from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
import gprMax.sources as sources_module
from gprMax.sources import EigenmodeSource

FREQUENCIES = (32e9, 45e9, 55e9, 65e9, 78e9)


def _fields(vector):
    values = np.asarray(vector, dtype=np.complex128)
    zero = np.zeros_like(values)
    return [values, zero.copy(), zero.copy()], [
        zero.copy(),
        zero.copy(),
        zero.copy(),
    ]


class _FakeSolver:
    def __init__(self, vectors, power_valid):
        self.num_modes = len(vectors)
        self.power_valid = np.asarray(power_valid, dtype=bool)
        self.complex_neff = np.asarray(
            [0.5 if valid else -0.5j for valid in self.power_valid],
            dtype=np.complex128,
        )
        self.raw_powers = np.asarray(
            [1.0 if valid else 1.0j for valid in self.power_valid],
            dtype=np.complex128,
        )
        self.forward_power_metrics = np.asarray(
            [1.0 if valid else 0.0 for valid in self.power_valid],
            dtype=np.float64,
        )
        self.fields = [
            (*_fields(vector), self.complex_neff[index]) for index, vector in enumerate(vectors)
        ]


def _solvers(vectors, power_valid):
    return tuple(
        _FakeSolver(frequency_vectors, frequency_valid)
        for frequency_vectors, frequency_valid in zip(vectors, power_valid)
    )


def _source(monkeypatch, *, automatic=True):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={"z0": 376.730313668, "c": 299792458.0},
            dtypes={"float_or_double": np.float64},
        ),
    )
    source = EigenmodeSource(None)
    source.port_index = 1
    source.mode_index = 1
    source.mode_count = 2
    source.mode_indices = (1, 2)
    source.dft_start = 45e9
    source.dft_stop = 65e9
    source.fallback_frequency = 55e9
    source.requested_anchor_policy = "auto" if automatic else "explicit"
    source.anchor_policy = source.requested_anchor_policy
    source._fields_from_solver_mode = lambda solver, mode_index: solver.fields[mode_index - 1]
    return source


def test_tracked_nonpropagating_guard_is_excluded_per_mode(monkeypatch):
    source = _source(monkeypatch)
    vectors = [((1, 0), (0, 1))] * len(FREQUENCIES)
    solvers = _solvers(
        vectors,
        [
            (False, True),
            (True, True),
            (True, True),
            (True, True),
            (True, True),
        ],
    )
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    valid, policies = source._prepare_port_anchor_bank(
        FREQUENCIES,
        solvers,
        (1, 2),
    )

    assert valid[:, 0].tolist() == [False, True, True, True, True]
    assert valid[:, 1].tolist() == [True, True, True, True, True]
    assert source.port_anchor_mode_propagating.tolist() == valid.tolist()
    assert policies == (
        "auto_broadband_nonpropagating_trimmed",
        "auto_broadband",
    )
    assert any("mode 1" in warning and "non-propagating anchor" in warning for warning in warnings)


def test_in_band_tracking_failure_falls_back_only_the_affected_mode(monkeypatch):
    source = _source(monkeypatch)
    vectors = [
        ((1, 0), (0, 1)),
        ((1, 0), (0, 1)),
        ((1, 0), (1, 0)),
        ((1, 0), (0, 1)),
        ((1, 0), (0, 1)),
    ]
    solvers = _solvers(vectors, [(True, True)] * len(FREQUENCIES))
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    valid, policies = source._prepare_port_anchor_bank(
        FREQUENCIES,
        solvers,
        (1, 2),
    )

    assert valid[:, 0].tolist() == [True, True, True, True, True]
    assert valid[:, 1].tolist() == [False, False, True, False, False]
    assert policies == ("auto_broadband", "auto_single_fallback")
    assert any(
        "mode 2" in warning and "only the centre-frequency anchor" in warning
        for warning in warnings
    )


def test_nonpropagating_centre_after_in_band_tracking_failure_raises(monkeypatch):
    source = _source(monkeypatch)
    vectors = [
        ((1, 0), (0, 1)),
        ((1, 0), (0, 1)),
        ((1, 0), (1, 0)),
        ((1, 0), (0, 1)),
        ((1, 0), (0, 1)),
    ]
    power_valid = [(True, True)] * len(FREQUENCIES)
    power_valid[2] = (True, False)
    solvers = _solvers(vectors, power_valid)

    with pytest.raises(
        ValueError,
        match=r"mode 2 centre-frequency anchor.*non-propagating",
    ):
        source._prepare_port_anchor_bank(FREQUENCIES, solvers, (1, 2))


def test_guard_tracking_failure_is_trimmed_before_centre_fallback(monkeypatch):
    source = _source(monkeypatch)
    vectors = [
        ((1, 0), (1, 0)),
        ((1, 0), (0, 1)),
        ((1, 0), (0, 1)),
        ((1, 0), (0, 1)),
        ((1, 0), (0, 1)),
    ]
    solvers = _solvers(vectors, [(True, True)] * len(FREQUENCIES))
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    valid, policies = source._prepare_port_anchor_bank(
        FREQUENCIES,
        solvers,
        (1, 2),
    )

    assert valid[:, 0].tolist() == [True, True, True, True, True]
    assert valid[:, 1].tolist() == [False, True, True, True, True]
    assert policies == ("auto_broadband", "auto_broadband_guard_trimmed")
    assert any("mode 2" in warning and "lower spectral guard" in warning for warning in warnings)
    assert not any("only the centre-frequency anchor" in warning for warning in warnings)


def test_disconnected_propagating_anchors_fall_back_only_for_auto(monkeypatch):
    vectors = [((1, 0), (0, 1))] * len(FREQUENCIES)
    power_valid = [
        (True, True),
        (False, True),
        (True, True),
        (True, True),
        (True, True),
    ]
    solvers = _solvers(vectors, power_valid)
    automatic = _source(monkeypatch)

    valid, policies = automatic._prepare_port_anchor_bank(
        FREQUENCIES,
        solvers,
        (1, 2),
    )

    assert valid[:, 0].tolist() == [False, False, True, False, False]
    assert valid[:, 1].tolist() == [True, True, True, True, True]
    assert policies == ("auto_single_fallback", "auto_broadband")

    explicit = _source(monkeypatch, automatic=False)
    with pytest.raises(ValueError, match="disconnected propagating anchor ranges"):
        explicit._prepare_port_anchor_bank(FREQUENCIES, solvers, (1, 2))


def test_active_excitation_uses_only_its_propagating_anchor_subset(monkeypatch):
    source = _source(monkeypatch)
    vectors = [((1, 0), (0, 1))] * len(FREQUENCIES)
    solvers = _solvers(
        vectors,
        [
            (False, True),
            (True, True),
            (True, True),
            (True, True),
            (True, True),
        ],
    )
    by_frequency = dict(zip(FREQUENCIES, solvers))
    source.frequencies = FREQUENCIES
    source._extract_frequency_dependent_materials = lambda grid: None
    source._solve_eigenmode = lambda grid: setattr(
        source,
        "mode_solver",
        by_frequency[source.frequency],
    )
    prepared = []
    source._prepare_broadband_time_traces = lambda grid, frequencies: prepared.append(
        tuple(frequencies)
    )

    source._solve_broadband_eigenmode(SimpleNamespace(), FREQUENCIES)

    expected = FREQUENCIES[1:]
    assert source.port_anchor_frequencies == FREQUENCIES
    assert source.port_anchor_mode_valid[:, 0].tolist() == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert source.frequencies == expected
    assert len(source.anchor_modal_e) == len(expected)
    assert len(source.mode_solvers) == len(expected)
    assert prepared == [expected]
