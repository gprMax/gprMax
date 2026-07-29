from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.sources import EigenmodeSource
from gprMax.waveforms import Waveform


def _field_set(values):
    values = np.asarray(values, dtype=np.complex128)
    return [values.copy(), np.zeros_like(values), np.zeros_like(values)]


def test_broadband_anchor_phase_alignment_is_phase_invariant(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.mode_index = 0
    source.mode_overlap_threshold = 0.9
    reference_e = _field_set([1.0, 2.0])
    reference_h = _field_set([0.25, 0.5])
    phase = np.exp(0.73j)
    anchor_e = [reference_e, [field * phase for field in reference_e]]
    anchor_h = [reference_h, [field * phase for field in reference_h]]

    overlaps = source._align_and_validate_anchors(
        anchor_e, anchor_h, (1e9, 2e9)
    )

    assert overlaps == pytest.approx([1.0])
    for expected, actual in zip(reference_e, anchor_e[1]):
        assert actual == pytest.approx(expected)
    for expected, actual in zip(reference_h, anchor_h[1]):
        assert actual == pytest.approx(expected)


def test_broadband_anchor_overlap_warns_and_continues(monkeypatch, caplog):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.mode_index = 2
    source.mode_overlap_threshold = 0.9
    anchor_e = [_field_set([1.0, 0.0]), _field_set([0.0, 1.0])]
    anchor_h = [_field_set([0.0, 0.0]), _field_set([0.0, 0.0])]

    with caplog.at_level("WARNING"):
        overlaps = source._align_and_validate_anchors(
            anchor_e, anchor_h, (1e9, 2e9)
        )

    assert overlaps == pytest.approx([0.0])
    assert "Continuing with the supplied anchor modes" in caplog.text


def test_broadband_invalid_anchor_norm_warns_and_continues(monkeypatch, caplog):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.mode_index = 0
    source.mode_overlap_threshold = 0.9
    zero_fields = _field_set([0.0, 0.0])
    anchor_e = [
        [field.copy() for field in zero_fields],
        [field.copy() for field in zero_fields],
    ]
    anchor_h = [
        [field.copy() for field in zero_fields],
        [field.copy() for field in zero_fields],
    ]

    with caplog.at_level("WARNING"):
        overlaps = source._align_and_validate_anchors(
            anchor_e, anchor_h, (1e9, 2e9)
        )

    assert overlaps == pytest.approx([0.0])
    assert "zero or invalid field norm" in caplog.text
    assert "Continuing with the supplied anchor modes" in caplog.text


def test_broadband_quality_safeguards_warn_and_continue(monkeypatch, caplog):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "z0": 376.730313668,
                "c": 299792458.0,
            },
            dtypes={"float_or_double": np.float64},
        ),
    )
    source = EigenmodeSource(None)
    source.start = 0.0
    source.normal_axis = 0
    source.spectral_threshold = 1e-3
    source.anchor_complex_neff = np.ones(2, dtype=np.complex128)
    modal_fields = [
        np.ones((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
    ]
    source.anchor_modal_e = [
        [field.copy() for field in modal_fields],
        [field.copy() for field in modal_fields],
    ]
    source.anchor_modal_h = [
        [field.copy() for field in modal_fields],
        [field.copy() for field in modal_fields],
    ]
    source._modal_cross_power = lambda electric, magnetic, grid: 0.0
    source.waveform = Waveform()
    source.waveform.type = "ricker"
    source.waveform.amp = 1.0
    source.waveform.freq = 5e9
    grid = SimpleNamespace(
        iterations=2048,
        dt=1e-12,
        dl=np.asarray([0.5e-3, 0.5e-3, 0.5e-3]),
    )

    with caplog.at_level("WARNING"):
        source._prepare_broadband_time_traces(grid, (4.9e9, 5.1e9))

    assert "do not cover the significant waveform spectrum" in caplog.text
    assert "fallback normalization" in caplog.text
    assert np.all(np.isfinite(source.broadband_e_envelopes))
    assert source.broadband_waveform_error < 1e-8


def test_linear_anchor_weights_are_local_and_sum_to_one():
    bins = np.asarray([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    anchors = np.asarray([1.0, 2.0, 4.0])

    weights = EigenmodeSource._linear_anchor_weights(bins, anchors)

    assert weights[:, 0] == pytest.approx([1.0, 0.0, 0.0])
    assert weights[:, 1] == pytest.approx([1.0, 0.0, 0.0])
    assert weights[:, 2] == pytest.approx([0.5, 0.5, 0.0])
    assert weights[:, 3] == pytest.approx([0.0, 1.0, 0.0])
    assert weights[:, 4] == pytest.approx([0.0, 0.5, 0.5])
    assert weights[:, 5] == pytest.approx([0.0, 0.0, 1.0])
    assert weights[:, 6] == pytest.approx([0.0, 0.0, 1.0])
    assert np.sum(weights, axis=0) == pytest.approx(np.ones(7))


@pytest.mark.parametrize("wave_type", ("ricker", "gaussiandot"))
def test_broadband_ifft_recovers_original_waveform(monkeypatch, wave_type):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "z0": 376.730313668,
                "c": 299792458.0,
            },
            dtypes={"float_or_double": np.float64},
        ),
    )
    source = EigenmodeSource(None)
    source.start = 0.0
    source.normal_axis = 0
    source.spectral_threshold = 1e-3
    source.anchor_complex_neff = np.ones(2, dtype=np.complex128)
    modal_fields = [
        np.ones((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
    ]
    source.anchor_modal_e = [
        [field.copy() for field in modal_fields],
        [field.copy() for field in modal_fields],
    ]
    source.anchor_modal_h = [
        [field.copy() for field in modal_fields],
        [field.copy() for field in modal_fields],
    ]
    source._modal_cross_power = lambda electric, magnetic, grid: 1.0
    source.waveform = Waveform()
    source.waveform.type = wave_type
    source.waveform.amp = 1.0
    source.waveform.freq = 5e9
    grid = SimpleNamespace(
        iterations=2048,
        dt=1e-12,
        dl=np.asarray([0.5e-3, 0.5e-3, 0.5e-3]),
    )

    source._prepare_broadband_time_traces(grid, (0.1e9, 25e9))

    reconstructed = np.sum(source.broadband_e_envelopes[:, 0, :], axis=0)
    times = np.arange(grid.iterations) * grid.dt
    expected = np.asarray(
        [source.waveform.calculate_value(time, grid.dt) for time in times]
    )
    peak = np.max(np.abs(expected))
    assert np.max(np.abs(reconstructed - expected)) / peak < 1e-8
    assert source.broadband_waveform_error < 1e-8


def test_magnetic_stagger_factor_uses_each_frequency_and_beta():
    omega = 2 * np.pi * np.asarray([1e9, 4e9])
    beta = np.asarray([12.0, 31.0])
    dt = 2e-12
    spacing = 0.5e-3

    factors = EigenmodeSource._magnetic_stagger_factor(
        omega, beta, dt, spacing
    )

    assert factors == pytest.approx(
        np.exp(1j * (omega * dt / 2 + beta * spacing / 2))
    )
    assert factors[0] != pytest.approx(factors[1])


def test_modal_cross_power_is_positive_in_requested_negative_direction(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.direction = "-"
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.invariant_axis = 2
    source.physical_transverse_axis = 1
    source.domain_polarization = "TM"
    electric = [
        np.zeros((4, 2), dtype=np.complex128),
        np.zeros((3, 2), dtype=np.complex128),
        np.ones((4, 1), dtype=np.complex128),
    ]
    magnetic = [
        np.zeros((3, 1), dtype=np.complex128),
        np.ones((4, 1), dtype=np.complex128),
        np.zeros((3, 2), dtype=np.complex128),
    ]
    grid = SimpleNamespace(dl=np.asarray([0.5, 0.25, 1.0]))

    assert source._modal_cross_power(electric, magnetic, grid) == pytest.approx(0.375)


def test_modal_cross_power_2d_te_uses_only_live_invariant_layer(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.direction = "+"
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.invariant_axis = 2
    source.physical_transverse_axis = 1
    source.domain_polarization = "TE"
    electric = [
        np.zeros((4, 2), dtype=np.complex128),
        np.zeros((3, 3), dtype=np.complex128),
        np.zeros((4, 2), dtype=np.complex128),
    ]
    magnetic = [
        np.zeros((4, 2), dtype=np.complex128),
        np.zeros((3, 3), dtype=np.complex128),
        np.zeros((3, 3), dtype=np.complex128),
    ]
    electric[1][:, 1] = 2.0
    magnetic[2][:, 1] = 0.5
    grid = SimpleNamespace(dl=np.asarray([0.5, 0.25, 1.0]))

    assert source._modal_cross_power(electric, magnetic, grid) == pytest.approx(0.375)
