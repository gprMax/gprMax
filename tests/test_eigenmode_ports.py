import csv
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
import gprMax.ports as ports_module
from gprMax.eigenmode_ports import (
    DFT_PHASE_REANCHOR_INTERVAL,
    EigenmodePortMonitor,
    EigenmodePortResult,
    accumulate_eigenmode_dft,
    finalise_eigenmode_ports,
)
from gprMax.ports import evaluate_port_power_spectrum
from gprMax.sources import EigenmodeSource


@pytest.mark.parametrize(
    ("real_dtype", "complex_dtype"),
    ((np.float32, np.complex64), (np.float64, np.complex128)),
)
def test_cython_dft_updates_every_bin_once_per_time_step(real_dtype, complex_dtype):
    field_shape = (2, 2, 2)
    zeros = np.zeros(field_shape, dtype=real_dtype)
    ey = np.full(field_shape, 2, dtype=real_dtype)
    hz = np.full(field_shape, 3, dtype=real_dtype)
    mode_shape = (2, 1, 1, 1)
    mode_zeros = np.zeros(mode_shape, dtype=complex_dtype)
    mode_ones = np.ones(mode_shape, dtype=complex_dtype)
    electric_phase = np.ones(2, dtype=complex_dtype)
    magnetic_phase = np.ones(2, dtype=complex_dtype)
    phase_step = np.asarray((1j, -1), dtype=complex_dtype)
    electric_dft = np.zeros((2, 1), dtype=complex_dtype)
    magnetic_dft = np.zeros((2, 1), dtype=complex_dtype)
    owned_lower = np.zeros(3, dtype=np.int32)
    owned_upper = np.asarray(field_shape, dtype=np.int32)
    real_signature = "float" if real_dtype is np.float32 else "double"
    kernel = accumulate_eigenmode_dft[f"{real_signature}|{real_signature} complex"]

    for _ in range(2):
        kernel(
            1,
            0,
            1,
            -1,
            0,
            0,
            1,
            1,
            1,
            owned_lower,
            owned_upper,
            real_dtype(0.1),
            real_dtype(1),
            1,
            electric_phase,
            magnetic_phase,
            phase_step,
            mode_ones,
            mode_zeros,
            mode_zeros,
            mode_ones,
            electric_dft,
            magnetic_dft,
            zeros,
            ey,
            zeros,
            zeros,
            zeros,
            hz,
        )

    np.testing.assert_allclose(electric_dft[:, 0], (0.1 + 0.1j, 0), atol=1e-6)
    np.testing.assert_allclose(magnetic_dft[:, 0], (0.15 + 0.15j, 0), atol=1e-6)
    np.testing.assert_allclose(electric_phase, (-1, 1), atol=1e-6)
    np.testing.assert_allclose(magnetic_phase, (-1, 1), atol=1e-6)


def test_complex64_eigenmode_phase_drift_is_bounded_by_periodic_reanchoring(
    monkeypatch,
):
    iterations = 65_537
    dt = 2e-12
    frequencies = np.asarray([0.137 / dt, 0.467 / dt], dtype=np.float32)
    check_iterations = {
        0,
        1,
        DFT_PHASE_REANCHOR_INTERVAL - 1,
        DFT_PHASE_REANCHOR_INTERVAL,
        DFT_PHASE_REANCHOR_INTERVAL + 1,
        32_767,
        iterations - 2,
        iterations - 1,
    }
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={
                "C_float_or_double": "float",
                "complex": np.complex64,
            }
        ),
    )
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(ompthreads=1),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.owner = SimpleNamespace(
        normal_axis=0,
        direction="+",
        transverse_start=(0, 0),
        transverse_stop=(1, 1),
        plane_index=1,
    )
    monitor.magnetic_side = -1
    monitor.measure = np.float32(1.0)
    monitor.handedness = 1
    monitor.frequency = frequencies
    phase_argument = -2j * np.pi * frequencies.astype(np.float64) * dt
    monitor.phase_step = np.exp(phase_argument).astype(np.complex64)
    monitor.electric_phase = np.ones(2, dtype=np.complex64)
    monitor.magnetic_phase = np.exp(0.5 * phase_argument).astype(np.complex64)
    monitor._next_iteration = 0
    mode_shape = (2, 1, 1, 1)
    mode_zeros = np.zeros(mode_shape, dtype=np.complex64)
    monitor.conj_eu = mode_zeros
    monitor.conj_ev = mode_zeros.copy()
    monitor.conj_hu = mode_zeros.copy()
    monitor.conj_hv = mode_zeros.copy()
    monitor.electric_dft = np.zeros((2, 1), dtype=np.complex64)
    monitor.magnetic_dft = np.zeros((2, 1), dtype=np.complex64)
    field = np.zeros((2, 2, 2), dtype=np.float32)
    grid = SimpleNamespace(
        dt=dt,
        Ex=field,
        Ey=field.copy(),
        Ez=field.copy(),
        Hx=field.copy(),
        Hy=field.copy(),
        Hz=field.copy(),
    )
    errors = {}

    for iteration in range(iterations):
        if iteration in check_iterations:
            expected_electric = np.exp(phase_argument * iteration)
            expected_magnetic = np.exp(phase_argument * (iteration + 0.5))
            errors[iteration] = max(
                float(np.max(np.abs(monitor.electric_phase - expected_electric))),
                float(np.max(np.abs(monitor.magnetic_phase - expected_magnetic))),
            )
        monitor.observe(grid, iteration)

    assert monitor.electric_phase.dtype == np.complex64
    assert monitor.magnetic_phase.dtype == np.complex64
    assert max(errors.values()) < 2e-5
    assert errors[DFT_PHASE_REANCHOR_INTERVAL] < 2e-7


def test_2d_te_monitor_preserves_absolute_power_after_cell_averaging(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={
                "float_or_double": np.float64,
                "complex": np.complex128,
            }
        ),
    )
    owner = EigenmodeSource(None)
    owner.normal_axis = 0
    owner.transverse_axes = (1, 2)
    owner.invariant_axis = 2
    owner.physical_transverse_axis = 1
    owner.domain_polarization = "TE"
    owner.transverse_start = np.zeros(2, dtype=np.int32)
    owner.transverse_stop = np.asarray((2, 2), dtype=np.int32)

    electric = [
        np.zeros((3, 3), dtype=np.complex128),
        np.zeros((2, 3), dtype=np.complex128),
        np.zeros((3, 2), dtype=np.complex128),
    ]
    magnetic = [
        np.zeros((2, 2), dtype=np.complex128),
        np.zeros((3, 2), dtype=np.complex128),
        np.zeros((2, 3), dtype=np.complex128),
    ]
    electric[1][:, 1] = 1.0
    magnetic[2][:, 1] = 1.0
    grid = SimpleNamespace(
        dt=1e-12,
        dl=np.ones(3),
        eigenmodeports=[],
    )
    assert owner._modal_cross_power(electric, magnetic, grid) == pytest.approx(1.0)

    monitor = EigenmodePortMonitor(
        owner=owner,
        port_index=1,
        port_id="te",
        is_source=False,
        excitation_mode_index=None,
        mode_indices=(1,),
        anchor_frequencies=np.asarray([1e9]),
        anchor_e=[[electric]],
        anchor_h=[[magnetic]],
        anchor_neff=np.asarray([[1.0]]),
        dft_start=1e9,
        dft_stop=1e9,
        dft_points=1,
    )

    monitor.prepare(grid)

    assert monitor.measure == pytest.approx(2.0)
    assert monitor.electric_gram[0, 0, 0] == pytest.approx(1.0)
    assert monitor.magnetic_gram[0, 0, 0] == pytest.approx(1.0)
    assert monitor.power_matrix[0, 0, 0] == pytest.approx(1.0)


def test_monitor_uses_tracked_nonpropagating_reference_for_generalized_tail(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={
                "float_or_double": np.float64,
                "complex": np.complex128,
            }
        ),
    )
    owner = SimpleNamespace(
        transverse_axes=(1, 2),
        invariant_axis=None,
        normal_axis=0,
        direction="+",
        _linear_anchor_weights=EigenmodeSource._linear_anchor_weights,
        _transverse_cell_shape=lambda: (1, 1),
        _modal_cross_power=lambda electric, magnetic, grid: 1.0,
        _average_to_transverse_cells=lambda values, component: values,
        _modal_basis_handedness=lambda: 1,
    )
    zero = np.zeros((1, 1), dtype=np.complex128)

    def modal_fields(electric_value, magnetic_value=None):
        electric = np.full((1, 1), electric_value, dtype=np.complex128)
        magnetic = np.full(
            (1, 1),
            electric_value if magnetic_value is None else magnetic_value,
            dtype=np.complex128,
        )
        return [zero, electric, zero], [zero, zero, magnetic]

    first_mode = [modal_fields(99, 9j), modal_fields(2), modal_fields(3)]
    second_mode = [modal_fields(4), modal_fields(5), modal_fields(88, 8j)]
    monitor = EigenmodePortMonitor(
        owner=owner,
        port_index=1,
        port_id="cutoff",
        is_source=False,
        excitation_mode_index=None,
        mode_indices=(1, 2),
        anchor_frequencies=np.asarray([22e9, 28e9, 34e9]),
        anchor_e=[[first_mode[index][0], second_mode[index][0]] for index in range(3)],
        anchor_h=[[first_mode[index][1], second_mode[index][1]] for index in range(3)],
        anchor_neff=np.asarray([[-0.5j, 0.2], [0.4, 0.4], [0.7, -0.3j]]),
        anchor_mode_valid=np.asarray([[False, True], [True, True], [True, False]]),
        anchor_mode_reference_valid=np.ones((3, 2), dtype=bool),
        anchor_mode_propagating=np.asarray([[False, True], [True, True], [True, False]]),
        mode_anchor_policies=(
            "auto_nonpropagating_trimmed",
            "auto_nonpropagating_trimmed",
        ),
        dft_start=22e9,
        dft_stop=34e9,
        dft_points=3,
    )
    grid = SimpleNamespace(
        dt=1e-12,
        dl=np.ones(3),
        eigenmodeports=[],
    )

    monitor.prepare(grid)

    # The generalized-only bins use the tracked evanescent E/H reference,
    # including its quadrature admittance, while propagating bins retain the
    # original unit-real-power basis.
    assert monitor.hv[0, 0, 0, 0] / monitor.eu[0, 0, 0, 0] == pytest.approx(9j / 99)
    assert monitor.hv[2, 1, 0, 0] / monitor.eu[2, 1, 0, 0] == pytest.approx(8j / 88)
    assert monitor.eu[1:, 0, 0, 0] == pytest.approx([2, 3])
    assert monitor.eu[:2, 1, 0, 0] == pytest.approx([4, 5])
    assert monitor.neff[:, 0] == pytest.approx([-0.5j, 0.4, 0.7])
    assert monitor.neff[:, 1] == pytest.approx([0.2, 0.4, -0.3j])
    assert monitor.mode_power_valid.tolist() == [
        [False, True],
        [True, True],
        [True, False],
    ]
    assert monitor.mode_decomposition_valid.tolist() == [
        [True, True],
        [True, True],
        [True, True],
    ]


def test_monitor_extrapolates_an_outer_propagating_endpoint(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={"float_or_double": np.float64, "complex": np.complex128},
            em_consts={"z0": 1.0},
        ),
    )
    owner = SimpleNamespace(
        transverse_axes=(1, 2),
        invariant_axis=None,
        normal_axis=0,
        direction="+",
        _linear_anchor_weights=EigenmodeSource._linear_anchor_weights,
        _transverse_cell_shape=lambda: (1, 1),
        _modal_cross_power=lambda electric, magnetic, grid: 1.0,
        _average_to_transverse_cells=lambda values, component: values,
        _modal_basis_handedness=lambda: 1,
    )
    zero = np.zeros((1, 1), dtype=np.complex128)

    def fields(magnetic_value):
        electric = np.ones((1, 1), dtype=np.complex128)
        magnetic = np.full((1, 1), magnetic_value, dtype=np.complex128)
        return [zero, electric, zero], [zero, zero, magnetic]

    evanescent = fields(0.1j)
    propagating = fields(1.0)
    monitor = EigenmodePortMonitor(
        owner=owner,
        port_index=1,
        port_id="outer-propagating-endpoint",
        is_source=False,
        excitation_mode_index=None,
        mode_indices=(1,),
        anchor_frequencies=np.asarray([1e9, 2e9]),
        anchor_e=[[evanescent[0]], [propagating[0]]],
        anchor_h=[[evanescent[1]], [propagating[1]]],
        anchor_neff=np.asarray([[-0.5j], [0.4]]),
        anchor_mode_valid=np.asarray([[False], [True]]),
        anchor_mode_reference_valid=np.ones((2, 1), dtype=bool),
        anchor_mode_propagating=np.asarray([[False], [True]]),
        mode_anchor_policies=("explicit_nonpropagating_trimmed",),
        dft_start=1e9,
        dft_stop=3e9,
        dft_points=3,
    )
    grid = SimpleNamespace(dt=1e-12, dl=np.ones(3), eigenmodeports=[])

    monitor.prepare(grid)

    assert monitor.power_wave_valid[:, 0].tolist() == [False, True, True]
    assert monitor.neff[:, 0] == pytest.approx([-0.5j, 0.4, 0.4])
    np.testing.assert_allclose(
        monitor.hv[:, 0, 0, 0] / monitor.eu[:, 0, 0, 0],
        [0.1j, 1.0, 1.0],
    )


def test_monitor_rejects_matching_zero_neff_cutoff_anchor(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={
                "float_or_double": np.float64,
                "complex": np.complex128,
            }
        ),
    )
    owner = SimpleNamespace(
        transverse_axes=(1, 2),
        invariant_axis=None,
        normal_axis=0,
        direction="+",
        _linear_anchor_weights=EigenmodeSource._linear_anchor_weights,
        _transverse_cell_shape=lambda: (1, 1),
        _modal_cross_power=lambda electric, magnetic, grid: 1.0,
        _average_to_transverse_cells=lambda values, component: values,
        _modal_basis_handedness=lambda: 1,
    )
    zero = np.zeros((1, 1), dtype=np.complex128)
    field = np.ones((1, 1), dtype=np.complex128)
    modal_fields = [zero, field, zero], [zero, zero, field]
    monitor = EigenmodePortMonitor(
        owner=owner,
        port_index=1,
        port_id="exact-cutoff",
        is_source=False,
        excitation_mode_index=None,
        mode_indices=(1,),
        anchor_frequencies=np.asarray([1e9, 2e9]),
        anchor_e=[[modal_fields[0]], [modal_fields[0]]],
        anchor_h=[[modal_fields[1]], [modal_fields[1]]],
        anchor_neff=np.asarray([[0.0], [0.5]]),
        anchor_mode_valid=np.asarray([[False], [True]]),
        anchor_mode_reference_valid=np.asarray([[True], [True]]),
        anchor_mode_propagating=np.asarray([[False], [True]]),
        mode_anchor_policies=("explicit_nonpropagating_trimmed",),
        dft_start=1e9,
        dft_stop=2e9,
        dft_points=2,
    )
    grid = SimpleNamespace(dt=1e-12, dl=np.ones(3), eigenmodeports=[])

    monitor.prepare(grid)

    assert monitor.mode_decomposition_valid[:, 0].tolist() == [False, True]
    assert monitor.mode_power_valid[:, 0].tolist() == [False, True]


def test_monitor_keeps_two_evanescent_reference_runs_separate(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={"float_or_double": np.float64, "complex": np.complex128},
            em_consts={"z0": 1.0},
        ),
    )
    owner = SimpleNamespace(
        transverse_axes=(1, 2),
        invariant_axis=None,
        normal_axis=0,
        direction="+",
        _linear_anchor_weights=EigenmodeSource._linear_anchor_weights,
        _transverse_cell_shape=lambda: (1, 1),
        _modal_cross_power=lambda electric, magnetic, grid: 1.0,
        _average_to_transverse_cells=lambda values, component: values,
        _modal_basis_handedness=lambda: 1,
    )
    zero = np.zeros((1, 1), dtype=np.complex128)

    def fields(magnetic_value):
        one = np.ones((1, 1), dtype=np.complex128)
        magnetic = np.full((1, 1), magnetic_value, dtype=np.complex128)
        return [zero, one, zero], [zero, zero, magnetic]

    anchors = [fields(0.1j), fields(1.0), fields(0.2j)]
    monitor = EigenmodePortMonitor(
        owner=owner,
        port_index=1,
        port_id="two-cutoffs",
        is_source=False,
        excitation_mode_index=None,
        mode_indices=(1,),
        anchor_frequencies=np.asarray([1e9, 2e9, 3e9]),
        anchor_e=[[anchor[0]] for anchor in anchors],
        anchor_h=[[anchor[1]] for anchor in anchors],
        anchor_neff=np.asarray([[-0.5j], [0.4], [-0.3j]]),
        anchor_mode_valid=np.asarray([[False], [True], [False]]),
        anchor_mode_reference_valid=np.ones((3, 1), dtype=bool),
        anchor_mode_propagating=np.asarray([[False], [True], [False]]),
        mode_anchor_policies=("explicit_nonpropagating_trimmed",),
        dft_start=1e9,
        dft_stop=3e9,
        dft_points=5,
    )
    grid = SimpleNamespace(dt=1e-12, dl=np.ones(3), eigenmodeports=[])

    monitor.prepare(grid)

    assert monitor.power_wave_valid[:, 0].tolist() == [False, False, True, False, False]
    assert monitor.neff[:, 0] == pytest.approx([-0.5j, -0.5j, 0.4, -0.3j, -0.3j])
    np.testing.assert_allclose(
        monitor.hv[:, 0, 0, 0] / monitor.eu[:, 0, 0, 0],
        [0.1j, 0.1j, 1.0, 0.2j, 0.2j],
    )


def test_monitor_uses_nearest_tracked_endpoint_outside_candidate_range(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={"float_or_double": np.float64, "complex": np.complex128},
            em_consts={"z0": 1.0},
        ),
    )
    owner = SimpleNamespace(
        transverse_axes=(1, 2),
        invariant_axis=None,
        normal_axis=0,
        direction="+",
        _linear_anchor_weights=EigenmodeSource._linear_anchor_weights,
        _transverse_cell_shape=lambda: (1, 1),
        _modal_cross_power=lambda electric, magnetic, grid: 1.0,
        _average_to_transverse_cells=lambda values, component: values,
        _modal_basis_handedness=lambda: 1,
    )
    zero = np.zeros((1, 1), dtype=np.complex128)

    def fields(magnetic_value):
        electric = np.ones((1, 1), dtype=np.complex128)
        magnetic = np.full((1, 1), magnetic_value, dtype=np.complex128)
        return [zero, electric, zero], [zero, zero, magnetic]

    anchor_fields = [
        [fields(10), fields(20)],
        [fields(1.0), fields(0.15j)],
        [fields(0.25j), fields(2.0)],
        [fields(30), fields(40)],
    ]
    monitor = EigenmodePortMonitor(
        owner=owner,
        port_index=1,
        port_id="trimmed-outer-candidates",
        is_source=False,
        excitation_mode_index=None,
        mode_indices=(1, 2),
        anchor_frequencies=np.asarray([1e9, 2e9, 3e9, 4e9]),
        anchor_e=[[fields_[0] for fields_ in anchor] for anchor in anchor_fields],
        anchor_h=[[fields_[1] for fields_ in anchor] for anchor in anchor_fields],
        anchor_neff=np.asarray([[10.0, 20.0], [0.4, -0.5j], [-0.3j, 0.6], [30.0, 40.0]]),
        anchor_mode_valid=np.asarray(
            [[False, False], [True, False], [False, True], [False, False]]
        ),
        anchor_mode_reference_valid=np.asarray(
            [[False, False], [True, True], [True, True], [False, False]]
        ),
        anchor_mode_propagating=np.asarray(
            [[False, False], [True, False], [False, True], [False, False]]
        ),
        mode_anchor_policies=("auto_guard_trimmed", "auto_guard_trimmed"),
        dft_start=0.5e9,
        dft_stop=4.5e9,
        dft_points=2,
    )
    grid = SimpleNamespace(dt=1e-12, dl=np.ones(3), eigenmodeports=[])

    monitor.prepare(grid)

    assert not np.any(monitor.power_wave_valid)
    np.testing.assert_allclose(
        monitor.neff,
        [[0.4, -0.5j], [-0.3j, 0.6]],
    )
    np.testing.assert_allclose(
        monitor.hv[:, :, 0, 0] / monitor.eu[:, :, 0, 0],
        [[1.0, 0.15j], [0.25j, 2.0]],
    )


def test_monitor_rejects_disconnected_usable_anchor_ranges():
    monitor = EigenmodePortMonitor(
        owner=SimpleNamespace(),
        port_index=1,
        port_id="gap",
        is_source=False,
        excitation_mode_index=None,
        mode_indices=(1,),
        anchor_frequencies=np.asarray([1e9, 2e9, 3e9]),
        anchor_e=[None, None, None],
        anchor_h=[None, None, None],
        anchor_neff=np.ones((3, 1)),
        anchor_mode_valid=np.asarray([[True], [False], [True]]),
        anchor_mode_propagating=np.asarray([[True], [False], [True]]),
        mode_anchor_policies=("explicit",),
        dft_start=1e9,
        dft_stop=3e9,
        dft_points=3,
    )

    with pytest.raises(ValueError, match="disconnected usable anchor ranges"):
        monitor._validate()


def test_multimode_gram_solve_separates_incident_and_outgoing_waves(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"complex": np.complex128}),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    gram = np.asarray([[1.0, 0.25], [0.25, 1.0]], dtype=np.complex128)
    expected_incident = np.asarray([2.0, 3.0], dtype=np.complex128)
    expected_outgoing = np.asarray([0.5, -0.25], dtype=np.complex128)
    monitor.electric_dft = (gram @ (expected_incident + expected_outgoing))[np.newaxis]
    monitor.magnetic_dft = (gram @ (expected_incident - expected_outgoing))[np.newaxis]
    monitor.electric_gram = gram[np.newaxis]
    monitor.magnetic_gram = gram[np.newaxis]
    monitor.frequency = np.asarray([1e9])
    monitor.neff = np.zeros((1, 2), dtype=np.complex128)
    monitor.mode_power_valid = np.ones((1, 2), dtype=bool)
    monitor.power_matrix_valid = np.ones(1, dtype=bool)
    monitor.owner = SimpleNamespace(normal_axis=0)
    monitor.magnetic_side = -1

    result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))

    np.testing.assert_allclose(result.incident[:, 0], expected_incident)
    np.testing.assert_allclose(result.outgoing[:, 0], expected_outgoing)
    assert result.generalized_valid[:, 0].all()
    assert result.valid[:, 0].all()


@pytest.mark.parametrize(
    ("complex_dtype", "expected_valid"),
    ((np.complex64, False), (np.complex128, True)),
)
def test_condition_validity_accounts_for_input_precision(
    monkeypatch, complex_dtype, expected_valid
):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"complex": complex_dtype}),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    gram = np.diag(np.asarray([1.0, 1e-5], dtype=complex_dtype))
    monitor.electric_gram = gram[np.newaxis]
    monitor.magnetic_gram = gram[np.newaxis]
    monitor.electric_dft = np.asarray([[1.0, 1e-5]], dtype=complex_dtype)
    monitor.magnetic_dft = monitor.electric_dft.copy()
    monitor.frequency = np.asarray([1e9])
    monitor.neff = np.zeros((1, 2), dtype=complex_dtype)
    monitor.mode_power_valid = np.ones((1, 2), dtype=bool)
    monitor.power_matrix_valid = np.ones(1, dtype=bool)
    monitor.owner = SimpleNamespace(normal_axis=0)
    monitor.magnetic_side = -1

    result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))

    if expected_valid:
        assert result.condition_number[0] == pytest.approx(1e5, rel=1e-6)
    else:
        assert np.isinf(result.condition_number[0])
    assert result.valid[:, 0].tolist() == [expected_valid, expected_valid]


def test_scalar_tiny_gram_is_invalid_despite_unit_condition_number(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"complex": np.complex128}),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.electric_gram = np.asarray([[[1e-20]]], dtype=np.complex128)
    monitor.magnetic_gram = np.asarray([[[1e-20j]]], dtype=np.complex128)
    monitor.electric_dft = np.asarray([[1e-20]], dtype=np.complex128)
    monitor.magnetic_dft = np.asarray([[1e-20j]], dtype=np.complex128)
    monitor.frequency = np.asarray([1e9])
    monitor.neff = np.zeros((1, 1), dtype=np.complex128)
    monitor.mode_decomposition_valid = np.ones((1, 1), dtype=bool)
    monitor.mode_power_valid = np.ones((1, 1), dtype=bool)
    monitor.power_matrix_valid = np.ones(1, dtype=bool)
    monitor.owner = SimpleNamespace(normal_axis=0)
    monitor.magnetic_side = -1

    result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))

    assert np.isinf(result.condition_number[0])
    assert not result.generalized_valid[0, 0]
    assert not result.valid[0, 0]


def test_finalise_rejects_fallback_power_normalization(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"complex": np.complex128}),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.electric_gram = np.ones((1, 1, 1), dtype=np.complex128)
    monitor.magnetic_gram = np.ones((1, 1, 1), dtype=np.complex128)
    monitor.electric_dft = np.ones((1, 1), dtype=np.complex128)
    monitor.magnetic_dft = np.ones((1, 1), dtype=np.complex128)
    monitor.frequency = np.asarray([1e9])
    monitor.neff = np.zeros((1, 1), dtype=np.complex128)
    monitor.mode_power_valid = np.asarray([[False]])
    monitor.power_matrix_valid = np.asarray([False])
    monitor.owner = SimpleNamespace(normal_axis=0)
    monitor.magnetic_side = -1

    result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))

    assert np.isfinite(result.incident[0, 0])
    assert np.isfinite(result.outgoing[0, 0])
    assert not result.valid[0, 0]


def test_finalise_keeps_conditioned_below_cutoff_generalized_coefficients(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"complex": np.complex128}),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.electric_gram = np.ones((1, 1, 1), dtype=np.complex128)
    monitor.magnetic_gram = np.ones((1, 1, 1), dtype=np.complex128)
    monitor.electric_dft = np.asarray([[2.5]], dtype=np.complex128)
    monitor.magnetic_dft = np.asarray([[1.5]], dtype=np.complex128)
    monitor.frequency = np.asarray([22e9])
    monitor.neff = np.asarray([[0.5]], dtype=np.complex128)
    monitor.power_wave_valid = np.asarray([[False]])
    monitor.mode_decomposition_valid = np.asarray([[True]])
    monitor.power_matrix_valid = np.asarray([False])
    monitor.owner = SimpleNamespace(normal_axis=0)
    monitor.magnetic_side = -1

    result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))

    assert result.incident[0, 0] == pytest.approx(2.0)
    assert result.outgoing[0, 0] == pytest.approx(0.5)
    assert result.generalized_valid[0, 0]
    assert not result.valid[0, 0]
    assert not monitor.power_wave_valid[0, 0]
    assert not monitor.power_matrix_valid[0]


def test_physical_port_power_rejects_generalized_only_coefficients(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={
                "float_or_double": np.float64,
                "complex": np.complex128,
            }
        ),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.port_id = "below-cutoff"
    monitor.port_index = 1
    monitor.owner = SimpleNamespace()
    monitor.mode_indices = (1,)
    monitor.is_source = True
    monitor.excitation_mode_index = 1
    monitor.power_wave_valid = np.asarray([[False], [True]])
    monitor.power_matrix_valid = np.asarray([True, True])
    monitor.power_matrix = np.ones((2, 1, 1), dtype=np.complex128)
    monitor.electric_gram = np.ones((2, 1, 1), dtype=np.complex128)
    monitor.result = EigenmodePortResult(
        frequency=np.asarray([22e9, 28e9]),
        incident=np.asarray([[1000 + 0j, 2 + 0j]]),
        outgoing=np.asarray([[500 + 0j, 1 + 0j]]),
        valid=np.asarray([[False, True]]),
        condition_number=np.asarray([1.0, 1.0]),
        generalized_valid=np.asarray([[True, True]]),
    )
    monkeypatch.setattr(
        ports_module,
        "_port_mesh_valid",
        lambda output, grid, frequency: np.ones(frequency.shape, dtype=bool),
    )

    spectrum = evaluate_port_power_spectrum(monitor, SimpleNamespace(), [22e9, 28e9])

    assert not spectrum.modal_valid[0, 0]
    assert not spectrum.terminal_valid[0]
    assert spectrum.modal_valid[0, 1]
    assert spectrum.terminal_valid[1]
    assert spectrum.incident_power.tolist() == pytest.approx([0.0, 4.0])
    assert spectrum.accepted_power.tolist() == pytest.approx([0.0, 3.0])
    assert np.max(spectrum.incident_power) == pytest.approx(4.0)


def test_finalise_solves_propagating_sibling_when_other_mode_is_nonpropagating(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"complex": np.complex128}),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    gram = np.diag([0.0, 1.0]).astype(np.complex128)
    monitor.electric_gram = gram[np.newaxis]
    monitor.magnetic_gram = gram[np.newaxis]
    monitor.electric_dft = np.asarray([[123.0, 2.0]], dtype=np.complex128)
    monitor.magnetic_dft = np.asarray([[456.0, 2.0]], dtype=np.complex128)
    monitor.frequency = np.asarray([28e9])
    monitor.neff = np.zeros((1, 2), dtype=np.complex128)
    monitor.mode_decomposition_valid = np.asarray([[True, True]])
    monitor.mode_power_valid = np.asarray([[False, True]])
    monitor.power_wave_valid = np.asarray([[False, True]])
    monitor.power_matrix_valid = np.asarray([True])
    monitor.owner = SimpleNamespace(normal_axis=0)
    monitor.magnetic_side = -1

    result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))

    assert result.incident[:, 0] == pytest.approx([0.0, 2.0])
    assert result.outgoing[:, 0] == pytest.approx([0.0, 0.0])
    assert result.generalized_valid[:, 0].tolist() == [False, True]
    assert result.valid[:, 0].tolist() == [False, True]
    assert result.condition_number[0] == pytest.approx(1.0)


def test_finalise_rejects_truncated_nullspace_coupled_to_power_mode(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"complex": np.complex128}),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    gram = np.ones((2, 2), dtype=np.complex128)
    incident = np.asarray([2.0, 3.0], dtype=np.complex128)
    outgoing = np.asarray([0.5, -0.25], dtype=np.complex128)
    monitor.electric_gram = gram[np.newaxis]
    monitor.magnetic_gram = gram[np.newaxis]
    monitor.electric_dft = (gram @ (incident + outgoing))[np.newaxis]
    monitor.magnetic_dft = (gram @ (incident - outgoing))[np.newaxis]
    monitor.frequency = np.asarray([28e9])
    monitor.neff = np.zeros((1, 2), dtype=np.complex128)
    monitor.mode_decomposition_valid = np.asarray([[True, True]])
    monitor.mode_power_valid = np.asarray([[False, True]])
    monitor.power_wave_valid = np.asarray([[False, True]])
    monitor.power_matrix_valid = np.asarray([True])
    monitor.owner = SimpleNamespace(normal_axis=0)
    monitor.magnetic_side = -1

    result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))

    np.testing.assert_array_equal(result.incident[:, 0], 0.0)
    np.testing.assert_array_equal(result.outgoing[:, 0], 0.0)
    assert not np.any(result.generalized_valid[:, 0])
    assert not np.any(result.valid[:, 0])
    assert np.isinf(result.condition_number[0])


def test_hdf5_metadata_distinguishes_power_and_reference_anchor_banks():
    class MemoryGroup(dict):
        def __init__(self):
            super().__init__()
            self.attrs = {}

        def create_group(self, name):
            group = MemoryGroup()
            self[name] = group
            return group

    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.port_index = 1
    monitor.port_id = "modal"
    monitor.is_source = False
    monitor.excitation_mode_index = None
    monitor.mode_indices = (1,)
    monitor.owner = SimpleNamespace(
        direction="+",
        normal="x",
        plane_index=2,
        requested_anchor_policy="explicit",
        resolved_anchor_policy="explicit_nonpropagating_trimmed",
    )
    monitor.anchor_frequencies = np.asarray([1e9, 2e9, 3e9])
    monitor.anchor_mode_valid = np.asarray([[False], [True], [True]])
    monitor.anchor_mode_reference_valid = np.asarray([[True], [True], [True]])
    monitor.anchor_mode_propagating = np.asarray([[False], [True], [True]])
    monitor.anchor_balanced_power = np.asarray([[1.0], [2.0], [3.0]])
    monitor.anchor_neff = np.asarray([[-0.5j], [0.4 - 0.01j], [0.7 - 0.02j]])
    monitor.mode_anchor_policies = ("explicit_nonpropagating_trimmed",)
    monitor.result = EigenmodePortResult(
        frequency=np.asarray([1e9]),
        incident=np.zeros((1, 1), dtype=np.complex128),
        outgoing=np.zeros((1, 1), dtype=np.complex128),
        valid=np.zeros((1, 1), dtype=bool),
        condition_number=np.ones(1),
        generalized_valid=np.ones((1, 1), dtype=bool),
    )
    monitor.electric_gram = np.ones((1, 1, 1), dtype=np.complex128)
    monitor.power_matrix = np.ones((1, 1, 1), dtype=np.complex128)
    monitor.mode_decomposition_valid = np.ones((1, 1), dtype=bool)
    monitor.power_wave_valid = np.zeros((1, 1), dtype=bool)
    monitor.power_matrix_valid = np.zeros(1, dtype=bool)
    monitor.s_parameters = np.asarray([[0.5 + 0j]])
    monitor.s_generalized_valid = np.ones((1, 1), dtype=bool)
    monitor.s_valid = np.zeros((1, 1), dtype=bool)
    monitor.s_power_wave_valid = monitor.s_valid.copy()
    monitor.active_s_parameters = None

    base = MemoryGroup()
    monitor.write_hdf5(base)
    group = base["eigenmode_ports/port1"]

    np.testing.assert_array_equal(group.attrs["AnchorFrequencies"], [2e9, 3e9])
    np.testing.assert_array_equal(
        group.attrs["ReferenceAnchorFrequencies"],
        [1e9, 2e9, 3e9],
    )
    np.testing.assert_array_equal(group["anchor_mode_reference_valid"], [[1], [1], [1]])
    np.testing.assert_allclose(group["anchor_balanced_power"], [[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(group["anchor_complex_neff"], monitor.anchor_neff)
    np.testing.assert_array_equal(group["generalized_valid"], [[1]])
    np.testing.assert_array_equal(group["coefficient_valid"], group["generalized_valid"])
    np.testing.assert_array_equal(group["valid"], [[0]])
    np.testing.assert_array_equal(group["power_wave_valid"], group["valid"])
    np.testing.assert_array_equal(group["reference_basis_valid"], group["decomposition_valid"])
    np.testing.assert_array_equal(group["power_basis_valid"], group["power_normalization_valid"])
    np.testing.assert_array_equal(group["generalized_valid_S"], [[1]])
    np.testing.assert_array_equal(group["coefficient_valid_S"], group["generalized_valid_S"])
    np.testing.assert_array_equal(group["valid_S"], [[0]])
    np.testing.assert_array_equal(group["power_wave_valid_S"], group["valid_S"])
    assert group["generalized_valid"].shape == group["valid"].shape == (1, 1)
    assert group["generalized_valid_S"].shape == group["valid_S"].shape == (1, 1)
    assert group["valid"].dtype == np.uint8
    assert group["generalized_valid"].dtype == np.uint8


def test_sparameter_csv_contains_s11_and_each_s21_mode(tmp_path, monkeypatch):
    frequency = np.asarray([5e9])
    source = SimpleNamespace(
        is_source=True,
        port_index=1,
        excitation_mode_index=2,
        mode_indices=(1, 2),
        result=EigenmodePortResult(
            frequency=frequency,
            incident=np.asarray([[100 + 0j], [2 + 0j]]),
            outgoing=np.asarray([[1 + 0j], [0.5 + 0j]]),
            valid=np.asarray([[True], [True]]),
            condition_number=np.asarray([1.0]),
        ),
        finalise=lambda grid: None,
        mode_power_valid=np.ones((1, 2), dtype=bool),
        power_matrix_valid=np.ones(1, dtype=bool),
    )
    receiver = SimpleNamespace(
        is_source=False,
        port_index=2,
        mode_indices=(1, 2),
        result=EigenmodePortResult(
            frequency=frequency,
            incident=np.zeros((2, 1), dtype=np.complex128),
            outgoing=np.asarray([[1 + 0j], [0.5 + 0j]]),
            valid=np.asarray([[True], [True]]),
            condition_number=np.asarray([1.0]),
        ),
        finalise=lambda grid: None,
        mode_power_valid=np.ones((1, 2), dtype=bool),
        power_matrix_valid=np.ones(1, dtype=bool),
    )
    grid = SimpleNamespace(name="main_grid", eigenmodeports=[source, receiver])
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(output_file_path=tmp_path / "modal_run"),
    )

    csv_path = finalise_eigenmode_ports(grid)

    assert csv_path == tmp_path / "modal_run_sparameters.csv"
    assert b"\r\n" not in csv_path.read_bytes()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert "coefficient_magnitude_squared" in rows[0]
    assert "power_ratio" not in rows[0]
    assert "power_wave_valid" in rows[0]
    assert "generalized_valid" in rows[0]
    assert all(row["valid"] == row["power_wave_valid"] for row in rows)
    assert all(row["valid"] == row["generalized_valid"] for row in rows)
    values = {
        (int(row["destination_port"]), int(row["destination_mode"])): float(row["S_magnitude"])
        for row in rows
    }
    assert values[(1, 1)] == pytest.approx(0.5)
    assert values[(1, 2)] == pytest.approx(0.25)
    assert values[(2, 1)] == pytest.approx(0.5)
    assert values[(2, 2)] == pytest.approx(0.25)
    assert {int(row["source_mode"]) for row in rows} == {2}


def test_multiple_drives_write_active_sparameters_not_an_sparameter_column(
    tmp_path,
    monkeypatch,
):
    frequency = np.asarray([5e9, 6e9])

    def monitor(port_index, modes):
        return SimpleNamespace(
            is_source=True,
            port_index=port_index,
            excitation_mode_index=None,
            excitation_mode_indices=tuple(modes),
            mode_indices=tuple(modes),
            result=EigenmodePortResult(
                frequency=frequency,
                incident=np.full((len(modes), 2), 2 * port_index, dtype=np.complex128),
                outgoing=np.full((len(modes), 2), 0.5 * port_index, dtype=np.complex128),
                valid=np.tile([[True, False]], (len(modes), 1)),
                condition_number=np.ones(2),
                generalized_valid=np.ones((len(modes), 2), dtype=bool),
            ),
            finalise=lambda grid: None,
            mode_power_valid=np.tile([[True], [False]], (1, len(modes))),
            power_matrix_valid=np.ones(2, dtype=bool),
            drive_metadata=tuple(
                {
                    "mode": mode,
                    "amplitude": 1.0,
                    "power": 1.0,
                    "phase_deg": 30.0 * port_index,
                    "delay_s": 0.0,
                }
                for mode in modes
            ),
        )

    grid = SimpleNamespace(
        name="main_grid",
        eigenmodeports=[monitor(1, (1,)), monitor(2, (1,))],
    )
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(output_file_path=tmp_path / "driven"),
    )

    csv_path = finalise_eigenmode_ports(grid)

    assert csv_path == tmp_path / "driven_active_sparameters.csv"
    assert all(not hasattr(port, "s_parameters") for port in grid.eigenmodeports)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(float(row["active_S_magnitude"]) == pytest.approx(0.25) for row in rows)
    assert all(row["coefficient_valid"] == "1" for row in rows)
    assert [row["power_wave_valid"] for row in rows] == ["1", "0", "1", "0"]
    assert all(port.response_type == "driven" for port in grid.eigenmodeports)


def test_invalid_source_bin_does_not_invalidate_other_sparameter_bins(tmp_path, monkeypatch):
    frequency = np.asarray([5e9, 6e9])
    source = SimpleNamespace(
        is_source=True,
        port_index=1,
        excitation_mode_index=1,
        mode_indices=(1,),
        result=EigenmodePortResult(
            frequency=frequency,
            incident=np.asarray([[np.nan + 0j, 2 + 0j]]),
            outgoing=np.asarray([[np.nan + 0j, 0.5 + 0j]]),
            valid=np.asarray([[False, True]]),
            condition_number=np.asarray([np.inf, 1.0]),
        ),
        finalise=lambda grid: None,
        mode_power_valid=np.ones((2, 1), dtype=bool),
        power_matrix_valid=np.ones(2, dtype=bool),
    )
    receiver = SimpleNamespace(
        is_source=False,
        port_index=2,
        mode_indices=(1,),
        result=EigenmodePortResult(
            frequency=frequency,
            incident=np.asarray([[np.nan + 0j, 0 + 0j]]),
            outgoing=np.asarray([[np.nan + 0j, 1 + 0j]]),
            valid=np.asarray([[False, True]]),
            condition_number=np.asarray([np.inf, 1.0]),
        ),
        finalise=lambda grid: None,
        mode_power_valid=np.ones((2, 1), dtype=bool),
        power_matrix_valid=np.ones(2, dtype=bool),
    )
    grid = SimpleNamespace(name="main_grid", eigenmodeports=[source, receiver])
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(output_file_path=tmp_path / "partially_valid"),
    )

    csv_path = finalise_eigenmode_ports(grid)

    assert not receiver.s_valid[0, 0]
    assert receiver.s_valid[0, 1]
    assert np.isnan(receiver.s_parameters[0, 0])
    assert receiver.s_parameters[0, 1] == pytest.approx(0.5)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    invalid_rows = [row for row in rows if row["frequency_hz"] == "5000000000.0"]
    assert invalid_rows
    assert all(np.isnan(float(row["S_magnitude_db"])) for row in invalid_rows)


def test_below_cutoff_generalized_s_is_finite_but_not_power_wave_valid(
    tmp_path,
    monkeypatch,
):
    frequency = np.asarray([22e9, 28e9])
    source = SimpleNamespace(
        is_source=True,
        port_index=1,
        excitation_mode_index=1,
        mode_indices=(1,),
        result=EigenmodePortResult(
            frequency=frequency,
            # Deliberately incomparable generalized/power-wave coefficient
            # scales: each normalization class needs its own incident floor.
            incident=np.asarray([[2e9 + 0j, 2 + 0j]]),
            outgoing=np.asarray([[0.2 + 0j, 0.2 + 0j]]),
            valid=np.asarray([[False, True]]),
            condition_number=np.ones(2),
            generalized_valid=np.asarray([[True, True]]),
        ),
        finalise=lambda grid: None,
        mode_power_valid=np.asarray([[False], [True]]),
        power_matrix_valid=np.ones(2, dtype=bool),
    )
    receiver = SimpleNamespace(
        is_source=False,
        port_index=2,
        mode_indices=(1,),
        result=EigenmodePortResult(
            frequency=frequency,
            incident=np.zeros((1, 2), dtype=np.complex128),
            outgoing=np.asarray([[1e9 + 0j, 1 + 0j]]),
            valid=np.asarray([[False, True]]),
            condition_number=np.ones(2),
            generalized_valid=np.asarray([[True, True]]),
        ),
        finalise=lambda grid: None,
        mode_power_valid=np.asarray([[False], [True]]),
        power_matrix_valid=np.ones(2, dtype=bool),
    )
    grid = SimpleNamespace(name="main_grid", eigenmodeports=[source, receiver])
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(output_file_path=tmp_path / "cutoff_band"),
    )

    csv_path = finalise_eigenmode_ports(grid)

    assert receiver.s_generalized_valid[0].tolist() == [True, True]
    assert receiver.s_valid[0].tolist() == [False, True]
    assert receiver.s_power_wave_valid[0].tolist() == [False, True]
    assert receiver.s_parameters[0, 0] == pytest.approx(0.5)
    assert receiver.s_parameters[0, 1] == pytest.approx(0.5)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["destination_port"] == "2"]
    assert [row["valid"] for row in rows] == ["0", "1"]
    assert [row["power_wave_valid"] for row in rows] == ["0", "1"]
    assert [row["generalized_valid"] for row in rows] == ["1", "1"]


def test_monitor_rejects_invalid_interpolated_power(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={
                "float_or_double": np.float64,
                "complex": np.complex128,
            }
        ),
    )
    field = np.ones((1, 1), dtype=np.complex128)
    zero = np.zeros_like(field)
    owner = SimpleNamespace(
        transverse_axes=(1, 2),
        invariant_axis=None,
        normal_axis=0,
        direction="+",
        _linear_anchor_weights=lambda frequency, anchors: np.ones((1, 1)),
        _transverse_cell_shape=lambda: (1, 1),
        _modal_cross_power=lambda electric, magnetic, grid: 0.0,
        _average_to_transverse_cells=lambda values, component: values,
        _modal_basis_handedness=lambda: 1,
    )
    source = EigenmodePortMonitor(
        owner=owner,
        port_index=1,
        port_id="fallback",
        is_source=True,
        excitation_mode_index=1,
        mode_indices=(1,),
        anchor_frequencies=np.asarray([5e9]),
        anchor_e=[[[zero, field, zero]]],
        anchor_h=[[[zero, zero, field]]],
        anchor_neff=np.asarray([[1.0]]),
        dft_start=5e9,
        dft_stop=5e9,
        dft_points=1,
    )
    grid = SimpleNamespace(
        name="main_grid",
        dt=1e-12,
        dl=np.ones(3),
        eigenmodeports=[],
    )
    with pytest.raises(ValueError, match="invalid interpolated power 0"):
        source.prepare(grid)
