import csv
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.eigenmode_ports import (
    DFT_PHASE_REANCHOR_INTERVAL,
    EigenmodePortMonitor,
    EigenmodePortResult,
    accumulate_eigenmode_dft,
    finalise_eigenmode_ports,
)


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


def test_multimode_gram_solve_separates_incident_and_outgoing_waves(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"complex": np.complex128}),
    )
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.electric_dft = np.asarray([[2.5, 2.75]], dtype=np.complex128)
    monitor.magnetic_dft = np.asarray([[1.5, 3.25]], dtype=np.complex128)
    monitor.electric_gram = np.eye(2, dtype=np.complex128)[np.newaxis]
    monitor.magnetic_gram = np.eye(2, dtype=np.complex128)[np.newaxis]
    monitor.frequency = np.asarray([1e9])
    monitor.neff = np.zeros((1, 2), dtype=np.complex128)
    monitor.mode_power_valid = np.ones((1, 2), dtype=bool)
    monitor.power_matrix_valid = np.ones(1, dtype=bool)
    monitor.owner = SimpleNamespace(normal_axis=0)
    monitor.magnetic_side = -1

    result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))

    np.testing.assert_allclose(result.incident[:, 0], (2, 3))
    np.testing.assert_allclose(result.outgoing[:, 0], (0.5, -0.25))
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

    assert result.condition_number[0] == pytest.approx(1e5, rel=1e-6)
    assert result.valid[:, 0].tolist() == [expected_valid, expected_valid]


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
    values = {(int(row["destination_port"]), int(row["destination_mode"])): float(row["S_magnitude"]) for row in rows}
    assert values[(1, 1)] == pytest.approx(0.5)
    assert values[(1, 2)] == pytest.approx(0.25)
    assert values[(2, 1)] == pytest.approx(0.5)
    assert values[(2, 2)] == pytest.approx(0.25)
    assert {int(row["source_mode"]) for row in rows} == {2}


def test_invalid_source_bin_does_not_invalidate_other_sparameter_bins(
    tmp_path, monkeypatch
):
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


def test_fallback_normalized_source_marks_every_csv_row_invalid(
    tmp_path, monkeypatch
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
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(output_file_path=tmp_path / "fallback"),
    )
    source.prepare(grid)
    grid.eigenmodeports.append(source)
    source.electric_dft[:] = source.electric_gram[:, 0, :]
    source.magnetic_dft[:] = source.magnetic_gram[:, 0, :]

    csv_path = finalise_eigenmode_ports(grid)

    assert not source.mode_power_valid[0, 0]
    assert not source.power_matrix_valid[0]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(row["valid"] == "0" for row in rows)
