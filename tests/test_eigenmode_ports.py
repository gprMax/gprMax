import csv
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.eigenmode_ports import (
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
    monitor.owner = SimpleNamespace(normal_axis=0)
    monitor.magnetic_side = -1

    result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))

    np.testing.assert_allclose(result.incident[:, 0], (2, 3))
    np.testing.assert_allclose(result.outgoing[:, 0], (0.5, -0.25))
    assert result.valid[:, 0].all()


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
    )
    grid = SimpleNamespace(name="main_grid", eigenmodeports=[source, receiver])
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(output_file_path=tmp_path / "modal_run"),
    )

    csv_path = finalise_eigenmode_ports(grid)

    assert csv_path == tmp_path / "modal_run_sparameters.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
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
