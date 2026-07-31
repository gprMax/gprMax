import numpy as np
import pytest

import gprMax.config as config
import gprMax.snapshots as snapshots_mod
from gprMax.cython.snapshots import calculate_snapshot_fields


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("3D", (1, 1, 1)),
        ("2D TMx", (1, 1, 1)),
        ("2D TMy", (1, 1, 1)),
        ("2D TMz", (1, 1, 1)),
        ("2D TEx", (0, 1, 1)),
        ("2D TEy", (1, 0, 1)),
        ("2D TEz", (1, 1, 0)),
    ],
)
def test_snapshot_axis_strides(monkeypatch, mode, expected):
    monkeypatch.setattr(
        config, "get_model_config", lambda: type("ModelConfig", (), {"mode": mode})()
    )

    assert snapshots_mod._snapshot_axis_strides() == expected


def test_zero_z_stride_avoids_averaging_tez_ex_with_boundary_zero():
    nx, ny, nz = 2, 2, 1
    ex = np.arange(3 * 3 * 2, dtype=np.float64).reshape(3, 3, 2)
    ex_snapshot = np.zeros((nx, ny, nz), dtype=np.float64)
    unused = np.zeros((1, 1, 1), dtype=np.float64)

    calculate_snapshot_fields(
        nx,
        ny,
        nz,
        1,
        True,
        False,
        False,
        False,
        False,
        False,
        ex,
        unused,
        unused,
        unused,
        unused,
        unused,
        ex_snapshot,
        unused,
        unused,
        unused,
        unused,
        unused,
        1,
        1,
        0,
    )

    expected = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            expected[i, j, 0] = (ex[i, j, 0] + ex[i, j + 1, 0]) / 2

    assert np.allclose(ex_snapshot, expected)


def test_zero_z_stride_preserves_tez_hz_value():
    nx, ny, nz = 2, 2, 1
    hz = np.arange(3 * 3 * 2, dtype=np.float64).reshape(3, 3, 2)
    hz_snapshot = np.zeros((nx, ny, nz), dtype=np.float64)
    unused = np.zeros((1, 1, 1), dtype=np.float64)

    calculate_snapshot_fields(
        nx,
        ny,
        nz,
        1,
        False,
        False,
        False,
        False,
        False,
        True,
        unused,
        unused,
        unused,
        unused,
        unused,
        hz,
        unused,
        unused,
        unused,
        unused,
        unused,
        hz_snapshot,
        1,
        1,
        0,
    )

    assert np.allclose(hz_snapshot, hz[:nx, :ny, :1])
