from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import param

from gprMax.grid.fdtd_grid import FDTDGrid


def get_current_in_3d_grid(
    get_current_func: Callable[[int, int, int], float], shape: tuple[int, ...]
) -> np.ndarray:
    """Helper function to get current as a 3D grid

    Args:
        get_current_func: Function that returns the current value at a
            point on a grid.
        shape: shape of the grid

    Returns:
        result: 3D grid containing current values
    """
    result = np.empty(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                result[i, j, k] = get_current_func(i, j, k)
    return result


def tile(a: float, b: float, n: int = 1) -> np.ndarray:
    """Helper function to tile two numbers

    Args:
        a: first value
        b: second value
        n: number of repititions

    Returns:
        c: tiled numpy array
    """
    return np.tile([[[a, b], [b, a]], [[b, a], [a, b]]], (n, n, n))


@pytest.mark.parametrize(
    "dy,dz,Hy,Hz,expected",
    [
        (0, 0, (1, 3), (0.5, 0.8), 0),
        (0.1, 0.5, (0, 0), (0.5, 0.8), -0.15),
        (0.1, 0.5, (1, 3), (0, 0), 0.2),
        (0.1, 0, (1, 3), (0.5, 0.8), 0.2),
        (0, 0.5, (1, 3), (0.5, 0.8), -0.15),
        (0.1, 0.5, (1, 3), (0.5, 0.8), 0.05),
    ],
)
def test_calculate_Ix(dy, dz, Hy, Hz, expected, size=2):
    grid = FDTDGrid()
    grid.dy = dy
    grid.dz = dz
    grid.Hy = tile(Hy[0], Hy[1], size)
    grid.Hz = tile(Hz[0], Hz[1], size)

    actual_current = get_current_in_3d_grid(grid.calculate_Ix, grid.Hy.shape)
    expected_current = tile(expected, -expected, size)
    expected_current[:, 0, :] = 0
    expected_current[:, :, 0] = 0

    assert_allclose(actual_current, expected_current)


@pytest.mark.parametrize(
    "dx,dz,Hx,Hz,expected",
    [
        (0, 0, (1, 3), (0.5, 0.8), 0),
        (0.1, 0.5, (0, 0), (0.5, 0.8), -0.15),
        (0.1, 0.5, (1, 3), (0, 0), 0.2),
        (0.1, 0, (1, 3), (0.5, 0.8), 0.2),
        (0, 0.5, (1, 3), (0.5, 0.8), -0.15),
        (0.1, 0.5, (1, 3), (0.5, 0.8), 0.05),
    ],
)
def test_calculate_Iy(dx, dz, Hx, Hz, expected, size=2):
    grid = FDTDGrid()
    grid.dx = dx
    grid.dz = dz
    grid.Hx = tile(Hx[0], Hx[1], size)
    grid.Hz = tile(Hz[0], Hz[1], size)

    actual_current = get_current_in_3d_grid(grid.calculate_Iy, grid.Hx.shape)
    expected_current = tile(-expected, expected, size)
    expected_current[0, :, :] = 0
    expected_current[:, :, 0] = 0

    assert_allclose(actual_current, expected_current)


@pytest.mark.parametrize(
    "dx,dy,Hx,Hy,expected",
    [
        (0, 0, (1, 3), (0.5, 0.8), 0),
        (0.1, 0.5, (0, 0), (0.5, 0.8), -0.15),
        (0.1, 0.5, (1, 3), (0, 0), 0.2),
        (0.1, 0, (1, 3), (0.5, 0.8), 0.2),
        (0, 0.5, (1, 3), (0.5, 0.8), -0.15),
        (0.1, 0.5, (1, 3), (0.5, 0.8), 0.05),
    ],
)
def test_calculate_Iz(dx, dy, Hx, Hy, expected, size=2):
    grid = FDTDGrid()
    grid.dx = dx
    grid.dy = dy
    grid.Hx = tile(Hx[0], Hx[1], size)
    grid.Hy = tile(Hy[0], Hy[1], size)

    actual_current = get_current_in_3d_grid(grid.calculate_Iz, grid.Hx.shape)
    expected_current = tile(expected, -expected, size)
    expected_current[0, :, :] = 0
    expected_current[:, 0, :] = 0

    assert_allclose(actual_current, expected_current)
