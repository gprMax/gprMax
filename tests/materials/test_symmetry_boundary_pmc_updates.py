"""Tests for the per-iteration PMC ghost-node E update
(gprMax.symmetry_boundaries.update_symmetry_boundaries_electric_normal), the
face-interior (non-edge) piece of the #symmetry_boundary PMC feature.

Ghost-node derivation: tangential H is odd under the PMC mirror, so the
"ghost" H node just outside the domain equals minus the real interior H node
it mirrors. Substituting into the standard curl term collapses the missing
outside-neighbour difference into double one real H value. For a "0" face
(x0/y0/z0) the doubled term uses the wall's own H index with the bulk
kernel's own sign; for a "max" face (xmax/ymax/zmax) it uses the
interior-adjacent H index with the opposite sign - both derived and
cross-checked against apply_TFSF_conditions_electric's existing identical
asymmetry.

Each expected-value helper below is a plain Python loop (not vectorized),
independently re-deriving the same formula the implementation computes with
numpy - so a transcription bug in one wouldn't be masked by the same bug in
the other.
"""
import numpy as np

from gprMax.cython.symmetry_boundaries import (
    update_symmetry_boundary_electric_x0,
    update_symmetry_boundary_electric_xmax,
    update_symmetry_boundary_electric_y0,
    update_symmetry_boundary_electric_ymax,
    update_symmetry_boundary_electric_z0,
    update_symmetry_boundary_electric_zmax,
)
from gprMax.grid.fdtd_grid import FDTDGrid

nx, ny, nz = 6, 5, 4
ca, cb1, cb2, cb3 = 0.7, 0.11, 0.22, 0.33

_FACE_FUNCS = {
    "x0": update_symmetry_boundary_electric_x0,
    "xmax": update_symmetry_boundary_electric_xmax,
    "y0": update_symmetry_boundary_electric_y0,
    "ymax": update_symmetry_boundary_electric_ymax,
    "z0": update_symmetry_boundary_electric_z0,
    "zmax": update_symmetry_boundary_electric_zmax,
}


def _update_face_interior(grid, face):
    """Thin wrapper matching the old Python-level signature, calling the
    real Cython implementation directly (nthreads=1 - single-threaded is
    enough for these small, deterministic correctness checks)."""
    _FACE_FUNCS[face](
        grid.nx,
        grid.ny,
        grid.nz,
        1,
        grid.updatecoeffsE,
        grid.ID,
        grid.Ex,
        grid.Ey,
        grid.Ez,
        grid.Hx,
        grid.Hy,
        grid.Hz,
    )


def _make_grid():
    grid = FDTDGrid()
    grid.nx, grid.ny, grid.nz = nx, ny, nz
    # A single, distinguishable material (numID 1) everywhere.
    grid.ID = np.ones((6, nx + 1, ny + 1, nz + 1), dtype=np.uint32)
    grid.updatecoeffsE = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [ca, cb1, cb2, cb3, 0.0],
        ]
    )

    def field(offset):
        arr = np.zeros((nx + 1, ny + 1, nz + 1))
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    arr[i, j, k] = 1000 * i + 100 * j + 10 * k + offset
        return arr

    grid.Ex, grid.Ey, grid.Ez = field(1), field(2), field(3)
    grid.Hx, grid.Hy, grid.Hz = field(4), field(5), field(6)
    return grid


def _expected_x0(grid):
    Ey, Ez = grid.Ey.copy(), grid.Ez.copy()
    for j in range(ny):
        for k in range(1, nz):
            Ey[0, j, k] = (
                ca * grid.Ey[0, j, k]
                + cb3 * (grid.Hx[0, j, k] - grid.Hx[0, j, k - 1])
                - cb1 * (2 * grid.Hz[0, j, k])
            )
    for j in range(1, ny):
        for k in range(nz):
            Ez[0, j, k] = (
                ca * grid.Ez[0, j, k]
                - cb2 * (grid.Hx[0, j, k] - grid.Hx[0, j - 1, k])
                + cb1 * (2 * grid.Hy[0, j, k])
            )
    return Ey, Ez


def _expected_xmax(grid):
    Ey, Ez = grid.Ey.copy(), grid.Ez.copy()
    for j in range(ny):
        for k in range(1, nz):
            Ey[nx, j, k] = (
                ca * grid.Ey[nx, j, k]
                + cb3 * (grid.Hx[nx, j, k] - grid.Hx[nx, j, k - 1])
                + cb1 * (2 * grid.Hz[nx - 1, j, k])
            )
    for j in range(1, ny):
        for k in range(nz):
            Ez[nx, j, k] = (
                ca * grid.Ez[nx, j, k]
                - cb2 * (grid.Hx[nx, j, k] - grid.Hx[nx, j - 1, k])
                - cb1 * (2 * grid.Hy[nx - 1, j, k])
            )
    return Ey, Ez


def _expected_y0(grid):
    Ex, Ez = grid.Ex.copy(), grid.Ez.copy()
    for i in range(nx):
        for k in range(1, nz):
            Ex[i, 0, k] = (
                ca * grid.Ex[i, 0, k]
                - cb3 * (grid.Hy[i, 0, k] - grid.Hy[i, 0, k - 1])
                + cb2 * (2 * grid.Hz[i, 0, k])
            )
    for i in range(1, nx):
        for k in range(nz):
            Ez[i, 0, k] = (
                ca * grid.Ez[i, 0, k]
                + cb1 * (grid.Hy[i, 0, k] - grid.Hy[i - 1, 0, k])
                - cb2 * (2 * grid.Hx[i, 0, k])
            )
    return Ex, Ez


def _expected_ymax(grid):
    Ex, Ez = grid.Ex.copy(), grid.Ez.copy()
    for i in range(nx):
        for k in range(1, nz):
            Ex[i, ny, k] = (
                ca * grid.Ex[i, ny, k]
                - cb3 * (grid.Hy[i, ny, k] - grid.Hy[i, ny, k - 1])
                - cb2 * (2 * grid.Hz[i, ny - 1, k])
            )
    for i in range(1, nx):
        for k in range(nz):
            Ez[i, ny, k] = (
                ca * grid.Ez[i, ny, k]
                + cb1 * (grid.Hy[i, ny, k] - grid.Hy[i - 1, ny, k])
                + cb2 * (2 * grid.Hx[i, ny - 1, k])
            )
    return Ex, Ez


def _expected_z0(grid):
    Ex, Ey = grid.Ex.copy(), grid.Ey.copy()
    for i in range(nx):
        for j in range(1, ny):
            Ex[i, j, 0] = (
                ca * grid.Ex[i, j, 0]
                + cb2 * (grid.Hz[i, j, 0] - grid.Hz[i, j - 1, 0])
                - cb3 * (2 * grid.Hy[i, j, 0])
            )
    for i in range(1, nx):
        for j in range(ny):
            Ey[i, j, 0] = (
                ca * grid.Ey[i, j, 0]
                - cb1 * (grid.Hz[i, j, 0] - grid.Hz[i - 1, j, 0])
                + cb3 * (2 * grid.Hx[i, j, 0])
            )
    return Ex, Ey


def _expected_zmax(grid):
    Ex, Ey = grid.Ex.copy(), grid.Ey.copy()
    for i in range(nx):
        for j in range(1, ny):
            Ex[i, j, nz] = (
                ca * grid.Ex[i, j, nz]
                + cb2 * (grid.Hz[i, j, nz] - grid.Hz[i, j - 1, nz])
                + cb3 * (2 * grid.Hy[i, j, nz - 1])
            )
    for i in range(1, nx):
        for j in range(ny):
            Ey[i, j, nz] = (
                ca * grid.Ey[i, j, nz]
                - cb1 * (grid.Hz[i, j, nz] - grid.Hz[i - 1, j, nz])
                - cb3 * (2 * grid.Hx[i, j, nz - 1])
            )
    return Ex, Ey


_FACES = {
    "x0": (_expected_x0, ("Ey", "Ez")),
    "xmax": (_expected_xmax, ("Ey", "Ez")),
    "y0": (_expected_y0, ("Ex", "Ez")),
    "ymax": (_expected_ymax, ("Ex", "Ez")),
    "z0": (_expected_z0, ("Ex", "Ey")),
    "zmax": (_expected_zmax, ("Ex", "Ey")),
}


def test_face_interior_matches_independent_hand_derivation():
    for face, (expected_fn, components) in _FACES.items():
        grid = _make_grid()
        expected = expected_fn(grid)

        _update_face_interior(grid, face)

        actual = tuple(getattr(grid, c) for c in components)
        for a, e, name in zip(actual, expected, components):
            assert np.allclose(a, e), f"{face}/{name} mismatch"


def test_face_interior_does_not_touch_the_other_four_components():
    """Only the two tangential components for the given face should change -
    the third E component and the H arrays must be left untouched."""
    for face, (_, components) in _FACES.items():
        grid = _make_grid()
        untouched = [c for c in ("Ex", "Ey", "Ez") if c not in components]
        before = {c: getattr(grid, c).copy() for c in untouched}
        before_H = {c: getattr(grid, c).copy() for c in ("Hx", "Hy", "Hz")}

        _update_face_interior(grid, face)

        for c in untouched:
            assert np.array_equal(getattr(grid, c), before[c]), f"{face}: {c} changed unexpectedly"
        for c in ("Hx", "Hy", "Hz"):
            assert np.array_equal(getattr(grid, c), before_H[c]), f"{face}: {c} changed unexpectedly"
