"""Shared fixtures for the geometry-primitives test suite.

The Cython rasterisers in ``gprMax/cython/geometry_primitives.pyx``
mutate four numpy arrays that normally live on ``FDTDGrid``:

- ``solid``  ``(nx, ny, nz)``          uint32 — smoothed material ID per voxel
- ``rigidE`` ``(12, nx, ny, nz)``      int8   — E-edges excluded from averaging
- ``rigidH`` ``(6, nx, ny, nz)``       int8   — H-edges excluded from averaging
- ``ID``     ``(6, nx+1, ny+1, nz+1)`` uint32 — material ID per field component

The functions read nothing else — no ``gprMax.config``, no grid object —
so a ``SimpleNamespace`` carrying freshly zeroed arrays is a complete
test environment.

All shape tests use a uniform spatial discretisation of ``DL`` (1 mm) in
every axis, so cell index ``i`` maps to coordinate ``i * DL`` and cell
*centres* sit at ``(i + 0.5) * DL`` — the point every inside-check
samples.

The dispatch tests (``test_build_dispatch.py``) additionally drive the
user-object ``build()`` methods end-to-end through the real
``MainGridUserInput``, so their ``dispatch_grid`` stub carries the small
extra surface that layer reads: ``dl`` / ``size`` / ``within_bounds``
and a materials list.
"""

from types import SimpleNamespace

import numpy as np
import pytest

# Uniform spatial discretisation shared by the shape-builder tests.
DL = 0.001


def nonzero_set(arr):
    """Set of index tuples at which ``arr`` is nonzero."""
    return set(map(tuple, np.argwhere(np.asarray(arr))))


@pytest.fixture
def grid_arrays():
    """Factory for the four grid arrays with production shapes/dtypes.

    Usage:
        g = grid_arrays()          # 8 x 8 x 8 cells
        g = grid_arrays(4, 4, 4)
        build_box(..., g.solid, g.rigidE, g.rigidH, g.ID)
    """

    def _make(nx=8, ny=8, nz=8):
        return SimpleNamespace(
            nx=nx,
            ny=ny,
            nz=nz,
            solid=np.zeros((nx, ny, nz), dtype=np.uint32),
            rigidE=np.zeros((12, nx, ny, nz), dtype=np.int8),
            rigidH=np.zeros((6, nx, ny, nz), dtype=np.int8),
            ID=np.zeros((6, nx + 1, ny + 1, nz + 1), dtype=np.uint32),
        )

    return _make


def make_material(numID, ID, averagable=True):
    """Stub with the attributes the geometry ``build()`` methods read."""
    return SimpleNamespace(
        numID=numID, ID=ID, averagable=averagable, er=1.0, se=0.0, mr=1.0, sm=0.0
    )


class _StubGrid(SimpleNamespace):
    """FDTDGrid stand-in for driving ``build()`` through a real
    ``MainGridUserInput`` — implements the same ``within_bounds``
    contract (raises ``ValueError`` carrying the axis letter)."""

    def within_bounds(self, p):
        if p[0] < 0 or p[0] > self.nx:
            raise ValueError("x")
        if p[1] < 0 or p[1] > self.ny:
            raise ValueError("y")
        if p[2] < 0 or p[2] > self.nz:
            raise ValueError("z")
        return True


@pytest.fixture
def dispatch_grid(grid_arrays):
    """Factory for a grid stub the geometry ``build()`` methods can run
    against end-to-end: the four arrays plus discretisation, size,
    bounds checking, and a materials list (``pec``/``free_space``
    builtins plus averagable ``metal``, ``mat_a``, ``mat_b``)."""

    def _make(nx=8, ny=8, nz=8):
        arrays = grid_arrays(nx, ny, nz)
        return _StubGrid(
            nx=nx,
            ny=ny,
            nz=nz,
            dx=DL,
            dy=DL,
            dz=DL,
            dl=np.array([DL, DL, DL]),
            size=np.array([nx, ny, nz]),
            averagevolumeobjects=True,
            materials=[
                make_material(0, "pec", averagable=False),
                make_material(1, "free_space"),
                make_material(2, "metal"),
                make_material(3, "mat_a"),
                make_material(4, "mat_b"),
            ],
            solid=arrays.solid,
            rigidE=arrays.rigidE,
            rigidH=arrays.rigidH,
            ID=arrays.ID,
        )

    return _make
