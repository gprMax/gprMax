# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Regression tests for domain-boundary-plane material IDs produced by the
dielectric-smoothing (averaging) geometry build.

Bug: build_electric_components()/build_magnetic_components()
(gprMax/cython/yee_cell_build.pyx) only ever computed each field component's
ID over the interior of its material-dependency axis/axes. The tangential E
(and corresponding H) components that sit exactly on a domain-boundary plane
(e.g. Ex at j=0/j=ny) were never written by either the main loop or the
per-component "extra loop" patches, regardless of what material a primitive
(box, cylinder, etc.) actually placed there in the averaging/dielectric-
smoothed build path - they were silently left at the array's free_space
default. This is inert for the standard FDTD update (which never reads those
same boundary planes), but is wrong for geometry inspection/export and for
any future feature (e.g. PMC symmetry boundaries) that reads ID at those
positions. The fix uses a "clamp to edge" scheme: a missing out-of-domain
neighbour is clamped onto the cell that does exist, collapsing the usual
4-cell/2-cell average to the correct reduced-neighbour result.
"""
import numpy as np
from numpy.testing import assert_array_equal

from gprMax.cython.geometry_primitives import build_voxel
from gprMax.cython.yee_cell_build import (
    build_electric_components,
    build_magnetic_components,
)
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material


def _build_grid(nx, ny, nz, materials):
    grid = FDTDGrid()
    grid.nx, grid.ny, grid.nz = nx, ny, nz
    grid.materials = materials
    grid.initialise_geometry_arrays()
    return grid


def _run_build(grid):
    build_electric_components(grid.solid, grid.rigidE, grid.ID, grid)
    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid)


def _valid_region(component, nx, ny, nz):
    """The ID array is uniformly (6, nx+1, ny+1, nz+1), but each component
    is only ever written over its own valid sub-range: full range on its
    dependency axis/axes, but only 0..n-1 (not n) on its own/free axis/axes
    (e.g. Ex's own axis i spans 0..nx-1, never nx). Slots outside a
    component's valid range are never written and stay at whatever
    initialise_geometry_arrays() filled them with - comparing the raw,
    uniformly-shaped array directly against an expected uniform value would
    incorrectly include those untouched slots.
    """
    ranges = {
        "Ex": (nx, ny + 1, nz + 1),
        "Ey": (nx + 1, ny, nz + 1),
        "Ez": (nx + 1, ny + 1, nz),
        "Hx": (nx + 1, ny, nz),
        "Hy": (nx, ny + 1, nz),
        "Hz": (nx, ny, nz + 1),
    }
    ni, nj, nk = ranges[component]
    return np.s_[:ni, :nj, :nk]


def test_uniform_material_reaches_every_boundary_plane():
    """A single material filling the whole grid must produce that material's
    ID everywhere in every component's array, including both the near (0)
    and far (n) domain-boundary plane of every dependency axis - previously
    those boundary entries were left at free_space instead.
    """
    free_space = Material(0, "free_space")
    matA = Material(1, "matA")
    matA.er = 3.0

    grid = _build_grid(3, 3, 3, [free_space, matA])
    grid.solid[:, :, :] = matA.numID
    _run_build(grid)

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        idx = grid.IDlookup[component]
        region = _valid_region(component, grid.nx, grid.ny, grid.nz)
        actual = np.asarray(grid.ID[idx])[region]
        assert_array_equal(
            actual,
            np.full(actual.shape, matA.numID, dtype=np.uint32),
            err_msg=f"{component} ID not uniform across its valid range (including boundary planes)",
        )


def test_thin_axis_degenerate_boundary_planes_collapse_correctly():
    """When a dependency axis has only 1 cell (as on a 2D-mode grid's
    invariant axis), the near (0) and far (1) boundary planes both clamp
    onto that same single cell. Must not crash and must still produce the
    correct material, not free_space.
    """
    free_space = Material(0, "free_space")
    matA = Material(1, "matA")
    matA.er = 5.0

    grid = _build_grid(2, 1, 2, [free_space, matA])
    grid.solid[:, :, :] = matA.numID
    _run_build(grid)

    for component in ("Ex", "Hy"):
        idx = grid.IDlookup[component]
        region = _valid_region(component, grid.nx, grid.ny, grid.nz)
        actual = np.asarray(grid.ID[idx])[region]
        assert_array_equal(
            actual, np.full(actual.shape, matA.numID, dtype=np.uint32)
        )


def test_ez_boundary_plane_averages_across_its_non_clamped_dependency_axis():
    """Ez depends on i and j. At the j=0 (and j=ny) wall, the j-side of the
    average collapses (no cell exists across that wall), but the i-side
    does not - two different materials sitting side-by-side along x, both
    at the wall, must still be genuinely averaged there (a reduced 2-way
    average, not free_space and not an arbitrary pick of one).
    """
    free_space = Material(0, "free_space")
    matA = Material(1, "matA")
    matA.er = 3.0
    matB = Material(2, "matB")
    matB.er = 6.0

    grid = _build_grid(2, 2, 1, [free_space, matA, matB])
    grid.solid[0, :, :] = matA.numID
    grid.solid[1, :, :] = matB.numID
    _run_build(grid)

    idEz = grid.IDlookup["Ez"]
    for j in (0, grid.ny):
        compound_numid = grid.ID[idEz, 1, j, 0]
        compound = next(m for m in grid.materials if m.numID == compound_numid)
        assert compound.er == np.mean([matA.er, matB.er]), (
            f"Ez at wall j={j} should be the 2-way average of matA/matB across x, "
            f"got material {compound.ID!r} with er={compound.er}"
        )


def test_hx_boundary_plane_preserves_distinct_values_along_its_own_axes():
    """Hx depends only on i. At the i=0 wall there is no averaging (nothing
    exists across that wall), but Hx's own (non-dependency) axes j, k must
    still each get their own distinct, correctly-assigned material - not
    merged together and not left at free_space.
    """
    free_space = Material(0, "free_space")
    matA = Material(1, "matA")
    matB = Material(2, "matB")

    grid = _build_grid(2, 2, 1, [free_space, matA, matB])
    grid.solid[:, 0, :] = matA.numID
    grid.solid[:, 1, :] = matB.numID
    _run_build(grid)

    idHx = grid.IDlookup["Hx"]
    for i in (0, grid.nx):
        assert grid.ID[idHx, i, 0, 0] == matA.numID
        assert grid.ID[idHx, i, 1, 0] == matB.numID


def test_rigid_material_at_extreme_corner_is_not_overwritten_by_averaging():
    """Regression guard for the get_rigid_* bounds-guard fix
    (yee_cell_setget_rigid.pyx): on a 1-cell grid, every ID position is a
    dependency-axis far-wall (n=1) for at least one component. Whatever
    build_voxel wrote for a rigid (non-averaged) material must be preserved
    exactly by the averaging pass - and querying get_rigid_* at these
    extreme positions must not crash (out-of-bounds read) or incorrectly
    report "not rigid" and overwrite it.

    Compares a before/after snapshot rather than an expected value - this
    keeps the test agnostic to build_voxel's own exact H/E placement
    (covered precisely by tests/materials/test_pec_h_components.py), and
    focused purely on what's in scope here: the averaging pass must never
    touch a position build_voxel already marked rigid.
    """
    free_space = Material(0, "free_space")
    matA = Material(1, "matA")

    grid = _build_grid(1, 1, 1, [free_space, matA])
    build_voxel(
        0, 0, 0,
        matA.numID, matA.numID, matA.numID, matA.numID,
        False, False, False, False,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )
    before = np.array(grid.ID, copy=True)

    _run_build(grid)

    assert_array_equal(
        np.asarray(grid.ID),
        before,
        err_msg="averaging pass modified ID values already set by a rigid build_voxel call",
    )
