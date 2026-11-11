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

"""Kernel-level guards: the thickness loops of ``build_triangle`` and
``build_cylindrical_sector`` must clamp against ``solid.shape`` like the
in-plane bounding boxes already do, so a slab handed an overhanging
``level + thickness`` writes only in-bounds cells.

A canary region carved from the same parent buffer directly after
``solid``'s memory proves no write escapes: under the global
``boundscheck(False)`` directive an out-of-bounds index corrupts
neighbouring memory, not the arrays' logical content, so array content
alone cannot detect the escape.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.cython.geometry_primitives import build_cylindrical_sector, build_triangle

from .conftest import DL, nonzero_set

pytestmark = pytest.mark.unit

NX, NY, NZ = 10, 8, 10
CELLS = NX * NY * NZ
CANARY = 32
NUM_ID = 5
IN_PLANE_CELLS = {(2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (2, 4)}


def _canary_arrays():
    """Grid arrays with production shapes, where ``solid``'s slots are
    followed in memory by a canary slice of one parent buffer."""
    parent = np.zeros(CELLS + CANARY, dtype=np.uint32)
    arrays = SimpleNamespace(
        solid=parent[:CELLS].reshape(NX, NY, NZ),
        canary=parent[CELLS:].copy(),
        rigidE=np.zeros((12, NX, NY, NZ), dtype=np.int8),
        rigidH=np.zeros((6, NX, NY, NZ), dtype=np.int8),
        ID=np.zeros((6, NX + 1, NY + 1, NZ + 1), dtype=np.uint32),
    )
    return arrays, parent


def test_triangle_overhang_writes_only_in_bounds_cells():
    """levelcells + thicknesscells = 11 on a 10-cell axis: cells 8 and 9
    must be written, and nothing beyond solid's memory."""
    arrays, parent = _canary_arrays()

    occupied = build_triangle(
        8 * DL,
        2 * DL,
        2 * DL,
        8 * DL,
        6 * DL,
        2 * DL,
        8 * DL,
        2 * DL,
        5 * DL,
        "x",
        3 * DL,
        DL,
        DL,
        DL,
        NUM_ID,
        NUM_ID,
        NUM_ID,
        NUM_ID,
        False,
        False,
        False,
        False,
        arrays.solid,
        arrays.rigidE,
        arrays.rigidH,
        arrays.ID,
    )

    assert nonzero_set(arrays.solid) == {(k, y, z) for (y, z) in IN_PLANE_CELLS for k in (8, 9)}
    assert occupied == 2 * len(IN_PLANE_CELLS)
    assert np.array_equal(parent[CELLS:], arrays.canary)


def test_cylindrical_sector_overhang_writes_only_in_bounds_cells():
    """Same overhang along the z axis: sector voxels confined to planes
    8 and 9, canary untouched."""
    arrays, parent = _canary_arrays()

    occupied = build_cylindrical_sector(
        3 * DL,
        3 * DL,
        8 * DL,
        0.0,
        np.pi / 2,
        2.5 * DL,
        "z",
        3 * DL,
        DL,
        DL,
        DL,
        NUM_ID,
        NUM_ID,
        NUM_ID,
        NUM_ID,
        False,
        False,
        False,
        False,
        arrays.solid,
        arrays.rigidE,
        arrays.rigidH,
        arrays.ID,
    )

    plane_cells = nonzero_set(arrays.solid)
    assert plane_cells and {z for (_, _, z) in plane_cells} == {8, 9}
    assert occupied == len(plane_cells)
    assert np.array_equal(parent[CELLS:], arrays.canary)


def test_triangle_fully_inside_unaffected():
    """A slab fully inside the axis writes all its cells as before."""
    arrays, parent = _canary_arrays()

    build_triangle(
        2 * DL,
        2 * DL,
        2 * DL,
        2 * DL,
        6 * DL,
        2 * DL,
        2 * DL,
        2 * DL,
        5 * DL,
        "x",
        3 * DL,
        DL,
        DL,
        DL,
        NUM_ID,
        NUM_ID,
        NUM_ID,
        NUM_ID,
        False,
        False,
        False,
        False,
        arrays.solid,
        arrays.rigidE,
        arrays.rigidH,
        arrays.ID,
    )

    assert nonzero_set(arrays.solid) == {(k, y, z) for (y, z) in IN_PLANE_CELLS for k in (2, 3, 4)}
    assert np.array_equal(parent[CELLS:], arrays.canary)
