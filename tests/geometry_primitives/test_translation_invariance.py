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

"""Origin-invariance tests for curved and planar geometry rasterisers.

MPI ranks and HSG subgrids translate global geometry into a local coordinate
frame before invoking these shared builders. An integer-cell translation must
therefore translate the voxel mask exactly, without changing its shape.
"""

import numpy as np
import pytest

from gprMax.cython.geometry_primitives import (
    build_cone,
    build_cylinder,
    build_cylindrical_sector,
    build_ellipsoid,
    build_sphere,
    build_triangle,
)

from .conftest import DL


def _material_args(grid):
    return (
        1,
        1,
        1,
        1,
        True,
        False,
        False,
        False,
        grid.solid,
        grid.rigidE,
        grid.rigidH,
        grid.ID,
    )


def _sphere(grid, shift):
    build_sphere(12 + shift, 13 + shift, 14 + shift, 3.2 * DL, DL, DL, DL, *_material_args(grid))


def _ellipsoid(grid, shift):
    build_ellipsoid(
        12 + shift,
        13 + shift,
        14 + shift,
        4.1 * DL,
        2.7 * DL,
        3.3 * DL,
        DL,
        DL,
        DL,
        *_material_args(grid),
    )


def _cylinder(grid, shift):
    s = shift * DL
    build_cylinder(
        8 * DL + s,
        9 * DL + s,
        10 * DL + s,
        18 * DL + s,
        16 * DL + s,
        15 * DL + s,
        2.4 * DL,
        DL,
        DL,
        DL,
        *_material_args(grid),
    )


def _cone(grid, shift):
    s = shift * DL
    build_cone(
        8 * DL + s,
        9 * DL + s,
        10 * DL + s,
        18 * DL + s,
        16 * DL + s,
        15 * DL + s,
        1.2 * DL,
        3.1 * DL,
        DL,
        DL,
        DL,
        *_material_args(grid),
    )


def _triangle(grid, shift):
    s = shift * DL
    build_triangle(
        8 * DL + s,
        8 * DL + s,
        12 * DL + s,
        20 * DL + s,
        9 * DL + s,
        12 * DL + s,
        13 * DL + s,
        21 * DL + s,
        12 * DL + s,
        "z",
        3 * DL,
        DL,
        DL,
        DL,
        *_material_args(grid),
    )


def _sector(grid, shift):
    s = shift * DL
    build_cylindrical_sector(
        14 * DL + s,
        14 * DL + s,
        10 * DL + s,
        np.pi / 7,
        1.4 * np.pi,
        5.2 * DL,
        "z",
        4 * DL,
        DL,
        DL,
        DL,
        *_material_args(grid),
    )


@pytest.mark.parametrize("builder", [_sphere, _ellipsoid, _cylinder, _cone, _triangle, _sector])
def test_integer_cell_translation_preserves_voxel_mask(grid_arrays, builder):
    shift = 29
    extent = 28
    base = grid_arrays(70, 70, 70)
    translated = grid_arrays(70, 70, 70)

    builder(base, 0)
    builder(translated, shift)

    np.testing.assert_array_equal(
        base.solid[:extent, :extent, :extent],
        translated.solid[
            shift : shift + extent,
            shift : shift + extent,
            shift : shift + extent,
        ],
    )


def test_cone_endpoint_reversal_preserves_geometry_and_radius_assignment(grid_arrays):
    forward = grid_arrays(32, 32, 32)
    reverse = grid_arrays(32, 32, 32)
    p1 = (6 * DL, 12 * DL, 12 * DL)
    p2 = (22 * DL, 12 * DL, 12 * DL)

    build_cone(*p1, *p2, 1.2 * DL, 4.2 * DL, DL, DL, DL, *_material_args(forward))
    build_cone(*p2, *p1, 4.2 * DL, 1.2 * DL, DL, DL, DL, *_material_args(reverse))

    np.testing.assert_array_equal(forward.solid, reverse.solid)
