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

"""Unit tests for the array-driven rasterisers in
``gprMax/cython/geometry_primitives.pyx``.

``build_voxels_from_array`` stamps a pre-computed 3D block of material
IDs into the grid (used by ``GeometryObjectsRead`` and ``FractalBox``);
``build_voxels_from_array_mask`` does the same through a per-cell mask
that selects between the data value, a water material, and a grass
material (used by ``AddGrass``). Both bottom out in ``build_voxel``,
so these tests focus on the placement/offset arithmetic, the sentinel
and mask dispatch, and the material-ID offsetting.
"""

import numpy as np
import pytest

from gprMax.cython.geometry_primitives import (
    build_voxels_from_array,
    build_voxels_from_array_mask,
)

from .conftest import nonzero_set


def make_data(*shape, fill=0):
    return np.full(shape, fill, dtype=np.int16)


def make_mask(*shape, fill=0):
    return np.full(shape, fill, dtype=np.int8)


# Material-property lookups must cover every material ID passed to the Cython
# rasterisers. Empty arrays would make the unchecked memoryview indexing
# undefined and therefore platform-dependent.
_NON_PEC_LOOKUP = np.zeros(256, dtype=np.uint8)
_AVERAGABLE_LOOKUP = np.ones(256, dtype=np.uint8)


class TestBuildVoxelsFromArray:
    def test_data_block_lands_at_the_offset(self, grid_arrays):
        g = grid_arrays()
        data = make_data(2, 2, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    data[i, j, k] = i * 4 + j * 2 + k  # values 0..7

        build_voxels_from_array(
            1,
            2,
            3,
            10,
            True,
            _NON_PEC_LOOKUP,
            _AVERAGABLE_LOOKUP,
            data,
            g.solid,
            g.rigidE,
            g.rigidH,
            g.ID,
        )

        # data[i, j, k] maps to solid[xs + i, ys + j, zs + k], offset by
        # numexistmaterials.
        expected = {
            (1 + i, 2 + j, 3 + k): 10 + data[i, j, k]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        }
        assert nonzero_set(g.solid) == set(expected)
        for cell, value in expected.items():
            assert g.solid[cell] == value

    def test_negative_values_are_skipped_as_no_material(self, grid_arrays):
        g = grid_arrays()
        g.rigidE[:] = 1
        data = make_data(2, 1, 1, fill=-1)
        data[1, 0, 0] = 0

        build_voxels_from_array(
            2,
            2,
            2,
            5,
            True,
            _NON_PEC_LOOKUP,
            _AVERAGABLE_LOOKUP,
            data,
            g.solid,
            g.rigidE,
            g.rigidH,
            g.ID,
        )

        # Only the non-negative entry is written (0 + numexistmaterials).
        assert nonzero_set(g.solid) == {(3, 2, 2)}
        assert g.solid[3, 2, 2] == 5
        # The sentinel cell keeps its rigid flags; the written cell has
        # its flags handled by the averaging path of build_voxel.
        assert g.rigidE[:, 2, 2, 2].all()

    def test_block_overhanging_the_far_boundary_is_truncated(self, grid_arrays):
        # A 4-cell block placed at xs = 6 in an 8-cell domain: only the
        # first two x-slices of the data fit and they keep their values —
        # the block is truncated, not shifted.
        g = grid_arrays()
        data = make_data(4, 1, 1)
        for i in range(4):
            data[i, 0, 0] = i + 1

        build_voxels_from_array(
            6,
            0,
            0,
            0,
            True,
            _NON_PEC_LOOKUP,
            _AVERAGABLE_LOOKUP,
            data,
            g.solid,
            g.rigidE,
            g.rigidH,
            g.ID,
        )

        assert nonzero_set(g.solid) == {(6, 0, 0), (7, 0, 0)}
        assert g.solid[6, 0, 0] == 1
        assert g.solid[7, 0, 0] == 2

    def test_hard_write_stamps_id_with_the_offset_material(self, grid_arrays):
        g = grid_arrays()
        data = make_data(1, 1, 1, fill=5)

        build_voxels_from_array(
            2,
            2,
            2,
            1,
            False,
            _NON_PEC_LOOKUP,
            _AVERAGABLE_LOOKUP,
            data,
            g.solid,
            g.rigidE,
            g.rigidH,
            g.ID,
        )

        assert g.solid[2, 2, 2] == 6
        assert np.all(g.rigidE[:, 2, 2, 2] == 1)
        assert np.all(g.rigidH[:, 2, 2, 2] == 1)
        # All six ID components carry the same offset material ID.
        written = nonzero_set(g.ID)
        assert {comp for (comp, *_) in written} == set(range(6))
        assert all(g.ID[slot] == 6 for slot in written)


class TestBuildVoxelsFromArrayMask:
    def test_mask_selects_data_water_grass_or_skip(self, grid_arrays):
        g = grid_arrays()
        data = make_data(4, 1, 1, fill=7)
        mask = make_mask(4, 1, 1)
        mask[0, 0, 0] = 1  # use the data value
        mask[1, 0, 0] = 2  # water
        mask[2, 0, 0] = 3  # grass
        mask[3, 0, 0] = 0  # skip

        build_voxels_from_array_mask(
            2,
            3,
            4,
            20,
            30,
            True,
            _NON_PEC_LOOKUP,
            _AVERAGABLE_LOOKUP,
            mask,
            data,
            g.solid,
            g.rigidE,
            g.rigidH,
            g.ID,
        )

        assert nonzero_set(g.solid) == {(2, 3, 4), (3, 3, 4), (4, 3, 4)}
        assert g.solid[2, 3, 4] == 7  # mask 1 -> data value, no offset
        assert g.solid[3, 3, 4] == 20  # mask 2 -> waternumID
        assert g.solid[4, 3, 4] == 30  # mask 3 -> grassnumID
        assert g.solid[5, 3, 4] == 0  # mask 0 -> untouched

    def test_masked_out_cells_keep_their_rigid_flags(self, grid_arrays):
        g = grid_arrays()
        g.rigidE[:] = 1
        g.rigidH[:] = 1
        data = make_data(2, 1, 1, fill=5)
        mask = make_mask(2, 1, 1)
        mask[0, 0, 0] = 1

        build_voxels_from_array_mask(
            1,
            1,
            1,
            20,
            30,
            True,
            _NON_PEC_LOOKUP,
            _AVERAGABLE_LOOKUP,
            mask,
            data,
            g.solid,
            g.rigidE,
            g.rigidH,
            g.ID,
        )

        # The written cell (mask=1) has its rigid flags handled by the
        # averaging path; the skipped cell (mask=0) keeps its flags.
        assert g.rigidE[:, 2, 1, 1].all()
        assert g.rigidH[:, 2, 1, 1].all()

    def test_hard_write_stamps_id_with_the_mask_material(self, grid_arrays):
        g = grid_arrays()
        data = make_data(1, 1, 1, fill=5)
        mask = make_mask(1, 1, 1, fill=2)  # water

        build_voxels_from_array_mask(
            3,
            3,
            3,
            20,
            30,
            False,
            _NON_PEC_LOOKUP,
            _AVERAGABLE_LOOKUP,
            mask,
            data,
            g.solid,
            g.rigidE,
            g.rigidH,
            g.ID,
        )

        assert g.solid[3, 3, 3] == 20
        assert np.all(g.rigidE[:, 3, 3, 3] == 1)
        written = nonzero_set(g.ID)
        assert written
        assert all(g.ID[slot] == 20 for slot in written)

    def test_all_zero_mask_writes_nothing(self, grid_arrays):
        g = grid_arrays()
        data = make_data(2, 2, 2, fill=9)
        mask = make_mask(2, 2, 2)

        build_voxels_from_array_mask(
            1,
            1,
            1,
            20,
            30,
            True,
            _NON_PEC_LOOKUP,
            _AVERAGABLE_LOOKUP,
            mask,
            data,
            g.solid,
            g.rigidE,
            g.rigidH,
            g.ID,
        )

        assert not g.solid.any()
        assert not g.rigidE.any()
        assert not g.rigidH.any()
        assert not g.ID.any()


pytestmark = pytest.mark.unit
