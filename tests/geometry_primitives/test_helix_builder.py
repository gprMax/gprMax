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

"""Direct tests for the helical-wire rasteriser ``build_helix``.

With ``pitch=0`` and an integer number of turns the centreline is a circle, so
the swept volume is a torus and can be checked against an analytic
rasterisation. Handedness, continuity, degenerate parameters, and tag writing
are tested separately.
"""

import numpy as np
import pytest

from gprMax.cython.geometry_primitives import build_helix

from .conftest import DL


def _build(grid, p1, p2, coil_radius, pitch, turns, wire_radius, right_handed=True, **overrides):
    kwargs = dict(
        averaging=True,
        dx=DL,
        dy=DL,
        dz=DL,
        pec_x=False,
        pec_y=False,
        pec_z=False,
        tag_data=None,
        tag_id=0,
    )
    kwargs.update(overrides)
    return build_helix(
        *p1,
        *p2,
        coil_radius,
        pitch,
        turns,
        wire_radius,
        right_handed,
        kwargs["dx"],
        kwargs["dy"],
        kwargs["dz"],
        1,
        1,
        1,
        1,
        kwargs["averaging"],
        kwargs["pec_x"],
        kwargs["pec_y"],
        kwargs["pec_z"],
        grid.solid,
        grid.rigidE,
        grid.rigidH,
        grid.ID,
        kwargs["tag_data"],
        kwargs["tag_id"],
    )


def _cell_centres(nx, ny, nz):
    axes = [(np.arange(n) + 0.5) * DL for n in (nx, ny, nz)]
    return np.meshgrid(*axes, indexing="ij")


def _pec_edge_components(grid):
    """Count connected components of the PEC edges in the ID array."""
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def add_edge(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for cell in np.argwhere(grid.solid):
        i, j, k = (int(value) for value in cell)
        for dj in (0, 1):
            for dk in (0, 1):
                if grid.ID[0, i, j + dj, k + dk] == 1:
                    add_edge((i, j + dj, k + dk), (i + 1, j + dj, k + dk))
        for di in (0, 1):
            for dk in (0, 1):
                if grid.ID[1, i + di, j, k + dk] == 1:
                    add_edge((i + di, j, k + dk), (i + di, j + 1, k + dk))
        for di in (0, 1):
            for dj in (0, 1):
                if grid.ID[2, i + di, j + dj, k] == 1:
                    add_edge((i + di, j + dj, k), (i + di, j + dj, k + 1))

    return len({find(node) for node in parent})


class TestHelixAnalytic:
    def test_single_turn_zero_pitch_matches_torus(self, grid_arrays):
        g = grid_arrays(40, 40, 40)
        cx, cy, cz = 20 * DL, 20 * DL, 20 * DL
        coil_radius = 10 * DL
        wire_radius = 2 * DL

        occupied = _build(
            g,
            (cx, cy, cz),
            (cx, cy, 30 * DL),
            coil_radius,
            0.0,
            1.0,
            wire_radius,
        )

        X, Y, Z = _cell_centres(40, 40, 40)
        distance = (np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - coil_radius) ** 2 + (Z - cz) ** 2
        exact = distance <= wire_radius ** 2
        mask = g.solid.astype(bool)

        assert occupied == int(mask.sum())
        assert occupied > 0
        # The sampled capsule chain approximates the torus from the inside;
        # a 0.5-cell sampling step keeps the mismatch to a thin surface band.
        assert np.count_nonzero(mask & exact) / np.count_nonzero(exact) >= 0.99

    def test_zero_pitch_ring_has_no_axial_advance(self, grid_arrays):
        g = grid_arrays(32, 32, 32)
        cz = 16 * DL

        _build(g, (16 * DL, 16 * DL, cz), (16 * DL, 16 * DL, 30 * DL), 8 * DL, 0.0, 1.0, 1.5 * DL)

        occupied_z = np.unique(np.argwhere(g.solid)[:, 2])
        assert occupied_z.min() >= int((cz - 1.5 * DL) / DL)
        assert occupied_z.max() <= int((cz + 1.5 * DL) / DL)


class TestHelixHandedness:
    def test_left_handed_mask_mirrors_right_handed(self, grid_arrays):
        right_grid = grid_arrays(32, 32, 32)
        left_grid = grid_arrays(32, 32, 32)
        p1 = (16 * DL, 16 * DL, 8 * DL)
        p2 = (16 * DL, 16 * DL, 30 * DL)

        _build(right_grid, p1, p2, 8 * DL, 4 * DL, 1.0, 1.5 * DL, True)
        _build(left_grid, p1, p2, 8 * DL, 4 * DL, 1.0, 1.5 * DL, False)

        # The builder's transverse basis for a +z axis is u = +y, v = -x,
        # so mirroring the right-handed wire in the yz-plane (x-index
        # reversal about the axis node) produces the left-handed wire.
        reflected = right_grid.solid[::-1, :, :]
        np.testing.assert_array_equal(reflected, left_grid.solid)


class TestHelixContinuity:
    @pytest.mark.parametrize("right_handed", (True, False))
    def test_turns_form_one_connected_wire(self, grid_arrays, right_handed):
        g = grid_arrays(48, 48, 48)
        _build(
            g,
            (24 * DL, 24 * DL, 8 * DL),
            (24 * DL, 24 * DL, 40 * DL),
            8 * DL,
            6 * DL,
            2.5,
            1.5 * DL,
            right_handed,
            averaging=False,
            pec_x=True,
            pec_y=True,
            pec_z=True,
        )

        assert _pec_edge_components(g) == 1

    @pytest.mark.parametrize(
        "wire_radius,components",
        [(0.0005, 7), (0.0006, 5)],
        ids=["seven-components", "five-components"],
    )
    def test_review_reproducer_pec_edge_components(self, grid_arrays, wire_radius, components):
        g = grid_arrays(64, 64, 64)
        _build(
            g,
            (0.032, 0.032, 0.008),
            (0.032, 0.032, 0.048),
            0.008,
            0.006,
            2.5,
            wire_radius,
            averaging=False,
            pec_x=True,
            pec_y=True,
            pec_z=True,
        )

        assert _pec_edge_components(g) == components

    def test_anisotropic_grid_uses_actual_pec_edges(self, grid_arrays):
        g = grid_arrays(64, 32, 32)
        _build(
            g,
            (0.032, 0.032, 0.008),
            (0.032, 0.032, 0.048),
            0.008,
            0.006,
            2.5,
            0.0008,
            averaging=False,
            dx=DL,
            dy=2 * DL,
            dz=2 * DL,
            pec_x=True,
            pec_y=True,
            pec_z=True,
        )

        assert _pec_edge_components(g) > 1


class TestHelixDegenerate:
    @pytest.mark.parametrize(
        "coil_radius,pitch,turns,wire_radius",
        [
            (0.0, 4 * DL, 1.0, 1.5 * DL),
            (-1.0 * DL, 4 * DL, 1.0, 1.5 * DL),
            (8 * DL, -1.0 * DL, 1.0, 1.5 * DL),
            (8 * DL, 4 * DL, 0.0, 1.5 * DL),
            (8 * DL, 4 * DL, -1.0, 1.5 * DL),
            (8 * DL, 4 * DL, 1.0, 0.0),
        ],
        ids=["zero-radius", "negative-radius", "negative-pitch", "zero-turns", "negative-turns", "zero-wire"],
    )
    def test_degenerate_parameters_write_nothing(self, grid_arrays, coil_radius, pitch, turns, wire_radius):
        g = grid_arrays()
        occupied = _build(
            g,
            (4 * DL, 4 * DL, 2 * DL),
            (4 * DL, 4 * DL, 6 * DL),
            coil_radius,
            pitch,
            turns,
            wire_radius,
        )

        assert occupied == 0
        assert not g.solid.any()
        assert not g.rigidE.any()
        assert not g.rigidH.any()
        assert not g.ID.any()

    def test_zero_length_axis_writes_nothing(self, grid_arrays):
        g = grid_arrays()
        p = (4 * DL, 4 * DL, 2 * DL)
        occupied = _build(g, p, p, 6 * DL, 4 * DL, 1.0, 1.5 * DL)

        assert occupied == 0
        assert not g.solid.any()


class TestHelixRigidAndTag:
    def test_hard_material_sets_rigid_at_written_cells(self, grid_arrays):
        g = grid_arrays(32, 32, 32)
        _build(
            g,
            (16 * DL, 16 * DL, 8 * DL),
            (16 * DL, 16 * DL, 24 * DL),
            8 * DL,
            4 * DL,
            1.0,
            1.5 * DL,
            averaging=False,
            pec_x=True,
            pec_y=True,
            pec_z=True,
        )

        cells = set(map(tuple, np.argwhere(g.solid)))
        assert cells
        for cell in cells:
            assert np.all(g.rigidE[(slice(None), *cell)] == 1)
            # PEC directions leave the magnetic ID slots untouched.
            assert np.all(g.ID[(slice(0, 3), *cell)] == 1)

    def test_tag_written_only_to_occupied_cells(self, grid_arrays):
        g = grid_arrays(32, 32, 32)
        tag_data = np.zeros((32, 32, 32), dtype=np.uint8)

        occupied = _build(
            g,
            (16 * DL, 16 * DL, 8 * DL),
            (16 * DL, 16 * DL, 24 * DL),
            8 * DL,
            4 * DL,
            1.0,
            1.5 * DL,
            tag_data=tag_data,
            tag_id=7,
        )

        cells = set(map(tuple, np.argwhere(g.solid)))
        np.testing.assert_array_equal(tag_data != 0, g.solid != 0)
        assert occupied == len(cells)
        assert np.all(tag_data[tuple(np.argwhere(g.solid).T)] == 7)


class TestHelixTranslation:
    def test_translation_by_reference_point_does_not_rescale(self, grid_arrays):
        near = grid_arrays(40, 40, 40)
        far = grid_arrays(40, 40, 40)
        offset = 8 * DL

        _build(near, (12 * DL, 12 * DL, 4 * DL), (12 * DL, 12 * DL, 20 * DL), 6 * DL, 3 * DL, 1.5, 1.2 * DL)
        _build(
            far,
            (12 * DL + offset, 12 * DL + offset, 4 * DL + offset),
            (12 * DL + offset, 12 * DL + offset, 20 * DL + offset),
            6 * DL,
            3 * DL,
            1.5,
            1.2 * DL,
        )

        np.testing.assert_array_equal(
            near.solid[:30, :30, :30],
            far.solid[8:38, 8:38, 8:38],
        )


pytestmark = pytest.mark.unit
