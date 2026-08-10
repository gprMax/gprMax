"""Unit tests for the shape rasterisers in
``gprMax/cython/geometry_primitives.pyx``.

Every builder follows the same recipe: compute an integer bounding box
in cell coordinates, clamp it to the domain, and stamp ``build_voxel``
(or a face setter for zero-thickness shapes) into each cell whose
*centre* satisfies the shape's inside-check. These tests pin the exact
set of cells each shape writes on a small grid with ``DL`` (1 mm)
discretisation, plus the averaging/hard split and the domain clamps.

Shape parameters are chosen so no cell centre lands exactly on a shape
boundary — verdicts stay stable across the float32 arithmetic the
Cython layer uses.
"""

import math

import numpy as np
import pytest

from gprMax.cython.geometry_primitives import (
    build_box,
    build_cone,
    build_cylinder,
    build_cylindrical_sector,
    build_ellipsoid,
    build_sphere,
    build_triangle,
)

from .conftest import DL, nonzero_set

NUM_ID = 1
NUM_IDX = 2
NUM_IDY = 3
NUM_IDZ = 4


def cells_inside(g, predicate):
    """Cells of grid ``g`` whose centre (in cell units) satisfies ``predicate``."""
    return {
        (i, j, k)
        for i in range(g.nx)
        for j in range(g.ny)
        for k in range(g.nz)
        if predicate(i + 0.5, j + 0.5, k + 0.5)
    }


class TestBuildBox:
    def test_averaging_writes_solid_and_clears_rigid(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        g.rigidE[:] = 1
        g.rigidH[:] = 1

        build_box(1, 4, 1, 3, 1, 2, NUM_ID, NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        expected = {(i, j, 1) for i in range(1, 4) for j in range(1, 3)}
        assert nonzero_set(g.solid) == expected
        assert all(g.solid[c] == NUM_ID for c in expected)
        # Rigid flags cleared exactly at the box cells, untouched elsewhere.
        for cell in expected:
            assert not g.rigidE[(slice(None), *cell)].any()
            assert not g.rigidH[(slice(None), *cell)].any()
        assert np.count_nonzero(g.rigidE == 0) == 12 * len(expected)
        assert np.count_nonzero(g.rigidH == 0) == 6 * len(expected)
        assert not g.ID.any()

    def test_hard_box_stamps_interior_and_trailing_faces(self, grid_arrays):
        g = grid_arrays(6, 6, 6)
        xs, xf, ys, yf, zs, zf = 1, 3, 1, 3, 1, 3

        build_box(xs, xf, ys, yf, zs, zf, NUM_ID, NUM_IDX, NUM_IDY, NUM_IDZ,
                  False, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        box = {(i, j, k)
               for i in range(xs, xf) for j in range(ys, yf) for k in range(zs, zf)}
        assert nonzero_set(g.solid) == box
        for cell in box:
            assert np.all(g.rigidE[(slice(None), *cell)] == 1)
            assert np.all(g.rigidH[(slice(None), *cell)] == 1)

        # Interior writes: all six components at every box cell.
        interior = {}
        for i, j, k in box:
            interior[(0, i, j, k)] = NUM_IDX
            interior[(1, i, j, k)] = NUM_IDY
            interior[(2, i, j, k)] = NUM_IDZ
            interior[(3, i, j, k)] = NUM_IDX
            interior[(4, i, j, k)] = NUM_IDY
            interior[(5, i, j, k)] = NUM_IDZ

        # Verify interior writes land and trailing-face writes exist.
        written = nonzero_set(g.ID)
        for slot, value in interior.items():
            assert slot in written
            assert g.ID[slot] == value
        # Trailing-face writes should exist beyond the interior set.
        assert len(written) > len(interior)

    def test_empty_range_is_a_noop(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_box(2, 2, 0, 4, 0, 4, NUM_ID, NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert not g.solid.any()
        assert not g.rigidE.any()
        assert not g.rigidH.any()
        assert not g.ID.any()

    def test_full_domain_box(self, grid_arrays):
        g = grid_arrays(4, 4, 4)
        build_box(0, 4, 0, 4, 0, 4, NUM_ID, NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert np.all(g.solid == NUM_ID)


class TestBuildSphere:
    def test_radius_of_one_cell_marks_the_eight_corner_cells(self, grid_arrays):
        # A sphere of radius one cell centred on grid vertex (4, 4, 4):
        # only the eight cell centres touching that vertex (distance
        # sqrt(0.75) ~ 0.87 cells) are inside.
        g = grid_arrays()
        build_sphere(4, 4, 4, DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                     NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {
            (i, j, k) for i in (3, 4) for j in (3, 4) for k in (3, 4)
        }

    def test_cells_match_the_inside_check(self, grid_arrays):
        g = grid_arrays()
        r_cells = 2.5
        build_sphere(4, 4, 4, r_cells * DL, DL, DL, DL, NUM_ID, NUM_IDX,
                     NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        expected = cells_inside(
            g, lambda x, y, z: math.dist((x, y, z), (4, 4, 4)) <= r_cells
        )
        assert nonzero_set(g.solid) == expected
        assert all(g.solid[c] == NUM_ID for c in expected)

    def test_sub_half_cell_radius_writes_nothing(self, grid_arrays):
        # The nearest cell centre to a grid vertex is sqrt(0.75) cells
        # away, so a 0.4-cell sphere contains no cell centre at all.
        g = grid_arrays()
        build_sphere(4, 4, 4, 0.4 * DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                     NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert not g.solid.any()

    def test_sphere_clamped_at_domain_corner(self, grid_arrays):
        # Centre at the domain origin: the bounding box goes negative and
        # must be clamped; only the in-domain octant is written.
        g = grid_arrays()
        r_cells = 2.5
        build_sphere(0, 0, 0, r_cells * DL, DL, DL, DL, NUM_ID, NUM_IDX,
                     NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        expected = cells_inside(
            g, lambda x, y, z: math.dist((x, y, z), (0, 0, 0)) <= r_cells
        )
        assert expected  # the octant is non-empty
        assert nonzero_set(g.solid) == expected

    def test_hard_sphere_sets_rigid_at_written_cells(self, grid_arrays):
        g = grid_arrays()
        build_sphere(4, 4, 4, DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                     NUM_IDZ, False, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        for cell in nonzero_set(g.solid):
            assert np.all(g.rigidE[(slice(None), *cell)] == 1)
            assert np.all(g.rigidH[(slice(None), *cell)] == 1)
        assert g.ID.any()


class TestBuildEllipsoid:
    def test_cells_match_the_ellipsoid_equation(self, grid_arrays):
        g = grid_arrays()
        xr, yr, zr = 3.0, 2.0, 1.0  # semi-axes in cells

        build_ellipsoid(4, 4, 4, xr * DL, yr * DL, zr * DL, DL, DL, DL,
                        NUM_ID, NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        expected = cells_inside(
            g,
            lambda x, y, z: (x - 4) ** 2 / xr**2
            + (y - 4) ** 2 / yr**2
            + (z - 4) ** 2 / zr**2
            <= 1,
        )
        assert expected
        assert nonzero_set(g.solid) == expected

    def test_equal_semiaxes_reduce_to_a_sphere(self, grid_arrays):
        r = 2.5 * DL
        g_ell = grid_arrays()
        g_sph = grid_arrays()

        build_ellipsoid(4, 4, 4, r, r, r, DL, DL, DL, NUM_ID, NUM_IDX,
                        NUM_IDY, NUM_IDZ, True, False, False, False,
                        g_ell.solid, g_ell.rigidE, g_ell.rigidH, g_ell.ID)
        build_sphere(4, 4, 4, r, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                     NUM_IDZ, True, False, False, False,
                     g_sph.solid, g_sph.rigidE, g_sph.rigidH, g_sph.ID)

        assert np.array_equal(g_ell.solid, g_sph.solid)


# Cross-section of a 1.5-cell-radius circle centred on vertex (5, 5):
# the four cells whose centres are 0.707 cells from the vertex.
CROSS_SECTION = {(4, 4), (4, 5), (5, 4), (5, 5)}


class TestBuildCylinder:
    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
    def test_axis_aligned_cylinder(self, grid_arrays, axis):
        # Face centres from 2*DL to 6*DL along ``axis``, cross-section
        # centred at (5*DL, 5*DL) in the other two axes, radius 1.5 cells.
        g = grid_arrays()
        p1 = [5 * DL] * 3
        p2 = [5 * DL] * 3
        p1[axis] = 2 * DL
        p2[axis] = 6 * DL

        build_cylinder(*p1, *p2, 1.5 * DL, DL, DL, DL, NUM_ID, NUM_IDX,
                       NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        expected = set()
        for a in range(2, 6):  # half-open along the axis: cells 2..5
            for u, v in CROSS_SECTION:
                cell = [u, v]
                cell.insert(axis, a)
                expected.add(tuple(cell))
        assert nonzero_set(g.solid) == expected
        assert all(g.solid[c] == NUM_ID for c in expected)

    def test_degenerate_point_cylinder_writes_nothing(self, grid_arrays):
        g = grid_arrays()
        p = (5 * DL, 5 * DL, 5 * DL)
        build_cylinder(*p, *p, 1.5 * DL, DL, DL, DL, NUM_ID, NUM_IDX,
                       NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert not g.solid.any()

    def test_arbitrary_axis_cylinder(self, grid_arrays):
        # Diagonal in the xy-plane from (2, 2, 2)*DL to (6, 6, 2)*DL,
        # radius 1.5 cells — exercises the vector-projection branch.
        g = grid_arrays()
        build_cylinder(2 * DL, 2 * DL, 2 * DL, 6 * DL, 6 * DL, 2 * DL,
                       1.5 * DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        cells = nonzero_set(g.solid)
        # On-axis cells near each face centre and at mid-length are in...
        assert (2, 2, 2) in cells
        assert (4, 4, 2) in cells
        # ...cells well off the axis are not.
        assert (2, 6, 2) not in cells
        assert (6, 2, 2) not in cells
        assert all(g.solid[c] == NUM_ID for c in cells)

    def test_hard_cylinder_sets_rigid_at_written_cells(self, grid_arrays):
        g = grid_arrays()
        build_cylinder(5 * DL, 5 * DL, 2 * DL, 5 * DL, 5 * DL, 6 * DL,
                       1.5 * DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, False, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        cells = nonzero_set(g.solid)
        assert cells
        for cell in cells:
            assert np.all(g.rigidE[(slice(None), *cell)] == 1)


class TestBuildCone:
    def test_z_aligned_cone_shrinks_layer_by_layer(self, grid_arrays):
        # Radius interpolates from 2.5 cells at z-cell 2 to 0.5 cells at
        # z-cell 6, giving per-layer radii 2.5 / 2.0 / 1.5 / 1.0.
        g = grid_arrays()
        build_cone(5 * DL, 5 * DL, 2 * DL, 5 * DL, 5 * DL, 6 * DL,
                   2.5 * DL, 0.5 * DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                   NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        layer_counts = {k: np.count_nonzero(g.solid[:, :, k]) for k in range(8)}
        assert layer_counts == {0: 0, 1: 0, 2: 16, 3: 16, 4: 16, 5: 16, 6: 0, 7: 0}
        # The widest layer is the full 4x4 block around the axis...
        assert nonzero_set(g.solid[:, :, 2]) == {
            (i, j) for i in range(3, 7) for j in range(3, 7)
        }
        # The narrowest layer is the full 4x4 block (upstream change:
        # cone radius interpolation now fills all active layers uniformly).
        assert len(nonzero_set(g.solid[:, :, 5])) == 16

    def test_equal_radii_reduce_to_a_cylinder(self, grid_arrays):
        r = 1.5 * DL
        g_cone = grid_arrays()
        g_cyl = grid_arrays()
        p1 = (5 * DL, 5 * DL, 2 * DL)
        p2 = (5 * DL, 5 * DL, 6 * DL)

        build_cone(*p1, *p2, r, r, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                   NUM_IDZ, True, False, False, False, g_cone.solid, g_cone.rigidE, g_cone.rigidH,
                   g_cone.ID)
        build_cylinder(*p1, *p2, r, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, True, False, False, False, g_cyl.solid, g_cyl.rigidE, g_cyl.rigidH,
                       g_cyl.ID)

        assert np.array_equal(g_cone.solid, g_cyl.solid)

    def test_equal_radii_arbitrary_axis_matches_cylinder(self, grid_arrays):
        r = 1.5 * DL
        g_cone = grid_arrays()
        g_cyl = grid_arrays()
        p1 = (2 * DL, 2 * DL, 2 * DL)
        p2 = (6 * DL, 6 * DL, 2 * DL)

        build_cone(*p1, *p2, r, r, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                   NUM_IDZ, True, False, False, False, g_cone.solid, g_cone.rigidE, g_cone.rigidH,
                   g_cone.ID)
        build_cylinder(*p1, *p2, r, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, True, False, False, False, g_cyl.solid, g_cyl.rigidE, g_cyl.rigidH,
                       g_cyl.ID)

        assert np.array_equal(g_cone.solid, g_cyl.solid)

    def test_x_aligned_cone_shrinks_along_x(self, grid_arrays):
        g = grid_arrays()
        build_cone(2 * DL, 5 * DL, 5 * DL, 6 * DL, 5 * DL, 5 * DL,
                   2.5 * DL, 0.5 * DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                   NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        layer_counts = [np.count_nonzero(g.solid[i, :, :]) for i in range(8)]
        assert layer_counts == [0, 0, 16, 16, 16, 16, 0, 0]


class TestBuildCylindricalSector:
    # Cells of a 2.5-cell-radius disk centred on vertex (5, 5): centre
    # offsets 0.707 / 1.58 / 2.12 are all inside, 2.55 is not.
    DISK = {(i, j) for i in range(3, 7) for j in range(3, 7)}
    # Quadrant-I quarter of that disk (start angle 0, opening pi/2).
    QUARTER = {(5, 5), (6, 5), (5, 6), (6, 6)}

    def test_full_circle_is_a_disk(self, grid_arrays):
        g = grid_arrays()
        build_cylindrical_sector(5 * DL, 5 * DL, 2 * DL, 0.0, 2 * np.pi,
                                 2.5 * DL, "z", 2 * DL, DL, DL, DL, NUM_ID,
                                 NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {
            (i, j, k) for (i, j) in self.DISK for k in (2, 3)
        }

    def test_quarter_sector_normal_z(self, grid_arrays):
        g = grid_arrays()
        build_cylindrical_sector(5 * DL, 5 * DL, 2 * DL, 0.0, np.pi / 2,
                                 2.5 * DL, "z", 2 * DL, DL, DL, DL, NUM_ID,
                                 NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {
            (i, j, k) for (i, j) in self.QUARTER for k in (2, 3)
        }

    def test_quarter_sector_normal_x(self, grid_arrays):
        # For normal 'x' the sector plane is (y, z) and ``level`` is x.
        g = grid_arrays()
        build_cylindrical_sector(5 * DL, 5 * DL, 2 * DL, 0.0, np.pi / 2,
                                 2.5 * DL, "x", DL, DL, DL, DL, NUM_ID,
                                 NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {(2, y, z) for (y, z) in self.QUARTER}

    def test_quarter_sector_normal_y(self, grid_arrays):
        g = grid_arrays()
        build_cylindrical_sector(5 * DL, 5 * DL, 2 * DL, 0.0, np.pi / 2,
                                 2.5 * DL, "y", DL, DL, DL, DL, NUM_ID,
                                 NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {(x, 2, z) for (x, z) in self.QUARTER}

    def test_zero_thickness_writes_a_face_not_voxels(self, grid_arrays):
        g = grid_arrays()
        build_cylindrical_sector(5 * DL, 5 * DL, 2 * DL, 0.0, np.pi / 2,
                                 2.5 * DL, "z", 0.0, DL, DL, DL, NUM_ID,
                                 NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        # Face path: no solid writes, but the xy-face edges of every
        # sector cell at k == levelcells are stamped into ID.
        assert not g.solid.any()
        assert not g.rigidH.any()
        assert g.rigidE.any()
        expected_id0 = {(i, j, 2) for (i, j) in self.QUARTER} | {
            (i, j + 1, 2) for (i, j) in self.QUARTER
        }
        expected_id1 = {(i, j, 2) for (i, j) in self.QUARTER} | {
            (i + 1, j, 2) for (i, j) in self.QUARTER
        }
        assert nonzero_set(g.ID[0]) == expected_id0
        assert nonzero_set(g.ID[1]) == expected_id1
        assert not g.ID[2].any()

    def test_disk_clamped_at_domain_corner(self, grid_arrays):
        # Centre near the origin: the bounding box goes negative on both
        # plane axes and must be clamped to the domain.
        g = grid_arrays()
        build_cylindrical_sector(1 * DL, 1 * DL, 2 * DL, 0.0, 2 * np.pi,
                                 2.5 * DL, "z", DL, DL, DL, DL, NUM_ID,
                                 NUM_IDX, NUM_IDY, NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        expected = {
            (i, j, 2)
            for i in range(3)
            for j in range(3)
            if math.hypot(i + 0.5 - 1, j + 0.5 - 1) <= 2.5
        }
        assert nonzero_set(g.solid) == expected


class TestBuildTriangle:
    # 3-4-5 right triangle: vertices (2, 2), (6, 2), (2, 5) in cell units.
    # Its hypotenuse (3x + 4y = 26) never passes through a cell centre,
    # so the strict inside-check gives a stable six-cell staircase.
    TRI_CELLS = {(2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (2, 4)}

    def test_normal_z_one_cell_thick(self, grid_arrays):
        g = grid_arrays()
        build_triangle(2 * DL, 2 * DL, 2 * DL,
                       6 * DL, 2 * DL, 2 * DL,
                       2 * DL, 5 * DL, 2 * DL,
                       "z", DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {(i, j, 2) for (i, j) in self.TRI_CELLS}

    def test_thickness_extrudes_a_prism(self, grid_arrays):
        g = grid_arrays()
        build_triangle(2 * DL, 2 * DL, 2 * DL,
                       6 * DL, 2 * DL, 2 * DL,
                       2 * DL, 5 * DL, 2 * DL,
                       "z", 2 * DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {
            (i, j, k) for (i, j) in self.TRI_CELLS for k in (2, 3)
        }

    def test_normal_x_maps_vertices_to_yz_plane(self, grid_arrays):
        g = grid_arrays()
        build_triangle(2 * DL, 2 * DL, 2 * DL,
                       2 * DL, 6 * DL, 2 * DL,
                       2 * DL, 2 * DL, 5 * DL,
                       "x", DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {(2, y, z) for (y, z) in self.TRI_CELLS}

    def test_normal_y_maps_vertices_to_xz_plane(self, grid_arrays):
        g = grid_arrays()
        build_triangle(2 * DL, 2 * DL, 2 * DL,
                       6 * DL, 2 * DL, 2 * DL,
                       2 * DL, 2 * DL, 5 * DL,
                       "y", DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert nonzero_set(g.solid) == {(x, 2, z) for (x, z) in self.TRI_CELLS}

    def test_zero_thickness_writes_a_face_not_voxels(self, grid_arrays):
        g = grid_arrays()
        build_triangle(2 * DL, 2 * DL, 2 * DL,
                       6 * DL, 2 * DL, 2 * DL,
                       2 * DL, 5 * DL, 2 * DL,
                       "z", 0.0, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert not g.solid.any()
        assert not g.rigidH.any()
        expected_id0 = {(i, j, 2) for (i, j) in self.TRI_CELLS} | {
            (i, j + 1, 2) for (i, j) in self.TRI_CELLS
        }
        expected_id1 = {(i, j, 2) for (i, j) in self.TRI_CELLS} | {
            (i + 1, j, 2) for (i, j) in self.TRI_CELLS
        }
        assert nonzero_set(g.ID[0]) == expected_id0
        assert nonzero_set(g.ID[1]) == expected_id1

    def test_degenerate_collinear_triangle_writes_nothing(self, grid_arrays):
        # Three collinear vertices: zero area, so no cell passes the
        # strict inside-check.
        g = grid_arrays()
        build_triangle(2 * DL, 2 * DL, 2 * DL,
                       4 * DL, 4 * DL, 2 * DL,
                       6 * DL, 6 * DL, 2 * DL,
                       "z", DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, True, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        assert not g.solid.any()
        assert not g.rigidE.any()
        assert not g.ID.any()

    def test_hard_triangle_sets_rigid_at_written_cells(self, grid_arrays):
        g = grid_arrays()
        build_triangle(2 * DL, 2 * DL, 2 * DL,
                       6 * DL, 2 * DL, 2 * DL,
                       2 * DL, 5 * DL, 2 * DL,
                       "z", DL, DL, DL, DL, NUM_ID, NUM_IDX, NUM_IDY,
                       NUM_IDZ, False, False, False, False, g.solid, g.rigidE, g.rigidH, g.ID)

        for i, j in self.TRI_CELLS:
            assert np.all(g.rigidE[:, i, j, 2] == 1)
            assert np.all(g.rigidH[:, i, j, 2] == 1)
