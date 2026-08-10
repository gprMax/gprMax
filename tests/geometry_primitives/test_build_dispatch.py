"""Unit tests for the ``build()`` dispatch layer of the geometry
user objects in ``gprMax/user_objects/cmds_geometry/``.

These are the seven wrappers whose ``build()`` methods were not
exercised by the user-objects suite: ``Edge``, ``Plate``, ``Triangle``,
``Cylinder``, ``Cone``, ``CylindricalSector`` and
``GeometryObjectsRead``. Each ``build()`` converts continuous
coordinates to cell indices (through a real ``MainGridUserInput`` — no
mocking of the discretisation), resolves the material and averaging
flag, validates the orientation, and hands off to the Cython
rasteriser. The tests drive that whole chain against the
``dispatch_grid`` stub and assert the final state of the grid arrays.

``GeometryObjectsRead.build()`` beyond its parameter guard is
integration territory (HDF5 + materials files and the scene builder);
its array stamping is covered directly in ``test_array_builders.py``.
"""

import numpy as np
import pytest
from unittest import mock

from gprMax.user_objects.cmds_geometry.cone import Cone
from gprMax.user_objects.cmds_geometry.cylinder import Cylinder
from gprMax.user_objects.cmds_geometry.cylindrical_sector import CylindricalSector
from gprMax.user_objects.cmds_geometry.edge import Edge
from gprMax.user_objects.cmds_geometry.geometry_objects_read import GeometryObjectsRead
from gprMax.user_objects.cmds_geometry.plate import Plate
from gprMax.user_objects.cmds_geometry.triangle import Triangle

from .conftest import DL, nonzero_set

METAL = 2  # numID of the averagable "metal" material in dispatch_grid


@pytest.fixture(autouse=True)
def _mock_model_config():
    """Mock ``get_model_config`` so dispatch tests don't crash on the
    uninitialised ``sim_config`` global."""
    fake = mock.MagicMock()
    fake.mode = "3D"
    fake.requested_2d_mode = None
    fake.debye_averaging = True
    fake.geometry_fixed = False
    with mock.patch("gprMax.config.get_model_config", return_value=fake):
        yield


class TestEdgeBuild:
    @pytest.mark.parametrize(
        "p1, p2, component, expected",
        [
            # x-oriented: cells 2..4 (half-open at 5)
            ((2, 3, 4), (5, 3, 4), 0, {(0, 2, 3, 4), (0, 3, 3, 4), (0, 4, 3, 4)}),
            # y-oriented
            ((3, 2, 4), (3, 5, 4), 1, {(1, 3, 2, 4), (1, 3, 3, 4), (1, 3, 4, 4)}),
            # z-oriented
            ((3, 4, 2), (3, 4, 5), 2, {(2, 3, 4, 2), (2, 3, 4, 3), (2, 3, 4, 4)}),
        ],
        ids=["x", "y", "z"],
    )
    def test_axis_oriented_edge_stamps_id_cells(self, dispatch_grid, p1, p2, component, expected):
        g = dispatch_grid()
        edge = Edge(
            p1=tuple(c * DL for c in p1),
            p2=tuple(c * DL for c in p2),
            material_id="metal",
        )
        edge.build(g)

        assert nonzero_set(g.ID) == expected
        assert all(g.ID[slot] == METAL for slot in expected)
        assert g.rigidE.any()
        assert not g.solid.any()

    def test_diagonal_edge_raises(self, dispatch_grid):
        g = dispatch_grid()
        edge = Edge(p1=(2 * DL, 2 * DL, 2 * DL), p2=(3 * DL, 3 * DL, 2 * DL),
                    material_id="metal")
        with pytest.raises(ValueError):
            edge.build(g)

    def test_unknown_material_raises(self, dispatch_grid):
        g = dispatch_grid()
        edge = Edge(p1=(2 * DL, 3 * DL, 4 * DL), p2=(5 * DL, 3 * DL, 4 * DL),
                    material_id="ghost")
        with pytest.raises(ValueError):
            edge.build(g)

    def test_missing_kwargs_raises(self, dispatch_grid):
        g = dispatch_grid()
        with pytest.raises(KeyError):
            Edge(p1=(2 * DL, 3 * DL, 4 * DL)).build(g)

    def test_do_rotate_turns_an_x_edge_into_a_y_edge(self, dispatch_grid):
        # An x-oriented edge rotated 90 degrees about z (around its own
        # first point) becomes y-oriented — _do_rotate rewrites the p1/p2
        # kwargs through rotate_2point_object.
        g = dispatch_grid()
        edge = Edge(p1=(2 * DL, 2 * DL, 2 * DL), p2=(5 * DL, 2 * DL, 2 * DL),
                    material_id="metal")
        edge.rotate("z", 90, origin=(2 * DL, 2 * DL, 0.0))
        assert edge.do_rotate
        edge._do_rotate(g)

        assert np.allclose(edge.kwargs["p1"], (2 * DL, 2 * DL, 2 * DL))
        assert np.allclose(edge.kwargs["p2"], (2 * DL, 5 * DL, 2 * DL))


class TestPlateBuild:
    # xy-plane plate from cell (1, 1) to (4, 3) at z-level 2: the face
    # builder runs over cells i in 1..3, j in 1..2.
    CELLS = {(i, j) for i in range(1, 4) for j in range(1, 3)}

    def test_xy_plate_stamps_face_edges(self, dispatch_grid):
        g = dispatch_grid()
        plate = Plate(p1=(1 * DL, 1 * DL, 2 * DL), p2=(4 * DL, 3 * DL, 2 * DL),
                      material_id="metal")
        plate.build(g)

        expected_id0 = {(i, j, 2) for (i, j) in self.CELLS} | {
            (i, j + 1, 2) for (i, j) in self.CELLS
        }
        expected_id1 = {(i, j, 2) for (i, j) in self.CELLS} | {
            (i + 1, j, 2) for (i, j) in self.CELLS
        }
        assert nonzero_set(g.ID[0]) == expected_id0
        assert nonzero_set(g.ID[1]) == expected_id1
        assert not g.ID[2].any()
        assert not g.solid.any()
        assert g.rigidE.any()

    def test_anisotropic_plate_uses_per_direction_materials(self, dispatch_grid):
        g = dispatch_grid()
        plate = Plate(p1=(1 * DL, 1 * DL, 2 * DL), p2=(4 * DL, 3 * DL, 2 * DL),
                      material_ids=["mat_a", "mat_b"])
        plate.build(g)

        # xy-plate: first material feeds the x-edges, second the y-edges.
        assert all(g.ID[0][slot] == 3 for slot in nonzero_set(g.ID[0]))
        assert all(g.ID[1][slot] == 4 for slot in nonzero_set(g.ID[1]))

    def test_volume_raises(self, dispatch_grid):
        g = dispatch_grid()
        plate = Plate(p1=(1 * DL, 1 * DL, 1 * DL), p2=(3 * DL, 3 * DL, 3 * DL),
                      material_id="metal")
        with pytest.raises(ValueError):
            plate.build(g)

    def test_line_raises(self, dispatch_grid):
        g = dispatch_grid()
        plate = Plate(p1=(1 * DL, 2 * DL, 2 * DL), p2=(4 * DL, 2 * DL, 2 * DL),
                      material_id="metal")
        with pytest.raises(ValueError):
            plate.build(g)

    def test_unknown_material_raises(self, dispatch_grid):
        g = dispatch_grid()
        plate = Plate(p1=(1 * DL, 1 * DL, 2 * DL), p2=(4 * DL, 3 * DL, 2 * DL),
                      material_id="ghost")
        with pytest.raises(ValueError):
            plate.build(g)


class TestTriangleBuild:
    # Same 3-4-5 staircase as the direct build_triangle tests.
    CELLS = {(2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (2, 4)}
    KWARGS = dict(
        p1=(2 * DL, 2 * DL, 2 * DL),
        p2=(6 * DL, 2 * DL, 2 * DL),
        p3=(2 * DL, 5 * DL, 2 * DL),
        material_id="metal",
    )

    def test_prism_writes_solid_with_grid_default_averaging(self, dispatch_grid):
        g = dispatch_grid()
        Triangle(thickness=DL, **self.KWARGS).build(g)

        assert nonzero_set(g.solid) == {(i, j, 2) for (i, j) in self.CELLS}
        assert all(g.solid[i, j, 2] == METAL for (i, j) in self.CELLS)
        # grid default averaging + averagable material -> smoothed path
        assert not g.rigidE.any()
        assert not g.ID.any()

    def test_averaging_off_kwarg_sets_rigid(self, dispatch_grid):
        g = dispatch_grid()
        Triangle(thickness=DL, averaging="n", **self.KWARGS).build(g)

        for i, j in self.CELLS:
            assert np.all(g.rigidE[:, i, j, 2] == 1)
        assert g.ID.any()

    def test_zero_thickness_builds_a_patch(self, dispatch_grid):
        g = dispatch_grid()
        Triangle(thickness=0.0, **self.KWARGS).build(g)

        assert not g.solid.any()
        assert g.ID[0].any()
        assert g.ID[1].any()

    def test_non_coplanar_vertices_raise(self, dispatch_grid):
        g = dispatch_grid()
        triangle = Triangle(
            p1=(2 * DL, 2 * DL, 2 * DL),
            p2=(6 * DL, 2 * DL, 3 * DL),
            p3=(2 * DL, 5 * DL, 4 * DL),
            thickness=DL,
            material_id="metal",
        )
        with pytest.raises(ValueError):
            triangle.build(g)

    def test_unknown_material_raises(self, dispatch_grid):
        g = dispatch_grid()
        kwargs = dict(self.KWARGS, material_id="ghost")
        with pytest.raises(ValueError):
            Triangle(thickness=DL, **kwargs).build(g)

    def test_missing_thickness_raises(self, dispatch_grid):
        g = dispatch_grid()
        with pytest.raises(KeyError):
            Triangle(**self.KWARGS).build(g)


class TestCylinderBuild:
    CROSS_SECTION = {(4, 4), (4, 5), (5, 4), (5, 5)}

    def test_z_aligned_cylinder_matches_direct_rasterisation(self, dispatch_grid):
        g = dispatch_grid()
        cylinder = Cylinder(p1=(5 * DL, 5 * DL, 2 * DL), p2=(5 * DL, 5 * DL, 6 * DL),
                            r=1.5 * DL, material_id="metal")
        cylinder.build(g)

        expected = {(i, j, k) for (i, j) in self.CROSS_SECTION for k in range(2, 6)}
        assert nonzero_set(g.solid) == expected
        assert all(g.solid[c] == METAL for c in expected)
        # averagable material + grid default -> smoothed path
        assert not g.rigidE.any()

    def test_averaging_off_kwarg_sets_rigid(self, dispatch_grid):
        g = dispatch_grid()
        cylinder = Cylinder(p1=(5 * DL, 5 * DL, 2 * DL), p2=(5 * DL, 5 * DL, 6 * DL),
                            r=1.5 * DL, material_id="metal", averaging="n")
        cylinder.build(g)

        for cell in nonzero_set(g.solid):
            assert np.all(g.rigidE[(slice(None), *cell)] == 1)

    def test_non_positive_radius_raises(self, dispatch_grid):
        g = dispatch_grid()
        cylinder = Cylinder(p1=(5 * DL, 5 * DL, 2 * DL), p2=(5 * DL, 5 * DL, 6 * DL),
                            r=0.0, material_id="metal")
        with pytest.raises(ValueError):
            cylinder.build(g)

    def test_unknown_material_raises(self, dispatch_grid):
        g = dispatch_grid()
        cylinder = Cylinder(p1=(5 * DL, 5 * DL, 2 * DL), p2=(5 * DL, 5 * DL, 6 * DL),
                            r=1.5 * DL, material_id="ghost")
        with pytest.raises(ValueError):
            cylinder.build(g)

    def test_missing_radius_raises(self, dispatch_grid):
        g = dispatch_grid()
        cylinder = Cylinder(p1=(5 * DL, 5 * DL, 2 * DL), p2=(5 * DL, 5 * DL, 6 * DL),
                            material_id="metal")
        with pytest.raises(KeyError):
            cylinder.build(g)


class TestConeBuild:
    def test_tapering_cone_matches_direct_rasterisation(self, dispatch_grid):
        g = dispatch_grid()
        cone = Cone(p1=(5 * DL, 5 * DL, 2 * DL), p2=(5 * DL, 5 * DL, 6 * DL),
                    r1=2.5 * DL, r2=0.5 * DL, material_id="metal")
        cone.build(g)

        layer_counts = [np.count_nonzero(g.solid[:, :, k]) for k in range(8)]
        assert layer_counts == [0, 0, 16, 16, 16, 16, 0, 0]

    def test_both_radii_zero_raises(self, dispatch_grid):
        g = dispatch_grid()
        cone = Cone(p1=(5 * DL, 5 * DL, 2 * DL), p2=(5 * DL, 5 * DL, 6 * DL),
                    r1=0.0, r2=0.0, material_id="metal")
        with pytest.raises(ValueError):
            cone.build(g)

    def test_negative_radius_raises(self, dispatch_grid):
        g = dispatch_grid()
        cone = Cone(p1=(5 * DL, 5 * DL, 2 * DL), p2=(5 * DL, 5 * DL, 6 * DL),
                    r1=-1.0 * DL, r2=1.0 * DL, material_id="metal")
        with pytest.raises(ValueError):
            cone.build(g)

    def test_missing_radius_raises(self, dispatch_grid):
        g = dispatch_grid()
        cone = Cone(p1=(5 * DL, 5 * DL, 2 * DL), p2=(5 * DL, 5 * DL, 6 * DL),
                    r1=1.0 * DL, material_id="metal")
        with pytest.raises(KeyError):
            cone.build(g)


class TestCylindricalSectorBuild:
    QUARTER = {(5, 5), (6, 5), (5, 6), (6, 6)}
    KWARGS = dict(
        normal="z",
        ctr1=5 * DL,
        ctr2=5 * DL,
        extent1=2 * DL,
        extent2=4 * DL,
        r=2.5 * DL,
        start=0.0,
        material_id="metal",
    )

    def test_quarter_sector_matches_direct_rasterisation(self, dispatch_grid):
        # start/end arrive in degrees from the user object and are
        # converted to radians before the Cython call.
        g = dispatch_grid()
        CylindricalSector(end=90.0, **self.KWARGS).build(g)

        expected = {(i, j, k) for (i, j) in self.QUARTER for k in (2, 3)}
        assert nonzero_set(g.solid) == expected
        assert all(g.solid[c] == METAL for c in expected)

    def test_full_circle_end_angle_raises(self, dispatch_grid):
        # The wrapper caps sector angles strictly below 360 degrees.
        g = dispatch_grid()
        with pytest.raises(ValueError):
            CylindricalSector(end=360.0, **self.KWARGS).build(g)

    def test_zero_end_angle_raises(self, dispatch_grid):
        g = dispatch_grid()
        with pytest.raises(ValueError):
            CylindricalSector(end=0.0, **self.KWARGS).build(g)

    def test_invalid_normal_raises(self, dispatch_grid):
        g = dispatch_grid()
        kwargs = dict(self.KWARGS, normal="q")
        with pytest.raises(ValueError):
            CylindricalSector(end=90.0, **kwargs).build(g)

    def test_missing_kwargs_raises(self, dispatch_grid):
        g = dispatch_grid()
        with pytest.raises(KeyError):
            CylindricalSector(normal="z", ctr1=5 * DL, ctr2=5 * DL).build(g)


class TestGeometryObjectsReadBuild:
    def test_missing_kwargs_raises(self, dispatch_grid):
        g = dispatch_grid()
        with pytest.raises(KeyError):
            GeometryObjectsRead(p1=(0.0, 0.0, 0.0)).build(g)
