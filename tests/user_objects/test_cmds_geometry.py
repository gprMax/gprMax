"""Tests for the geometry user-object classes under
``gprMax.user_objects.cmds_geometry``.

14 classes in total: 11 primitive shapes (Edge, Plate, Triangle, Box,
Cylinder, Cone, CylindricalSector, Sphere, Ellipsoid, FractalBox,
GeometryObjectsRead) plus 3 modifiers that decorate a fractal box
(AddSurfaceRoughness, AddSurfaceWater, AddGrass).

All extend ``GeometryUserObject``, so ``order == 1`` is inherited from
the base class (geometry primitives build in arrival order, not by
``order``). Most also mix in ``RotatableMixin``.

We test:

* constructor stores ``self.kwargs`` verbatim;
* ``order == 1`` is inherited;
* ``hash`` matches the documented value;
* ``build()`` raises on missing required kwargs or unknown materials —
  but we do **not** drive the full ``build()`` chain (the underlying
  Cython primitives like ``build_box``/``build_sphere`` are out of scope
  for unit tests).
"""

from unittest.mock import MagicMock, patch

import pytest

from gprMax.user_objects.cmds_geometry.add_grass import AddGrass
from gprMax.user_objects.cmds_geometry.add_surface_roughness import AddSurfaceRoughness
from gprMax.user_objects.cmds_geometry.add_surface_water import AddSurfaceWater
from gprMax.user_objects.cmds_geometry.box import Box
from gprMax.user_objects.cmds_geometry.cone import Cone
from gprMax.user_objects.cmds_geometry.cylinder import Cylinder
from gprMax.user_objects.cmds_geometry.cylindrical_sector import CylindricalSector
from gprMax.user_objects.cmds_geometry.edge import Edge
from gprMax.user_objects.cmds_geometry.ellipsoid import Ellipsoid
from gprMax.user_objects.cmds_geometry.fractal_box import FractalBox
from gprMax.user_objects.cmds_geometry.geometry_objects_read import GeometryObjectsRead
from gprMax.user_objects.cmds_geometry.plate import Plate
from gprMax.user_objects.cmds_geometry.sphere import Sphere
from gprMax.user_objects.cmds_geometry.triangle import Triangle

# ---------------------------------------------------------------------------
# Common contract: every geometry class has order == 1 and the right hash
# ---------------------------------------------------------------------------


PRIMITIVES = [
    (Edge, "#edge"),
    (Plate, "#plate"),
    (Triangle, "#triangle"),
    (Box, "#box"),
    (Cylinder, "#cylinder"),
    (Cone, "#cone"),
    (CylindricalSector, "#cylindrical_sector"),
    (Sphere, "#sphere"),
    (Ellipsoid, "#ellipsoid"),
    (FractalBox, "#fractal_box"),
    (GeometryObjectsRead, "#geometry_objects_read"),
]

MODIFIERS = [
    (AddSurfaceRoughness, "#add_surface_roughness"),
    (AddSurfaceWater, "#add_surface_water"),
    (AddGrass, "#add_grass"),
]


class TestCommonContract:
    @pytest.mark.parametrize("cls,hash_name", PRIMITIVES + MODIFIERS)
    def test_order_is_one(self, cls, hash_name):
        # ``GeometryUserObject.order == 1`` is the contract for geometry
        # objects (they build in arrival order, not by ``order``).
        assert cls().order == 1

    @pytest.mark.parametrize("cls,hash_name", PRIMITIVES + MODIFIERS)
    def test_hash_matches(self, cls, hash_name):
        assert cls().hash == hash_name

    @pytest.mark.parametrize("cls,_", PRIMITIVES + MODIFIERS)
    def test_constructor_stores_kwargs_verbatim(self, cls, _):
        # Every geometry class uses ``__init__(**kwargs)`` → ``self.kwargs``
        obj = cls(arbitrary=1, payload="x", values=(0.1, 0.2))
        assert obj.kwargs == {
            "arbitrary": 1,
            "payload": "x",
            "values": (0.1, 0.2),
        }


# ---------------------------------------------------------------------------
# Box — the canonical isotropic/anisotropic material branch test target
# ---------------------------------------------------------------------------


class TestBox:
    def test_constructor_kwargs(self):
        b = Box(p1=(0, 0, 0), p2=(0.1, 0.1, 0.1), material_id="free_space")
        assert b.kwargs["p1"] == (0, 0, 0)
        assert b.kwargs["material_id"] == "free_space"

    def test_rotatable_defaults(self):
        b = Box(p1=(0, 0, 0), p2=(0.1, 0.1, 0.1), material_id="x")
        assert b.do_rotate is False

    def test_build_missing_p1_raises(self, stub_grid):
        b = Box(material_id="free_space")
        with pytest.raises(KeyError):
            b.build(stub_grid)

    def test_build_missing_material_raises(self, stub_grid):
        # Neither material_id nor material_ids set
        b = Box(p1=(0, 0, 0), p2=(0.1, 0.1, 0.1))
        uip = MagicMock()
        uip.check_box_points.return_value = (True, (0, 0, 0), (10, 10, 10))
        with patch.object(Box, "_create_uip", return_value=uip):
            with pytest.raises(KeyError):
                b.build(stub_grid)

    def test_build_unknown_material_raises(self, stub_grid):
        b = Box(p1=(0, 0, 0), p2=(0.1, 0.1, 0.1), material_id="ghost")
        uip = MagicMock()
        uip.check_box_points.return_value = (True, (0, 0, 0), (10, 10, 10))
        uip.round_to_grid_static_point.return_value = (0.0, 0.0, 0.0)
        with patch.object(Box, "_create_uip", return_value=uip):
            with pytest.raises(ValueError):
                b.build(stub_grid)

    def test_build_returns_early_when_box_not_in_grid(self, stub_grid):
        # When uip says ``grid_contains_box=False``, ``build()`` returns
        # without touching materials.
        b = Box(p1=(0, 0, 0), p2=(0.1, 0.1, 0.1), material_id="free_space")
        uip = MagicMock()
        uip.check_box_points.return_value = (False, None, None)
        with patch.object(Box, "_create_uip", return_value=uip):
            # Should return None without raising
            assert b.build(stub_grid) is None


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------


class TestSphere:
    def test_constructor_kwargs(self):
        s = Sphere(p1=(0.05, 0.05, 0.05), r=0.02, material_id="free_space")
        assert s.kwargs["p1"] == (0.05, 0.05, 0.05)
        assert s.kwargs["r"] == 0.02

    def test_build_missing_p1_raises(self, stub_grid):
        s = Sphere(r=0.02, material_id="x")
        with pytest.raises(KeyError):
            s.build(stub_grid)

    def test_build_missing_radius_raises(self, stub_grid):
        s = Sphere(p1=(0.05, 0.05, 0.05), material_id="x")
        with pytest.raises(KeyError):
            s.build(stub_grid)

    def test_build_missing_material_raises(self, stub_grid):
        s = Sphere(p1=(0.05, 0.05, 0.05), r=0.02)
        uip = MagicMock()
        uip.round_to_grid_static_point.return_value = (0.0, 0.0, 0.0)
        uip.discretise_point.return_value = (25, 25, 25)
        with patch.object(Sphere, "_create_uip", return_value=uip):
            with pytest.raises(KeyError):
                s.build(stub_grid)

    def test_build_unknown_material_raises(self, stub_grid):
        s = Sphere(p1=(0.05, 0.05, 0.05), r=0.02, material_id="ghost")
        uip = MagicMock()
        uip.round_to_grid_static_point.return_value = (0.0, 0.0, 0.0)
        uip.discretise_point.return_value = (25, 25, 25)
        with patch.object(Sphere, "_create_uip", return_value=uip):
            with pytest.raises(ValueError):
                s.build(stub_grid)


# ---------------------------------------------------------------------------
# Ellipsoid
# ---------------------------------------------------------------------------


class TestEllipsoid:
    def test_constructor_kwargs(self):
        e = Ellipsoid(
            p1=(0.05, 0.05, 0.05),
            xr=0.02,
            yr=0.03,
            zr=0.04,
            material_id="free_space",
        )
        assert e.kwargs["xr"] == 0.02
        assert e.kwargs["yr"] == 0.03

    def test_build_missing_semiaxis_raises(self, stub_grid):
        # zr is missing
        e = Ellipsoid(p1=(0.05, 0.05, 0.05), xr=0.02, yr=0.03)
        with pytest.raises(KeyError):
            e.build(stub_grid)


# ---------------------------------------------------------------------------
# Cylinder / Cone / Cylindrical sector
# ---------------------------------------------------------------------------


class TestCylinder:
    def test_constructor_kwargs(self):
        c = Cylinder(p1=(0, 0, 0), p2=(0.1, 0, 0), r=0.02, material_id="free_space")
        assert c.kwargs["r"] == 0.02


class TestCone:
    def test_constructor_kwargs(self):
        c = Cone(
            p1=(0, 0, 0),
            p2=(0.1, 0, 0),
            r1=0.02,
            r2=0.01,
            material_id="free_space",
        )
        assert c.kwargs["r1"] == 0.02
        assert c.kwargs["r2"] == 0.01


class TestCylindricalSector:
    def test_constructor_kwargs(self):
        c = CylindricalSector(
            axis="z",
            ctr1=0.05,
            ctr2=0.05,
            t1=0.0,
            t2=0.05,
            r=0.02,
            sectorstartangle=0.0,
            sectorangle=90.0,
            material_id="free_space",
        )
        assert c.kwargs["axis"] == "z"
        assert c.kwargs["sectorangle"] == 90.0


# ---------------------------------------------------------------------------
# Edge / Plate / Triangle
# ---------------------------------------------------------------------------


class TestEdge:
    def test_constructor_kwargs(self):
        e = Edge(p1=(0, 0, 0), p2=(0.1, 0, 0), material_id="free_space")
        assert e.kwargs["p1"] == (0, 0, 0)
        assert e.kwargs["material_id"] == "free_space"

    def test_rotatable_defaults(self):
        e = Edge(p1=(0, 0, 0), p2=(0.1, 0, 0), material_id="x")
        assert e.do_rotate is False


class TestPlate:
    def test_constructor_kwargs(self):
        p = Plate(p1=(0, 0, 0), p2=(0.1, 0.1, 0), material_id="free_space")
        assert p.kwargs["material_id"] == "free_space"


class TestTriangle:
    def test_constructor_kwargs(self):
        t = Triangle(
            p1=(0, 0, 0),
            p2=(0.1, 0, 0),
            p3=(0, 0.1, 0),
            thickness=0.01,
            material_id="free_space",
        )
        assert t.kwargs["thickness"] == 0.01
        assert t.kwargs["p3"] == (0, 0.1, 0)


# ---------------------------------------------------------------------------
# FractalBox + modifiers
# ---------------------------------------------------------------------------


class TestFractalBox:
    def test_constructor_kwargs(self):
        f = FractalBox(
            p1=(0, 0, 0),
            p2=(0.1, 0.1, 0.1),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=1,
            mixing_model_id="m1",
            id="fb1",
            seed=42,
        )
        assert f.kwargs["frac_dim"] == 1.5
        assert f.kwargs["seed"] == 42
        assert f.kwargs["id"] == "fb1"

    def test_do_pre_build_default(self):
        f = FractalBox()
        assert f.do_pre_build is True

    def test_pre_build_missing_kwarg_raises(self, stub_grid):
        # ``mixing_model_id`` missing
        f = FractalBox(
            p1=(0, 0, 0),
            p2=(0.1, 0.1, 0.1),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=1,
            id="fb1",
        )
        with pytest.raises(KeyError):
            f.pre_build(stub_grid)


class TestModifiers:
    """Surface roughness, surface water, grass — fractal-box decorators."""

    def test_add_surface_roughness_kwargs(self):
        r = AddSurfaceRoughness(
            p1=(0, 0, 0),
            p2=(0.1, 0.1, 0),
            frac_dim=1.5,
            weighting=(1, 1),
            limits=(0.0, 0.005),
            fractal_box_id="fb1",
            seed=1,
        )
        assert r.kwargs["fractal_box_id"] == "fb1"
        assert r.kwargs["limits"] == (0.0, 0.005)

    def test_add_surface_water_kwargs(self):
        w = AddSurfaceWater(
            p1=(0, 0, 0),
            p2=(0.1, 0.1, 0),
            depth=0.005,
            fractal_box_id="fb1",
        )
        assert w.kwargs["depth"] == 0.005
        assert w.kwargs["fractal_box_id"] == "fb1"

    def test_add_grass_kwargs(self):
        g = AddGrass(
            p1=(0, 0, 0),
            p2=(0.1, 0.1, 0),
            frac_dim=1.5,
            limits=(0.0, 0.005),
            n_blades=10,
            fractal_box_id="fb1",
            seed=1,
        )
        assert g.kwargs["n_blades"] == 10
        assert g.kwargs["fractal_box_id"] == "fb1"


# ---------------------------------------------------------------------------
# GeometryObjectsRead
# ---------------------------------------------------------------------------


class TestGeometryObjectsRead:
    def test_constructor_kwargs(self):
        g = GeometryObjectsRead(p1=(0, 0, 0), geofile="objs.h5", matfile="objs_materials.txt")
        assert g.kwargs["geofile"] == "objs.h5"
        assert g.kwargs["matfile"] == "objs_materials.txt"


pytestmark = pytest.mark.unit
