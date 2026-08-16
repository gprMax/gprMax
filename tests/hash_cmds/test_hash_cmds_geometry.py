"""Tests for ``gprMax.hash_cmds_geometry.process_geometrycmds``.

The geometry dispatcher walks a list of raw command strings (preserved in
their original input order) and ``split()``-tokenises each one. A trailing
colon is part of ``tmp[0]`` (``"#box:"``, not ``"#box"``). Per shape it
validates arity and constructs the matching ``GeometryUserObject``.

Tests drive the dispatcher with hand-built lists — no file I/O, no globals.
"""

import numpy as np
import pytest

from gprMax.hash_cmds_geometry import process_geometrycmds
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
from gprMax.user_objects.cmds_geometry.plate import Plate
from gprMax.user_objects.cmds_geometry.sphere import Sphere
from gprMax.user_objects.cmds_geometry.triangle import Triangle

# ---------------------------------------------------------------------------
# Sanity: empty input
# ---------------------------------------------------------------------------


class TestEmptyDispatch:
    def test_empty_list_yields_empty_list(self):
        assert process_geometrycmds([]) == []


# ---------------------------------------------------------------------------
# Geometry objects read
# ---------------------------------------------------------------------------


class TestGeometryObjectsRead:
    def test_six_token_form(self):
        from gprMax.user_objects.cmds_geometry.geometry_objects_read import GeometryObjectsRead

        objs = process_geometrycmds(["#geometry_objects_read: 0 0 0 geo.bin mat.txt"])
        gor = objs[0]
        assert isinstance(gor, GeometryObjectsRead)
        assert gor.kwargs["p1"] == (0.0, 0.0, 0.0)
        assert gor.kwargs["geofile"] == "geo.bin"
        assert gor.kwargs["matfile"] == "mat.txt"

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#geometry_objects_read: 0 0 0 geo.bin"])


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


class TestEdge:
    def test_eight_token_form(self):
        objs = process_geometrycmds(["#edge: 0 0 0 0.1 0.2 0.3 wire"])
        edge = objs[0]
        assert isinstance(edge, Edge)
        assert edge.kwargs["p1"] == (0.0, 0.0, 0.0)
        assert edge.kwargs["p2"] == (0.1, 0.2, 0.3)
        assert edge.kwargs["material_id"] == "wire"

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#edge: 0 0 0 0.1 0.2 0.3"])


# ---------------------------------------------------------------------------
# Plate (isotropic / anisotropic)
# ---------------------------------------------------------------------------


class TestPlate:
    def test_eight_token_isotropic(self):
        objs = process_geometrycmds(["#plate: 0 0 0 0.1 0.1 0 m1"])
        plate = objs[0]
        assert isinstance(plate, Plate)
        assert plate.kwargs["material_id"] == "m1"

    def test_nine_token_anisotropic(self):
        objs = process_geometrycmds(["#plate: 0 0 0 0.1 0.1 0 m1 m2"])
        plate = objs[0]
        assert plate.kwargs["material_ids"] == ["m1", "m2"]

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#plate: 0 0 0 0.1 0.1"])


# ---------------------------------------------------------------------------
# Triangle (3 isotropic widths + anisotropic)
# ---------------------------------------------------------------------------


class TestTriangle:
    def test_twelve_token_isotropic_no_averaging(self):
        objs = process_geometrycmds(["#triangle: 0 0 0 1 0 0 0 1 0 0.01 m1"])
        tri = objs[0]
        assert isinstance(tri, Triangle)
        assert tri.kwargs["thickness"] == 0.01
        assert tri.kwargs["material_id"] == "m1"

    def test_thirteen_token_isotropic_with_averaging(self):
        objs = process_geometrycmds(["#triangle: 0 0 0 1 0 0 0 1 0 0.01 m1 y"])
        assert objs[0].kwargs["averaging"] == "y"

    def test_fourteen_token_anisotropic(self):
        objs = process_geometrycmds(["#triangle: 0 0 0 1 0 0 0 1 0 0.01 mx my mz"])
        assert objs[0].kwargs["material_ids"] == ["mx", "my", "mz"]

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#triangle: 0 0 0 1 0 0 0 1 0 0.01"])


# ---------------------------------------------------------------------------
# Box (8 / 9 / 10)
# ---------------------------------------------------------------------------


class TestBox:
    def test_eight_token_isotropic(self):
        objs = process_geometrycmds(["#box: 0 0 0 0.1 0.1 0.1 m1"])
        box = objs[0]
        assert isinstance(box, Box)
        assert box.kwargs["p1"] == (0.0, 0.0, 0.0)
        assert box.kwargs["p2"] == (0.1, 0.1, 0.1)
        assert box.kwargs["material_id"] == "m1"

    def test_nine_token_isotropic_with_averaging(self):
        objs = process_geometrycmds(["#box: 0 0 0 0.1 0.1 0.1 m1 n"])
        assert objs[0].kwargs["averaging"] == "n"

    def test_ten_token_anisotropic(self):
        objs = process_geometrycmds(["#box: 0 0 0 0.1 0.1 0.1 mx my mz"])
        assert objs[0].kwargs["material_ids"] == ["mx", "my", "mz"]

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#box: 0 0 0 0.1 0.1 0.1"])

    def test_too_many_tokens_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#box: 0 0 0 0.1 0.1 0.1 mx my mz extra"])


# ---------------------------------------------------------------------------
# Cylinder (9 / 10 / 11)
# ---------------------------------------------------------------------------


class TestCylinder:
    def test_nine_token_isotropic(self):
        objs = process_geometrycmds(["#cylinder: 0 0 0 0.1 0.1 0.1 0.05 m1"])
        cyl = objs[0]
        assert isinstance(cyl, Cylinder)
        assert cyl.kwargs["r"] == 0.05
        assert cyl.kwargs["material_id"] == "m1"

    def test_ten_token_isotropic_with_averaging(self):
        objs = process_geometrycmds(["#cylinder: 0 0 0 0.1 0.1 0.1 0.05 m1 y"])
        assert objs[0].kwargs["averaging"] == "y"

    def test_eleven_token_anisotropic(self):
        objs = process_geometrycmds(["#cylinder: 0 0 0 0.1 0.1 0.1 0.05 mx my mz"])
        assert objs[0].kwargs["material_ids"] == ["mx", "my", "mz"]

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#cylinder: 0 0 0 0.1 0.1 0.1 0.05"])


# ---------------------------------------------------------------------------
# Cone (10 / 11 / 12)
# ---------------------------------------------------------------------------


class TestCone:
    def test_ten_token_isotropic(self):
        objs = process_geometrycmds(["#cone: 0 0 0 0.1 0.1 0.1 0.05 0.02 m1"])
        cone = objs[0]
        assert isinstance(cone, Cone)
        assert cone.kwargs["r1"] == 0.05
        assert cone.kwargs["r2"] == 0.02
        assert cone.kwargs["material_id"] == "m1"

    def test_eleven_token_isotropic_with_averaging(self):
        objs = process_geometrycmds(["#cone: 0 0 0 0.1 0.1 0.1 0.05 0.02 m1 n"])
        assert objs[0].kwargs["averaging"] == "n"

    def test_twelve_token_anisotropic(self):
        objs = process_geometrycmds(["#cone: 0 0 0 0.1 0.1 0.1 0.05 0.02 mx my mz"])
        assert objs[0].kwargs["material_ids"] == ["mx", "my", "mz"]

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#cone: 0 0 0 0.1 0.1 0.1 0.05 0.02"])


# ---------------------------------------------------------------------------
# Cylindrical sector (10 / 11 / 12)
# ---------------------------------------------------------------------------


class TestCylindricalSector:
    def test_ten_token_isotropic(self):
        objs = process_geometrycmds(["#cylindrical_sector: X 0.1 0.1 0 0.1 0.05 0 90 m1"])
        sec = objs[0]
        assert isinstance(sec, CylindricalSector)
        # axis is lowercased
        assert sec.kwargs["normal"] == "x"
        assert sec.kwargs["material_id"] == "m1"

    def test_eleven_token_isotropic_with_averaging(self):
        objs = process_geometrycmds(["#cylindrical_sector: x 0.1 0.1 0 0.1 0.05 0 90 m1 y"])
        assert objs[0].kwargs["averaging"] == "y"

    def test_twelve_token_anisotropic(self):
        objs = process_geometrycmds(["#cylindrical_sector: x 0.1 0.1 0 0.1 0.05 0 90 mx my mz"])
        assert objs[0].kwargs["material_ids"] == ["mx", "my", "mz"]

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#cylindrical_sector: x 0.1 0.1 0 0.1 0.05 0 90"])


# ---------------------------------------------------------------------------
# Sphere (6 / 7 / 8) — anisotropic branch is buggy (see tripwire below)
# ---------------------------------------------------------------------------


class TestSphere:
    def test_six_token_isotropic(self):
        objs = process_geometrycmds(["#sphere: 0.05 0.05 0.05 0.02 m1"])
        sphere = objs[0]
        assert isinstance(sphere, Sphere)
        assert sphere.kwargs["material_id"] == "m1"

    def test_seven_token_isotropic_with_averaging(self):
        objs = process_geometrycmds(["#sphere: 0.05 0.05 0.05 0.02 m1 y"])
        assert objs[0].kwargs["averaging"] == "y"

    def test_too_few_tokens_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#sphere: 0.05 0.05 0.05 0.02"])

    def test_too_many_tokens_rejected(self):
        # 9 tokens: longer than any valid form
        with pytest.raises(ValueError):
            process_geometrycmds(["#sphere: 0.05 0.05 0.05 0.02 mx my mz extra"])


class TestSphereAnisotropicMaterials:
    def test_anisotropic_branch_uses_plural_material_ids(self):
        objs = process_geometrycmds(["#sphere: 0 0 0 0.02 mx my mz"])
        sphere = objs[0]
        assert sphere.kwargs["material_ids"] == ["mx", "my", "mz"]
        assert "material_id" not in sphere.kwargs


# ---------------------------------------------------------------------------
# Ellipsoid — duplicate elif clauses make the anisotropic branch dead code
# ---------------------------------------------------------------------------


class TestEllipsoid:
    def test_eight_token_treated_as_isotropic(self):
        objs = process_geometrycmds(["#ellipsoid: 0 0 0 0.1 0.1 0.1 m1"])
        ell = objs[0]
        assert isinstance(ell, Ellipsoid)
        assert ell.kwargs["material_id"] == "m1"

    def test_nine_token_with_averaging(self):
        objs = process_geometrycmds(["#ellipsoid: 0 0 0 0.1 0.1 0.1 m1 y"])
        assert objs[0].kwargs["averaging"] == "y"

    def test_too_few_tokens_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#ellipsoid: 0 0 0 0.1 0.1 0.1"])


class TestEllipsoidAnisotropicDeadCodeBug:
    """Bug tripwire: ``hash_cmds_geometry.py:349``.

    ``#ellipsoid`` lists branches in the order:

    * ``len == 8`` → isotropic (no averaging)
    * ``len == 9`` → isotropic with averaging
    * ``len == 8`` → *anisotropic* — duplicate condition, never reached
    * ``else``    → ``ValueError``

    So passing an 8-token line always lands on the first branch (a single
    ``material_id`` string) regardless of whether the user intended three
    anisotropic materials. There is no way to invoke the anisotropic
    branch through this dispatcher.

    Pin the current behaviour: 8 tokens → ``material_id`` is a string
    (the last token); ``material_ids`` is never populated.
    """

    def test_8_tokens_route_to_isotropic_not_anisotropic(self):
        # An 8-token input where the user *wanted* anisotropy: the
        # last three tokens read like material names, but only the
        # final one is taken as ``material_id`` (a string).
        objs = process_geometrycmds(["#ellipsoid: 0 0 0 0.1 0.1 0.1 mz"])
        ell = objs[0]
        assert ell.kwargs["material_id"] == "mz"
        assert isinstance(ell.kwargs["material_id"], str)
        assert "material_ids" not in ell.kwargs


# ---------------------------------------------------------------------------
# Fractal box and modifiers
# ---------------------------------------------------------------------------


class TestFractalBox:
    def test_fourteen_token_minimal(self):
        objs = process_geometrycmds(["#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1"])
        fb = objs[0]
        assert isinstance(fb, FractalBox)
        assert fb.kwargs["frac_dim"] == 1.5
        assert fb.kwargs["mixing_model_id"] == "mix"
        assert fb.kwargs["id"] == "fb1"
        np.testing.assert_array_equal(fb.kwargs["weighting"], np.array([1.0, 1.0, 1.0]))

    def test_fifteen_token_with_seed(self):
        objs = process_geometrycmds(["#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1 42"])
        # Seed is included
        assert "seed" in objs[0].kwargs

    def test_sixteen_token_with_seed_and_averaging(self):
        objs = process_geometrycmds(["#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1 42 y"])
        assert objs[0].kwargs["averaging"] == "y"

    def test_too_few_tokens_rejected(self):
        with pytest.raises(ValueError):
            process_geometrycmds(["#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix"])


class TestFractalBoxSeedTypeBug:
    """Bug tripwire: ``hash_cmds_geometry.py:393``.

    The 15-token ``#fractal_box`` branch passes ``seed=tmp[14]`` —
    *unconverted string*. Sibling commands convert: ``#add_surface_roughness``
    line 453 and ``#add_grass`` line 517 both pass ``seed=int(tmp[N])``.
    This inconsistency means a fractal-box ``seed`` lands in ``kwargs`` as
    a ``str`` while modifier seeds land as ``int``.

    Pin the current behaviour. If the dispatcher is later fixed to cast
    consistently (``seed=int(tmp[14])``), this assertion fails — flip to
    ``isinstance(..., int)``.
    """

    def test_fractal_box_seed_stored_as_string(self):
        objs = process_geometrycmds(["#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1 42"])
        assert isinstance(objs[0].kwargs["seed"], str)
        assert objs[0].kwargs["seed"] == "42"


class TestFractalBoxModifiers:
    """Modifier commands (#add_surface_roughness, #add_surface_water,
    #add_grass) are only processed *inside* a matching #fractal_box branch.
    The inner loop iterates all geometry commands and accepts only those
    whose fractal_box_id matches the current ID.
    """

    def test_surface_roughness_attached_to_fractal_box(self):
        geometry = [
            "#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1",
            "#add_surface_roughness: 0 0.1 0 0.1 0.1 0.1 1.5 1 1 0 0.01 fb1 7",
        ]
        objs = process_geometrycmds(geometry)
        types = [type(o) for o in objs]
        # FractalBox parsed first, modifier scanned and appended afterwards
        assert types == [FractalBox, AddSurfaceRoughness]

    def test_surface_water_attached_to_fractal_box(self):
        geometry = [
            "#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1",
            "#add_surface_water: 0 0.1 0 0.1 0.1 0.1 0.005 fb1",
        ]
        objs = process_geometrycmds(geometry)
        assert any(isinstance(o, AddSurfaceWater) for o in objs)

    def test_grass_attached_to_fractal_box(self):
        geometry = [
            "#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1",
            "#add_grass: 0 0.1 0 0.1 0.1 0.1 1.5 0.001 0.01 50 fb1",
        ]
        objs = process_geometrycmds(geometry)
        assert any(isinstance(o, AddGrass) for o in objs)

    def test_modifier_with_mismatched_id_rejected(self):
        geometry = [
            "#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1",
            "#add_surface_roughness: 0 0.1 0 0.1 0.1 0.1 1.5 1 1 0 0.01 fb_other 7",
        ]
        with pytest.raises(ValueError, match="cannot find #fractal_box"):
            process_geometrycmds(geometry)

    def test_modifier_without_parent_fractal_box_rejected(self):
        geometry = [
            "#add_surface_roughness: 0 0.1 0 0.1 0.1 0.1 1.5 1 1 0 0.01 fb1 7",
        ]
        with pytest.raises(ValueError, match="cannot find #fractal_box"):
            process_geometrycmds(geometry)


class TestFractalBoxModifierArity:
    def test_surface_roughness_wrong_arity_rejected(self):
        geometry = [
            "#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1",
            "#add_surface_roughness: 0 0.1 0 0.1 0.1 0.1 1.5 1 1 0 0.01",
        ]
        with pytest.raises(ValueError):
            process_geometrycmds(geometry)

    def test_surface_water_wrong_arity_rejected(self):
        geometry = [
            "#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1",
            "#add_surface_water: 0 0.1 0 0.1 0.1 0.1 0.005",
        ]
        with pytest.raises(ValueError):
            process_geometrycmds(geometry)

    def test_grass_wrong_arity_rejected(self):
        geometry = [
            "#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 4 mix fb1",
            "#add_grass: 0 0.1 0 0.1 0.1 0.1 1.5 0.001 0.01 50",
        ]
        with pytest.raises(ValueError):
            process_geometrycmds(geometry)


# ---------------------------------------------------------------------------
# Unknown commands — outer chain has no else, so they're silently ignored
# ---------------------------------------------------------------------------


class TestUnknownCommandSilentlyDropped:
    """The outer ``for object in geometry`` loop has no final ``else``
    clause. An unrecognised hash command produces no scene object and no
    error — pin this so it's intentional, not a regression.
    """

    def test_unknown_command_yields_nothing(self):
        objs = process_geometrycmds(["#not_a_real_command: 1 2 3"])
        assert objs == []


pytestmark = pytest.mark.unit
