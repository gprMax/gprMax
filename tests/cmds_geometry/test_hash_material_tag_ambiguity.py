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

"""A smoothing-like material ID must never silently erase anisotropy."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod
from gprMax.hash_cmds_file import get_user_objects
from gprMax.hash_cmds_geometry import process_geometrycmds


PREFIXES = (
    "#box: .01 .01 .01 .03 .03 .03",
    "#sphere: .02 .02 .02 .008",
    "#ellipsoid: .02 .02 .02 .008 .006 .004",
    "#cylinder: .01 .02 .02 .03 .02 .02 .006",
    "#cone: .01 .02 .02 .03 .02 .02 .006 .004",
    "#cylindrical_sector: z .02 .02 .01 .03 .006 0 90",
    "#triangle: .01 .01 .01 .03 .01 .01 .01 .03 .01 .004",
    "#helix: .02 .02 .01 .02 .02 .03 .008 .01 2 .002 r",
)
HEADER = """#domain: .04 .04 .04
#dx_dy_dz: .002 .002 .002
#time_window: 1e-11
#pml_cells: 0
#num_threads: 1
#material: 2 0 1 0 a
#material: 4 0 1 0 n
#material: 6 0 1 0 c
"""


def _grid(*names, surface_names=()):
    return SimpleNamespace(
        materials=[SimpleNamespace(ID=name) for name in names],
        surface_impedance_models=dict.fromkeys(surface_names),
    )


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("marker", ("y", "n", "Y", "N"))
def test_all_volume_parsers_retain_collision_for_build_time_check(prefix, marker):
    (obj,) = process_geometrycmds([f"{prefix} a {marker} c"])
    with pytest.raises(ValueError, match="ambiguous.*material ID.*smoothing flag"):
        obj.validate_hash_material_tag(_grid("a", marker, "c"))
    # A missing third material must not hide a likely anisotropic typo either.
    with pytest.raises(ValueError, match="tag=<name>"):
        obj.validate_hash_material_tag(_grid("a", marker))
    # Ordinary tagged geometry without a colliding material still works.
    obj.validate_hash_material_tag(_grid("a"))


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("marker", ("y", "n", "Y", "N"))
def test_explicit_tags_and_explicit_directional_tags_are_unambiguous(prefix, marker):
    (isotropic,) = process_geometrycmds([f"{prefix} a {marker} tag=c"])
    isotropic.validate_hash_material_tag(_grid("a", marker, "c"))
    assert isotropic.kwargs["material_id"] == "a"
    assert isotropic.kwargs["tag"] == "c"
    for tag in ("body", "tag=body"):
        (directional,) = process_geometrycmds([f"{prefix} a {marker} c n {tag}"])
        directional.validate_hash_material_tag(_grid("a", marker, "c"))
        assert directional.kwargs["material_ids"] == ["a", marker, "c"]
        assert directional.kwargs["tag"] == "body"


def test_material_ids_remain_case_sensitive_and_surface_ids_are_checked():
    (obj,) = process_geometrycmds([f"{PREFIXES[0]} a n c"])
    obj.validate_hash_material_tag(_grid("a", "N", "c"))
    with pytest.raises(ValueError, match="ambiguous"):
        obj.validate_hash_material_tag(_grid("a", surface_names=("n",)))


def test_material_table_is_checked_afresh_not_cached_between_builds():
    (obj,) = process_geometrycmds([f"{PREFIXES[0]} a n c"])
    obj.validate_hash_material_tag(_grid("a"))
    with pytest.raises(ValueError, match="ambiguous"):
        obj.validate_hash_material_tag(_grid("a", "n"))
    obj.validate_hash_material_tag(_grid("a"))


def test_fractal_explicit_tag_still_requires_seed_and_smoothing():
    prefix = "#fractal_box: 0 0 0 .01 .01 .01 1.5 1 1 1 3 mix fb"
    (obj,) = process_geometrycmds([prefix + " 42 y tag=soil"])
    assert obj.kwargs["tag"] == "soil"
    assert obj.kwargs["seed"] == 42
    with pytest.raises(ValueError, match="seed must be a whole number"):
        process_geometrycmds([prefix + " y tag=soil"])


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("marker", ("y", "n", "Y", "N"))
def test_complete_hash_model_rejects_silent_isotropic_substitution(tmp_path, prefix, marker):
    path = tmp_path / "ambiguous.in"
    path.write_text(HEADER.replace("0 n\n", f"0 {marker}\n") + f"{prefix} a {marker} c\n")
    with pytest.raises(ValueError, match="ambiguous"):
        gprMax.run(inputfile=path, geometry_only=True, hide_progress_bars=True)


@pytest.mark.parametrize(
    "tail, expected",
    (
        ("a n tag=c", (2, 2, 2)),
        ("a n c n tag=body", (2, 4, 6)),
        ("n a c", (4, 2, 6)),
        ("a c n", (2, 6, 4)),
    ),
)
def test_complete_hash_models_preserve_intended_component_materials(tmp_path, monkeypatch, tail, expected):
    captured = {}
    original = model_mod.Model.build

    def build(self):
        original(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", build)
    path = tmp_path / "control.in"
    path.write_text(HEADER + f"{PREFIXES[0]} {tail}\n")
    gprMax.run(inputfile=path, geometry_only=True, hide_progress_bars=True)
    grid = captured["grid"]
    actual = [grid.materials[int(i)].er for i in grid.ID[:3, 10, 10, 10]]
    np.testing.assert_array_equal(actual, expected)


def test_api_directional_material_named_n_is_unchanged(tmp_path, monkeypatch):
    captured = {}
    original = model_mod.Model.build

    def build(self):
        original(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", build)
    scene = gprMax.Scene()
    for obj in get_user_objects(HEADER.splitlines()):
        scene.add(obj)
    scene.add(gprMax.Box(p1=(0.01,) * 3, p2=(0.03,) * 3, material_ids=("a", "n", "c")))
    gprMax.run(scenes=[scene], geometry_only=True, outputfile=tmp_path / "api", hide_progress_bars=True)
    grid = captured["grid"]
    np.testing.assert_array_equal([grid.materials[int(i)].er for i in grid.ID[:3, 10, 10, 10]], (2, 4, 6))


@pytest.mark.parametrize("explicit", (False, True))
def test_subgrid_geometry_checks_its_material_table(tmp_path, explicit):
    scene = gprMax.Scene()
    for obj in get_user_objects(HEADER.splitlines()):
        scene.add(obj)
    subgrid = gprMax.SubGridHSG(p1=(0.012,) * 3, p2=(0.028,) * 3, ratio=1, id="patch")
    for name, er in (("a", 2), ("n", 4), ("c", 6)):
        subgrid.add(gprMax.Material(er=er, se=0, mr=1, sm=0, id=name))
    tail = "a n tag=c" if explicit else "a n c"
    (obj,) = process_geometrycmds([f"#box: .016 .016 .016 .024 .024 .024 {tail}"])
    subgrid.add(obj)
    scene.add(subgrid)
    options = dict(
        scenes=[scene], subgrid=True, geometry_only=True, outputfile=tmp_path / "subgrid", hide_progress_bars=True
    )
    if explicit:
        gprMax.run(**options)
    else:
        with pytest.raises(ValueError, match="ambiguous"):
            gprMax.run(**options)


@pytest.mark.parametrize("tag", ("tag=", "tag=bad/name", "tag=untagged"))
def test_explicit_tags_keep_existing_name_validation(tmp_path, tag):
    path = tmp_path / "invalid_tag.in"
    path.write_text(HEADER + f"{PREFIXES[0]} a n {tag}\n")
    with pytest.raises(ValueError, match="Geometry tag"):
        gprMax.run(inputfile=path, geometry_only=True, hide_progress_bars=True)
