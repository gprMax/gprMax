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

"""Hash-command parsing tests for #rebar_mesh."""

import pytest

from gprMax.hash_cmds_file import check_cmd_names
from gprMax.hash_cmds_geometry import process_geometrycmds
from gprMax.user_objects.cmds_geometry.rebar_mesh import RebarMesh

BASE = "#rebar_mesh: 0.008 0.008 0.010 0.040 0.040 0.030 0.008 0.008 0.002 0.002 0.004"


def test_rebar_mesh_is_a_registered_geometry_command():
    line = f"{BASE} pec\n"

    _, _, geometry = check_cmd_names([line], checkessential=False)

    assert geometry == [line.strip()]


def test_isotropic_rebar_mesh_without_optional_fields():
    mesh = process_geometrycmds([f"{BASE} concrete"])[0]

    assert isinstance(mesh, RebarMesh)
    assert mesh.kwargs["p1"] == (0.008, 0.008, 0.010)
    assert mesh.kwargs["p2"] == (0.040, 0.040, 0.030)
    assert mesh.kwargs["spacing"] == (0.008, 0.008)
    assert mesh.kwargs["diameter"] == 0.002
    assert mesh.kwargs["cover"] == 0.002
    assert mesh.kwargs["layer_spacing"] == 0.004
    assert mesh.kwargs["material_id"] == "concrete"
    assert mesh.kwargs["tag"] is None
    assert mesh.bar_count == 8


def test_rebar_mesh_accepts_averaging_and_tag():
    mesh = process_geometrycmds([f"{BASE} concrete n slab"])[0]

    assert mesh.kwargs["material_id"] == "concrete"
    assert mesh.kwargs["averaging"] == "n"
    assert mesh.kwargs["tag"] == "slab"


def test_rebar_mesh_accepts_directional_materials():
    mesh = process_geometrycmds([f"{BASE} mat_x mat_y mat_z"])[0]

    assert mesh.kwargs["material_ids"] == ["mat_x", "mat_y", "mat_z"]
    assert "material_id" not in mesh.kwargs


def test_rebar_mesh_directional_materials_accept_tag_with_marker():
    mesh = process_geometrycmds([f"{BASE} mat_x mat_y mat_z n slab"])[0]

    assert mesh.kwargs["material_ids"] == ["mat_x", "mat_y", "mat_z"]
    assert mesh.kwargs["tag"] == "slab"
    assert "averaging" not in mesh.kwargs


def test_rebar_mesh_directional_materials_accept_smoothing_without_tag():
    mesh = process_geometrycmds([f"{BASE} mat_x mat_y mat_z n"])[0]

    assert mesh.kwargs["material_ids"] == ["mat_x", "mat_y", "mat_z"]
    assert mesh.kwargs["tag"] is None
    assert "averaging" not in mesh.kwargs


def test_rebar_mesh_hash_diagnostic_contains_layout_values():
    mesh = process_geometrycmds([f"{BASE} concrete n slab"])[0]

    text = str(mesh)

    assert text.startswith("#rebar_mesh:")
    assert "0.008" in text
    assert "concrete" in text
    assert "slab" in text


def test_rebar_mesh_requires_twelve_parameters():
    with pytest.raises(ValueError):
        process_geometrycmds(["#rebar_mesh: 0.008 0.008 0.010 0.040 0.040 0.030"])


def test_rebar_mesh_rejects_too_many_parameters():
    with pytest.raises(ValueError):
        process_geometrycmds([f"{BASE} concrete n slab extra"])


def test_rebar_mesh_rejects_invalid_layout_after_parsing():
    with pytest.raises(ValueError, match="bar spacing must be at least"):
        process_geometrycmds(
            [
                "#rebar_mesh: 0.008 0.008 0.010 0.040 0.040 0.030 "
                "0.001 0.008 0.002 0.002 0.004 concrete"
            ]
        )
