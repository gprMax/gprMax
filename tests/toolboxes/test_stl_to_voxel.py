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

import json
import sys

import h5py
import numpy as np

from toolboxes.STLtoVoxel import stltovoxel
from toolboxes.STLtoVoxel.convert import convert_meshes
from toolboxes.STLtoVoxel.slice import calculate_scale_shift


def test_parallel_conversion_matches_serial_conversion():
    mesh = np.array(
        [
            [[0, 0, 0], [10, 0, 0], [0, 10, 0]],
            [[0, 0, 0], [10, 0, 0], [0, 0, 10]],
            [[0, 0, 0], [0, 10, 0], [0, 0, 10]],
            [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        ],
        dtype=float,
    )

    serial, *_ = convert_meshes([mesh], (0.001, 0.001, 0.001), parallel=False)
    parallel, *_ = convert_meshes([mesh], (0.001, 0.001, 0.001), parallel=True)

    assert np.count_nonzero(serial >= 0) > 0
    np.testing.assert_array_equal(parallel, serial)


def test_anisotropic_discretisation_sets_each_axis_independently():
    mesh = np.array(
        [[[0, 0, 0], [10, 0, 0], [0, 20, 30]]],
        dtype=float,
    )

    scale, _, shape = calculate_scale_shift([mesh], (0.001, 0.002, 0.003))

    np.testing.assert_allclose(scale, [1, 0.5, 1 / 3])
    assert shape == [11, 11, 11]


def test_cli_writes_stable_material_keys_and_editable_json(tmp_path, monkeypatch):
    stl = tmp_path / "patch metal.stl"
    stl.touch()
    model = np.asarray([[[0]], [[-1]]], dtype=np.int16)
    monkeypatch.setattr(
        stltovoxel,
        "convert_files",
        lambda files, dxdydz, source_unit="mm": model,
    )
    monkeypatch.setattr(sys, "argv", ["stltovoxel", str(stl), "-dxdydz", "0.001"])

    stltovoxel.main()

    geometry_path = tmp_path / "patch metal_geo.h5"
    database_path = tmp_path / "patch_metal_geo_materials.json"
    with h5py.File(geometry_path) as geometry:
        keys = [value.decode() for value in geometry["material_keys"][:]]
        assert keys == ["material_000_patch_metal"]
        assert geometry.attrs["MaterialDatabase"] == "patch_metal_geo_materials"
        assert [value.decode() for value in geometry["tag_names"][:]] == [
            "untagged",
            "patch_metal",
        ]
        np.testing.assert_array_equal(geometry["tag_data"][:].ravel(), [1, 0])
    database = json.loads(database_path.read_text(encoding="utf-8"))
    assert list(database["materials"]) == keys
    entry = database["materials"][keys[0]]
    assert entry["base"]["relative_permittivity"] is None
    assert entry["metadata"]["original_id"] == "patch metal"

    entry["base"] = {
        "relative_permittivity": 4.0,
        "electric_conductivity_s_per_m": 0.01,
        "relative_permeability": 1.0,
        "magnetic_conductivity_s_per_m": 0.0,
    }
    database_path.write_text(json.dumps(database), encoding="utf-8")

    # Re-voxelising must not destroy properties that the user entered into
    # the companion database.
    stltovoxel.main()
    preserved = json.loads(database_path.read_text(encoding="utf-8"))
    assert preserved["materials"][keys[0]]["base"]["relative_permittivity"] == 4.0


def test_assignments_separate_part_tags_from_shared_material(tmp_path):
    first = tmp_path / "left eye.stl"
    second = tmp_path / "right eye.stl"
    first.touch()
    second.touch()
    assignments = tmp_path / "anatomy.csv"
    assignments.write_text(
        "file,include,priority,material_name,geometry_tag\n"
        "left eye.stl,y,0,vitreous,left_eye\n"
        "right eye.stl,y,1,vitreous,right_eye\n",
        encoding="utf-8",
    )

    result = stltovoxel.read_assignments([first, second], assignments)

    assert [item.material_name for item in result] == ["vitreous", "vitreous"]
    assert [item.geometry_tag for item in result] == ["left_eye", "right_eye"]


def test_excluded_stl_may_leave_material_and_tag_blank(tmp_path):
    source = tmp_path / "external surface.stl"
    source.touch()
    assignments = tmp_path / "anatomy.csv"
    assignments.write_text(
        "file,include,priority,material_name,geometry_tag\n" "external surface.stl,n,0,,\n",
        encoding="utf-8",
    )

    result = stltovoxel.read_assignments([source], assignments)

    assert result[0].include is False
    assert result[0].material_name is None
    assert result[0].geometry_tag is None


def test_prepare_assignments_does_not_require_voxel_size(tmp_path, monkeypatch):
    source = tmp_path / "organ.stl"
    source.touch()
    assignments = tmp_path / "anatomy.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["stltovoxel", str(source), "--prepare", str(assignments)],
    )

    stltovoxel.main()

    assert assignments.read_text(encoding="utf-8").startswith(
        "file,include,priority,material_name,geometry_tag\n"
    )
