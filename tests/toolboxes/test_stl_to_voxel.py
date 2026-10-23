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
import pytest
from stl import mesh as stl_mesh

import gprMax
import gprMax.model as model_mod
from gprMax.toolboxes.STLtoVoxel import stltovoxel
from gprMax.toolboxes.STLtoVoxel.convert import (
    STLVoxels,
    convert_files,
    convert_meshes,
    rotation_matrix,
    voxelise_stl_files,
)

DX_MM = 2.0
CUBE_FACES = np.array(
    [
        [0, 3, 1],
        [1, 3, 2],
        [0, 4, 7],
        [0, 7, 3],
        [4, 5, 6],
        [4, 6, 7],
        [5, 1, 2],
        [5, 2, 6],
        [2, 3, 6],
        [3, 7, 6],
        [0, 1, 5],
        [0, 5, 4],
    ]
)


def _box_triangles(lower, upper):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    corners = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=float,
    )
    return (lower + corners * (upper - lower))[CUBE_FACES]


def _sphere_triangles(centre, radius, segments=96, rings=48):
    theta = np.linspace(0, np.pi, rings + 1)
    phi = np.linspace(0, 2 * np.pi, segments + 1)
    points = np.stack(
        [
            np.outer(np.sin(theta), np.cos(phi)),
            np.outer(np.sin(theta), np.sin(phi)),
            np.outer(np.cos(theta), np.ones_like(phi)),
        ],
        axis=-1,
    )
    points = np.asarray(centre, dtype=float) + radius * points
    triangles = []
    for i in range(rings):
        for j in range(segments):
            a, b = points[i, j], points[i, j + 1]
            c, d = points[i + 1, j], points[i + 1, j + 1]
            if i > 0:
                triangles.append([a, c, b])
            if i < rings - 1:
                triangles.append([b, c, d])
    return np.asarray(triangles)


def _write_stl(path, triangles):
    surface = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    surface.vectors[:] = triangles
    surface.save(str(path))
    return path


def _cell_centres(n=50, dx=DX_MM):
    centres = (np.arange(n) + 0.5) * dx
    return np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), axis=-1)


def _on_model_grid(voxels, n=50, dx=DX_MM):
    """Place the file's cells at p1 = origin on an n^3 grid of dx (mm) cells."""

    first = np.rint(np.asarray(voxels.origin) * 1000 / dx).astype(int)
    np.testing.assert_allclose(first * dx, np.asarray(voxels.origin) * 1000, atol=1e-9)
    model = np.zeros((n, n, n), dtype=bool)
    occupied = np.argwhere(voxels.data >= 0) + first
    model[tuple(occupied.T)] = True
    return model


def test_on_grid_cube_fills_exactly_its_cells(tmp_path):
    stl = _write_stl(tmp_path / "cube.stl", _box_triangles((30, 30, 30), (50, 50, 50)))

    voxels = voxelise_stl_files([stl], (0.002, 0.002, 0.002))

    assert voxels.data.shape == (10, 10, 10)
    assert np.count_nonzero(voxels.data >= 0) == 1000
    np.testing.assert_allclose(voxels.origin, (0.03, 0.03, 0.03), rtol=0, atol=1e-15)
    np.testing.assert_allclose(voxels.spacing, (0.002, 0.002, 0.002))


def test_off_grid_cube_fills_the_cells_whose_centres_it_contains(tmp_path):
    lower = np.array((31.3, 32.7, 34.1))
    stl = _write_stl(tmp_path / "cube.stl", _box_triangles(lower, lower + 20))

    voxels = voxelise_stl_files([stl], (0.002, 0.002, 0.002))

    np.testing.assert_allclose(voxels.origin, (0.030, 0.032, 0.034), rtol=0, atol=1e-15)
    centres = _cell_centres()
    expected = np.all((centres >= lower) & (centres <= lower + 20), axis=-1)
    assert np.count_nonzero(expected) == 1000
    np.testing.assert_array_equal(_on_model_grid(voxels), expected)


def test_sphere_fills_the_cells_whose_centres_it_contains(tmp_path):
    centre = np.array((50.0, 50.0, 50.0))
    stl = _write_stl(tmp_path / "sphere.stl", _sphere_triangles(centre, 15.0))

    voxels = voxelise_stl_files([stl], (0.002, 0.002, 0.002))

    expected = np.sum((_cell_centres() - centre) ** 2, axis=-1) <= 15.0**2
    assert np.count_nonzero(expected) == 1736
    np.testing.assert_array_equal(_on_model_grid(voxels), expected)


def test_rotate_option_turns_a_box_about_its_centre(tmp_path, monkeypatch):
    centre = np.array((50.15, 49.85, 50.37))
    half = np.array((15.0, 5.0, 5.0))
    stl = _write_stl(tmp_path / "bar.stl", _box_triangles(centre - half, centre + half))
    monkeypatch.setattr(
        sys,
        "argv",
        ["stltovoxel", str(stl), "-dxdydz", "0.002", "--rotate", "0", "0", "30"],
    )

    stltovoxel.main()

    with h5py.File(tmp_path / "bar_geo.h5") as geometry:
        voxels = STLVoxels(
            geometry["data"][:],
            tuple(geometry.attrs["origin_xyz"]),
            tuple(geometry.attrs["dx_dy_dz"]),
        )
        settings = json.loads(geometry.attrs["conversion_settings"])
    local = (_cell_centres() - centre) @ rotation_matrix((0, 0, 30))
    expected = np.all(np.abs(local) <= half, axis=-1)
    assert np.count_nonzero(expected) == 375
    np.testing.assert_array_equal(_on_model_grid(voxels), expected)
    assert settings["rotate"] == [0, 0, 30]
    assert settings["rotate_about"] == "bbox"
    assert settings["unit"] == "mm"


def test_scale_and_translate_move_the_geometry_in_the_stl_unit(tmp_path):
    stl = _write_stl(tmp_path / "cube.stl", _box_triangles((0, 0, 0), (1, 1, 1)))

    voxels = voxelise_stl_files(
        [stl],
        (0.002, 0.002, 0.002),
        source_unit="mm",
        scale=20,
        translate=(30, 40, 50),
    )

    assert np.count_nonzero(voxels.data >= 0) == 1000
    np.testing.assert_allclose(voxels.origin, (0.03, 0.04, 0.05), rtol=0, atol=1e-15)


def test_rotation_order_is_x_then_y_then_z():
    np.testing.assert_allclose(rotation_matrix((0, 0, 90)) @ (1, 0, 0), (0, 1, 0), atol=1e-15)
    # About x first, then y: x leaves (1, 0, 0) alone and y then takes it to -z;
    # x takes (0, 1, 0) to +z and y then takes that to +x.
    np.testing.assert_allclose(rotation_matrix((90, 90, 0)) @ (1, 0, 0), (0, 0, -1), atol=1e-15)
    np.testing.assert_allclose(rotation_matrix((90, 90, 0)) @ (0, 1, 0), (1, 0, 0), atol=1e-15)


def test_later_files_overwrite_earlier_files_where_they_overlap(tmp_path):
    first = _write_stl(tmp_path / "first.stl", _box_triangles((0, 0, 0), (8, 8, 8)))
    second = _write_stl(tmp_path / "second.stl", _box_triangles((4, 4, 4), (12, 12, 12)))

    data = convert_files([first, second], (0.002, 0.002, 0.002))

    assert np.count_nonzero(data == 1) == 4**3
    assert np.count_nonzero(data == 0) == 4**3 - 2**3


def test_parallel_option_is_deprecated_and_does_not_change_the_result():
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
    with pytest.warns(DeprecationWarning, match="parallel"):
        parallel, *_ = convert_meshes([mesh], (0.001, 0.001, 0.001), parallel=True)

    # Cell centres (i + 0.5) mm inside x + y + z <= 10 mm.
    centres = np.arange(10) + 0.5
    x, y, z = np.meshgrid(centres, centres, centres, indexing="ij")
    assert np.count_nonzero(serial >= 0) == np.count_nonzero(x + y + z <= 10)
    np.testing.assert_array_equal(parallel, serial)


def test_anisotropic_discretisation_sets_each_axis_independently():
    mesh = _box_triangles((0, 0, 0), (10, 20, 30))

    vol, scale, shift = convert_meshes([mesh], (0.001, 0.002, 0.003))

    np.testing.assert_allclose(scale, [1, 0.5, 1 / 3])
    np.testing.assert_allclose(shift, [0, 0, 0], atol=1e-12)
    # convert_meshes keeps its historical z, y, x axis order.
    assert vol.shape == (10, 10, 10)
    assert np.all(vol == 0)


def _capture_built_grid(monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _imported_cells(tmp_path, monkeypatch, name, objects):
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=name))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.Domain(p1=(0.1, 0.1, 0.1)))
    scene.add(gprMax.TimeWindow(iterations=1))
    for user_object in objects:
        scene.add(user_object)
    captured = _capture_built_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / name,
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    ids = {material.ID: material.numID for material in grid.materials}
    solid = grid.solid[:50, :50, :50]
    return solid, ids


def _convert_cube(tmp_path, monkeypatch, lower, dxdydz="0.002"):
    """Convert a 20 mm cube with the CLI and fill in its material."""

    stl = _write_stl(tmp_path / "cube.stl", _box_triangles(lower, np.asarray(lower) + 20))
    monkeypatch.setattr(sys, "argv", ["stltovoxel", str(stl), "-dxdydz", dxdydz])
    stltovoxel.main()
    database = tmp_path / "cube_geo_materials.json"
    document = json.loads(database.read_text(encoding="utf-8"))
    (material,) = document["materials"].values()
    material["base"] = {
        "relative_permittivity": 4.0,
        "electric_conductivity_s_per_m": 0.0,
        "relative_permeability": 1.0,
        "magnetic_conductivity_s_per_m": 0.0,
    }
    database.write_text(json.dumps(document), encoding="utf-8")
    with h5py.File(tmp_path / "cube_geo.h5") as geometry:
        origin = tuple(float(value) for value in geometry.attrs["origin_xyz"])
    return origin, database


@pytest.mark.integration
def test_geometry_objects_read_at_origin_puts_cells_where_a_box_does(tmp_path, monkeypatch):
    lower = np.array((31.3, 32.7, 34.1))
    origin, database = _convert_cube(tmp_path, monkeypatch, lower)

    imported, ids = _imported_cells(
        tmp_path,
        monkeypatch,
        "stl_import",
        [
            gprMax.GeometryObjectsRead(
                p1=origin, geofile=str(tmp_path / "cube_geo.h5"), material_database=database.stem
            )
        ],
    )
    box, box_ids = _imported_cells(
        tmp_path,
        monkeypatch,
        "box",
        [
            gprMax.Material(er=4, se=0, mr=1, sm=0, id="cad"),
            gprMax.Box(p1=tuple(lower / 1000), p2=tuple((lower + 20) / 1000), material_id="cad"),
        ],
    )

    imported_cells = imported == ids[f"cube{{{database.stem}}}"]
    box_cells = box == box_ids["cad"]
    assert np.count_nonzero(box_cells) == 1000
    np.testing.assert_array_equal(imported_cells, box_cells)


@pytest.mark.integration
def test_geometry_objects_read_names_both_cell_sizes_when_they_differ(tmp_path, monkeypatch):
    origin, database = _convert_cube(tmp_path, monkeypatch, (30, 30, 30), dxdydz="0.001")
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.Domain(p1=(0.1, 0.1, 0.1)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=origin, geofile=str(tmp_path / "cube_geo.h5"), material_database=database.stem
        )
    )

    with pytest.raises(
        ValueError,
        match=r"file \(0\.001 x 0\.001 x 0\.001 m\) to match the model's "
        r"\(0\.002 x 0\.002 x 0\.002 m\)",
    ):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "mismatch",
            hide_progress_bars=True,
        )


@pytest.mark.integration
def test_geometry_objects_read_warns_when_p1_is_rounded(tmp_path, monkeypatch, capsys):
    origin, database = _convert_cube(tmp_path, monkeypatch, (30, 30, 30))
    shifted = (origin[0] + 0.0007, origin[1], origin[2])

    capsys.readouterr()
    imported, ids = _imported_cells(
        tmp_path,
        monkeypatch,
        "rounded",
        [
            gprMax.GeometryObjectsRead(
                p1=shifted, geofile=str(tmp_path / "cube_geo.h5"), material_database=database.stem
            )
        ],
    )

    output = capsys.readouterr()
    assert "is not on a cell corner" in output.out + output.err
    cells = np.argwhere(imported == ids[f"cube{{{database.stem}}}"])
    np.testing.assert_array_equal(cells.min(axis=0), (15, 15, 15))


def test_cli_writes_stable_material_keys_and_editable_json(tmp_path, monkeypatch):
    stl = tmp_path / "patch metal.stl"
    stl.touch()
    model = np.asarray([[[0]], [[-1]]], dtype=np.int16)
    monkeypatch.setattr(
        stltovoxel,
        "voxelise_stl_files",
        lambda files, dxdydz, **options: STLVoxels(model, (0.0, 0.0, 0.0), dxdydz),
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


def test_prepare_assignments_refuses_to_overwrite_without_flag(tmp_path, monkeypatch, capsys):
    source = tmp_path / "organ.stl"
    source.touch()
    assignments = tmp_path / "anatomy.csv"
    assignments.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["stltovoxel", str(source), "--prepare", str(assignments)],
    )

    with pytest.raises(SystemExit) as error:
        stltovoxel.main()

    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert assignments.read_text(encoding="utf-8") == "existing"


def test_prepare_assignments_overwrites_with_flag(tmp_path, monkeypatch):
    source = tmp_path / "organ.stl"
    source.touch()
    assignments = tmp_path / "anatomy.csv"
    assignments.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["stltovoxel", str(source), "--prepare", str(assignments), "--overwrite"],
    )

    stltovoxel.main()

    assert assignments.read_text(encoding="utf-8").startswith(
        "file,include,priority,material_name,geometry_tag\n"
    )


@pytest.mark.parametrize("name", ["soil+air", "+soil", "soil+", "Hmag_a+a+b+b"])
def test_stl_assignments_reject_reserved_material_names(tmp_path, name):
    source = tmp_path / "part.stl"
    assignments = tmp_path / "assignments.csv"
    assignments.write_text(
        "file,include,priority,material_name,geometry_tag\n" f"part.stl,y,0,{name},part\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved for automatically averaged material"):
        stltovoxel.read_assignments([source], assignments)


def test_default_stl_material_name_requires_assignment_if_stem_contains_plus(tmp_path):
    with pytest.raises(ValueError, match="reserved for automatically averaged material"):
        stltovoxel.read_assignments([tmp_path / "part+detail.stl"])


def test_stl_filename_can_contain_plus_with_explicit_safe_material(tmp_path):
    source = tmp_path / "part+detail.stl"
    assignments = tmp_path / "assignments.csv"
    assignments.write_text(
        "file,include,priority,material_name,geometry_tag\n"
        "part+detail.stl,y,0,plastic,part\n",
        encoding="utf-8",
    )

    result = stltovoxel.read_assignments([source], assignments)

    assert result[0].path == source
    assert result[0].material_name == "plastic"


@pytest.mark.parametrize("field", ["name", "original_id"])
def test_stl_preserved_database_rejects_reserved_material_names(tmp_path, field):
    path = tmp_path / "materials.json"
    entry = {"name": "soil", "metadata": {"original_id": "soil"}}
    stltovoxel._write_or_preserve_database(path, "materials", ["soil"], {"soil": entry})
    document = json.loads(path.read_text(encoding="utf-8"))
    material = document["materials"]["soil"]
    (material if field == "name" else material["metadata"])[field] = "soil+air"
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ValueError, match="reserved for automatically averaged material"):
        stltovoxel._write_or_preserve_database(path, "materials", ["soil"], {"soil": entry})

    assert path.read_bytes() == original


def test_stl_invalid_material_stops_before_voxelisation(tmp_path, monkeypatch, capsys):
    source = tmp_path / "soil+air.stl"
    source.touch()
    monkeypatch.setattr(sys, "argv", ["stltovoxel", str(source), "-dxdydz", "0.001"])

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("An invalid material name reached voxelisation")

    monkeypatch.setattr(stltovoxel, "voxelise_stl_files", unexpected_conversion)
    with pytest.raises(SystemExit) as error:
        stltovoxel.main()

    assert error.value.code == 2
    assert "reserved for automatically averaged material" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.h5"))
