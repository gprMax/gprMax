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
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PIL import Image

import gprMax
import gprMax.model as model_mod
from toolboxes.Utilities.convert_png2h5 import Cursor, convert_png


def test_cursor_records_material_on_its_instance(monkeypatch):
    monkeypatch.setattr("toolboxes.Utilities.convert_png2h5.plt.connect", lambda *args: None)
    materials = []
    image = np.array([[[0.1, 0.2, 0.3, 1.0]]])
    cursor = Cursor(image, materials)

    cursor(SimpleNamespace(dblclick=False, xdata=0, ydata=0))

    assert len(materials) == 1
    np.testing.assert_array_equal(materials[0], [25, 51, 76, 255])


def test_cursor_preserves_integer_rgb_values(monkeypatch):
    monkeypatch.setattr("toolboxes.Utilities.convert_png2h5.plt.connect", lambda *args: None)
    materials = []
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    cursor = Cursor(image, materials)

    cursor(SimpleNamespace(dblclick=False, xdata=0, ydata=0))

    np.testing.assert_array_equal(materials[0], [10, 20, 30])


def _png(filename):
    # Rows are stored top to bottom. The converter's established clockwise
    # rotation maps these to an (x, y) cell-centred geometry array.
    pixels = np.asarray(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[0, 0, 255], [255, 0, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(pixels).save(filename)


def _complete_database(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    for index, material in enumerate(document["materials"].values(), start=1):
        material["base"] = {
            "relative_permittivity": float(index + 2),
            "electric_conductivity_s_per_m": 0.0,
            "relative_permeability": 1.0,
            "magnetic_conductivity_s_per_m": 0.0,
        }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def test_convert_png_writes_current_geometry_and_json_material_schema(tmp_path):
    image = tmp_path / "layers.png"
    _png(image)

    result = convert_png(
        image,
        (0.001, 0.002, 0.003),
        ([255, 0, 0], [0, 255, 0]),
        zcells=2,
    )

    assert result.geometry_file == tmp_path / "layers.h5"
    assert result.material_database_file == tmp_path / "layers_materials.json"
    assert result.material_database_id == "layers_materials"
    assert result.material_names == ("rgb_255_0_0", "rgb_0_255_0")
    assert result.shape == (3, 2, 2)
    with h5py.File(result.geometry_file, "r") as geometry:
        assert geometry.attrs["MaterialDatabase"] == "layers_materials"
        assert geometry.attrs["MaterialDatabaseSchemaVersion"] == 1
        assert geometry.attrs["dx_dy_dz"] == pytest.approx((0.001, 0.002, 0.003))
        assert tuple(geometry.attrs["shape_nxyz"]) == (3, 2, 2)
        assert [value.decode() for value in geometry["material_keys"][:]] == list(result.material_keys)
        expected_xy = np.asarray([[-1, 0], [0, 1], [1, -1]], dtype=np.int16)
        np.testing.assert_array_equal(geometry["data"][:, :, 0], expected_xy)
        np.testing.assert_array_equal(geometry["data"][:, :, 1], expected_xy)

    database = json.loads(result.material_database_file.read_text(encoding="utf-8"))
    assert database["schema"] == "gprMax-material-database"
    assert database["schema_version"] == 1
    assert database["database"]["id"] == "layers_materials"
    assert list(database["materials"]) == list(result.material_keys)
    red = database["materials"][result.material_keys[0]]
    assert red["base"]["relative_permittivity"] is None
    assert red["metadata"]["original_id"] == "rgb_255_0_0"
    assert red["metadata"]["selected_colour"] == [255, 0, 0]


def test_convert_png_preserves_edited_material_database(tmp_path):
    image = tmp_path / "layers.png"
    _png(image)
    result = convert_png(image, (0.001, 0.001, 0.001), ([255, 0, 0],), zcells=1)
    _complete_database(result.material_database_file)
    edited = result.material_database_file.read_text(encoding="utf-8")

    convert_png(image, (0.001, 0.001, 0.001), ([255, 0, 0],), zcells=1)

    assert result.material_database_file.read_text(encoding="utf-8") == edited


def test_convert_png_rejects_changed_mapping_before_replacing_geometry(tmp_path):
    image = tmp_path / "layers.png"
    _png(image)
    result = convert_png(image, (0.001, 0.001, 0.001), ([255, 0, 0],), zcells=1)
    original_geometry = result.geometry_file.read_bytes()

    with pytest.raises(ValueError, match="keys which do not match"):
        convert_png(
            image,
            (0.001, 0.001, 0.001),
            ([255, 0, 0], [0, 255, 0]),
            zcells=1,
        )

    assert result.geometry_file.read_bytes() == original_geometry


def _capture_built_grid(monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


@pytest.mark.integration
@pytest.mark.parametrize("averaging", ("n", "y"))
def test_png_geometry_pair_round_trips_through_geometry_objects_read(tmp_path, monkeypatch, averaging):
    image = tmp_path / "layers.png"
    _png(image)
    result = convert_png(
        image,
        (0.001, 0.001, 0.001),
        ([255, 0, 0], [0, 255, 0]),
        zcells=2,
    )
    _complete_database(result.material_database_file)

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="PNG geometry import"))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.Domain(p1=(0.005, 0.004, 0.003)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0.001, 0.001, 0.0),
            geofile=str(result.geometry_file),
            material_database=result.material_database_id,
            averaging=averaging,
        )
    )
    captured = _capture_built_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "png_geometry_import",
        hide_progress_bars=True,
    )

    grid = captured["grid"]
    names_by_id = {material.numID: material.ID for material in grid.materials}
    imported = np.vectorize(names_by_id.__getitem__)(grid.solid[1:4, 1:3, 0:2])
    database = result.material_database_id
    red = f"rgb_255_0_0{{{database}}}"
    green = f"rgb_0_255_0{{{database}}}"
    expected_xy = np.asarray(
        [["free_space", red], [red, green], [green, "free_space"]],
        dtype=object,
    )
    np.testing.assert_array_equal(imported[:, :, 0], expected_xy)
    np.testing.assert_array_equal(imported[:, :, 1], expected_xy)
