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

"""Surface-impedance geometry survives #geometry_objects_write/read."""

import json
import logging

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.model import Model

DL = 1e-3
DOMAIN = 0.032
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@pytest.fixture(autouse=True)
def restore_package_logging():
    yield
    logger = logging.getLogger("gprMax")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def _scene(title, *, solve):
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=title))
    scene.add(gprMax.Domain(p1=(DOMAIN,) * 3))
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(iterations=800))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    if solve:
        scene.add(
            gprMax.HertzianDipole(
                polarisation="z", p1=(0.005, 0.016, 0.016), waveform_id="pulse"
            )
        )
        scene.add(gprMax.Rx(p1=(0.026, 0.016, 0.016)))
    return scene


def _add_surface_impedance(scene):
    scene.add(gprMax.SurfaceImpedance(id="wall", resistance=50.0))
    scene.add(
        gprMax.Box(p1=(0.012, 0.012, 0.012), p2=(0.020, 0.020, 0.020), material_id="wall")
    )


def _export(tmp_path):
    scene = _scene("sib-export", solve=False)
    _add_surface_impedance(scene)
    outfile = tmp_path / "sib_geometry"
    scene.add(
        gprMax.GeometryObjectsWrite(
            p1=(0.0, 0.0, 0.0), p2=(DOMAIN,) * 3, filename=str(outfile)
        )
    )
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "export_run",
        hide_progress_bars=True,
        cpu_precision="double",
    )
    return outfile.with_suffix(".h5"), tmp_path / "sib_geometry_materials.json"


def _capture_grid(monkeypatch):
    captured = {}
    original = Model.build

    def build(self):
        original(self)
        captured["grid"] = self.G

    monkeypatch.setattr(Model, "build", build)
    return captured


def _database_entries(database):
    document = json.loads(database.read_text(encoding="utf-8"))
    return document["materials"]


def _entry_with_original_id(entries, original_id):
    matches = [
        entry
        for entry in entries.values()
        if entry.get("metadata", {}).get("original_id") == original_id
    ]
    assert len(matches) == 1, original_id
    return matches[0]


def test_database_serialises_surface_impedance_and_private_roles(tmp_path):
    _, database = _export(tmp_path)
    entries = _database_entries(database)

    marker = _entry_with_original_id(entries, "__impedance_volume__wall")
    model = marker["metadata"]["surface_impedance_model"]
    assert model["ID"] == "wall"
    assert model["D"] == pytest.approx(50.0)
    assert np.asarray(model["A"]).size == 0

    hold = _entry_with_original_id(entries, "__impedance_surface_hold__")
    void = _entry_with_original_id(entries, "__impedance_volume_void__")
    assert hold["metadata"]["impedance_private_role"] == "surface-hold"
    assert void["metadata"]["impedance_private_role"] == "volume-void"


def test_read_restores_model_marker_and_private_materials(tmp_path, monkeypatch):
    geofile, database = _export(tmp_path)
    scene = _scene("sib-read", solve=False)
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0.0, 0.0, 0.0),
            geofile=str(geofile),
            material_database=database.stem,
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "read_run",
        hide_progress_bars=True,
        cpu_precision="double",
    )
    grid = captured["grid"]

    assert set(grid.surface_impedance_models) == {"wall"}
    assert grid.surface_impedance_models["wall"].D == pytest.approx(50.0)
    assert grid.impedance_surfaces is not None
    assert grid.impedance_surfaces.model_ids == ("wall",)

    marker_numid = next(
        numid for numid, model_id in grid.impedance_marker_models.items() if model_id == "wall"
    )
    marker = grid.materials[marker_numid]
    assert marker.surface_impedance_id == "wall"
    assert np.count_nonzero(grid.solid == marker_numid) == 8 * 8 * 8

    sentinels = {
        material.impedance_role: material.numID
        for material in grid.materials
        if hasattr(material, "impedance_role")
    }
    assert set(sentinels) == {"surface-hold", "volume-void"}
    assert np.count_nonzero(grid.ID == marker_numid) == 0


def test_read_scene_solves_identically_to_directly_built_model(tmp_path, monkeypatch):
    geofile, database = _export(tmp_path)
    captured = _capture_grid(monkeypatch)

    direct = _scene("sib-direct", solve=True)
    _add_surface_impedance(direct)
    gprMax.run(
        scenes=[direct],
        n=1,
        outputfile=tmp_path / "direct",
        hide_progress_bars=True,
        cpu_precision="double",
    )

    imported = _scene("sib-import", solve=True)
    imported.add(
        gprMax.GeometryObjectsRead(
            p1=(0.0, 0.0, 0.0),
            geofile=str(geofile),
            material_database=database.stem,
        )
    )
    gprMax.run(
        scenes=[imported],
        n=1,
        outputfile=tmp_path / "imported",
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(tmp_path / "direct.h5") as output:
        reference = np.stack([output[f"rxs/rx1/{field}"][:] for field in FIELDS])
    with h5py.File(tmp_path / "imported.h5") as output:
        actual = np.stack([output[f"rxs/rx1/{field}"][:] for field in FIELDS])
    assert np.linalg.norm(reference) > 0
    np.testing.assert_array_equal(actual, reference)
    assert captured["grid"].impedance_surfaces is not None


def test_conflicting_model_with_same_id_is_rejected(tmp_path):
    geofile, database = _export(tmp_path)
    scene = _scene("sib-conflict", solve=False)
    scene.add(gprMax.SurfaceImpedance(id="wall", resistance=100.0))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0.0, 0.0, 0.0),
            geofile=str(geofile),
            material_database=database.stem,
        )
    )

    with pytest.raises(ValueError, match="conflicts with the model already defined"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "conflict",
            hide_progress_bars=True,
            cpu_precision="double",
        )
