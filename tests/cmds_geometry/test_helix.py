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

"""End-to-end scene tests for ``Helix``.

These run the real ``gprMax.run`` pipeline (discretisation, materials,
averaging, tags, rasterisation) and inspect the captured grid; the Cython
builder is tested directly in ``tests/geometry_primitives/test_helix_builder.py``.
"""

import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod
from gprMax.user_objects.cmds_geometry import helix as helix_module

DL = 2e-3
P1 = (0.012, 0.012, 0.004)
P2 = (0.012, 0.012, 0.020)


def _capture_grid(monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _scene(**overrides):
    kwargs = dict(
        p1=P1,
        p2=P2,
        coil_radius=6e-3,
        pitch=4e-3,
        turns=1.0,
        wire_radius=2e-3,
        handedness="r",
        material_id="diel",
    )
    kwargs.update(overrides)
    if "material_ids" in kwargs:
        kwargs.pop("material_id")

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=(0.024, 0.024, 0.024)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    scene.add(gprMax.Helix(**kwargs))
    return scene


def _review_scene(wire_radius, discretisation=(1e-3, 1e-3, 1e-3), material_id="pec"):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=discretisation))
    scene.add(gprMax.Domain(p1=(0.064, 0.064, 0.064)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    if material_id != "pec":
        scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id=material_id))
    scene.add(
        gprMax.Helix(
            p1=(0.032, 0.032, 0.008),
            p2=(0.032, 0.032, 0.048),
            coil_radius=0.008,
            pitch=0.006,
            turns=2.5,
            wire_radius=wire_radius,
            handedness="r",
            material_id=material_id,
        )
    )
    return scene


def _run(monkeypatch, tmp_path, label, scene):
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )
    return captured["grid"]


def _occupied_cells(grid):
    free_space = next(material for material in grid.materials if material.ID == "free_space")
    return grid.solid != free_space.numID


def test_helix_writes_material_voxels(monkeypatch, tmp_path):
    grid = _run(monkeypatch, tmp_path, "helix", _scene())

    diel = next(material for material in grid.materials if material.ID == "diel")
    assert np.count_nonzero(grid.solid == diel.numID) > 0
    assert np.count_nonzero(_occupied_cells(grid)) > 0


def test_pec_helix_sets_rigid_and_id_edges(monkeypatch, tmp_path):
    grid = _run(monkeypatch, tmp_path, "pec_helix", _scene(material_id="pec"))

    assert grid.rigidE.any()
    assert np.count_nonzero(grid.ID) > 0


def test_axis_reference_distance_does_not_change_geometry(monkeypatch, tmp_path):
    near = _run(monkeypatch, tmp_path, "near", _scene(p2=(0.012, 0.012, 0.016)))
    far = _run(monkeypatch, tmp_path, "far", _scene(p2=(0.012, 0.012, 0.060)))

    np.testing.assert_array_equal(near.solid, far.solid)


@pytest.mark.parametrize("handedness", ("R", "L"))
def test_handedness_is_case_insensitive(monkeypatch, tmp_path, handedness):
    upper = _run(monkeypatch, tmp_path, f"handed_{handedness}", _scene(handedness=handedness))
    lower = _run(
        monkeypatch, tmp_path, f"handed_{handedness.lower()}", _scene(handedness=handedness.lower())
    )

    assert np.count_nonzero(_occupied_cells(upper)) > 0
    np.testing.assert_array_equal(upper.solid, lower.solid)


def test_zero_pitch_builds_closed_ring(monkeypatch, tmp_path):
    grid = _run(monkeypatch, tmp_path, "ring", _scene(pitch=0.0))

    occupied = np.argwhere(_occupied_cells(grid))
    assert len(occupied) > 0
    z_cells = occupied[:, 2]
    assert z_cells.min() >= 0
    assert z_cells.max() <= int((P1[2] + 2e-3) / DL)


def test_directional_materials(monkeypatch, tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=(0.024, 0.024, 0.024)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    for material_id in ("mat_x", "mat_y", "mat_z"):
        scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id=material_id))
    scene.add(gprMax.Helix(**dict(
        p1=P1,
        p2=P2,
        coil_radius=6e-3,
        pitch=4e-3,
        turns=1.0,
        wire_radius=2e-3,
        handedness="r",
        material_ids=("mat_x", "mat_y", "mat_z"),
    )))

    grid = _run(monkeypatch, tmp_path, "directional", scene)

    directional = [material for material in grid.materials if material.directional_materials is not None]
    assert len(directional) == 1
    assert np.count_nonzero(grid.solid == directional[0].numID) > 0


def test_tag_is_written_to_helix_cells(monkeypatch, tmp_path):
    grid = _run(monkeypatch, tmp_path, "tagged", _scene(tag="wire"))

    assert grid.geometry_tag_map is not None
    tags = grid.geometry_tag_map.data
    assert np.count_nonzero(tags) == np.count_nonzero(_occupied_cells(grid))
    assert np.count_nonzero(tags) > 0


@pytest.mark.parametrize("mode", ("TM", "TE"))
def test_rejected_in_2d_mode(monkeypatch, tmp_path, mode):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=(0.024, 0.024, float("inf"))))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Helix(p1=P1, p2=P2, coil_radius=6e-3, pitch=4e-3, turns=1.0,
                           wire_radius=2e-3, handedness="r", material_id="pec"))

    with pytest.raises(ValueError, match="2D mode"):
        _run(monkeypatch, tmp_path, f"2d_{mode}", scene)


def test_underresolved_wire_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="wire diameter"):
        _run(monkeypatch, tmp_path, "thin", _scene(wire_radius=0.5e-3))


@pytest.mark.parametrize("wire_radius", (0.0005, 0.0006))
def test_disconnected_pec_helix_rejected(monkeypatch, tmp_path, wire_radius):
    with pytest.raises(ValueError, match="disconnected PEC Yee-edge components"):
        _run(monkeypatch, tmp_path, f"disconnected_{wire_radius}", _review_scene(wire_radius))


def test_anisotropic_disconnected_pec_helix_rejected(monkeypatch, tmp_path):
    scene = _review_scene(0.0008, discretisation=(1e-3, 2e-3, 2e-3))
    with pytest.raises(ValueError, match="under-resolved"):
        _run(monkeypatch, tmp_path, "anisotropic", scene)


def test_connected_pec_wire_below_smallest_cell_accepted(monkeypatch, tmp_path):
    grid = _run(monkeypatch, tmp_path, "small_pec", _review_scene(0.0009))

    assert np.count_nonzero(_occupied_cells(grid)) > 0


def test_anisotropic_non_pec_wire_warns(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(helix_module.logger, "warning", warnings.append)
    scene = _review_scene(0.00075, discretisation=(1e-3, 2e-3, 2e-3), material_id="diel")
    _run(monkeypatch, tmp_path, "anisotropic_warning", scene)

    assert any("largest grid cell size" in message for message in warnings)


def test_directional_pec_wire_warns_connectivity_unchecked(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(helix_module.logger, "warning", warnings.append)
    scene = _scene(material_ids=("pec", "pec", "diel"))
    _run(monkeypatch, tmp_path, "directional_pec", scene)

    assert any("only some material directions are PEC" in message for message in warnings)


def test_pec_tag_is_written_to_helix_cells(monkeypatch, tmp_path):
    grid = _run(monkeypatch, tmp_path, "pec_tagged", _scene(material_id="pec", tag="wire"))

    tags = grid.geometry_tag_map.data
    assert np.count_nonzero(tags) == np.count_nonzero(_occupied_cells(grid))
    assert np.count_nonzero(tags) > 0


def test_invalid_handedness_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="handedness"):
        _run(monkeypatch, tmp_path, "spin", _scene(handedness="x"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"coil_radius": 0.0},
        {"turns": 0.0},
        {"wire_radius": 0.0},
        {"pitch": -1e-3},
        {"p1": P2, "p2": P2},
    ],
    ids=["zero-coil", "zero-turns", "zero-wire", "negative-pitch", "coincident-points"],
)
def test_invalid_geometry_rejected(monkeypatch, tmp_path, overrides):
    with pytest.raises(ValueError):
        _run(monkeypatch, tmp_path, "invalid", _scene(**overrides))
