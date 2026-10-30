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

"""Layout and rasterisation tests for #rebar_mesh."""

import math

import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod

pytestmark = pytest.mark.integration

DL = 0.001
DOMAIN = (0.048, 0.048, 0.040)
MATERIALS = ("mat_x", "mat_y", "mat_z")
MESH_KWARGS = {
    "p1": (0.008, 0.008, 0.010),
    "p2": (0.040, 0.040, 0.030),
    "spacing": (0.008, 0.008),
    "diameter": 0.002,
    "cover": 0.002,
    "layer_spacing": 0.004,
    "material_id": "pec",
}


def _mesh(**overrides):
    kwargs = dict(MESH_KWARGS)
    if "material_ids" in overrides and "material_id" not in overrides:
        kwargs.pop("material_id")
    kwargs.update(overrides)
    return gprMax.RebarMesh(**kwargs)


def _base_scene(objects=(), *, mode=None, domain=DOMAIN):
    scene = gprMax.Scene()
    if mode is not None:
        scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.Material(er=5, se=0, mr=1, sm=0, id="concrete"))
    for index, material_id in enumerate(MATERIALS, start=2):
        scene.add(
            gprMax.Material(er=index, se=0, mr=1, sm=0, id=material_id)
        )
    for obj in objects:
        scene.add(obj)
    return scene


def _capture_grid(monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _run(scene, path, monkeypatch):
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=path,
        hide_progress_bars=True,
        log_level=30,
    )
    return captured["grid"]


def _cylinders(mesh, **overrides):
    kwargs = {
        "material_id": mesh.kwargs.get("material_id"),
        "material_ids": mesh.kwargs.get("material_ids"),
        "averaging": mesh.kwargs.get("averaging"),
        "tag": mesh.kwargs.get("tag"),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    kwargs.update(overrides)
    return [
        gprMax.Cylinder(
            p1=bar.p1,
            p2=bar.p2,
            r=bar.radius,
            **kwargs,
        )
        for bar in mesh.bars
    ]


def test_rebar_mesh_generates_two_orthogonal_layers():
    mesh = _mesh()

    assert mesh.hash == "#rebar_mesh"
    assert mesh.bar_count == 8
    assert mesh.bars[0].layer == "top"
    assert mesh.bars[-1].layer == "bottom"
    assert all(bar.axis == "x" for bar in mesh.bars if bar.layer == "top")
    assert all(bar.axis == "y" for bar in mesh.bars if bar.layer == "bottom")
    assert all(bar.radius == pytest.approx(0.001) for bar in mesh.bars)
    assert all(bar.p1[2] == pytest.approx(0.027) for bar in mesh.bars[:4])
    assert all(bar.p1[2] == pytest.approx(0.023) for bar in mesh.bars[4:])


def test_spacing_controls_bar_positions_along_each_axis():
    mesh = _mesh(spacing=(0.010, 0.006))
    top = [bar for bar in mesh.bars if bar.layer == "top"]
    bottom = [bar for bar in mesh.bars if bar.layer == "bottom"]

    assert len(top) == 5
    assert len(bottom) == 3
    assert tuple(bar.p1[1] for bar in top) == pytest.approx(
        (0.011, 0.017, 0.023, 0.029, 0.035)
    )
    assert tuple(bar.p1[0] for bar in bottom) == pytest.approx(
        (0.011, 0.021, 0.031)
    )


def test_zero_layer_spacing_places_both_layers_at_one_height():
    mesh = _mesh(layer_spacing=0)

    assert mesh.bar_count == 8
    assert all(bar.p1[2] == pytest.approx(0.027) for bar in mesh.bars)


def test_layer_spacing_defaults_to_zero():
    kwargs = dict(MESH_KWARGS)
    kwargs.pop("layer_spacing")

    mesh = gprMax.RebarMesh(**kwargs)

    assert all(bar.p1[2] == pytest.approx(0.027) for bar in mesh.bars)


def test_scalar_spacing_applies_to_both_axes():
    mesh = _mesh(spacing=0.008)

    assert mesh.bar_count == 8


def test_reversed_corners_produce_the_same_layout():
    mesh = _mesh()
    reversed_mesh = _mesh(p1=mesh.kwargs["p2"], p2=mesh.kwargs["p1"])

    assert reversed_mesh.bars == mesh.bars


def test_bar_lengths_include_clear_cover_at_both_ends():
    mesh = _mesh()

    for bar in mesh.bars:
        assert bar.length == pytest.approx(0.028)
    assert mesh.total_length == pytest.approx(8 * 0.028)
    assert math.isclose(mesh.total_length, sum(bar.length for bar in mesh.bars))


def test_single_bar_per_direction_is_allowed():
    mesh = _mesh(
        p1=(0.008, 0.008, 0.010),
        p2=(0.014, 0.014, 0.030),
        spacing=(0.010, 0.010),
    )

    assert mesh.bar_count == 2
    assert [bar.layer for bar in mesh.bars] == ["top", "bottom"]
    assert mesh.bars[0].length == pytest.approx(0.002)
    assert mesh.bars[1].length == pytest.approx(0.002)


def test_numpy_arrays_are_accepted():
    mesh = _mesh(
        p1=np.array((0.008, 0.008, 0.010)),
        p2=np.array((0.040, 0.040, 0.030)),
        spacing=np.array((0.008, 0.008)),
    )

    assert mesh.bar_count == 8


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"p1": None}, "three finite coordinates"),
        ({"p1": (0.008, 0.008)}, "three finite coordinates"),
        ({"p2": (0.040, 0.040, float("nan"))}, "three finite coordinates"),
        ({"spacing": 0}, "positive scalar"),
        ({"spacing": (0.008, 0.008, 0.008)}, "positive scalar"),
        ({"spacing": (0.001, 0.008)}, "bar spacing must be at least"),
        ({"diameter": 0}, "positive value"),
        ({"diameter": float("inf")}, "positive value"),
        ({"cover": -0.001}, "non-negative value"),
        ({"layer_spacing": -0.001}, "non-negative value"),
        ({"layer_spacing": 0.001}, "zero or at least one"),
        ({"p2": (0.040, 0.040, 0.011)}, "z span is too small"),
        ({"p2": (0.0105, 0.040, 0.030)}, "x span is too small"),
        ({"p2": (0.040, 0.0105, 0.030)}, "y span is too small"),
        ({"layer_spacing": 0.020}, "outside the reinforced volume"),
        ({"material_id": "pec", "material_ids": MATERIALS}, "not both"),
        ({"material_id": None, "material_ids": MATERIALS[:2]}, "three identifiers"),
        ({"material_id": None, "material_ids": 3}, "three identifiers"),
        ({"material_id": None}, "no materials"),
    ),
)
def test_layout_rejects_invalid_parameters(overrides, message):
    with pytest.raises(ValueError, match=message):
        _mesh(**overrides)


def test_pec_mesh_writes_material_and_tag(monkeypatch, tmp_path):
    mesh = _mesh(tag="bars")
    grid = _run(_base_scene((mesh,)), tmp_path / "pec_mesh", monkeypatch)
    pec = next(material for material in grid.materials if material.ID == "pec")

    occupied = grid.solid == pec.numID
    assert np.count_nonzero(occupied) > 0
    assert grid.geometry_tag_map is not None
    tag_id = grid.geometry_tag_map.id_for("bars")
    assert np.all(grid.geometry_tag_map.data[occupied] == tag_id)
    assert np.all(grid.geometry_tag_map.data[~occupied] == 0)


def test_untagged_mesh_does_not_allocate_tag_map(monkeypatch, tmp_path):
    grid = _run(_base_scene((_mesh(),)), tmp_path / "untagged_mesh", monkeypatch)

    assert grid.geometry_tag_map is None


def test_isotropic_mesh_matches_individual_cylinders(monkeypatch, tmp_path):
    mesh = _mesh(tag="bars")
    cylinders = _cylinders(mesh)
    mesh_grid = _run(
        _base_scene((mesh,)), tmp_path / "isotropic_mesh", monkeypatch
    )
    cylinder_grid = _run(
        _base_scene(cylinders), tmp_path / "isotropic_cylinders", monkeypatch
    )

    for array in ("solid", "ID", "rigidE", "rigidH"):
        assert np.array_equal(
            getattr(mesh_grid, array), getattr(cylinder_grid, array)
        )
    assert np.array_equal(
        mesh_grid.geometry_tag_map.data, cylinder_grid.geometry_tag_map.data
    )


def test_dielectric_averaging_matches_individual_cylinders(monkeypatch, tmp_path):
    mesh = _mesh(material_id="concrete", averaging="y", tag="bars")
    cylinders = _cylinders(mesh)
    mesh_grid = _run(
        _base_scene((mesh,)), tmp_path / "averaged_mesh", monkeypatch
    )
    cylinder_grid = _run(
        _base_scene(cylinders), tmp_path / "averaged_cylinders", monkeypatch
    )

    for array in ("solid", "ID", "rigidE", "rigidH"):
        assert np.array_equal(
            getattr(mesh_grid, array), getattr(cylinder_grid, array)
        )


def test_anisotropic_mesh_matches_individual_cylinders(monkeypatch, tmp_path):
    mesh = _mesh(material_ids=MATERIALS, tag="bars")
    cylinders = _cylinders(mesh)
    mesh_grid = _run(
        _base_scene((mesh,)), tmp_path / "anisotropic_mesh", monkeypatch
    )
    cylinder_grid = _run(
        _base_scene(cylinders), tmp_path / "anisotropic_cylinders", monkeypatch
    )

    for array in ("solid", "ID", "rigidE", "rigidH"):
        assert np.array_equal(
            getattr(mesh_grid, array), getattr(cylinder_grid, array)
        )
    records = [
        material
        for material in mesh_grid.materials
        if material.directional_materials is not None
    ]
    assert len(records) == 1
    assert records[0].directional_materials == tuple(
        next(material for material in mesh_grid.materials if material.ID == material_id)
        for material_id in MATERIALS
    )


def test_missing_material_is_rejected_during_build(monkeypatch, tmp_path):
    mesh = _mesh(material_id="missing")

    with pytest.raises(ValueError, match=r"material\(s\) \['missing'\] do not exist"):
        _run(_base_scene((mesh,)), tmp_path / "missing_material", monkeypatch)


def test_invalid_averaging_is_rejected_during_build(monkeypatch, tmp_path):
    mesh = _mesh(averaging="maybe")

    with pytest.raises(ValueError, match="Averaging should be"):
        _run(_base_scene((mesh,)), tmp_path / "invalid_averaging", monkeypatch)


def test_mesh_is_rejected_in_2d_mode(monkeypatch, tmp_path):
    scene = _base_scene(
        (_mesh(),),
        mode="TM",
        domain=(0.048, 0.048, float("inf")),
    )

    with pytest.raises(ValueError, match="cannot be used in 2D mode"):
        _run(scene, tmp_path / "rebar_2d", monkeypatch)


def test_underresolved_bars_are_rejected(monkeypatch, tmp_path):
    mesh = _mesh(diameter=0.0004)

    with pytest.raises(ValueError, match="does not occupy any Yee cells or faces"):
        _run(_base_scene((mesh,)), tmp_path / "underresolved", monkeypatch)
