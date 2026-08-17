"""Semantic geometry-tag registry and end-to-end voxel semantics."""

import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod
from gprMax.geometry_tags import GeometryTagRegistry, validate_geometry_tag


def _capture_grid(monkeypatch):
    captured = {}
    original = model_mod.Model.build

    def build(self):
        original(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", build)
    return captured


def _base_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


def _build(scene, tmp_path, monkeypatch, name="tags"):
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / name,
        hide_progress_bars=True,
    )
    return captured["grid"]


@pytest.mark.parametrize("tag", ["", "has space", "slash/tag", "untagged", 7])
def test_invalid_or_reserved_tags_are_rejected(tag):
    with pytest.raises((TypeError, ValueError)):
        validate_geometry_tag(tag)


def test_registry_reuses_ids_and_selects_smallest_dtype():
    registry = GeometryTagRegistry()
    assert registry.register("housing") == 1
    assert registry.register("housing") == 1
    registry.register_many(f"part_{i}" for i in range(1, 255))
    registry.freeze()
    assert registry.dtype == np.dtype(np.uint8)

    registry = GeometryTagRegistry()
    registry.register_many(f"part_{i}" for i in range(256))
    registry.freeze()
    assert registry.dtype == np.dtype(np.uint16)

    registry = GeometryTagRegistry()
    registry.register_many(f"part_{i}" for i in range(65536))
    registry.freeze()
    assert registry.dtype == np.dtype(np.uint32)


def test_no_tags_means_no_registry_or_map_allocation(tmp_path, monkeypatch):
    scene = _base_scene()
    scene.add(gprMax.Box(p1=(0.002, 0.002, 0.002), p2=(0.008, 0.008, 0.008), material_id="pec"))
    grid = _build(scene, tmp_path, monkeypatch)
    assert grid.geometry_tag_registry is None
    assert grid.geometry_tag_map is None


def test_final_voxels_follow_tag_overwrite_and_clear_semantics(tmp_path, monkeypatch):
    scene = _base_scene()
    scene.add(
        gprMax.Box(
            p1=(0.001, 0.001, 0.001),
            p2=(0.009, 0.009, 0.009),
            material_id="pec",
            tag="outer",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0.003, 0.003, 0.003),
            p2=(0.007, 0.007, 0.007),
            material_id="free_space",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0.004, 0.004, 0.004),
            p2=(0.006, 0.006, 0.006),
            material_id="free_space",
            tag="void_marker",
        )
    )
    grid = _build(scene, tmp_path, monkeypatch)
    tags = grid.geometry_tag_map.data
    outer = grid.geometry_tag_registry.id_for("outer")
    void = grid.geometry_tag_registry.id_for("void_marker")

    assert np.all(tags[1:3, 1:3, 1:3] == outer)
    assert np.all(tags[3:4, 3:4, 3:4] == 0)
    assert np.all(tags[4:6, 4:6, 4:6] == void)


def test_same_tag_on_multiple_primitives_uses_one_id(tmp_path, monkeypatch):
    scene = _base_scene()
    scene.add(
        gprMax.Box(
            p1=(0.001, 0.001, 0.001),
            p2=(0.003, 0.003, 0.003),
            material_id="pec",
            tag="metal",
        )
    )
    scene.add(gprMax.Sphere(p1=(0.007, 0.007, 0.007), r=0.001, material_id="pec", tag="metal"))
    grid = _build(scene, tmp_path, monkeypatch)
    assert grid.geometry_tag_registry.names == ("untagged", "metal")
    assert set(np.unique(grid.geometry_tag_map.data)) == {0, 1}


def test_smoothing_does_not_change_cell_centred_tag_membership(tmp_path, monkeypatch):
    scene = _base_scene()
    scene.add(
        gprMax.Box(
            p1=(0.001, 0.001, 0.001),
            p2=(0.004, 0.004, 0.004),
            material_id="free_space",
            averaging="n",
            tag="unsmoothed",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0.005, 0.005, 0.005),
            p2=(0.008, 0.008, 0.008),
            material_id="free_space",
            averaging="y",
            tag="smoothed",
        )
    )
    grid = _build(scene, tmp_path, monkeypatch)
    tags = grid.geometry_tag_map.data
    assert np.count_nonzero(tags == grid.geometry_tag_registry.id_for("unsmoothed")) == 27
    assert np.count_nonzero(tags == grid.geometry_tag_registry.id_for("smoothed")) == 27


def test_clipped_tagged_primitive_writes_only_valid_domain_cells(tmp_path, monkeypatch):
    scene = _base_scene()
    scene.add(
        gprMax.Sphere(
            p1=(0.0, 0.005, 0.005),
            r=0.003,
            material_id="pec",
            tag="clipped_sphere",
        )
    )
    grid = _build(scene, tmp_path, monkeypatch)
    tags = grid.geometry_tag_map.data
    assert tags.shape == (10, 10, 10)
    assert np.count_nonzero(tags) > 0
    assert np.count_nonzero(tags[:, :2, :2]) == 0


def test_geometry_tag_map_is_preserved_across_geometry_fixed_runs(tmp_path, monkeypatch):
    scene = _base_scene()
    scene.add(
        gprMax.Box(
            p1=(0.002, 0.002, 0.002),
            p2=(0.008, 0.008, 0.008),
            material_id="pec",
            tag="anatomy",
        )
    )

    observed = []
    original = model_mod.Model.build

    def build(self):
        original(self)
        observed.append((id(self.G.geometry_tag_map), self.G.geometry_tag_map.data.copy()))

    monkeypatch.setattr(model_mod.Model, "build", build)
    gprMax.run(
        scenes=[scene],
        n=2,
        geometry_fixed=True,
        geometry_only=True,
        outputfile=tmp_path / "geometry_fixed_tags",
        hide_progress_bars=True,
    )

    assert len(observed) == 2
    assert observed[0][0] == observed[1][0]
    assert np.array_equal(observed[0][1], observed[1][1])
