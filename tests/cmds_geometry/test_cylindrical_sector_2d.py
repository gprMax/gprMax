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

"""Regression tests: #cylindrical_sector in 2D TM/TE mode.

Design: only supported when the sector's own (`normal`) axis matches the
model's invariant axis - like #cylinder, this lets `extent1`/`extent2`
(the scalar thickness bounds along `normal`) span the full 1-cell (TM) or
2-cell (TE) thickness via `inf`, wall-to-wall. Unlike #cylinder's p1/p2
(full 3D points), a sector's cross-section coordinates (ctr1/ctr2) are two
scalars in the plane perpendicular to `normal` - a sector whose normal is a
*transverse* axis would need one of ctr1/ctr2 itself to span the invariant
thickness, which isn't supported, so that orientation is rejected outright
in 2D mode rather than silently producing an invariance-breaking shape.

`extent1`/`extent2` are resolved via a throwaway 3-tuple through the same
resolve_inf_point() used everywhere else (role="lower"/"upper"), which is a
no-op when there's no `inf` present - so the same call also gives a clean
"'inf' is only allowed... in 2D mode" rejection for 3D+inf, instead of a
raw crash, at no extra cost.
"""
import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod

INF = float("inf")


def _capture_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _base_scene(mode=None, domain=(0.02, 0.02, INF), dl=1e-3):
    scene = gprMax.Scene()
    if mode is not None:
        scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Material(er=5, se=0, mr=1, sm=0, id="diel"))
    return scene


def _sector(**overrides):
    kwargs = dict(
        normal="z", ctr1=0.01, ctr2=0.01, extent1=INF, extent2=INF, r=0.006,
        start=15, end=200, material_id="diel",
    )
    kwargs.update(overrides)
    return gprMax.CylindricalSector(**kwargs)


def test_tez_sector_spans_both_cells_and_is_invariant(monkeypatch, tmp_path):
    scene = _base_scene(mode="TE")
    scene.add(_sector())
    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "te", hide_progress_bars=True)
    grid = captured["grid"]
    diel = next(m for m in grid.materials if m.ID == "diel")
    assert grid.solid.shape == (20, 20, 2)
    assert np.array_equal(grid.solid[:, :, 0], grid.solid[:, :, 1])
    assert np.sum(grid.solid[:, :, 0] == diel.numID) > 0


def test_tmz_sector_spans_single_cell(monkeypatch, tmp_path):
    scene = _base_scene(mode="TM")
    scene.add(_sector())
    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "tm", hide_progress_bars=True)
    grid = captured["grid"]
    diel = next(m for m in grid.materials if m.ID == "diel")
    assert grid.solid.shape == (20, 20, 1)
    assert np.sum(grid.solid[:, :, 0] == diel.numID) > 0


def test_tm_and_te_produce_identical_cross_section(monkeypatch, tmp_path):
    scene_tm = _base_scene(mode="TM")
    scene_tm.add(_sector())
    captured_tm = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene_tm], n=1, geometry_only=True, outputfile=tmp_path / "cross_tm", hide_progress_bars=True)
    grid_tm = captured_tm["grid"]

    scene_te = _base_scene(mode="TE")
    scene_te.add(_sector())
    captured_te = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene_te], n=1, geometry_only=True, outputfile=tmp_path / "cross_te", hide_progress_bars=True)
    grid_te = captured_te["grid"]

    diel_tm = next(m for m in grid_tm.materials if m.ID == "diel")
    diel_te = next(m for m in grid_te.materials if m.ID == "diel")
    assert np.array_equal(
        grid_tm.solid[:, :, 0] == diel_tm.numID,
        grid_te.solid[:, :, 0] == diel_te.numID,
    )


def test_te_sector_normal_mismatch_rejected(tmp_path):
    scene = _base_scene(mode="TE")
    scene.add(_sector(normal="x", extent1=0.008, extent2=0.012))
    with pytest.raises(ValueError, match="normal axis must match the invariant axis"):
        gprMax.run(
            scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "mismatch", hide_progress_bars=True
        )


def test_3d_sector_with_inf_is_rejected(tmp_path):
    scene = _base_scene(mode=None, domain=(0.02, 0.02, 0.02))
    scene.add(_sector())
    with pytest.raises(ValueError, match="2D mode"):
        gprMax.run(
            scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "3d_inf", hide_progress_bars=True
        )


def test_3d_sector_without_inf_unaffected(monkeypatch, tmp_path):
    scene = _base_scene(mode=None, domain=(0.02, 0.02, 0.02))
    scene.add(_sector(normal="x", extent1=0.008, extent2=0.012))
    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "3d_plain", hide_progress_bars=True)
    grid = captured["grid"]
    diel = next(m for m in grid.materials if m.ID == "diel")
    assert np.sum(grid.solid == diel.numID) > 0
