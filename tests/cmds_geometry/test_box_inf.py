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

"""End-to-end tests for `inf` coordinates in #box / Box, covering the
range-endpoint resolution rules from gprMax/user_inputs.py's
resolve_inf_point(): purely positional (lower->0, upper->axis extent),
correctly spanning the full invariant-axis thickness in TM/TE 2D mode
with no special-casing, unaffected by sign, and rejected outright in a
3D model (see gprMax/user_inputs.py's resolve_inf_point() docstring for
why 3D usage was removed - it isn't needed there and can't be resolved
correctly for subgrid-scoped objects).
"""
import tempfile
from pathlib import Path

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


def _scene_with_diel(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    return scene


def test_3d_box_with_inf_is_rejected(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.Box(p1=(0, 0, INF), p2=(0.005, 0.005, INF), material_id="diel"))

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, "box_3d", scene)


def test_tm_box_spans_full_invariant_thickness(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Box(p1=(0, 0, INF), p2=(0.005, 0.005, INF), material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "box_tm", scene)
    assert grid.solid[0, 0, :].tolist() == [3]


def test_te_box_spans_full_invariant_thickness(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Box(p1=(0, 0, INF), p2=(0.005, 0.005, INF), material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "box_te", scene)
    assert grid.solid[0, 0, :].tolist() == [3, 3]


def test_box_inf_sign_is_irrelevant_for_range_endpoints(monkeypatch, tmp_path):
    """Mismatched sign (+inf in the lower slot, -inf in the upper slot)
    still resolves purely by position - sign is ignored for range
    endpoints. Uses the y axis (non-invariant in this TEz model) so the
    ordinary positional rule applies, not the invariant-axis override;
    z is pinned to a real 1-cell range covering just the interior
    reference layer (not inf) so only the y-axis sign behaviour is
    under test here."""
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(
        gprMax.Box(p1=(0, INF, 0.001), p2=(0.005, -INF, 0.002), material_id="diel")
    )

    grid = _run(monkeypatch, tmp_path, "box_signed", scene)
    assert grid.solid[0, :, 1].tolist() == [3] * 10
