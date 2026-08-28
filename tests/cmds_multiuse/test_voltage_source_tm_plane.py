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

"""Regression test: in 2D TM mode, a #voltage_source / VoltageSource must
be positioned at index 0 on the invariant axis, not just correctly
polarised.

Once polarisation matches the invariant axis (already enforced), the
own-axis E component (e.g. Ez for TMz) is only ever computed by the
interior update loop at index 0 - index 1 exists in the padded field
array but is never read or written by anything. `inf` already resolves
to index 0 correctly for TM (see resolve_inf_point's mode-aware
override), but an explicit numeric coordinate could still land on the
dead index 1 - this guard catches that regardless of how the coordinate
was specified.

This matters more for a hard (resistance=0) source than a resistive one:
a hard source directly overwrites the field value every iteration
regardless of material/ID, bypassing the material-lookup protection a
resistive source's VoltageSource.create_material() would otherwise get.
"""
import tempfile
from pathlib import Path

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


def _base_scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    return scene


def test_voltage_source_at_plane_0_via_inf_is_accepted_and_changes_id(monkeypatch, tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.VoltageSource(polarisation="z", p1=(0.01, 0.01, INF), resistance=50, waveform_id="w")
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene], n=1, outputfile=tmp_path / "vsrc_ok", hide_progress_bars=True
    )

    grid = captured["grid"]
    numid = grid.ID[2, 10, 10, 0]
    mat = next(m for m in grid.materials if m.numID == numid)
    assert "VoltageSource" in mat.ID
    assert mat.se != float("inf")  # a real, finite conductivity - genuinely live here


def test_voltage_source_at_plane_1_explicit_coordinate_is_rejected(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.VoltageSource(
            polarisation="z", p1=(0.01, 0.01, 0.001), resistance=50, waveform_id="w"
        )
    )

    with pytest.raises(ValueError, match="index 0"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "vsrc_bad",
            hide_progress_bars=True,
        )


def test_voltage_source_plane_check_only_applies_in_2d_tm(monkeypatch, tmp_path):
    """A 3D model must be completely unaffected by this guard."""
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.VoltageSource(
            polarisation="z", p1=(0.01, 0.01, 0.015), resistance=50, waveform_id="w"
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "vsrc_3d", hide_progress_bars=True)
    assert not np.isnan(captured["grid"].Ez).any()
