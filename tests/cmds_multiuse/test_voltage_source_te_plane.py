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

"""Regression tests: in 2D TE mode, #voltage_source / VoltageSource
follows the same polarisation and positioning rule as HertzianDipole
(both are E-side sources): polarisation must be perpendicular to the
invariant axis, and position must be at index 1 (the interior layer) on
that axis. See tests/cmds_multiuse/test_dipole_te_polarisation_and_plane.py
for the equivalent HertzianDipole/MagneticDipole tests, and
test_voltage_source_tm_plane.py for the TM-mode VoltageSource tests this
mirrors.

Confirms G.ID genuinely changes at plane 1 with a real, finite
conductivity (not stuck at se=inf the way a source landing on a
tex()/tey()/tez()-forced position would be) - matching the equivalent TM
verification.
"""
import tempfile
from pathlib import Path

import gprMax
import gprMax.model as model_mod
import pytest

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
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    return scene


def test_voltage_source_perpendicular_polarisation_and_plane_1_via_inf_changes_id(
    monkeypatch, tmp_path
):
    scene = _base_scene()
    scene.add(
        gprMax.VoltageSource(polarisation="x", p1=(0.01, 0.01, INF), resistance=50, waveform_id="w")
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "vsrc_te_ok", hide_progress_bars=True)

    grid = captured["grid"]
    numid = grid.ID[0, 10, 10, 1]
    mat = next(m for m in grid.materials if m.numID == numid)
    assert "VoltageSource" in mat.ID
    assert mat.se != float("inf")


def test_voltage_source_invariant_axis_polarisation_is_rejected_in_te(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.VoltageSource(polarisation="z", p1=(0.01, 0.01, INF), resistance=50, waveform_id="w")
    )
    with pytest.raises(ValueError, match="polarisation"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "vsrc_te_bad_pol",
            hide_progress_bars=True,
        )


def test_voltage_source_correct_polarisation_wrong_plane_is_rejected_in_te(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.VoltageSource(polarisation="y", p1=(0.01, 0.01, 0.0), resistance=50, waveform_id="w")
    )
    with pytest.raises(ValueError, match="index 1"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "vsrc_te_bad_plane",
            hide_progress_bars=True,
        )
