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

"""End-to-end tests for `inf` coordinates in single-point commands
(#hertzian_dipole / HertzianDipole, #rx / Rx), covering the sign-based
resolution rule from gprMax/user_inputs.py's resolve_inf_point() and its
mode-aware override on the invariant axis of an active 2D mode. Only
allowed in an active 2D mode - a 3D model rejects it outright, see
resolve_inf_point()'s docstring for why.
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


def test_3d_source_with_inf_is_rejected(monkeypatch, tmp_path):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="mypulse"))
    scene.add(
        gprMax.HertzianDipole(polarisation="z", p1=(-INF, 0.005, 0.005), waveform_id="mypulse")
    )

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, "src_3d", scene)


def test_3d_rx_with_inf_is_rejected(monkeypatch, tmp_path):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Rx(p1=(INF, 0.005, 0.005)))

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, "rx_3d", scene)


def test_te_source_and_rx_resolve_to_interior_layer_via_inf(monkeypatch, tmp_path):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(INF, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="mypulse"))
    scene.add(gprMax.HertzianDipole(polarisation="y", p1=(INF, 0.01, 0.01), waveform_id="mypulse"))
    scene.add(gprMax.Rx(p1=(-INF, 0.015, 0.015)))

    grid = _run(monkeypatch, tmp_path, "src_rx_te", scene)
    # both +inf (source) and -inf (rx) redirect to the interior layer
    # (index 1), not to the dead pec/pmc-forced walls at index 0 or 2.
    assert grid.hertziandipoles[0].coord[0] == 1
    assert grid.rxs[0].coord[0] == 1


def test_tm_source_and_rx_resolve_to_reference_layer_via_inf(monkeypatch, tmp_path):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(INF, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="mypulse"))
    scene.add(gprMax.HertzianDipole(polarisation="x", p1=(INF, 0.01, 0.01), waveform_id="mypulse"))
    scene.add(gprMax.Rx(p1=(-INF, 0.015, 0.015)))

    grid = _run(monkeypatch, tmp_path, "src_rx_tm", scene)
    assert grid.hertziandipoles[0].coord[0] == 0
    assert grid.rxs[0].coord[0] == 0


def test_te_source_non_invariant_axis_unaffected_by_mode(monkeypatch, tmp_path):
    """inf on a non-invariant axis in a 2D model still uses the plain
    sign-based 3D rule (snap to that axis's own domain edge)."""
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(INF, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="mypulse"))
    scene.add(
        gprMax.HertzianDipole(polarisation="y", p1=(0.001, INF, 0.01), waveform_id="mypulse")
    )

    grid = _run(monkeypatch, tmp_path, "src_te_other_axis", scene)
    assert grid.hertziandipoles[0].coord[1] == 20
