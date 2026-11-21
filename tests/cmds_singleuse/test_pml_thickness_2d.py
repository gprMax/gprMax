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

"""End-to-end tests for #pml_cells in 2D models.

In a 2D model the invariant axis carries no PML: Domain.build() zeroes it
for the default thickness, and a 1-cell (TM) or 2-cell (TE) axis cannot
host opposing slabs anyway. PMLThickness used to expand the single-value
form onto all six faces and validate that raw request, so the documented
`#pml_cells: N` syntax raised "too many cells" in every 2D model, and the
only escape was the six-value form with explicit zeros. The command now
validates and stores the thickness the faces will actually have, so the
single-value form applies to the transverse faces and requested
invariant-axis cells are dropped (with a message).
"""
import pytest

import gprMax
import gprMax.config as config
import gprMax.model as model_mod

INF = float("inf")


def _capture_model_config(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["mode"] = config.get_model_config().mode
        captured["dims"] = (int(self.G.nx), int(self.G.ny), int(self.G.nz))
        captured["pmls"] = dict(self.G.pmls["thickness"])

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _run(monkeypatch, tmp_path, label, scene):
    captured = _capture_model_config(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )
    return captured


def _base_scene(mode=None, domain=(0.03, 0.03, INF), dl=1e-3):
    scene = gprMax.Scene()
    if mode is not None:
        scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


def test_single_value_pml_cells_on_2d_tmz_model(monkeypatch, tmp_path):
    scene = _base_scene(mode="TM")
    scene.add(gprMax.PMLThickness(thickness=10))
    captured = _run(monkeypatch, tmp_path, "pml_2d_tmz", scene)

    assert captured["mode"] == "2D TMz"
    assert captured["dims"] == (30, 30, 1)
    assert captured["pmls"]["z0"] == 0
    assert captured["pmls"]["zmax"] == 0
    assert captured["pmls"]["x0"] == 10
    assert captured["pmls"]["xmax"] == 10
    assert captured["pmls"]["y0"] == 10
    assert captured["pmls"]["ymax"] == 10


def test_single_value_pml_cells_on_2d_tez_model(monkeypatch, tmp_path):
    # Declared TE resolves the inf axis to 2 cells.
    scene = _base_scene(mode="TE")
    scene.add(gprMax.PMLThickness(thickness=10))
    captured = _run(monkeypatch, tmp_path, "pml_2d_tez", scene)

    assert captured["mode"] == "2D TEz"
    assert captured["dims"] == (30, 30, 2)
    assert captured["pmls"]["z0"] == 0
    assert captured["pmls"]["zmax"] == 0
    for face in ("x0", "xmax", "y0", "ymax"):
        assert captured["pmls"][face] == 10


def test_single_value_pml_cells_on_legacy_2d_model(monkeypatch, tmp_path):
    # Old-style 2D: no #domain_mode, a 1-cell z axis. The invariant PML
    # faces are zeroed and the single-value form applies to x/y.
    scene = _base_scene(domain=(0.03, 0.03, 0.001))
    scene.add(gprMax.PMLThickness(thickness=10))
    captured = _run(monkeypatch, tmp_path, "pml_2d_legacy", scene)

    assert captured["mode"] == "2D TMz"
    assert captured["dims"] == (30, 30, 1)
    assert captured["pmls"]["z0"] == 0
    assert captured["pmls"]["zmax"] == 0
    for face in ("x0", "xmax", "y0", "ymax"):
        assert captured["pmls"][face] == 10


def test_six_value_pml_cells_with_explicit_zeros_unchanged(monkeypatch, tmp_path):
    # The documented escape hatch keeps working, with the same result.
    scene = _base_scene(mode="TM")
    scene.add(gprMax.PMLThickness(thickness=(10, 10, 0, 10, 10, 0)))
    captured = _run(monkeypatch, tmp_path, "pml_2d_six_zero", scene)

    assert captured["pmls"]["z0"] == 0
    assert captured["pmls"]["zmax"] == 0
    for face in ("x0", "xmax", "y0", "ymax"):
        assert captured["pmls"][face] == 10


def test_six_value_requested_invariant_cells_dropped(monkeypatch, tmp_path):
    # Explicit invariant-axis cells cannot be honoured in a 2D model - they
    # are dropped with a message instead of silently overwriting the zeros
    # Domain.build wrote.
    scene = _base_scene(mode="TM")
    scene.add(gprMax.PMLThickness(thickness=(10, 10, 3, 10, 10, 3)))
    captured = _run(monkeypatch, tmp_path, "pml_2d_six_invariant", scene)

    assert captured["pmls"]["z0"] == 0
    assert captured["pmls"]["zmax"] == 0
    for face in ("x0", "xmax", "y0", "ymax"):
        assert captured["pmls"][face] == 10


def test_2d_transverse_overflow_still_rejected(monkeypatch, tmp_path):
    # Dropping the invariant pair must not weaken the check on the
    # transverse axes: 10+10 >= 15 is still too many cells.
    scene = _base_scene(mode="TM", domain=(0.015, 0.03, INF))
    scene.add(gprMax.PMLThickness(thickness=10))
    with pytest.raises(ValueError, match="too many cells"):
        _run(monkeypatch, tmp_path, "pml_2d_overflow", scene)
