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

"""Tests for FDTDGrid.symmetry_boundary_edges - the build-time resolution of
which of the 12 domain edges need a per-iteration PMC update, and with which
fixed flags (gprMax.symmetry_boundaries.build_symmetry_boundary_edges).

An edge is included only if at least one of its two bordering faces is a
declared PMC symmetry boundary; an edge where neither face is PMC is
dropped entirely, so the per-iteration dispatcher never calls into it.
"""
import gprMax
import gprMax.model as model_mod


def _capture_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


def _edge_flags(grid):
    """Returns {cython_func_name: (a_pmc, b_pmc)} for easy assertions."""
    return {func.__name__: (a, b) for func, a, b, *_ in grid.symmetry_boundary_edges}


def test_no_symmetry_boundaries_gives_no_edges(monkeypatch, tmp_path):
    scene = _scene()

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    assert grid.symmetry_boundary_edges == []


def test_single_pmc_face_resolves_its_four_edges_single_sided(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    flags = _edge_flags(grid)
    assert len(flags) == 4
    assert flags["update_symmetry_boundary_electric_Ez_X0_Y0"] == (True, False)
    assert flags["update_symmetry_boundary_electric_Ez_X0_YMax"] == (True, False)
    assert flags["update_symmetry_boundary_electric_Ey_X0_Z0"] == (True, False)
    assert flags["update_symmetry_boundary_electric_Ey_X0_ZMax"] == (True, False)


def test_two_adjacent_pmc_faces_resolve_shared_edge_both_true(monkeypatch, tmp_path):
    """x0 and y0 share the Ez_X0_Y0 edge - both flags should be True there,
    while their other (non-shared) edges stay single-sided."""
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    scene.add(gprMax.SymmetryBoundary(face="y0", type="pmc"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    flags = _edge_flags(grid)
    # x0 touches 4 edges, y0 touches 4 edges, one (X0_Y0) is shared -> 7 total.
    assert len(flags) == 7
    assert flags["update_symmetry_boundary_electric_Ez_X0_Y0"] == (True, True)
    assert flags["update_symmetry_boundary_electric_Ez_X0_YMax"] == (True, False)
    assert flags["update_symmetry_boundary_electric_Ez_XMax_Y0"] == (False, True)
    assert flags["update_symmetry_boundary_electric_Ey_X0_Z0"] == (True, False)
    assert flags["update_symmetry_boundary_electric_Ey_X0_ZMax"] == (True, False)
    assert flags["update_symmetry_boundary_electric_Ex_Y0_Z0"] == (True, False)
    assert flags["update_symmetry_boundary_electric_Ex_Y0_ZMax"] == (True, False)
    # Edges not touching either x0 or y0 must not appear at all.
    assert "update_symmetry_boundary_electric_Ez_XMax_YMax" not in flags


def test_pec_face_contributes_no_edges(monkeypatch, tmp_path):
    """A pec-type symmetry boundary needs no per-iteration edge update at
    all - only pmc faces populate symmetry_boundary_edges."""
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pec"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    assert grid.symmetry_boundary_edges == []


def test_all_six_faces_pmc_resolves_all_twelve_edges_both_true(monkeypatch, tmp_path):
    scene = _scene()
    for face in ("x0", "y0", "z0", "xmax", "ymax", "zmax"):
        scene.add(gprMax.SymmetryBoundary(face=face, type="pmc"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    flags = _edge_flags(grid)
    assert len(flags) == 12
    assert all(a and b for a, b in flags.values())
