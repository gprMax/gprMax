"""Tests for the #magnetic_edge command / MagneticEdge class - the magnetic
dual of #edge, added alongside the reversion of set_rigid_Hx/Hy/Hz to the
self-consistent single-position formula (see
tests/materials/test_pec_h_components.py for the underlying rigid-flag
mechanics tests).
"""
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


def _scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


def test_magnetic_edge_sets_single_hx_edge(monkeypatch, tmp_path):
    # p1->p2 spans x=2mm to 4mm at 1mm resolution: 2 cells, i=2 and i=3
    # (matching exactly how #edge spans multiple cells for the same span).
    scene = _scene()
    scene.add(gprMax.MagneticEdge(p1=(0.002, 0.001, 0.001), p2=(0.004, 0.001, 0.001), material_id="pmc"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    pmc_numid = next(m.numID for m in grid.materials if m.ID == "pmc")
    idHx = grid.IDlookup["Hx"]
    assert grid.ID[idHx, 2, 1, 1] == pmc_numid
    assert grid.ID[idHx, 3, 1, 1] == pmc_numid
    # Doesn't leak to neighbouring Hx positions outside the specified span.
    assert grid.ID[idHx, 1, 1, 1] != pmc_numid
    assert grid.ID[idHx, 4, 1, 1] != pmc_numid


def test_magnetic_edge_multi_cell_wire(monkeypatch, tmp_path):
    """A magnetic edge spanning multiple cells sets every cell along the
    run axis, like #edge does for a multi-cell wire."""
    scene = _scene()
    scene.add(gprMax.MagneticEdge(p1=(0.001, 0.002, 0.002), p2=(0.006, 0.002, 0.002), material_id="pmc"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    pmc_numid = next(m.numID for m in grid.materials if m.ID == "pmc")
    idHx = grid.IDlookup["Hx"]
    for i in range(1, 6):
        assert grid.ID[idHx, i, 2, 2] == pmc_numid


def test_magnetic_edge_invalid_orientation_rejected(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.MagneticEdge(p1=(0.002, 0.002, 0.002), p2=(0.004, 0.004, 0.002), material_id="pmc"))

    with pytest.raises(ValueError):
        gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)


def test_magnetic_edge_missing_material_rejected(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.MagneticEdge(p1=(0.002, 0.001, 0.001), p2=(0.004, 0.001, 0.001), material_id="doesnotexist"))

    with pytest.raises(ValueError):
        gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)


def test_magnetic_edge_rejected_in_2d_mode(monkeypatch, tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.MagneticEdge(p1=(0.002, 0.001, INF), p2=(0.004, 0.001, INF), material_id="pmc"))

    with pytest.raises(ValueError, match="not yet supported in 2D mode"):
        gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)


def test_magnetic_edge_text_command_parses_correctly(monkeypatch, tmp_path: Path):
    """Exercises the hash_cmds_geometry.py text-parsing path specifically
    (the #magnetic_edge: line -> MagneticEdge construction), not just the
    Python Scene API used by the other tests in this file."""
    infile = tmp_path / "magnetic_edge.in"
    infile.write_text(
        "#title: magnetic edge text parsing\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.01 0.01 0.01\n"
        "#pml_cells: 0\n"
        "#time_window: 1e-12\n"
        "#magnetic_edge: 0.002 0.001 0.001 0.004 0.001 0.001 pmc\n"
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(inputfile=str(infile), n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    pmc_numid = next(m.numID for m in grid.materials if m.ID == "pmc")
    idHx = grid.IDlookup["Hx"]
    assert grid.ID[idHx, 2, 1, 1] == pmc_numid
