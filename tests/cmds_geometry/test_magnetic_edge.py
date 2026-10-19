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

"""Tests for the #magnetic_edge command / MagneticEdge class - the magnetic
dual of #edge, added alongside the reversion of set_rigid_Hx/Hy/Hz to the
self-consistent single-position formula (see
tests/materials/test_pec_h_components.py for the underlying rigid-flag
mechanics tests).
"""
from pathlib import Path

import h5py
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


def _scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


def _scene_2d(mode, dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
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


@pytest.mark.parametrize(
    ("p1", "p2", "component", "cells"),
    [
        ((0.002, 0.001, INF), (0.004, 0.001, INF), "Hx", [(2, 1, 0), (3, 1, 0)]),
        ((0.001, 0.002, INF), (0.001, 0.004, INF), "Hy", [(1, 2, 0), (1, 3, 0)]),
    ],
)
def test_magnetic_edge_tm_runs_on_in_plane_axis(monkeypatch, tmp_path, p1, p2, component, cells):
    scene = _scene_2d("TM")
    scene.add(gprMax.MagneticEdge(p1=p1, p2=p2, material_id="pmc"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    pmc_numid = next(m.numID for m in grid.materials if m.ID == "pmc")
    idH = grid.IDlookup[component]
    for i, j, k in cells:
        assert grid.ID[idH, i, j, k] == pmc_numid


def test_magnetic_edge_te_runs_along_invariant_axis(monkeypatch, tmp_path):
    scene = _scene_2d("TE")
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    scene.add(gprMax.MagneticEdge(p1=(0.001, 0.001, INF), p2=(0.001, 0.001, INF), material_id="diel"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    diel_numid = next(m.numID for m in grid.materials if m.ID == "diel")
    idHz = grid.IDlookup["Hz"]
    assert grid.ID[idHz, 1, 1, 1] == diel_numid
    assert grid.ID[idHz, 1, 1, 0] != diel_numid


def test_magnetic_edge_rejected_in_tm_along_invariant_axis(monkeypatch, tmp_path):
    scene = _scene_2d("TM")
    scene.add(gprMax.MagneticEdge(p1=(0.001, 0.001, 0.0), p2=(0.001, 0.001, 0.001), material_id="pmc"))

    with pytest.raises(ValueError, match="non-invariant axis"):
        gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)


def test_magnetic_edge_rejected_in_te_along_in_plane_axis(monkeypatch, tmp_path):
    scene = _scene_2d("TE")
    scene.add(gprMax.MagneticEdge(p1=(0.001, 0.001, 0.001), p2=(0.003, 0.001, 0.001), material_id="pmc"))

    with pytest.raises(ValueError, match="invariant axis"):
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


def _pmc_sheet_scene(mode, wall):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.16, 0.24, INF)))
    scene.add(gprMax.TimeWindow(time=1.6e-9))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.DiscretePlaneWaveAngles(
            p1=(0.020, 0.030, INF),
            p2=(0.140, 0.210, INF),
            theta=90,
            phi=0,
            psi=90 if mode == "TM" else 0,
            waveform_id="w",
        )
    )
    field = "Ez" if mode == "TM" else "Ey"
    scene.add(gprMax.Rx(p1=(0.035, 0.120, INF), outputs=[field], id="front"))
    scene.add(gprMax.Rx(p1=(0.135, 0.120, INF), outputs=[field], id="back"))
    if wall:
        if mode == "TM":
            # Freezes the Hy component tangential to the x-normal sheet.
            scene.add(
                gprMax.MagneticEdge(
                    p1=(0.130, 0.010, INF), p2=(0.130, 0.230, INF), material_id="pmc"
                )
            )
        else:
            # In TE one invariant-axis edge per cross-line cell spans the 2-cell thickness.
            for j in range(10, 230):
                scene.add(
                    gprMax.MagneticEdge(
                        p1=(0.130, j * 1e-3, INF),
                        p2=(0.130, j * 1e-3, INF),
                        material_id="pmc",
                    )
                )
    return scene


def _run_pmc_sheet(mode, wall, tmp_path, label):
    outputfile = tmp_path / label
    gprMax.run(
        scenes=[_pmc_sheet_scene(mode, wall)],
        n=1,
        outputfile=outputfile,
        cpu_precision="double",
        hide_progress_bars=True,
    )
    return outputfile.with_suffix(".h5")


@pytest.mark.parametrize("mode", ("TM", "TE"))
def test_pmc_sheet_reflection_is_unit_and_transmission_zero(tmp_path, mode):
    """Normal incidence on a PMC sheet: E reflects with coefficient +1 and the gated field behind is zero."""
    field = "Ez" if mode == "TM" else "Ey"
    with h5py.File(_run_pmc_sheet(mode, False, tmp_path, f"{mode}_free")) as f:
        dt = f.attrs["dt"]
        incident_front = f[f"rxs/rx1/{field}"][:]
        incident_back = f[f"rxs/rx2/{field}"][:]
    with h5py.File(_run_pmc_sheet(mode, True, tmp_path, f"{mode}_wall")) as f:
        total_front = f[f"rxs/rx1/{field}"][:]
        total_back = f[f"rxs/rx2/{field}"][:]

    times = np.arange(len(incident_front)) * dt
    incident_window = (times > 0.10e-9) & (times < 0.45e-9)
    reflected_window = (times > 0.75e-9) & (times < 1.15e-9)
    transmitted_window = (times > 0.45e-9) & (times < 0.75e-9)

    incident = incident_front[incident_window]
    reflected = (total_front - incident_front)[reflected_window]
    coefficient = np.max(np.correlate(reflected, incident, mode="full"))
    coefficient /= np.max(np.correlate(incident, incident, mode="full"))
    assert coefficient == pytest.approx(1, abs=0.03)

    transmitted = np.max(np.abs(total_back[transmitted_window]))
    assert transmitted < 1e-4 * np.max(np.abs(incident_back[transmitted_window]))
