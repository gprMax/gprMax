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

"""Tests for tex()/tey()/tez() (gprMax/grid/fdtd_grid.py) - the PEC/PMC
boundary forcing that makes 2D TE mode (a 2-cell-thick slice of the 3D
grid, see #domain_mode) physically equivalent to the existing 1-cell TM
mode.

For an invariant x axis (TEx, analogous derivation for TEy/TEz):
  - Ex (own-axis E) and Hy, Hz (tangential H) are forced to pec/pmc at
    both their live x-positions (i=0,1) - this is load-bearing on both
    the CPU (fields_updates_normal.pyx) and GPU (knl_fields_updates.py)
    backends, confirmed by tracing both kernels' loop/gate structure:
    without it they would pick up genuine non-zero curl-driven values.
  - Ey, Ez (tangential E) and Hx (own-axis H) are the surviving/
    propagating components. They are additionally forced to pec/pmc at
    the two outer wall planes (i=0 and i=nx) only - defence-in-depth,
    not strictly required today (both backends already gate these
    components to a strictly-interior x range for this axis, so they
    never pick up a non-zero value at the walls on their own) - but kept
    so a future change to those loops can't silently break the
    reduction. The interior plane (i=1, the genuinely propagating field)
    is deliberately left untouched.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod

INF = float("inf")

PEC = 0
PMC = 1

# (own E component, tangential H components, tangential E components, own H component)
AXIS_COMPONENTS = {
    "x": (0, (4, 5), (1, 2), 3),
    "y": (1, (3, 5), (0, 2), 4),
    "z": (2, (3, 4), (0, 1), 5),
}


def _capture_id(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["ID"] = self.G.ID.copy()

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _run_te(monkeypatch, tmp_path, axis):
    domain = [0.01, 0.01, 0.01]
    domain["xyz".index(axis)] = INF

    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=tuple(domain)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))

    captured = _capture_id(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"te_{axis}",
        hide_progress_bars=True,
    )
    return captured["ID"]


def _plane(ID, axis, comp, index):
    sl = [slice(None)] * 4
    sl[0] = comp
    sl["xyz".index(axis) + 1] = index
    return ID[tuple(sl)]


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_own_axis_e_forced_to_pec_at_both_live_positions(monkeypatch, tmp_path, axis):
    own_e, _, _, _ = AXIS_COMPONENTS[axis]
    ID = _run_te(monkeypatch, tmp_path, axis)
    assert (_plane(ID, axis, own_e, 0) == PEC).all()
    assert (_plane(ID, axis, own_e, 1) == PEC).all()


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_tangential_h_forced_to_pmc_at_both_live_positions(monkeypatch, tmp_path, axis):
    _, tangential_h, _, _ = AXIS_COMPONENTS[axis]
    ID = _run_te(monkeypatch, tmp_path, axis)
    for h in tangential_h:
        assert (_plane(ID, axis, h, 0) == PMC).all()
        assert (_plane(ID, axis, h, 1) == PMC).all()


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_surviving_e_forced_to_pec_at_outer_walls_only(monkeypatch, tmp_path, axis):
    _, _, tangential_e, _ = AXIS_COMPONENTS[axis]
    ID = _run_te(monkeypatch, tmp_path, axis)
    for e in tangential_e:
        assert (_plane(ID, axis, e, 0) == PEC).all()
        assert (_plane(ID, axis, e, 2) == PEC).all()
        # interior plane must be untouched (still free_space, not forced)
        assert not (_plane(ID, axis, e, 1) == PEC).all()


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_surviving_h_forced_to_pmc_at_outer_walls_only(monkeypatch, tmp_path, axis):
    _, _, _, own_h = AXIS_COMPONENTS[axis]
    ID = _run_te(monkeypatch, tmp_path, axis)
    assert (_plane(ID, axis, own_h, 0) == PMC).all()
    assert (_plane(ID, axis, own_h, 2) == PMC).all()
    assert not (_plane(ID, axis, own_h, 1) == PMC).all()


def test_te_solve_runs_without_nan_and_respects_forcing_over_time():
    """End-to-end: run real time-stepping (not geometry_only) with a
    source and confirm the forcing holds throughout, and that the
    interior layer carries a genuine non-zero propagating field.
    """
    import gprMax.model as model_mod

    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    model_mod.Model.build = patched_build
    try:
        dl = 1e-3
        scene = gprMax.Scene()
        scene.add(gprMax.DomainMode(mode="TE"))
        scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
        scene.add(gprMax.Domain(p1=(INF, 0.02, 0.02)))
        scene.add(gprMax.PMLThickness(thickness=0))
        scene.add(gprMax.TimeWindow(time=20e-12))
        scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="mypulse"))
        scene.add(
            gprMax.HertzianDipole(
                polarisation="y", p1=(0.001, 0.01, 0.01), waveform_id="mypulse"
            )
        )

        with tempfile.TemporaryDirectory() as td:
            gprMax.run(
                scenes=[scene],
                n=1,
                outputfile=Path(td) / "te_solve",
                hide_progress_bars=True,
            )
    finally:
        model_mod.Model.build = orig_build

    grid = captured["grid"]

    assert not np.isnan(grid.Ey).any()
    assert np.all(grid.Ex == 0)
    assert np.all(grid.Hy == 0)
    assert np.all(grid.Hz == 0)
    assert np.all(grid.Ey[0, :, :] == 0)
    assert np.all(grid.Ey[2, :, :] == 0)
    assert np.all(grid.Ez[0, :, :] == 0)
    assert np.all(grid.Ez[2, :, :] == 0)
    assert np.all(grid.Hx[0, :, :] == 0)
    assert np.all(grid.Hx[2, :, :] == 0)
    assert np.any(grid.Ey[1, :, :] != 0)


def test_te_solve_with_full_time_window_runs_dispersion_analysis_without_crashing():
    """Regression guard: FDTDGrid._dispersion_analysis() (fdtd_grid.py)
    computed its maximum-spatial-step `delta` via `self.nx==1`/`ny==1`/
    `nz==1` checks - none of which match TE's 2-cell invariant axis,
    leaving `delta` unbound and raising UnboundLocalError. This only
    fires once the waveform is long enough relative to the time window
    to NOT be flagged as truncated (short/degenerate windows skip the
    delta computation entirely, which is why the tex()/tey()/tez() tests
    above, with a deliberately tiny 20ps window, never exercised this
    path) - so this test uses a full, realistic time window.
    """
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, INF)))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="mypulse"))
    scene.add(
        gprMax.HertzianDipole(
            polarisation="y", p1=(0.025, 0.025, 0.001), waveform_id="mypulse"
        )
    )

    with tempfile.TemporaryDirectory() as td:
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=Path(td) / "te_dispersion",
            hide_progress_bars=True,
        )
