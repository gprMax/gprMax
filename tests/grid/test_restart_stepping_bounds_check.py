"""Regression test for FDTDGrid._update_positions() (external review,
2026-07-21, "Restarted stepped scans bypass bounds checking"): the
one-time "won't be stepped outside the grid" check was gated by
`step_number == 0`, but step_number is the ABSOLUTE model index, so with
a restart (-i/i= > 1) the first model actually processed in this run has
step_number == model_start, never 0 - the check never ran at all on any
restarted run. A stepped source/receiver could then be silently
positioned outside the domain, later indexing field arrays out of
bounds (Cython source paths are less safely bounded than a plain Python
IndexError).

Confirmed empirically before fixing: a scan whose absolute step count
(n=30, step=1 cell) would run the receiver past a 20-cell domain by the
final model raised no error at all when restarted from model 25 with
the old code - it silently completed all 6 (restarted) models with the
receiver positioned outside the domain.

Fixed by changing the gate to `step_number == config.sim_config.model_start`
- but this alone would have regressed a DIFFERENT thing: the `else`
branch's repositioning (`item.coord = item.coordorigin + step_number *
step_size`) is itself already correct for restarts (it only depends on
the absolute step_number), so simply swapping which branch the first
processed model falls into would make a restarted run's first model
skip repositioning entirely (leaving it stuck at coordorigin, unstepped)
- confirmed this exact regression before catching it. Fixed by
decoupling the two: the bounds check runs only on the first model
processed (step_number == model_start), and the repositioning always
runs (harmless no-op for a non-restarted model 0, genuinely needed for
a restarted run's first model).
"""
import numpy as np
import pytest

import gprMax


def _make_scene(domain=(0.02, 0.02, 0.02), dl=1e-3, rx_step=(1e-3, 0.0, 0.0)):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.005, 0.005, 0.005), waveform_id="w"))
    scene.add(gprMax.Rx(p1=(0.005, 0.005, 0.005)))
    scene.add(gprMax.RxSteps(p1=rx_step))
    return scene


def _capture_rx_coords(monkeypatch):
    import gprMax.grid.fdtd_grid as fdtd_grid_mod

    captured = []
    orig = fdtd_grid_mod.FDTDGrid.update_sources_and_recievers

    def patched(self):
        orig(self)
        if self.rxs:
            captured.append(tuple(int(c) for c in self.rxs[0].coord))

    monkeypatch.setattr(fdtd_grid_mod.FDTDGrid, "update_sources_and_recievers", patched)
    return captured


def test_restarted_run_positions_match_the_equivalent_full_run(monkeypatch, tmp_path):
    """The repositioning itself must give IDENTICAL results whether a
    model was reached via a straight n=6 run or via a restart (i=4,
    n=3) picking up partway through the same conceptual 6-model scan."""
    captured = _capture_rx_coords(monkeypatch)
    gprMax.run(
        scenes=[_make_scene()], n=6, geometry_fixed=True, geometry_only=True,
        outputfile=tmp_path / "full", hide_progress_bars=True,
    )
    full_coords = list(captured)
    captured.clear()

    gprMax.run(
        scenes=[_make_scene()], n=3, i=4, geometry_fixed=True, geometry_only=True,
        outputfile=tmp_path / "restart", hide_progress_bars=True,
    )
    restart_coords = list(captured)

    assert restart_coords == full_coords[3:6]


def test_restarted_run_still_rejects_out_of_bounds_scan(tmp_path):
    """The scenario the review specifically flagged: a scan whose full
    (absolute, n=30) step count would run the receiver outside a
    20-cell domain by the final model must still be rejected when
    restarted partway through, not silently allowed to complete."""
    scene = _make_scene(domain=(0.02, 0.02, 0.02))
    with pytest.raises(ValueError):
        gprMax.run(
            scenes=[scene], n=6, i=25, geometry_fixed=True, geometry_only=True,
            outputfile=tmp_path / "restart_oob", hide_progress_bars=True,
        )


def test_non_restarted_scan_still_rejects_out_of_bounds(tmp_path):
    """No regression: the ordinary (non-restart) bounds-check path must
    still work exactly as before."""
    scene = _make_scene(domain=(0.02, 0.02, 0.02))
    with pytest.raises(ValueError):
        gprMax.run(
            scenes=[scene], n=30, geometry_fixed=True, geometry_only=True,
            outputfile=tmp_path / "nonrestart_oob", hide_progress_bars=True,
        )


def test_non_restarted_in_bounds_scan_still_completes(monkeypatch, tmp_path):
    """No regression: an in-bounds, non-restarted scan must still
    complete and reposition correctly."""
    captured = _capture_rx_coords(monkeypatch)
    gprMax.run(
        scenes=[_make_scene(domain=(0.05, 0.05, 0.05))], n=6, geometry_fixed=True,
        geometry_only=True, outputfile=tmp_path / "inbounds", hide_progress_bars=True,
    )
    assert captured == [(5 + i, 5, 5) for i in range(6)]


def test_restart_with_negative_step_still_rejects_out_of_bounds(tmp_path):
    """A negative step is a valid request to move backward each model -
    confirm the restart-aware bounds check also catches an out-of-bounds
    negative-direction scan, not just positive-direction ones."""
    scene = _make_scene(domain=(0.02, 0.02, 0.02), rx_step=(-1e-3, 0.0, 0.0))
    with pytest.raises(ValueError):
        gprMax.run(
            scenes=[scene], n=6, i=25, geometry_fixed=True, geometry_only=True,
            outputfile=tmp_path / "restart_oob_neg", hide_progress_bars=True,
        )
