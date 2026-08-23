"""Regression tests: #rx_array, #src_steps, #rx_steps in 2D TM/TE mode.

- RxArray.build() now resolves `inf` per-axis like a source: the invariant
  axis uses the single-point mode-aware rule (redirecting to the interior
  reference layer, matching HertzianDipole/Rx), while the other two axes
  use the ordinary lower/upper range rule to define the array's real
  extent and step count.
- SrcSteps/RxSteps now reject a nonzero step component on the invariant
  axis in 2D mode - previously this silently moved sources/receivers onto
  the forced-dead outer wall from the second model run onwards (the
  within_bounds() check used internally doesn't know about "dead but
  in-bounds" positions).
- Two unrelated, pre-existing bugs found and fixed alongside this, in the
  same RxArray.build() method:
  - discretised_upper_point was computed from self.lower_point twice (a
    copy-paste bug), making the lower<upper validation check vacuous.
  - The step size (dx, dy, dz) used to build the array was re-derived from
    the raw, uncorrected `self.dl` instead of the already-corrected
    `discretised_dl` (which maps a 0 step to 1 cell, a common "single row
    along this axis" pattern) - causing np.arange()'s internal division to
    divide by zero whenever any axis's dl was exactly 0.
"""
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


def _base_scene(mode, dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    return scene


def test_rx_array_te_inf_lands_on_interior_layer(monkeypatch, tmp_path):
    scene = _base_scene("TE")
    scene.add(gprMax.RxArray(p1=(0.002, 0.002, INF), p2=(0.006, 0.006, INF), dl=(0.002, 0.002, 0)))
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "rxarray_te", hide_progress_bars=True
    )
    grid = captured["grid"]
    assert len(grid.rxs) == 9
    assert all(rx.coord[2] == 1 for rx in grid.rxs)


def test_rx_array_tm_inf_lands_on_layer_zero(monkeypatch, tmp_path):
    scene = _base_scene("TM")
    scene.add(gprMax.RxArray(p1=(0.002, 0.002, INF), p2=(0.004, 0.004, INF), dl=(0.002, 0.002, 0)))
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "rxarray_tm", hide_progress_bars=True
    )
    grid = captured["grid"]
    assert len(grid.rxs) == 4
    assert all(rx.coord[2] == 0 for rx in grid.rxs)


def test_rx_array_rejects_lower_greater_than_upper(tmp_path):
    """Regression for a copy-paste bug: discretised_upper_point was
    computed from self.lower_point twice, making this check vacuous."""
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.RxArray(p1=(0.008, 0.008, 0.008), p2=(0.002, 0.002, 0.002), dl=(0.001, 0.001, 0.001))
    )
    with pytest.raises(ValueError, match="lower coordinates should be less"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "rxarray_bad_range",
            hide_progress_bars=True,
        )


def test_rx_array_zero_dl_axis_does_not_crash(tmp_path):
    """Regression for a divide-by-zero: dx/dy/dz were re-derived from the
    raw (uncorrected) dl instead of the already-corrected discretised_dl.
    """
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.RxArray(p1=(0.002, 0.002, 0.005), p2=(0.006, 0.006, 0.005), dl=(0.002, 0.002, 0)))
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "rxarray_zero_dl", hide_progress_bars=True
    )


def test_src_steps_rejects_invariant_axis_step(tmp_path):
    scene = _base_scene("TE")
    scene.add(gprMax.SrcSteps(p1=(0.001, 0.0, 0.001)))
    with pytest.raises(ValueError, match="invariant axis"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "srcsteps_bad",
            hide_progress_bars=True,
        )


def test_src_steps_allows_transverse_axis_step(tmp_path):
    scene = _base_scene("TE")
    scene.add(gprMax.SrcSteps(p1=(0.001, 0.0, 0.0)))
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "srcsteps_good", hide_progress_bars=True
    )


def test_rx_steps_rejects_invariant_axis_step(tmp_path):
    scene = _base_scene("TM")
    scene.add(gprMax.RxSteps(p1=(0.0, 0.0, 0.001)))
    with pytest.raises(ValueError, match="invariant axis"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "rxsteps_bad",
            hide_progress_bars=True,
        )


def test_rx_steps_allows_transverse_axis_step(tmp_path):
    scene = _base_scene("TM")
    scene.add(gprMax.RxSteps(p1=(0.001, 0.0, 0.0)))
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "rxsteps_good", hide_progress_bars=True
    )


def test_src_steps_and_rx_steps_unaffected_in_3d(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.SrcSteps(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.RxSteps(p1=(0.001, 0.001, 0.001)))
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "steps_3d", hide_progress_bars=True
    )
