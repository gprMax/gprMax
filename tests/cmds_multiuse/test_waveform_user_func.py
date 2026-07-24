"""Tests for #waveform's `user_func` option (wave_type='user') - lets a
Python callable be used directly as a waveform's amplitude-vs-time mapping,
as an alternative to the existing `user_values`/`user_time` array +
scipy.interpolate.interp1d path. Python API only (a callable can't be
expressed in a text input file).
"""
import numpy as np
import pytest

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


def test_user_func_is_used_directly_as_userfunc(monkeypatch, tmp_path):
    def my_waveform(time):
        return np.sin(2 * np.pi * 1e9 * time)

    scene = _scene()
    scene.add(gprMax.Waveform(wave_type="user", user_func=my_waveform, id="mywave"))

    grid = _run(monkeypatch, tmp_path, "user_func_basic", scene)
    waveform = next(w for w in grid.waveforms if w.ID == "mywave")
    assert waveform.userfunc is my_waveform
    assert waveform.calculate_value(1.23e-10, grid.dt) == my_waveform(1.23e-10)


def test_user_func_precompute_matches_direct_call(monkeypatch, tmp_path):
    """End-to-end: the per-source waveform precompute loop (calculate_waveform_values,
    called from HertzianDipole.build - runs even in geometry_only mode) must
    actually invoke user_func at each iteration's real time, not just store it.
    HertzianDipole specifically precomputes waveformvalues_halfdt (time offset
    by +0.5*dt, used in its update_electric formula), not waveformvalues_wholedt."""

    def my_waveform(time):
        return np.cos(2 * np.pi * 5e8 * time) * np.exp(-time / 1e-9)

    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    # Long enough relative to dt (~1.9e-12s at this discretisation) that
    # several genuine (non-start/stop-windowed-out) iterations exist to
    # compare, not just iteration 0.
    scene.add(gprMax.TimeWindow(time=2e-11))
    scene.add(gprMax.Waveform(wave_type="user", user_func=my_waveform, id="mywave"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.005, 0.005, 0.005), waveform_id="mywave"))

    grid = _run(monkeypatch, tmp_path, "user_func_precompute", scene)
    assert grid.iterations >= 5
    src = grid.hertziandipoles[0]
    expected = np.array(
        [my_waveform(grid.dt * i + 0.5 * grid.dt) for i in range(grid.iterations + 1)]
    )
    # Compare only the safely-interior iterations: the source's own
    # start/stop-vs-TimeWindow floating-point boundary check (unrelated to
    # user_func) can leave the very last one or two entries at their
    # np.zeros() default rather than a precomputed value - not a bug, just
    # not what this test is about.
    assert np.allclose(src.waveformvalues_halfdt[:-2], expected[:-2])


def test_closure_factory_produces_independent_waveforms(monkeypatch, tmp_path):
    def make_waveform(freq):
        def waveform(time):
            return np.sin(2 * np.pi * freq * time)

        return waveform

    scene = _scene()
    scene.add(gprMax.Waveform(wave_type="user", user_func=make_waveform(1e9), id="wave_a"))
    scene.add(gprMax.Waveform(wave_type="user", user_func=make_waveform(2e9), id="wave_b"))

    grid = _run(monkeypatch, tmp_path, "user_func_closures", scene)
    wave_a = next(w for w in grid.waveforms if w.ID == "wave_a")
    wave_b = next(w for w in grid.waveforms if w.ID == "wave_b")

    t = 1e-10
    assert wave_a.calculate_value(t, grid.dt) == np.sin(2 * np.pi * 1e9 * t)
    assert wave_b.calculate_value(t, grid.dt) == np.sin(2 * np.pi * 2e9 * t)


def test_user_func_and_user_values_together_rejected(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(
        gprMax.Waveform(
            wave_type="user",
            user_func=lambda time: 0.0,
            user_values=np.array([0.0, 1.0]),
            id="mywave",
        )
    )

    with pytest.raises(ValueError, match="exactly one of"):
        _run(monkeypatch, tmp_path, "user_func_and_values", scene)


def test_user_func_missing_both_rejected(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.Waveform(wave_type="user", id="mywave"))

    with pytest.raises(ValueError, match="user_func.*user_values"):
        _run(monkeypatch, tmp_path, "user_func_missing", scene)


def test_user_func_not_callable_rejected(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.Waveform(wave_type="user", user_func="not a function", id="mywave"))

    with pytest.raises(ValueError, match="callable"):
        _run(monkeypatch, tmp_path, "user_func_not_callable", scene)


def test_user_func_wrong_signature_rejected_at_build_time(monkeypatch, tmp_path):
    def bad_waveform(time, extra_required_arg):
        return time + extra_required_arg

    scene = _scene()
    scene.add(gprMax.Waveform(wave_type="user", user_func=bad_waveform, id="mywave"))

    with pytest.raises(ValueError, match="single float"):
        _run(monkeypatch, tmp_path, "user_func_bad_sig", scene)


def test_user_func_non_numeric_return_rejected_at_build_time(monkeypatch, tmp_path):
    def bad_waveform(time):
        return "not a number"

    scene = _scene()
    scene.add(gprMax.Waveform(wave_type="user", user_func=bad_waveform, id="mywave"))

    with pytest.raises(ValueError, match="single float"):
        _run(monkeypatch, tmp_path, "user_func_bad_return", scene)


def test_user_values_path_still_works_unchanged(monkeypatch, tmp_path):
    """Regression check: refactoring the user_values branch to sit alongside
    user_func must not change its existing behaviour."""
    dl = 1e-3
    scene = _scene(dl=dl)
    scene.add(
        gprMax.Waveform(
            wave_type="user",
            user_values=np.array([0.0, 1.0, 0.0, -1.0, 0.0]),
            user_time=np.array([0.0, 1e-13, 2e-13, 3e-13, 4e-13]),
            id="mywave",
        )
    )

    grid = _run(monkeypatch, tmp_path, "user_values_regression", scene)
    waveform = next(w for w in grid.waveforms if w.ID == "mywave")
    # Sampled exactly at the given user_time points - interp1d should
    # reproduce user_values exactly there, same as before this change.
    assert np.isclose(waveform.calculate_value(0.0, grid.dt), 0.0)
    assert np.isclose(waveform.calculate_value(1e-13, grid.dt), 1.0)
    assert np.isclose(waveform.calculate_value(3e-13, grid.dt), -1.0)
