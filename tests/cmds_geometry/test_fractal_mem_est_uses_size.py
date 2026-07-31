"""Regression test for FDTDGrid.mem_est_fractals() (Codex-reported):
it estimated each fractal volume's memory as `np.prod(vol.start) *
vol.dtype.itemsize`, where `vol.start` is the volume's starting grid
COORDINATE, not its size. A volume starting at the origin (start=(0,0,0))
was therefore always estimated as exactly 0 bytes regardless of its real
extent, and otherwise the "estimate" tracked the box's position in the
domain rather than how many cells it actually occupies.

Fixed by using `vol.size` (`self.stop - self.start`, the volume's true
(nx, ny, nz) extent) instead of `vol.start`.

Note: the pre-existing test in test_fractal_box_double_registration.py
used a box positioned such that start == size numerically by coincidence
((10, 10, 10) cells offset, (10, 10, 10) cells in extent), so it could
not distinguish the buggy formula from the fixed one - it has been
updated to use `vol.size`, and this file adds tests that genuinely
distinguish position from size.
"""
import numpy as np

import gprMax
import gprMax.model as model_mod


def _capture(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched(self):
        orig_build(self)
        captured["mem_est_fractals"] = self.G.mem_est_fractals()

    monkeypatch.setattr(model_mod.Model, "build", patched)
    return captured


def _base_scene(domain=(0.06, 0.06, 0.06)):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.SoilPeplinski(
            sand_fraction=0.5, clay_fraction=0.5, bulk_density=2.0, sand_density=2.66,
            water_fraction_lower=0.001, water_fraction_upper=0.25, id="soil1",
        )
    )
    return scene


def _add_fractal_box(scene, p1, p2, box_id):
    scene.add(
        gprMax.FractalBox(
            p1=p1, p2=p2, frac_dim=1.5, weighting=(1, 1, 1), n_materials=2,
            mixing_model_id="soil1", id=box_id,
        )
    )


def test_fractal_box_at_origin_has_nonzero_mem_estimate(tmp_path, monkeypatch):
    """A box starting exactly at the domain origin used to estimate to 0
    bytes (np.prod((0, 0, 0)) == 0), no matter how large it was."""
    captured = _capture(monkeypatch)
    scene = _base_scene()
    _add_fractal_box(scene, (0.0, 0.0, 0.0), (0.02, 0.02, 0.02), "fb1")

    gprMax.run(
        scenes=[scene], n=1, geometry_only=True,
        outputfile=tmp_path / "origin", hide_progress_bars=True,
    )

    assert captured["mem_est_fractals"] > 0


def test_mem_estimate_depends_on_size_not_position(tmp_path, monkeypatch):
    """Two identically-sized boxes at different positions must produce
    the same memory estimate; the old formula tracked position instead."""
    captured_a = _capture(monkeypatch)
    scene_a = _base_scene()
    _add_fractal_box(scene_a, (0.0, 0.0, 0.0), (0.02, 0.02, 0.02), "fb1")
    gprMax.run(
        scenes=[scene_a], n=1, geometry_only=True,
        outputfile=tmp_path / "pos_a", hide_progress_bars=True,
    )
    mem_a = captured_a["mem_est_fractals"]

    captured_b = _capture(monkeypatch)
    scene_b = _base_scene()
    _add_fractal_box(scene_b, (0.03, 0.03, 0.03), (0.05, 0.05, 0.05), "fb1")
    gprMax.run(
        scenes=[scene_b], n=1, geometry_only=True,
        outputfile=tmp_path / "pos_b", hide_progress_bars=True,
    )
    mem_b = captured_b["mem_est_fractals"]

    assert mem_a == mem_b


def test_mem_estimate_scales_with_volume_size(tmp_path, monkeypatch):
    """A bigger box (same start-offset pattern) must estimate to more
    memory than a smaller one - sanity check that size is actually driving
    the number, not some other coincidental invariant."""
    captured_small = _capture(monkeypatch)
    scene_small = _base_scene()
    _add_fractal_box(scene_small, (0.01, 0.01, 0.01), (0.02, 0.02, 0.02), "fb1")
    gprMax.run(
        scenes=[scene_small], n=1, geometry_only=True,
        outputfile=tmp_path / "small", hide_progress_bars=True,
    )
    mem_small = captured_small["mem_est_fractals"]

    captured_large = _capture(monkeypatch)
    scene_large = _base_scene()
    _add_fractal_box(scene_large, (0.01, 0.01, 0.01), (0.04, 0.04, 0.04), "fb1")
    gprMax.run(
        scenes=[scene_large], n=1, geometry_only=True,
        outputfile=tmp_path / "large", hide_progress_bars=True,
    )
    mem_large = captured_large["mem_est_fractals"]

    assert mem_large > mem_small
