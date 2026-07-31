"""Regression test for FDTDGrid.fractalvolumes double registration
(Codex-reported): FDTDGrid.add_fractal_volume() (and MPIGrid's override)
already appends the new FractalVolume to grid.fractalvolumes and returns
it. FractalBox.pre_build() then appended the SAME object a second time,
so every fractal box ended up with two references to the identical
volume in grid.fractalvolumes.

Confirmed this is unrelated to 2D TE mode's 2-cell invariant axis (that's
handled entirely within a single FractalVolume's own array dimensions,
never by duplicate object references) and does not cause real double
processing anywhere - add_grass/add_surface_roughness/add_surface_water
only ever take the first match from a filtered list (harmless whether
duplicated or not), and fractal generation itself is triggered directly
via FractalBox.build()'s own `self.volume` reference, never by iterating
grid.fractalvolumes. The one real, confirmed effect: FDTDGrid.
mem_est_fractals() sums memory per list entry, so every fractal box's
contribution to the pre-run "Memory required" estimate (and the host
memory-sufficiency warning in gprMax/utilities/host_info.py) was
silently doubled - not real wasted memory (the same array was only ever
allocated once), just an inflated estimate.

Fixed by removing the redundant append in FractalBox.pre_build() - the
one grid.add_fractal_volume() already does is sufficient.
"""
import numpy as np

import gprMax
import gprMax.model as model_mod


def _capture(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched(self):
        orig_build(self)
        captured["fractalvolumes"] = list(self.G.fractalvolumes)
        captured["mem_est_fractals"] = self.G.mem_est_fractals()

    monkeypatch.setattr(model_mod.Model, "build", patched)
    return captured


def _base_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
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


def test_single_fractal_box_registered_exactly_once(monkeypatch, tmp_path):
    captured = _capture(monkeypatch)
    scene = _base_scene()
    _add_fractal_box(scene, (0.01, 0.01, 0.01), (0.02, 0.02, 0.02), "fb1")

    gprMax.run(
        scenes=[scene], n=1, geometry_only=True,
        outputfile=tmp_path / "single", hide_progress_bars=True,
    )

    assert len(captured["fractalvolumes"]) == 1


def test_two_fractal_boxes_registered_exactly_once_each(monkeypatch, tmp_path):
    captured = _capture(monkeypatch)
    scene = _base_scene()
    _add_fractal_box(scene, (0.01, 0.01, 0.01), (0.02, 0.02, 0.02), "fb1")
    _add_fractal_box(scene, (0.03, 0.03, 0.03), (0.04, 0.04, 0.04), "fb2")

    gprMax.run(
        scenes=[scene], n=1, geometry_only=True,
        outputfile=tmp_path / "two", hide_progress_bars=True,
    )

    volumes = captured["fractalvolumes"]
    assert len(volumes) == 2
    assert volumes[0] is not volumes[1]
    # No entry should be a duplicate reference to another.
    assert len({id(v) for v in volumes}) == 2


def test_mem_est_fractals_matches_manual_single_count(monkeypatch, tmp_path):
    captured = _capture(monkeypatch)
    scene = _base_scene()
    _add_fractal_box(scene, (0.01, 0.01, 0.01), (0.02, 0.02, 0.02), "fb1")

    gprMax.run(
        scenes=[scene], n=1, geometry_only=True,
        outputfile=tmp_path / "mem_est", hide_progress_bars=True,
    )

    vol = captured["fractalvolumes"][0]
    expected = np.prod(vol.size) * vol.dtype.itemsize
    for surface in vol.fractalsurfaces:
        surfacedims = surface.get_surface_dims()
        expected += surfacedims[0] * surfacedims[1] * surface.dtype.itemsize

    assert captured["mem_est_fractals"] == expected
