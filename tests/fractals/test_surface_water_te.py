"""Regression tests: #add_surface_water gains the same `inf` wiring as
#add_surface_roughness/#add_grass - same bespoke flat-axis pattern (one
flat/normal axis resolved sign-based via role=None, two extent axes
resolved positionally via role="lower"/"upper"), since it has the identical
one-flat-axis-plus-two-extent-axes p1/p2 structure.

No separate Case-A (normal==invariant axis) guard is needed here:
AddSurfaceWater requires an *existing* FractalSurface on the same face,
which #add_surface_roughness's own Case-A guard already prevents from ever
being created on a normal==invariant-axis face - so AddSurfaceWater
naturally can't reach that face either, via the pre-existing "does not
have a rough surface applied" check.
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
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    # Domain is only 20 cells transverse; the default 10-cell PML on every
    # side would overlap itself (now correctly rejected - see
    # FDTDGrid._validate_pml_thickness()). PML is irrelevant to surface
    # water TE-invariance, so just disable it.
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.MaterialRange(
            er_lower=2,
            er_upper=6,
            sigma_lower=0,
            sigma_upper=0,
            mr_lower=1,
            mr_upper=1,
            ro_lower=0,
            ro_upper=0,
            id="mr1",
        )
    )
    scene.add(
        gprMax.FractalBox(
            p1=(0.005, 0.005, INF),
            p2=(0.015, 0.015, INF),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=3,
            mixing_model_id="mr1",
            id="fb1",
            seed=42,
        )
    )
    return scene


def test_add_surface_water_te_inf_and_invariant(monkeypatch, tmp_path):
    scene = _base_scene("TE")
    scene.add(
        gprMax.AddSurfaceRoughness(
            fractal_box_id="fb1",
            seed=42,
            p1=(0.015, 0.005, INF),
            p2=(0.015, 0.015, INF),
            frac_dim=1.5,
            weighting=(1, 1),
            limits=(0.010, 0.020),
        )
    )
    scene.add(
        gprMax.AddSurfaceWater(
            fractal_box_id="fb1", p1=(0.015, 0.005, INF), p2=(0.015, 0.015, INF), depth=0.012
        )
    )
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "water_te",
        hide_progress_bars=True,
    )
    grid = captured["grid"]

    water_material = next((m for m in grid.materials if m.ID == "water"), None)
    assert water_material is not None
    assert np.array_equal(grid.solid[:, :, 0], grid.solid[:, :, 1])
    assert np.sum(grid.solid[:, :, 0] == water_material.numID) > 0


def test_add_surface_water_without_matching_roughness_rejected(tmp_path):
    """A face with no AddSurfaceRoughness applied (whether because none was
    added, or because Case A rejected it) must still be rejected cleanly."""
    scene = _base_scene("TE")
    scene.add(
        gprMax.AddSurfaceWater(
            fractal_box_id="fb1", p1=(0.015, 0.005, INF), p2=(0.015, 0.015, INF), depth=0.012
        )
    )
    with pytest.raises(ValueError, match="does not have a rough surface applied"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "water_no_roughness",
            hide_progress_bars=True,
        )
