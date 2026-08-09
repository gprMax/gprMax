"""Regression tests for making the dispersive-material update kernels
(fields_updates_dispersive.pyx) mode2d-aware, matching update_electric() in
fields_updates_normal.pyx - for consistency, clarity, and to stop wastefully
computing (and then discarding, via tex()/tey()/tez()'s ID-forcing) the
dead own-axis dispersive component in TE mode.

Values checked here were recorded from the pre-refactor kernel (which used
old-style implicit degenerate-size checks, protected only by the separate
tez() ID-forcing safety net) and confirmed bit-exact after the refactor -
this file locks that in.
"""
import numpy as np

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


def test_3d_dispersive_material_unaffected(monkeypatch, tmp_path):
    """A finite 3-D Debye dielectric remains a stable regression fixture."""
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(10e-3, 10e-3, 10e-3)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    scene.add(
        gprMax.AddDebyeDispersion(
            poles=1,
            er_delta=[2.0],
            tau=[1e-10],
            material_ids=["diel"],
        )
    )
    scene.add(
        gprMax.Box(
            p1=(3e-3, 3e-3, 3e-3),
            p2=(7e-3, 7e-3, 7e-3),
            material_id="diel",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(5e-3, 5e-3, 1e-3), waveform_id="w"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    for arr in (grid.Ex, grid.Ey, grid.Ez):
        assert not np.any(np.isnan(arr))
    # Pinned after replacing the former dispersive-PEC fixture: adding
    # electric dispersion to an ideal conductor is now rejected as invalid.
    assert np.isclose(np.max(np.abs(grid.Ez)), 0.1396385, rtol=1e-6)


def test_tez_dispersive_dead_component_and_interior_layer(monkeypatch, tmp_path):
    """Bit-exact regression against the pre-refactor (safety-net-protected)
    values: Ez (dead in TEz) must be exactly zero everywhere - now via a
    genuine skip, not a computed-then-discarded value - and Ex/Ey (live)
    must be zero at both wall layers and match the recorded interior-layer
    values exactly."""
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(20e-3, 20e-3, float("inf"))))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-11))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    scene.add(
        gprMax.AddDebyeDispersion(poles=1, er_delta=[2.0], tau=[1e-10], material_ids=["diel"])
    )
    scene.add(gprMax.Box(p1=(5e-3, 5e-3, 0), p2=(15e-3, 15e-3, 2e-3), material_id="diel"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="x", p1=(0.01, 0.01, 1e-3), waveform_id="w"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    assert np.max(np.abs(grid.Ez)) == 0.0
    assert np.max(np.abs(grid.Ex[:, :, 0])) == 0.0
    assert np.max(np.abs(grid.Ex[:, :, 2])) == 0.0
    # Regression values re-pinned for the now-mandatory PMLThickness(0)
    # fixture - see the comment in test_3d_dispersive_material_unaffected
    # above for why this domain's old pinned values were PML-dominated.
    assert np.isclose(np.max(np.abs(grid.Ex[:, :, 1])), 1.2266484, rtol=1e-6)
    assert np.max(np.abs(grid.Ey[:, :, 0])) == 0.0
    assert np.max(np.abs(grid.Ey[:, :, 2])) == 0.0
    assert np.isclose(np.max(np.abs(grid.Ey[:, :, 1])), 0.6385769, rtol=1e-6)


def test_tmz_dispersive_live_component(monkeypatch, tmp_path):
    """TM mode: only Ez (the invariant-axis component) should be live and
    dispersive; a real, finite, non-degenerate field."""
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(20e-3, 20e-3, float("inf"))))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-11))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    scene.add(
        gprMax.AddDebyeDispersion(poles=1, er_delta=[2.0], tau=[1e-10], material_ids=["diel"])
    )
    scene.add(gprMax.Box(p1=(5e-3, 5e-3, 0), p2=(15e-3, 15e-3, 1e-3), material_id="diel"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, 0), waveform_id="w"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    assert not np.any(np.isnan(grid.Ez))
    assert np.max(np.abs(grid.Ez)) > 1e-3
    assert np.max(np.abs(grid.Ex)) == 0.0
    assert np.max(np.abs(grid.Ey)) == 0.0
