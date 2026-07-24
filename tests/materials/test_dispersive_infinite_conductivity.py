"""Regression test: DispersiveMaterial.calculate_update_coeffsE() was missing
the infinite-conductivity/PEC guard present in the base Material class
(materials.py). Without it, a dispersive material (Debye/Lorentz/Drude) with
se=inf - either the literal builtin 'pec', or any user-defined material with
infinite conductivity - computed EA/EB both containing a 0.5*inf term, making
CA = EB/EA the indeterminate form -inf/inf, i.e. NaN, instead of the intended
0. NaN then propagates through the whole simulation from the very first
update (NaN * 0 is still NaN). Confirmed empirically before fixing; this
locks in the fix at both the coefficient level and a real, full solve.
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


def _scene_with_dispersive_infinite_conductivity_material():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(10e-3, 10e-3, 10e-3)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Material(er=1, se=float("inf"), mr=1, sm=0, id="mypec"))
    scene.add(
        gprMax.AddDebyeDispersion(poles=1, er_delta=[2.0], tau=[1e-10], material_ids=["mypec"])
    )
    scene.add(gprMax.Box(p1=(3e-3, 3e-3, 3e-3), p2=(7e-3, 7e-3, 7e-3), material_id="mypec"))
    return scene


def test_dispersive_infinite_conductivity_material_gets_zero_coefficients_not_nan(
    monkeypatch, tmp_path
):
    scene = _scene_with_dispersive_infinite_conductivity_material()

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True
    )
    grid = captured["grid"]
    mypec = next(m for m in grid.materials if m.ID == "mypec")

    assert mypec.CA == 0
    assert mypec.CBx == 0
    assert mypec.CBy == 0
    assert mypec.CBz == 0
    assert mypec.srce == 0
    assert np.array_equal(grid.updatecoeffsE[mypec.numID], [0, 0, 0, 0, 0])


def test_dispersive_infinite_conductivity_material_full_solve_has_no_nan(monkeypatch, tmp_path):
    scene = _scene_with_dispersive_infinite_conductivity_material()
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(5e-3, 5e-3, 1e-3), waveform_id="w"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    for arr in (grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz):
        assert not np.any(np.isnan(arr))

    # A genuinely propagating, finite field - not a degenerate all-zero
    # result masking the check.
    assert np.max(np.abs(grid.Ez)) > 1e-3
