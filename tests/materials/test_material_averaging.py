"""Regression tests for dielectric-smoothing eligibility on user-defined
materials.

Bug: Material.build() (gprMax/user_objects/cmds_multiuse.py) only disabled
`averagable` for infinite electric conductivity (se == inf, i.e. a
PEC-like material). There was no equivalent check for infinite magnetic
loss (sm == inf, i.e. a PMC-like material) - built-in `pmc` is protected by
a separate hardcoded `averagable = False` in create_built_in_materials(),
but a user-defined material with sm=inf was not, and would silently
participate in dielectric smoothing at geometry boundaries (e.g. a Box),
producing physically meaningless "smoothed" compound materials at what
should be a sharp perfect-magnetic-conductor boundary.
"""
from pathlib import Path

import gprMax
import gprMax.model as model_mod
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import create_built_in_materials
from gprMax.user_objects.cmds_multiuse import Material


def _build_custom_material(**kwargs):
    grid = FDTDGrid()
    create_built_in_materials(grid)
    material = Material(**kwargs)
    material.build(grid)
    return next(m for m in grid.materials if m.ID == kwargs["id"])


def test_custom_pmc_like_material_is_not_averagable():
    m = _build_custom_material(er=1, se=0, mr=1, sm=float("inf"), id="myPMC")
    assert m.averagable is False


def test_custom_pec_like_material_is_not_averagable():
    m = _build_custom_material(er=1, se=float("inf"), mr=1, sm=0, id="myPEC")
    assert m.averagable is False


def test_ordinary_dielectric_material_is_averagable():
    m = _build_custom_material(er=3, se=0.01, mr=1, sm=0, id="dielectric")
    assert m.averagable is True


def _capture_built_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def test_pmc_like_box_produces_no_smoothed_boundary_material(tmp_path: Path, monkeypatch):
    """End-to-end: a Box filled with a custom sm=inf material must not
    produce any dielectric-smoothed compound material at its boundary,
    exactly like a Box filled with a custom se=inf material doesn't.
    """
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="pmc_like_box"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=1, se=0, mr=1, sm=float("inf"), id="myPMC"))
    scene.add(gprMax.Box(p1=(0.005, 0.005, 0.005), p2=(0.009, 0.009, 0.009), material_id="myPMC"))

    captured = _capture_built_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "pmc_like_box",
        hide_progress_bars=True,
    )
    grid = captured["grid"]

    my_pmc = next(m for m in grid.materials if m.ID == "myPMC")
    assert my_pmc.averagable is False
    assert not any("+" in m.ID for m in grid.materials), (
        "no compound/smoothed material should have been created at the "
        f"myPMC box boundary, found: {[m.ID for m in grid.materials if '+' in m.ID]}"
    )
