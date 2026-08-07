"""Construction tests for the experimental internal RIPML slab."""

import numpy as np

import gprMax
import gprMax.model as model_mod


def _capture_grid(monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def test_x0_internal_pml_is_local_graded_and_pec_capped(monkeypatch, tmp_path):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(
        gprMax.PMLSlab(
            p1=(0.005, 0.010, 0.010),
            p2=(0.015, 0.020, 0.020),
            termination_face="x0",
            id="feed_load",
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "run",
        hide_progress_bars=True,
    )
    grid = captured["grid"]

    assert len(grid.pmls["slabs"]) == 1
    pml = grid.pmls["slabs"][0]
    assert pml.internal
    assert pml.ID == "feed_load"
    assert pml.direction == "xminus"
    assert pml.termination_face == "x0"
    assert (pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf) == (
        5,
        15,
        10,
        20,
        10,
        20,
    )

    # The xminus profile is zero at the open +x entrance and grows towards
    # the -x cap. Its correction arrays cover only the selected y-z region.
    assert pml.EPhi1.shape[1:] == (11, 10, 11)
    assert pml.EPhi2.shape[1:] == (11, 11, 10)
    assert pml.ERA[0, 0] != pml.ERA[0, -1]

    pec_numid = next(material.numID for material in grid.materials if material.ID == "pec")
    assert np.all(grid.ID[1, 5, 10:20, 10:21] == pec_numid)
    assert np.all(grid.ID[2, 5, 10:21, 10:20] == pec_numid)

    # The open entrance and the normal E component are not converted to PEC.
    assert not np.any(grid.ID[1, 15, 10:20, 10:21] == pec_numid)
    assert not np.any(grid.ID[2, 15, 10:21, 10:20] == pec_numid)
    assert not np.any(grid.ID[0, 5, 10:20, 10:20] == pec_numid)

