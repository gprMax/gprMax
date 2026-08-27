# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Regression tests: in 2D TE mode, #add_grass must remain invariant across
the invariant axis's 2 cells, and must reproduce exactly what an equivalent
TM-mode (1-cell) grass surface with the same seed would produce.

Design (see gprMax/user_objects/cmds_geometry/add_grass.py and
gprMax/user_objects/cmds_geometry/fractal_box.py):

- AddGrass.build() samples blade position/height on a reduced (1-cell-thick)
  slice of the invariant axis (matching TM's own shape/computation exactly),
  then places the result at only ONE of the two invariant-axis cells -
  deliberately NOT broadcasting it directly, since the blade/root-building
  loop in FractalBox.build() increments a sequential index into Grass's
  fixed-size geometryparams array once per grid point visited; if both
  cells showed height>0 for the same row, that loop would visit the same
  logical blade twice, overflowing geometryparams (a real crash, not just
  an invariance bug) and giving the two cells different wobble geometry.

- FractalBox.build()'s blade/root loops (6 near-duplicate blocks: blade and
  root for each of xplus/yplus/zplus) treat a nonzero wobble offset along
  the invariant axis as out-of-bounds, exactly replicating what TM's own
  (naturally 1-cell-thick) bounds check already does - this keeps TM and TE
  bit-for-bit reproducible for the same seed, not just internally
  invariant.

- FractalBox.build() also has a general post-hoc mask broadcast (copy
  invariant-axis index 0 to index 1) right before voxels are built from the
  mask, as a safety net - a no-op for content that's already invariant by
  construction, a real backstop for grass specifically.

- #add_grass gets the same Case-A guard as #add_surface_roughness: a grass
  surface whose normal axis IS the invariant axis is rejected (no
  meaningful depth for grass on a 1/2-cell axis).

Also exercises a real, unrelated, pre-existing bug found while verifying
this: gprMax/utilities/utilities.py's round_int() crashed on numpy.float32
(the default single-precision dtype used by Grass's geometry parameters),
via decimal.Decimal() not accepting numpy.float32 directly - reproduced
independently in a plain 3D grass model with no #domain_mode involved, so
fixed as a narrow, unrelated side-fix (cast to float() first).
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


def _base_scene(mode, invariant_axis, dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    if invariant_axis == "z":
        scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    elif invariant_axis == "y":
        scene.add(gprMax.Domain(p1=(0.02, INF, 0.02)))
    # Domain is only 20 cells transverse; the default 10-cell PML on every
    # side would overlap itself (now correctly rejected - see
    # FDTDGrid._validate_pml_thickness()). PML is irrelevant to grass
    # TE-invariance, so just disable it.
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
    return scene


def _run_grass(monkeypatch, mode, tmp_path, invariant_axis="z", seed=42, n_blades=5):
    scene = _base_scene(mode, invariant_axis)
    if invariant_axis == "z":
        box_p1, box_p2 = (0.005, 0.005, INF), (0.015, 0.015, INF)
        grass_p1, grass_p2 = (0.015, 0.005, INF), (0.015, 0.015, INF)
    else:
        box_p1, box_p2 = (0.005, INF, 0.005), (0.015, INF, 0.015)
        grass_p1, grass_p2 = (0.005, INF, 0.015), (0.015, INF, 0.015)

    scene.add(
        gprMax.FractalBox(
            p1=box_p1,
            p2=box_p2,
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=3,
            mixing_model_id="mr1",
            id="fb1",
            seed=seed,
        )
    )
    scene.add(
        gprMax.AddGrass(
            p1=grass_p1,
            p2=grass_p2,
            frac_dim=1.5,
            limits=(0.015, 0.020),
            n_blades=n_blades,
            fractal_box_id="fb1",
            seed=seed,
        )
    )
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"{mode}_{invariant_axis}_grass",
        hide_progress_bars=True,
    )
    return captured["grid"]


@pytest.mark.parametrize("invariant_axis", ["z", "y"])
def test_te_grass_internally_invariant(monkeypatch, tmp_path, invariant_axis):
    grid = _run_grass(monkeypatch, "TE", tmp_path, invariant_axis=invariant_axis)
    idx = {"x": 0, "y": 1, "z": 2}[invariant_axis]
    slicer0 = [slice(None)] * 3
    slicer0[idx] = 0
    slicer1 = [slice(None)] * 3
    slicer1[idx] = 1
    assert np.array_equal(grid.solid[tuple(slicer0)], grid.solid[tuple(slicer1)])


@pytest.mark.parametrize("invariant_axis", ["z", "y"])
def test_te_grass_matches_tm_with_same_seed(monkeypatch, tmp_path, invariant_axis):
    grid_te = _run_grass(monkeypatch, "TE", tmp_path, invariant_axis=invariant_axis, seed=42)
    grid_tm = _run_grass(monkeypatch, "TM", tmp_path, invariant_axis=invariant_axis, seed=42)
    idx = {"x": 0, "y": 1, "z": 2}[invariant_axis]
    slicer = [slice(None)] * 3
    slicer[idx] = 0
    assert np.array_equal(grid_te.solid[tuple(slicer)], grid_tm.solid[tuple(slicer)])

    grassnumid = next(m.numID for m in grid_te.materials if m.ID == "grass")
    assert np.sum(grid_te.solid[tuple(slicer)] == grassnumid) > 0


def test_te_grass_different_seed_diverges(monkeypatch, tmp_path):
    grid_te = _run_grass(monkeypatch, "TE", tmp_path, seed=99)
    grid_tm = _run_grass(monkeypatch, "TM", tmp_path, seed=42)
    assert not np.array_equal(grid_te.solid[:, :, 0], grid_tm.solid[:, :, 0])


def test_te_grass_case_a_rejected_normal_equals_invariant_axis(monkeypatch, tmp_path):
    scene = _base_scene("TE", "z")
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
    scene.add(
        gprMax.AddGrass(
            p1=(0.005, 0.005, 0.002),
            p2=(0.015, 0.015, 0.002),
            frac_dim=1.5,
            limits=(0.0, 0.001),
            n_blades=5,
            fractal_box_id="fb1",
            seed=42,
        )
    )
    with pytest.raises(ValueError, match="normal is the invariant axis"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "grass_case_a",
            hide_progress_bars=True,
        )


def test_tm_grass_unaffected(monkeypatch, tmp_path):
    """TM's invariant axis is already 1 cell - the TE-specific shadow/
    reduce/bounds-check logic must never trigger there."""
    grid = _run_grass(monkeypatch, "TM", tmp_path)
    assert grid.solid.shape[2] == 1


def test_round_int_accepts_float32():
    """Pre-existing, unrelated bug found while verifying grass in TE mode:
    round_int() crashed on numpy.float32 (the default single-precision
    dtype), blocking any grass model - TM, TE, or 3D - whenever blade
    growth was computed. See gprMax/utilities/utilities.py.
    """
    from gprMax.utilities.utilities import round_int

    assert round_int(np.float32(3.7)) == 4
    assert round_int(np.float32(-2.3)) == -2
