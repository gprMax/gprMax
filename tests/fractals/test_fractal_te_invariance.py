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

"""Regression tests: in 2D TE mode, FractalBox and #add_surface_roughness
must remain invariant across the invariant axis's 2 cells, and must
reproduce exactly what an equivalent TM-mode (1-cell) box/surface with the
same seed/dimension/weighting would generate.

Design: FractalVolume.generate_fractal_volume() / FractalSurface.
generate_fractal_surface() detect a 2-cell-thick invariant axis in 2D TE
mode and, instead of generating independently over both cells (which would
break invariance and TM/TE reproducibility), generate a single 1-cell-thick
"shadow" volume/surface once and broadcast it to both cells. See
gprMax/fractals/fractal_volume.py and gprMax/fractals/fractal_surface.py.

#add_surface_roughness also gets a Case-A guard: a rough surface whose
normal axis IS the invariant axis is rejected in 2D mode (no meaningful
roughness depth on a 1/2-cell axis), matching the existing #plate 2D
restriction.
"""
import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod
from gprMax.fractals.fractal_volume import FractalVolume

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
    # FDTDGrid._validate_pml_thickness()). PML is irrelevant to fractal
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


def _run_fractal_box(monkeypatch, mode, tmp_path, seed=42, n_materials=3):
    # FractalBox.build() frees volume.fractalvolume/.mask once it has
    # consumed them into grid.solid/rigidE/rigidH/ID (see
    # gprMax/gprMax#389 - a large fractal box's array otherwise persists
    # for as long as the caller's own Scene object does, accumulating
    # across a multi-scene sweep). This test needs to inspect the raw
    # per-bin fractal pattern afterward to verify TE-mode invariance, so
    # snapshot it via generate_fractal_volume() (before the later
    # bin->material-ID remapping and the free) and restore it onto the
    # volume once the run completes - the shape/equality/invariance
    # properties asserted below hold identically at this point, since the
    # remapping that follows is a deterministic per-element lookup applied
    # uniformly to both invariant-axis cells.
    fractalvolume_snapshot = {}
    orig_generate = FractalVolume.generate_fractal_volume

    def patched_generate(self):
        result = orig_generate(self)
        fractalvolume_snapshot["value"] = self.fractalvolume.copy()
        return result

    monkeypatch.setattr(FractalVolume, "generate_fractal_volume", patched_generate)

    scene = _base_scene(mode)
    scene.add(
        gprMax.FractalBox(
            p1=(0.005, 0.005, INF),
            p2=(0.015, 0.015, INF),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=n_materials,
            mixing_model_id="mr1",
            id="fb1",
            seed=seed,
        )
    )
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"{mode}_fb",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    volume = next(v for v in grid.fractalvolumes if v.ID == "fb1")
    volume.fractalvolume = fractalvolume_snapshot["value"]
    return volume


def test_te_fractal_volume_invariant_across_both_cells(monkeypatch, tmp_path):
    volume = _run_fractal_box(monkeypatch, "TE", tmp_path)
    assert volume.fractalvolume.shape == (10, 10, 2)
    assert np.array_equal(volume.fractalvolume[:, :, 0], volume.fractalvolume[:, :, 1])


def test_te_fractal_box_grid_solid_invariant(monkeypatch, tmp_path):
    captured = _capture_grid(monkeypatch)
    scene = _base_scene("TE")
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
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "te_fb_solid",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    assert np.array_equal(grid.solid[5:15, 5:15, 0], grid.solid[5:15, 5:15, 1])


def test_te_fractal_volume_matches_tm_with_same_seed(monkeypatch, tmp_path):
    volume_te = _run_fractal_box(monkeypatch, "TE", tmp_path, seed=42)
    volume_tm = _run_fractal_box(monkeypatch, "TM", tmp_path, seed=42)
    assert volume_tm.fractalvolume.shape == (10, 10, 1)
    assert np.array_equal(volume_te.fractalvolume[:, :, 0], volume_tm.fractalvolume[:, :, 0])


def test_te_fractal_volume_differs_with_different_seed(monkeypatch, tmp_path):
    volume_te = _run_fractal_box(monkeypatch, "TE", tmp_path, seed=99)
    volume_tm = _run_fractal_box(monkeypatch, "TM", tmp_path, seed=42)
    assert not np.array_equal(volume_te.fractalvolume[:, :, 0], volume_tm.fractalvolume[:, :, 0])


def _run_surface(monkeypatch, mode, surface_kwargs, tmp_path, seed=42):
    scene = _base_scene(mode)
    scene.add(
        gprMax.FractalBox(
            p1=(0.005, 0.005, INF),
            p2=(0.015, 0.015, INF),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=3,
            mixing_model_id="mr1",
            id="fb1",
            seed=seed,
        )
    )
    scene.add(gprMax.AddSurfaceRoughness(fractal_box_id="fb1", seed=seed, **surface_kwargs))
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"{mode}_surf",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    volume = next(v for v in grid.fractalvolumes if v.ID == "fb1")
    return volume.fractalsurfaces[0]


def test_te_fractal_surface_case_b_invariant_across_both_cells(monkeypatch, tmp_path):
    surface = _run_surface(
        monkeypatch,
        "TE",
        dict(p1=(0.005, 0.005, INF), p2=(0.005, 0.015, INF), frac_dim=1.5, weighting=(1, 1), limits=(0.003, 0.007)),
        tmp_path,
    )
    assert surface.surfaceID == "xminus"
    assert surface.fractalsurface.shape == (10, 2)
    assert np.array_equal(surface.fractalsurface[:, 0], surface.fractalsurface[:, 1])


def test_te_fractal_surface_case_b_matches_tm_with_same_seed(monkeypatch, tmp_path):
    surface_te = _run_surface(
        monkeypatch,
        "TE",
        dict(p1=(0.005, 0.005, INF), p2=(0.005, 0.015, INF), frac_dim=1.5, weighting=(1, 1), limits=(0.003, 0.007)),
        tmp_path,
        seed=42,
    )
    surface_tm = _run_surface(
        monkeypatch,
        "TM",
        dict(p1=(0.005, 0.005, INF), p2=(0.005, 0.015, INF), frac_dim=1.5, weighting=(1, 1), limits=(0.003, 0.007)),
        tmp_path,
        seed=42,
    )
    assert surface_tm.fractalsurface.shape == (10, 1)
    assert np.allclose(surface_te.fractalsurface[:, 0], surface_tm.fractalsurface[:, 0])


def test_te_fractal_surface_case_a_rejected_explicit_wall_coords(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="normal is the invariant axis"):
        _run_surface(
            monkeypatch,
            "TE",
            dict(
                p1=(0.005, 0.005, 0.002),
                p2=(0.015, 0.015, 0.002),
                frac_dim=1.5,
                weighting=(1, 1),
                limits=(0.0, 0.004),
            ),
            tmp_path,
        )


def test_te_fractal_surface_case_a_inf_redirects_off_the_wall_and_is_rejected(monkeypatch, tmp_path):
    # inf on the (flat) invariant axis redirects to the TE interior layer
    # (index 1), which is neither the box's zs nor zf wall - so this is
    # rejected too, just via the pre-existing "external surfaces" check
    # rather than the Case-A message (there is no valid wall-attached
    # surface reachable via inf on the invariant axis in TE mode).
    with pytest.raises(ValueError, match="external surfaces"):
        _run_surface(
            monkeypatch,
            "TE",
            dict(
                p1=(0.005, 0.005, INF),
                p2=(0.015, 0.015, INF),
                frac_dim=1.5,
                weighting=(1, 1),
                limits=(0.0, 0.004),
            ),
            tmp_path,
        )


def test_tm_fractal_volume_single_cell_unaffected(monkeypatch, tmp_path):
    """TM mode's invariant axis is already 1 cell - the shadow/broadcast
    path must never trigger there (size check is == 2, TM is always 1)."""
    volume = _run_fractal_box(monkeypatch, "TM", tmp_path)
    assert volume.fractalvolume.shape == (10, 10, 1)
