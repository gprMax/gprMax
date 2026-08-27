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

"""Regression tests for PML thickness validation gaps (Codex-reported):

1. Grids default to a 10-cell PML on every side (FDTDGrid.__init__), but
   the overlap check ("no PML may take up more than half the domain")
   only ran inside PMLThickness.build() - which only runs when the user
   explicitly supplies `#pml_cells`. A domain with a transverse dimension
   of 20 cells or fewer and NO explicit PML command at all would
   silently let the default 10+10 PMLs meet/overlap with no error.
   Fixed by adding FDTDGrid._validate_pml_thickness(), called
   unconditionally from FDTDGrid.build().

2. PMLThickness.build() didn't reject negative thickness values -
   FDTDGrid._build_pmls() only builds a slab when thickness > 0, so a
   negative value silently behaved like 0 (no PML) instead of raising an
   error for a nonsensical request.
"""
import pytest

import gprMax


def _run(scene, tmp_path, label):
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )


def _base_scene(domain, dl=1e-3, time=1e-12):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.TimeWindow(time=time))
    return scene


def test_default_pml_overlaps_on_small_domain_without_explicit_pml_cells(tmp_path):
    # 15 cells in x with the default 10-cell PML on both x0/xmax (20
    # cells total) overlaps - and this model never declares #pml_cells
    # at all, so the old check (inside PMLThickness.build()) never ran.
    scene = _base_scene((0.015, 0.05, 0.05))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "default_pml_overlap")


def test_default_pml_fits_on_large_enough_domain(tmp_path):
    scene = _base_scene((0.05, 0.05, 0.05))
    _run(scene, tmp_path, "default_pml_ok")


def test_negative_pml_thickness_rejected(tmp_path):
    scene = _base_scene((0.05, 0.05, 0.05))
    scene.add(gprMax.PMLThickness(thickness=-1))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "neg_pml")


def test_positive_pml_thickness_still_works(tmp_path):
    scene = _base_scene((0.05, 0.05, 0.05))
    scene.add(gprMax.PMLThickness(thickness=5))
    _run(scene, tmp_path, "pos_pml")
