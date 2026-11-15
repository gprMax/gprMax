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

"""Regression tests: slabs overhanging a domain edge by less than half a
cell must be clipped at the last cell, never written past it.

check_thickness used to return level + thickness == size + 1 cells when
the lower face and the thickness both rounded up onto the terminal node
(a documented use case - objects are permitted to extend outwith the
model domain). The unclamped Cython volume loops then wrote past
solid/rigidE/rigidH with bounds checking disabled, silently corrupting
neighbouring memory. The clip now yields the in-domain remainder, and a
slab whose discretised footprint is empty becomes a face on the boundary
plane, matching the pre-existing zero-thickness-face-at-the-end
behaviour.
"""
import numpy as np
import pytest

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


def _scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="diel"))
    return scene


def test_triangle_overhang_clipped_to_domain(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(
        gprMax.Triangle(
            p1=(0.0096, 0.002, 0.002),
            p2=(0.0096, 0.006, 0.002),
            p3=(0.0096, 0.002, 0.005),
            thickness=0.0006,
            material_id="diel",
        )
    )
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "tri_overhang",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    diel = next(m for m in grid.materials if m.ID == "diel")

    # The slab rounds onto the terminal node; it becomes a face there,
    # not voxels past the last cell.
    assert not np.any(grid.solid == diel.numID)
    assert np.any(grid.ID[1, 10] == diel.numID)
    assert np.any(grid.ID[2, 10] == diel.numID)
    assert not np.any(grid.ID[0, 10] == diel.numID)
    assert np.any(grid.rigidE[:, 9])
    assert set(np.nonzero((grid.rigidE > 0).any(axis=(0, 2, 3)))[0]) == {9}


def test_cylindrical_sector_overhang_clipped_to_domain(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(
        gprMax.CylindricalSector(
            normal="x",
            ctr1=0.005,
            ctr2=0.005,
            extent1=0.0096,
            extent2=0.0102,
            r=0.004,
            start=0,
            end=180,
            material_id="diel",
        )
    )
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "sec_overhang",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    diel = next(m for m in grid.materials if m.ID == "diel")

    assert not np.any(grid.solid == diel.numID)
    assert np.any(grid.ID[1, 10] == diel.numID)
    assert np.any(grid.ID[2, 10] == diel.numID)
    assert not np.any(grid.ID[0, 10] == diel.numID)
    assert np.any(grid.rigidE[:, 9])
    assert set(np.nonzero((grid.rigidE > 0).any(axis=(0, 2, 3)))[0]) == {9}


def test_triangle_fully_inside_still_builds(monkeypatch, tmp_path):
    """The same prism fully inside the domain is unaffected."""
    scene = _scene()
    scene.add(
        gprMax.Triangle(
            p1=(0.004, 0.002, 0.002),
            p2=(0.004, 0.006, 0.002),
            p3=(0.004, 0.002, 0.005),
            thickness=0.0006,
            material_id="diel",
        )
    )
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "tri_inside",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    diel = next(m for m in grid.materials if m.ID == "diel")

    assert np.any(grid.solid == diel.numID)
    assert not np.any(grid.ID[1, 10] == diel.numID)
