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
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Step-size validation for #rx_array.

The documented minimum step is one spatial discretisation, and zero keeps
its role as the "single row on this axis" marker. Validation used to run
on the already-rounded step, so a nonzero step below one cell silently
became one cell (0.4 mm and 0.6 mm requests on a 1 mm grid both delivered
1 mm spacing), and a step in (-0.5 dl, 0) rounded to zero, was replaced by
one cell, and silently reversed direction. The checks now run on the raw
requested step, exempting singleton axes (equal endpoints, including the
2D invariant axis), where any non-negative step resolves to one point.
"""

from itertools import product

import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod

DL = 1e-3


def _capture_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _scene(domain=(0.02, 0.02, 0.02), dl=(DL, DL, DL)):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=dl))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    return scene


def _build(scene, monkeypatch, tmp_path, name):
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / name, hide_progress_bars=True
    )
    return captured["grid"]


@pytest.mark.parametrize("step", (0.0004, 0.0006, 0.0009))
def test_sub_cell_step_on_range_is_rejected(tmp_path, step):
    """A nonzero step below one cell cannot resolve a range axis and must
    raise the documented error instead of being silently rounded up."""
    scene = _scene()
    scene.add(gprMax.RxArray(p1=(0.002, 0.006, 0.006), p2=(0.010, 0.006, 0.006), dl=(step, 0, 0)))
    with pytest.raises(
        ValueError, match="step size should not be less than the spatial discretisation"
    ):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "reject",
            hide_progress_bars=True,
        )


@pytest.mark.parametrize("step", (-0.0004, -0.0005))
def test_negative_sub_cell_step_is_rejected(tmp_path, step):
    """Steps in (-0.5 dl, 0) round to zero before the old negative check,
    so they used to be replaced by +1 cell (a silent sign flip)."""
    scene = _scene()
    scene.add(gprMax.RxArray(p1=(0.002, 0.006, 0.006), p2=(0.010, 0.006, 0.006), dl=(step, 0, 0)))
    with pytest.raises(ValueError, match="step size should not be less than zero"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "reject_neg",
            hide_progress_bars=True,
        )


def test_step_equal_to_discretisation_is_accepted(monkeypatch, tmp_path):
    """One cell is the documented minimum and keeps its exact meaning."""
    scene = _scene()
    scene.add(gprMax.RxArray(p1=(0.002, 0.006, 0.006), p2=(0.010, 0.006, 0.006), dl=(0.001, 0, 0)))
    grid = _build(scene, monkeypatch, tmp_path, "min_step")
    assert [rx.coord[0] for rx in grid.rxs] == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert all(rx.coord[1] == 6 and rx.coord[2] == 6 for rx in grid.rxs)


def test_step_above_discretisation_rounds_to_nearest_cells(monkeypatch, tmp_path):
    """Steps of at least one cell keep nearest-cell rounding."""
    scene = _scene()
    scene.add(gprMax.RxArray(p1=(0.002, 0.006, 0.006), p2=(0.010, 0.006, 0.006), dl=(0.0016, 0, 0)))
    grid = _build(scene, monkeypatch, tmp_path, "rounded_step")
    assert [rx.coord[0] for rx in grid.rxs] == [2, 4, 6, 8, 10]


def test_zero_step_on_range_keeps_per_cell_stepping(monkeypatch, tmp_path):
    """Zero remains the documented single-row marker and keeps its pinned
    per-cell behaviour on ranges."""
    scene = _scene()
    scene.add(gprMax.RxArray(p1=(0.002, 0.002, 0.002), p2=(0.004, 0.004, 0.004), dl=(0, 0, 0)))
    grid = _build(scene, monkeypatch, tmp_path, "zero_step")
    coords = [tuple(rx.coord) for rx in grid.rxs]
    assert coords == list(product((2, 3, 4), repeat=3))


def test_zero_step_on_singleton_yields_one_receiver(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.RxArray(p1=(0.005, 0.005, 0.005), p2=(0.005, 0.005, 0.005), dl=(0, 0, 0)))
    grid = _build(scene, monkeypatch, tmp_path, "singleton_zero")
    assert [tuple(rx.coord) for rx in grid.rxs] == [(5, 5, 5)]


def test_sub_cell_step_on_singleton_axis_is_accepted(monkeypatch, tmp_path):
    """A singleton axis resolves to one point whatever the non-negative
    step, so the minimum-step rule does not apply there."""
    scene = _scene()
    scene.add(
        gprMax.RxArray(p1=(0.002, 0.006, 0.006), p2=(0.010, 0.006, 0.006), dl=(0.001, 0.0004, 0))
    )
    grid = _build(scene, monkeypatch, tmp_path, "singleton_subcell")
    assert [rx.coord[0] for rx in grid.rxs] == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert all(rx.coord[1] == 6 and rx.coord[2] == 6 for rx in grid.rxs)


def test_2d_invariant_axis_accepts_sub_cell_step(monkeypatch, tmp_path):
    """The resolved 2D invariant axis is a singleton, so a sub-cell step
    there stays valid (it lands on the interior reference layer)."""
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, float("inf"))))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(
        gprMax.RxArray(
            p1=(0.002, 0.002, float("inf")),
            p2=(0.006, 0.006, float("inf")),
            dl=(0.002, 0.002, 0.0004),
        )
    )
    grid = _build(scene, monkeypatch, tmp_path, "rxarray_2d_subcell")
    assert len(grid.rxs) == 9
    assert all(rx.coord[2] == 1 for rx in grid.rxs)
    assert [rx.coord[0] for rx in grid.rxs] == [2, 2, 2, 4, 4, 4, 6, 6, 6]
    assert [rx.coord[1] for rx in grid.rxs] == [2, 4, 6] * 3


@pytest.mark.parametrize("axis", range(3))
def test_one_cell_step_with_float_round_off_is_accepted(monkeypatch, tmp_path, axis):
    """A step that means exactly one cell must build even when float
    arithmetic lands it a hair under: 0.011 - 0.010 on a 1 mm grid is
    0.0009999999999999992, and rejecting it would break models that
    compute spacing from coordinates."""
    step = 0.011 - 0.010
    dl = [0.0, 0.0, 0.0]
    dl[axis] = step
    lower = [0.005] * 3
    upper = list(lower)
    upper[axis] = 0.007
    scene = _scene()
    scene.add(gprMax.RxArray(p1=tuple(lower), p2=tuple(upper), dl=tuple(dl)))
    grid = _build(scene, monkeypatch, tmp_path, f"round_off_axis_{axis}")
    assert len(grid.rxs) == 3
    assert [tuple(rx.coord)[axis] for rx in grid.rxs] == [5, 6, 7]
    assert all(
        tuple(rx.coord)[(axis + 1) % 3] == 5 and tuple(rx.coord)[(axis + 2) % 3] == 5
        for rx in grid.rxs
    )


def test_nextafter_below_one_cell_is_accepted(monkeypatch, tmp_path):
    """np.nextafter(dl, 0) is the closest float under one cell and is a
    one-cell request, not a sub-cell one."""
    scene = _scene()
    scene.add(
        gprMax.RxArray(
            p1=(0.002, 0.006, 0.006), p2=(0.010, 0.006, 0.006), dl=(np.nextafter(DL, 0.0), 0, 0)
        )
    )
    grid = _build(scene, monkeypatch, tmp_path, "nextafter_step")
    assert [rx.coord[0] for rx in grid.rxs] == [2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_float32_origin_step_is_accepted(monkeypatch, tmp_path):
    """Spacing imported from float32 data sits just above one cell and
    must keep stepping at one cell."""
    scene = _scene()
    scene.add(
        gprMax.RxArray(
            p1=(0.002, 0.006, 0.006), p2=(0.010, 0.006, 0.006), dl=(float(np.float32(DL)), 0, 0)
        )
    )
    grid = _build(scene, monkeypatch, tmp_path, "float32_step")
    assert [rx.coord[0] for rx in grid.rxs] == [2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_genuinely_sub_cell_step_still_rejected(tmp_path):
    """The tolerance only covers round-off: a real sub-cell request stays
    an error."""
    scene = _scene()
    scene.add(
        gprMax.RxArray(p1=(0.002, 0.006, 0.006), p2=(0.010, 0.006, 0.006), dl=(0.00099, 0, 0))
    )
    with pytest.raises(
        ValueError, match="step size should not be less than the spatial discretisation"
    ):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "still_reject",
            hide_progress_bars=True,
        )
