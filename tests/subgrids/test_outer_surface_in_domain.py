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

"""A subgrid's Outer Surface (is_os_sep main cells outside its Inner Surface) must lie at least
one main cell inside the domain. Before this was checked, a subgrid near the domain's edge
failed only once time stepping started: a segmentation fault (Outer Surface two or more cells
past the lower edge), an OverflowError (on or just past it) or an IndexError (past the upper
edge)."""

import pytest

import gprMax


def _scene(p1, p2, is_os_sep=3):
    """A 100 mm cube of 2 mm cells (50 per side) with a subgrid from p1 to p2."""
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.1, 0.1, 0.1)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(gprMax.SubGridHSG(p1=p1, p2=p2, ratio=3, id="sg", is_os_sep=is_os_sep))
    return scene


def _build(scene, tmp_path):
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "m", subgrid=True)


@pytest.mark.parametrize("cell", [0, 1, 2, 3])
def test_an_outer_surface_on_or_past_the_lower_edge_is_refused(cell, tmp_path):
    x = cell * 0.002
    with pytest.raises(
        ValueError, match=r"Outer Surface .* inside the domain; it does not along x"
    ):
        _build(_scene((x, 0.03, 0.03), (x + 0.04, 0.07, 0.07)), tmp_path)


@pytest.mark.parametrize("below", [0, 2, 3])
def test_an_outer_surface_on_or_past_the_upper_edge_is_refused(below, tmp_path):
    top = 0.1 - below * 0.002
    with pytest.raises(ValueError, match=r"does not along z"):
        _build(_scene((0.03, 0.03, top - 0.04), (0.07, 0.07, top)), tmp_path)


def test_one_cell_inside_is_enough(tmp_path):
    # Outer Surface at cell 1 below and cell 49 above (of 0..50): built
    _build(_scene((0.008, 0.03, 0.03), (0.092, 0.07, 0.07)), tmp_path)


def test_the_separation_counts(tmp_path):
    # the same subgrid with a 1-cell separation is well inside
    _build(_scene((0.004, 0.03, 0.03), (0.04, 0.07, 0.07), is_os_sep=1), tmp_path)
    with pytest.raises(ValueError, match=r"3 main grid cells outside"):
        _build(_scene((0.004, 0.03, 0.03), (0.04, 0.07, 0.07)), tmp_path)
