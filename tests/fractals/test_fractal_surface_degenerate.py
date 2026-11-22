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

"""Regression tests for degenerate (single-cell) fractal-surface footprints.

A surface that discretises to a single cell in both in-plane directions
has a one-sample spectrum: zeroing the DC component zeroes the entire
field, so the min/max rescale in ``generate_fractal_surface()`` divided
by zero and crashed with a raw ``FloatingPointError`` (numpy is set to
``divide='raise'`` in this module). A single cell in one direction only
(a 1-cell-wide strip) still has spectral variation along its long axis
and has always been fine.

Both layers are pinned here:

- ``FractalSurface._reject_single_sample_surface()`` - the shared guard
  the serial, TE-shadow and MPI generators all call;
- ``AddSurfaceRoughness`` / ``AddGrass`` ``build()`` - the
  message-first check that rejects the footprint before the surface
  object is even created.
"""

import pytest

from gprMax.fractals.fractal_surface import FractalSurface
from gprMax.user_objects.cmds_geometry.add_grass import AddGrass
from gprMax.user_objects.cmds_geometry.add_surface_roughness import AddSurfaceRoughness

from .conftest import add_mixing_model
from .test_surface_modifiers import GRASS_KWARGS, ROUGHNESS_KWARGS, make_boxed

pytestmark = pytest.mark.unit


@pytest.fixture
def boxed_grid(fractal_grid):
    """A grid with a fractal box already registered (pass 1 only); same
    construction as test_surface_modifiers.boxed_grid (the fixture lives
    in that module, so rebuild it here rather than importing it)."""
    g = fractal_grid()
    add_mixing_model(g)
    return make_boxed(g)


class TestFractalSurfaceGuard:
    def test_a_1x1_cell_surface_is_rejected(self):
        surface = FractalSurface(0, 0, 0, 1, 0, 1, 1.5, 42)
        with pytest.raises(ValueError, match="at least 2 cells"):
            surface.generate_fractal_surface()

    @pytest.mark.parametrize(
        "bounds",
        [
            (0, 0, 0, 1, 0, 8),  # x-plane, 1 x 8 strip
            (0, 0, 0, 8, 0, 1),  # x-plane, 8 x 1 strip
            (0, 1, 0, 0, 0, 8),  # y-plane, 1 x 8 strip
            (0, 8, 0, 0, 0, 1),  # y-plane, 8 x 1 strip
            (0, 8, 0, 1, 1, 1),  # z-plane, 8 x 1 strip
        ],
    )
    def test_a_1_cell_wide_strip_still_generates(self, bounds):
        xs, xf, ys, yf, zs, zf = bounds
        surface = FractalSurface(xs, xf, ys, yf, zs, zf, 1.5, 42)
        assert surface.generate_fractal_surface() is True

    def test_a_1x1_surface_on_every_plane_is_rejected(self):
        # The guard must cover all three surface orientations. Each bounds
        # tuple is (xs, xf, ys, yf, zs, zf) with exactly one flat axis and
        # one-cell extents on both in-plane axes.
        for bounds in [(0, 0, 2, 3, 4, 5), (2, 3, 0, 0, 4, 5), (2, 3, 4, 5, 0, 0)]:
            surface = FractalSurface(*bounds, 1.5, 42)
            with pytest.raises(ValueError, match="at least 2 cells"):
                surface.generate_fractal_surface()

    def test_the_message_names_the_requested_size(self):
        surface = FractalSurface(0, 0, 0, 1, 0, 1, 1.5, 42)
        with pytest.raises(ValueError, match="1 x 1 cells"):
            surface.generate_fractal_surface()

    def test_a_2x2_surface_still_generates(self):
        surface = FractalSurface(0, 0, 0, 2, 0, 2, 1.5, 42)
        assert surface.generate_fractal_surface() is True


class TestAddSurfaceRoughnessGuard:
    def test_a_1x1_cell_footprint_is_rejected_with_a_geometry_message(self, boxed_grid):
        kwargs = {
            **ROUGHNESS_KWARGS,
            "p1": (0.002, 0.002, 0.010),
            "p2": (0.003, 0.003, 0.010),
        }
        with pytest.raises(ValueError, match="at least 2 cells") as excinfo:
            AddSurfaceRoughness(**kwargs).build(boxed_grid)
        # The command-level check must fire (message carries the command's
        # own __str__), not the generator's backstop - otherwise the user
        # gets a raw "Fractal surface must..." instead of which command in
        # their input is at fault.
        assert "#add_surface_roughness:" in str(excinfo.value)

    def test_a_footprint_that_rounds_down_to_1x1_is_rejected(self, boxed_grid):
        # 1.4 cells rounds down to 1 cell (ROUND_HALF_DOWN at 0.5 only).
        kwargs = {
            **ROUGHNESS_KWARGS,
            "p1": (0.002, 0.002, 0.010),
            "p2": (0.0034, 0.0034, 0.010),
        }
        with pytest.raises(ValueError, match="at least 2 cells"):
            AddSurfaceRoughness(**kwargs).build(boxed_grid)

    def test_a_1x1_footprint_leaves_no_surface_attached(self, boxed_grid):
        kwargs = {
            **ROUGHNESS_KWARGS,
            "p1": (0.002, 0.002, 0.010),
            "p2": (0.003, 0.003, 0.010),
        }
        with pytest.raises(ValueError):
            AddSurfaceRoughness(**kwargs).build(boxed_grid)
        assert boxed_grid.fractalvolumes[0].fractalsurfaces == []

    @pytest.mark.parametrize(
        "p2",
        [
            (0.004, 0.010, 0.010),  # 2 x 8 strip
            (0.010, 0.004, 0.010),  # 8 x 2 strip
            (0.003, 0.010, 0.010),  # 1 x 8 strip (accepted: varies in y)
        ],
    )
    def test_1_cell_wide_strips_are_still_accepted(self, boxed_grid, p2):
        kwargs = {**ROUGHNESS_KWARGS, "p1": (0.002, 0.002, 0.010), "p2": p2}
        AddSurfaceRoughness(**kwargs).build(boxed_grid)
        assert len(boxed_grid.fractalvolumes[0].fractalsurfaces) == 1


class TestAddGrassGuard:
    def test_a_1x1_cell_footprint_is_rejected_with_a_geometry_message(self, boxed_grid):
        kwargs = {
            **GRASS_KWARGS,
            "p1": (0.002, 0.002, 0.010),
            "p2": (0.003, 0.003, 0.010),
            "n_blades": 1,
        }
        with pytest.raises(ValueError, match="at least 2 cells") as excinfo:
            AddGrass(**kwargs).build(boxed_grid)
        # The command-level check must fire (message carries the command's
        # own __str__), not the generator's backstop.
        assert "#add_grass:" in str(excinfo.value)

    def test_a_1x8_strip_is_still_accepted(self, boxed_grid):
        kwargs = {
            **GRASS_KWARGS,
            "p1": (0.002, 0.002, 0.010),
            "p2": (0.003, 0.010, 0.010),
            "n_blades": 2,
        }
        AddGrass(**kwargs).build(boxed_grid)
        assert len(boxed_grid.fractalvolumes[0].fractalsurfaces) == 1
