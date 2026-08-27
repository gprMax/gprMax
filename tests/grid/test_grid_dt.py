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

"""Time-step and current-calculation tests for ``FDTDGrid``.

``calculate_dt`` is the single most consequential line in the grid: choose a
time step above the Courant-Friedrichs-Lewy (CFL) limit and the simulation
does not degrade gracefully, it diverges. The limit is a closed-form
expression, so every branch is assertable exactly.

The 2D TM modes drop the invariant axis from the sum, which is the whole
difference between the four branches.
"""

import decimal

import numpy as np
import pytest
from scipy.constants import c as C_LIGHT

from .conftest import DL, DL_ANISO


def cfl_limit(*spacings):
    """The exact CFL time step for the given axis spacings."""
    return 1 / (C_LIGHT * np.sqrt(sum(1 / s**2 for s in spacings)))


class TestCalculateDt3D:
    def test_matches_the_closed_form(self, make_grid):
        g = make_grid(dl=DL_ANISO, arrays=False)
        g.calculate_dt()
        assert g.dt == pytest.approx(cfl_limit(*DL_ANISO))

    def test_isotropic_grid(self, make_grid):
        g = make_grid(dl=DL, arrays=False)
        g.calculate_dt()
        assert g.dt == pytest.approx(DL / (C_LIGHT * np.sqrt(3)))

    def test_never_exceeds_the_cfl_limit(self, make_grid):
        """The rounding is ``ROUND_FLOOR`` precisely so the stored value can
        never land above the limit through binary representation.
        """
        g = make_grid(dl=DL_ANISO, arrays=False)
        g.calculate_dt()
        assert g.dt <= cfl_limit(*DL_ANISO)

    @pytest.mark.parametrize(
        "dl",
        [
            (0.001, 0.001, 0.001),
            (0.002, 0.001, 0.001),
            (0.01, 0.02, 0.04),
            (0.0005, 0.0005, 0.002),
        ],
    )
    def test_stays_within_the_limit_for_various_spacings(self, make_grid, dl):
        g = make_grid(dl=dl, arrays=False)
        g.calculate_dt()
        assert g.dt <= cfl_limit(*dl)
        assert g.dt == pytest.approx(cfl_limit(*dl))

    def test_finer_grid_gives_a_smaller_time_step(self, make_grid):
        coarse = make_grid(dl=0.002, arrays=False)
        fine = make_grid(dl=0.001, arrays=False)
        coarse.calculate_dt()
        fine.calculate_dt()
        assert fine.dt < coarse.dt

    def test_halving_the_spacing_halves_the_time_step(self, make_grid):
        """The limit is linear in the spacing for an isotropic grid."""
        coarse = make_grid(dl=0.002, arrays=False)
        fine = make_grid(dl=0.001, arrays=False)
        coarse.calculate_dt()
        fine.calculate_dt()
        assert fine.dt == pytest.approx(coarse.dt / 2)

    def test_rounded_to_one_less_than_hardware_precision(self, make_grid):
        """``decimal.getcontext().prec - 1`` decimal places."""
        g = make_grid(dl=DL_ANISO, arrays=False)
        g.calculate_dt()
        places = decimal.getcontext().prec - 1
        requantised = decimal.Decimal(g.dt).quantize(
            decimal.Decimal(f"1.{'0' * places}"), rounding=decimal.ROUND_FLOOR
        )
        assert float(requantised) == g.dt


class TestCalculateDt2D:
    """Each 2D TM mode drops its invariant axis from the sum."""

    def test_tmx_uses_y_and_z(self, make_grid, grid_config):
        grid_config.model_config.mode = "2D TMx"
        g = make_grid(dl=DL_ANISO, arrays=False)
        g.calculate_dt()
        assert g.dt == pytest.approx(cfl_limit(DL_ANISO[1], DL_ANISO[2]))

    def test_tmy_uses_x_and_z(self, make_grid, grid_config):
        grid_config.model_config.mode = "2D TMy"
        g = make_grid(dl=DL_ANISO, arrays=False)
        g.calculate_dt()
        assert g.dt == pytest.approx(cfl_limit(DL_ANISO[0], DL_ANISO[2]))

    def test_tmz_uses_x_and_y(self, make_grid, grid_config):
        grid_config.model_config.mode = "2D TMz"
        g = make_grid(dl=DL_ANISO, arrays=False)
        g.calculate_dt()
        assert g.dt == pytest.approx(cfl_limit(DL_ANISO[0], DL_ANISO[1]))

    @pytest.mark.parametrize("mode", ["2D TMx", "2D TMy", "2D TMz"])
    def test_2d_time_step_is_larger_than_3d(self, make_grid, grid_config, mode):
        """Dropping a term from the sum under the square root always relaxes
        the limit — a 2D model may take longer steps than a 3D one.
        """
        three_d = make_grid(dl=DL_ANISO, arrays=False)
        three_d.calculate_dt()

        grid_config.model_config.mode = mode
        two_d = make_grid(dl=DL_ANISO, arrays=False)
        two_d.calculate_dt()

        assert two_d.dt > three_d.dt

    def test_unknown_mode_falls_back_to_3d(self, make_grid, grid_config):
        """The final ``else`` catches anything that is not a named 2D mode."""
        grid_config.model_config.mode = "something else"
        g = make_grid(dl=DL_ANISO, arrays=False)
        g.calculate_dt()
        assert g.dt == pytest.approx(cfl_limit(*DL_ANISO))


class TestCalculateCurrents:
    """``calculate_Ix/Iy/Iz`` sum the magnetic field around a Yee contour.

    Each has a guard returning exactly zero on the low faces, where the
    contour would need cells at index -1.
    """

    @pytest.fixture
    def current_grid(self, make_grid):
        """A grid with deterministic, distinguishable H fields.

        Each component is filled with a different constant multiple of a
        linear ramp so that a test reading the wrong component or the wrong
        neighbour cannot coincidentally agree.
        """
        g = make_grid(nx=6, ny=6, nz=6, dl=DL_ANISO)
        shape = g.Hx.shape
        ramp = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        g.Hx = ramp.copy()
        g.Hy = (ramp * 2).copy()
        g.Hz = (ramp * 3).copy()
        return g

    @pytest.mark.parametrize("point", [(3, 0, 3), (3, 3, 0), (3, 0, 0)])
    def test_ix_is_zero_on_the_low_y_or_z_faces(self, current_grid, point):
        assert current_grid.calculate_Ix(*point) == 0

    @pytest.mark.parametrize("point", [(0, 3, 3), (3, 3, 0), (0, 3, 0)])
    def test_iy_is_zero_on_the_low_x_or_z_faces(self, current_grid, point):
        assert current_grid.calculate_Iy(*point) == 0

    @pytest.mark.parametrize("point", [(0, 3, 3), (3, 0, 3), (0, 0, 3)])
    def test_iz_is_zero_on_the_low_x_or_y_faces(self, current_grid, point):
        assert current_grid.calculate_Iz(*point) == 0

    def test_ix_contour_sum(self, current_grid):
        g = current_grid
        x, y, z = 3, 3, 3
        expected = g.dy * (g.Hy[x, y, z - 1] - g.Hy[x, y, z]) + g.dz * (
            g.Hz[x, y, z] - g.Hz[x, y - 1, z]
        )
        assert g.calculate_Ix(x, y, z) == pytest.approx(expected)

    def test_iy_contour_sum(self, current_grid):
        g = current_grid
        x, y, z = 3, 3, 3
        expected = g.dx * (g.Hx[x, y, z] - g.Hx[x, y, z - 1]) + g.dz * (
            g.Hz[x - 1, y, z] - g.Hz[x, y, z]
        )
        assert g.calculate_Iy(x, y, z) == pytest.approx(expected)

    def test_iz_contour_sum(self, current_grid):
        g = current_grid
        x, y, z = 3, 3, 3
        expected = g.dx * (g.Hx[x, y - 1, z] - g.Hx[x, y, z]) + g.dy * (
            g.Hy[x, y, z] - g.Hy[x - 1, y, z]
        )
        assert g.calculate_Iz(x, y, z) == pytest.approx(expected)

    def test_uniform_field_gives_zero_current(self, make_grid):
        """A spatially constant H has no curl, so every contour sum cancels."""
        g = make_grid(nx=6, ny=6, nz=6, dl=DL_ANISO)
        g.Hx[:] = 1.0
        g.Hy[:] = 1.0
        g.Hz[:] = 1.0
        assert g.calculate_Ix(3, 3, 3) == pytest.approx(0.0)
        assert g.calculate_Iy(3, 3, 3) == pytest.approx(0.0)
        assert g.calculate_Iz(3, 3, 3) == pytest.approx(0.0)

    def test_zero_field_gives_zero_current(self, make_grid):
        g = make_grid(nx=6, ny=6, nz=6, dl=DL_ANISO)
        assert g.calculate_Ix(3, 3, 3) == 0.0
        assert g.calculate_Iy(3, 3, 3) == 0.0
        assert g.calculate_Iz(3, 3, 3) == 0.0

    def test_current_scales_with_field_magnitude(self, make_grid):
        """The contour sum is linear in H."""
        g = make_grid(nx=6, ny=6, nz=6, dl=DL_ANISO)
        shape = g.Hx.shape
        ramp = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        g.Hy = ramp.copy()
        g.Hz = ramp.copy()
        single = g.calculate_Ix(3, 3, 3)

        g.Hy = (ramp * 2).copy()
        g.Hz = (ramp * 2).copy()
        assert g.calculate_Ix(3, 3, 3) == pytest.approx(2 * single)

    def test_uses_the_matching_axes_of_dl(self, make_grid):
        """``Ix`` is weighted by ``dy`` and ``dz`` — never ``dx``. With
        anisotropic spacing, swapping any of them changes the answer.
        """
        g = make_grid(nx=6, ny=6, nz=6, dl=DL_ANISO)
        g.Hy[3, 3, 2] = 1.0
        g.Hz[3, 3, 3] = 1.0
        # dy * (Hy[x,y,z-1] - Hy[x,y,z]) + dz * (Hz[x,y,z] - Hz[x,y-1,z])
        assert g.calculate_Ix(3, 3, 3) == pytest.approx(DL_ANISO[1] * 1.0 + DL_ANISO[2] * 1.0)


pytestmark = pytest.mark.unit
