# Copyright (C) 2015-2019: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
from scipy import interpolate
import sys


def calculate_weighting_coefficients(x1, x):
    c1 = (x - x1) / x
    c2 = x1 / x
    return (c1, c2)


class PrecusorNodes2dBase(object):

    def __init__(self, fdtd_grid, sub_grid):
        self.G = fdtd_grid
        self.ratio = sub_grid.ratio
        self.nwx = sub_grid.nwx
        self.nwy = sub_grid.nwy
        self.sub_grid = sub_grid
        self.interpolation = sub_grid.interpolation

        self.Hx = fdtd_grid.Hx
        self.Hy = fdtd_grid.Hy
        self.Hz = fdtd_grid.Hz
        self.Ex = fdtd_grid.Ex
        self.Ey = fdtd_grid.Ey
        self.Ez = fdtd_grid.Ez

        # Main grid indices of subgrids
        self.i0 = sub_grid.i0
        self.j0 = sub_grid.j0
        self.k0 = sub_grid.k0
        self.i1 = sub_grid.i1
        self.j1 = sub_grid.j1
        self.k1 = sub_grid.k1

        # dl / 2 sub cell
        self.d = 1 / (2 * self.ratio)

        self._initialize_fields()
        self._initialize_field_names()

        self.l_weight = self.ratio // 2
        self.r_weight = self.ratio - self.l_weight
        #self.l_weight = 1
        #self.r_weight = 2

    def _initialize_fields(self):
        print('dont call me')
        sys.exit()
        pass

    def _initialize_field_names(self):
        print('dont call me')
        sys.exit()
        pass

    def interpolate_magnetic_in_time(self, m):
        self.weight_pre_and_current_fields(m, self.fn_m)

    def interpolate_electric_in_time(self, m):
        self.weight_pre_and_current_fields(m, self.fn_e)

    def weight_pre_and_current_fields(self, m, field_names):
        c1, c2 = calculate_weighting_coefficients(m, self.ratio)

        for f in field_names:
            try:
                val = c1 * getattr(self, f + '_0') + c2 * getattr(self, f + '_1')
            except ValueError:
                print(self.ex_front_0.shape)
                print(self.ex_front_1.shape)
                raise Exception(f)
            setattr(self, f, val)

    def calc_exact_field(self, field_names):
        """
            Function to set the fields used in update calculations to the
            values at the current main time step.
            i.e. ey_left = copy.ey_left_1
        """
        for f in field_names:
            val = np.copy(getattr(self, f + '_1'))
            setattr(self, f, val)

    def calc_exact_magnetic_in_time(self):
        self.calc_exact_field(self.fn_m)

    def calc_exact_electric_in_time(self):
        self.calc_exact_field(self.fn_e)


class PrecursorNodes2dTM(PrecusorNodes2dBase):

    def __init__(self, fdtd_grid, sub_grid):
        super().__init__(fdtd_grid, sub_grid)

    def _initialize_fields(self):

        # H precursors
        self.hx_front_0 = np.zeros(self.nwx + 1)
        self.hx_front_1 = np.zeros(self.nwx + 1)

        self.hx_back_0 = np.zeros(self.nwx + 1)
        self.hx_back_1 = np.zeros(self.nwx + 1)

        self.hy_left_0 = np.zeros(self.nwy + 1)
        self.hy_left_1 = np.zeros(self.nwy + 1)

        self.hy_right_0 = np.zeros(self.nwy + 1)
        self.hy_right_1 = np.zeros(self.nwy + 1)

        # E precursors
        self.ez_front_0 = np.zeros(self.nwx + 1)
        self.ez_front_1 = np.zeros(self.nwx + 1)

        self.ez_back_0 = np.zeros(self.nwx + 1)
        self.ez_back_1 = np.zeros(self.nwx + 1)

        self.ez_left_0 = np.zeros(self.nwy + 1)
        self.ez_left_1 = np.zeros(self.nwy + 1)

        self.ez_right_0 = np.zeros(self.nwy + 1)
        self.ez_right_1 = np.zeros(self.nwy + 1)

    def _initialize_field_names(self):
        self.fn_m = ['hy_left', 'hy_right', 'hx_back', 'hx_front']
        self.fn_e = ['ez_left', 'ez_right', 'ez_back', 'ez_front']

    def interpolate_across_sub_cells(self, y, n0, n1):
        n = n1 - n0
        x = np.arange(0, y.shape[0], 1.0)
        x_new = np.linspace(1, y.shape[0] - 2, n * self.ratio + 1)
        f = interpolate.interp1d(x, y, kind='linear')
        y_new = f(x_new)
        """plt.plot(x, y, 'x', x_new, y_new, '.')
        plt.show()
        sys.exit()"""
        return y_new

    def update_electric(self):
        # f = np.sin(np.arange(0, 13))

        # line of Ez nodes along the left edge of the IS
        self.ez_left_0 = np.copy(self.ez_left_1)
        # interpolate nodes between two Ez nodes 1 cell beyond/infront the IS
        f = self.Ez[self.i0, self.j0 - 1:self.j1 + 2, 0]
        self.ez_left_1 = self.interpolate_across_sub_cells(f, self.j0, self.j1)

        self.ez_right_0 = np.copy(self.ez_right_1)
        f = self.Ez[self.i1, self.j0 - 1:self.j1 + 2, 0]
        self.ez_right_1 = self.interpolate_across_sub_cells(f, self.j0, self.j1)

        self.ez_front_0 = np.copy(self.ez_front_1)
        f = self.Ez[self.i0 - 1:self.i1 + 2, self.j0, 0]
        self.ez_front_1 = self.interpolate_across_sub_cells(f, self.i0, self.i1)

        self.ez_back_0 = np.copy(self.ez_back_1)
        f = self.Ez[self.i0 - 1:self.i1 + 2, self.j1, 0]
        self.ez_back_1 = self.interpolate_across_sub_cells(f, self.i0, self.i1)

    def update_hy_left_1(self):

        self.hy_left_0 = np.copy(self.hy_left_1)
        c1, c2 = calculate_weighting_coefficients(self.l_weight, self.ratio)
        l = self.Hy[self.i0 - 1, self.j0 - 1:self.j1 + 2, 0]
        r = self.Hy[self.i0, self.j0 - 1:self.j1 + 2, 0]

        # Tranverse interpolated hz
        hz_t_i = c1 * l + c2 * r
        # Tangential interpolated hz
        self.hy_left_1 = self.interpolate_across_sub_cells(hz_t_i, self.j0, self.j1)

    def update_hy_right_1(self):

        self.hy_right_0 = np.copy(self.hy_right_1)
        c1, c2 = calculate_weighting_coefficients(self.r_weight, self.ratio)
        l = self.Hy[self.i1 - 1, self.j0 - 1:self.j1 + 2, 0]
        r = self.Hy[self.i1, self.j0 - 1:self.j1 + 2, 0]
        hz_t_i = c1 * l + c2 * r
        self.hy_right_1 = self.interpolate_across_sub_cells(hz_t_i, self.j0, self.j1)

    def update_hx_back_1(self):
        self.hx_back_0 = np.copy(self.hx_back_1)
        c1, c2 = calculate_weighting_coefficients(self.r_weight, self.ratio)
        l = self.Hx[self.i0 - 1:self.i1 + 2, self.j1 - 1, 0]
        r = self.Hx[self.i0 - 1:self.i1 + 2, self.j1, 0]
        hz_t_i = c1 * l + c2 * r
        self.hx_back_1 = self.interpolate_across_sub_cells(hz_t_i, self.i0, self.i1)

    def update_hx_front_1(self):

        self.hx_front_0 = np.copy(self.hx_front_1)
        c1, c2 = calculate_weighting_coefficients(self.l_weight, self.ratio)
        l = self.Hx[self.i0 - 1:self.i1 + 2, self.j0 - 1, 0]
        r = self.Hx[self.i0 - 1:self.i1 + 2, self.j0, 0]
        hz_t_i = c1 * l + c2 * r
        self.hx_front_1 = self.interpolate_across_sub_cells(hz_t_i, self.i0, self.i1)

    def update_magnetic(self):
        self.update_hy_left_1()
        self.update_hy_right_1()
        self.update_hx_back_1()
        self.update_hx_front_1()


class PrecursorNodes2d(PrecusorNodes2dBase):

    def __init__(self, fdtd_grid, sub_grid):
        super().__init__(fdtd_grid, sub_grid)

    def _initialize_fields(self):
        # E precursors
        self.ex_front_0 = np.zeros(self.nwx)
        self.ex_front_1 = np.zeros(self.nwx)

        self.ex_back_0 = np.zeros(self.nwx)
        self.ex_back_1 = np.zeros(self.nwx)

        self.ey_left_0 = np.zeros(self.nwy)
        self.ey_left_1 = np.zeros(self.nwy)

        self.ey_right_0 = np.zeros(self.nwy)
        self.ey_right_1 = np.zeros(self.nwy)

        # H precursors
        self.hz_front_0 = np.zeros(self.nwx)
        self.hz_front_1 = np.zeros(self.nwx)

        self.hz_back_0 = np.zeros(self.nwx)
        self.hz_back_1 = np.zeros(self.nwx)

        self.hz_left_0 = np.zeros(self.nwy)
        self.hz_left_1 = np.zeros(self.nwy)

        self.hz_right_0 = np.zeros(self.nwy)
        self.hz_right_1 = np.zeros(self.nwy)

    def _initialize_field_names(self):
        # Field names
        self.fn_m = ['hz_left', 'hz_right', 'hz_back', 'hz_front']
        self.fn_e = ['ey_left', 'ey_right', 'ex_back', 'ex_front']

    def update_hz_left_1(self):
        self.hz_left_0 = np.copy(self.hz_left_1)
        c1, c2 = calculate_weighting_coefficients(1, self.ratio)
        l = self.Hz[self.i0 - 1, self.j0 - 1:self.j1 + 1, 1]
        r = self.Hz[self.i0, self.j0 - 1:self.j1 + 1, 1]

        # Tranverse interpolated hz
        hz_t_i = c1 * l + c2 * r
        # Tangential interpolated hz
        self.hz_left_1 = self.interpolate_across_sub_cells(hz_t_i, self.j0, self.j1)

    def update_hz_right_1(self):

        self.hz_right_0 = np.copy(self.hz_right_1)
        c1, c2 = calculate_weighting_coefficients(2, self.ratio)
        l = self.Hz[self.i1 - 1, self.j0 - 1:self.j1 + 1, 1]
        r = self.Hz[self.i1, self.j0 - 1:self.j1 + 1, 1]
        hz_t_i = c1 * l + c2 * r
        self.hz_right_1 = self.interpolate_across_sub_cells(hz_t_i, self.j0, self.j1)

    def update_hz_back_1(self):
        self.hz_back_0 = np.copy(self.hz_back_1)
        c1, c2 = calculate_weighting_coefficients(2, self.ratio)
        l = self.Hz[self.i0 - 1:self.i1 + 1, self.j1 - 1, 1]
        r = self.Hz[self.i0 - 1:self.i1 + 1, self.j1, 1]
        hz_t_i = c1 * l + c2 * r
        self.hz_back_1 = self.interpolate_across_sub_cells(hz_t_i, self.i0, self.i1)

    def update_hz_front_1(self):
        self.hz_front_0 = np.copy(self.hz_front_1)
        c1, c2 = calculate_weighting_coefficients(1, self.ratio)
        l = self.Hz[self.i0 - 1:self.i1 + 1, self.j0 - 1, 1]
        r = self.Hz[self.i0 - 1:self.i1 + 1, self.j0, 1]
        hz_t_i = c1 * l + c2 * r
        self.hz_front_1 = self.interpolate_across_sub_cells(hz_t_i, self.i0, self.i1)

    def update_magnetic(self):
        self.update_hz_left_1()
        self.update_hz_right_1()
        self.update_hz_back_1()
        self.update_hz_front_1()

    def update_electric(self):

        """Function to calculate the precursor nodes at the next main
            grid time-step
        """

        # LEFT BOUNDARY
        # Save the last precursor node values - these will be used interpolate
        # field values at sub grid time step values.
        self.ey_left_0 = np.copy(self.ey_left_1)
        # line of Ex nodes along the left edge of the IS
        f = self.Ey[self.i0, self.j0 - 1:self.j1 + 1, 1]
        self.ey_left_1 = self.interpolate_across_sub_cells(f, self.j0, self.j1)

        # RIGHT BOUNDARY
        self.ey_right_0 = np.copy(self.ey_right_1)
        f = self.Ey[self.i1, self.j0 - 1:self.j1 + 1, 1]
        self.ey_right_1 = self.interpolate_across_sub_cells(f, self.j0, self.j1)

        # BACK BOUNDARY
        self.ex_back_0 = np.copy(self.ex_back_1)
        f = self.Ex[self.i0 - 1:self.i1 + 1, self.j1, 1]
        self.ex_back_1 = self.interpolate_across_sub_cells(f, self.i0, self.i1)

        # FRONT BOUNDARY
        self.ex_front_0 = np.copy(self.ex_front_1)
        f = self.Ex[self.i0 - 1:self.i1 + 1, self.j0, 1]
        self.ex_front_1 = self.interpolate_across_sub_cells(f, self.i0, self.i1)

    def interpolate_across_sub_cells(self, y, n0, n1):
        n = n1 - n0
        x = np.arange(0.5, y.shape[0], 1.0)
        x_new = np.linspace(1 + self.d, (1 + n) - self.d, n * self.ratio)
        f = interpolate.interp1d(x, y, kind='linear')
        y_new = f(x_new)

        return y_new


class PrecursorNodes(PrecusorNodes2dBase):

    def __init__(self, fdtd_grid, sub_grid):

        self.nwz = sub_grid.nwz
        super().__init__(fdtd_grid, sub_grid)
        self.initialize_magnetic_slices_array()
        self.initialize_electric_slices_array()

    def build_coefficient_matrix(self, update_coefficients, lookup_id, inc_field, name):
        """Function builds a 2d matrix of update coefficients for each face and field.
            This allows the is nodes to be updated via slicing rather than iterating which is much faster
        """
        pass

    def _initialize_fields(self):

        # Initialise the precursor arrays

        # The precursors are divided up into the 6. Each represent 1
        # face of a huygens cube surface. We represent each face as a 2d array
        # containing a field in a particular direction.

        # _1 are the fields at the current main grid timestep
        # _0 are the fields at the previous main grid timestep
        # We store both fields so we can do different interpolations between
        # them on the fly.

        # Front face
        self.ex_front_1 = np.zeros((self.nwx, self.nwz + 1))
        self.ex_front_0 = np.copy(self.ex_front_1)
        self.ez_front_1 = np.zeros((self.nwx + 1, self.nwz))
        self.ez_front_0 = np.copy(self.ez_front_1)

        # The same as the opposite face
        self.ex_back_1 = np.copy(self.ex_front_1)
        self.ex_back_0 = np.copy(self.ex_front_1)
        self.ez_back_1 = np.copy(self.ez_front_1)
        self.ez_back_0 = np.copy(self.ez_front_1)

        self.ey_left_1 = np.zeros((self.nwy, self.nwz + 1))
        self.ey_left_0 = np.copy(self.ey_left_1)
        self.ez_left_1 = np.zeros((self.nwy + 1, self.nwz))
        self.ez_left_0 = np.copy(self.ez_left_1)

        self.ey_right_1 = np.copy(self.ey_left_1)
        self.ey_right_0 = np.copy(self.ey_left_1)
        self.ez_right_1 = np.copy(self.ez_left_1)
        self.ez_right_0 = np.copy(self.ez_left_1)

        self.ex_bottom_1 = np.zeros((self.nwx, self.nwy + 1))
        self.ex_bottom_0 = np.copy(self.ex_bottom_1)
        self.ey_bottom_1 = np.zeros((self.nwx + 1, self.nwy))
        self.ey_bottom_0 = np.copy(self.ey_bottom_1)

        self.ex_top_1 = np.copy(self.ex_bottom_1)
        self.ex_top_0 = np.copy(self.ex_bottom_1)
        self.ey_top_1 = np.copy(self.ey_bottom_1)
        self.ey_top_0 = np.copy(self.ey_bottom_1)

        # Initialize the H precursor fields
        self.hx_front_1 = np.copy(self.ez_front_1)
        self.hx_front_0 = np.copy(self.ez_front_1)
        self.hz_front_1 = np.copy(self.ex_front_1)
        self.hz_front_0 = np.copy(self.ex_front_1)

        self.hx_back_1 = np.copy(self.hx_front_1)
        self.hx_back_0 = np.copy(self.hx_front_1)
        self.hz_back_1 = np.copy(self.hz_front_1)
        self.hz_back_0 = np.copy(self.hz_front_1)

        self.hy_left_1 = np.copy(self.ez_left_1)
        self.hy_left_0 = np.copy(self.ez_left_1)
        self.hz_left_1 = np.copy(self.ey_left_1)
        self.hz_left_0 = np.copy(self.ey_left_1)

        self.hy_right_1 = np.copy(self.hy_left_1)
        self.hy_right_0 = np.copy(self.hy_left_1)
        self.hz_right_1 = np.copy(self.hz_left_1)
        self.hz_right_0 = np.copy(self.hz_left_1)

        self.hx_top_1 = np.copy(self.ey_top_1)
        self.hx_top_0 = np.copy(self.ey_top_1)
        self.hy_top_1 = np.copy(self.ex_top_1)
        self.hy_top_0 = np.copy(self.ex_top_1)

        self.hx_bottom_1 = np.copy(self.hx_top_1)
        self.hx_bottom_0 = np.copy(self.hx_top_1)
        self.hy_bottom_1 = np.copy(self.hy_top_1)
        self.hy_bottom_0 = np.copy(self.hy_top_1)

    def _initialize_field_names(self):

        self.fn_m = [
            'hx_front', 'hz_front',
            'hx_back', 'hz_back',
            'hy_left', 'hz_left',
            'hy_right', 'hz_right',
            'hx_top', 'hy_top',
            'hx_bottom', 'hy_bottom'
        ]

        self.fn_e = [
            'ex_front', 'ez_front',
            'ex_back', 'ez_back',
            'ey_left', 'ez_left',
            'ey_right', 'ez_right',
            'ex_top', 'ey_top',
            'ex_bottom', 'ey_bottom'
        ]

    def update_previous_timestep_fields(self, field_names):
        for fn in field_names:
            val = getattr(self, fn + '_1')
            val_c = np.copy(val)
            setattr(self, fn + '_0', val_c)

    def initialize_magnetic_slices_array(self):

        # Array contains the indices at which the main grid should be sliced
        # to obtain the 2 2d array of H nodes required for interpolation
        # across the IS boundary for each h field on each face of the subgrid

        # Extend the surface so that the outer fields can be interpolated
        # more accurately
        #i0 = self.i0 - 1
        #j0 = self.j0 - 1
        #k0 = self.k0 - 1
        #i1 = self.i1 + 1
        #j1 = self.j1 + 1
        #k1 = self.k1 + 1

        # not extended
        i0 = self.i0
        j0 = self.j0
        k0 = self.k0
        i1 = self.i1
        j1 = self.j1
        k1 = self.k1

        slices = [
            ['hy_left_1', False,
                (self.i0 - 2, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i0 - 1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i0, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i0 + 1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)), self.Hy],
            ['hy_right_1', False,
                (self.i1 - 2, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i1 - 1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i1 + 1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)), self.Hy],
            ['hz_left_1', True,
                (self.i0 - 2, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i0 - 1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i0, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i0 + 1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)), self.Hz],
            ['hz_right_1', True,
                (self.i1 - 2, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i1 - 1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i1 + 1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)), self.Hz],
            ['hx_front_1', False,
                (slice(i0, i1 + 1, 1), self.j0 - 2, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j0 - 1, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j0, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j0 + 1, slice(k0, k1, 1)), self.Hx],
            ['hx_back_1', False,
                (slice(i0, i1 + 1, 1), self.j1 - 2, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j1 - 1, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j1, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j1 + 1, slice(k0, k1, 1)), self.Hx],
            ['hz_front_1', True,
                (slice(i0, i1, 1), self.j0 - 2, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j0 - 1, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j0, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j0 + 1, slice(k0, k1 + 1, 1)), self.Hz],
            ['hz_back_1', True,
                (slice(i0, i1, 1), self.j1 - 2, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j1 - 1, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j1, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j1 + 1, slice(k0, k1 + 1, 1)), self.Hz],
            ['hx_bottom_1', False,
                # check these indexes
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k0 - 2),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k0 - 1),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k0),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k0 + 1), self.Hx],
            ['hx_top_1', False,
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k1 - 2),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k1 - 1),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k1),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k1 + 1), self.Hx],
            ['hy_bottom_1', True,
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k0 - 2),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k0 - 1),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k0),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k0 + 1), self.Hy],
            ['hy_top_1', True,
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k1 - 2),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k1 - 1),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k1),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k1 + 1), self.Hy]
        ]

        for obj in slices:
            sliced_field = obj[-1][obj[2]]
            obj[1] = self.create_interpolated_coords(obj[1], sliced_field)

        self.magnetic_slices = slices

    def initialize_electric_slices_array(self):
        # Extend the region sliced from the main grid by 1 cell.
        # this allows more accurate interpolation for the outernodes
        #i0 = self.i0 - 1
        #j0 = self.j0 - 1
        #k0 = self.k0 - 1
        #i1 = self.i1 + 1
        #j1 = self.j1 + 1
        #k1 = self.k1 + 1

        # not extended
        i0 = self.i0
        j0 = self.j0
        k0 = self.k0
        i1 = self.i1
        j1 = self.j1
        k1 = self.k1

        # Spatially interpolate nodes
        slices = [
            ['ex_front_1', True,
                (slice(i0, i1, 1), self.j0 - 1, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j0, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j0 + 1, slice(k0, k1 + 1, 1)),
                self.Ex],
            ['ex_back_1', True,
                (slice(i0, i1, 1), self.j1 - 1, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j1, slice(k0, k1 + 1, 1)),
                (slice(i0, i1, 1), self.j1 + 1, slice(k0, k1 + 1, 1)),
                self.Ex],
            ['ez_front_1', False,
                (slice(i0, i1 + 1, 1), self.j0 - 1, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j0, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j0 + 1, slice(k0, k1, 1)),
                self.Ez],
            ['ez_back_1', False,
                (slice(i0, i1 + 1, 1), self.j1 - 1, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j1, slice(k0, k1, 1)),
                (slice(i0, i1 + 1, 1), self.j1 + 1, slice(k0, k1, 1)),
                self.Ez],
            ['ey_left_1', True,
                (self.i0 - 1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i0, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i0 + 1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                self.Ey],
            ['ey_right_1', True,
                (self.i1 - 1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                (self.i1 + 1, slice(j0, j1, 1), slice(k0, k1 + 1, 1)),
                self.Ey],
            ['ez_left_1', False,
                (self.i0 - 1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i0, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i0 + 1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                self.Ez],
            ['ez_right_1', False,
                (self.i1 - 1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                (self.i1 + 1, slice(j0, j1 + 1, 1), slice(k0, k1, 1)),
                self.Ez],

            ['ex_bottom_1', True,
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k0 - 1),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k0),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k0 + 1),
                self.Ex],
            ['ex_top_1', True,
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k1 - 1),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k1),
                (slice(i0, i1, 1), slice(j0, j1 + 1, 1), self.k1 + 1),
                self.Ex],
            ['ey_bottom_1', False,
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k0 - 1),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k0),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k0 + 1),
                self.Ey],
            ['ey_top_1', False,
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k1 - 1),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k1),
                (slice(i0, i1 + 1, 1), slice(j0, j1, 1), self.k1 + 1),
                self.Ey]
        ]

        for obj in slices:
            sliced_field = obj[-1][obj[2]]
            obj[1] = self.create_interpolated_coords(obj[1], sliced_field)

        self.electric_slices = slices

    def create_interpolated_coords(self, mid, field):

        n_x = field.shape[0]
        n_y = field.shape[1]

        if mid:
            x = np.arange(0.5, n_x, 1.0)
            z = np.arange(0, n_y, 1.0)

            # Coordinates that require interpolated values
            x_sg = np.linspace(self.d, n_x - self.d, n_x * self.ratio)
            z_sg = np.linspace(0, n_y - 1, (n_y - 1) * self.ratio + 1)

        else:
            x = np.arange(0, n_x, 1.0)
            z = np.arange(0.5, n_y, 1.0)

            # Coordinates that require interpolated values
            x_sg = np.linspace(0, n_x - 1, (n_x - 1) * self.ratio + 1)
            z_sg = np.linspace(self.d, n_y - self.d, n_y * self.ratio)

        return (x, z, x_sg, z_sg)

    def update_magnetic(self):

        # Copy previous time step magnetic field values to the previous
        # time step variables
        self.update_previous_timestep_fields(self.fn_m)

        for obj in self.magnetic_slices:

            # Grab the main grid fields used to interpolate across the IS
            # f = self.Hi[slice]

            f_u_1 = obj[-1][obj[2]]
            f_u_2 = obj[-1][obj[3]]
            f_u_3 = obj[-1][obj[4]]
            f_u_4 = obj[-1][obj[5]]

            f_1 = 0.25 * f_u_1 + 0.5 * f_u_2 + 0.25 * f_u_3
            f_2 = 0.25 * f_u_2 + 0.5 * f_u_3 + 0.25 * f_u_4

            #f_1 = obj[-1][obj[2]]
            #f_2 = obj[-1][obj[3]]
            if ('left' in obj[0] or
                'bottom' in obj[0] or
                    'front' in obj[0]):
                w = self.l_weight
            else:
                w = self.r_weight
            c1, c2 = calculate_weighting_coefficients(w, self.ratio)
            # transverse interpolated h field
            f_t = c1 * f_1 + c2 * f_2

            # interpolate over a fine grid
            f_i = self.interpolate_to_sub_grid(f_t, obj[1])

            if f_i == f_t:
                raise ValueError

            if obj[0] == 'hz_front_1' or obj[0] == 'hz_back_1':

                pre_0 = int(f_i.shape[0] / 2 - 1 - self.ratio // 2)
                pre_1 = int(f_i.shape[0] / 2 + self.ratio // 2)

                f_i[pre_0 + 1:pre_0 + 1 + self.ratio // 2, :] = np.stack((f_i[pre_0, :], f_i[pre_0, :]))
                f_i[pre_1 - self.ratio // 2:pre_1, :] = np.stack((f_i[pre_1, :], f_i[pre_1, :]))


            # discard the outer nodes only required for interpolation
            #f = f_i[self.ratio:-self.ratio, self.ratio:-self.ratio]
            f = f_i
            setattr(self, obj[0], f)

    def update_electric(self):

        self.update_previous_timestep_fields(self.fn_e)

        for obj in self.electric_slices:

            f_u_1 = obj[-1][obj[2]]
            f_u_2 = obj[-1][obj[3]]
            f_u_3 = obj[-1][obj[4]]

            f_m = 0.25 * f_u_1 + 0.5 * f_u_2 + 0.25 * f_u_3

            f_i = self.interpolate_to_sub_grid(f_m, obj[1])

            # correction interpolated points at the pec traversal point
            if obj[0] == 'ex_front_1' or obj[0] == 'ex_back_1':

                pre_0 = int(f_i.shape[0] / 2 - 1 - self.ratio // 2)
                pre_1 = int(f_i.shape[0] / 2 + self.ratio // 2)

                f_i[pre_0 + 1:pre_0 + 1 + self.ratio // 2, :] = np.stack((f_i[pre_0, :], f_i[pre_0, :]))
                f_i[pre_1 - self.ratio // 2:pre_1, :] = np.stack((f_i[pre_1, :], f_i[pre_1, :]))

            f = f_i
            #f = f_i[self.ratio:-self.ratio, self.ratio:-self.ratio]
            setattr(self, obj[0], f)

    def interpolate_to_sub_grid(self, field, coords):
        x, z, x_sg, z_sg = coords
        interp_f = interpolate.RectBivariateSpline(x, z, field, kx=self.interpolation, ky=self.interpolation)
        f_i = interp_f(x_sg, z_sg)
        return f_i


class PlaneError(Exception):
    pass
