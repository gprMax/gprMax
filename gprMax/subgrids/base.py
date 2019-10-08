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
from scipy.constants import c

from ..grid import FDTDGrid


class SubGridBase(FDTDGrid):

    def __init__(self, **kwargs):
        super().__init__()

        self.mode = '3D'
        self.ratio = kwargs['ratio']

        if self.ratio % 2 == 0:
            raise ValueError('Subgrid Error: Only odd ratios are supported')

        # Name of the grid
        self.name = kwargs['id']

        self.filter = kwargs['filter']

        # Number of main grid cells between the IS and OS
        self.is_os_sep = kwargs['is_os_sep']
        # Number of subgrid grid cells between the IS and OS
        self.s_is_os_sep = self.is_os_sep * self.ratio

        # Distance from OS to pml or the edge of the grid when pml is off
        self.pml_separation = kwargs['pml_separation']

        self.pmlthickness['x0'] = kwargs['subgrid_pml_thickness']
        self.pmlthickness['y0'] = kwargs['subgrid_pml_thickness']
        self.pmlthickness['z0'] = kwargs['subgrid_pml_thickness']
        self.pmlthickness['xmax'] = kwargs['subgrid_pml_thickness']
        self.pmlthickness['ymax'] = kwargs['subgrid_pml_thickness']
        self.pmlthickness['zmax'] = kwargs['subgrid_pml_thickness']

        # Number of sub cells to extend the sub grid beyond the IS boundary
        d_to_pml = self.s_is_os_sep + self.pml_separation
        # index of the is
        self.n_boundary_cells = d_to_pml + self.pmlthickness['x0']
        self.n_boundary_cells_x = d_to_pml + self.pmlthickness['x0']
        self.n_boundary_cells_y = d_to_pml + self.pmlthickness['y0']
        self.n_boundary_cells_z = d_to_pml + self.pmlthickness['z0']

        self.interpolation = kwargs['interpolation']

    def calculate_dt(self):
        """Calculate dt at the CFL limit."""
        self.dt = (1 / (c * np.sqrt(
                  (1 / self.dx) * (1 / self.dx) +
                  (1 / self.dy) * (1 / self.dy) +
                  (1 / self.dz) * (1 / self.dz))))

    def main_grid_index_to_subgrid_index(self, i, j, k):
        """Calculate local subgrid index from global main grid index."""
        i_s = self.n_boundary_cells_x + (i - self.i0) * self.ratio
        j_s = self.n_boundary_cells_y + (j - self.j0) * self.ratio
        k_s = self.n_boundary_cells_z + (k - self.k0) * self.ratio
        return (i_s, j_s, k_s)

    def initialise_geometry_arrays(self):
        super().initialise_geometry_arrays()
