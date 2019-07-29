from ..grid import FDTDGrid

from scipy.constants import c
import numpy as np


class SubGridBase(FDTDGrid):

    def __init__(self, **kwargs):
        super().__init__()

        self.mode = '3D'

        # subgridding ratio
        self.ratio = kwargs['ratio']

        if self.ratio % 2 == 0:
            raise ValueError('Subgrid Error: Only odd ratios are supported')

        # Name of the grid
        self.name = kwargs['ID']

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
        self.n_boundary_cells = d_to_pml + self.pmlthickness['x0']
        self.n_boundary_cells_x = d_to_pml + self.pmlthickness['x0']
        self.n_boundary_cells_y = d_to_pml + self.pmlthickness['y0']
        self.n_boundary_cells_z = d_to_pml + self.pmlthickness['z0']

        # vectorise coordinates
        self.p0 = np.array(self.i0, self.j0, self.k0)
        self.n_boundary_cells_p = np.array(self.n_boundary_cells_x,
                                           self.n_boundary_cells_y, self.n_boundary_cells_z)

        # interpolation scheme
        self.interpolation = kwargs['interpolation']

    def main_grid_index_to_subgrid_index(self, p):
        """
        Return the equivalent spatial index of the global (i, j, k) in the subgrid.
        Args:
              p (numpy.array): i, j, k indices of a point in the main grid.
        """
        return self.n_boundary_cells_p + (p - self.p0) * self.ratio
