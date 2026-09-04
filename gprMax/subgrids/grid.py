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

import logging
from abc import ABC, abstractmethod

import numpy as np

from gprMax.grid.fdtd_grid import FDTDGrid

logger = logging.getLogger(__name__)


class SubGridBaseGrid(FDTDGrid, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.ratio = kwargs["ratio"]

        if self.ratio % 2 == 0:
            logger.exception("Subgrid Error: Only odd ratios are supported")
            raise ValueError

        # ratio=1 is an equal-resolution embedded region rather than a
        # refining subgrid. It uses the HSG ownership/coupling machinery, but
        # needs none of the stabilisation or interpolation used at a change of
        # resolution.
        self.equal_resolution = self.ratio == 1
        self.coupling_mode = (
            "equal_resolution" if self.equal_resolution else "refining_hsg"
        )

        # Name of the grid
        self.name = kwargs["id"]
        self.parent_grid: FDTDGrid
        self.iterations = 0

        self.filter = False if self.equal_resolution else kwargs["filter"]

        # Number of main grid cells between the IS and OS
        self.is_os_sep = kwargs["is_os_sep"]
        # Number of subgrid grid cells between the IS and OS
        self.s_is_os_sep = self.is_os_sep * self.ratio

        # Distance from OS to PML or the edge of the grid when PML is off
        self.pml_separation = kwargs["pml_separation"]

        pml_thickness = 0 if self.equal_resolution else kwargs["subgrid_pml_thickness"]
        self.pmls["thickness"]["x0"] = pml_thickness
        self.pmls["thickness"]["y0"] = pml_thickness
        self.pmls["thickness"]["z0"] = pml_thickness
        self.pmls["thickness"]["xmax"] = pml_thickness
        self.pmls["thickness"]["ymax"] = pml_thickness
        self.pmls["thickness"]["zmax"] = pml_thickness

        # Number of sub cells to extend the sub grid beyond the IS boundary
        d_to_pml = self.s_is_os_sep + self.pml_separation
        # Index of the IS
        self.n_boundary_cells = d_to_pml + self.pmls["thickness"]["x0"]
        self.n_boundary_cells_x = d_to_pml + self.pmls["thickness"]["x0"]
        self.n_boundary_cells_y = d_to_pml + self.pmls["thickness"]["y0"]
        self.n_boundary_cells_z = d_to_pml + self.pmls["thickness"]["z0"]

        # Zero records direct one-to-one transfer; positive values are spline
        # degrees used only by refining HSG interfaces.
        self.interpolation = 0 if self.equal_resolution else kwargs["interpolation"]

    def local_to_global(self, coord):
        """Converts a local (subgrid array) cell index to a physical
        position in the main grid's/global coordinate frame.

        Local index 0 is offset from the global origin by the subgrid's
        boundary padding (n_boundary_cells_*) and its placement (i0, j0,
        k0) within the main grid - this reverses that offset. See
        SubgridUserInput.translate_to_gap in user_inputs.py, which
        performs the forward transform when building objects.
        """
        boundary = np.array([self.n_boundary_cells_x, self.n_boundary_cells_y, self.n_boundary_cells_z])
        i0 = np.array([self.i0, self.j0, self.k0])
        return (np.asarray(coord) - boundary + i0 * self.ratio) * self.dl

    @abstractmethod
    def update_magnetic_is(self, precursors):
        pass

    @abstractmethod
    def update_electric_is(self, precursors):
        pass

    @abstractmethod
    def update_electric_os(self, main_grid):
        pass

    @abstractmethod
    def update_magnetic_os(self, main_grid):
        pass

    @abstractmethod
    def print_info(self):
        pass
