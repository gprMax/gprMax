# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

import logging

import numpy as np
from mpi4py import MPI

from gprMax._version import __version__
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.output_controllers.geometry_views import GeometryView, GridViewType, Metadata
from gprMax.output_controllers.grid_view import MPIGridView
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.vtkhdf_filehandlers.vtk_image_data import VtkImageData

logger = logging.getLogger(__name__)


class GeometryViewVoxels(GeometryView[GridViewType]):
    """Image data for a per-cell geometry view."""

    def prep_vtk(self):
        """Prepares data for writing to VTKHDF file."""

        self.material_data = self.grid_view.get_solid()

        if isinstance(self.grid, SubGridBaseGrid):
            self.origin = np.array(
                [
                    (self.grid.i0 * self.grid.dx * self.grid.ratio),
                    (self.grid.j0 * self.grid.dy * self.grid.ratio),
                    (self.grid.k0 * self.grid.dz * self.grid.ratio),
                ]
            )
        else:
            self.origin = self.grid_view.start * self.grid.dl

        self.spacing = self.grid_view.step * self.grid.dl

        # Write information about any PMLs, sources, receivers
        self.metadata = Metadata(self.grid_view)

        self.nbytes = self.material_data.nbytes

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        with VtkImageData(self.filename, self.grid_view.size, self.origin, self.spacing) as f:
            f.add_cell_data("Material", self.material_data)
            self.metadata.write_to_vtkhdf(f)


class MPIGeometryViewVoxels(GeometryViewVoxels[MPIGridView]):
    """Image data for a per-cell geometry view."""

    def __init__(self, grid_view: MPIGridView, filename: str, comm: MPI.Comm):
        super().__init__(grid_view, filename)

        self.comm = comm

    @property
    def grid(self) -> MPIGrid:
        return self.grid_view.grid

    def prep_vtk(self):
        """Prepares data for writing to VTKHDF file."""

        super().prep_vtk()

        # Use global material IDs rather than local IDs
        self.material_data = self.grid.local_to_global_material_id_map(self.material_data)

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        with VtkImageData(
            self.filename, self.grid_view.global_size, self.origin, self.spacing, comm=self.comm
        ) as f:
            f.add_cell_data("Material", self.material_data, self.grid_view.offset)
            self.metadata.write_to_vtkhdf(f)
