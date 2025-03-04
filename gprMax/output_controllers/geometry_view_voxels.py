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
from gprMax.output_controllers.geometry_views import GeometryView, Metadata, MPIMetadata
from gprMax.output_controllers.grid_view import GridType, MPIGridView
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.vtkhdf_filehandlers.vtk_image_data import VtkImageData

logger = logging.getLogger(__name__)


class GeometryViewVoxels(GeometryView[GridType]):
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

        self.nbytes = self.material_data.nbytes

        # Write information about any PMLs, sources, receivers
        self.metadata = Metadata(self.grid_view)

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        with VtkImageData(self.filename, self.grid_view.size, self.origin, self.spacing) as f:
            f.add_cell_data("Material", self.material_data)
            self.metadata.write_to_vtkhdf(f)


class MPIGeometryViewVoxels(GeometryViewVoxels[MPIGrid]):
    """Image data for a per-cell geometry view."""

    @property
    def GRID_VIEW_TYPE(self) -> type[MPIGridView]:
        return MPIGridView

    def prep_vtk(self):
        """Prepares data for writing to VTKHDF file."""

        assert isinstance(self.grid_view, self.GRID_VIEW_TYPE)

        self.material_data = self.grid_view.get_solid()

        self.origin = self.grid_view.global_start * self.grid.dl
        self.spacing = self.grid_view.step * self.grid.dl

        self.nbytes = self.material_data.nbytes

        # Write information about any PMLs, sources, receivers
        self.metadata = MPIMetadata(self.grid_view)

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        assert isinstance(self.grid_view, self.GRID_VIEW_TYPE)

        with VtkImageData(
            self.filename,
            self.grid_view.global_size,
            self.origin,
            self.spacing,
            comm=self.grid_view.comm,
        ) as f:
            f.add_cell_data("Material", self.material_data, self.grid_view.offset)
            self.metadata.write_to_vtkhdf(f)
