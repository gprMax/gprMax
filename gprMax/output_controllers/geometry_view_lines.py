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
from gprMax.cython.geometry_outputs import get_line_properties
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.output_controllers.geometry_views import GeometryView, GridViewType, Metadata
from gprMax.output_controllers.grid_view import MPIGridView
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.vtkhdf_filehandlers.vtk_unstructured_grid import VtkUnstructuredGrid
from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

logger = logging.getLogger(__name__)


class GeometryViewLines(GeometryView[GridViewType]):
    """Unstructured grid for a per-cell-edge geometry view."""

    def prep_vtk(self):
        """Prepares data for writing to VTKHDF file."""

        ID = self.grid_view.get_ID()

        x = np.arange(self.grid_view.nx + 1, dtype=np.float64)
        y = np.arange(self.grid_view.ny + 1, dtype=np.float64)
        z = np.arange(self.grid_view.nz + 1, dtype=np.float64)
        coords = np.meshgrid(x, y, z, indexing="ij")
        self.points = np.vstack(list(map(np.ravel, coords))).T
        self.points += self.grid_view.start
        self.points *= self.grid_view.step * self.grid.dl

        # Add offset to subgrid geometry to correctly locate within main grid
        if isinstance(self.grid, SubGridBaseGrid):
            offset = [self.grid.i0, self.grid.j0, self.grid.k0]
            self.points += offset * self.grid.dl * self.grid.ratio

        # Each point is the 'source' for 3 lines.
        # NB: Excluding points at the far edge of the geometry as those
        # are the 'source' for no lines
        n_lines = 3 * np.prod(self.grid_view.size)

        self.cell_types = np.full(n_lines, VtkCellType.LINE)
        self.cell_offsets = np.arange(0, 2 * n_lines + 2, 2, dtype=np.intc)

        self.connectivity, self.material_data = get_line_properties(
            n_lines, *self.grid_view.size, ID
        )

        assert isinstance(self.connectivity, np.ndarray)
        assert isinstance(self.material_data, np.ndarray)

        # Write information about any PMLs, sources, receivers
        self.metadata = Metadata(self.grid_view, averaged_materials=True, materials_only=True)

        # Number of bytes of data to be written to file
        self.nbytes = (
            self.points.nbytes
            + self.cell_types.nbytes
            + self.connectivity.nbytes
            + self.cell_offsets.nbytes
            + self.material_data.nbytes
        )

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        # Write the VTK file
        with VtkUnstructuredGrid(
            self.filename,
            self.points,
            self.cell_types,
            self.connectivity,
            self.cell_offsets,
        ) as f:
            f.add_cell_data("Material", self.material_data)
            self.metadata.write_to_vtkhdf(f)


class MPIGeometryViewLines(GeometryViewLines[MPIGridView]):
    """Image data for a per-cell geometry view."""

    def __init__(self, grid_view: MPIGridView, filename: str, comm: MPI.Comm):
        super().__init__(grid_view, filename)

        self.comm = comm

    @property
    def grid(self) -> MPIGrid:
        return self.grid_view.grid

    def prep_vtk(self):
        """Prepares data for writing to VTKHDF file."""

        ID = self.grid_view.get_ID()

        x = np.arange(self.grid_view.gx + 1, dtype=np.float64)
        y = np.arange(self.grid_view.gy + 1, dtype=np.float64)
        z = np.arange(self.grid_view.gz + 1, dtype=np.float64)
        coords = np.meshgrid(x, y, z, indexing="ij")
        self.points = np.vstack(list(map(np.ravel, coords))).T
        self.points += self.grid_view.global_start
        self.points *= self.grid_view.step * self.grid.dl

        # Add offset to subgrid geometry to correctly locate within main grid
        if isinstance(self.grid, SubGridBaseGrid):
            offset = [self.grid.i0, self.grid.j0, self.grid.k0]
            self.points += offset * self.grid.dl * self.grid.ratio

        # Each point is the 'source' for 3 lines.
        # NB: Excluding points at the far edge of the geometry as those
        # are the 'source' for no lines
        n_lines = 3 * np.prod(self.grid_view.global_size)

        self.cell_types = np.full(n_lines, VtkCellType.LINE)
        self.cell_offsets = np.arange(0, 2 * n_lines + 2, 2, dtype=np.intc)

        self.connectivity, self.material_data = get_line_properties(
            n_lines, *self.grid_view.size, ID
        )

        assert isinstance(self.connectivity, np.ndarray)
        assert isinstance(self.material_data, np.ndarray)

        # Write information about any PMLs, sources, receivers
        self.metadata = Metadata(self.grid_view, averaged_materials=True, materials_only=True)

        # Number of bytes of data to be written to file
        self.nbytes = (
            self.points.nbytes
            + self.cell_types.nbytes
            + self.connectivity.nbytes
            + self.cell_offsets.nbytes
            + self.material_data.nbytes
        )

        # Use global material IDs rather than local IDs
        self.material_data = self.grid.local_to_global_material_id_map(self.material_data)

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        with VtkUnstructuredGrid(
            self.filename,
            self.points,
            self.cell_types,
            self.connectivity,
            self.cell_offsets,
            comm=self.comm,
        ) as f:
            f.add_cell_data("Material", self.material_data, self.grid_view.offset)
            self.metadata.write_to_vtkhdf(f)
