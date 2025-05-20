# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley, 
#                          and Nathan Mannall
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

from gprMax._version import __version__
from gprMax.cython.geometry_outputs import get_line_properties
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.output_controllers.geometry_views import GeometryView, Metadata, MPIMetadata
from gprMax.output_controllers.grid_view import GridType, MPIGridView
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.vtkhdf_filehandlers.vtk_unstructured_grid import VtkUnstructuredGrid
from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

logger = logging.getLogger(__name__)


class GeometryViewLines(GeometryView[GridType]):
    """Unstructured grid for a per-cell-edge geometry view."""

    def __init__(
        self,
        xs: int,
        ys: int,
        zs: int,
        xf: int,
        yf: int,
        zf: int,
        filename: str,
        grid: GridType,
    ):
        super().__init__(xs, ys, zs, xf, yf, zf, 1, 1, 1, filename, grid)

    def prep_vtk(self):
        """Prepares data for writing to VTKHDF file."""

        self.grid_view.initialise_materials(filter_materials=False)

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

        # Use sorted material IDs rather than default ordering
        self.material_data = self.grid_view.map_to_view_materials(self.material_data)

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


class MPIGeometryViewLines(GeometryViewLines[MPIGrid]):
    """Image data for a per-cell geometry view."""

    @property
    def GRID_VIEW_TYPE(self) -> type[MPIGridView]:
        return MPIGridView

    def prep_vtk(self):
        """Prepares data for writing to VTKHDF file."""

        assert isinstance(self.grid_view, MPIGridView)

        self.grid_view.initialise_materials(filter_materials=False)

        ID = self.grid_view.get_ID()

        x = np.arange(self.grid_view.gx + 1, dtype=np.float64)
        y = np.arange(self.grid_view.gy + 1, dtype=np.float64)
        z = np.arange(self.grid_view.gz + 1, dtype=np.float64)
        coords = np.meshgrid(x, y, z, indexing="ij")
        self.points = np.vstack(list(map(np.ravel, coords))).T
        self.points += self.grid_view.global_start
        self.points *= self.grid_view.step * self.grid.dl

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
        self.metadata = MPIMetadata(self.grid_view, averaged_materials=True, materials_only=True)

        # Number of bytes of data to be written to file
        self.nbytes = (
            self.points.nbytes
            + self.cell_types.nbytes
            + self.connectivity.nbytes
            + self.cell_offsets.nbytes
            + self.material_data.nbytes
        )

        # Use global material IDs rather than local IDs
        self.material_data = self.grid_view.map_to_view_materials(self.material_data)

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        assert isinstance(self.grid_view, MPIGridView)

        with VtkUnstructuredGrid(
            self.filename,
            self.points,
            self.cell_types,
            self.connectivity,
            self.cell_offsets,
            comm=self.grid_view.comm,
        ) as f:
            self.metadata.write_to_vtkhdf(f)
            f.add_cell_data("Material", self.material_data, self.grid_view.offset)
