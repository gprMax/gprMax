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
from os import PathLike
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType, VtkHdfFile

logger = logging.getLogger(__name__)


class VtkUnstructuredGrid(VtkHdfFile):
    """File handler for creating a VTKHDF Unstructured Grid file.

    File format information is available here:
    https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#unstructured-grid
    """

    class Dataset(VtkHdfFile.Dataset):
        CONNECTIVITY = "Connectivity"
        NUMBER_OF_CELLS = "NumberOfCells"
        NUMBER_OF_CONNECTIVITY_IDS = "NumberOfConnectivityIds"
        NUMBER_OF_POINTS = "NumberOfPoints"
        OFFSETS = "Offsets"
        POINTS = "Points"
        TYPES = "Types"

    @property
    def TYPE(self) -> Literal["UnstructuredGrid"]:
        return "UnstructuredGrid"

    def __init__(
        self,
        filename: Union[str, PathLike],
        points: npt.NDArray,
        cell_types: npt.NDArray[VtkCellType],
        connectivity: npt.NDArray,
        cell_offsets: npt.NDArray,
        comm: Optional[MPI.Comm] = None,
    ) -> None:
        """Create a new VtkUnstructuredGrid file.

        An unstructured grid has N points and C cells. A cell is defined
        as a collection of points which is specified by the connectivity
        and cell_offsets arguments along with the list of cell_types.

        If the file already exists, it will be overriden. Required
        attributes (Type and Version) will be written to the file.

        The file will be opened using the 'mpio' h5py driver if an MPI
        communicator is provided.

        Args:
            filename: Name of the file (can be a file path). The file
                extension will be set to '.vtkhdf'.
            points: Array of point coordinates of shape (N, 3).
            cell_types: Array of VTK cell types of shape (C,).
            connectivity: Array of point IDs that together with
                cell_offsets, defines the points that make up each cell.
                Each point ID has a value between 0 and (N-1) inclusive
                and corresponds to a point in the points array.
            cell_offsets: Array listing where each cell starts and ends
                in the connectivity array. It has shape (C + 1,).
            comm (optional): MPI communicator containing all ranks that
                want to write to the file.

        Raises:
            Value Error: Raised if argument dimensions are invalid.
        """
        super().__init__(filename, comm)

        if len(cell_offsets) != len(cell_types) + 1:
            raise ValueError(
                "cell_offsets should be one longer than cell_types."
                " I.e. one longer than the number of cells"
            )

        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        if not is_sorted(cell_offsets):
            raise ValueError("cell_offsets should be sorted in ascending order")

        if len(connectivity) < cell_offsets[-1]:
            raise ValueError("Connectivity array is shorter than final cell_offsets value")

        elif len(connectivity) > cell_offsets[-1]:
            logger.warning(
                "Connectivity array longer than final cell_offsets value."
                " Some connectivity data will be ignored"
            )

        self._write_root_dataset(self.Dataset.CONNECTIVITY, connectivity)
        self._write_root_dataset(self.Dataset.NUMBER_OF_CELLS, len(cell_types))
        self._write_root_dataset(self.Dataset.NUMBER_OF_CONNECTIVITY_IDS, len(connectivity))
        self._write_root_dataset(self.Dataset.NUMBER_OF_POINTS, len(points))
        self._write_root_dataset(self.Dataset.OFFSETS, cell_offsets)
        self._write_root_dataset(self.Dataset.POINTS, points, xyz_data_ordering=False)
        self._write_root_dataset(self.Dataset.TYPES, cell_types)

    @property
    def number_of_cells(self) -> int:
        number_of_cells = self._get_root_dataset(self.Dataset.NUMBER_OF_CELLS)
        return np.sum(number_of_cells, dtype=np.int32)

    @property
    def number_of_connectivity_ids(self) -> int:
        number_of_connectivity_ids = self._get_root_dataset(self.Dataset.NUMBER_OF_CONNECTIVITY_IDS)
        return np.sum(number_of_connectivity_ids, dtype=np.int32)

    @property
    def number_of_points(self) -> int:
        number_of_points = self._get_root_dataset(self.Dataset.NUMBER_OF_POINTS)
        return np.sum(number_of_points, dtype=np.int32)

    def add_point_data(
        self, name: str, data: npt.NDArray, offset: Optional[npt.NDArray[np.int32]] = None
    ):
        """Add point data to the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to be saved.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.

        Raises:
            ValueError: Raised if data has invalid dimensions.
        """
        shape = np.array(data.shape)
        number_of_dimensions = len(shape)

        if number_of_dimensions < 1 or number_of_dimensions > 2:
            raise ValueError(f"Data must have 1 or 2 dimensions, not {number_of_dimensions}")
        elif len(data) != self.number_of_points:
            raise ValueError(
                "Length of data must match the number of points in the vtkUnstructuredGrid."
                f" {len(data)} != {self.number_of_points}"
            )
        elif number_of_dimensions == 2 and shape[1] != 1 and shape[1] != 3:
            raise ValueError(f"The second dimension should have shape 1 or 3, not {shape[1]}")

        return super().add_point_data(name, data, shape, offset)

    def add_cell_data(
        self, name: str, data: npt.NDArray, offset: Optional[npt.NDArray[np.int32]] = None
    ):
        """Add cell data to the VTKHDF file.

        Args:
            name: Name of the dataset.
            data: Data to be saved.
            offset (optional): Offset to store the provided data at. Can
                be omitted if data provides the full dataset.

        Raises:
            ValueError: Raised if data has invalid dimensions.
        """
        shape = np.array(data.shape)
        number_of_dimensions = len(shape)

        if number_of_dimensions < 1 or number_of_dimensions > 2:
            raise ValueError(f"Data must have 1 or 2 dimensions, not {number_of_dimensions}.")
        elif len(data) != self.number_of_cells:
            raise ValueError(
                "Length of data must match the number of cells in the vtkUnstructuredGrid."
                f" {len(data)} != {self.number_of_cells}"
            )
        elif number_of_dimensions == 2 and shape[1] != 1 and shape[1] != 3:
            raise ValueError(f"The second dimension should have shape 1 or 3, not {shape[1]}")

        return super().add_cell_data(name, data, shape, offset)
