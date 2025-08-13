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

from os import PathLike
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from mpi4py.MPI import Intracomm

from gprMax.vtkhdf_filehandlers.vtkhdf import VtkFileType, VtkHdfFile


class VtkImageData(VtkHdfFile):
    """File handler for creating a VTKHDF Image Data file.

    File format information is available here:
    https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#image-data
    """

    TYPE = VtkFileType.IMAGE_DATA

    DIRECTION_ATTR = "Direction"
    ORIGIN_ATTR = "Origin"
    SPACING_ATTR = "Spacing"
    WHOLE_EXTENT_ATTR = "WholeExtent"

    DIMENSIONS = 3

    def __init__(
        self,
        filename: Union[str, PathLike],
        shape: npt.NDArray[np.int32],
        origin: Optional[npt.NDArray[np.float32]] = None,
        spacing: Optional[npt.NDArray[np.float32]] = None,
        direction: Optional[npt.NDArray[np.float32]] = None,
        comm: Optional[Intracomm] = None,
    ):
        """Create a new VtkImageData file.

        If the file already exists, it will be overriden. Required
        attributes (Type and Version) will be written to the file.

        The file will be opened using the 'mpio' h5py driver if an MPI
        communicator is provided.

        Args:
            filename: Name of the file (can be a file path). The file
                extension will be set to '.vtkhdf'.
            shape: Shape of the image data to be stored in the file.
                This specifies the number of cells. Image data can be
                1D, 2D, or 3D.
            origin (optional): Origin of the image data. Default
                [0, 0, 0].
            spacing (optional): Discritisation of the image data.
                Default [1, 1, 1].
            direction (optional): Array of direction vectors for each
                dimension of the image data. Can be a flattened array.
                I.e. [[1, 0, 0], [0, 1, 0], [0, 0, 1]] and
                [1, 0, 0, 0, 1, 0, 0, 0, 1] are equivalent. Default
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]].
            comm (optional): MPI communicator containing all ranks that
                want to write to the file.
        """
        super().__init__(filename, self.TYPE, "w", comm)

        if len(shape) == 0:
            raise ValueError(f"Shape must not be empty.")
        if len(shape) > self.DIMENSIONS:
            raise ValueError(f"Shape must not have more than {self.DIMENSIONS} dimensions.")
        elif len(shape) < self.DIMENSIONS:
            shape = np.concatenate((shape, np.ones(self.DIMENSIONS - len(shape), dtype=np.int32)))

        self.shape = shape

        whole_extent = np.zeros(2 * self.DIMENSIONS, dtype=np.int32)
        whole_extent[1::2] = self.shape
        self._set_root_attribute(self.WHOLE_EXTENT_ATTR, whole_extent)

        if origin is None:
            origin = np.zeros(self.DIMENSIONS, dtype=np.float32)
        self.set_origin(origin)

        if spacing is None:
            spacing = np.ones(self.DIMENSIONS, dtype=np.float32)
        self.set_spacing(spacing)

        if direction is None:
            direction = np.diag(np.ones(self.DIMENSIONS, dtype=np.float32))
        self.set_direction(direction)

    @property
    def whole_extent(self) -> npt.NDArray[np.int32]:
        return self._get_root_attribute(self.WHOLE_EXTENT_ATTR)

    @property
    def origin(self) -> npt.NDArray[np.float32]:
        return self._get_root_attribute(self.ORIGIN_ATTR)

    @property
    def spacing(self) -> npt.NDArray[np.float32]:
        return self._get_root_attribute(self.SPACING_ATTR)

    @property
    def direction(self) -> npt.NDArray[np.float32]:
        return self._get_root_attribute(self.DIRECTION_ATTR)

    def set_origin(self, origin: npt.NDArray[np.float32]):
        """Set the origin coordinate of the image data.

        Args:
            origin: x, y, z coordinates to set as the origin.
        """
        if len(origin) != self.DIMENSIONS:
            raise ValueError(f"Origin attribute must have {self.DIMENSIONS} dimensions.")
        self._set_root_attribute(self.ORIGIN_ATTR, origin)

    def set_spacing(self, spacing: npt.NDArray[np.float32]):
        """Set the discritisation of the image data.

        Args:
            spacing: Discritisation of the x, y, and z dimensions.
        """
        if len(spacing) != self.DIMENSIONS:
            raise ValueError(f"Spacing attribute must have {self.DIMENSIONS} dimensions.")
        self._set_root_attribute(self.SPACING_ATTR, spacing)

    def set_direction(self, direction: npt.NDArray[np.float32]):
        """Set the coordinate system of the image data.

        Args:
            direction: Array of direction vectors for each dimension of
                the image data. Can be a flattened array. I.e.
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]] and
                [1, 0, 0, 0, 1, 0, 0, 0, 1] are equivalent.
        """
        direction = direction.flatten()
        if len(direction) != self.DIMENSIONS * self.DIMENSIONS:
            raise ValueError(
                f"Direction array must contain {self.DIMENSIONS * self.DIMENSIONS} elements."
            )
        self._set_root_attribute(self.DIRECTION_ATTR, direction)

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
        points_shape = self.shape + 1
        if offset is None and any(data.shape != points_shape):  # type: ignore
            raise ValueError(
                "If no offset is specified, data.shape must be one larger in each dimension than"
                f" this vtkImageData object. {data.shape} != {points_shape}"
            )
        return super()._add_point_data(name, data, points_shape, offset)

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
        if offset is None and any(data.shape != self.shape):  # type: ignore
            raise ValueError(
                "If no offset is specified, data.shape must match the dimensions of this"
                f" VtkImageData object. {data.shape} != {self.shape}"
            )
        return super()._add_cell_data(name, data, self.shape, offset)
