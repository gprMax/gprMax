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

from contextlib import AbstractContextManager
from os import PathLike
from types import TracebackType
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.output_controllers.grid_view import GridView, MPIGridView


class ReadGeometryObject(AbstractContextManager):
    def __init__(
        self,
        filename: PathLike,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        num_existing_materials: int,
    ) -> None:
        self.file_handler = h5py.File(filename)

        data = self.file_handler["/data"]
        assert isinstance(data, h5py.Dataset)
        stop = start + data.shape

        if isinstance(grid, MPIGrid):
            if grid.local_bounds_overlap_grid(start, stop):
                self.grid_view = MPIGridView(
                    grid, start[0], start[1], start[2], stop[0], stop[1], stop[2]
                )
            else:
                # The MPIGridView will create a new communicator using
                # MPI_Split. Calling this here prevents deadlock if not
                # all ranks need to read the geometry object.
                grid.comm.Split(MPI.UNDEFINED)
                self.grid_view = None

        else:
            self.grid_view = GridView(grid, start[0], start[1], start[2], stop[0], stop[1], stop[2])

        self.num_existing_materials = num_existing_materials

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Close the file when the context is exited.

        The parameters describe the exception that caused the context to
        be exited. If the context was exited without an exception, all
        three arguments will be None. Any exception will be
        processed normally upon exit from this method.

        Returns:
            suppress_exception (optional): Returns True if the exception
                should be suppressed (i.e. not propagated). Otherwise,
                the exception will be processed normally upon exit from
                this method.
        """
        self.close()

    def close(self) -> None:
        """Close the file handler"""
        self.file_handler.close()

    def has_valid_discritisation(self) -> bool:
        if self.grid_view is None:
            return True

        dx_dy_dz = self.file_handler.attrs["dx_dy_dz"]
        return not isinstance(dx_dy_dz, h5py.Empty) and all(dx_dy_dz == self.grid_view.grid.dl)

    def has_ID_array(self) -> bool:
        ID_class = self.file_handler.get("ID", getclass=True)
        return ID_class == h5py.Dataset

    def has_rigid_arrays(self) -> bool:
        rigidE_class = self.file_handler.get("rigidE", getclass=True)
        rigidH_class = self.file_handler.get("rigidH", getclass=True)
        return rigidE_class == h5py.Dataset and rigidH_class == h5py.Dataset

    def read_data(self):
        if self.grid_view is None:
            return

        data = self.file_handler["/data"]
        assert isinstance(data, h5py.Dataset)
        data = data[self.grid_view.get_3d_read_slice()]

        # Should be int16 to allow for -1 which indicates background, i.e.
        # don't build anything, but AustinMan/Woman maybe uint16
        if data.dtype != "int16":
            data = data.astype("int16")

        self.grid_view.set_solid(data + self.num_existing_materials)

    def get_data(self) -> Optional[npt.NDArray[np.int16]]:
        if self.grid_view is None:
            return None

        data = self.file_handler["/data"]
        assert isinstance(data, h5py.Dataset)
        data = data[self.grid_view.get_3d_read_slice()]

        # Should be int16 to allow for -1 which indicates background, i.e.
        # don't build anything, but AustinMan/Woman maybe uint16
        if data.dtype != "int16":
            data = data.astype("int16")

        return data + self.num_existing_materials

    def read_rigidE(self):
        if self.grid_view is None:
            return

        rigidE = self.file_handler["/rigidE"]
        assert isinstance(rigidE, h5py.Dataset)

        dset_slice = self.grid_view.get_3d_read_slice()
        self.grid_view.set_rigidE(rigidE[:, dset_slice[0], dset_slice[1], dset_slice[2]])

    def read_rigidH(self):
        if self.grid_view is None:
            return

        rigidH = self.file_handler["/rigidH"]
        assert isinstance(rigidH, h5py.Dataset)

        dset_slice = self.grid_view.get_3d_read_slice()
        self.grid_view.set_rigidH(rigidH[:, dset_slice[0], dset_slice[1], dset_slice[2]])

    def read_ID(self):
        if self.grid_view is None:
            return

        ID = self.file_handler["/ID"]
        assert isinstance(ID, h5py.Dataset)

        dset_slice = self.grid_view.get_3d_read_slice(upper_bound_exclusive=False)
        self.grid_view.set_ID(
            ID[:, dset_slice[0], dset_slice[1], dset_slice[2]] + self.num_existing_materials
        )
