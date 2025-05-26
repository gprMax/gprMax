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

from importlib import import_module

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.pml import PML, OpenCLPML


class OpenCLGrid(FDTDGrid):
    """Additional grid methods for solving on compute device using OpenCL."""

    def __init__(self):
        super().__init__()

        self.clarray = import_module("pyopencl.array")

    def _construct_pml(self, pml_ID: str, thickness: int) -> OpenCLPML:
        return super()._construct_pml(pml_ID, thickness, OpenCLPML)

    def htod_geometry_arrays(self, queue):
        """Initialise an array for cell edge IDs (ID) on compute device.

        Args:
            queue: pyopencl queue.
        """

        self.ID_dev = self.clarray.to_device(queue, self.ID)

    def htod_field_arrays(self, queue):
        """Initialise field arrays on compute device.

        Args:
            queue: pyopencl queue.
        """

        self.Ex_dev = self.clarray.to_device(queue, self.Ex)
        self.Ey_dev = self.clarray.to_device(queue, self.Ey)
        self.Ez_dev = self.clarray.to_device(queue, self.Ez)
        self.Hx_dev = self.clarray.to_device(queue, self.Hx)
        self.Hy_dev = self.clarray.to_device(queue, self.Hy)
        self.Hz_dev = self.clarray.to_device(queue, self.Hz)

    def htod_dispersive_arrays(self, queue):
        """Initialise dispersive material coefficient arrays on compute device.

        Args:
            queue: pyopencl queue.
        """

        self.updatecoeffsdispersive_dev = self.clarray.to_device(queue, self.updatecoeffsdispersive)
        # self.updatecoeffsdispersive_dev = self.clarray.to_device(queue, np.ones((95,95,95), dtype=np.float32))
        self.Tx_dev = self.clarray.to_device(queue, self.Tx)
        self.Ty_dev = self.clarray.to_device(queue, self.Ty)
        self.Tz_dev = self.clarray.to_device(queue, self.Tz)
