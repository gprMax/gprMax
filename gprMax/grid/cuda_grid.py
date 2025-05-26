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

import numpy as np

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.pml import CUDAPML


class CUDAGrid(FDTDGrid):
    """Additional grid methods for solving on GPU using CUDA."""

    def __init__(self):
        super().__init__()

        self.gpuarray = import_module("pycuda.gpuarray")

        # Threads per block - used for main electric/magnetic field updates
        self.tpb = (128, 1, 1)
        # Blocks per grid - used for main electric/magnetic field updates
        self.bpg = None

    def _construct_pml(self, pml_ID: str, thickness: int) -> CUDAPML:
        return super()._construct_pml(pml_ID, thickness, CUDAPML)

    def set_blocks_per_grid(self):
        """Set the blocks per grid size used for updating the electric and
        magnetic field arrays on a GPU.
        """

        self.bpg = (
            int(np.ceil(((self.nx + 1) * (self.ny + 1) * (self.nz + 1)) / self.tpb[0])),
            1,
            1,
        )

    def htod_geometry_arrays(self):
        """Initialise an array for cell edge IDs (ID) on compute device."""

        self.ID_dev = self.gpuarray.to_gpu(self.ID)

    def htod_field_arrays(self):
        """Initialise field arrays on compute device."""

        self.Ex_dev = self.gpuarray.to_gpu(self.Ex)
        self.Ey_dev = self.gpuarray.to_gpu(self.Ey)
        self.Ez_dev = self.gpuarray.to_gpu(self.Ez)
        self.Hx_dev = self.gpuarray.to_gpu(self.Hx)
        self.Hy_dev = self.gpuarray.to_gpu(self.Hy)
        self.Hz_dev = self.gpuarray.to_gpu(self.Hz)

    def htod_dispersive_arrays(self):
        """Initialise dispersive material coefficient arrays on compute device."""

        self.updatecoeffsdispersive_dev = self.gpuarray.to_gpu(self.updatecoeffsdispersive)
        self.Tx_dev = self.gpuarray.to_gpu(self.Tx)
        self.Ty_dev = self.gpuarray.to_gpu(self.Ty)
        self.Tz_dev = self.gpuarray.to_gpu(self.Tz)
