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
from gprMax.pml import PML


class MetalGrid(FDTDGrid):
    """Additional grid methods for solving on compute device using Apple Metal."""

    def __init__(self):
        super().__init__()

        self.metal = import_module("Metal")
        self.storage = self.metal.MTLResourceStorageModeShared

        # Current iteration counter (for tracking during solve)
        self.iteration = 0

        # Threads per thread group - used for main electric/magnetic field updates
        self.tptg = None
        # Thread group size - used for main electric/magnetic field updates
        self.tgs = None

    def set_threads_per_thread_group(self):
        """Set the threads per thread group used for updating the electric and
            magnetic field arrays on a GPU.
        """

        self.tptg = self.metal.MTLSizeMake(
            int(np.ceil(((self.nx + 1) * (self.ny + 1) * (self.nz + 1)))), 1, 1)

    def set_thread_group_size(self, pso):
        """Set the thread group size used for updating the electric and magnetic 
            field arrays on a GPU.

        Args:
            pso: pipeline state object.
        """

        self.tgs = self.metal.MTLSizeMake(
            pso.maxTotalThreadsPerThreadgroup(), 1, 1)

    def htod_geometry_arrays(self, dev):
        """Initialise an array for cell edge IDs (ID) on compute device.

        Args:
            dev: device object.
        """

        self.ID_dev = dev.newBufferWithBytes_length_options_(self.ID,
                                                             self.ID.nbytes,
                                                             self.storage)

    def htod_field_arrays(self, dev):
        """Initialise field arrays on compute device.

        Args:
            dev: device object.
        """

        self.Ex_dev = dev.newBufferWithBytes_length_options_(self.Ex,
                                                             self.Ex.nbytes,
                                                             self.storage)
        self.Ey_dev = dev.newBufferWithBytes_length_options_(self.Ey,
                                                             self.Ey.nbytes,
                                                             self.storage)
        self.Ez_dev = dev.newBufferWithBytes_length_options_(self.Ez,
                                                             self.Ez.nbytes,
                                                             self.storage)
        self.Hx_dev = dev.newBufferWithBytes_length_options_(self.Hx,
                                                             self.Hx.nbytes,
                                                             self.storage)
        self.Hy_dev = dev.newBufferWithBytes_length_options_(self.Hy,
                                                             self.Hy.nbytes,
                                                             self.storage)
        self.Hz_dev = dev.newBufferWithBytes_length_options_(self.Hz,
                                                             self.Hz.nbytes,
                                                             self.storage)

    def htod_dispersive_arrays(self, dev):
        """Initialise dispersive material coefficient arrays on compute device.

        Args:
            dev: device object.
        """

        self.updatecoeffsdispersive_dev = dev.newBufferWithBytes_length_options_(self.updatecoeffsdispersive,
                                                                                 self.updatecoeffsdispersive.nbytes,
                                                                                 self.storage)
        self.Tx_dev = dev.newBufferWithBytes_length_options_(self.Tx,
                                                             self.Tx.nbytes,
                                                             self.storage)
        self.Ty_dev = dev.newBufferWithBytes_length_options_(self.Ty,
                                                             self.Ty.nbytes,
                                                             self.storage)
        self.Tz_dev = dev.newBufferWithBytes_length_options_(self.Tz,
                                                             self.Tz.nbytes,
                                                             self.storage)

    def htod_material_arrays(self, dev):
        """Initialise material coefficient arrays on compute device.

        Args:
            dev: device object.
        """

        self.updatecoeffsE_dev = dev.newBufferWithBytes_length_options_(self.updatecoeffsE,
                                                                        self.updatecoeffsE.nbytes,
                                                                        self.storage)
        self.updatecoeffsH_dev = dev.newBufferWithBytes_length_options_(self.updatecoeffsH,
                                                                        self.updatecoeffsH.nbytes,
                                                                        self.storage)
