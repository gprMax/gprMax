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

    pml_type = CUDAPML

    def __init__(self):
        super().__init__()

        self.gpuarray = import_module("pycuda.gpuarray")

        # Threads per block -used for main electric/magnetic field updates
        self.tpb = (128, 1, 1)
        # Blocks per grid - used for main electric/magnetic field updates
        self.bpg = None

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

    def htod_mat_coeff_arrays(self):
        """Initialise material coefficient arrays on compute device.
        Used by axial plane wave kernels which pass these as pointer arguments
        instead of constant memory — removes 64KB constant memory limit.
        """
        self.updatecoeffsH_dev = self.gpuarray.to_gpu(self.updatecoeffsH)
        self.updatecoeffsE_dev = self.gpuarray.to_gpu(self.updatecoeffsE)

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

    def htod_planewave_arrays(self, dpw):
        """Initialise 1D DPW auxiliary grid arrays on compute device.

        Uploads all plane wave arrays to GPU once at simulation init.
        Arrays remain resident on GPU for the entire simulation.

        Args:
            dpw: DiscretePlaneWave object containing 1D auxiliary grid arrays.
                 dpw.axial == 0 for standard (homogeneous) case.
                 dpw.axial == 1/2/3 for axial case (x/y/z direction).
        """

        # -Standard arrays  (uploaded for both standard and axial) -

        # 1D DPW field arrays - shape [3, n]
        # E_fields[0]=Ex, E_fields[1]=Ey, E_fields[2]=Ez
        dpw.E_fields_dev = self.gpuarray.to_gpu(dpw.E_fields)
        dpw.H_fields_dev = self.gpuarray.to_gpu(dpw.H_fields)

        # PML integral arrays — shape [4, pml_length]
        dpw.Ix_dev = self.gpuarray.to_gpu(dpw.Ix)
        dpw.Iy_dev = self.gpuarray.to_gpu(dpw.Iy)
        dpw.Iz_dev = self.gpuarray.to_gpu(dpw.Iz)

        # PML coefficient arrays - shape [4, pml_length]
        # Used in 1D DPW PML update kernel
        dpw.pml_rhx_dev = self.gpuarray.to_gpu(dpw.pml_rhx)
        dpw.pml_rhy_dev = self.gpuarray.to_gpu(dpw.pml_rhy)
        dpw.pml_rhz_dev = self.gpuarray.to_gpu(dpw.pml_rhz)
        dpw.pml_rex_dev = self.gpuarray.to_gpu(dpw.pml_rex)
        dpw.pml_rey_dev = self.gpuarray.to_gpu(dpw.pml_rey)
        dpw.pml_rez_dev = self.gpuarray.to_gpu(dpw.pml_rez)

        # -Axial-only arrays (uploaded only when dpw.axial != 0)-

        if dpw.axial != 0:

            # Source 1D grid field arrays - shape [3, n]
            # Runs in background material, feeds into main 1D grid at origin_axial
            dpw.E_fields_s_dev = self.gpuarray.to_gpu(dpw.E_fields_s)
            dpw.H_fields_s_dev = self.gpuarray.to_gpu(dpw.H_fields_s)

            # Source grid PML integral arrays - shape [4, pml_length]
            dpw.Ix_s_dev = self.gpuarray.to_gpu(dpw.Ix_s)
            dpw.Iy_s_dev = self.gpuarray.to_gpu(dpw.Iy_s)
            dpw.Iz_s_dev = self.gpuarray.to_gpu(dpw.Iz_s)

            # Second PML region integral arrays - shape [4, pml_length]
            # Axial main grid has two PML regions (start and end)
            dpw.Ix0_dev = self.gpuarray.to_gpu(dpw.Ix0)
            dpw.Iy0_dev = self.gpuarray.to_gpu(dpw.Iy0)
            dpw.Iz0_dev = self.gpuarray.to_gpu(dpw.Iz0)

            # Second PML region coefficient arrays - shape [4, pml_length]
            dpw.pml_rhx0_dev = self.gpuarray.to_gpu(dpw.pml_rhx0)
            dpw.pml_rhy0_dev = self.gpuarray.to_gpu(dpw.pml_rhy0)
            dpw.pml_rhz0_dev = self.gpuarray.to_gpu(dpw.pml_rhz0)
            dpw.pml_rex0_dev = self.gpuarray.to_gpu(dpw.pml_rex0)
            dpw.pml_rey0_dev = self.gpuarray.to_gpu(dpw.pml_rey0)
            dpw.pml_rez0_dev = self.gpuarray.to_gpu(dpw.pml_rez0)

            # 1D material ID array - shape [6, n]
            # Maps each position j in the 1D grid to a material ID
            # Used by axial 1D update kernel for per-cell coefficient lookup
            # updatecoeffsH[ID[component, j], col]
            dpw.ID_dev = self.gpuarray.to_gpu(dpw.ID)

        # -Dispersive polarization arrays (uploaded when dpw.dispersive)-
        # Shape [max_poles, n], flattened row-major for kernel access as T[pole*N + j]
        if dpw.dispersive:
            dpw.Px_dev = self.gpuarray.to_gpu(dpw.Px)
            dpw.Py_dev = self.gpuarray.to_gpu(dpw.Py)
            dpw.Pz_dev = self.gpuarray.to_gpu(dpw.Pz)
            if dpw.axial != 0:
                dpw.Px_s_dev = self.gpuarray.to_gpu(dpw.Px_s)
                dpw.Py_s_dev = self.gpuarray.to_gpu(dpw.Py_s)
                dpw.Pz_s_dev = self.gpuarray.to_gpu(dpw.Pz_s)
