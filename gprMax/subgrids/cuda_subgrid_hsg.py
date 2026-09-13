# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom

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

"""CUDA-capable HSG subgrid.

SubGridHSG carries the interface methods but calls Cython; CUDAGrid carries
the device arrays but knows nothing about subgrids. CUDASubGridHSG is both:
it inherits the subgrid geometry and setup from one and the device array
handling from the other, and overrides the four interface methods to launch
kernels instead.

Kept in its own module so that importing subgrid_hsg.py never pulls in
PyCUDA - the CPU path must not gain a GPU dependency.
"""

import logging

from gprMax.grid.cuda_grid import CUDAGrid
from gprMax.subgrids.cuda_subgrid_interface import (
    CUDASubgridInterface,
    SubgridGeometry,
)
from gprMax.subgrids.subgrid_hsg import SubGridHSG

logger = logging.getLogger(__name__)

FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class CUDASubGridHSG(SubGridHSG, CUDAGrid):
    """HSG subgrid whose interface corrections run on the GPU.

    MRO is CUDASubGridHSG -> SubGridHSG -> SubGridBaseGrid -> CUDAGrid ->
    FDTDGrid, so SubGridBaseGrid.__init__'s super().__init__() reaches
    CUDAGrid.__init__ and the gpuarray handle, tpb and bpg are set up as for
    any other CUDA grid.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iface = None
        self._main_grid = None

    # ------------------------------------------------------------ setup
    def setup_cuda_interface(self, kernels, main_grid):
        """Build the launch interface. Called once, after both grids have
        uploaded their arrays.

        The kernels are built by CUDASubgridUpdater._set_subgrid_knls() and
        passed in, so this class never compiles anything itself - it only
        needs to know the geometry.

        Args:
            kernels: dict of built CUDA functions, keyed 'update_is',
                        'update_electric_os', 'update_magnetic_os'.
            main_grid: the CUDAGrid this subgrid sits inside.
        """
        geom = SubgridGeometry(
            nwx=self.nwx, nwy=self.nwy, nwz=self.nwz,
            ratio=self.ratio,
            is_os_sep=self.is_os_sep,
            n_boundary_cells=self.n_boundary_cells,
            i0=self.i0, j0=self.j0, k0=self.k0,
            i1=self.i1, j1=self.j1, k1=self.k1,
            sub_shape=(self.nx + 1, self.ny + 1, self.nz + 1),
            main_shape=(main_grid.nx + 1, main_grid.ny + 1, main_grid.nz + 1),
            sub_id=self.ID_dev.gpudata,
            main_id=main_grid.ID_dev.gpudata,
            id_lookup=self.IDlookup,
            main_id_lookup=main_grid.IDlookup,
            ny_matcoeffs=self.updatecoeffsE.shape[1],
        )
        # Turns a silently out-of-bounds read into an error at setup
        geom.check()

        self.iface = CUDASubgridInterface(kernels, geom, tpb=self.tpb[0])
        self._main_grid = main_grid

    def _sub_fields(self):
        return {n: getattr(self, f"{n}_dev").gpudata for n in FIELD_NAMES}

    def _main_fields(self):
        return {n: getattr(self._main_grid, f"{n}_dev").gpudata
                for n in FIELD_NAMES}

    # ------------------------------------------------------------ IS / OS
    #
    # Same four entry points as SubGridHSG, same call order, same arguments.
    # The phase logic in HSGPhaseMixin cannot tell which backend it is
    # driving.

    def update_magnetic_is(self, precursors):
        self.iface.update_magnetic_is(
            self.updatecoeffsH_dev.gpudata,
            self._sub_fields(),
            precursors.device_slices(),
        )

    def update_electric_is(self, precursors):
        self.iface.update_electric_is(
            self.updatecoeffsE_dev.gpudata,
            self._sub_fields(),
            precursors.device_slices(),
        )

    def update_electric_os(self, main_grid):
        self.iface.update_electric_os(
            main_grid.updatecoeffsE_dev.gpudata,
            self._main_fields(),
            self._sub_fields(),
        )

    def update_magnetic_os(self, main_grid):
        self.iface.update_magnetic_os(
            main_grid.updatecoeffsH_dev.gpudata,
            self._main_fields(),
            self._sub_fields(),
        )
