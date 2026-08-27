# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

from importlib import import_module

import numpy as np

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.pml import MetalPML


class MetalBufferView:
    """Byte-offset view into a shared Metal buffer."""

    def __init__(self, buffer, offset=0):
        self.buffer = buffer
        self.offset = int(offset)


class MetalArray(MetalBufferView):
    """Minimal first-axis slicing used by the plane-wave kernels."""

    def __init__(self, buffer, shape, dtype):
        super().__init__(buffer, 0)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Metal device arrays support integer first-axis views only")
        if index < 0:
            index += self.shape[0]
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)
        stride = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        return MetalBufferView(self.buffer, index * stride)


class MetalGrid(FDTDGrid):
    """Additional grid methods for solving on compute device using Apple Metal."""

    pml_type = MetalPML

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

    def htod_mat_coeff_arrays(self, _queue=None):
        """OpenCL-compatible alias used by shared plane-wave orchestration."""

        from gprMax import config

        self.htod_material_arrays(config.get_model_config().device["dev"])

    def htod_planewave_arrays(self, dpw, _queue=None):
        """Upload all auxiliary plane-wave arrays as offset-capable buffers."""

        from gprMax import config

        dev = config.get_model_config().device["dev"]

        def upload(values):
            values = np.ascontiguousarray(values)
            buffer = dev.newBufferWithBytes_length_options_(
                values, values.nbytes, self.storage
            )
            return MetalArray(buffer, values.shape, values.dtype)

        for name in (
            "E_fields", "H_fields", "Ix", "Iy", "Iz", "pml_rhx",
            "pml_rhy", "pml_rhz", "pml_rex", "pml_rey", "pml_rez",
        ):
            setattr(dpw, f"{name}_dev", upload(getattr(dpw, name)))

        real = dpw.E_fields.dtype
        source_e = np.asarray(
            dpw.waveformvalues_wholedt * dpw.projections[None, :3, None],
            dtype=real,
            order="C",
        )
        source_h = np.asarray(
            dpw.waveformvalues_halfdt * dpw.projections[None, 3:, None],
            dtype=real,
            order="C",
        )
        if source_e.size > np.iinfo(np.int32).max:
            raise ValueError("Plane-wave source arrays exceed the signed 32-bit index range.")
        dpw.source_e_dev = upload(source_e)
        dpw.source_h_dev = upload(source_h)

        if dpw.axial != 0:
            for name in (
                "E_fields_s", "H_fields_s", "Ix_s", "Iy_s", "Iz_s",
                "Ix0", "Iy0", "Iz0", "pml_rhx0", "pml_rhy0",
                "pml_rhz0", "pml_rex0", "pml_rey0", "pml_rez0", "ID",
            ):
                setattr(dpw, f"{name}_dev", upload(getattr(dpw, name)))
        if dpw.dispersive:
            for name in ("Px", "Py", "Pz"):
                setattr(dpw, f"{name}_dev", upload(getattr(dpw, name)))
            if dpw.axial != 0:
                for name in ("Px_s", "Py_s", "Pz_s"):
                    setattr(dpw, f"{name}_dev", upload(getattr(dpw, name)))
