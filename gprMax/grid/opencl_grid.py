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
from gprMax.pml import OpenCLPML


class OpenCLGrid(FDTDGrid):
    """Additional grid methods for solving on compute device using OpenCL."""

    pml_type = OpenCLPML

    def __init__(self):
        super().__init__()

        self.clarray = import_module("pyopencl.array")

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

    def htod_mat_coeff_arrays(self, queue):
        """Initialise plane-wave material coefficient arrays on the device."""

        self.updatecoeffsH_dev = self.clarray.to_device(queue, self.updatecoeffsH)
        self.updatecoeffsE_dev = self.clarray.to_device(queue, self.updatecoeffsE)

    def htod_planewave_arrays(self, dpw, queue):
        """Initialise all auxiliary plane-wave arrays on the device."""

        for name in (
            "E_fields",
            "H_fields",
            "Ix",
            "Iy",
            "Iz",
            "pml_rhx",
            "pml_rhy",
            "pml_rhz",
            "pml_rex",
            "pml_rey",
            "pml_rez",
        ):
            setattr(dpw, f"{name}_dev", self.clarray.to_device(queue, getattr(dpw, name)))

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
        dpw.source_e_dev = self.clarray.to_device(queue, source_e)
        dpw.source_h_dev = self.clarray.to_device(queue, source_h)

        if dpw.axial != 0:
            for name in (
                "E_fields_s",
                "H_fields_s",
                "Ix_s",
                "Iy_s",
                "Iz_s",
                "Ix0",
                "Iy0",
                "Iz0",
                "pml_rhx0",
                "pml_rhy0",
                "pml_rhz0",
                "pml_rex0",
                "pml_rey0",
                "pml_rez0",
                "ID",
            ):
                setattr(dpw, f"{name}_dev", self.clarray.to_device(queue, getattr(dpw, name)))

        if dpw.dispersive:
            for name in ("Px", "Py", "Pz"):
                setattr(dpw, f"{name}_dev", self.clarray.to_device(queue, getattr(dpw, name)))
            if dpw.axial != 0:
                for name in ("Px_s", "Py_s", "Pz_s"):
                    setattr(
                        dpw,
                        f"{name}_dev",
                        self.clarray.to_device(queue, getattr(dpw, name)),
                    )
