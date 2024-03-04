# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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
from importlib import import_module

import humanize
import numpy as np
from jinja2 import Environment, PackageLoader

from gprMax import config
from gprMax.cuda_opencl import (
    knl_fields_updates,
    knl_snapshots,
    knl_source_updates,
    knl_store_outputs,
)
from gprMax.grid.cuda_grid import CUDAGrid
from gprMax.receivers import dtoh_rx_array, htod_rx_arrays
from gprMax.snapshots import Snapshot, dtoh_snapshot_array, htod_snapshot_array
from gprMax.sources import htod_src_arrays
from gprMax.updates.updates import Updates
from gprMax.utilities.utilities import round32

logger = logging.getLogger(__name__)


class CUDAUpdates(Updates):
    """Defines update functions for GPU-based (CUDA) solver."""

    def __init__(self, G: CUDAGrid):
        """
        Args:
            G: CUDAGrid class describing a grid in a model.
        """

        self.grid = G

        # Import PyCUDA modules
        self.drv = import_module("pycuda.driver")
        self.source_module = getattr(import_module("pycuda.compiler"), "SourceModule")
        self.drv.init()

        # Create device handle and context on specific GPU device (and make it current context)
        self.dev = config.get_model_config().device["dev"]
        self.ctx = self.dev.make_context()

        # Set common substitutions for use in kernels
        # Substitutions in function arguments
        self.subs_name_args = {
            "REAL": config.sim_config.dtypes["C_float_or_double"],
            "COMPLEX": config.get_model_config().materials["dispersiveCdtype"],
        }
        # Substitutions in function bodies
        self.subs_func = {
            "REAL": config.sim_config.dtypes["C_float_or_double"],
            "CUDA_IDX": "int i = blockIdx.x * blockDim.x + threadIdx.x;",
            "NX_FIELDS": self.grid.nx + 1,
            "NY_FIELDS": self.grid.ny + 1,
            "NZ_FIELDS": self.grid.nz + 1,
            "NX_ID": self.grid.ID.shape[1],
            "NY_ID": self.grid.ID.shape[2],
            "NZ_ID": self.grid.ID.shape[3],
        }

        # Enviroment for templating kernels
        self.env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))

        # Initialise arrays on GPU, prepare kernels, and get kernel functions
        self._set_macros()
        self._set_field_knls()
        if self.grid.pmls["slabs"]:
            self._set_pml_knls()
        if self.grid.rxs:
            self._set_rx_knl()
        if self.grid.voltagesources + self.grid.hertziandipoles + self.grid.magneticdipoles:
            self._set_src_knls()
        if self.grid.snapshots:
            self._set_snapshot_knl()

    def _build_knl(self, knl_func, subs_name_args, subs_func):
        """Builds a CUDA kernel from templates: 1) function name and args;
            and 2) function (kernel) body.

        Args:
            knl_func: dict containing templates for function name and args,
                        and function body.
            subs_name_args: dict containing substitutions to be used with
                                function name and args.
            subs_func: dict containing substitutions to be used with function
                        (kernel) body.

        Returns:
            knl: string with complete kernel
        """

        name_plus_args = knl_func["args_cuda"].substitute(subs_name_args)
        func_body = knl_func["func"].substitute(subs_func)
        knl = self.knl_common + "\n" + name_plus_args + "{" + func_body + "}"

        return knl

    def _set_macros(self):
        """Common macros to be used in kernels."""

        # Set specific values for any dispersive materials
        if config.get_model_config().materials["maxpoles"] > 0:
            NY_MATDISPCOEFFS = self.grid.updatecoeffsdispersive.shape[1]
            NX_T = self.grid.Tx.shape[1]
            NY_T = self.grid.Tx.shape[2]
            NZ_T = self.grid.Tx.shape[3]
        else:  # Set to one any substitutions for dispersive materials.
            NY_MATDISPCOEFFS = 1
            NX_T = 1
            NY_T = 1
            NZ_T = 1

        self.knl_common = self.env.get_template("knl_common_cuda.tmpl").render(
            REAL=config.sim_config.dtypes["C_float_or_double"],
            N_updatecoeffsE=self.grid.updatecoeffsE.size,
            N_updatecoeffsH=self.grid.updatecoeffsH.size,
            NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1],
            NY_MATDISPCOEFFS=NY_MATDISPCOEFFS,
            NX_FIELDS=self.grid.nx + 1,
            NY_FIELDS=self.grid.ny + 1,
            NZ_FIELDS=self.grid.nz + 1,
            NX_ID=self.grid.ID.shape[1],
            NY_ID=self.grid.ID.shape[2],
            NZ_ID=self.grid.ID.shape[3],
            NX_T=NX_T,
            NY_T=NY_T,
            NZ_T=NZ_T,
            NY_RXCOORDS=3,
            NX_RXS=6,
            NY_RXS=self.grid.iterations,
            NZ_RXS=len(self.grid.rxs),
            NY_SRCINFO=4,
            NY_SRCWAVES=self.grid.iterations,
            NX_SNAPS=Snapshot.nx_max,
            NY_SNAPS=Snapshot.ny_max,
            NZ_SNAPS=Snapshot.nz_max,
        )

    def _set_field_knls(self):
        """Electric and magnetic field updates - prepares kernels, and
        gets kernel functions.
        """

        bld = self._build_knl(
            knl_fields_updates.update_electric, self.subs_name_args, self.subs_func
        )
        knlE = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
        self.update_electric_dev = knlE.get_function("update_electric")

        bld = self._build_knl(
            knl_fields_updates.update_magnetic, self.subs_name_args, self.subs_func
        )
        knlH = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
        self.update_magnetic_dev = knlH.get_function("update_magnetic")

        self._copy_mat_coeffs(knlE, knlH)

        # If there are any dispersive materials (updates are split into two
        # parts as they require present and updated electric field values).
        if config.get_model_config().materials["maxpoles"] > 0:
            self.subs_func.update(
                {
                    "REAL": config.sim_config.dtypes["C_float_or_double"],
                    "REALFUNC": config.get_model_config().materials["crealfunc"],
                    "NX_T": self.grid.Tx.shape[1],
                    "NY_T": self.grid.Tx.shape[2],
                    "NZ_T": self.grid.Tx.shape[3],
                }
            )

            bld = self._build_knl(
                knl_fields_updates.update_electric_dispersive_A, self.subs_name_args, self.subs_func
            )
            knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            self.dispersive_update_a = knl.get_function("update_electric_dispersive_A")
            self._copy_mat_coeffs(knl, knl)

            bld = self._build_knl(
                knl_fields_updates.update_electric_dispersive_B, self.subs_name_args, self.subs_func
            )
            knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            self.dispersive_update_b = knl.get_function("update_electric_dispersive_B")
            self._copy_mat_coeffs(knl, knl)

        # Set blocks per grid and initialise field arrays on GPU
        self.grid.set_blocks_per_grid()
        self.grid.htod_geometry_arrays()
        self.grid.htod_field_arrays()
        if config.get_model_config().materials["maxpoles"] > 0:
            self.grid.htod_dispersive_arrays()

    def _set_pml_knls(self):
        """PMLS - prepares kernels and gets kernel functions."""
        knl_pml_updates_electric = import_module(
            "gprMax.cuda_opencl.knl_pml_updates_electric_" + self.grid.pmls["formulation"]
        )
        knl_pml_updates_magnetic = import_module(
            "gprMax.cuda_opencl.knl_pml_updates_magnetic_" + self.grid.pmls["formulation"]
        )

        # Initialise arrays on GPU, set block per grid, and get kernel functions
        for pml in self.grid.pmls["slabs"]:
            pml.htod_field_arrays()
            pml.set_blocks_per_grid()
            knl_name = f"order{len(pml.CFS)}_{pml.direction}"
            self.subs_name_args["FUNC"] = knl_name

            knl_electric = getattr(knl_pml_updates_electric, knl_name)
            bld = self._build_knl(knl_electric, self.subs_name_args, self.subs_func)
            knlE = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            pml.update_electric_dev = knlE.get_function(knl_name)

            knl_magnetic = getattr(knl_pml_updates_magnetic, knl_name)
            bld = self._build_knl(knl_magnetic, self.subs_name_args, self.subs_func)
            knlH = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            pml.update_magnetic_dev = knlH.get_function(knl_name)

            # Copy material coefficient arrays to constant memory of GPU - must
            # be done for each kernel
            self._copy_mat_coeffs(knlE, knlH)

    def _set_rx_knl(self):
        """Receivers - initialises arrays on GPU, prepares kernel and gets kernel
        function.
        """
        self.rxcoords_dev, self.rxs_dev = htod_rx_arrays(self.grid)

        self.subs_func.update(
            {
                "REAL": config.sim_config.dtypes["C_float_or_double"],
                "NY_RXCOORDS": 3,
                "NX_RXS": 6,
                "NY_RXS": self.grid.iterations,
                "NZ_RXS": len(self.grid.rxs),
            }
        )

        bld = self._build_knl(knl_store_outputs.store_outputs, self.subs_name_args, self.subs_func)
        knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
        self.store_outputs_dev = knl.get_function("store_outputs")

    def _set_src_knls(self):
        """Sources - initialises arrays on GPU, prepares kernel and gets kernel
        function.
        """
        self.subs_func.update({"NY_SRCINFO": 4, "NY_SRCWAVES": self.grid.iteration})

        if self.grid.hertziandipoles:
            (
                self.srcinfo1_hertzian_dev,
                self.srcinfo2_hertzian_dev,
                self.srcwaves_hertzian_dev,
            ) = htod_src_arrays(self.grid.hertziandipoles, self.grid)
            bld = self._build_knl(
                knl_source_updates.update_hertzian_dipole, self.subs_name_args, self.subs_func
            )
            knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            self.update_hertzian_dipole_dev = knl.get_function("update_hertzian_dipole")
        if self.grid.magneticdipoles:
            (
                self.srcinfo1_magnetic_dev,
                self.srcinfo2_magnetic_dev,
                self.srcwaves_magnetic_dev,
            ) = htod_src_arrays(self.grid.magneticdipoles, self.grid)
            bld = self._build_knl(
                knl_source_updates.update_magnetic_dipole, self.subs_name_args, self.subs_func
            )
            knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            self.update_magnetic_dipole_dev = knl.get_function("update_magnetic_dipole")
        if self.grid.voltagesources:
            (
                self.srcinfo1_voltage_dev,
                self.srcinfo2_voltage_dev,
                self.srcwaves_voltage_dev,
            ) = htod_src_arrays(self.grid.voltagesources, self.grid)
            bld = self._build_knl(
                knl_source_updates.update_voltage_source, self.subs_name_args, self.subs_func
            )
            knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            self.update_voltage_source_dev = knl.get_function("update_voltage_source")

        self._copy_mat_coeffs(knl, knl)

    def _set_snapshot_knl(self):
        """Snapshots - initialises arrays on GPU, prepares kernel and gets kernel
        function.
        """
        (
            self.snapEx_dev,
            self.snapEy_dev,
            self.snapEz_dev,
            self.snapHx_dev,
            self.snapHy_dev,
            self.snapHz_dev,
        ) = htod_snapshot_array(self.grid)

        self.subs_func.update(
            {
                "REAL": config.sim_config.dtypes["C_float_or_double"],
                "NX_SNAPS": Snapshot.nx_max,
                "NY_SNAPS": Snapshot.ny_max,
                "NZ_SNAPS": Snapshot.nz_max,
            }
        )

        bld = self._build_knl(knl_snapshots.store_snapshot, self.subs_name_args, self.subs_func)
        knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
        self.store_snapshot_dev = knl.get_function("store_snapshot")

    def _copy_mat_coeffs(self, knlE, knlH):
        """Copies material coefficient arrays to constant memory of GPU
            (must be <64KB).

        Args:
            knlE: kernel for electric field.
            knlH: kernel for magnetic field.
        """

        # Check if coefficient arrays will fit on constant memory of GPU
        if (
            self.grid.updatecoeffsE.nbytes + self.grid.updatecoeffsH.nbytes
            > config.get_model_config().device["dev"].total_constant_memory
        ):
            device = config.get_model_config().device["dev"]
            logger.exception(
                f"Too many materials in the model to fit onto "
                + f"constant memory of size {humanize.naturalsize(device.total_constant_memory)} "
                + f"on {device.deviceID}: {' '.join(device.name().split())}"
            )
            raise ValueError

        updatecoeffsE = knlE.get_global("updatecoeffsE")[0]
        updatecoeffsH = knlH.get_global("updatecoeffsH")[0]
        self.drv.memcpy_htod(updatecoeffsE, self.grid.updatecoeffsE)
        self.drv.memcpy_htod(updatecoeffsH, self.grid.updatecoeffsH)

    def store_outputs(self):
        """Stores field component values for every receiver."""
        if self.grid.rxs:
            self.store_outputs_dev(
                np.int32(len(self.grid.rxs)),
                np.int32(self.grid.iteration),
                self.rxcoords_dev.gpudata,
                self.rxs_dev.gpudata,
                self.grid.Ex_dev.gpudata,
                self.grid.Ey_dev.gpudata,
                self.grid.Ez_dev.gpudata,
                self.grid.Hx_dev.gpudata,
                self.grid.Hy_dev.gpudata,
                self.grid.Hz_dev.gpudata,
                block=(1, 1, 1),
                grid=(round32(len(self.grid.rxs)), 1, 1),
            )

    def store_snapshots(self, iteration):
        """Stores any snapshots.

        Args:
            iteration: int for iteration number.
        """

        for i, snap in enumerate(self.grid.snapshots):
            if snap.time == iteration + 1:
                snapno = 0 if config.get_model_config().device["snapsgpu2cpu"] else i
                self.store_snapshot_dev(
                    np.int32(snapno),
                    np.int32(snap.xs),
                    np.int32(snap.xf),
                    np.int32(snap.ys),
                    np.int32(snap.yf),
                    np.int32(snap.zs),
                    np.int32(snap.zf),
                    np.int32(snap.dx),
                    np.int32(snap.dy),
                    np.int32(snap.dz),
                    self.grid.Ex_dev.gpudata,
                    self.grid.Ey_dev.gpudata,
                    self.grid.Ez_dev.gpudata,
                    self.grid.Hx_dev.gpudata,
                    self.grid.Hy_dev.gpudata,
                    self.grid.Hz_dev.gpudata,
                    self.snapEx_dev.gpudata,
                    self.snapEy_dev.gpudata,
                    self.snapEz_dev.gpudata,
                    self.snapHx_dev.gpudata,
                    self.snapHy_dev.gpudata,
                    self.snapHz_dev.gpudata,
                    block=Snapshot.tpb,
                    grid=Snapshot.bpg,
                )
                if config.get_model_config().device["snapsgpu2cpu"]:
                    dtoh_snapshot_array(
                        self.snapEx_dev.get(),
                        self.snapEy_dev.get(),
                        self.snapEz_dev.get(),
                        self.snapHx_dev.get(),
                        self.snapHy_dev.get(),
                        self.snapHz_dev.get(),
                        0,
                        snap,
                    )

    def update_magnetic(self):
        """Updates magnetic field components."""
        self.update_magnetic_dev(
            np.int32(self.grid.nx),
            np.int32(self.grid.ny),
            np.int32(self.grid.nz),
            self.grid.ID_dev.gpudata,
            self.grid.Hx_dev.gpudata,
            self.grid.Hy_dev.gpudata,
            self.grid.Hz_dev.gpudata,
            self.grid.Ex_dev.gpudata,
            self.grid.Ey_dev.gpudata,
            self.grid.Ez_dev.gpudata,
            block=self.grid.tpb,
            grid=self.grid.bpg,
        )

    def update_magnetic_pml(self):
        """Updates magnetic field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_magnetic()

    def update_magnetic_sources(self):
        """Updates magnetic field components from sources."""
        if self.grid.magneticdipoles:
            self.update_magnetic_dipole_dev(
                np.int32(len(self.grid.magneticdipoles)),
                np.int32(self.grid.iteration),
                config.sim_config.dtypes["float_or_double"](self.grid.dx),
                config.sim_config.dtypes["float_or_double"](self.grid.dy),
                config.sim_config.dtypes["float_or_double"](self.grid.dz),
                self.srcinfo1_magnetic_dev.gpudata,
                self.srcinfo2_magnetic_dev.gpudata,
                self.srcwaves_magnetic_dev.gpudata,
                self.grid.ID_dev.gpudata,
                self.grid.Hx_dev.gpudata,
                self.grid.Hy_dev.gpudata,
                self.grid.Hz_dev.gpudata,
                block=(1, 1, 1),
                grid=(round32(len(self.grid.magneticdipoles)), 1, 1),
            )

    def update_electric_a(self):
        """Updates electric field components."""
        # All materials are non-dispersive so do standard update.
        if config.get_model_config().materials["maxpoles"] == 0:
            self.update_electric_dev(
                np.int32(self.grid.nx),
                np.int32(self.grid.ny),
                np.int32(self.grid.nz),
                self.grid.ID_dev.gpudata,
                self.grid.Ex_dev.gpudata,
                self.grid.Ey_dev.gpudata,
                self.grid.Ez_dev.gpudata,
                self.grid.Hx_dev.gpudata,
                self.grid.Hy_dev.gpudata,
                self.grid.Hz_dev.gpudata,
                block=self.grid.tpb,
                grid=self.grid.bpg,
            )

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            self.dispersive_update_a(
                np.int32(self.grid.nx),
                np.int32(self.grid.ny),
                np.int32(self.grid.nz),
                np.int32(config.get_model_config().materials["maxpoles"]),
                self.grid.updatecoeffsdispersive_dev.gpudata,
                self.grid.Tx_dev.gpudata,
                self.grid.Ty_dev.gpudata,
                self.grid.Tz_dev.gpudata,
                self.grid.ID_dev.gpudata,
                self.grid.Ex_dev.gpudata,
                self.grid.Ey_dev.gpudata,
                self.grid.Ez_dev.gpudata,
                self.grid.Hx_dev.gpudata,
                self.grid.Hy_dev.gpudata,
                self.grid.Hz_dev.gpudata,
                block=self.grid.tpb,
                grid=self.grid.bpg,
            )

    def update_electric_pml(self):
        """Updates electric field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_electric()

    def update_electric_sources(self):
        """Updates electric field components from sources -
        update any Hertzian dipole sources last.
        """
        if self.grid.voltagesources:
            self.update_voltage_source_dev(
                np.int32(len(self.grid.voltagesources)),
                np.int32(self.grid.iteration),
                config.sim_config.dtypes["float_or_double"](self.grid.dx),
                config.sim_config.dtypes["float_or_double"](self.grid.dy),
                config.sim_config.dtypes["float_or_double"](self.grid.dz),
                self.srcinfo1_voltage_dev.gpudata,
                self.srcinfo2_voltage_dev.gpudata,
                self.srcwaves_voltage_dev.gpudata,
                self.grid.ID_dev.gpudata,
                self.grid.Ex_dev.gpudata,
                self.grid.Ey_dev.gpudata,
                self.grid.Ez_dev.gpudata,
                block=(1, 1, 1),
                grid=(round32(len(self.grid.voltagesources)), 1, 1),
            )

        if self.grid.hertziandipoles:
            self.update_hertzian_dipole_dev(
                np.int32(len(self.grid.hertziandipoles)),
                np.int32(self.grid.iteration),
                config.sim_config.dtypes["float_or_double"](self.grid.dx),
                config.sim_config.dtypes["float_or_double"](self.grid.dy),
                config.sim_config.dtypes["float_or_double"](self.grid.dz),
                self.srcinfo1_hertzian_dev.gpudata,
                self.srcinfo2_hertzian_dev.gpudata,
                self.srcwaves_hertzian_dev.gpudata,
                self.grid.ID_dev.gpudata,
                self.grid.Ex_dev.gpudata,
                self.grid.Ey_dev.gpudata,
                self.grid.Ez_dev.gpudata,
                block=(1, 1, 1),
                grid=(round32(len(self.grid.hertziandipoles)), 1, 1),
            )

        self.grid.iteration += 1

    def update_electric_b(self):
        """If there are any dispersive materials do 2nd part of dispersive
        update - it is split into two parts as it requires present and
        updated electric field values. Therefore it can only be completely
        updated after the electric field has been updated by the PML and
        source updates.
        """
        if config.get_model_config().materials["maxpoles"] > 0:
            self.dispersive_update_b(
                np.int32(self.grid.nx),
                np.int32(self.grid.ny),
                np.int32(self.grid.nz),
                np.int32(config.get_model_config().materials["maxpoles"]),
                self.grid.updatecoeffsdispersive_dev.gpudata,
                self.grid.Tx_dev.gpudata,
                self.grid.Ty_dev.gpudata,
                self.grid.Tz_dev.gpudata,
                self.grid.ID_dev.gpudata,
                self.grid.Ex_dev.gpudata,
                self.grid.Ey_dev.gpudata,
                self.grid.Ez_dev.gpudata,
                block=self.grid.tpb,
                grid=self.grid.bpg,
            )

    def time_start(self):
        """Starts event timers used to calculate solving time for model."""
        self.iterstart = self.drv.Event()
        self.iterend = self.drv.Event()
        self.iterstart.record()
        self.iterstart.synchronize()

    def calculate_memory_used(self, iteration):
        """Calculates memory used on last iteration.

        Args:
            iteration: int for iteration number.

        Returns:
            Memory (RAM) used on GPU.
        """
        if iteration == self.grid.iterations - 1:
            # Total minus free memory in current context
            return self.drv.mem_get_info()[1] - self.drv.mem_get_info()[0]

    def calculate_solve_time(self):
        """Calculates solving time for model."""
        self.iterend.record()
        self.iterend.synchronize()
        return self.iterstart.time_till(self.iterend) * 1e-3

    def finalise(self):
        """Copies data from GPU back to CPU to save to file(s)."""
        # Copy output from receivers array back to correct receiver objects
        if self.grid.rxs:
            dtoh_rx_array(self.rxs_dev.get(), self.rxcoords_dev.get(), self.grid)

        # Copy data from any snapshots back to correct snapshot objects
        if self.grid.snapshots and not config.get_model_config().device["snapsgpu2cpu"]:
            for i, snap in enumerate(self.grid.snapshots):
                dtoh_snapshot_array(
                    self.snapEx_dev.get(),
                    self.snapEy_dev.get(),
                    self.snapEz_dev.get(),
                    self.snapHx_dev.get(),
                    self.snapHy_dev.get(),
                    self.snapHz_dev.get(),
                    i,
                    snap,
                )

    def cleanup(self):
        """Cleanup GPU context."""
        # Remove context from top of stack and clear
        self.ctx.pop()
        self.ctx = None
