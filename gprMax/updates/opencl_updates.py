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

import numpy as np
from jinja2 import Environment, PackageLoader

from gprMax import config
from gprMax.cuda_opencl import knl_fields_updates, knl_snapshots, knl_source_updates, knl_store_outputs
from gprMax.receivers import dtoh_rx_array, htod_rx_arrays
from gprMax.snapshots import Snapshot, dtoh_snapshot_array, htod_snapshot_array
from gprMax.sources import htod_src_arrays

logger = logging.getLogger(__name__)


class OpenCLUpdates:
    """Defines update functions for OpenCL-based solver."""

    def __init__(self, G):
        """
        Args:
            G: OpenCLGrid class describing a grid in a model.
        """

        self.grid = G

        # Import pyopencl module
        self.cl = import_module("pyopencl")
        self.elwiseknl = getattr(import_module("pyopencl.elementwise"), "ElementwiseKernel")

        # Select device, create context and command queue
        self.dev = config.get_model_config().device["dev"]
        self.ctx = self.cl.Context(devices=[self.dev])
        self.queue = self.cl.CommandQueue(self.ctx, properties=self.cl.command_queue_properties.PROFILING_ENABLE)

        # Enviroment for templating kernels
        self.env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))

        # Initialise arrays on device, prepare kernels, and get kernel functions
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

        self.knl_common = self.env.get_template("knl_common_opencl.tmpl").render(
            updatecoeffsE=self.grid.updatecoeffsE.ravel(),
            updatecoeffsH=self.grid.updatecoeffsH.ravel(),
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

        subs = {
            "CUDA_IDX": "",
            "NX_FIELDS": self.grid.nx + 1,
            "NY_FIELDS": self.grid.ny + 1,
            "NZ_FIELDS": self.grid.nz + 1,
            "NX_ID": self.grid.ID.shape[1],
            "NY_ID": self.grid.ID.shape[2],
            "NZ_ID": self.grid.ID.shape[3],
        }

        self.update_electric_dev = self.elwiseknl(
            self.ctx,
            knl_fields_updates.update_electric["args_opencl"].substitute(
                {"REAL": config.sim_config.dtypes["C_float_or_double"]}
            ),
            knl_fields_updates.update_electric["func"].substitute(subs),
            "update_electric",
            preamble=self.knl_common,
            options=config.sim_config.devices["compiler_opts"],
        )

        self.update_magnetic_dev = self.elwiseknl(
            self.ctx,
            knl_fields_updates.update_magnetic["args_opencl"].substitute(
                {"REAL": config.sim_config.dtypes["C_float_or_double"]}
            ),
            knl_fields_updates.update_magnetic["func"].substitute(subs),
            "update_magnetic",
            preamble=self.knl_common,
            options=config.sim_config.devices["compiler_opts"],
        )

        # If there are any dispersive materials (updates are split into two
        # parts as they require present and updated electric field values).
        if config.get_model_config().materials["maxpoles"] > 0:
            subs = {
                "CUDA_IDX": "",
                "REAL": config.sim_config.dtypes["C_float_or_double"],
                "REALFUNC": config.get_model_config().materials["crealfunc"],
                "NX_FIELDS": self.grid.nx + 1,
                "NY_FIELDS": self.grid.ny + 1,
                "NZ_FIELDS": self.grid.nz + 1,
                "NX_ID": self.grid.ID.shape[1],
                "NY_ID": self.grid.ID.shape[2],
                "NZ_ID": self.grid.ID.shape[3],
                "NX_T": self.grid.Tx.shape[1],
                "NY_T": self.grid.Tx.shape[2],
                "NZ_T": self.grid.Tx.shape[3],
            }

            self.dispersive_update_a = self.elwiseknl(
                self.ctx,
                knl_fields_updates.update_electric_dispersive_A["args_opencl"].substitute(
                    {
                        "REAL": config.sim_config.dtypes["C_float_or_double"],
                        "COMPLEX": config.get_model_config().materials["dispersiveCdtype"],
                    }
                ),
                knl_fields_updates.update_electric_dispersive_A["func"].substitute(subs),
                "update_electric_dispersive_A",
                preamble=self.knl_common,
                options=config.sim_config.devices["compiler_opts"],
            )

            self.dispersive_update_b = self.elwiseknl(
                self.ctx,
                knl_fields_updates.update_electric_dispersive_B["args_opencl"].substitute(
                    {
                        "REAL": config.sim_config.dtypes["C_float_or_double"],
                        "COMPLEX": config.get_model_config().materials["dispersiveCdtype"],
                    }
                ),
                knl_fields_updates.update_electric_dispersive_B["func"].substitute(subs),
                "update_electric_dispersive_B",
                preamble=self.knl_common,
                options=config.sim_config.devices["compiler_opts"],
            )

        # Initialise field arrays on compute device
        self.grid.htod_geometry_arrays(self.queue)
        self.grid.htod_field_arrays(self.queue)
        if config.get_model_config().materials["maxpoles"] > 0:
            self.grid.htod_dispersive_arrays(self.queue)

    def _set_pml_knls(self):
        """PMLS - prepares kernels and gets kernel functions."""
        knl_pml_updates_electric = import_module(
            "gprMax.cuda_opencl.knl_pml_updates_electric_" + self.grid.pmls["formulation"]
        )
        knl_pml_updates_magnetic = import_module(
            "gprMax.cuda_opencl.knl_pml_updates_magnetic_" + self.grid.pmls["formulation"]
        )

        subs = {
            "CUDA_IDX": "",
            "REAL": config.sim_config.dtypes["C_float_or_double"],
            "NX_FIELDS": self.grid.nx + 1,
            "NY_FIELDS": self.grid.ny + 1,
            "NZ_FIELDS": self.grid.nz + 1,
            "NX_ID": self.grid.ID.shape[1],
            "NY_ID": self.grid.ID.shape[2],
            "NZ_ID": self.grid.ID.shape[3],
        }

        # Set workgroup size, initialise arrays on compute device, and get
        # kernel functions
        for pml in self.grid.pmls["slabs"]:
            pml.set_queue(self.queue)
            pml.htod_field_arrays()
            knl_name = f"order{len(pml.CFS)}_{pml.direction}"
            knl_electric_name = getattr(knl_pml_updates_electric, knl_name)
            knl_magnetic_name = getattr(knl_pml_updates_magnetic, knl_name)

            pml.update_electric_dev = self.elwiseknl(
                self.ctx,
                knl_electric_name["args_opencl"].substitute({"REAL": config.sim_config.dtypes["C_float_or_double"]}),
                knl_electric_name["func"].substitute(subs),
                f"pml_updates_electric_{knl_name}",
                preamble=self.knl_common,
                options=config.sim_config.devices["compiler_opts"],
            )

            pml.update_magnetic_dev = self.elwiseknl(
                self.ctx,
                knl_magnetic_name["args_opencl"].substitute({"REAL": config.sim_config.dtypes["C_float_or_double"]}),
                knl_magnetic_name["func"].substitute(subs),
                f"pml_updates_magnetic_{knl_name}",
                preamble=self.knl_common,
                options=config.sim_config.devices["compiler_opts"],
            )

    def _set_rx_knl(self):
        """Receivers - initialises arrays on compute device, prepares kernel and
        gets kernel function.
        """
        self.rxcoords_dev, self.rxs_dev = htod_rx_arrays(self.grid, self.queue)
        self.store_outputs_dev = self.elwiseknl(
            self.ctx,
            knl_store_outputs.store_outputs["args_opencl"].substitute(
                {"REAL": config.sim_config.dtypes["C_float_or_double"]}
            ),
            knl_store_outputs.store_outputs["func"].substitute({"CUDA_IDX": ""}),
            "store_outputs",
            preamble=self.knl_common,
            options=config.sim_config.devices["compiler_opts"],
        )

    def _set_src_knls(self):
        """Sources - initialises arrays on compute device, prepares kernel and
        gets kernel function.
        """
        if self.grid.hertziandipoles:
            self.srcinfo1_hertzian_dev, self.srcinfo2_hertzian_dev, self.srcwaves_hertzian_dev = htod_src_arrays(
                self.grid.hertziandipoles, self.grid, self.queue
            )
            self.update_hertzian_dipole_dev = self.elwiseknl(
                self.ctx,
                knl_source_updates.update_hertzian_dipole["args_opencl"].substitute(
                    {"REAL": config.sim_config.dtypes["C_float_or_double"]}
                ),
                knl_source_updates.update_hertzian_dipole["func"].substitute(
                    {"CUDA_IDX": "", "REAL": config.sim_config.dtypes["C_float_or_double"]}
                ),
                "update_hertzian_dipole",
                preamble=self.knl_common,
                options=config.sim_config.devices["compiler_opts"],
            )
        if self.grid.magneticdipoles:
            self.srcinfo1_magnetic_dev, self.srcinfo2_magnetic_dev, self.srcwaves_magnetic_dev = htod_src_arrays(
                self.grid.magneticdipoles, self.grid, self.queue
            )
            self.update_magnetic_dipole_dev = self.elwiseknl(
                self.ctx,
                knl_source_updates.update_magnetic_dipole["args_opencl"].substitute(
                    {"REAL": config.sim_config.dtypes["C_float_or_double"]}
                ),
                knl_source_updates.update_magnetic_dipole["func"].substitute(
                    {"CUDA_IDX": "", "REAL": config.sim_config.dtypes["C_float_or_double"]}
                ),
                "update_magnetic_dipole",
                preamble=self.knl_common,
                options=config.sim_config.devices["compiler_opts"],
            )
        if self.grid.voltagesources:
            self.srcinfo1_voltage_dev, self.srcinfo2_voltage_dev, self.srcwaves_voltage_dev = htod_src_arrays(
                self.grid.voltagesources, self.grid, self.queue
            )
            self.update_voltage_source_dev = self.elwiseknl(
                self.ctx,
                knl_source_updates.update_voltage_source["args_opencl"].substitute(
                    {"REAL": config.sim_config.dtypes["C_float_or_double"]}
                ),
                knl_source_updates.update_voltage_source["func"].substitute(
                    {"CUDA_IDX": "", "REAL": config.sim_config.dtypes["C_float_or_double"]}
                ),
                "update_voltage_source",
                preamble=self.knl_common,
                options=config.sim_config.devices["compiler_opts"],
            )

    def _set_snapshot_knl(self):
        """Snapshots - initialises arrays on compute device, prepares kernel and
        gets kernel function.
        """
        (
            self.snapEx_dev,
            self.snapEy_dev,
            self.snapEz_dev,
            self.snapHx_dev,
            self.snapHy_dev,
            self.snapHz_dev,
        ) = htod_snapshot_array(self.grid, self.queue)
        self.store_snapshot_dev = self.elwiseknl(
            self.ctx,
            knl_snapshots.store_snapshot["args_opencl"].substitute(
                {"REAL": config.sim_config.dtypes["C_float_or_double"]}
            ),
            knl_snapshots.store_snapshot["func"].substitute(
                {"CUDA_IDX": "", "NX_SNAPS": Snapshot.nx_max, "NY_SNAPS": Snapshot.ny_max, "NZ_SNAPS": Snapshot.nz_max}
            ),
            "store_snapshot",
            preamble=self.knl_common,
            options=config.sim_config.devices["compiler_opts"],
        )

    def store_outputs(self):
        """Stores field component values for every receiver."""
        if self.grid.rxs:
            self.store_outputs_dev(
                np.int32(len(self.grid.rxs)),
                np.int32(self.grid.iteration),
                self.rxcoords_dev,
                self.rxs_dev,
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                self.grid.Hx_dev,
                self.grid.Hy_dev,
                self.grid.Hz_dev,
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
                    self.grid.Ex_dev,
                    self.grid.Ey_dev,
                    self.grid.Ez_dev,
                    self.grid.Hx_dev,
                    self.grid.Hy_dev,
                    self.grid.Hz_dev,
                    self.snapEx_dev,
                    self.snapEy_dev,
                    self.snapEz_dev,
                    self.snapHx_dev,
                    self.snapHy_dev,
                    self.snapHz_dev,
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
            self.grid.ID_dev,
            self.grid.Hx_dev,
            self.grid.Hy_dev,
            self.grid.Hz_dev,
            self.grid.Ex_dev,
            self.grid.Ey_dev,
            self.grid.Ez_dev,
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
                self.srcinfo1_magnetic_dev,
                self.srcinfo2_magnetic_dev,
                self.srcwaves_magnetic_dev,
                self.grid.ID_dev,
                self.grid.Hx_dev,
                self.grid.Hy_dev,
                self.grid.Hz_dev,
            )

    def update_electric_a(self):
        """Updates electric field components."""
        # All materials are non-dispersive so do standard update.
        if config.get_model_config().materials["maxpoles"] == 0:
            self.update_electric_dev(
                np.int32(self.grid.nx),
                np.int32(self.grid.ny),
                np.int32(self.grid.nz),
                self.grid.ID_dev,
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                self.grid.Hx_dev,
                self.grid.Hy_dev,
                self.grid.Hz_dev,
            )

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            self.dispersive_update_a(
                np.int32(self.grid.nx),
                np.int32(self.grid.ny),
                np.int32(self.grid.nz),
                np.int32(config.get_model_config().materials["maxpoles"]),
                self.grid.ID_dev,
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                self.grid.Hx_dev,
                self.grid.Hy_dev,
                self.grid.Hz_dev,
                self.grid.updatecoeffsdispersive_dev,
                self.grid.Tx_dev,
                self.grid.Ty_dev,
                self.grid.Tz_dev,
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
                self.srcinfo1_voltage_dev,
                self.srcinfo2_voltage_dev,
                self.srcwaves_voltage_dev,
                self.grid.ID_dev,
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
            )

        if self.grid.hertziandipoles:
            self.update_hertzian_dipole_dev(
                np.int32(len(self.grid.hertziandipoles)),
                np.int32(self.grid.iteration),
                config.sim_config.dtypes["float_or_double"](self.grid.dx),
                config.sim_config.dtypes["float_or_double"](self.grid.dy),
                config.sim_config.dtypes["float_or_double"](self.grid.dz),
                self.srcinfo1_hertzian_dev,
                self.srcinfo2_hertzian_dev,
                self.srcwaves_hertzian_dev,
                self.grid.ID_dev,
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
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
                self.grid.ID_dev,
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                self.grid.updatecoeffsdispersive_dev,
                self.grid.Tx_dev,
                self.grid.Ty_dev,
                self.grid.Tz_dev,
            )

    def time_start(self):
        """Starts event timers used to calculate solving time for model."""
        self.event_marker1 = self.cl.enqueue_marker(self.queue)
        self.event_marker1.wait()

    def calculate_memory_used(self, iteration):
        """Calculates memory used on last iteration.

        Args:
            iteration: int for iteration number.

        Returns:
            Memory (RAM) used on compute device.
        """
        # No clear way to determine memory used from PyOpenCL unlike PyCUDA.
        pass

    def calculate_solve_time(self):
        """Calculates solving time for model."""
        event_marker2 = self.cl.enqueue_marker(self.queue)
        event_marker2.wait()
        return (event_marker2.profile.end - self.event_marker1.profile.start) * 1e-9

    def finalise(self):
        """Copies data from compute device back to CPU to save to file(s)."""
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
        pass
