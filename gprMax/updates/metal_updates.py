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

import logging
from importlib import import_module

import numpy as np
from jinja2 import Environment, PackageLoader

from gprMax import config
from gprMax.cuda_opencl import (
    knl_fields_updates,
    knl_snapshots,
    knl_source_updates,
    knl_store_outputs,
)
from gprMax.grid.metal_grid import MetalGrid
from gprMax.receivers import dtoh_rx_array, htod_rx_arrays
from gprMax.snapshots import Snapshot, dtoh_snapshot_array, htod_snapshot_array
from gprMax.sources import htod_src_arrays
from gprMax.updates.updates import Updates
from gprMax.utilities.utilities import round32

logger = logging.getLogger(__name__)

class MetalUpdates:
    """Defines update functions for Apple Metal-based solver."""

    def __init__(self, G):
        """
        Args:
            G: OpenCLGrid class describing a grid in a model.
        """

        self.grid = G

        self.metal = import_module("Metal")
        self.opts = self.metal.MTLCompileOptions.new()

        # Select device and create command queue
        self.dev = config.get_model_config().device["dev"]
        self.cmdqueue = self.dev.newCommandQueue()

        # Set common substitutions for use in kernels
        # Substitutions in function arguments
        self.subs_name_args = {
            "REAL": config.sim_config.dtypes["C_float_or_double"],
            "COMPLEX": config.get_model_config().materials["dispersiveCdtype"],
        }
        # Substitutions in function bodies
        self.subs_func = {
            "REAL": config.sim_config.dtypes["C_float_or_double"],
            "CUDA_IDX": "",
            "NX_FIELDS": self.grid.nx + 1,
            "NY_FIELDS": self.grid.ny + 1,
            "NZ_FIELDS": self.grid.nz + 1,
            "NX_ID": self.grid.ID.shape[1],
            "NY_ID": self.grid.ID.shape[2],
            "NZ_ID": self.grid.ID.shape[3],
        }

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
        # if self.grid.snapshots:
        #     self._set_snapshot_knl()

    def _build_knl(self, knl_func, subs_name_args, subs_func):
        """Builds an Apple Metal kernel from templates: 1) function name and args;
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

        name_plus_args = knl_func["args_metal"].substitute(subs_name_args)
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

        self.knl_common = self.env.get_template("knl_common_metal.tmpl").render(
            REAL=config.sim_config.dtypes["C_float_or_double"],
            N_updatecoeffsE=self.grid.updatecoeffsE.size,
            N_updatecoeffsH=self.grid.updatecoeffsH.size,
            NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1],
            NY_MATDISPCOEFFS=NY_MATDISPCOEFFS,
            updatecoeffsE=self.grid.updatecoeffsE.flatten(),
            updatecoeffsH=self.grid.updatecoeffsH.flatten(),
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

        bld = self._build_knl(knl_fields_updates.update_electric, self.subs_name_args, self.subs_func)
        lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
        self.update_electric_dev = lib.newFunctionWithName_("update_electric")
        self.psoE = self.dev.newComputePipelineStateWithFunction_error_(self.update_electric_dev, None)[0]
        
        # Set thread sizes based on electric (same for magnetic)
        self.grid.set_threads_per_thread_group()
        self.grid.set_thread_group_size(self.psoE)
        
        bld = self._build_knl(knl_fields_updates.update_magnetic, self.subs_name_args, self.subs_func)
        lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
        self.update_magnetic_dev = lib.newFunctionWithName_("update_magnetic")
        self.psoH = self.dev.newComputePipelineStateWithFunction_error_(self.update_magnetic_dev, None)[0]
        

        # If there are any dispersive materials (updates are split into two
        # parts as they require present and updated electric field values).
        if config.get_model_config().materials["maxpoles"] > 0:
            # TODO: Implement Metal compute pipeline for dispersive materials
            # This needs to be implemented when Metal kernels are available
            pass

        if config.get_model_config().materials["maxpoles"] > 0:
            # TODO: Initialize dispersive arrays for Metal
            pass

    def _set_pml_knls(self):
        """PMLS - prepares kernels and gets kernel functions."""
        knl_pml_updates_electric = import_module(
            "gprMax.cuda_opencl.knl_pml_updates_electric_" + self.grid.pmls["formulation"]
        )
        knl_pml_updates_magnetic = import_module(
            "gprMax.cuda_opencl.knl_pml_updates_magnetic_" + self.grid.pmls["formulation"]
        )

        # Set workgroup size, initialise arrays on compute device, and get
        # kernel functions
        for pml in self.grid.pmls["slabs"]:
            pml.set_queue(self.cmdqueue)
            pml.htod_field_arrays(self.dev)
            knl_name = f"order{len(pml.CFS)}_{pml.direction}"
            knl_electric_name = getattr(knl_pml_updates_electric, knl_name)
            knl_magnetic_name = getattr(knl_pml_updates_magnetic, knl_name)

            # Build and compile electric field PML kernel
            func_name = f"pml_updates_electric_{knl_name}"
            subs_name_args_pml = self.subs_name_args.copy()
            subs_name_args_pml["FUNC"] = func_name
            bld = self._build_knl(knl_electric_name, subs_name_args_pml, self.subs_func)
            
            lib, error = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            if lib is None:
                print(f"Electric PML kernel compilation failed: {error}")
                raise RuntimeError(f"Failed to compile electric PML kernel: {error}")
            pml.update_electric_dev = lib.newFunctionWithName_(func_name)
            pml.psoE = self.dev.newComputePipelineStateWithFunction_error_(pml.update_electric_dev, None)[0]

            # Build and compile magnetic field PML kernel
            func_name = f"pml_updates_magnetic_{knl_name}"
            subs_name_args_pml = self.subs_name_args.copy()
            subs_name_args_pml["FUNC"] = func_name
            bld = self._build_knl(knl_magnetic_name, subs_name_args_pml, self.subs_func)
            
            lib, error = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            if lib is None:
                print(f"Magnetic PML kernel compilation failed: {error}")
                raise RuntimeError(f"Failed to compile magnetic PML kernel: {error}")
            pml.update_magnetic_dev = lib.newFunctionWithName_(func_name)
            pml.psoH = self.dev.newComputePipelineStateWithFunction_error_(pml.update_magnetic_dev, None)[0]

    def _set_rx_knl(self):
        """Receivers - initialises arrays on compute device, prepares kernel and
        gets kernel function.
        """
        self.rxcoords_dev, self.rxs_dev = htod_rx_arrays(self.grid, None, self.dev)
        
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
        lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
        self.store_outputs_dev = lib.newFunctionWithName_("store_outputs")
        self.pso_store_outputs = self.dev.newComputePipelineStateWithFunction_error_(self.store_outputs_dev, None)[0]
        
        # Set thread sizes
        self.grid.set_threads_per_thread_group()
        self.grid.set_thread_group_size(self.pso_store_outputs)

    def _set_src_knls(self):
        """Sources - initialises arrays on compute device, prepares kernel and
        gets kernel function.
        """
        if self.grid.hertziandipoles:
            self.srcinfo1_hertzian_dev, self.srcinfo2_hertzian_dev, self.srcwaves_hertzian_dev = htod_src_arrays(
                self.grid.hertziandipoles, self.grid, self.dev
            )
            
            bld = self._build_knl(knl_source_updates.update_hertzian_dipole, self.subs_name_args, self.subs_func)
            lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            self.update_hertzian_dipole_dev = lib.newFunctionWithName_("update_hertzian_dipole")
            self.pso_hertzian_dipole = self.dev.newComputePipelineStateWithFunction_error_(self.update_hertzian_dipole_dev, None)[0]
            
        if self.grid.magneticdipoles:
            self.srcinfo1_magnetic_dev, self.srcinfo2_magnetic_dev, self.srcwaves_magnetic_dev = htod_src_arrays(
                self.grid.magneticdipoles, self.grid, self.dev
            )
            
            bld = self._build_knl(knl_source_updates.update_magnetic_dipole, self.subs_name_args, self.subs_func)
            lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            self.update_magnetic_dipole_dev = lib.newFunctionWithName_("update_magnetic_dipole")
            self.pso_magnetic_dipole = self.dev.newComputePipelineStateWithFunction_error_(self.update_magnetic_dipole_dev, None)[0]
            
        if self.grid.voltagesources:
            self.srcinfo1_voltage_dev, self.srcinfo2_voltage_dev, self.srcwaves_voltage_dev = htod_src_arrays(
                self.grid.voltagesources, self.grid, self.dev
            )
            
            bld = self._build_knl(knl_source_updates.update_voltage_source, self.subs_name_args, self.subs_func)
            lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            self.update_voltage_source_dev = lib.newFunctionWithName_("update_voltage_source")
            self.pso_voltage_source = self.dev.newComputePipelineStateWithFunction_error_(self.update_voltage_source_dev, None)[0]

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

    def store_outputs(self, iteration):
        """Stores field component values for every receiver.
        
        Args:
            iteration: int for iteration number.
        """
        if self.grid.rxs:
            # Check if field device arrays exist
            field_attrs = ['Ex_dev', 'Ey_dev', 'Ez_dev', 'Hx_dev', 'Hy_dev', 'Hz_dev']
            missing_attrs = [attr for attr in field_attrs if not hasattr(self.grid, attr)]
            if missing_attrs:
                # Try to initialize field arrays if they don't exist
                if not hasattr(self.grid, 'Ex_dev'):
                    self.grid.htod_field_arrays(self.dev)
            
            self.cmdbuffer_store_outputs = self.cmdqueue.commandBuffer()
            self.cmpencoder_store_outputs = self.cmdbuffer_store_outputs.computeCommandEncoder()
            self.cmpencoder_store_outputs.setComputePipelineState_(self.pso_store_outputs)
            
            # Set buffer arguments for the kernel
            # NRX (number of receivers)
            nrx_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(len(self.grid.rxs)).tobytes(), 4, 0)
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(nrx_buffer, 0, 0)
            
            # iteration
            iteration_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(iteration).tobytes(), 4, 0)
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(iteration_buffer, 0, 1)
            
            # rxcoords - receiver coordinates
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(self.rxcoords_dev, 0, 2)
            
            # rxs - receiver data storage array
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(self.rxs_dev, 0, 3)
            
            # Field component buffers (Ex, Ey, Ez, Hx, Hy, Hz)
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(self.grid.Ex_dev, 0, 4)
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(self.grid.Ey_dev, 0, 5)
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(self.grid.Ez_dev, 0, 6)
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(self.grid.Hx_dev, 0, 7)
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(self.grid.Hy_dev, 0, 8)
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(self.grid.Hz_dev, 0, 9)
            
            self.cmpencoder_store_outputs.dispatchThreads_threadsPerThreadgroup_(
                self.metal.MTLSizeMake(round32(len(self.grid.rxs)), 1, 1), 
                self.metal.MTLSizeMake(self.pso_store_outputs.maxTotalThreadsPerThreadgroup(), 1, 1)
                )
            self.cmpencoder_store_outputs.endEncoding()
            self.cmdbuffer_store_outputs.commit()
            self.cmdbuffer_store_outputs.waitUntilCompleted()

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
        # Initialize buffers if not already done (since magnetic update is called before electric update)
        if not hasattr(self.grid, 'ID_dev') or not hasattr(self.grid, 'Ex_dev'):
            self.grid.htod_geometry_arrays(self.dev)
            self.grid.htod_field_arrays(self.dev)
            self.grid.htod_material_arrays(self.dev)
        
        self.cmdbufferH = self.cmdqueue.commandBuffer()
        self.cmpencoderH = self.cmdbufferH.computeCommandEncoder()
        self.cmpencoderH.setComputePipelineState_(self.psoH)
        
        # Set scalar values for H update kernel (similar to E update)
        nx_value = np.int32(self.grid.nx + 1)
        ny_value = np.int32(self.grid.ny + 1) 
        nz_value = np.int32(self.grid.nz + 1)
        
        self.cmpencoderH.setBytes_length_atIndex_(nx_value.tobytes(), 4, 0)
        self.cmpencoderH.setBytes_length_atIndex_(ny_value.tobytes(), 4, 1)
        self.cmpencoderH.setBytes_length_atIndex_(nz_value.tobytes(), 4, 2)
        
        # Set buffer arguments for magnetic field update kernel
        self.cmpencoderH.setBuffer_offset_atIndex_(self.grid.ID_dev, 0, 3)
        self.cmpencoderH.setBuffer_offset_atIndex_(self.grid.Hx_dev, 0, 4)
        self.cmpencoderH.setBuffer_offset_atIndex_(self.grid.Hy_dev, 0, 5)
        self.cmpencoderH.setBuffer_offset_atIndex_(self.grid.Hz_dev, 0, 6)
        self.cmpencoderH.setBuffer_offset_atIndex_(self.grid.Ex_dev, 0, 7)
        self.cmpencoderH.setBuffer_offset_atIndex_(self.grid.Ey_dev, 0, 8)
        self.cmpencoderH.setBuffer_offset_atIndex_(self.grid.Ez_dev, 0, 9)
        
        self.cmpencoderH.dispatchThreads_threadsPerThreadgroup_(self.grid.tptg, self.grid.tgs)
        self.cmpencoderH.endEncoding()
        self.cmdbufferH.commit()
        self.cmdbufferH.waitUntilCompleted()

    def update_magnetic_pml(self):
        """Updates magnetic field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_magnetic()

    def update_magnetic_sources(self, iteration):
        """Updates magnetic field components from sources."""
        if self.grid.magneticdipoles:
            # TODO: Implement Metal compute pipeline execution for magnetic dipole sources
            # This needs to be implemented when Metal kernels are available
            pass

    def update_electric_a(self):
        """Updates electric field components."""
        
        # All materials are non-dispersive so do standard update.
        if config.get_model_config().materials["maxpoles"] == 0:
            # Only initialize buffers once (on first iteration) - don't re-initialize as it resets field values!
            if not hasattr(self.grid, 'ID_dev') or not hasattr(self.grid, 'Ex_dev'):
                self.grid.htod_geometry_arrays(self.dev)
                self.grid.htod_field_arrays(self.dev)
                self.grid.htod_material_arrays(self.dev)
            # Skip re-initialization to preserve field data on GPU
            
            self.cmdbufferE = self.cmdqueue.commandBuffer()
            self.cmpencoderE = self.cmdbufferE.computeCommandEncoder()
            self.cmpencoderE.setComputePipelineState_(self.psoE)
            
            # For Metal, we need to set the scalar values using setBytes, not buffers
            # Set NX, NY, NZ as scalar values (Metal expects device const int&)
            nx_value = np.int32(self.grid.nx + 1)
            ny_value = np.int32(self.grid.ny + 1) 
            nz_value = np.int32(self.grid.nz + 1)
            
            self.cmpencoderE.setBytes_length_atIndex_(nx_value.tobytes(), 4, 0)
            self.cmpencoderE.setBytes_length_atIndex_(ny_value.tobytes(), 4, 1)
            self.cmpencoderE.setBytes_length_atIndex_(nz_value.tobytes(), 4, 2)
            
            # Set buffer arguments for electric field update kernel
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.ID_dev, 0, 3)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Ex_dev, 0, 4)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Ey_dev, 0, 5)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Ez_dev, 0, 6)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Hx_dev, 0, 7)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Hy_dev, 0, 8)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Hz_dev, 0, 9)
            
            self.cmpencoderE.dispatchThreads_threadsPerThreadgroup_(self.grid.tptg, self.grid.tgs)

            self.cmpencoderE.endEncoding()
            self.cmdbufferE.commit()
            self.cmdbufferE.waitUntilCompleted()

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            # TODO: Implement dispersive material updates for Metal
            # For now, just do the standard update
            
            # Only initialize buffers once (on first iteration) - don't re-initialize as it resets field values!
            if not hasattr(self.grid, 'ID_dev') or not hasattr(self.grid, 'Ex_dev'):
                self.grid.htod_geometry_arrays(self.dev)
                self.grid.htod_field_arrays(self.dev)
                self.grid.htod_material_arrays(self.dev)
            # Skip re-initialization to preserve field data on GPU
            
            self.cmdbufferE = self.cmdqueue.commandBuffer()
            self.cmpencoderE = self.cmdbufferE.computeCommandEncoder()
            self.cmpencoderE.setComputePipelineState_(self.psoE)
            
            # For Metal, we need to set the scalar values using setBytes, not buffers
            # Set NX, NY, NZ as scalar values (Metal expects device const int&)
            nx_value = np.int32(self.grid.nx + 1)
            ny_value = np.int32(self.grid.ny + 1) 
            nz_value = np.int32(self.grid.nz + 1)
            
            self.cmpencoderE.setBytes_length_atIndex_(nx_value.tobytes(), 4, 0)
            self.cmpencoderE.setBytes_length_atIndex_(ny_value.tobytes(), 4, 1)
            self.cmpencoderE.setBytes_length_atIndex_(nz_value.tobytes(), 4, 2)
            
            # Set buffer arguments for electric field update kernel
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.ID_dev, 0, 3)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Ex_dev, 0, 4)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Ey_dev, 0, 5)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Ez_dev, 0, 6)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Hx_dev, 0, 7)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Hy_dev, 0, 8)
            self.cmpencoderE.setBuffer_offset_atIndex_(self.grid.Hz_dev, 0, 9)
            
            self.cmpencoderE.dispatchThreads_threadsPerThreadgroup_(self.grid.tptg, self.grid.tgs)

            self.cmpencoderE.endEncoding()
            self.cmdbufferE.commit()
            self.cmdbufferE.waitUntilCompleted()
        
    def update_electric_pml(self):
        """Updates electric field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_electric()

    def update_electric_sources(self, iteration):
        """Updates electric field components from sources -
        update any Hertzian dipole sources last.
        """
        if self.grid.voltagesources:
            # Create command buffer for voltage sources
            cmdbuffer_voltage = self.cmdqueue.commandBuffer()
            cmpencoder_voltage = cmdbuffer_voltage.computeCommandEncoder()
            cmpencoder_voltage.setComputePipelineState_(self.pso_voltage_source)
            
            # Set buffer arguments for voltage source kernel
            n_voltage_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(len(self.grid.voltagesources)).tobytes(), 4, 0)
            cmpencoder_voltage.setBuffer_offset_atIndex_(n_voltage_buffer, 0, 0)
            
            iteration_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(iteration).tobytes(), 4, 0)
            cmpencoder_voltage.setBuffer_offset_atIndex_(iteration_buffer, 0, 1)
            
            # Set other required buffers
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.srcinfo1_voltage_dev, 0, 2)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.srcinfo2_voltage_dev, 0, 3)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.srcwaves_voltage_dev, 0, 4)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.grid.Ex_dev, 0, 5)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.grid.Ey_dev, 0, 6)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.grid.Ez_dev, 0, 7)
            
            # Dispatch the kernel
            cmpencoder_voltage.dispatchThreads_threadsPerThreadgroup_(
                self.metal.MTLSizeMake(round32(len(self.grid.voltagesources)), 1, 1),
                self.metal.MTLSizeMake(self.pso_voltage_source.maxTotalThreadsPerThreadgroup(), 1, 1)
            )
            cmpencoder_voltage.endEncoding()
            cmdbuffer_voltage.commit()
            cmdbuffer_voltage.waitUntilCompleted()

        if self.grid.hertziandipoles:
            # Optional debug logging for first iteration only
            if iteration == 1:
                print(f"Metal backend: {len(self.grid.hertziandipoles)} Hertzian dipole(s) at iteration {iteration}")
                for i, src in enumerate(self.grid.hertziandipoles):
                    print(f"  Source {i}: position=({src.xcoord},{src.ycoord},{src.zcoord}), polarisation={src.polarisation}")
            
            # Create command buffer for Hertzian dipoles
            cmdbuffer_hertzian = self.cmdqueue.commandBuffer()
            cmpencoder_hertzian = cmdbuffer_hertzian.computeCommandEncoder()
            cmpencoder_hertzian.setComputePipelineState_(self.pso_hertzian_dipole)
            
            # Set buffer arguments for Hertzian dipole kernel
            n_hertzian_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(len(self.grid.hertziandipoles)).tobytes(), 4, 0)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(n_hertzian_buffer, 0, 0)
            
            iteration_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(iteration).tobytes(), 4, 0)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(iteration_buffer, 0, 1)
            
            # Set spatial discretization buffers
            dx_buffer = self.dev.newBufferWithBytes_length_options_(
                np.float32(self.grid.dx).tobytes(), 4, 0)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(dx_buffer, 0, 2)
            
            dy_buffer = self.dev.newBufferWithBytes_length_options_(
                np.float32(self.grid.dy).tobytes(), 4, 0)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(dy_buffer, 0, 3)
            
            dz_buffer = self.dev.newBufferWithBytes_length_options_(
                np.float32(self.grid.dz).tobytes(), 4, 0)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(dz_buffer, 0, 4)
            
            # Set source info and waveform buffers
            cmpencoder_hertzian.setBuffer_offset_atIndex_(self.srcinfo1_hertzian_dev, 0, 5)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(self.srcinfo2_hertzian_dev, 0, 6)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(self.srcwaves_hertzian_dev, 0, 7)
            
            # Set ID and field buffers
            cmpencoder_hertzian.setBuffer_offset_atIndex_(self.grid.ID_dev, 0, 8)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(self.grid.Ex_dev, 0, 9)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(self.grid.Ey_dev, 0, 10)
            cmpencoder_hertzian.setBuffer_offset_atIndex_(self.grid.Ez_dev, 0, 11)
            
            # Dispatch the kernel
            cmpencoder_hertzian.dispatchThreads_threadsPerThreadgroup_(
                self.metal.MTLSizeMake(round32(len(self.grid.hertziandipoles)), 1, 1),
                self.metal.MTLSizeMake(self.pso_hertzian_dipole.maxTotalThreadsPerThreadgroup(), 1, 1)
            )
            cmpencoder_hertzian.endEncoding()
            cmdbuffer_hertzian.commit()
            cmdbuffer_hertzian.waitUntilCompleted()
            
            # Check Ex field before and after kernel execution
            # Optional debug: Check source fields briefly for first iteration
            if iteration == 1:
                try:
                    total_elements = (self.grid.nx + 1) * (self.grid.ny + 1) * (self.grid.nz + 1)
                    buffer_size = total_elements * 4
                    ex_buffer = self.grid.Ex_dev.contents().as_buffer(buffer_size)
                    ex_array = np.frombuffer(ex_buffer, dtype=np.float32)
                    
                    max_abs_ex = np.max(np.abs(ex_array))
                    nonzero_count = np.count_nonzero(ex_array)
                    print(f"Metal backend after source kernel (iteration {iteration}): Ex max_abs={max_abs_ex:.2e}, nonzero={nonzero_count}")
                        
                except Exception as e:
                    print(f"Error checking fields after source kernel: {e}")

        self.grid.iteration += 1

    def update_electric_b(self):
        """If there are any dispersive materials do 2nd part of dispersive
        update - it is split into two parts as it requires present and
        updated electric field values. Therefore it can only be completely
        updated after the electric field has been updated by the PML and
        source updates.
        """
        if config.get_model_config().materials["maxpoles"] > 0:
            # TODO: Implement 2nd part of dispersive update for Metal
            pass

    def time_start(self):
        """Starts event timers used to calculate solving time for model."""
        pass

    def calculate_memory_used(self, iteration):
        """Calculates memory used on last iteration. 

        Args:
            iteration: int for iteration number.

        Returns:
            Memory (RAM) used on compute device.
        """
        return 0

    def calculate_solve_time(self):
        """Calculates solving time for model."""
        return 0

    def finalise(self):
        """Copies data from compute device back to CPU to save to file(s)."""
        # Copy output from receivers array back to correct receiver objects
        if self.grid.rxs:
            dtoh_rx_array(self.rxs_dev, self.rxcoords_dev, self.grid, self.dev)

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

