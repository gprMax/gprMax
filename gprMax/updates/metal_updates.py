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

import logging
from importlib import import_module

import numpy as np
from jinja2 import Environment, PackageLoader

from gprMax import config
from gprMax.cuda_opencl import (
    knl_eigenmode,
    knl_fields_updates,
    knl_magnetic_frill_source,
    knl_rational_network,
    knl_snapshots,
    knl_source_updates,
    knl_store_outputs,
    knl_symmetry_boundaries,
    knl_transmission_line,
    knl_virtual_waveguide,
)
from gprMax.eigenmode_device import (
    eigenmode_source_envelopes,
    finalise_device_eigenmode_monitors,
    prepare_device_eigenmode_monitor,
    prepare_device_eigenmode_source,
    reanchor_device_eigenmode_monitor,
)
from gprMax.grid.metal_grid import MetalGrid
from gprMax.network_ports import dtoh_rational_network_outputs, htod_rational_network_arrays
from gprMax.ntff.device import MetalCombinedKSIRCollector
from gprMax.receivers import dtoh_rx_array, htod_rx_arrays, requested_current_outputs
from gprMax.sar_device import MetalSARCollector
from gprMax.snapshots import (
    Snapshot,
    _snapshot_axis_strides,
    dtoh_snapshot_array,
    htod_snapshot_array,
    update_snapshot_max_dims,
)
from gprMax.sources import (
    MAGNETIC_FRILL_MAX_TERMS,
    dtoh_magnetic_frill_source_outputs,
    dtoh_transmission_line_outputs,
    htod_magnetic_frill_source_arrays,
    htod_src_arrays,
    htod_transmission_line_arrays,
)
from gprMax.updates.metal_plane_waves import MetalPlaneWaveController
from gprMax.updates.updates import Updates
from gprMax.utilities.utilities import round32, timer

logger = logging.getLogger(__name__)


class MetalUpdates(Updates[MetalGrid]):
    """Defines update functions for Apple Metal-based solver."""

    def __init__(self, G: MetalGrid, shared=None):
        """
        Args:
            G: MetalGrid class describing a grid in a model.
        """

        super().__init__(G)

        self.metal = import_module("Metal")
        self.opts = self.metal.MTLCompileOptions.new()

        # Auxiliary virtual guides share the parent's device and command
        # queue so their fields can be coupled without host transfers.
        if shared is None:
            self.dev = config.get_model_config().device["dev"]
            self.cmdqueue = self.dev.newCommandQueue()
        else:
            self.dev = shared.dev
            self.cmdqueue = shared.cmdqueue

        # Set common substitutions for use in kernels
        # Substitutions in function arguments
        self.subs_name_args = {
            "REAL": config.sim_config.dtypes["C_float_or_double"],
            "COMPLEX": self.grid.dispersiveCdtype,
        }
        # Substitutions in function bodies
        self.subs_func = {
            "REAL": config.sim_config.dtypes["C_float_or_double"],
            "CUDA_IDX": "",
            "METAL_DFT_PARAMETERS": """
                int NF = parameters.NF;
                int NM = parameters.NM;
                int normal_axis = parameters.normal_axis;
                int direction_sign = parameters.direction_sign;
                int magnetic_side = parameters.magnetic_side;
                int u0 = parameters.u0;
                int v0 = parameters.v0;
                int u1 = parameters.u1;
                int v1 = parameters.v1;
                int plane_index = parameters.plane_index;
                float dt = parameters.dt;
                float measure = parameters.measure;
                int handedness = parameters.handedness;
            """,
            "NX_FIELDS": self.grid.nx + 1,
            "NY_FIELDS": self.grid.ny + 1,
            "NZ_FIELDS": self.grid.nz + 1,
            "NX_ID": self.grid.ID.shape[1],
            "NY_ID": self.grid.ID.shape[2],
            "NZ_ID": self.grid.ID.shape[3],
        }

        # Enviroment for templating kernels
        self.env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))

        # Must happen before _set_macros(), which bakes NX_SNAPS/NY_SNAPS/
        # NZ_SNAPS into the shared kernel preamble - see
        # update_snapshot_max_dims()'s docstring.
        if self.grid.snapshots:
            update_snapshot_max_dims(self.grid.snapshots)

        # Initialise arrays on device, prepare kernels, and get kernel functions
        self._set_macros()
        self._set_field_knls()
        if "pmc" in self.grid.symmetry_boundaries.values():
            self._set_symmetry_boundary_knl()
        if self.grid.pmls["slabs"]:
            self._set_pml_knls()
        if self.grid.rxs:
            self._set_rx_knl()
        if self.grid.voltagesources + self.grid.hertziandipoles + self.grid.magneticdipoles:
            self._set_src_knls()
        if getattr(self.grid, "transmissionlines", ()):
            self._set_transmission_line_knls()
        if self.grid.magneticfrillsources:
            self._set_magnetic_frill_knl()
        if self.grid.networkterminals:
            self._set_rational_network_knl()
        if self.grid.eigenmodesources:
            self._set_eigenmode_source_knls()
        if self.grid.eigenmodeports:
            self._set_eigenmode_monitor_knl()
        if self.grid.snapshots:
            self._set_snapshot_knl()
        self.planewave_controller = None
        if self.grid.discreteplanewaves:
            self.planewave_controller = MetalPlaneWaveController(self)
        self.ntff_collector = None
        if self.grid.ntff_monitors:
            self.ntff_c_real = config.sim_config.dtypes["C_float_or_double"]
            self.ntff_collector = MetalCombinedKSIRCollector(self)
        self.sar_collector = MetalSARCollector(self) if self.grid.sar_monitors else None
        if self.grid.virtual_waveguides:
            self._set_virtual_waveguide_knls()
        for guide in self.grid.virtual_waveguides:
            guide.initialise_device(self)

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

    def _compile_pipeline(self, specification, function_name, substitutions=None):
        """Compile one templated Metal kernel and return its pipeline state."""

        source = self._build_knl(
            specification,
            self.subs_name_args,
            substitutions or self.subs_func,
        )
        library, error = self.dev.newLibraryWithSource_options_error_(source, self.opts, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal {function_name} kernel: {error}")
        function = library.newFunctionWithName_(function_name)
        return self.dev.newComputePipelineStateWithFunction_error_(function, None)[0]

    def _set_macros(self):
        """Common macros to be used in kernels."""

        # Set specific values for any dispersive materials
        if self.grid.maxpoles > 0:
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
            DRUDELORENTZ=self.grid.drudelorentz,
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
            # Must match htod_src_arrays()'s actual row stride (sources.py:
            # (len(sources), G.iterations + 1)), not G.iterations - see
            # cuda_updates.py's equivalent comment for the full mechanism.
            NY_SRCWAVES=self.grid.iterations + 1,
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
        lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
        self.update_electric_dev = lib.newFunctionWithName_("update_electric")
        self.psoE = self.dev.newComputePipelineStateWithFunction_error_(
            self.update_electric_dev, None
        )[0]

        # Set thread sizes based on electric (same for magnetic)
        self.grid.set_threads_per_thread_group()
        self.grid.set_thread_group_size(self.psoE)

        bld = self._build_knl(
            knl_fields_updates.update_magnetic, self.subs_name_args, self.subs_func
        )
        lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
        self.update_magnetic_dev = lib.newFunctionWithName_("update_magnetic")
        self.psoH = self.dev.newComputePipelineStateWithFunction_error_(
            self.update_magnetic_dev, None
        )[0]

        # If there are any dispersive materials (updates are split into two
        # parts as they require present and updated electric field values).
        # Mirrors CUDAUpdates/OpenCLUpdates._set_field_knls()'s equivalent
        # block exactly - same subs_func keys, same kernel-building pattern.
        if self.grid.maxpoles > 0:
            self.subs_func.update(
                {
                    "REAL": config.sim_config.dtypes["C_float_or_double"],
                    "REALFUNC": self.grid.crealfunc,
                    "NX_T": self.grid.Tx.shape[1],
                    "NY_T": self.grid.Tx.shape[2],
                    "NZ_T": self.grid.Tx.shape[3],
                }
            )

            bld = self._build_knl(
                knl_fields_updates.update_electric_dispersive_A,
                self.subs_name_args,
                self.subs_func,
            )
            lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            self.dispersive_update_a_dev = lib.newFunctionWithName_("update_electric_dispersive_A")
            self.pso_dispersive_a = self.dev.newComputePipelineStateWithFunction_error_(
                self.dispersive_update_a_dev, None
            )[0]

            bld = self._build_knl(
                knl_fields_updates.update_electric_dispersive_B,
                self.subs_name_args,
                self.subs_func,
            )
            lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            self.dispersive_update_b_dev = lib.newFunctionWithName_("update_electric_dispersive_B")
            self.pso_dispersive_b = self.dev.newComputePipelineStateWithFunction_error_(
                self.dispersive_update_b_dev, None
            )[0]

            # Tx/Ty/Tz + updatecoeffsdispersive host arrays already exist by
            # this point (allocated during grid.build(), same as for every
            # other solver) - upload them once, eagerly, matching CUDA's own
            # placement/timing exactly.
            self.grid.htod_dispersive_arrays(self.dev)

        # Initialise geometry/field/material arrays on device unconditionally,
        # matching CUDAUpdates/OpenCLUpdates (which call the CUDA/OpenCL
        # equivalents here with no guard). A fresh MetalUpdates is constructed
        # every model run (see solvers.py), so this always runs once per run -
        # including with geometry_fixed=True, where self.grid (and any device
        # buffers previously attached to it) survives across runs while the
        # host field arrays are freshly zeroed by reuse_geometry(). The
        # previous per-iteration `hasattr(self.grid, "Ex_dev")` guards skipped
        # this upload whenever the grid already carried buffers from a prior
        # run, silently resuming from that run's final GPU field values
        # instead of the freshly-reset host arrays.
        self.grid.htod_geometry_arrays(self.dev)
        self.grid.htod_field_arrays(self.dev)
        self.grid.htod_material_arrays(self.dev)

    def _set_symmetry_boundary_knl(self):
        """Build normal or dispersive PMC ghost-image boundary kernels."""
        substitutions = dict(self.subs_func)
        substitutions.update(knl_symmetry_boundaries.nondispersive_substitutions())
        source = self._build_knl(
            knl_symmetry_boundaries.update_electric_pmc,
            self.subs_name_args,
            substitutions,
        )
        library, error = self.dev.newLibraryWithSource_options_error_(source, self.opts, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal PMC kernel: {error}")
        function = library.newFunctionWithName_("update_electric_pmc")
        self.pso_electric_pmc = self.dev.newComputePipelineStateWithFunction_error_(function, None)[
            0
        ]
        if self.grid.maxpoles > 0:
            substitutions = dict(self.subs_func)
            substitutions.update(
                knl_symmetry_boundaries.dispersive_substitutions(
                    config.sim_config.dtypes["C_float_or_double"]
                )
            )
            self.pso_electric_pmc_dispersive = self._compile_pipeline(
                knl_symmetry_boundaries.update_electric_pmc_dispersive,
                "update_electric_pmc_dispersive",
                substitutions,
            )
            self.pso_electric_pmc_dispersive_b = self._compile_pipeline(
                knl_symmetry_boundaries.update_electric_pmc_dispersive_b,
                "update_electric_pmc_dispersive_b",
                self.subs_func,
            )

    def _set_transmission_line_knls(self):
        """Initialise device-resident transmission lines on Metal."""

        arrays = htod_transmission_line_arrays(self.grid.transmissionlines, self.grid)
        for name, array in arrays.items():
            setattr(self, f"tl_{name}_dev", array)
        real = config.sim_config.dtypes["float_or_double"]
        line = self.grid.transmissionlines[0]
        self.tl_line_coefficient = real(config.c * self.grid.dt / line.dl)
        self.tl_abc_coefficient = real(
            (config.c * self.grid.dt - line.dl) / (config.c * self.grid.dt + line.dl)
        )
        substitutions = dict(self.subs_func)
        substitutions.update(
            {
                "NY_TLINFO": 10,
                "NY_TLWAVES": self.grid.iterations + 1,
                "NY_TLOUTPUTS": self.grid.iterations + 1,
            }
        )
        self.pso_transmission_line_magnetic = self._compile_pipeline(
            knl_transmission_line.update_transmission_line_magnetic,
            "update_transmission_line_magnetic",
            substitutions,
        )
        self.pso_transmission_line_electric = self._compile_pipeline(
            knl_transmission_line.update_transmission_line_electric,
            "update_transmission_line_electric",
            substitutions,
        )

    def _pmc_flags(self):
        boundaries = self.grid.symmetry_boundaries
        return tuple(
            np.int32(boundaries.get(face) == "pmc")
            for face in ("x0", "xmax", "y0", "ymax", "z0", "zmax")
        )

    def _set_pml_knls(self):
        """PMLS - prepares kernels and gets kernel functions."""
        # Set workgroup size, initialise arrays on compute device, and get
        # kernel functions
        for pml in self.grid.pmls["slabs"]:
            knl_pml_updates_electric = import_module(
                "gprMax.cuda_opencl.knl_pml_updates_electric_" + pml.formulation
            )
            knl_pml_updates_magnetic = import_module(
                "gprMax.cuda_opencl.knl_pml_updates_magnetic_" + pml.formulation
            )
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
                logger.debug(f"Electric PML kernel compilation failed: {error}")
                raise RuntimeError(f"Failed to compile electric PML kernel: {error}")
            pml.update_electric_dev = lib.newFunctionWithName_(func_name)
            pml.psoE = self.dev.newComputePipelineStateWithFunction_error_(
                pml.update_electric_dev, None
            )[0]

            # Build and compile magnetic field PML kernel
            func_name = f"pml_updates_magnetic_{knl_name}"
            subs_name_args_pml = self.subs_name_args.copy()
            subs_name_args_pml["FUNC"] = func_name
            bld = self._build_knl(knl_magnetic_name, subs_name_args_pml, self.subs_func)

            lib, error = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            if lib is None:
                logger.debug(f"Magnetic PML kernel compilation failed: {error}")
                raise RuntimeError(f"Failed to compile magnetic PML kernel: {error}")
            pml.update_magnetic_dev = lib.newFunctionWithName_(func_name)
            pml.psoH = self.dev.newComputePipelineStateWithFunction_error_(
                pml.update_magnetic_dev, None
            )[0]

    def _set_rx_knl(self):
        """Receivers - initialises arrays on compute device, prepares kernel and
        gets kernel function.
        """
        (
            self.rxcoords_dev,
            self.rxs_dev,
            self.rxcurrentinfo_dev,
            self.rxcurrents_dev,
        ) = htod_rx_arrays(self.grid, None, self.dev)
        self.nrxcurrent = len(requested_current_outputs(self.grid))

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
        self.pso_store_outputs = self.dev.newComputePipelineStateWithFunction_error_(
            self.store_outputs_dev, None
        )[0]
        if self.nrxcurrent:
            bld = self._build_knl(
                knl_store_outputs.store_current_outputs,
                self.subs_name_args,
                self.subs_func,
            )
            lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            self.store_current_outputs_dev = lib.newFunctionWithName_("store_current_outputs")
            self.pso_store_current_outputs = self.dev.newComputePipelineStateWithFunction_error_(
                self.store_current_outputs_dev, None
            )[0]

        # No self.grid.set_thread_group_size() call here - store_outputs()'s
        # own dispatch always computes its thread-group size directly from
        # self.pso_store_outputs.maxTotalThreadsPerThreadgroup(), never from
        # self.grid.tgs. Calling it here would only clobber the field-sized
        # self.grid.tgs that _set_field_knls() already set (used by the bulk
        # electric/magnetic/PML dispatches) with this unrelated, differently
        # sized pipeline's own limit - a real, previously-found bug.

    def _set_src_knls(self):
        """Sources - initialises arrays on compute device, prepares kernel and
        gets kernel function.
        """
        if self.grid.hertziandipoles:
            (
                self.srcinfo1_hertzian_dev,
                self.srcinfo2_hertzian_dev,
                self.srcwaves_hertzian_dev,
            ) = htod_src_arrays(self.grid.hertziandipoles, self.grid, self.dev)

            bld = self._build_knl(
                knl_source_updates.update_hertzian_dipole,
                self.subs_name_args,
                self.subs_func,
            )
            lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            self.update_hertzian_dipole_dev = lib.newFunctionWithName_("update_hertzian_dipole")
            self.pso_hertzian_dipole = self.dev.newComputePipelineStateWithFunction_error_(
                self.update_hertzian_dipole_dev, None
            )[0]

        if self.grid.magneticdipoles:
            (
                self.srcinfo1_magnetic_dev,
                self.srcinfo2_magnetic_dev,
                self.srcwaves_magnetic_dev,
            ) = htod_src_arrays(self.grid.magneticdipoles, self.grid, self.dev)

            bld = self._build_knl(
                knl_source_updates.update_magnetic_dipole,
                self.subs_name_args,
                self.subs_func,
            )
            lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            self.update_magnetic_dipole_dev = lib.newFunctionWithName_("update_magnetic_dipole")
            self.pso_magnetic_dipole = self.dev.newComputePipelineStateWithFunction_error_(
                self.update_magnetic_dipole_dev, None
            )[0]

        if self.grid.voltagesources:
            (
                self.srcinfo1_voltage_dev,
                self.srcinfo2_voltage_dev,
                self.srcwaves_voltage_dev,
            ) = htod_src_arrays(self.grid.voltagesources, self.grid, self.dev)

            bld = self._build_knl(
                knl_source_updates.update_voltage_source,
                self.subs_name_args,
                self.subs_func,
            )
            lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
            self.update_voltage_source_dev = lib.newFunctionWithName_("update_voltage_source")
            self.pso_voltage_source = self.dev.newComputePipelineStateWithFunction_error_(
                self.update_voltage_source_dev, None
            )[0]

    def _set_magnetic_frill_knl(self):
        """Initialise corrected device-resident magnetic-frill sources."""

        arrays = htod_magnetic_frill_source_arrays(self.grid.magneticfrillsources, self.grid)
        for name, array in arrays.items():
            setattr(self, f"frill_{name}_dev", array)

        substitutions = dict(self.subs_func)
        substitutions.update(
            {
                "MAX_FRILLTERMS": MAGNETIC_FRILL_MAX_TERMS,
                "NY_FRILLTERMINFO": 4,
                "NY_FRILLTERMPARAMS": 2,
                "NY_FRILLPARAMS": 3,
                "NY_FRILLWAVES": self.grid.iterations + 1,
                "NY_FRILLOUT": self.grid.iterations + 1,
            }
        )
        source = self._build_knl(
            knl_magnetic_frill_source.update_magnetic_frill_source,
            self.subs_name_args,
            substitutions,
        )
        library, error = self.dev.newLibraryWithSource_options_error_(source, self.opts, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal magnetic-frill kernel: {error}")
        function = library.newFunctionWithName_("update_magnetic_frill_source")
        self.pso_magnetic_frill = self.dev.newComputePipelineStateWithFunction_error_(
            function, None
        )[0]

    def _set_rational_network_knl(self):
        """Initialise sparse device-resident rational-network terminals."""

        arrays = htod_rational_network_arrays(self.grid.networkterminals, self.grid)
        for name, array in arrays.items():
            setattr(self, f"rn_{name}_dev", array)
        substitutions = dict(self.subs_func)
        substitutions.update(
            {
                "NY_RNINFO": 6,
                "NY_RNPARAMS": 7,
                "NY_RNWAVEWHOLE": self.grid.iterations + 1,
                "NY_RNWAVEHALF": self.grid.iterations,
                "NY_RNVOLTAGE": self.grid.iterations + 1,
                "NY_RNCURRENT": self.grid.iterations,
            }
        )
        source = self._build_knl(
            knl_rational_network.update_rational_network,
            self.subs_name_args,
            substitutions,
        )
        library, error = self.dev.newLibraryWithSource_options_error_(source, self.opts, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal rational-network kernel: {error}")
        function = library.newFunctionWithName_("update_rational_network")
        self.pso_rational_network = self.dev.newComputePipelineStateWithFunction_error_(
            function, None
        )[0]

    def _set_eigenmode_source_knls(self):
        """Upload modal bases and compile device TF/SF source kernels."""

        for source in self.grid.eigenmodesources:
            prepare_device_eigenmode_source(source, "metal", dev=self.dev)
        self.pso_eigenmode_magnetic = self._compile_pipeline(
            knl_eigenmode.update_eigenmode_magnetic,
            "update_eigenmode_magnetic",
        )
        self.pso_eigenmode_electric = self._compile_pipeline(
            knl_eigenmode.update_eigenmode_electric,
            "update_eigenmode_electric",
        )

    def _set_eigenmode_monitor_knl(self):
        """Upload modal projection bases and compile the DFT kernel."""

        for monitor in self.grid.eigenmodeports:
            prepare_device_eigenmode_monitor(monitor, "metal", dev=self.dev)
        self.pso_eigenmode_dft = self._compile_pipeline(
            knl_eigenmode.accumulate_eigenmode_dft,
            "accumulate_eigenmode_dft",
        )

    def _set_virtual_waveguide_knls(self):
        """Compile auxiliary-grid aperture coupling kernels."""

        self.pso_virtual_magnetic = self._compile_pipeline(
            knl_virtual_waveguide.couple_magnetic,
            "couple_virtual_waveguide_magnetic",
        )
        self.pso_virtual_clear_magnetic = self._compile_pipeline(
            knl_virtual_waveguide.clear_rear_magnetic,
            "clear_virtual_waveguide_rear_magnetic",
        )
        self.pso_virtual_electric = self._compile_pipeline(
            knl_virtual_waveguide.couple_electric,
            "couple_virtual_waveguide_electric",
        )
        self.pso_virtual_clear_electric = self._compile_pipeline(
            knl_virtual_waveguide.clear_rear_electric,
            "clear_virtual_waveguide_rear_electric",
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
        ) = htod_snapshot_array(self.grid.snapshots, self.dev)

        subs_func_snap = dict(self.subs_func)
        subs_func_snap.update(
            {
                "NX_SNAPS": Snapshot.nx_max,
                "NY_SNAPS": Snapshot.ny_max,
                "NZ_SNAPS": Snapshot.nz_max,
            }
        )
        bld = self._build_knl(knl_snapshots.store_snapshot, self.subs_name_args, subs_func_snap)
        lib, _ = self.dev.newLibraryWithSource_options_error_(bld, self.opts, None)
        self.update_store_snapshot_dev = lib.newFunctionWithName_("store_snapshot")
        self.pso_store_snapshot = self.dev.newComputePipelineStateWithFunction_error_(
            self.update_store_snapshot_dev, None
        )[0]

    def _metal_snapshot_buffers_to_numpy(self):
        """Converts the six device-resident snapshot buffers into host numpy
        arrays with the same shape htod_snapshot_array() allocated them
        with - MTLBuffer has no .get() (that's the CUDA/OpenCL array API);
        Metal buffers are read back via .contents().as_buffer(size)."""
        numsnaps = (
            1 if config.get_model_config().device["snapsgpu2cpu"] else len(self.grid.snapshots)
        )
        shape = (numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max)
        dtype = config.sim_config.dtypes["float_or_double"]
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        def _to_numpy(buf):
            return (
                np.frombuffer(buf.contents().as_buffer(nbytes), dtype=dtype).reshape(shape).copy()
            )

        return (
            _to_numpy(self.snapEx_dev),
            _to_numpy(self.snapEy_dev),
            _to_numpy(self.snapEz_dev),
            _to_numpy(self.snapHx_dev),
            _to_numpy(self.snapHy_dev),
            _to_numpy(self.snapHz_dev),
        )

    def store_outputs(self, iteration):
        """Stores field component values for every receiver.

        Args:
            iteration: int for iteration number.
        """
        if self.grid.rxs:
            self.cmdbuffer_store_outputs = self.cmdqueue.commandBuffer()
            self.cmpencoder_store_outputs = self.cmdbuffer_store_outputs.computeCommandEncoder()
            self.cmpencoder_store_outputs.setComputePipelineState_(self.pso_store_outputs)

            # Set buffer arguments for the kernel
            # NRX (number of receivers)
            nrx_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(len(self.grid.rxs)).tobytes(), 4, 0
            )
            self.cmpencoder_store_outputs.setBuffer_offset_atIndex_(nrx_buffer, 0, 0)

            # iteration
            iteration_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(iteration).tobytes(), 4, 0
            )
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
                self.metal.MTLSizeMake(
                    self.pso_store_outputs.maxTotalThreadsPerThreadgroup(), 1, 1
                ),
            )
            self.cmpencoder_store_outputs.endEncoding()
            if self.nrxcurrent:
                encoder = self.cmdbuffer_store_outputs.computeCommandEncoder()
                encoder.setComputePipelineState_(self.pso_store_current_outputs)
                ncurrent_buffer = self.dev.newBufferWithBytes_length_options_(
                    np.int32(self.nrxcurrent).tobytes(), 4, 0
                )
                real_dtype = config.sim_config.dtypes["float_or_double"]
                dx_buffer = self.dev.newBufferWithBytes_length_options_(
                    real_dtype(self.grid.dx).tobytes(), np.dtype(real_dtype).itemsize, 0
                )
                dy_buffer = self.dev.newBufferWithBytes_length_options_(
                    real_dtype(self.grid.dy).tobytes(), np.dtype(real_dtype).itemsize, 0
                )
                dz_buffer = self.dev.newBufferWithBytes_length_options_(
                    real_dtype(self.grid.dz).tobytes(), np.dtype(real_dtype).itemsize, 0
                )
                for index, buffer in enumerate(
                    (
                        ncurrent_buffer,
                        iteration_buffer,
                        self.rxcurrentinfo_dev,
                        self.rxcurrents_dev,
                        dx_buffer,
                        dy_buffer,
                        dz_buffer,
                        self.grid.Hx_dev,
                        self.grid.Hy_dev,
                        self.grid.Hz_dev,
                    )
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, index)
                encoder.dispatchThreads_threadsPerThreadgroup_(
                    self.metal.MTLSizeMake(round32(self.nrxcurrent), 1, 1),
                    self.metal.MTLSizeMake(
                        self.pso_store_current_outputs.maxTotalThreadsPerThreadgroup(),
                        1,
                        1,
                    ),
                )
                encoder.endEncoding()
            self.cmdbuffer_store_outputs.commit()
            self.cmdbuffer_store_outputs.waitUntilCompleted()

    def store_snapshots(self, iteration):
        """Stores any snapshots.

        Args:
            iteration: int for iteration number.
        """

        sx, sy, sz = _snapshot_axis_strides()
        for i, snap in enumerate(self.grid.snapshots):
            if snap.iteration == iteration:
                snapno = 0 if config.get_model_config().device["snapsgpu2cpu"] else i

                cmdbuffer_snap = self.cmdqueue.commandBuffer()
                cmpencoder_snap = cmdbuffer_snap.computeCommandEncoder()
                cmpencoder_snap.setComputePipelineState_(self.pso_store_snapshot)

                scalar_args = (
                    snapno,
                    snap.xs,
                    snap.ys,
                    snap.zs,
                    snap.nx,
                    snap.ny,
                    snap.nz,
                    snap.dx,
                    snap.dy,
                    snap.dz,
                    sx,
                    sy,
                    sz,
                )
                for index, value in enumerate(scalar_args):
                    buf = self.dev.newBufferWithBytes_length_options_(
                        np.int32(value).tobytes(), 4, 0
                    )
                    cmpencoder_snap.setBuffer_offset_atIndex_(buf, 0, index)

                field_args = (
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
                for offset, buf in enumerate(field_args):
                    cmpencoder_snap.setBuffer_offset_atIndex_(buf, 0, len(scalar_args) + offset)

                total_threads = Snapshot.nx_max * Snapshot.ny_max * Snapshot.nz_max
                cmpencoder_snap.dispatchThreads_threadsPerThreadgroup_(
                    self.metal.MTLSizeMake(round32(total_threads), 1, 1),
                    self.metal.MTLSizeMake(
                        self.pso_store_snapshot.maxTotalThreadsPerThreadgroup(), 1, 1
                    ),
                )
                cmpencoder_snap.endEncoding()
                cmdbuffer_snap.commit()
                cmdbuffer_snap.waitUntilCompleted()

                if config.get_model_config().device["snapsgpu2cpu"]:
                    dtoh_snapshot_array(*self._metal_snapshot_buffers_to_numpy(), 0, snap)

    def observe_ntff_electric(self, iteration):
        """Collect electric frequency- and time-domain KSIR data on Metal."""

        collector = getattr(self, "ntff_collector", None)
        if collector is not None:
            collector.observe_electric(iteration)

    def observe_ntff_magnetic(self, iteration):
        """Collect magnetic frequency- and time-domain KSIR data on Metal."""

        collector = getattr(self, "ntff_collector", None)
        if collector is not None:
            collector.observe_magnetic(iteration)

    def observe_sar_electric(self, iteration):
        """Collect sparse electric-field DFTs for SAR on Metal."""

        collector = getattr(self, "sar_collector", None)
        if collector is not None:
            collector.observe_electric(iteration)

    def _dispatch_1d(self, pipeline, scalars, buffers, npoints):
        """Bind and execute one bounds-checked one-dimensional Metal kernel."""

        command = self.cmdqueue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, value in enumerate(scalars):
            encoder.setBytes_length_atIndex_(value.tobytes(), value.nbytes, index)
        for index, buffer in enumerate(buffers, start=len(scalars)):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.metal.MTLSizeMake(int(npoints), 1, 1),
            self.metal.MTLSizeMake(pipeline.maxTotalThreadsPerThreadgroup(), 1, 1),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

    def observe_eigenmode_ports(self, iteration):
        """Project modal port fields and accumulate DFTs on Metal."""

        real = config.sim_config.dtypes["float_or_double"]
        for monitor in self.grid.eigenmodeports:
            if iteration != monitor._next_iteration:
                raise RuntimeError(
                    f"expected eigenmode DFT iteration {monitor._next_iteration}, "
                    f"received {iteration}"
                )
            owner = monitor.owner
            arrays = monitor.device_arrays
            nf, nm = monitor.electric_dft.shape
            parameters = monitor.device_parameters
            parameters[0] = (
                nf,
                nm,
                owner.normal_axis,
                1 if owner.direction == "+" else -1,
                monitor.magnetic_side,
                owner.transverse_start[0],
                owner.transverse_start[1],
                owner.transverse_stop[0],
                owner.transverse_stop[1],
                owner.plane_index,
                self.grid.dt,
                monitor.measure,
                monitor.handedness,
            )
            buffers = (
                arrays["electric_phase_real"],
                arrays["electric_phase_imag"],
                arrays["magnetic_phase_real"],
                arrays["magnetic_phase_imag"],
                arrays["phase_step_real"],
                arrays["phase_step_imag"],
                arrays["conj_eu_real"],
                arrays["conj_eu_imag"],
                arrays["conj_ev_real"],
                arrays["conj_ev_imag"],
                arrays["conj_hu_real"],
                arrays["conj_hu_imag"],
                arrays["conj_hv_real"],
                arrays["conj_hv_imag"],
                arrays["electric_dft_real"],
                arrays["electric_dft_imag"],
                arrays["magnetic_dft_real"],
                arrays["magnetic_dft_imag"],
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                self.grid.Hx_dev,
                self.grid.Hy_dev,
                self.grid.Hz_dev,
            )
            self._dispatch_1d(self.pso_eigenmode_dft, (parameters,), buffers, nf)
            monitor._next_iteration += 1
            reanchor_device_eigenmode_monitor(monitor, self.grid, "metal")

    def _launch_eigenmode_source(self, source, iteration, magnetic):
        nu, nv = np.asarray(source.transverse_stop) - np.asarray(source.transverse_start)
        npoints = int((nu + 1) * (nv + 1))
        pipeline = self.pso_eigenmode_magnetic if magnetic else self.pso_eigenmode_electric
        profiles = source.device_electric_profiles if magnetic else source.device_magnetic_profiles
        coefficients = self.grid.updatecoeffsH_dev if magnetic else self.grid.updatecoeffsE_dev
        fields = (
            (self.grid.Hx_dev, self.grid.Hy_dev, self.grid.Hz_dev)
            if magnetic
            else (self.grid.Ex_dev, self.grid.Ey_dev, self.grid.Ez_dev)
        )
        real = config.sim_config.dtypes["float_or_double"]
        for basis, envelope in eigenmode_source_envelopes(source, self.grid, iteration, magnetic):
            scalars = (
                np.int32(npoints),
                np.int32(source.normal_axis),
                np.int32(1 if source.direction == "+" else -1),
                np.int32(source.transverse_start[0]),
                np.int32(source.transverse_start[1]),
                np.int32(source.transverse_stop[0]),
                np.int32(source.transverse_stop[1]),
                np.int32(source.plane_index),
                np.int32(basis),
                real(envelope),
            )
            self._dispatch_1d(
                pipeline,
                scalars,
                (profiles, coefficients, self.grid.ID_dev, *fields),
                npoints,
            )

    def update_eigenmode_sources_magnetic(self, iteration):
        for source in self.grid.eigenmodesources:
            self._launch_eigenmode_source(source, iteration, True)
        for guide in self.grid.virtual_waveguides:
            self._update_virtual_waveguide_magnetic(guide, iteration)

    def update_eigenmode_sources_electric(self, iteration):
        for source in self.grid.eigenmodesources:
            self._launch_eigenmode_source(source, iteration, False)
        for guide in self.grid.virtual_waveguides:
            self._update_virtual_waveguide_electric(guide, iteration)

    @staticmethod
    def _virtual_scalars(guide, npoints):
        aux = guide.aux_grid
        return (
            np.int32(npoints),
            np.int32(guide.normal_axis),
            np.int32(guide.direction_sign),
            np.int32(guide.u0),
            np.int32(guide.v0),
            np.int32(guide.u1),
            np.int32(guide.v1),
            np.int32(guide.plane_index),
            np.int32(aux.nx),
            np.int32(aux.ny),
            np.int32(aux.nz),
        )

    def _update_virtual_waveguide_magnetic(self, guide, iteration):
        child = guide.aux_updates
        child.update_magnetic()
        child.update_magnetic_pml()
        child.update_eigenmode_sources_magnetic(iteration)
        nface = (guide.nu + 1) * (guide.nv + 1)
        main_h = (self.grid.Hx_dev, self.grid.Hy_dev, self.grid.Hz_dev)
        aux_h = (
            guide.aux_grid.Hx_dev,
            guide.aux_grid.Hy_dev,
            guide.aux_grid.Hz_dev,
        )
        self._dispatch_1d(
            self.pso_virtual_magnetic,
            self._virtual_scalars(guide, nface),
            (*main_h, *aux_h),
            nface,
        )
        nmain = self.grid.Ex.size
        self._dispatch_1d(
            self.pso_virtual_clear_magnetic,
            self._virtual_scalars(guide, nmain),
            (*main_h, *aux_h),
            nmain,
        )

    def _update_virtual_waveguide_electric(self, guide, iteration):
        child = guide.aux_updates
        child.update_electric_a()
        child.update_electric_pml()
        child.update_eigenmode_sources_electric(iteration)
        child.update_electric_b()
        nface = (guide.nu + 1) * (guide.nv + 1)
        main_fields = (
            self.grid.Ex_dev,
            self.grid.Ey_dev,
            self.grid.Ez_dev,
            self.grid.Hx_dev,
            self.grid.Hy_dev,
            self.grid.Hz_dev,
        )
        aux_fields = (
            guide.aux_grid.Ex_dev,
            guide.aux_grid.Ey_dev,
            guide.aux_grid.Ez_dev,
            guide.aux_grid.Hx_dev,
            guide.aux_grid.Hy_dev,
            guide.aux_grid.Hz_dev,
        )
        self._dispatch_1d(
            self.pso_virtual_electric,
            self._virtual_scalars(guide, nface),
            (
                guide.aux_grid.updatecoeffsE_dev,
                guide.aux_grid.ID_dev,
                *main_fields,
                *aux_fields,
            ),
            nface,
        )
        nmain = self.grid.Ex.size
        self._dispatch_1d(
            self.pso_virtual_clear_electric,
            self._virtual_scalars(guide, nmain),
            (
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                guide.aux_grid.Ex_dev,
                guide.aux_grid.Ey_dev,
                guide.aux_grid.Ez_dev,
            ),
            nmain,
        )

    def update_magnetic(self):
        """Updates magnetic field components."""
        self.cmdbufferH = self.cmdqueue.commandBuffer()
        self.cmpencoderH = self.cmdbufferH.computeCommandEncoder()
        self.cmpencoderH.setComputePipelineState_(self.psoH)

        # Set scalar values for H update kernel (similar to E update)
        # See update_electric_a()'s comment - must be the raw cell count
        # (matching CUDA/OpenCL), not the field-array dimension.
        nx_value = np.int32(self.grid.nx)
        ny_value = np.int32(self.grid.ny)
        nz_value = np.int32(self.grid.nz)

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
        if getattr(self.grid, "transmissionlines", ()):
            real = config.sim_config.dtypes["float_or_double"]
            self._dispatch_1d(
                self.pso_transmission_line_magnetic,
                (
                    np.int32(len(self.grid.transmissionlines)),
                    np.int32(iteration),
                    real(self.grid.dx),
                    real(self.grid.dy),
                    real(self.grid.dz),
                    self.tl_line_coefficient,
                ),
                (
                    self.tl_info_dev,
                    self.tl_resistance_dev,
                    self.tl_waveform_half_dev,
                    self.tl_voltage_dev,
                    self.tl_current_dev,
                    self.tl_Vtotal_dev,
                    self.tl_Itotal_dev,
                    self.grid.Hx_dev,
                    self.grid.Hy_dev,
                    self.grid.Hz_dev,
                ),
                len(self.grid.transmissionlines),
            )

        if self.grid.magneticdipoles:
            real_dtype = config.sim_config.dtypes["float_or_double"]
            real_nbytes = np.dtype(real_dtype).itemsize

            # Create command buffer for magnetic dipoles
            cmdbuffer_magnetic = self.cmdqueue.commandBuffer()
            cmpencoder_magnetic = cmdbuffer_magnetic.computeCommandEncoder()
            cmpencoder_magnetic.setComputePipelineState_(self.pso_magnetic_dipole)

            # Set buffer arguments for magnetic dipole kernel
            n_magnetic_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(len(self.grid.magneticdipoles)).tobytes(), 4, 0
            )
            cmpencoder_magnetic.setBuffer_offset_atIndex_(n_magnetic_buffer, 0, 0)

            iteration_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(iteration).tobytes(), 4, 0
            )
            cmpencoder_magnetic.setBuffer_offset_atIndex_(iteration_buffer, 0, 1)

            # Set spatial discretization buffers
            dx_buffer = self.dev.newBufferWithBytes_length_options_(
                real_dtype(self.grid.dx).tobytes(), real_nbytes, 0
            )
            cmpencoder_magnetic.setBuffer_offset_atIndex_(dx_buffer, 0, 2)

            dy_buffer = self.dev.newBufferWithBytes_length_options_(
                real_dtype(self.grid.dy).tobytes(), real_nbytes, 0
            )
            cmpencoder_magnetic.setBuffer_offset_atIndex_(dy_buffer, 0, 3)

            dz_buffer = self.dev.newBufferWithBytes_length_options_(
                real_dtype(self.grid.dz).tobytes(), real_nbytes, 0
            )
            cmpencoder_magnetic.setBuffer_offset_atIndex_(dz_buffer, 0, 4)

            # Set source info and waveform buffers
            cmpencoder_magnetic.setBuffer_offset_atIndex_(self.srcinfo1_magnetic_dev, 0, 5)
            cmpencoder_magnetic.setBuffer_offset_atIndex_(self.srcinfo2_magnetic_dev, 0, 6)
            cmpencoder_magnetic.setBuffer_offset_atIndex_(self.srcwaves_magnetic_dev, 0, 7)

            # Set ID and field buffers
            cmpencoder_magnetic.setBuffer_offset_atIndex_(self.grid.ID_dev, 0, 8)
            cmpencoder_magnetic.setBuffer_offset_atIndex_(self.grid.Hx_dev, 0, 9)
            cmpencoder_magnetic.setBuffer_offset_atIndex_(self.grid.Hy_dev, 0, 10)
            cmpencoder_magnetic.setBuffer_offset_atIndex_(self.grid.Hz_dev, 0, 11)

            # Dispatch the kernel
            cmpencoder_magnetic.dispatchThreads_threadsPerThreadgroup_(
                self.metal.MTLSizeMake(round32(len(self.grid.magneticdipoles)), 1, 1),
                self.metal.MTLSizeMake(
                    self.pso_magnetic_dipole.maxTotalThreadsPerThreadgroup(), 1, 1
                ),
            )
            cmpencoder_magnetic.endEncoding()
            cmdbuffer_magnetic.commit()
            cmdbuffer_magnetic.waitUntilCompleted()

        if self.grid.magneticfrillsources:
            cmdbuffer = self.cmdqueue.commandBuffer()
            encoder = cmdbuffer.computeCommandEncoder()
            encoder.setComputePipelineState_(self.pso_magnetic_frill)

            nfrill = np.int32(len(self.grid.magneticfrillsources))
            iteration_value = np.int32(iteration)
            encoder.setBytes_length_atIndex_(nfrill.tobytes(), 4, 0)
            encoder.setBytes_length_atIndex_(iteration_value.tobytes(), 4, 1)
            buffers = (
                self.frill_term_counts_dev,
                self.frill_term_info_dev,
                self.frill_term_params_dev,
                self.frill_params_dev,
                self.frill_state_dev,
                self.frill_waveform_dev,
                self.frill_Vinc_dev,
                self.frill_Vtotal_dev,
                self.frill_Itot_dev,
                self.grid.Hx_dev,
                self.grid.Hy_dev,
                self.grid.Hz_dev,
            )
            for index, buffer in enumerate(buffers, start=2):
                encoder.setBuffer_offset_atIndex_(buffer, 0, index)

            encoder.dispatchThreads_threadsPerThreadgroup_(
                self.metal.MTLSizeMake(int(nfrill), 1, 1),
                self.metal.MTLSizeMake(
                    self.pso_magnetic_frill.maxTotalThreadsPerThreadgroup(), 1, 1
                ),
            )
            encoder.endEncoding()
            cmdbuffer.commit()
            cmdbuffer.waitUntilCompleted()

    def update_plane_waves_magnetic(self, iteration):
        """Advance Metal auxiliary plane waves and apply magnetic TF/SF corrections."""

        if self.planewave_controller is not None:
            self.planewave_controller.update_plane_waves_magnetic(iteration)

    def update_network_terminals(self, iteration):
        """Apply sparse rational-network corrections entirely on Metal."""

        if not self.grid.networkterminals:
            return
        command = self.cmdqueue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pso_rational_network)
        terminal_count = np.int32(len(self.grid.networkterminals))
        iteration_value = np.int32(iteration)
        dt_value = config.sim_config.dtypes["float_or_double"](self.grid.dt)
        encoder.setBytes_length_atIndex_(terminal_count.tobytes(), 4, 0)
        encoder.setBytes_length_atIndex_(iteration_value.tobytes(), 4, 1)
        encoder.setBytes_length_atIndex_(dt_value.tobytes(), dt_value.nbytes, 2)
        buffers = (
            self.rn_info_dev,
            self.rn_params_dev,
            self.rn_waveform_whole_dev,
            self.rn_waveform_half_dev,
            self.rn_voltage_dev,
            self.rn_current_dev,
            self.rn_exp_half_real_dev,
            self.rn_exp_half_imag_dev,
            self.rn_coeff_half_new_real_dev,
            self.rn_coeff_half_new_imag_dev,
            self.rn_coeff_half_old_real_dev,
            self.rn_coeff_half_old_imag_dev,
            self.rn_exp_full_real_dev,
            self.rn_exp_full_imag_dev,
            self.rn_coeff_full_new_real_dev,
            self.rn_coeff_full_new_imag_dev,
            self.rn_coeff_full_old_real_dev,
            self.rn_coeff_full_old_imag_dev,
            self.rn_state_real_dev,
            self.rn_state_imag_dev,
            self.grid.Ex_dev,
            self.grid.Ey_dev,
            self.grid.Ez_dev,
        )
        for index, buffer in enumerate(buffers, start=3):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.metal.MTLSizeMake(int(terminal_count), 1, 1),
            self.metal.MTLSizeMake(self.pso_rational_network.maxTotalThreadsPerThreadgroup(), 1, 1),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

    def update_electric_a(self):
        """Updates electric field components."""

        # All materials are non-dispersive so do standard update.
        if self.grid.maxpoles == 0:
            self.cmdbufferE = self.cmdqueue.commandBuffer()
            self.cmpencoderE = self.cmdbufferE.computeCommandEncoder()
            self.cmpencoderE.setComputePipelineState_(self.psoE)

            # For Metal, we need to set the scalar values using setBytes, not buffers
            # Set NX, NY, NZ as scalar values (Metal expects device const int&)
            # NX/NY/NZ here are the kernel's bounds-check comparison values
            # (x < NX etc in knl_fields_updates.py), which must be the raw
            # cell count, matching CUDA/OpenCL (np.int32(self.grid.nx), no
            # +1) - NOT the field-array dimension (nx+1). The +1 previously
            # used here let every boundary-plane bounds check admit one
            # extra plane it shouldn't, corrupting domain-boundary fields.
            nx_value = np.int32(self.grid.nx)
            ny_value = np.int32(self.grid.ny)
            nz_value = np.int32(self.grid.nz)

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
            cmdbuffer = self.cmdqueue.commandBuffer()
            cmpencoder = cmdbuffer.computeCommandEncoder()
            cmpencoder.setComputePipelineState_(self.pso_dispersive_a)

            nx_value = np.int32(self.grid.nx)
            ny_value = np.int32(self.grid.ny)
            nz_value = np.int32(self.grid.nz)
            maxpoles_value = np.int32(self.grid.maxpoles)

            cmpencoder.setBytes_length_atIndex_(nx_value.tobytes(), 4, 0)
            cmpencoder.setBytes_length_atIndex_(ny_value.tobytes(), 4, 1)
            cmpencoder.setBytes_length_atIndex_(nz_value.tobytes(), 4, 2)
            cmpencoder.setBytes_length_atIndex_(maxpoles_value.tobytes(), 4, 3)

            # Buffer index contract matches knl_fields_updates.
            # update_electric_dispersive_A's args_metal signature exactly:
            # NX, NY, NZ, MAXPOLES, updatecoeffsdispersive, Tx, Ty, Tz, ID,
            # Ex, Ey, Ez, Hx, Hy, Hz (indices 0-14).
            cmpencoder.setBuffer_offset_atIndex_(self.grid.updatecoeffsdispersive_dev, 0, 4)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Tx_dev, 0, 5)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Ty_dev, 0, 6)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Tz_dev, 0, 7)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.ID_dev, 0, 8)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Ex_dev, 0, 9)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Ey_dev, 0, 10)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Ez_dev, 0, 11)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Hx_dev, 0, 12)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Hy_dev, 0, 13)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Hz_dev, 0, 14)

            # Per-group thread size must come from THIS kernel's own
            # pipeline, not the shared self.grid.tgs (which is sized for
            # psoE specifically) - the dispersive kernel's per-pole loop
            # likely has different register pressure/occupancy limits, and
            # reusing a differently-sized pipeline's threadgroup limit is
            # exactly the class of bug fixed in _set_rx_knl() above.
            cmpencoder.dispatchThreads_threadsPerThreadgroup_(
                self.grid.tptg,
                self.metal.MTLSizeMake(self.pso_dispersive_a.maxTotalThreadsPerThreadgroup(), 1, 1),
            )

            cmpencoder.endEncoding()
            cmdbuffer.commit()
            cmdbuffer.waitUntilCompleted()

    def update_symmetry_boundaries_electric(self):
        """Apply the PMC ghost-image correction on Metal."""
        if "pmc" not in self.grid.symmetry_boundaries.values():
            return
        dispersive = self.grid.maxpoles > 0
        pipeline = self.pso_electric_pmc_dispersive if dispersive else self.pso_electric_pmc
        command = self.cmdqueue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        scalars = [
            np.int32(self.grid.nx),
            np.int32(self.grid.ny),
            np.int32(self.grid.nz),
        ]
        if dispersive:
            scalars.append(np.int32(self.grid.maxpoles))
        scalars.extend(self._pmc_flags())
        for index, value in enumerate(scalars):
            encoder.setBytes_length_atIndex_(value.tobytes(), 4, index)
        buffers = []
        if dispersive:
            buffers.extend(
                [
                    self.grid.updatecoeffsdispersive_dev,
                    self.grid.Tx_dev,
                    self.grid.Ty_dev,
                    self.grid.Tz_dev,
                ]
            )
        buffers.extend(
            [
                self.grid.ID_dev,
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                self.grid.Hx_dev,
                self.grid.Hy_dev,
                self.grid.Hz_dev,
            ]
        )
        for index, buffer in enumerate(buffers, start=len(scalars)):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)

        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.grid.tptg,
            self.metal.MTLSizeMake(pipeline.maxTotalThreadsPerThreadgroup(), 1, 1),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

    def update_electric_pml(self):
        """Updates electric field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_electric()

    def update_electric_sources(self, iteration):
        """Updates electric field components from sources -
        update any Hertzian dipole sources last.
        """
        if self.grid.voltagesources:
            real_dtype = config.sim_config.dtypes["float_or_double"]
            real_nbytes = np.dtype(real_dtype).itemsize

            # Create command buffer for voltage sources
            cmdbuffer_voltage = self.cmdqueue.commandBuffer()
            cmpencoder_voltage = cmdbuffer_voltage.computeCommandEncoder()
            cmpencoder_voltage.setComputePipelineState_(self.pso_voltage_source)

            # Set buffer arguments for voltage source kernel
            n_voltage_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(len(self.grid.voltagesources)).tobytes(), 4, 0
            )
            cmpencoder_voltage.setBuffer_offset_atIndex_(n_voltage_buffer, 0, 0)

            iteration_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(iteration).tobytes(), 4, 0
            )
            cmpencoder_voltage.setBuffer_offset_atIndex_(iteration_buffer, 0, 1)

            # Set spatial discretization buffers - matches
            # knl_source_updates.update_voltage_source's args_metal
            # signature (NVOLTSRC, iteration, dx, dy, dz, srcinfo1,
            # srcinfo2, srcwaveforms, ID, Ex, Ey, Ez): dx/dy/dz and ID were
            # previously never bound at all, and every argument after
            # iteration was shifted by 4 slots as a result.
            dx_buffer = self.dev.newBufferWithBytes_length_options_(
                real_dtype(self.grid.dx).tobytes(), real_nbytes, 0
            )
            cmpencoder_voltage.setBuffer_offset_atIndex_(dx_buffer, 0, 2)

            dy_buffer = self.dev.newBufferWithBytes_length_options_(
                real_dtype(self.grid.dy).tobytes(), real_nbytes, 0
            )
            cmpencoder_voltage.setBuffer_offset_atIndex_(dy_buffer, 0, 3)

            dz_buffer = self.dev.newBufferWithBytes_length_options_(
                real_dtype(self.grid.dz).tobytes(), real_nbytes, 0
            )
            cmpencoder_voltage.setBuffer_offset_atIndex_(dz_buffer, 0, 4)

            # Set source info and waveform buffers
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.srcinfo1_voltage_dev, 0, 5)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.srcinfo2_voltage_dev, 0, 6)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.srcwaves_voltage_dev, 0, 7)

            # Set ID and field buffers
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.grid.ID_dev, 0, 8)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.grid.Ex_dev, 0, 9)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.grid.Ey_dev, 0, 10)
            cmpencoder_voltage.setBuffer_offset_atIndex_(self.grid.Ez_dev, 0, 11)

            # Dispatch the kernel
            cmpencoder_voltage.dispatchThreads_threadsPerThreadgroup_(
                self.metal.MTLSizeMake(round32(len(self.grid.voltagesources)), 1, 1),
                self.metal.MTLSizeMake(
                    self.pso_voltage_source.maxTotalThreadsPerThreadgroup(), 1, 1
                ),
            )
            cmpencoder_voltage.endEncoding()
            cmdbuffer_voltage.commit()
            cmdbuffer_voltage.waitUntilCompleted()

        if getattr(self.grid, "transmissionlines", ()):
            real = config.sim_config.dtypes["float_or_double"]
            self._dispatch_1d(
                self.pso_transmission_line_electric,
                (
                    np.int32(len(self.grid.transmissionlines)),
                    np.int32(iteration),
                    real(self.grid.dx),
                    real(self.grid.dy),
                    real(self.grid.dz),
                    self.tl_line_coefficient,
                    self.tl_abc_coefficient,
                ),
                (
                    self.tl_info_dev,
                    self.tl_resistance_dev,
                    self.tl_waveform_whole_dev,
                    self.tl_voltage_dev,
                    self.tl_current_dev,
                    self.tl_abcv0_dev,
                    self.tl_abcv1_dev,
                    self.grid.Ex_dev,
                    self.grid.Ey_dev,
                    self.grid.Ez_dev,
                ),
                len(self.grid.transmissionlines),
            )

        if self.grid.hertziandipoles:
            real_dtype = config.sim_config.dtypes["float_or_double"]
            real_nbytes = np.dtype(real_dtype).itemsize

            # Optional debug logging for first iteration only
            if iteration == 1:
                logger.debug(
                    f"Metal backend: {len(self.grid.hertziandipoles)} Hertzian dipole(s) at iteration {iteration}"
                )
                for i, src in enumerate(self.grid.hertziandipoles):
                    logger.debug(
                        f"  Source {i}: position=({src.xcoord},{src.ycoord},{src.zcoord}), polarisation={src.polarisation}"
                    )

            # Create command buffer for Hertzian dipoles
            cmdbuffer_hertzian = self.cmdqueue.commandBuffer()
            cmpencoder_hertzian = cmdbuffer_hertzian.computeCommandEncoder()
            cmpencoder_hertzian.setComputePipelineState_(self.pso_hertzian_dipole)

            # Set buffer arguments for Hertzian dipole kernel
            n_hertzian_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(len(self.grid.hertziandipoles)).tobytes(), 4, 0
            )
            cmpencoder_hertzian.setBuffer_offset_atIndex_(n_hertzian_buffer, 0, 0)

            iteration_buffer = self.dev.newBufferWithBytes_length_options_(
                np.int32(iteration).tobytes(), 4, 0
            )
            cmpencoder_hertzian.setBuffer_offset_atIndex_(iteration_buffer, 0, 1)

            # Set spatial discretization buffers
            dx_buffer = self.dev.newBufferWithBytes_length_options_(
                real_dtype(self.grid.dx).tobytes(), real_nbytes, 0
            )
            cmpencoder_hertzian.setBuffer_offset_atIndex_(dx_buffer, 0, 2)

            dy_buffer = self.dev.newBufferWithBytes_length_options_(
                real_dtype(self.grid.dy).tobytes(), real_nbytes, 0
            )
            cmpencoder_hertzian.setBuffer_offset_atIndex_(dy_buffer, 0, 3)

            dz_buffer = self.dev.newBufferWithBytes_length_options_(
                real_dtype(self.grid.dz).tobytes(), real_nbytes, 0
            )
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
                self.metal.MTLSizeMake(
                    self.pso_hertzian_dipole.maxTotalThreadsPerThreadgroup(), 1, 1
                ),
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
                    logger.debug(
                        f"Metal backend after source kernel (iteration {iteration}): Ex max_abs={max_abs_ex:.2e}, nonzero={nonzero_count}"
                    )

                except Exception as e:
                    logger.exception(f"Error checking fields after source kernel: {e}")

        self.grid.iteration += 1

    def update_plane_waves_electric(self, iteration):
        """Advance Metal auxiliary plane waves and apply electric TF/SF corrections."""

        if self.planewave_controller is not None:
            self.planewave_controller.update_plane_waves_electric(iteration)

    def update_electric_b(self):
        """If there are any dispersive materials do 2nd part of dispersive
        update - it is split into two parts as it requires present and
        updated electric field values. Therefore it can only be completely
        updated after the electric field has been updated by the PML and
        source updates.
        """
        if self.grid.maxpoles > 0:
            cmdbuffer = self.cmdqueue.commandBuffer()
            cmpencoder = cmdbuffer.computeCommandEncoder()
            cmpencoder.setComputePipelineState_(self.pso_dispersive_b)

            nx_value = np.int32(self.grid.nx)
            ny_value = np.int32(self.grid.ny)
            nz_value = np.int32(self.grid.nz)
            maxpoles_value = np.int32(self.grid.maxpoles)

            cmpencoder.setBytes_length_atIndex_(nx_value.tobytes(), 4, 0)
            cmpencoder.setBytes_length_atIndex_(ny_value.tobytes(), 4, 1)
            cmpencoder.setBytes_length_atIndex_(nz_value.tobytes(), 4, 2)
            cmpencoder.setBytes_length_atIndex_(maxpoles_value.tobytes(), 4, 3)

            # Buffer index contract matches knl_fields_updates.
            # update_electric_dispersive_B's args_metal signature exactly:
            # NX, NY, NZ, MAXPOLES, updatecoeffsdispersive, Tx, Ty, Tz, ID,
            # Ex, Ey, Ez (indices 0-11 - no H components, unlike phase A).
            cmpencoder.setBuffer_offset_atIndex_(self.grid.updatecoeffsdispersive_dev, 0, 4)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Tx_dev, 0, 5)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Ty_dev, 0, 6)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Tz_dev, 0, 7)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.ID_dev, 0, 8)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Ex_dev, 0, 9)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Ey_dev, 0, 10)
            cmpencoder.setBuffer_offset_atIndex_(self.grid.Ez_dev, 0, 11)

            cmpencoder.dispatchThreads_threadsPerThreadgroup_(
                self.grid.tptg,
                self.metal.MTLSizeMake(self.pso_dispersive_b.maxTotalThreadsPerThreadgroup(), 1, 1),
            )

            cmpencoder.endEncoding()
            cmdbuffer.commit()
            cmdbuffer.waitUntilCompleted()

    def update_symmetry_boundaries_electric_b(self):
        """Complete the dispersive PMC ADE update on Metal."""
        if (
            "pmc" not in self.grid.symmetry_boundaries.values()
            or self.grid.maxpoles == 0
        ):
            return
        command = self.cmdqueue.commandBuffer()
        encoder = command.computeCommandEncoder()
        pipeline = self.pso_electric_pmc_dispersive_b
        encoder.setComputePipelineState_(pipeline)
        scalars = (
            np.int32(self.grid.nx),
            np.int32(self.grid.ny),
            np.int32(self.grid.nz),
            np.int32(self.grid.maxpoles),
            *self._pmc_flags(),
        )
        for index, value in enumerate(scalars):
            encoder.setBytes_length_atIndex_(value.tobytes(), 4, index)
        buffers = (
            self.grid.updatecoeffsdispersive_dev,
            self.grid.Tx_dev,
            self.grid.Ty_dev,
            self.grid.Tz_dev,
            self.grid.ID_dev,
            self.grid.Ex_dev,
            self.grid.Ey_dev,
            self.grid.Ez_dev,
        )
        for index, buffer in enumerate(buffers, start=len(scalars)):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.grid.tptg,
            self.metal.MTLSizeMake(pipeline.maxTotalThreadsPerThreadgroup(), 1, 1),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

    def time_start(self):
        """Start the wall timer used to calculate solving time for a model.

        Metal launches in this backend wait for each command buffer to finish,
        so elapsed wall time measures the completed GPU work without requiring
        a separate end event.
        """
        self.timestart = timer()

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
        return timer() - self.timestart

    def finalise(self):
        """Copies data from compute device back to CPU to save to file(s)."""
        collector = getattr(self, "ntff_collector", None)
        if collector is not None:
            collector.finalise()
        sar_collector = getattr(self, "sar_collector", None)
        if sar_collector is not None:
            sar_collector.finalise()

        # Copy output from receivers array back to correct receiver objects
        if self.grid.rxs:
            dtoh_rx_array(
                self.rxs_dev,
                self.rxcoords_dev,
                self.grid,
                self.rxcurrents_dev if self.nrxcurrent else None,
            )

        if self.grid.magneticfrillsources:
            dtoh_magnetic_frill_source_outputs(
                self.frill_Vinc_dev,
                self.frill_Vtotal_dev,
                self.frill_Itot_dev,
                self.grid,
            )

        if getattr(self.grid, "transmissionlines", ()):
            dtoh_transmission_line_outputs(self.tl_Vtotal_dev, self.tl_Itotal_dev, self.grid)

        if getattr(self.grid, "networkterminals", ()):
            dtoh_rational_network_outputs(self.rn_voltage_dev, self.rn_current_dev, self.grid)

        if getattr(self.grid, "eigenmodeports", ()):
            finalise_device_eigenmode_monitors(self.grid.eigenmodeports, "metal")

        # Copy data from any snapshots back to correct snapshot objects
        if self.grid.snapshots and not config.get_model_config().device["snapsgpu2cpu"]:
            snap_arrays = self._metal_snapshot_buffers_to_numpy()
            for i, snap in enumerate(self.grid.snapshots):
                dtoh_snapshot_array(*snap_arrays, i, snap)

    def cleanup(self):
        pass
