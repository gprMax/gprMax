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

import humanize
import numpy as np
from jinja2 import Environment, PackageLoader

from gprMax import config
from gprMax.cuda_opencl import (
    knl_fields_updates,
    knl_planewave_updates,
    knl_snapshots,
    knl_source_updates,
    knl_store_outputs,
    knl_symmetry_boundaries,
    knl_tfsf_injection,
    knl_transmission_line,
)
from gprMax.grid.cuda_grid import CUDAGrid
from gprMax.ntff.device import CUDACombinedKSIRCollector
from gprMax.receivers import dtoh_rx_array, htod_rx_arrays
from gprMax.snapshots import (
    Snapshot,
    _snapshot_axis_strides,
    dtoh_snapshot_array,
    htod_snapshot_array,
    update_snapshot_max_dims,
)
from gprMax.sources import (
    dtoh_transmission_line_outputs,
    htod_src_arrays,
    htod_transmission_line_arrays,
)
from gprMax.updates.updates import Updates
from gprMax.utilities.utilities import round32

logger = logging.getLogger(__name__)


class CUDAUpdates(Updates[CUDAGrid]):
    """Defines update functions for GPU-based (CUDA) solver."""

    def __init__(self, G: CUDAGrid):
        """
        Args:
            G: CUDAGrid class describing a grid in a model.
        """
        super().__init__(G)

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

        # Must happen before _set_macros(), which bakes NX_SNAPS/NY_SNAPS/
        # NZ_SNAPS into the shared kernel preamble - see
        # update_snapshot_max_dims()'s docstring.
        if self.grid.snapshots:
            update_snapshot_max_dims(self.grid.snapshots)

        # Initialise arrays on GPU, prepare kernels, and get kernel functions
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
        if self.grid.transmissionlines:
            self._set_transmission_line_knls()
        if self.grid.snapshots:
            self._set_snapshot_knl()
        if self.grid.discreteplanewaves:
            self._set_planewave_knls()
        self.ntff_collector = None
        if self.grid.ntff_monitors:
            self.ntff_c_real = config.sim_config.dtypes["C_float_or_double"]
            self.ntff_compiler_options = config.sim_config.devices["nvcc_opts"]
            self.ntff_collector = CUDACombinedKSIRCollector(self)

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
            DRUDELORENTZ=config.get_model_config().materials["drudelorentz"],
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
            # Row stride for srcwaveforms must match htod_src_arrays()'s
            # actual allocation (sources.py: (len(sources), G.iterations + 1),
            # not G.iterations - the extra column holds a half-timestep
            # sample). A stride one short here doesn't just misindex the
            # last column: every row after the first is read shifted, since
            # the stride mismatch compounds per source (source i's data
            # starts 1*i elements early).
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

    def _set_symmetry_boundary_knl(self):
        """Build the nondispersive PMC ghost-image boundary kernel."""
        source = self._build_knl(
            knl_symmetry_boundaries.update_electric_pmc,
            self.subs_name_args,
            self.subs_func,
        )
        module = self.source_module(source, options=config.sim_config.devices["nvcc_opts"])
        self.update_electric_pmc_dev = module.get_function("update_electric_pmc")
        self._copy_mat_coeffs(module, module)

    def _pmc_flags(self):
        boundaries = self.grid.symmetry_boundaries
        return tuple(
            np.int32(boundaries.get(face) == "pmc")
            for face in ("x0", "xmax", "y0", "ymax", "z0", "zmax")
        )

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
        self.subs_func.update({"NY_SRCINFO": 4, "NY_SRCWAVES": self.grid.iterations + 1})

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

    def _set_transmission_line_knls(self):
        """Initialise device-resident transmission lines and their kernels."""

        arrays = htod_transmission_line_arrays(self.grid.transmissionlines, self.grid)
        for name, array in arrays.items():
            setattr(self, f"tl_{name}_dev", array)

        real = config.sim_config.dtypes["float_or_double"]
        line = self.grid.transmissionlines[0]
        self.tl_line_coefficient = real(config.c * self.grid.dt / line.dl)
        self.tl_abc_coefficient = real(
            (config.c * self.grid.dt - line.dl) / (config.c * self.grid.dt + line.dl)
        )
        self.tl_tpb = (32, 1, 1)
        self.tl_bpg = (
            int(np.ceil(len(self.grid.transmissionlines) / self.tl_tpb[0])),
            1,
            1,
        )

        substitutions = dict(self.subs_func)
        substitutions.update(
            {
                "NY_TLINFO": 10,
                "NY_TLWAVES": self.grid.iterations + 1,
                "NY_TLOUTPUTS": self.grid.iterations + 1,
            }
        )

        source = self._build_knl(
            knl_transmission_line.update_transmission_line_magnetic,
            self.subs_name_args,
            substitutions,
        )
        module = self.source_module(source, options=config.sim_config.devices["nvcc_opts"])
        self.update_transmission_line_magnetic_dev = module.get_function(
            "update_transmission_line_magnetic"
        )

        source = self._build_knl(
            knl_transmission_line.update_transmission_line_electric,
            self.subs_name_args,
            substitutions,
        )
        module = self.source_module(source, options=config.sim_config.devices["nvcc_opts"])
        self.update_transmission_line_electric_dev = module.get_function(
            "update_transmission_line_electric"
        )

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
        ) = htod_snapshot_array(self.grid.snapshots)

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

    def _set_planewave_knls(self):
        """Plane wave (TF/SF) - prepares kernels, uploads 1D DPW arrays,
        and gets kernel functions for standard and axial variants.
        Called once at simulation init if discreteplanewaves exist.
        """

        # Additional subs needed for plane wave kernels.
        # These are used in the func body template substitutions
        subs_pw = dict(self.subs_func)

        # Upload updatecoeffsH and updatecoeffsE to GPU as global arrays
        # Used by axial kernels as pointer arguments - no 64KB constant memory limit
        self.grid.htod_mat_coeff_arrays()

        for dpw in self.grid.discreteplanewaves:

            # Upload all 1D DPW arrays to GPU
            self.grid.htod_planewave_arrays(dpw)

            # Standard (homogeneous) kernels 
            # Scalar coefficients passed as kernel args - no constant memory needed

            bld = self._build_knl(
                knl_planewave_updates.update_1d_magnetic, self.subs_name_args, subs_pw
            )
            knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            dpw.update_1d_magnetic_dev = knl.get_function("update_1d_magnetic")

            bld = self._build_knl(
                knl_planewave_updates.update_1d_magnetic_pml, self.subs_name_args, subs_pw
            )
            knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            dpw.update_1d_magnetic_pml_dev = knl.get_function("update_1d_magnetic_pml")

            bld = self._build_knl(
                knl_planewave_updates.update_1d_electric, self.subs_name_args, subs_pw
            )
            knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            dpw.update_1d_electric_dev = knl.get_function("update_1d_electric")

            bld = self._build_knl(
                knl_planewave_updates.update_1d_electric_pml, self.subs_name_args, subs_pw
            )
            knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
            dpw.update_1d_electric_pml_dev = knl.get_function("update_1d_electric_pml")

            # Standard dispersive electric kernels (compiled only when dpw.dispersive)
            if dpw.dispersive:
                bld = self._build_knl(
                    knl_planewave_updates.update_1d_electric_dispersive_A,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_electric_dispersive_A_dev = knl.get_function(
                    "update_1d_electric_dispersive_A"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_electric_dispersive_B,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_electric_dispersive_B_dev = knl.get_function(
                    "update_1d_electric_dispersive_B"
                )

            # Standard face injection kernels - 12 kernels (6 faces x H and E)
            dpw.std_H_face_devs = []
            for knl_dict in knl_tfsf_injection.STANDARD_H_KERNELS:
                bld = self._build_knl(knl_dict, self.subs_name_args, subs_pw)
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                func_name = knl_dict["name"]
                dpw.std_H_face_devs.append(knl.get_function(func_name))

            dpw.std_E_face_devs = []
            for knl_dict in knl_tfsf_injection.STANDARD_E_KERNELS:
                bld = self._build_knl(knl_dict, self.subs_name_args, subs_pw)
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                func_name = knl_dict["name"]
                dpw.std_E_face_devs.append(knl.get_function(func_name))

            # Axial kernels (only if dpw.axial != 0) 
            # updatecoeffsH/E passed as pointer args - no constant memory, no 64KB limit
            if dpw.axial != 0:

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_magnetic_axial_source,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_magnetic_axial_source_dev = knl.get_function(
                    "update_1d_magnetic_axial_source"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_magnetic_axial_source_pml,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_magnetic_axial_source_pml_dev = knl.get_function(
                    "update_1d_magnetic_axial_source_pml"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_magnetic_axial_inject,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_magnetic_axial_inject_dev = knl.get_function(
                    "update_1d_magnetic_axial_inject"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_magnetic_axial_main,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_magnetic_axial_main_dev = knl.get_function(
                    "update_1d_magnetic_axial_main"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_magnetic_axial_main_pml_end,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_magnetic_axial_main_pml_end_dev = knl.get_function(
                    "update_1d_magnetic_axial_main_pml_end"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_magnetic_axial_main_pml_start,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_magnetic_axial_main_pml_start_dev = knl.get_function(
                    "update_1d_magnetic_axial_main_pml_start"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_electric_axial_source,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_electric_axial_source_dev = knl.get_function(
                    "update_1d_electric_axial_source"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_electric_axial_source_pml,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_electric_axial_source_pml_dev = knl.get_function(
                    "update_1d_electric_axial_source_pml"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_electric_axial_inject,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_electric_axial_inject_dev = knl.get_function(
                    "update_1d_electric_axial_inject"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_electric_axial_main,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_electric_axial_main_dev = knl.get_function(
                    "update_1d_electric_axial_main"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_electric_axial_main_pml_end,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_electric_axial_main_pml_end_dev = knl.get_function(
                    "update_1d_electric_axial_main_pml_end"
                )

                bld = self._build_knl(
                    knl_planewave_updates.update_1d_electric_axial_main_pml_start,
                    self.subs_name_args, subs_pw
                )
                knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                dpw.update_1d_electric_axial_main_pml_start_dev = knl.get_function(
                    "update_1d_electric_axial_main_pml_start"
                )

                # Dispersive axial kernels (compiled only when dpw.dispersive)
                if dpw.dispersive:
                    bld = self._build_knl(
                        knl_planewave_updates.update_1d_electric_dispersive_axial_source,
                        self.subs_name_args, subs_pw
                    )
                    knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                    dpw.update_1d_electric_dispersive_axial_source_dev = knl.get_function(
                        "update_1d_electric_dispersive_axial_source"
                    )

                    bld = self._build_knl(
                        knl_planewave_updates.update_1d_electric_dispersive_axial_source_T,
                        self.subs_name_args, subs_pw
                    )
                    knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                    dpw.update_1d_electric_dispersive_axial_source_T_dev = knl.get_function(
                        "update_1d_electric_dispersive_axial_source_T"
                    )

                    bld = self._build_knl(
                        knl_planewave_updates.update_1d_electric_dispersive_axial_main,
                        self.subs_name_args, subs_pw
                    )
                    knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                    dpw.update_1d_electric_dispersive_axial_main_dev = knl.get_function(
                        "update_1d_electric_dispersive_axial_main"
                    )

                    bld = self._build_knl(
                        knl_planewave_updates.update_1d_electric_dispersive_axial_main_T,
                        self.subs_name_args, subs_pw
                    )
                    knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                    dpw.update_1d_electric_dispersive_axial_main_T_dev = knl.get_function(
                        "update_1d_electric_dispersive_axial_main_T"
                    )

                # Axial face injection kernels - 12 kernels (6 faces x H and E)
                # note- these kernels read updatecoeffsH/E from constant memory
                # (declared by knl_common), so _copy_mat_coeffs must populate it
                # on each compiled module - otherwise reads return uninitialised data.
                dpw.axial_H_face_devs = []
                for knl_dict in knl_tfsf_injection.AXIAL_H_KERNELS:
                    bld = self._build_knl(knl_dict, self.subs_name_args, subs_pw)
                    knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                    self._copy_mat_coeffs(knl, knl)
                    func_name = knl_dict["name"]
                    dpw.axial_H_face_devs.append(knl.get_function(func_name))

                dpw.axial_E_face_devs = []
                for knl_dict in knl_tfsf_injection.AXIAL_E_KERNELS:
                    bld = self._build_knl(knl_dict, self.subs_name_args, subs_pw)
                    knl = self.source_module(bld, options=config.sim_config.devices["nvcc_opts"])
                    self._copy_mat_coeffs(knl, knl)
                    func_name = knl_dict["name"]
                    dpw.axial_E_face_devs.append(knl.get_function(func_name))

    def _dpw_skip_axis(self, dpw):
        """Resolves which face pair (if any) to skip for 2D TM/TE runs.

        Mirrors the skip_axis parameter added to applyTFSF* in
        plane_wave.pyx: 0/1/2 -> skip the x/y/z faces (the invariant
        axis has no physical faces to inject through), -1 -> 3D, launch
        all six. Reads dpw.skip_axis if the 2D-aware sources.py provides
        it; otherwise derives it from the grid's mode2d encoding
        (-1 = 3D; 0/1/2 = 2D TM invariant x/y/z; 3/4/5 = 2D TE
        invariant x/y/z, so invariant axis = mode2d % 3); otherwise -1
        (pure 3D behaviour, identical to before this change).
        """
        skip = getattr(dpw, "skip_axis", None)
        if skip is not None:
            return int(skip)
        mode2d = getattr(self.grid, "mode2d", -1)
        if mode2d is None or mode2d < 0:
            return -1
        return mode2d % 3

    def _launch_std_H_face_kernels(self, dpw):
        """Launches 6 standard H face injection kernels for one dpw object.
        Called after 1D magnetic update every timestep.
        """
        corners = dpw.corners
        # corners layout is [x_start, y_start, z_start, x_stop, y_stop, z_stop]
        # (see plane_wave.pyx applyTFSF* and sources.py loops)
        xs, ys, zs = corners[0], corners[1], corners[2]
        xf, yf, zf = corners[3], corners[4], corners[5]
        m = dpw.m
        origin = dpw.origin

        # Face sizes for thread decomposition
        NY_xface = yf - ys + 1
        NZ_xface = zf - zs + 1
        NX_yface = xf - xs + 1
        NZ_yface = zf - zs + 1
        NX_zface = xf - xs + 1
        NY_zface = yf - ys + 1

        REAL = config.sim_config.dtypes["float_or_double"]

        # x_low, x_high, y_low, y_high, z_low, z_high
        face_configs = [
            (NY_xface * NZ_xface, NY_xface, NZ_xface, xs, xs, ys, yf, zs, zf),
            (NY_xface * NZ_xface, NY_xface, NZ_xface, xf, xf, ys, yf, zs, zf),
            (NX_yface * NZ_yface, NX_yface, NZ_yface, xs, xf, ys, ys, zs, zf),
            (NX_yface * NZ_yface, NX_yface, NZ_yface, xs, xf, yf, yf, zs, zf),
            (NX_zface * NY_zface, NX_zface, NY_zface, xs, xf, ys, yf, zs, zs),
            (NX_zface * NY_zface, NX_zface, NY_zface, xs, xf, ys, yf, zf, zf),
        ]

        mat = self.grid.updatecoeffsH[dpw.material.numID]
        skip_axis = self._dpw_skip_axis(dpw)

        for idx, (face_dev, fc) in enumerate(zip(dpw.std_H_face_devs, face_configs)):
            # 2D TM/TE: no injection through the faces perpendicular to
            # the invariant axis (mirrors skip_axis in applyTFSFMagnetic)
            if skip_axis == idx // 2:
                continue
            face_size, NY_F, NZ_F, x_s, x_e, y_s, y_e, z_s, z_e = fc
            if face_size == 0:
                continue

            # coef_H_1 and coef_H_2 depend on face axis
            if idx < 2:   # x faces — use DBx (col 1)
                c1 = REAL(mat[1]); c2 = REAL(mat[1])
            elif idx < 4: # y faces — use DBy (col 2)
                c1 = REAL(mat[2]); c2 = REAL(mat[2])
            else:         # z faces — use DBz (col 3)
                c1 = REAL(mat[3]); c2 = REAL(mat[3])

            face_dev(
                np.int32(self.grid.ny + 1),
                np.int32(self.grid.nz + 1),
                np.int32(NY_F),
                np.int32(NZ_F),
                np.int32(x_s), np.int32(x_e),
                np.int32(y_s), np.int32(y_e),
                np.int32(z_s), np.int32(z_e),
                np.int32(m[0]), np.int32(m[1]), np.int32(m[2]),
                np.int32(origin[0]), np.int32(origin[1]), np.int32(origin[2]),
                c1, c2,
                self.grid.Hx_dev,
                self.grid.Hy_dev,
                self.grid.Hz_dev,
                dpw.E_fields_dev[0],   # E_x row
                dpw.E_fields_dev[1],   # E_y row
                dpw.E_fields_dev[2],   # E_z row
                block=(256, 1, 1),
                grid=(int(np.ceil(face_size / 256)), 1, 1),
            )

    def _launch_std_E_face_kernels(self, dpw):
        """Launches 6 standard E face injection kernels for one dpw object.
        Called after 1D electric update every timestep.
        """
        corners = dpw.corners
        # corners layout is [x_start, y_start, z_start, x_stop, y_stop, z_stop]
        # (see plane_wave.pyx applyTFSF* and sources.py loops)
        xs, ys, zs = corners[0], corners[1], corners[2]
        xf, yf, zf = corners[3], corners[4], corners[5]
        m = dpw.m
        origin = dpw.origin

        NY_xface = yf - ys + 1
        NZ_xface = zf - zs + 1
        NX_yface = xf - xs + 1
        NZ_yface = zf - zs + 1
        NX_zface = xf - xs + 1
        NY_zface = yf - ys + 1

        REAL = config.sim_config.dtypes["float_or_double"]

        face_configs = [
            (NY_xface * NZ_xface, NY_xface, NZ_xface, xs, xs, ys, yf, zs, zf),
            (NY_xface * NZ_xface, NY_xface, NZ_xface, xf, xf, ys, yf, zs, zf),
            (NX_yface * NZ_yface, NX_yface, NZ_yface, xs, xf, ys, ys, zs, zf),
            (NX_yface * NZ_yface, NX_yface, NZ_yface, xs, xf, yf, yf, zs, zf),
            (NX_zface * NY_zface, NX_zface, NY_zface, xs, xf, ys, yf, zs, zs),
            (NX_zface * NY_zface, NX_zface, NY_zface, xs, xf, ys, yf, zf, zf),
        ]

        mat = self.grid.updatecoeffsE[dpw.material.numID]
        skip_axis = self._dpw_skip_axis(dpw)

        for idx, (face_dev, fc) in enumerate(zip(dpw.std_E_face_devs, face_configs)):
            # 2D TM/TE: skip the invariant-axis faces (applyTFSFElectric)
            if skip_axis == idx // 2:
                continue
            face_size, NY_F, NZ_F, x_s, x_e, y_s, y_e, z_s, z_e = fc
            if face_size == 0:
                continue

            if idx < 2:   # x faces — use CBx (col 1)
                c1 = REAL(mat[1]); c2 = REAL(mat[1])
            elif idx < 4: # y faces — use CBy (col 2)
                c1 = REAL(mat[2]); c2 = REAL(mat[2])
            else:         # z faces — use CBz (col 3)
                c1 = REAL(mat[3]); c2 = REAL(mat[3])

            face_dev(
                np.int32(self.grid.ny + 1),
                np.int32(self.grid.nz + 1),
                np.int32(NY_F),
                np.int32(NZ_F),
                np.int32(x_s), np.int32(x_e),
                np.int32(y_s), np.int32(y_e),
                np.int32(z_s), np.int32(z_e),
                np.int32(m[0]), np.int32(m[1]), np.int32(m[2]),
                np.int32(origin[0]), np.int32(origin[1]), np.int32(origin[2]),
                c1, c2,
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                dpw.H_fields_dev[0],   # H_x row
                dpw.H_fields_dev[1],   # H_y row
                dpw.H_fields_dev[2],   # H_z row
                block=(256, 1, 1),
                grid=(int(np.ceil(face_size / 256)), 1, 1),
            )

    def _launch_axial_H_face_kernels(self, dpw):
        """Launches 6 axial H face injection kernels for one dpw object.
        updatecoeffsH passed as pointer arg — no constant memory limit.
        """
        corners = dpw.corners
        # corners layout is [x_start, y_start, z_start, x_stop, y_stop, z_stop]
        # (see plane_wave.pyx applyTFSF* and sources.py loops)
        xs, ys, zs = corners[0], corners[1], corners[2]
        xf, yf, zf = corners[3], corners[4], corners[5]
        m = dpw.m
        origin = dpw.origin

        NY_xface = yf - ys + 1
        NZ_xface = zf - zs + 1
        NX_yface = xf - xs + 1
        NZ_yface = zf - zs + 1
        NX_zface = xf - xs + 1
        NY_zface = yf - ys + 1

        face_configs = [
            (NY_xface * NZ_xface, NY_xface, NZ_xface, xs, xs, ys, yf, zs, zf),
            (NY_xface * NZ_xface, NY_xface, NZ_xface, xf, xf, ys, yf, zs, zf),
            (NX_yface * NZ_yface, NX_yface, NZ_yface, xs, xf, ys, ys, zs, zf),
            (NX_yface * NZ_yface, NX_yface, NZ_yface, xs, xf, yf, yf, zs, zf),
            (NX_zface * NY_zface, NX_zface, NY_zface, xs, xf, ys, yf, zs, zs),
            (NX_zface * NY_zface, NX_zface, NY_zface, xs, xf, ys, yf, zf, zf),
        ]

        skip_axis = self._dpw_skip_axis(dpw)

        for idx, (face_dev, fc) in enumerate(zip(dpw.axial_H_face_devs, face_configs)):
            # 2D TM/TE: skip the invariant-axis faces
            # (mirrors skip_axis in applyTFSFMagnetic_axial)
            if skip_axis == idx // 2:
                continue
            face_size, NY_F, NZ_F, x_s, x_e, y_s, y_e, z_s, z_e = fc
            if face_size == 0:
                continue

            face_dev(
                np.int32(self.grid.ny + 1),
                np.int32(self.grid.nz + 1),
                np.int32(NY_F),
                np.int32(NZ_F),
                np.int32(x_s), np.int32(x_e),
                np.int32(y_s), np.int32(y_e),
                np.int32(z_s), np.int32(z_e),
                np.int32(m[0]), np.int32(m[1]), np.int32(m[2]),
                np.int32(origin[0]), np.int32(origin[1]), np.int32(origin[2]),
                np.int32(dpw.origin_axial),
                self.grid.Hx_dev,
                self.grid.Hy_dev,
                self.grid.Hz_dev,
                dpw.E_fields_dev[0],   # E_x row (main 1D grid, matches Cython applyTFSFMagnetic_axial)
                dpw.E_fields_dev[1],   # E_y row
                dpw.E_fields_dev[2],   # E_z row
                self.grid.ID_dev,
                # updatecoeffsH is read from constant memory (populated by _copy_mat_coeffs)
                block=(256, 1, 1),
                grid=(int(np.ceil(face_size / 256)), 1, 1),
            )

    def _launch_axial_E_face_kernels(self, dpw):
        """Launches 6 axial E face injection kernels for one dpw object.
        updatecoeffsE passed as pointer arg — no constant memory limit.
        """
        corners = dpw.corners
        # corners layout is [x_start, y_start, z_start, x_stop, y_stop, z_stop]
        # (see plane_wave.pyx applyTFSF* and sources.py loops)
        xs, ys, zs = corners[0], corners[1], corners[2]
        xf, yf, zf = corners[3], corners[4], corners[5]
        m = dpw.m
        origin = dpw.origin

        NY_xface = yf - ys + 1
        NZ_xface = zf - zs + 1
        NX_yface = xf - xs + 1
        NZ_yface = zf - zs + 1
        NX_zface = xf - xs + 1
        NY_zface = yf - ys + 1

        face_configs = [
            (NY_xface * NZ_xface, NY_xface, NZ_xface, xs, xs, ys, yf, zs, zf),
            (NY_xface * NZ_xface, NY_xface, NZ_xface, xf, xf, ys, yf, zs, zf),
            (NX_yface * NZ_yface, NX_yface, NZ_yface, xs, xf, ys, ys, zs, zf),
            (NX_yface * NZ_yface, NX_yface, NZ_yface, xs, xf, yf, yf, zs, zf),
            (NX_zface * NY_zface, NX_zface, NY_zface, xs, xf, ys, yf, zs, zs),
            (NX_zface * NY_zface, NX_zface, NY_zface, xs, xf, ys, yf, zf, zf),
        ]

        skip_axis = self._dpw_skip_axis(dpw)

        for idx, (face_dev, fc) in enumerate(zip(dpw.axial_E_face_devs, face_configs)):
            # 2D TM/TE: skip the invariant-axis faces
            # (mirrors skip_axis in applyTFSFElectric_axial)
            if skip_axis == idx // 2:
                continue
            face_size, NY_F, NZ_F, x_s, x_e, y_s, y_e, z_s, z_e = fc
            if face_size == 0:
                continue

            face_dev(
                np.int32(self.grid.ny + 1),
                np.int32(self.grid.nz + 1),
                np.int32(NY_F),
                np.int32(NZ_F),
                np.int32(x_s), np.int32(x_e),
                np.int32(y_s), np.int32(y_e),
                np.int32(z_s), np.int32(z_e),
                np.int32(m[0]), np.int32(m[1]), np.int32(m[2]),
                np.int32(origin[0]), np.int32(origin[1]), np.int32(origin[2]),
                np.int32(dpw.origin_axial),
                self.grid.Ex_dev,
                self.grid.Ey_dev,
                self.grid.Ez_dev,
                dpw.H_fields_dev[0],   # H_x row (main 1D grid, matches Cython applyTFSFElectric_axial)
                dpw.H_fields_dev[1],   # H_y row
                dpw.H_fields_dev[2],   # H_z row
                self.grid.ID_dev,
                # updatecoeffsE is read from constant memory (populated by _copy_mat_coeffs)
                block=(256, 1, 1),
                grid=(int(np.ceil(face_size / 256)), 1, 1),
            )

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

    def store_outputs(self, iteration):
        """Stores field component values for every receiver."""
        if self.grid.rxs:
            self.store_outputs_dev(
                np.int32(len(self.grid.rxs)),
                np.int32(iteration),
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

        sx, sy, sz = _snapshot_axis_strides()
        for i, snap in enumerate(self.grid.snapshots):
            if snap.time == iteration + 1:
                snapno = 0 if config.get_model_config().device["snapsgpu2cpu"] else i
                self.store_snapshot_dev(
                    np.int32(snapno),
                    np.int32(snap.xs),
                    np.int32(snap.ys),
                    np.int32(snap.zs),
                    np.int32(snap.nx),
                    np.int32(snap.ny),
                    np.int32(snap.nz),
                    np.int32(snap.dx),
                    np.int32(snap.dy),
                    np.int32(snap.dz),
                    np.int32(sx),
                    np.int32(sy),
                    np.int32(sz),
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

    def observe_ntff_electric(self, iteration):
        """Collect electric frequency- and time-domain KSIR data on CUDA."""

        collector = getattr(self, "ntff_collector", None)
        if collector is not None:
            collector.observe_electric(iteration)

    def observe_ntff_magnetic(self, iteration):
        """Collect magnetic frequency- and time-domain KSIR data on CUDA."""

        collector = getattr(self, "ntff_collector", None)
        if collector is not None:
            collector.observe_magnetic(iteration)

    def update_magnetic_pml(self):
        """Updates magnetic field components with the PML correction."""
        for pml in self.grid.pmls["slabs"]:
            pml.update_magnetic()

    def update_magnetic_sources(self, iteration):
        """Updates magnetic field components from sources."""
        if self.grid.transmissionlines:
            self.update_transmission_line_magnetic_dev(
                np.int32(len(self.grid.transmissionlines)),
                np.int32(iteration),
                config.sim_config.dtypes["float_or_double"](self.grid.dx),
                config.sim_config.dtypes["float_or_double"](self.grid.dy),
                config.sim_config.dtypes["float_or_double"](self.grid.dz),
                self.tl_line_coefficient,
                self.tl_info_dev.gpudata,
                self.tl_resistance_dev.gpudata,
                self.tl_waveform_half_dev.gpudata,
                self.tl_voltage_dev.gpudata,
                self.tl_current_dev.gpudata,
                self.tl_Vtotal_dev.gpudata,
                self.tl_Itotal_dev.gpudata,
                self.grid.Hx_dev.gpudata,
                self.grid.Hy_dev.gpudata,
                self.grid.Hz_dev.gpudata,
                block=self.tl_tpb,
                grid=self.tl_bpg,
            )

        if self.grid.magneticdipoles:
            self.update_magnetic_dipole_dev(
                np.int32(len(self.grid.magneticdipoles)),
                np.int32(iteration),
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

    def update_symmetry_boundaries_electric(self):
        """Apply the nondispersive PMC ghost-image correction on CUDA."""
        if "pmc" not in self.grid.symmetry_boundaries.values():
            return

        self.update_electric_pmc_dev(
            np.int32(self.grid.nx),
            np.int32(self.grid.ny),
            np.int32(self.grid.nz),
            *self._pmc_flags(),
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

    def update_electric_sources(self, iteration):
        """Updates electric field components from sources -
        update any Hertzian dipole sources last.
        """
        if self.grid.voltagesources:
            self.update_voltage_source_dev(
                np.int32(len(self.grid.voltagesources)),
                np.int32(iteration),
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

        if self.grid.transmissionlines:
            self.update_transmission_line_electric_dev(
                np.int32(len(self.grid.transmissionlines)),
                np.int32(iteration),
                config.sim_config.dtypes["float_or_double"](self.grid.dx),
                config.sim_config.dtypes["float_or_double"](self.grid.dy),
                config.sim_config.dtypes["float_or_double"](self.grid.dz),
                self.tl_line_coefficient,
                self.tl_abc_coefficient,
                self.tl_info_dev.gpudata,
                self.tl_resistance_dev.gpudata,
                self.tl_waveform_whole_dev.gpudata,
                self.tl_voltage_dev.gpudata,
                self.tl_current_dev.gpudata,
                self.tl_abcv0_dev.gpudata,
                self.tl_abcv1_dev.gpudata,
                self.grid.Ex_dev.gpudata,
                self.grid.Ey_dev.gpudata,
                self.grid.Ez_dev.gpudata,
                block=self.tl_tpb,
                grid=self.tl_bpg,
            )

        if self.grid.hertziandipoles:
            self.update_hertzian_dipole_dev(
                np.int32(len(self.grid.hertziandipoles)),
                np.int32(iteration),
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


    def update_plane_waves_magnetic(self, iteration):
        """Updates 1D DPW auxiliary grid H fields and applies TF/SF
        corrections to 3D H field arrays at all 6 faces.
        Called every timestep after update_magnetic().
        """
        if not self.grid.discreteplanewaves:
            return

        REAL = config.sim_config.dtypes["float_or_double"]

        for dpw in self.grid.discreteplanewaves:
            n  = np.int32(dpw.length)
            p  = np.int32(dpw.pml_length)
            M  = np.int32(dpw.m[3])
            mx = np.int32(dpw.m[0])
            my = np.int32(dpw.m[1])
            mz = np.int32(dpw.m[2])
            dx = REAL(self.grid.dx)
            dy = REAL(self.grid.dy)
            dz = REAL(self.grid.dz)

            bulk_size  = int(np.ceil((dpw.length - 2 * dpw.m[3]) / 256))
            pml_size   = int(np.ceil(dpw.pml_length / 256))

            #  Source injection (initialize 1D grid source region) 
            # Ports initializeMagneticFields() from plane_wave.pyx
            # Sets H_fields[comp, r] = projections[3+comp] * waveformvalues_halfdt[iteration, comp, r]
            # for r in [0, M) - the source region at start of 1D grid
            M_int = int(dpw.m[3])
            if M_int > 0:
                wave_H = dpw.waveformvalues_halfdt[iteration]   # shape [3, M]
                # Axial variant injects the source into the auxiliary source
                # grid (H_fields_s), standard into the main 1D grid (H_fields)
                # - matches initializeMagneticFields() call sites in plane_wave.pyx
                target = dpw.H_fields_dev if dpw.axial == 0 else dpw.H_fields_s_dev
                for comp in range(3):
                    proj = dpw.projections[3 + comp]
                    src_vals = (proj * wave_H[comp]).astype(
                        config.sim_config.dtypes["float_or_double"]
                    )
                    # Write source values to first M positions of row `comp`.
                    # Must compute the row offset manually — int(arr[comp].gpudata)
                    # returns the base allocation address regardless of comp.
                    row_ptr = int(target.gpudata) + comp * target.strides[0]
                    self.drv.memcpy_htod(row_ptr, src_vals)

            if dpw.axial == 0:
                #  Standard (homogeneous) magnetic 1D update 
                # Get background material coefficients
                mat = self.grid.updatecoeffsH[dpw.material.numID]
                DA  = REAL(mat[0])
                DBx = REAL(mat[1])
                DBy = REAL(mat[2])
                DBz = REAL(mat[3])
                srcm = REAL(mat[4])

                # Bulk update
                # Row offsets for [3, N] arrays: row_i starts at i * N * itemsize
                Hx_ptr = dpw.H_fields_dev[0]
                Hy_ptr = dpw.H_fields_dev[1]
                Hz_ptr = dpw.H_fields_dev[2]
                Ex_ptr = dpw.E_fields_dev[0]
                Ey_ptr = dpw.E_fields_dev[1]
                Ez_ptr = dpw.E_fields_dev[2]

                # Coefficient order MUST match kernel signature:
                # (xt, xy, xz), (yt, yx, yz), (zt, zx, zy)
                # where xy=DBy(col2), xz=DBz(col3), yx=DBx(col1), yz=DBz(col3),
                #       zx=DBx(col1), zy=DBy(col2)  per plane_wave.pyx
                dpw.update_1d_magnetic_dev(
                    n, M, mx, my, mz,
                    DA,  DBy, DBz,   # coef_H_xt, coef_H_xy, coef_H_xz
                    DA,  DBx, DBz,   # coef_H_yt, coef_H_yx, coef_H_yz
                    DA,  DBx, DBy,   # coef_H_zt, coef_H_zx, coef_H_zy
                    Hx_ptr, Hy_ptr, Hz_ptr,
                    Ex_ptr, Ey_ptr, Ez_ptr,
                    block=(256, 1, 1),
                    grid=(bulk_size, 1, 1),
                )

                # PML update
                # Kernel expects 6 integral-row pointers + 12 coeff-row pointers.
                # Magnetic PML uses integral rows 2,3 of Ix/Iy/Iz (per plane_wave.pyx)-
                #   Ixmyz=Ix[2], Ixmzy=Ix[3], Iymxz=Iy[2], Iymzx=Iy[3],
                #   Izmxy=Iz[2], Izmyx=Iz[3]
                # Kernel param order is: Ixmzy, Ixmyz, Iymzx, Iymxz, Izmyx, Izmxy
                # Coeff rows: RA=row0, RB=row1, RC=row2, RD=row3 of rcHx/rcHy/rcHz
                if pml_size > 0:
                    dpw.update_1d_magnetic_pml_dev(
                        n, p, M, mx, my, mz, srcm, dx, dy, dz,
                        Hx_ptr, Hy_ptr, Hz_ptr,
                        Ex_ptr, Ey_ptr, Ez_ptr,
                        dpw.Ix_dev[3],   # Ixmzy
                        dpw.Ix_dev[2],   # Ixmyz
                        dpw.Iy_dev[3],   # Iymzx
                        dpw.Iy_dev[2],   # Iymxz
                        dpw.Iz_dev[3],   # Izmyx
                        dpw.Iz_dev[2],   # Izmxy
                        dpw.pml_rhx_dev[0],  # RAHx
                        dpw.pml_rhx_dev[1],  # RBHx
                        dpw.pml_rhx_dev[2],  # RCHx
                        dpw.pml_rhx_dev[3],  # RDHx
                        dpw.pml_rhy_dev[0],  # RAHy
                        dpw.pml_rhy_dev[1],  # RBHy
                        dpw.pml_rhy_dev[2],  # RCHy
                        dpw.pml_rhy_dev[3],  # RDHy
                        dpw.pml_rhz_dev[0],  # RAHz
                        dpw.pml_rhz_dev[1],  # RBHz
                        dpw.pml_rhz_dev[2],  # RCHz
                        dpw.pml_rhz_dev[3],  # RDHz
                        block=(256, 1, 1),
                        grid=(pml_size, 1, 1),
                    )

                # Standard face corrections
                self._launch_std_H_face_kernels(dpw)

                if iteration == 500:
                    hf = dpw.H_fields_dev.get()
                    print("GPU 1D H_fields max per row:", abs(hf).max(axis=1))
                    ef = dpw.E_fields_dev.get()
                    print("GPU 1D E_fields max per row:", abs(ef).max(axis=1))
                    print("Projections:", dpw.projections)
                    print("waveformvalues_halfdt[500] max:", abs(dpw.waveformvalues_halfdt[500]).max())

            else:
                # Axial magnetic 1D update (3 sequential launches)

                # Row offsets for axial source grid arrays
                Hx_s = dpw.H_fields_s_dev[0]
                Hy_s = dpw.H_fields_s_dev[1]
                Hz_s = dpw.H_fields_s_dev[2]
                Ex_s = dpw.E_fields_s_dev[0]
                Ey_s = dpw.E_fields_s_dev[1]
                Ez_s = dpw.E_fields_s_dev[2]
                Hx_m = dpw.H_fields_dev[0]
                Hy_m = dpw.H_fields_dev[1]
                Hz_m = dpw.H_fields_dev[2]
                Ex_m = dpw.E_fields_dev[0]
                Ey_m = dpw.E_fields_dev[1]
                Ez_m = dpw.E_fields_dev[2]

                # Launch 1: Update source grid bulk
                if bulk_size > 0:
                    dpw.update_1d_magnetic_axial_source_dev(
                        n, M, mx, my, mz,
                        Hx_s, Hy_s, Hz_s,
                        Ex_s, Ey_s, Ez_s,
                        dpw.ID_dev,
                        self.grid.updatecoeffsH_dev,
                        self.grid.updatecoeffsE_dev,
                        block=(256, 1, 1),
                        grid=(bulk_size, 1, 1),
                    )

                # Source grid PML
                # Kernel expects 6 integral-row pointers + 12 coeff-row pointers.
                # Source PML integrals occupy rows 2,3 of Ix_s/Iy_s/Iz_s
                # (Ixmyz_s=Ix_s[2], Ixmzy_s=Ix_s[3], etc., per plane_wave.pyx).
                # Kernel param order: Ixmzy_s, Ixmyz_s, Iymzx_s, Iymxz_s, Izmyx_s, Izmxy_s
                # Coeff rows: RA=0, RB=1, RC=2, RD=3 of pml_rhx0/pml_rhy0/pml_rhz0
                if pml_size > 0:
                    dpw.update_1d_magnetic_axial_source_pml_dev(
                        n, p, M, mx, my, mz, dx, dy, dz,
                        Hx_s, Hy_s, Hz_s,
                        Ex_s, Ey_s, Ez_s,
                        dpw.Ix_s_dev[3],   # Ixmzy_s
                        dpw.Ix_s_dev[2],   # Ixmyz_s
                        dpw.Iy_s_dev[3],   # Iymzx_s
                        dpw.Iy_s_dev[2],   # Iymxz_s
                        dpw.Iz_s_dev[3],   # Izmyx_s
                        dpw.Iz_s_dev[2],   # Izmxy_s
                        dpw.pml_rhx0_dev[0],  # RAHx0
                        dpw.pml_rhx0_dev[1],  # RBHx0
                        dpw.pml_rhx0_dev[2],  # RCHx0
                        dpw.pml_rhx0_dev[3],  # RDHx0
                        dpw.pml_rhy0_dev[0],  # RAHy0
                        dpw.pml_rhy0_dev[1],  # RBHy0
                        dpw.pml_rhy0_dev[2],  # RCHy0
                        dpw.pml_rhy0_dev[3],  # RDHy0
                        dpw.pml_rhz0_dev[0],  # RAHz0
                        dpw.pml_rhz0_dev[1],  # RBHz0
                        dpw.pml_rhz0_dev[2],  # RCHz0
                        dpw.pml_rhz0_dev[3],  # RDHz0
                        dpw.ID_dev,
                        self.grid.updatecoeffsH_dev,
                        self.grid.updatecoeffsE_dev,
                        block=(256, 1, 1),
                        grid=(pml_size, 1, 1),
                    )

                # Launch 2-Inject source into main grid at src-2
                dpw.update_1d_magnetic_axial_inject_dev(
                    n, np.int32(dpw.origin_axial), mx, my, mz,
                    Hx_m, Hy_m, Hz_m,
                    Ex_s, Ey_s, Ez_s,
                    dpw.ID_dev,
                    self.grid.updatecoeffsH_dev,
                    self.grid.updatecoeffsE_dev,
                    block=(1, 1, 1),
                    grid=(1, 1, 1),
                )

                # Launch 3- Update main grid bulk
                # Range [M-1, N-M) gives size N - 2M + 1 (includes origin point)
                main_bulk_threads = dpw.length - 2 * dpw.m[3] + 1
                main_bulk_size = int(np.ceil(main_bulk_threads / 256)) if main_bulk_threads > 0 else 0
                if main_bulk_size > 0:
                    dpw.update_1d_magnetic_axial_main_dev(
                        n, M, mx, my, mz,
                        Hx_m, Hy_m, Hz_m,
                        Ex_m, Ey_m, Ez_m,
                        dpw.ID_dev,
                        self.grid.updatecoeffsH_dev,
                        self.grid.updatecoeffsE_dev,
                        block=(256, 1, 1),
                        grid=(main_bulk_size, 1, 1),
                    )

                # Main grid PML end region
                # Main PML integrals at rows 2,3 of Ix/Iy/Iz; coeffs at rows 0-3 of pml_rh*
                if pml_size > 0:
                    dpw.update_1d_magnetic_axial_main_pml_end_dev(
                        n, p, M, mx, my, mz, dx, dy, dz,
                        Hx_m, Hy_m, Hz_m,
                        Ex_m, Ey_m, Ez_m,
                        dpw.Ix_dev[3],   # Ixmzy
                        dpw.Ix_dev[2],   # Ixmyz
                        dpw.Iy_dev[3],   # Iymzx
                        dpw.Iy_dev[2],   # Iymxz
                        dpw.Iz_dev[3],   # Izmyx
                        dpw.Iz_dev[2],   # Izmxy
                        dpw.pml_rhx_dev[0], dpw.pml_rhx_dev[1],
                        dpw.pml_rhx_dev[2], dpw.pml_rhx_dev[3],
                        dpw.pml_rhy_dev[0], dpw.pml_rhy_dev[1],
                        dpw.pml_rhy_dev[2], dpw.pml_rhy_dev[3],
                        dpw.pml_rhz_dev[0], dpw.pml_rhz_dev[1],
                        dpw.pml_rhz_dev[2], dpw.pml_rhz_dev[3],
                        dpw.ID_dev,
                        self.grid.updatecoeffsH_dev,
                        self.grid.updatecoeffsE_dev,
                        block=(256, 1, 1),
                        grid=(pml_size, 1, 1),
                    )

                # Main grid PML start region
                # Uses Ix0/Iy0/Iz0 (separate from main Ix/Iy/Iz) and pml_rh*0 (source coeffs reused)
                if pml_size > 0:
                    dpw.update_1d_magnetic_axial_main_pml_start_dev(
                        n, p, mx, my, mz, dx, dy, dz,
                        Hx_m, Hy_m, Hz_m,
                        Ex_m, Ey_m, Ez_m,
                        dpw.Ix0_dev[3],   # Ixmzy0
                        dpw.Ix0_dev[2],   # Ixmyz0
                        dpw.Iy0_dev[3],   # Iymzx0
                        dpw.Iy0_dev[2],   # Iymxz0
                        dpw.Iz0_dev[3],   # Izmyx0
                        dpw.Iz0_dev[2],   # Izmxy0
                        dpw.pml_rhx0_dev[0], dpw.pml_rhx0_dev[1],
                        dpw.pml_rhx0_dev[2], dpw.pml_rhx0_dev[3],
                        dpw.pml_rhy0_dev[0], dpw.pml_rhy0_dev[1],
                        dpw.pml_rhy0_dev[2], dpw.pml_rhy0_dev[3],
                        dpw.pml_rhz0_dev[0], dpw.pml_rhz0_dev[1],
                        dpw.pml_rhz0_dev[2], dpw.pml_rhz0_dev[3],
                        dpw.ID_dev,
                        self.grid.updatecoeffsH_dev,
                        self.grid.updatecoeffsE_dev,
                        block=(256, 1, 1),
                        grid=(pml_size, 1, 1),
                    )

                # Axial face corrections
                self._launch_axial_H_face_kernels(dpw)

                if iteration == 500:
                    hf = dpw.H_fields_s_dev.get()
                    print("GPU 1D H_fields_s max per row:", abs(hf).max(axis=1))
                    ef = dpw.E_fields_s_dev.get()
                    print("GPU 1D E_fields_s max per row:", abs(ef).max(axis=1))
                    print("Projections:", dpw.projections)
                    print("waveformvalues_halfdt[500] max:", abs(dpw.waveformvalues_halfdt[500]).max())

    def update_plane_waves_electric(self, iteration):
        """Updates 1D DPW auxiliary grid E fields and applies TF/SF
        corrections to 3D E field arrays at all 6 faces.
        Called every timestep after update_electric_a().
        """
        if not self.grid.discreteplanewaves:
            return

        REAL = config.sim_config.dtypes["float_or_double"]

        for dpw in self.grid.discreteplanewaves:
            n  = np.int32(dpw.length)
            p  = np.int32(dpw.pml_length)
            M  = np.int32(dpw.m[3])
            mx = np.int32(dpw.m[0])
            my = np.int32(dpw.m[1])
            mz = np.int32(dpw.m[2])
            dx = REAL(self.grid.dx)
            dy = REAL(self.grid.dy)
            dz = REAL(self.grid.dz)

            bulk_size  = int(np.ceil((dpw.length - 2 * dpw.m[3]) / 256))
            pml_size   = int(np.ceil(dpw.pml_length / 256))

            #  Source injection for electric fields
            # Ports initializeElectricFields() from plane_wave.pyx
            # Sets E_fields[comp, r] = projections[comp] * waveformvalues_wholedt[iteration+1, comp, r]
            M_int = int(dpw.m[3])
            if M_int > 0:
                wave_E = dpw.waveformvalues_wholedt[iteration + 1]   # shape [3, M]
                # Axial variant injects into the source grid (E_fields_s),
                # standard into the main 1D grid (E_fields)
                target = dpw.E_fields_dev if dpw.axial == 0 else dpw.E_fields_s_dev
                for comp in range(3):
                    proj = dpw.projections[comp]
                    src_vals = (proj * wave_E[comp]).astype(
                        config.sim_config.dtypes["float_or_double"]
                    )
                    row_ptr = int(target.gpudata) + comp * target.strides[0]
                    self.drv.memcpy_htod(row_ptr, src_vals)

            if dpw.axial == 0:
                #  Standard (homogeneous) electric 1D update 
                mat = self.grid.updatecoeffsE[dpw.material.numID]
                CA  = REAL(mat[0])
                CBx = REAL(mat[1])
                CBy = REAL(mat[2])
                CBz = REAL(mat[3])
                srce = REAL(mat[4])

                Ex_ptr = dpw.E_fields_dev[0]
                Ey_ptr = dpw.E_fields_dev[1]
                Ez_ptr = dpw.E_fields_dev[2]
                Hx_ptr = dpw.H_fields_dev[0]
                Hy_ptr = dpw.H_fields_dev[1]
                Hz_ptr = dpw.H_fields_dev[2]

                # Non-dispersive update runs ONLY when the source material is
                # not dispersive - mirrors the if/else dispatch in
                # cpu_updates.update_plane_waves_electric. Without this guard
                # both the standard AND dispersive kernels would update E every
                # timestep (double update -> compounding amplitude error).
                if not dpw.dispersive:
                    dpw.update_1d_electric_dev(
                        n, M, mx, my, mz,
                        CA,  CBy, CBz,   # coef_E_xt, coef_E_xy, coef_E_xz
                        CA,  CBx, CBz,   # coef_E_yt, coef_E_yx, coef_E_yz
                        CA,  CBx, CBy,   # coef_E_zt, coef_E_zx, coef_E_zy
                        Ex_ptr, Ey_ptr, Ez_ptr,
                        Hx_ptr, Hy_ptr, Hz_ptr,
                        block=(256, 1, 1),
                        grid=(bulk_size, 1, 1),
                    )

                # Electric PML uses integral rows 0,1 of Ix/Iy/Iz (per plane_wave.pyx)-
                #   Ixjyz=Ix[0], Ixjzy=Ix[1], Iyjxz=Iy[0], Iyjzx=Iy[1],
                #   Izjxy=Iz[0], Izjyx=Iz[1]
                # Kernel param order: Jxmzy, Jxmyz, Jymzx, Jymxz, Jzmyx, Jzmxy
                #   (Jxmzy pairs with dHzy -> Ix[1]; Jxmyz pairs with dHyz -> Ix[0])
                # Guarded like the bulk update above - the dispersive branch
                # launches its own PML pass.
                if pml_size > 0 and not dpw.dispersive:
                    dpw.update_1d_electric_pml_dev(
                        n, p, M, mx, my, mz, srce, dx, dy, dz,
                        Ex_ptr, Ey_ptr, Ez_ptr,
                        Hx_ptr, Hy_ptr, Hz_ptr,
                        dpw.Ix_dev[1],   # Jxmzy
                        dpw.Ix_dev[0],   # Jxmyz
                        dpw.Iy_dev[1],   # Jymzx
                        dpw.Iy_dev[0],   # Jymxz
                        dpw.Iz_dev[1],   # Jzmyx
                        dpw.Iz_dev[0],   # Jzmxy
                        dpw.pml_rex_dev[0],  # RAEx
                        dpw.pml_rex_dev[1],  # RBEx
                        dpw.pml_rex_dev[2],  # RCEx
                        dpw.pml_rex_dev[3],  # RDEx
                        dpw.pml_rey_dev[0],  # RAEy
                        dpw.pml_rey_dev[1],  # RBEy
                        dpw.pml_rey_dev[2],  # RCEy
                        dpw.pml_rey_dev[3],  # RDEy
                        dpw.pml_rez_dev[0],  # RAEz
                        dpw.pml_rez_dev[1],  # RBEz
                        dpw.pml_rez_dev[2],  # RCEz
                        dpw.pml_rez_dev[3],  # RDEz
                        block=(256, 1, 1),
                        grid=(pml_size, 1, 1),
                    )

                # Dispersive standard electric update
                # Mirrors updatePlaneWave_electric_dispersive() in plane_wave.pyx.
                # Part A: bulk + polarization update (before PML).
                # PML: reuse existing update_1d_electric_pml_dev (identical math).
                # Part B: final T subtraction (after PML, uses post-PML E values).
                if dpw.dispersive:
                    mat = self.grid.updatecoeffsE[dpw.material.numID]
                    CA   = REAL(mat[0])
                    CBx  = REAL(mat[1])
                    CBy  = REAL(mat[2])
                    CBz  = REAL(mat[3])
                    srce = REAL(mat[4])
                    num_poles = np.int32(config.get_model_config().materials["maxpoles"])

                    Ex_ptr = dpw.E_fields_dev[0]
                    Ey_ptr = dpw.E_fields_dev[1]
                    Ez_ptr = dpw.E_fields_dev[2]
                    Hx_ptr = dpw.H_fields_dev[0]
                    Hy_ptr = dpw.H_fields_dev[1]
                    Hz_ptr = dpw.H_fields_dev[2]

                    # coef_D = dispersive coefficients for the source material
                    mat_id = dpw.material.numID
                    coef_D_ptr = self.grid.updatecoeffsdispersive_dev[mat_id]

                    if bulk_size > 0:
                        dpw.update_1d_electric_dispersive_A_dev(
                            n, M, mx, my, mz, num_poles,
                            CA, CBx, CBy, CBz, srce,
                            coef_D_ptr,
                            dpw.Px_dev, dpw.Py_dev, dpw.Pz_dev,
                            Ex_ptr, Ey_ptr, Ez_ptr,
                            Hx_ptr, Hy_ptr, Hz_ptr,
                            block=(256, 1, 1),
                            grid=(bulk_size, 1, 1),
                        )

                    if pml_size > 0:
                        dpw.update_1d_electric_pml_dev(
                            n, p, M, mx, my, mz, srce, dx, dy, dz,
                            Ex_ptr, Ey_ptr, Ez_ptr,
                            Hx_ptr, Hy_ptr, Hz_ptr,
                            dpw.Ix_dev[1], dpw.Ix_dev[0],
                            dpw.Iy_dev[1], dpw.Iy_dev[0],
                            dpw.Iz_dev[1], dpw.Iz_dev[0],
                            dpw.pml_rex_dev[0], dpw.pml_rex_dev[1],
                            dpw.pml_rex_dev[2], dpw.pml_rex_dev[3],
                            dpw.pml_rey_dev[0], dpw.pml_rey_dev[1],
                            dpw.pml_rey_dev[2], dpw.pml_rey_dev[3],
                            dpw.pml_rez_dev[0], dpw.pml_rez_dev[1],
                            dpw.pml_rez_dev[2], dpw.pml_rez_dev[3],
                            block=(256, 1, 1),
                            grid=(pml_size, 1, 1),
                        )

                    if bulk_size > 0:
                        dpw.update_1d_electric_dispersive_B_dev(
                            n, M, num_poles,
                            coef_D_ptr,
                            dpw.Px_dev, dpw.Py_dev, dpw.Pz_dev,
                            Ex_ptr, Ey_ptr, Ez_ptr,
                            block=(256, 1, 1),
                            grid=(bulk_size, 1, 1),
                        )

                self._launch_std_E_face_kernels(dpw)

            else:
                #  Axial electric 1D update (3 sequential launches)

                # Row offsets for axial electric arrays
                Ex_s = dpw.E_fields_s_dev[0]
                Ey_s = dpw.E_fields_s_dev[1]
                Ez_s = dpw.E_fields_s_dev[2]
                Hx_s = dpw.H_fields_s_dev[0]
                Hy_s = dpw.H_fields_s_dev[1]
                Hz_s = dpw.H_fields_s_dev[2]
                Ex_m = dpw.E_fields_dev[0]
                Ey_m = dpw.E_fields_dev[1]
                Ez_m = dpw.E_fields_dev[2]
                Hx_m = dpw.H_fields_dev[0]
                Hy_m = dpw.H_fields_dev[1]
                Hz_m = dpw.H_fields_dev[2]

                if not dpw.dispersive:
                    # Launch 1: Source grid bulk
                    if bulk_size > 0:
                        dpw.update_1d_electric_axial_source_dev(
                            n, M, mx, my, mz,
                            Ex_s, Ey_s, Ez_s,
                            Hx_s, Hy_s, Hz_s,
                            dpw.ID_dev,
                            self.grid.updatecoeffsH_dev,
                            self.grid.updatecoeffsE_dev,
                            block=(256, 1, 1),
                            grid=(bulk_size, 1, 1),
                        )

                    # Source grid PML
                    # Electric uses integral rows 0,1 of Ix_s/Iy_s/Iz_s
                    # (Ixjyz_s=Ix_s[0], Ixjzy_s=Ix_s[1], etc., per plane_wave.pyx).
                    # Kernel param order: Jxmzy_s..Jzmxy_s pairs with E PML naming.
                    if pml_size > 0:
                        dpw.update_1d_electric_axial_source_pml_dev(
                            n, p, M, mx, my, mz, dx, dy, dz,
                            Ex_s, Ey_s, Ez_s,
                            Hx_s, Hy_s, Hz_s,
                            dpw.Ix_s_dev[1],   # Ixjzy_s
                            dpw.Ix_s_dev[0],   # Ixjyz_s
                            dpw.Iy_s_dev[1],   # Iyjzx_s
                            dpw.Iy_s_dev[0],   # Iyjxz_s
                            dpw.Iz_s_dev[1],   # Izjyx_s
                            dpw.Iz_s_dev[0],   # Izjxy_s
                            dpw.pml_rex0_dev[0], dpw.pml_rex0_dev[1],
                            dpw.pml_rex0_dev[2], dpw.pml_rex0_dev[3],
                            dpw.pml_rey0_dev[0], dpw.pml_rey0_dev[1],
                            dpw.pml_rey0_dev[2], dpw.pml_rey0_dev[3],
                            dpw.pml_rez0_dev[0], dpw.pml_rez0_dev[1],
                            dpw.pml_rez0_dev[2], dpw.pml_rez0_dev[3],
                            dpw.ID_dev,
                            self.grid.updatecoeffsH_dev,
                            self.grid.updatecoeffsE_dev,
                            block=(256, 1, 1),
                            grid=(pml_size, 1, 1),
                        )

                    # Launch 2- Inject source into main grid at src-1
                    dpw.update_1d_electric_axial_inject_dev(
                        n, np.int32(dpw.origin_axial), mx, my, mz,
                        Ex_m, Ey_m, Ez_m,
                        Hx_s, Hy_s, Hz_s,
                        dpw.ID_dev,
                        self.grid.updatecoeffsH_dev,
                        self.grid.updatecoeffsE_dev,
                        block=(1, 1, 1),
                        grid=(1, 1, 1),
                    )

                    # Launch 3- Main grid bulk
                    # Electric main range is [M, N-M) - size N - 2M (no -1, unlike magnetic)
                    main_bulk_threads = dpw.length - 2 * dpw.m[3]
                    main_bulk_size = int(np.ceil(main_bulk_threads / 256)) if main_bulk_threads > 0 else 0
                    if main_bulk_size > 0:
                        dpw.update_1d_electric_axial_main_dev(
                            n, M, mx, my, mz,
                            Ex_m, Ey_m, Ez_m,
                            Hx_m, Hy_m, Hz_m,
                            dpw.ID_dev,
                            self.grid.updatecoeffsH_dev,
                            self.grid.updatecoeffsE_dev,
                            block=(256, 1, 1),
                            grid=(main_bulk_size, 1, 1),
                        )

                    # Main grid PML end
                    if pml_size > 0:
                        dpw.update_1d_electric_axial_main_pml_end_dev(
                            n, p, M, mx, my, mz, dx, dy, dz,
                            Ex_m, Ey_m, Ez_m,
                            Hx_m, Hy_m, Hz_m,
                            dpw.Ix_dev[1],   # Ixjzy
                            dpw.Ix_dev[0],   # Ixjyz
                            dpw.Iy_dev[1],   # Iyjzx
                            dpw.Iy_dev[0],   # Iyjxz
                            dpw.Iz_dev[1],   # Izjyx
                            dpw.Iz_dev[0],   # Izjxy
                            dpw.pml_rex_dev[0], dpw.pml_rex_dev[1],
                            dpw.pml_rex_dev[2], dpw.pml_rex_dev[3],
                            dpw.pml_rey_dev[0], dpw.pml_rey_dev[1],
                            dpw.pml_rey_dev[2], dpw.pml_rey_dev[3],
                            dpw.pml_rez_dev[0], dpw.pml_rez_dev[1],
                            dpw.pml_rez_dev[2], dpw.pml_rez_dev[3],
                            dpw.ID_dev,
                            self.grid.updatecoeffsH_dev,
                            self.grid.updatecoeffsE_dev,
                            block=(256, 1, 1),
                            grid=(pml_size, 1, 1),
                        )

                    # Main grid PML start
                    if pml_size > 0:
                        dpw.update_1d_electric_axial_main_pml_start_dev(
                            n, p, mx, my, mz, dx, dy, dz,
                            Ex_m, Ey_m, Ez_m,
                            Hx_m, Hy_m, Hz_m,
                            dpw.Ix0_dev[1],   # Ixjzy0
                            dpw.Ix0_dev[0],   # Ixjyz0
                            dpw.Iy0_dev[1],   # Iyjzx0
                            dpw.Iy0_dev[0],   # Iyjxz0
                            dpw.Iz0_dev[1],   # Izjyx0
                            dpw.Iz0_dev[0],   # Izjxy0
                            dpw.pml_rex0_dev[0], dpw.pml_rex0_dev[1],
                            dpw.pml_rex0_dev[2], dpw.pml_rex0_dev[3],
                            dpw.pml_rey0_dev[0], dpw.pml_rey0_dev[1],
                            dpw.pml_rey0_dev[2], dpw.pml_rey0_dev[3],
                            dpw.pml_rez0_dev[0], dpw.pml_rez0_dev[1],
                            dpw.pml_rez0_dev[2], dpw.pml_rez0_dev[3],
                            dpw.ID_dev,
                            self.grid.updatecoeffsH_dev,
                            self.grid.updatecoeffsE_dev,
                            block=(256, 1, 1),
                            grid=(pml_size, 1, 1),
                        )

                # Dispersive axial electric update
                # Mirrors updateElectricFields_dispersive_axial() in plane_wave.pyx.
                # Sequence: source bulk - source PML - source T - inject - main bulk - main PML end - main PML start - main T
                # Non-dispersive source/inject/PML kernels are reused where the math is identical.
                if dpw.dispersive:
                    num_poles = np.int32(config.get_model_config().materials["maxpoles"])
                    matD = self.grid.updatecoeffsdispersive_dev

                    # Source grid dispersive bulk
                    if bulk_size > 0:
                        dpw.update_1d_electric_dispersive_axial_source_dev(
                            n, M, mx, my, mz, num_poles,
                            dpw.Px_s_dev, dpw.Py_s_dev, dpw.Pz_s_dev,
                            Ex_s, Ey_s, Ez_s,
                            Hx_s, Hy_s, Hz_s,
                            dpw.ID_dev,
                            self.grid.updatecoeffsE_dev,
                            matD,
                            block=(256, 1, 1),
                            grid=(bulk_size, 1, 1),
                        )

                    # Source grid PML (reuse non-dispersive — same PML math)
                    if pml_size > 0:
                        dpw.update_1d_electric_axial_source_pml_dev(
                            n, p, M, mx, my, mz, dx, dy, dz,
                            Ex_s, Ey_s, Ez_s,
                            Hx_s, Hy_s, Hz_s,
                            dpw.Ix_s_dev[1], dpw.Ix_s_dev[0],
                            dpw.Iy_s_dev[1], dpw.Iy_s_dev[0],
                            dpw.Iz_s_dev[1], dpw.Iz_s_dev[0],
                            dpw.pml_rex0_dev[0], dpw.pml_rex0_dev[1],
                            dpw.pml_rex0_dev[2], dpw.pml_rex0_dev[3],
                            dpw.pml_rey0_dev[0], dpw.pml_rey0_dev[1],
                            dpw.pml_rey0_dev[2], dpw.pml_rey0_dev[3],
                            dpw.pml_rez0_dev[0], dpw.pml_rez0_dev[1],
                            dpw.pml_rez0_dev[2], dpw.pml_rez0_dev[3],
                            dpw.ID_dev,
                            self.grid.updatecoeffsH_dev,
                            self.grid.updatecoeffsE_dev,
                            block=(256, 1, 1),
                            grid=(pml_size, 1, 1),
                        )

                    # Source grid final T subtraction
                    if bulk_size > 0:
                        dpw.update_1d_electric_dispersive_axial_source_T_dev(
                            n, M, num_poles,
                            dpw.Px_s_dev, dpw.Py_s_dev, dpw.Pz_s_dev,
                            Ex_s, Ey_s, Ez_s,
                            dpw.ID_dev,
                            matD,
                            block=(256, 1, 1),
                            grid=(bulk_size, 1, 1),
                        )

                    # Inject source into main grid at src-1 (reuse non-dispersive inject)
                    dpw.update_1d_electric_axial_inject_dev(
                        n, np.int32(dpw.origin_axial), mx, my, mz,
                        Ex_m, Ey_m, Ez_m,
                        Hx_s, Hy_s, Hz_s,
                        dpw.ID_dev,
                        self.grid.updatecoeffsH_dev,
                        self.grid.updatecoeffsE_dev,
                        block=(1, 1, 1),
                        grid=(1, 1, 1),
                    )

                    # Main grid dispersive bulk
                    main_bulk_threads = dpw.length - 2 * dpw.m[3]
                    main_bulk_size = int(np.ceil(main_bulk_threads / 256)) if main_bulk_threads > 0 else 0
                    if main_bulk_size > 0:
                        dpw.update_1d_electric_dispersive_axial_main_dev(
                            n, M, mx, my, mz, num_poles,
                            dpw.Px_dev, dpw.Py_dev, dpw.Pz_dev,
                            Ex_m, Ey_m, Ez_m,
                            Hx_m, Hy_m, Hz_m,
                            dpw.ID_dev,
                            self.grid.updatecoeffsE_dev,
                            matD,
                            block=(256, 1, 1),
                            grid=(main_bulk_size, 1, 1),
                        )

                    # Main grid PML end (reuse non-dispersive)
                    if pml_size > 0:
                        dpw.update_1d_electric_axial_main_pml_end_dev(
                            n, p, M, mx, my, mz, dx, dy, dz,
                            Ex_m, Ey_m, Ez_m,
                            Hx_m, Hy_m, Hz_m,
                            dpw.Ix_dev[1], dpw.Ix_dev[0],
                            dpw.Iy_dev[1], dpw.Iy_dev[0],
                            dpw.Iz_dev[1], dpw.Iz_dev[0],
                            dpw.pml_rex_dev[0], dpw.pml_rex_dev[1],
                            dpw.pml_rex_dev[2], dpw.pml_rex_dev[3],
                            dpw.pml_rey_dev[0], dpw.pml_rey_dev[1],
                            dpw.pml_rey_dev[2], dpw.pml_rey_dev[3],
                            dpw.pml_rez_dev[0], dpw.pml_rez_dev[1],
                            dpw.pml_rez_dev[2], dpw.pml_rez_dev[3],
                            dpw.ID_dev,
                            self.grid.updatecoeffsH_dev,
                            self.grid.updatecoeffsE_dev,
                            block=(256, 1, 1),
                            grid=(pml_size, 1, 1),
                        )

                        # Main grid PML start (reuse non-dispersive)
                        dpw.update_1d_electric_axial_main_pml_start_dev(
                            n, p, mx, my, mz, dx, dy, dz,
                            Ex_m, Ey_m, Ez_m,
                            Hx_m, Hy_m, Hz_m,
                            dpw.Ix0_dev[1], dpw.Ix0_dev[0],
                            dpw.Iy0_dev[1], dpw.Iy0_dev[0],
                            dpw.Iz0_dev[1], dpw.Iz0_dev[0],
                            dpw.pml_rex0_dev[0], dpw.pml_rex0_dev[1],
                            dpw.pml_rex0_dev[2], dpw.pml_rex0_dev[3],
                            dpw.pml_rey0_dev[0], dpw.pml_rey0_dev[1],
                            dpw.pml_rey0_dev[2], dpw.pml_rey0_dev[3],
                            dpw.pml_rez0_dev[0], dpw.pml_rez0_dev[1],
                            dpw.pml_rez0_dev[2], dpw.pml_rez0_dev[3],
                            dpw.ID_dev,
                            self.grid.updatecoeffsH_dev,
                            self.grid.updatecoeffsE_dev,
                            block=(256, 1, 1),
                            grid=(pml_size, 1, 1),
                        )

                    # Main grid final T subtraction
                    if main_bulk_size > 0:
                        dpw.update_1d_electric_dispersive_axial_main_T_dev(
                            n, M, num_poles,
                            dpw.Px_dev, dpw.Py_dev, dpw.Pz_dev,
                            Ex_m, Ey_m, Ez_m,
                            dpw.ID_dev,
                            matD,
                            block=(256, 1, 1),
                            grid=(main_bulk_size, 1, 1),
                        )

                self._launch_axial_E_face_kernels(dpw)

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
        collector = getattr(self, "ntff_collector", None)
        if collector is not None:
            collector.finalise()

        # Copy output from receivers array back to correct receiver objects
        if self.grid.rxs:
            dtoh_rx_array(self.rxs_dev.get(), self.rxcoords_dev.get(), self.grid)

        if self.grid.transmissionlines:
            dtoh_transmission_line_outputs(
                self.tl_Vtotal_dev.get(), self.tl_Itotal_dev.get(), self.grid
            )

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
