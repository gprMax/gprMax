# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
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
from string import Template

import numpy as np
from jinja2 import Environment, PackageLoader

import gprMax.config as config

from .cuda.snapshots import knl_template_store_snapshot
from .cuda_opencl_el import (knl_fields_updates, knl_snapshots,
                             knl_source_updates, knl_store_outputs)
from .cython.fields_updates_normal import \
    update_electric as update_electric_cpu
from .cython.fields_updates_normal import \
    update_magnetic as update_magnetic_cpu
from .fields_outputs import knl_template_store_outputs
from .fields_outputs import store_outputs as store_outputs_cpu
from .receivers import dtoh_rx_array, htod_rx_arrays
from .snapshots import Snapshot, dtoh_snapshot_array, htod_snapshot_array
from .sources import htod_src_arrays
from .utilities.utilities import human_size, round32, timer

logger = logging.getLogger(__name__)


class CPUUpdates:
    """Defines update functions for CPU-based solver."""

    def __init__(self, G):
        """
        Args:
            G (FDTDGrid): Holds essential parameters describing the model.
        """
        self.grid = G
        self.dispersive_update_a = None
        self.dispersive_update_b = None

    def store_outputs(self):
        """Store field component values for every receiver and transmission line."""
        store_outputs_cpu(self.grid)

    def store_snapshots(self, iteration):
        """Store any snapshots.

        Args:
            iteration (int): Iteration number.
        """
        for snap in self.grid.snapshots:
            if snap.time == iteration + 1:
                snap.store(self.grid)

    def update_magnetic(self):
        """Update magnetic field components."""
        update_magnetic_cpu(self.grid.nx,
                        self.grid.ny,
                        self.grid.nz,
                        config.get_model_config().ompthreads,
                        self.grid.updatecoeffsH,
                        self.grid.ID,
                        self.grid.Ex,
                        self.grid.Ey,
                        self.grid.Ez,
                        self.grid.Hx,
                        self.grid.Hy,
                        self.grid.Hz)

    def update_magnetic_pml(self):
        """Update magnetic field components with the PML correction."""
        for pml in self.grid.pmls:
            pml.update_magnetic()

    def update_magnetic_sources(self):
        """Update magnetic field components from sources."""
        for source in self.grid.transmissionlines + self.grid.magneticdipoles:
            source.update_magnetic(self.grid.iteration,
                                   self.grid.updatecoeffsH,
                                   self.grid.ID,
                                   self.grid.Hx,
                                   self.grid.Hy,
                                   self.grid.Hz,
                                   self.grid)

    def update_electric_a(self):
        """Update electric field components."""
        # All materials are non-dispersive so do standard update.
        if config.get_model_config().materials['maxpoles'] == 0:
            update_electric_cpu(self.grid.nx,
                                self.grid.ny,
                                self.grid.nz,
                                config.get_model_config().ompthreads,
                                self.grid.updatecoeffsE,
                                self.grid.ID,
                                self.grid.Ex,
                                self.grid.Ey,
                                self.grid.Ez,
                                self.grid.Hx,
                                self.grid.Hy,
                                self.grid.Hz)

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            self.dispersive_update_a(self.grid.nx,
                                     self.grid.ny,
                                     self.grid.nz,
                                     config.get_model_config().ompthreads,
                                     config.get_model_config().materials['maxpoles'],
                                     self.grid.updatecoeffsE,
                                     self.grid.updatecoeffsdispersive,
                                     self.grid.ID,
                                     self.grid.Tx,
                                     self.grid.Ty,
                                     self.grid.Tz,
                                     self.grid.Ex,
                                     self.grid.Ey,
                                     self.grid.Ez,
                                     self.grid.Hx,
                                     self.grid.Hy,
                                     self.grid.Hz)

    def update_electric_pml(self):
        """Update electric field components with the PML correction."""
        for pml in self.grid.pmls:
            pml.update_electric()

    def update_electric_sources(self):
        """Update electric field components from sources -
            update any Hertzian dipole sources last.
        """
        for source in self.grid.voltagesources + self.grid.transmissionlines + self.grid.hertziandipoles:
            source.update_electric(self.grid.iteration,
                                   self.grid.updatecoeffsE,
                                   self.grid.ID,
                                   self.grid.Ex,
                                   self.grid.Ey,
                                   self.grid.Ez,
                                   self.grid)
        self.grid.iteration += 1

    def update_electric_b(self):
        """If there are any dispersive materials do 2nd part of dispersive
            update - it is split into two parts as it requires present and
            updated electric field values. Therefore it can only be completely
            updated after the electric field has been updated by the PML and
            source updates.
        """
        if config.get_model_config().materials['maxpoles'] != 0:
            self.dispersive_update_b(self.grid.nx,
                                     self.grid.ny,
                                     self.grid.nz,
                                     config.get_model_config().ompthreads,
                                     config.get_model_config().materials['maxpoles'],
                                     self.grid.updatecoeffsdispersive,
                                     self.grid.ID,
                                     self.grid.Tx,
                                     self.grid.Ty,
                                     self.grid.Tz,
                                     self.grid.Ex,
                                     self.grid.Ey,
                                     self.grid.Ez)

    def adapt_dispersive_config(self):
        """Set properties for disperive materials.

        Returns:
            props (Props): Dispersive material properties.
        """
        if config.get_model_config().materials['maxpoles'] > 1:
            poles = 'multi'
        else:
            poles = '1'

        if config.sim_config.general['precision'] == 'single':
            type = 'float'
        else:
            type = 'double'

        if config.get_model_config().materials['dispersivedtype'] == config.sim_config.dtypes['complex']:
            dispersion = 'complex'
        else:
            dispersion = 'real'

        class Props():
            pass

        props = Props()
        props.poles = poles
        props.precision = type
        props.dispersion_type = dispersion

        return props

    def set_dispersive_updates(self, props):
        """Set dispersive update functions.

        Args:
            props (Props): Dispersive material properties.
        """
        update_f = 'update_electric_dispersive_{}pole_{}_{}_{}'
        disp_a = update_f.format(props.poles, 'A', props.precision, props.dispersion_type)
        disp_b = update_f.format(props.poles, 'B', props.precision, props.dispersion_type)

        disp_a_f = getattr(import_module('gprMax.cython.fields_updates_dispersive'), disp_a)
        disp_b_f = getattr(import_module('gprMax.cython.fields_updates_dispersive'), disp_b)

        self.dispersive_update_a = disp_a_f
        self.dispersive_update_b = disp_b_f

    def time_start(self):
        """Start timer used to calculate solving time for model."""
        self.timestart = timer()

    def calculate_tsolve(self):
        """Calculate solving time for model."""
        return timer() - self.timestart

    def finalise(self):
        pass

    def cleanup(self):
        pass


class CUDAUpdates:
    """Defines update functions for GPU-based (CUDA) solver."""

    def __init__(self, G):
        """
        Args:
            G (FDTDGrid): Parameters describing a grid in a model.
        """

        self.grid = G
        self.dispersive_update_a = None
        self.dispersive_update_b = None

        # Import PyCUDA modules
        self.drv = import_module('pycuda.driver')
        self.source_module = getattr(import_module('pycuda.compiler'), 'SourceModule')
        self.drv.init()

        # Create device handle and context on specific GPU device (and make it current context)
        self.dev = config.get_model_config().device['dev']
        self.ctx = self.dev.make_context()

        # Set common substitutions for use in kernels
        self.subs_name_args = {'REAL': config.sim_config.dtypes['C_float_or_double'], 
                               'COMPLEX': config.get_model_config().materials['dispersiveCdtype']}
        self.subs_func = {'CUDA_IDX': 'int i = blockIdx.x * blockDim.x + threadIdx.x;',
                          'NX_FIELDS': self.grid.nx + 1,
                          'NY_FIELDS': self.grid.ny + 1,
                          'NZ_FIELDS': self.grid.nz + 1,
                          'NX_ID': self.grid.ID.shape[1],
                          'NY_ID': self.grid.ID.shape[2],
                          'NZ_ID': self.grid.ID.shape[3]}

        # Initialise arrays on GPU, prepare kernels, and get kernel functions
        self._set_macros()
        self._set_field_knls()
        if self.grid.pmls:
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

        name_plus_args = knl_func['args_cuda'].substitute(subs_name_args)
        func_body = knl_func['func'].substitute(subs_func)
        knl = self.knl_common + '\n' + name_plus_args + '{' + func_body + '}'

        return knl

    def _set_macros(self):
        """Common macros to be used in kernels."""

        if config.get_model_config().materials['maxpoles'] > 0:
            NY_MATDISPCOEFFS = self.grid.updatecoeffsdispersive.shape[1]
            NX_T = self.grid.Tx.shape[1]
            NY_T = self.grid.Tx.shape[2]
            NZ_T = self.grid.Tx.shape[3]
        else: # Set to one any substitutions for dispersive materials.
            NY_MATDISPCOEFFS = 1
            NX_T = 1
            NY_T = 1
            NZ_T = 1

        self.knl_common = self.env.get_template('knl_common_cuda.tmpl').render(
                            REAL=config.sim_config.dtypes['C_float_or_double'],
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
                            NZ_SNAPS=Snapshot.nz_max)

    def _set_field_knls(self):
        """Electric and magnetic field updates - prepare kernels, and
            get kernel functions.
        """
        
        bld = self._build_knl(knl_fields_updates.update_electric['func'], 
                              self.subs_name_args, self.subs_func)
        knlE = self.source_module(bld, 
                                  options=config.sim_config.cuda['nvcc_opts'])
        self.update_electric_gpu = knlE.get_function("update_electric")

        bld = self._build_knl(knl_fields_updates.update_magnetic['func'], 
                              self.subs_name_args, self.subs_func)
        knlH = self.source_module(bld, 
                                  options=config.sim_config.cuda['nvcc_opts'])
        self.update_magnetic_gpu = knlH.get_function("update_magnetic")

        self._copy_mat_coeffs(knlE, knlH)

        # Electric and magnetic field updates - dispersive materials
        # - get kernel functions and initialise array on GPU
        # If there are any dispersive materials (updates are split into two
        # parts as they require present and updated electric field values).
        if config.get_model_config().materials['maxpoles'] > 0:
            self.subs_func.update({'REAL': config.sim_config.dtypes['C_float_or_double'],
                                'REALFUNC': config.get_model_config().materials['cudarealfunc'],
                                'NX_T': self.grid.Tx.shape[1],
                                'NY_T': self.grid.Tx.shape[2],
                                'NZ_T': self.grid.Tx.shape[3]})

            bld = self._build_knl(knl_fields_updates.update_electric_dispersive_A['func'], 
                                  self.subs_name_args, self.subs_func)
            knl = self.source_module(bld, 
                                     options=config.sim_config.cuda['nvcc_opts'])
            self.dispersive_update_a = knl.get_function("update_electric_dispersive_A")

            bld = self._build_knl(knl_fields_updates.update_electric_dispersive_B['func'], 
                                  self.subs_name_args, self.subs_func)
            knl = self.source_module(bld, 
                                     options=config.sim_config.cuda['nvcc_opts'])
            self.dispersive_update_b = knl.get_function("update_electric_dispersive_B")

        # Electric and magnetic field updates - set blocks per grid and
        # initialise field arrays on GPU
        self.grid.set_blocks_per_grid()
        self.grid.htod_geometry_arrays()
        self.grid.htod_field_arrays()
        if config.get_model_config().materials['maxpoles'] > 0:
            self.grid.htod_dispersive_arrays()

    def _set_pml_knls(self):
        """PMLS - prepare kernels and get kernel functions."""
        knl_pml_updates_electric = import_module('gprMax.cuda_opencl_el.knl_pml_updates_electric_' + self.grid.pmlformulation)
        knl_pml_updates_magnetic = import_module('gprMax.cuda_opencl_el.knl_pml_updates_magnetic_' + self.grid.pmlformulation)

        # Set block per grid, initialise arrays on GPU, and get kernel functions
        for pml in self.grid.pmls:
            pml.set_blocks_per_grid()
            pml.htod_field_arrays()
            knl_name = 'order' + str(len(pml.CFS)) + '_' + pml.direction
            self.subs_name_args['FUNC'] = knl_name

            knl_electric = getattr(knl_pml_updates_electric, knl_name)
            bld = self._build_knl(knl_electric['func'], 
                                  self.subs_name_args, self.subs_func)
            knlE = self.source_module(bld, options=config.sim_config.cuda['nvcc_opts'])
            pml.update_electric_dev = knlE.get_function(knl_name)

            knl_magnetic = getattr(knl_pml_updates_magnetic, knl_name)
            bld = self._build_knl(knl_magnetic['func'], 
                                  self.subs_name_args, self.subs_func)
            knlH = self.source_module(bld, options=config.sim_config.cuda['nvcc_opts'])
            pml.update_magnetic_dev = knlH.get_function(knl_name)

        self._copy_mat_coeffs(knlE, knlH)

    def _set_rx_knl(self):
        """Receivers - initialise arrays on GPU, prepare kernel and get kernel
                        function.
        """
        self.rxcoords_gpu, self.rxs_gpu = htod_rx_arrays(self.grid)

        self.subs_func.update({'REAL': config.sim_config.dtypes['C_float_or_double'],
                               'NY_RXCOORDS': 3,
                               'NX_RXS': 6,
                               'NY_RXS': self.grid.iterations,
                               'NZ_RXS': len(self.grid.rxs)})

        bld = self._build_knl(knl_store_outputs.store_outputs['func'], 
                              self.subs_name_args, self.subs_func)
        knl = self.source_module(bld, options=config.sim_config.cuda['nvcc_opts'])
        self.store_outputs_gpu = knl.get_function("store_outputs")

    def _set_src_knls(self):
        """Sources - initialise arrays on GPU, prepare kernel and get kernel
                        function.
        """
        self.subs_func.update({'NY_SRCINFO': 4, 
                               'NY_SRCWAVES': self.grid.iteration})
        
        if self.grid.hertziandipoles:
            self.srcinfo1_hertzian_gpu, self.srcinfo2_hertzian_gpu, self.srcwaves_hertzian_gpu = htod_src_arrays(self.grid.hertziandipoles, self.grid)
            bld = self._build_knl(knl_source_updates.update_hertzian_dipole['func'], 
                                  self.subs_name_args, self.subs_func)
            knl = self.source_module(bld, options=config.sim_config.cuda['nvcc_opts'])
            self.update_hertzian_dipole_gpu = knl.get_function("update_hertzian_dipole")
        if self.grid.magneticdipoles:
            self.srcinfo1_magnetic_gpu, self.srcinfo2_magnetic_gpu, self.srcwaves_magnetic_gpu = htod_src_arrays(self.grid.magneticdipoles, self.grid)
            bld = self._build_knl(knl_source_updates.update_magnetic_dipole['func'], 
                                  self.subs_name_args, self.subs_func)
            knl = self.source_module(bld, options=config.sim_config.cuda['nvcc_opts'])
            self.update_magnetic_dipole_gpu = knl.get_function("update_magnetic_dipole")
        if self.grid.voltagesources:
            self.srcinfo1_voltage_gpu, self.srcinfo2_voltage_gpu, self.srcwaves_voltage_gpu = htod_src_arrays(self.grid.voltagesources, self.grid)
            bld = self._build_knl(knl_source_updates.update_voltage_source['func'], 
                                  self.subs_name_args, self.subs_func)
            knl = self.source_module(bld, options=config.sim_config.cuda['nvcc_opts'])
            self.update_voltage_source_gpu = knl.get_function("update_voltage_source")

        if (self.grid.hertziandipoles + 
            self.grid.magneticdipoles + 
            self.grid.voltagesources):
            self._copy_mat_coeffs(knl, knl)

    def _set_snapshot_knl(self):
        """Snapshots - initialise arrays on GPU, prepare kernel and get kernel
                        function.
        """
        self.snapEx_gpu, self.snapEy_gpu, self.snapEz_gpu, self.snapHx_gpu, self.snapHy_gpu, self.snapHz_gpu = htod_snapshot_array(self.grid)

        self.subs_func.update({'REAL': config.sim_config.dtypes['C_float_or_double'],
                               'NX_SNAPS': Snapshot.nx_max,
                               'NY_SNAPS': Snapshot.ny_max,
                               'NZ_SNAPS': Snapshot.nz_max})

        bld = self._build_knl(knl_snapshots.store_snapshot['func'], 
                              self.subs_name_args, self.subs_func)
        knl = self.source_module(bld, options=config.sim_config.cuda['nvcc_opts'])
        self.store_snapshot_gpu = knl.get_function("store_snapshot")

    def _copy_mat_coeffs(self, knlE, knlH):
        """Copy material coefficient arrays to constant memory of GPU
            (must be <64KB).

        Args:
            knlE (kernel): electric field kernel.
            knlH (kernel): magnetic field kernel.
        """

        # Check if coefficient arrays will fit on constant memory of GPU
        if (self.grid.updatecoeffsE.nbytes + self.grid.updatecoeffsH.nbytes
            > config.get_model_config().cuda['gpu'].total_constant_memory):
            logger.exception(f"Too many materials in the model to fit onto constant memory of size {human_size(config.get_model_config().cuda['gpu'].total_constant_memory)} on {config.get_model_config().cuda['gpu'].deviceID} - {config.get_model_config().cuda['gpu'].name} GPU")
            raise ValueError

        updatecoeffsE = knlE.get_global('updatecoeffsE')[0]
        updatecoeffsH = knlH.get_global('updatecoeffsH')[0]
        self.drv.memcpy_htod(updatecoeffsE, self.grid.updatecoeffsE)
        self.drv.memcpy_htod(updatecoeffsH, self.grid.updatecoeffsH)

    def store_outputs(self):
        """Store field component values for every receiver."""
        if self.grid.rxs:
            self.store_outputs_gpu(np.int32(len(self.grid.rxs)),
                                   np.int32(self.grid.iteration),
                                   self.rxcoords_gpu.gpudata,
                                   self.rxs_gpu.gpudata,
                                   self.grid.Ex_gpu.gpudata,
                                   self.grid.Ey_gpu.gpudata,
                                   self.grid.Ez_gpu.gpudata,
                                   self.grid.Hx_gpu.gpudata,
                                   self.grid.Hy_gpu.gpudata,
                                   self.grid.Hz_gpu.gpudata,
                                   block=(1, 1, 1),
                                   grid=(round32(len(self.grid.rxs)), 1, 1))

    def store_snapshots(self, iteration):
        """Store any snapshots.

        Args:
            iteration (int): iteration number.
        """

        for i, snap in enumerate(self.grid.snapshots):
            if snap.time == iteration + 1:
                snapno = 0 if config.get_model_config().cuda['snapsgpu2cpu'] else i
                self.store_snapshot_gpu(np.int32(snapno),
                                        np.int32(snap.xs),
                                        np.int32(snap.xf),
                                        np.int32(snap.ys),
                                        np.int32(snap.yf),
                                        np.int32(snap.zs),
                                        np.int32(snap.zf),
                                        np.int32(snap.dx),
                                        np.int32(snap.dy),
                                        np.int32(snap.dz),
                                        self.grid.Ex_gpu.gpudata,
                                        self.grid.Ey_gpu.gpudata,
                                        self.grid.Ez_gpu.gpudata,
                                        self.grid.Hx_gpu.gpudata,
                                        self.grid.Hy_gpu.gpudata,
                                        self.grid.Hz_gpu.gpudata,
                                        self.snapEx_gpu.gpudata,
                                        self.snapEy_gpu.gpudata,
                                        self.snapEz_gpu.gpudata,
                                        self.snapHx_gpu.gpudata,
                                        self.snapHy_gpu.gpudata,
                                        self.snapHz_gpu.gpudata,
                                        block=Snapshot.tpb,
                                        grid=Snapshot.bpg)
                if config.get_model_config().cuda['snapsgpu2cpu']:
                    dtoh_snapshot_array(self.snapEx_gpu.get(),
                                        self.snapEy_gpu.get(),
                                        self.snapEz_gpu.get(),
                                        self.snapHx_gpu.get(),
                                        self.snapHy_gpu.get(),
                                        self.snapHz_gpu.get(),
                                        0, snap)

    def update_magnetic(self):
        """Update magnetic field components."""
        self.update_magnetic_gpu(np.int32(self.grid.nx),
                                 np.int32(self.grid.ny),
                                 np.int32(self.grid.nz),
                                 self.grid.ID_gpu.gpudata,
                                 self.grid.Hx_gpu.gpudata,
                                 self.grid.Hy_gpu.gpudata,
                                 self.grid.Hz_gpu.gpudata,
                                 self.grid.Ex_gpu.gpudata,
                                 self.grid.Ey_gpu.gpudata,
                                 self.grid.Ez_gpu.gpudata,
                                 block=self.grid.tpb,
                                 grid=self.grid.bpg)

    def update_magnetic_pml(self):
        """Update magnetic field components with the PML correction."""
        for pml in self.grid.pmls:
            pml.update_magnetic()

    def update_magnetic_sources(self):
        """Update magnetic field components from sources."""
        if self.grid.magneticdipoles:
            self.update_magnetic_dipole_gpu(np.int32(len(self.grid.magneticdipoles)),
                                            np.int32(self.grid.iteration),
                                            config.sim_config.dtypes['float_or_double'](self.grid.dx),
                                            config.sim_config.dtypes['float_or_double'](self.grid.dy),
                                            config.sim_config.dtypes['float_or_double'](self.grid.dz),
                                            self.srcinfo1_magnetic_gpu.gpudata,
                                            self.srcinfo2_magnetic_gpu.gpudata,
                                            self.srcwaves_magnetic_gpu.gpudata,
                                            self.grid.ID_gpu.gpudata,
                                            self.grid.Hx_gpu.gpudata,
                                            self.grid.Hy_gpu.gpudata,
                                            self.grid.Hz_gpu.gpudata,
                                            block=(1, 1, 1),
                                            grid=(round32(len(self.grid.magneticdipoles)), 1, 1))

    def update_electric_a(self):
        """Update electric field components."""
        # All materials are non-dispersive so do standard update.
        if config.get_model_config().materials['maxpoles'] == 0:
            self.update_electric_gpu(np.int32(self.grid.nx),
                                     np.int32(self.grid.ny),
                                     np.int32(self.grid.nz),
                                     self.grid.ID_gpu.gpudata,
                                     self.grid.Ex_gpu.gpudata,
                                     self.grid.Ey_gpu.gpudata,
                                     self.grid.Ez_gpu.gpudata,
                                     self.grid.Hx_gpu.gpudata,
                                     self.grid.Hy_gpu.gpudata,
                                     self.grid.Hz_gpu.gpudata,
                                     block=self.grid.tpb,
                                     grid=self.grid.bpg)

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            self.dispersive_update_a(np.int32(self.grid.nx),
                                     np.int32(self.grid.ny),
                                     np.int32(self.grid.nz),
                                     np.int32(config.get_model_config().materials['maxpoles']),
                                     self.grid.updatecoeffsdispersive_gpu.gpudata,
                                     self.grid.Tx_gpu.gpudata,
                                     self.grid.Ty_gpu.gpudata,
                                     self.grid.Tz_gpu.gpudata,
                                     self.grid.ID_gpu.gpudata,
                                     self.grid.Ex_gpu.gpudata,
                                     self.grid.Ey_gpu.gpudata,
                                     self.grid.Ez_gpu.gpudata,
                                     self.grid.Hx_gpu.gpudata,
                                     self.grid.Hy_gpu.gpudata,
                                     self.grid.Hz_gpu.gpudata,
                                     block=self.grid.tpb,
                                     grid=self.grid.bpg)

    def update_electric_pml(self):
        """Update electric field components with the PML correction."""
        for pml in self.grid.pmls:
            pml.update_electric()

    def update_electric_sources(self):
        """Update electric field components from sources -
            update any Hertzian dipole sources last.
        """
        if self.grid.voltagesources:
            self.update_voltage_source_gpu(np.int32(len(self.grid.voltagesources)),
                                           np.int32(self.grid.iteration),
                                           config.sim_config.dtypes['float_or_double'](self.grid.dx),
                                           config.sim_config.dtypes['float_or_double'](self.grid.dy),
                                           config.sim_config.dtypes['float_or_double'](self.grid.dz),
                                           self.srcinfo1_voltage_gpu.gpudata,
                                           self.srcinfo2_voltage_gpu.gpudata,
                                           self.srcwaves_voltage_gpu.gpudata,
                                           self.grid.ID_gpu.gpudata,
                                           self.grid.Ex_gpu.gpudata,
                                           self.grid.Ey_gpu.gpudata,
                                           self.grid.Ez_gpu.gpudata,
                                           block=(1, 1, 1),
                                           grid=(round32(len(self.grid.voltagesources)), 1, 1))

        if self.grid.hertziandipoles:
            self.update_hertzian_dipole_gpu(np.int32(len(self.grid.hertziandipoles)),
                                            np.int32(self.grid.iteration),
                                            config.sim_config.dtypes['float_or_double'](self.grid.dx),
                                            config.sim_config.dtypes['float_or_double'](self.grid.dy),
                                            config.sim_config.dtypes['float_or_double'](self.grid.dz),
                                            self.srcinfo1_hertzian_gpu.gpudata,
                                            self.srcinfo2_hertzian_gpu.gpudata,
                                            self.srcwaves_hertzian_gpu.gpudata,
                                            self.grid.ID_gpu.gpudata,
                                            self.grid.Ex_gpu.gpudata,
                                            self.grid.Ey_gpu.gpudata,
                                            self.grid.Ez_gpu.gpudata,
                                            block=(1, 1, 1),
                                            grid=(round32(len(self.grid.hertziandipoles)), 1, 1))

        self.grid.iteration += 1

    def update_electric_b(self):
        """If there are any dispersive materials do 2nd part of dispersive
            update - it is split into two parts as it requires present and
            updated electric field values. Therefore it can only be completely
            updated after the electric field has been updated by the PML and
            source updates.
        """
        if config.get_model_config().materials['maxpoles'] > 0:
            self.dispersive_update_b(np.int32(self.grid.nx),
                                     np.int32(self.grid.ny),
                                     np.int32(self.grid.nz),
                                     np.int32(config.get_model_config().materials['maxpoles']),
                                     self.grid.updatecoeffsdispersive_gpu.gpudata,
                                     self.grid.Tx_gpu.gpudata,
                                     self.grid.Ty_gpu.gpudata,
                                     self.grid.Tz_gpu.gpudata,
                                     self.grid.ID_gpu.gpudata,
                                     self.grid.Ex_gpu.gpudata,
                                     self.grid.Ey_gpu.gpudata,
                                     self.grid.Ez_gpu.gpudata,
                                     block=self.grid.tpb,
                                     grid=self.grid.bpg)

    def time_start(self):
        """Start event timers used to calculate solving time for model."""
        self.iterstart = self.drv.Event()
        self.iterend = self.drv.Event()
        self.iterstart.record()
        self.iterstart.synchronize()

    def calculate_memsolve(self, iteration):
        """Calculate memory used on last iteration.

        Args:
            iteration (int): Iteration number.

        Returns:
            Memory (RAM) used on GPU.
        """
        if iteration == self.grid.iterations - 1:
            return self.drv.mem_get_info()[1] - self.drv.mem_get_info()[0]

    def calculate_tsolve(self):
        """Calculate solving time for model."""
        self.iterend.record()
        self.iterend.synchronize()
        tsolve = self.iterstart.time_till(self.iterend) * 1e-3

        return tsolve

    def finalise(self):
        """Copy data from GPU back to CPU to save to file(s)."""
        # Copy output from receivers array back to correct receiver objects
        if self.grid.rxs:
            dtoh_rx_array(self.rxs_gpu.get(),
                          self.rxcoords_gpu.get(),
                          self.grid)

        # Copy data from any snapshots back to correct snapshot objects
        if self.grid.snapshots and not config.get_model_config().cuda['snapsgpu2cpu']:
            for i, snap in enumerate(self.grid.snapshots):
                dtoh_snapshot_array(self.snapEx_gpu.get(),
                                   self.snapEy_gpu.get(),
                                   self.snapEz_gpu.get(),
                                   self.snapHx_gpu.get(),
                                   self.snapHy_gpu.get(),
                                   self.snapHz_gpu.get(),
                                   i, snap)

    def cleanup(self):
        """Cleanup GPU context."""
        # Remove context from top of stack and delete
        self.ctx.pop()
        del self.ctx


class OpenCLUpdates:
    """Defines update functions for OpenCL-based solver."""

    def __init__(self, G):
        """
        Args:
            G: FDTDObject of parameters describing a grid in a model.
        """

        self.grid = G
        self.dispersive_update_a = None
        self.dispersive_update_b = None
        self.compute_time = 0

        # Import pyopencl module
        self.cl = import_module('pyopencl')
        self.elwise = getattr(import_module('pyopencl.elementwise'), 'ElementwiseKernel')

        # Select device, create context and command queue
        self.dev = config.get_model_config().device['dev']
        self.ctx = self.cl.Context(devices=[self.dev])
        self.queue = self.cl.CommandQueue(self.ctx, 
                        properties=self.cl.command_queue_properties.PROFILING_ENABLE)

        # Enviroment for templating kernels
        self.env = Environment(loader=PackageLoader('gprMax', 'cuda_opencl_el'))

        # Initialise arrays on device, prepare kernels, and get kernel functions
        self._set_field_knls()
        if self.grid.pmls:
            self._set_pml_knls()
        if self.grid.rxs:
            self._set_rx_knl()
        if self.grid.voltagesources + self.grid.hertziandipoles + self.grid.magneticdipoles:
            self._set_src_knls()
        if self.grid.snapshots:
            self._set_snapshot_knl()        

    def _set_field_knls(self):
        """Electric and magnetic field updates - prepare kernels, and
            get kernel functions.
        """
        if config.get_model_config().materials['maxpoles'] > 0:
            NY_MATDISPCOEFFS = self.grid.updatecoeffsdispersive.shape[1]
            NX_T = self.grid.Tx.shape[1]
            NY_T = self.grid.Tx.shape[2]
            NZ_T = self.grid.Tx.shape[3]
        else: # Set to one any substitutions for dispersive materials.
            NY_MATDISPCOEFFS = 1
            NX_T = 1
            NY_T = 1
            NZ_T = 1

        self.knl_common = self.env.get_template('knl_common_opencl.tmpl').render(
                            updatecoeffsE = self.grid.updatecoeffsE.ravel(),
                            updatecoeffsH = self.grid.updatecoeffsH.ravel(),
                            REAL=config.sim_config.dtypes['C_float_or_double'],
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
                            NZ_SNAPS=Snapshot.nz_max)

        subs = {'CUDA_IDX': '',
                'NX_FIELDS': self.grid.nx + 1,
                'NY_FIELDS': self.grid.ny + 1,
                'NZ_FIELDS': self.grid.nz + 1,
                'NX_ID': self.grid.ID.shape[1],
                'NY_ID': self.grid.ID.shape[2],
                'NZ_ID': self.grid.ID.shape[3]}

        self.update_electric_dev = self.elwise(self.ctx, 
                                    knl_fields_updates.update_electric['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double']}), 
                                    knl_fields_updates.update_electric['func'].substitute(subs), 
                                    'update_electric', preamble=self.knl_common,
                                    options=config.sim_config.devices['compiler_opts'])

        self.update_magnetic_dev = self.elwise(self.ctx, 
                                    knl_fields_updates.update_magnetic['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double']}),
                                    knl_fields_updates.update_magnetic['func'].substitute(subs), 
                                    'update_magnetic', preamble=self.knl_common,
                                    options=config.sim_config.devices['compiler_opts'])

        # Electric and magnetic field updates - dispersive materials - 
        #                                        get kernel functions
        # If there are any dispersive materials (updates are split into two
        # parts as they require present and updated electric field values).
        if config.get_model_config().materials['maxpoles'] > 0:
            subs = {'CUDA_IDX': '',
                    'REAL': config.sim_config.dtypes['C_float_or_double'],
                    'REALFUNC': config.get_model_config().materials['crealfunc'],
                    'NX_FIELDS': self.grid.nx + 1,
                    'NY_FIELDS': self.grid.ny + 1,
                    'NZ_FIELDS': self.grid.nz + 1,
                    'NX_ID': self.grid.ID.shape[1],
                    'NY_ID': self.grid.ID.shape[2],
                    'NZ_ID': self.grid.ID.shape[3],
                    'NX_T': NX_T,
                    'NY_T': NY_T,
                    'NZ_T': NZ_T}
            self.dispersive_update_a = self.elwise(self.ctx, 
                                        knl_fields_updates.update_electric_dispersive_A['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double'], 'COMPLEX': config.get_model_config().materials['dispersiveCdtype']}), 
                                        knl_fields_updates.update_electric_dispersive_A['func'].substitute(subs), 
                                        'update_electric_dispersive_A', preamble=self.knl_common,
                                        options=config.sim_config.devices['compiler_opts'])
            self.dispersive_update_b = self.elwise(self.ctx, 
                                        knl_fields_updates.update_electric_dispersive_B['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double'] ,'COMPLEX': config.get_model_config().materials['dispersiveCdtype']}), 
                                        knl_fields_updates.update_electric_dispersive_B['func'].substitute(subs), 
                                        'update_electric_dispersive_B', preamble=self.knl_common,
                                        options=config.sim_config.devices['compiler_opts'])
                                                                      
        # Electric and magnetic field updates - initialise field arrays on 
        #                                       compute device
        self.grid.htod_geometry_arrays(self.queue)
        self.grid.htod_field_arrays(self.queue)
        if config.get_model_config().materials['maxpoles'] > 0:
            self.grid.htod_dispersive_arrays(self.queue)

    def _set_pml_knls(self):
        """PMLS - prepare kernels and get kernel functions."""
        knl_pml_updates_electric = import_module('gprMax.cuda_opencl_el.knl_pml_updates_electric_' + self.grid.pmlformulation)
        knl_pml_updates_magnetic = import_module('gprMax.cuda_opencl_el.knl_pml_updates_magnetic_' + self.grid.pmlformulation)

        subs = {'CUDA_IDX': '',
                'REAL': config.sim_config.dtypes['C_float_or_double'],
                'NX_FIELDS': self.grid.nx + 1,
                'NY_FIELDS': self.grid.ny + 1,
                'NZ_FIELDS': self.grid.nz + 1,
                'NX_ID': self.grid.ID.shape[1],
                'NY_ID': self.grid.ID.shape[2],
                'NZ_ID': self.grid.ID.shape[3]}

        # Set workgroup size, initialise arrays on compute device, and get 
        # kernel functions
        for pml in self.grid.pmls:
            pml.set_queue(self.queue)
            pml.htod_field_arrays()
            knl_name = 'order' + str(len(pml.CFS)) + '_' + pml.direction
            knl_electric_name = getattr(knl_pml_updates_electric, knl_name)
            knl_magnetic_name = getattr(knl_pml_updates_magnetic, knl_name)   

            pml.update_electric_dev = self.elwise(self.ctx, 
                                        knl_electric_name['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double']}), 
                                        knl_electric_name['func'].substitute(subs),
                                        'pml_updates_electric_' + knl_name, 
                                        preamble=self.knl_common,
                                        options=config.sim_config.devices['compiler_opts'])
            
            pml.update_magnetic_dev = self.elwise(self.ctx, 
                                        knl_magnetic_name['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double']}), 
                                        knl_magnetic_name['func'].substitute(subs),
                                        'pml_updates_magnetic_' + knl_name, 
                                        preamble=self.knl_common,
                                        options=config.sim_config.devices['compiler_opts'])

    def _set_rx_knl(self):
        """Receivers - initialise arrays on compute device, prepare kernel and 
                        get kernel function.
        """
        self.rxcoords_dev, self.rxs_dev = htod_rx_arrays(self.grid, self.queue)
        self.store_outputs_dev = self.elwise(self.ctx, 
                                    knl_store_outputs.store_outputs['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double']}), 
                                    knl_store_outputs.store_outputs['func'].substitute({'CUDA_IDX': ''}), 
                                             'store_outputs', preamble=self.knl_common,
                                             options=config.sim_config.devices['compiler_opts'])

    def _set_src_knls(self):
        """Sources - initialise arrays on compute device, prepare kernel and 
                    get kernel function.
        """
        if self.grid.hertziandipoles:
            self.srcinfo1_hertzian_dev, self.srcinfo2_hertzian_dev, self.srcwaves_hertzian_dev = htod_src_arrays(self.grid.hertziandipoles, self.grid, self.queue)
            self.update_hertzian_dipole_dev = self.elwise(self.ctx, 
                                                knl_source_updates.update_hertzian_dipole['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double']}), 
                                                knl_source_updates.update_hertzian_dipole['func'].substitute({'CUDA_IDX': '', 'REAL': config.sim_config.dtypes['C_float_or_double']}), 
                                                'update_hertzian_dipole', preamble=self.knl_common,
                                                options=config.sim_config.devices['compiler_opts'])
        if self.grid.magneticdipoles:
            self.srcinfo1_magnetic_dev, self.srcinfo2_magnetic_dev, self.srcwaves_magnetic_dev = htod_src_arrays(self.grid.magneticdipoles, self.grid, self.queue)
            self.update_magnetic_dipole_dev = self.elwise(self.ctx, 
                                                knl_source_updates.update_magnetic_dipole['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double']}), 
                                                knl_source_updates.update_magnetic_dipole['func'].substitute({'CUDA_IDX': '', 'REAL': config.sim_config.dtypes['C_float_or_double']}),
                                                'update_magnetic_dipole', preamble=self.knl_common,
                                                options=config.sim_config.devices['compiler_opts'])
        if self.grid.voltagesources:
            self.srcinfo1_voltage_dev, self.srcinfo2_voltage_dev,self.srcwaves_voltage_dev = htod_src_arrays(self.grid.voltagesources, self.grid, self.queue)
            self.update_voltage_source_dev = self.elwise(self.ctx, 
                                                knl_source_updates.update_voltage_source['args_opencl'].substitute({'CUDA_IDX': '', 'REAL': config.sim_config.dtypes['C_float_or_double']}), 
                                                knl_source_updates.update_voltage_source['func'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double']}), 'update_voltage_source', preamble=self.knl_common,
                                                options=config.sim_config.devices['compiler_opts'])

    def _set_snapshot_knl(self):
        """Snapshots - initialise arrays on compute device, prepare kernel and 
                        get kernel function.
        """
        self.snapEx_dev, self.snapEy_dev, self.snapEz_dev, self.snapHx_dev, self.snapHy_dev, self.snapHz_dev = htod_snapshot_array(self.grid, self.queue)
        self.store_snapshot_dev = self.elwise(self.ctx, 
                                    knl_snapshots.store_snapshot['args_opencl'].substitute({'REAL': config.sim_config.dtypes['C_float_or_double']}), 
                                    knl_snapshots.store_snapshot['func'].substitute(        {'CUDA_IDX': '',
                                              'NX_SNAPS': Snapshot.nx_max,
                                              'NY_SNAPS': Snapshot.ny_max,
                                              'NZ_SNAPS': Snapshot.nz_max}), 
                                    'store_snapshot', preamble=self.knl_common,
                                    options=config.sim_config.devices['compiler_opts'])

    def store_outputs(self):
        """Store field component values for every receiver."""
        if self.grid.rxs:
            event = self.store_outputs_dev(np.int32(len(self.grid.rxs)),
                                           np.int32(self.grid.iteration),
                                           self.rxcoords_dev,
                                           self.rxs_dev,
                                           self.grid.Ex_dev,
                                           self.grid.Ey_dev,
                                           self.grid.Ez_dev,
                                           self.grid.Hx_dev,
                                           self.grid.Hy_dev,
                                           self.grid.Hz_dev)
            event.wait()
            self.compute_time += (event.profile.end - event.profile.start)*1e-9

    def store_snapshots(self, iteration):
        """Store any snapshots.

        Args:
            iteration: int for iteration number.
        """

        for i, snap in enumerate(self.grid.snapshots):
            if snap.time == iteration + 1:
                snapno = 0 if config.get_model_config().device['snapsgpu2cpu'] else i
                event = self.store_snapshot_dev(np.int32(snapno),
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
                                                self.snapHz_dev)
                event.wait()
                if config.get_model_config().device['snapsgpu2cpu']:
                    dtoh_snapshot_array(self.snapEx_dev.get(),
                                        self.snapEy_dev.get(),
                                        self.snapEz_dev.get(),
                                        self.snapHx_dev.get(),
                                        self.snapHy_dev.get(),
                                        self.snapHz_dev.get(),
                                        0,
                                        snap)
                self.compute_time += (event.profile.end - event.profile.start)*1e-9

    def update_magnetic(self):
        """Update magnetic field components."""
        event = self.update_magnetic_dev(np.int32(self.grid.nx),
                                         np.int32(self.grid.ny),
                                         np.int32(self.grid.nz),
                                         self.grid.ID_dev,
                                         self.grid.Hx_dev,
                                         self.grid.Hy_dev,
                                         self.grid.Hz_dev,
                                         self.grid.Ex_dev,
                                         self.grid.Ey_dev,
                                         self.grid.Ez_dev)
        event.wait()
        self.compute_time += (event.profile.end - event.profile.start)*1e-9

    def update_magnetic_pml(self):
        """Update magnetic field components with the PML correction."""
        for pml in self.grid.pmls:
            pml.update_magnetic()
            self.compute_time += pml.compute_time

    def update_magnetic_sources(self):
        """Update magnetic field components from sources."""
        if self.grid.magneticdipoles:
            event = self.update_magnetic_dipole_dev(np.int32(len(self.grid.magneticdipoles)),
                        np.int32(self.grid.iteration),
                        config.sim_config.dtypes['float_or_double'](self.grid.dx),
                        config.sim_config.dtypes['float_or_double'](self.grid.dy),
                        config.sim_config.dtypes['float_or_double'](self.grid.dz),
                        self.srcinfo1_magnetic_dev,
                        self.srcinfo2_magnetic_dev,
                        self.srcwaves_magnetic_dev,
                        self.grid.ID_dev,
                        self.grid.Hx_dev,
                        self.grid.Hy_dev,
                        self.grid.Hz_dev)
            event.wait()
            self.compute_time += (event.profile.end - event.profile.start)*1e-9

    def update_electric_a(self):
        """Update electric field components."""
        # All materials are non-dispersive so do standard update.
        if config.get_model_config().materials['maxpoles'] == 0:
            event = self.update_electric_dev(np.int32(self.grid.nx),
                                             np.int32(self.grid.ny),
                                             np.int32(self.grid.nz),
                                             self.grid.ID_dev,
                                             self.grid.Ex_dev,
                                             self.grid.Ey_dev,
                                             self.grid.Ez_dev,
                                             self.grid.Hx_dev,
                                             self.grid.Hy_dev,
                                             self.grid.Hz_dev)
            event.wait()
            self.compute_time += (event.profile.end - event.profile.start)*1e-9

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            event = self.dispersive_update_a(np.int32(self.grid.nx),
                        np.int32(self.grid.ny),
                        np.int32(self.grid.nz),
                        np.int32(config.get_model_config().materials['maxpoles']),
                        self.grid.updatecoeffsdispersive_dev,
                        self.grid.Tx_dev,
                        self.grid.Ty_dev,
                        self.grid.Tz_dev,
                        self.grid.ID_dev,
                        self.grid.Ex_dev,
                        self.grid.Ey_dev,
                        self.grid.Ez_dev,
                        self.grid.Hx_dev,
                        self.grid.Hy_dev,
                        self.grid.Hz_dev)
            event.wait()
            self.compute_time += (event.profile.end - event.profile.start)*1e-9

    def update_electric_pml(self):
        """Update electric field components with the PML correction."""
        for pml in self.grid.pmls:
            pml.update_electric()
            self.compute_time += pml.compute_time

    def update_electric_sources(self):
        """Update electric field components from sources -
            update any Hertzian dipole sources last.
        """
        if self.grid.voltagesources:
            event = self.update_voltage_source_dev(np.int32(len(self.grid.voltagesources)),
                        np.int32(self.grid.iteration),
                        config.sim_config.dtypes['float_or_double'](self.grid.dx),
                        config.sim_config.dtypes['float_or_double'](self.grid.dy),
                        config.sim_config.dtypes['float_or_double'](self.grid.dz),
                        self.srcinfo1_voltage_dev,
                        self.srcinfo2_voltage_dev,
                        self.srcwaves_voltage_dev,
                        self.grid.ID_dev,
                        self.grid.Ex_dev,
                        self.grid.Ey_dev,
                        self.grid.Ez_dev)
            event.wait()
            self.compute_time += (event.profile.end - event.profile.start)*1e-9

        if self.grid.hertziandipoles:
            event = self.update_hertzian_dipole_dev(np.int32(len(self.grid.hertziandipoles)),
                        np.int32(self.grid.iteration),
                        config.sim_config.dtypes['float_or_double'](self.grid.dx),
                        config.sim_config.dtypes['float_or_double'](self.grid.dy),
                        config.sim_config.dtypes['float_or_double'](self.grid.dz),
                        self.srcinfo1_hertzian_dev,
                        self.srcinfo2_hertzian_dev,
                        self.srcwaves_hertzian_dev,
                        self.grid.ID_dev,
                        self.grid.Ex_dev,
                        self.grid.Ey_dev,
                        self.grid.Ez_dev)
            event.wait()
            self.compute_time += (event.profile.end - event.profile.start)*1e-9

        self.grid.iteration += 1

    def update_electric_b(self):
        """If there are any dispersive materials do 2nd part of dispersive
            update - it is split into two parts as it requires present and
            updated electric field values. Therefore it can only be completely
            updated after the electric field has been updated by the PML and
            source updates.
        """
        if config.get_model_config().materials['maxpoles'] > 0:
            event = self.dispersive_update_b(np.int32(self.grid.nx),
                        np.int32(self.grid.ny),
                        np.int32(self.grid.nz),
                        np.int32(config.get_model_config().materials['maxpoles']),
                        self.grid.updatecoeffsdispersive_dev,
                        self.grid.Tx_dev,
                        self.grid.Ty_dev,
                        self.grid.Tz_dev,
                        self.grid.ID_dev,
                        self.grid.Ex_dev,
                        self.grid.Ey_dev,
                        self.grid.Ez_dev)
            event.wait()
            self.compute_time += (event.profile.end - event.profile.start)*1e-9

    def time_start(self):
        self.event_marker1 = self.cl.enqueue_marker(self.queue)
        

    def calculate_memsolve(self, iteration):
        """Calculate memory used on last iteration.

        Args:
            iteration: int of iteration number.

        Returns:
            Memory (RAM) used on compute device.
        """
        # if iteration == self.grid.iterations - 1:
        #     return self.drv.mem_get_info()[1] - self.drv.mem_get_info()[0]
        logger.debug('Look at memory estimate for pyopencl')
        pass

    def calculate_tsolve(self):
        """Calculate solving time for model."""
        self.event_marker1.wait()
        event_marker2 = self.cl.enqueue_marker(self.queue)
        event_marker2.wait()
        compute_time = (event_marker2.profile.end - self.event_marker1.profile.end)*1e-9
        print(compute_time)
        print(self.compute_time)
        return self.compute_time

    def finalise(self):
        """Copy data from compute device back to CPU to save to file(s)."""
        # Copy output from receivers array back to correct receiver objects
        if self.grid.rxs:
            dtoh_rx_array(self.rxs_dev.get(), self.rxcoords_dev.get(), self.grid)

        # Copy data from any snapshots back to correct snapshot objects
        if self.grid.snapshots and not config.get_model_config().device['snapsgpu2cpu']:
            for i, snap in enumerate(self.grid.snapshots):
                dtoh_snapshot_array(self.snapEx_dev.get(),
                                    self.snapEy_dev.get(),
                                    self.snapEz_dev.get(),
                                    self.snapHx_dev.get(),
                                    self.snapHy_dev.get(),
                                    self.snapHz_dev.get(),
                                    i, snap)

    def cleanup(self):
        """Cleanup compute device context."""
        logger.debug('Check if pyopencl needs explicit cleanup.')
        # Remove context from top of stack and delete
        # self.ctx.pop()
        # del self.ctx
