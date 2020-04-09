# Copyright (C) 2015-2020: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

import gprMax.config as config
import numpy as np

from .cuda.fields_updates import kernel_template_fields
from .cuda.snapshots import kernel_template_store_snapshot
from .cuda.source_updates import kernel_template_sources
from .cython.fields_updates_normal import \
    update_electric as update_electric_cpu
from .cython.fields_updates_normal import \
    update_magnetic as update_magnetic_cpu
from .fields_outputs import kernel_template_store_outputs
from .fields_outputs import store_outputs as store_outputs_cpu
from .receivers import dtoh_rx_array, htod_rx_arrays
from .snapshots import Snapshot, dtoh_snapshot_array, htod_snapshot_array
from .sources import htod_src_arrays
from .utilities import human_size, round32, timer


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
                        config.sim_config.hostinfo['ompthreads'],
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
                                config.sim_config.hostinfo['ompthreads'],
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
                                     config.sim_config.hostinfo['ompthreads'],
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
                                     config.sim_config.hostinfo['ompthreads'],
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

        # Create device handle and context on specifc GPU device (and make it current context)
        self.dev = self.drv.Device(config.get_model_config().cuda['gpu'].deviceID)
        self.ctx = self.dev.make_context()

        # Initialise arrays on GPU, prepare kernels, and get kernel functions
        self._set_field_kernels()
        if self.grid.pmls:
            self._set_pml_kernels()
        if self.grid.rxs:
            self._set_rx_kernel()
        if self.grid.voltagesources + self.grid.hertziandipoles + self.grid.magneticdipoles:
            self._set_src_kernels()
        if self.grid.snapshots:
            self._set_snapshot_kernel()

    def _set_field_kernels(self):
        """Electric and magnetic field updates - prepare kernels, and
            get kernel functions.
        """
        if config.get_model_config().materials['maxpoles'] > 0:
            kernels_fields = self.source_module(kernel_template_fields.substitute(
                                                REAL=config.sim_config.dtypes['C_float_or_double'],
                                                REALFUNC=config.get_model_config().materials['cudarealfunc'],
                                                COMPLEX=config.get_model_config().materials['dispersiveCdtype'],
                                                N_updatecoeffsE=self.grid.updatecoeffsE.size,
                                                N_updatecoeffsH=self.grid.updatecoeffsH.size,
                                                NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1],
                                                NY_MATDISPCOEFFS=self.grid.updatecoeffsdispersive.shape[1],
                                                NX_FIELDS=self.grid.nx + 1,
                                                NY_FIELDS=self.grid.ny + 1,
                                                NZ_FIELDS=self.grid.nz + 1,
                                                NX_ID=self.grid.ID.shape[1],
                                                NY_ID=self.grid.ID.shape[2],
                                                NZ_ID=self.grid.ID.shape[3],
                                                NX_T=self.grid.Tx.shape[1],
                                                NY_T=self.grid.Tx.shape[2],
                                                NZ_T=self.grid.Tx.shape[3]),
                                                options=config.sim_config.cuda['nvcc_opts'])
        else: # Set to one any substitutions for dispersive materials.
              # Value of COMPLEX is not relevant.
            kernels_fields = self.source_module(kernel_template_fields.substitute(
                                                REAL=config.sim_config.dtypes['C_float_or_double'],
                                                REALFUNC=config.get_model_config().materials['cudarealfunc'],
                                                COMPLEX=config.sim_config.dtypes['C_float_or_double'],
                                                N_updatecoeffsE=self.grid.updatecoeffsE.size,
                                                N_updatecoeffsH=self.grid.updatecoeffsH.size,
                                                NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1],
                                                NY_MATDISPCOEFFS=1,
                                                NX_FIELDS=self.grid.nx + 1,
                                                NY_FIELDS=self.grid.ny + 1,
                                                NZ_FIELDS=self.grid.nz + 1,
                                                NX_ID=self.grid.ID.shape[1],
                                                NY_ID=self.grid.ID.shape[2],
                                                NZ_ID=self.grid.ID.shape[3],
                                                NX_T=1,
                                                NY_T=1,
                                                NZ_T=1),
                                                options=config.sim_config.cuda['nvcc_opts'])
        self.update_electric_gpu = kernels_fields.get_function("update_electric")
        self.update_magnetic_gpu = kernels_fields.get_function("update_magnetic")
        self._copy_mat_coeffs(kernels_fields, kernels_fields)

        # Electric and magnetic field updates - dispersive materials
        # - get kernel functions and initialise array on GPU
        # If there are any dispersive materials (updates are split into two
        # parts as they require present and updated electric field values).
        if config.get_model_config().materials['maxpoles'] > 0:
            self.dispersive_update_a = kernels_fields.get_function("update_electric_dispersive_A")
            self.dispersive_update_b = kernels_fields.get_function("update_electric_dispersive_B")

        # Electric and magnetic field updates - set blocks per grid and
        # initialise field arrays on GPU
        self.grid.set_blocks_per_grid()
        self.grid.htod_geometry_arrays()
        self.grid.htod_field_arrays()
        if config.get_model_config().materials['maxpoles'] > 0:
            self.grid.htod_dispersive_arrays()

    def _set_pml_kernels(self):
        """PMLS - prepare kernels and get kernel functions."""
        pmlmodulelectric = 'gprMax.cuda.pml_updates_electric_' + self.grid.pmlformulation
        kernelelectricfunc = getattr(import_module(pmlmodulelectric),
                                     'kernels_template_pml_electric_' +
                                     self.grid.pmlformulation)
        pmlmodulemagnetic = 'gprMax.cuda.pml_updates_magnetic_' + self.grid.pmlformulation
        kernelmagneticfunc = getattr(import_module(pmlmodulemagnetic),
                                     'kernels_template_pml_magnetic_' +
                                     self.grid.pmlformulation)
        kernels_pml_electric = self.source_module(kernelelectricfunc.substitute(
                                                  REAL=config.sim_config.dtypes['C_float_or_double'],
                                                  N_updatecoeffsE=self.grid.updatecoeffsE.size,
                                                  NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1],
                                                  NX_FIELDS=self.grid.nx + 1,
                                                  NY_FIELDS=self.grid.ny + 1,
                                                  NZ_FIELDS=self.grid.nz + 1,
                                                  NX_ID=self.grid.ID.shape[1],
                                                  NY_ID=self.grid.ID.shape[2],
                                                  NZ_ID=self.grid.ID.shape[3]),
                                                  options=config.sim_config.cuda['nvcc_opts'])
        kernels_pml_magnetic = self.source_module(kernelmagneticfunc.substitute(
                                                  REAL=config.sim_config.dtypes['C_float_or_double'],
                                                  N_updatecoeffsH=self.grid.updatecoeffsH.size,
                                                  NY_MATCOEFFS=self.grid.updatecoeffsH.shape[1],
                                                  NX_FIELDS=self.grid.nx + 1,
                                                  NY_FIELDS=self.grid.ny + 1,
                                                  NZ_FIELDS=self.grid.nz + 1,
                                                  NX_ID=self.grid.ID.shape[1],
                                                  NY_ID=self.grid.ID.shape[2],
                                                  NZ_ID=self.grid.ID.shape[3]),
                                                  options=config.sim_config.cuda['nvcc_opts'])
        self._copy_mat_coeffs(kernels_pml_electric, kernels_pml_magnetic)
        # Set block per grid, initialise arrays on GPU, and get kernel functions
        for pml in self.grid.pmls:
            pml.htod_field_arrays()
            pml.set_blocks_per_grid()
            pml.get_update_funcs(kernels_pml_electric, kernels_pml_magnetic)

    def _set_rx_kernel(self):
        """Receivers - initialise arrays on GPU, prepare kernel and get kernel
                        function.
        """
        self.rxcoords_gpu, self.rxs_gpu = htod_rx_arrays(self.grid)
        kernel_store_outputs = self.source_module(kernel_template_store_outputs.substitute(
                                                  REAL=config.sim_config.dtypes['C_float_or_double'],
                                                  NY_RXCOORDS=3,
                                                  NX_RXS=6,
                                                  NY_RXS=self.grid.iterations,
                                                  NZ_RXS=len(self.grid.rxs),
                                                  NX_FIELDS=self.grid.nx + 1,
                                                  NY_FIELDS=self.grid.ny + 1,
                                                  NZ_FIELDS=self.grid.nz + 1),
                                                  options=config.sim_config.cuda['nvcc_opts'])
        self.store_outputs_gpu = kernel_store_outputs.get_function("store_outputs")

    def _set_src_kernels(self):
        """Sources - initialise arrays on GPU, prepare kernel and get kernel
                        function.
        """
        kernels_sources = self.source_module(kernel_template_sources.substitute(
                                             REAL=config.sim_config.dtypes['C_float_or_double'],
                                             N_updatecoeffsE=self.grid.updatecoeffsE.size,
                                             N_updatecoeffsH=self.grid.updatecoeffsH.size,
                                             NY_MATCOEFFS=self.grid.updatecoeffsE.shape[1],
                                             NY_SRCINFO=4,
                                             NY_SRCWAVES=self.grid.iterations,
                                             NX_FIELDS=self.grid.nx + 1,
                                             NY_FIELDS=self.grid.ny + 1,
                                             NZ_FIELDS=self.grid.nz + 1,
                                             NX_ID=self.grid.ID.shape[1],
                                             NY_ID=self.grid.ID.shape[2],
                                             NZ_ID=self.grid.ID.shape[3]),
                                             options=config.sim_config.cuda['nvcc_opts'])
        self._copy_mat_coeffs(kernels_sources, kernels_sources)
        if self.grid.hertziandipoles:
            self.srcinfo1_hertzian_gpu, self.srcinfo2_hertzian_gpu, self.srcwaves_hertzian_gpu = htod_src_arrays(self.grid.hertziandipoles, self.grid)
            self.update_hertzian_dipole_gpu = kernels_sources.get_function("update_hertzian_dipole")
        if self.grid.magneticdipoles:
            self.srcinfo1_magnetic_gpu, self.srcinfo2_magnetic_gpu, self.srcwaves_magnetic_gpu = htod_src_arrays(self.grid.magneticdipoles, self.grid)
            self.update_magnetic_dipole_gpu = kernels_sources.get_function("update_magnetic_dipole")
        if self.grid.voltagesources:
            self.srcinfo1_voltage_gpu, self.srcinfo2_voltage_gpu, self.srcwaves_voltage_gpu = htod_src_arrays(self.grid.voltagesources, self.grid)
            self.update_voltage_source_gpu = kernels_sources.get_function("update_voltage_source")

    def _set_snapshot_kernel(self):
        """Snapshots - initialise arrays on GPU, prepare kernel and get kernel
                        function.
        """
        self.snapEx_gpu, self.snapEy_gpu, self.snapEz_gpu, self.snapHx_gpu, self.snapHy_gpu, self.snapHz_gpu = htod_snapshot_array(self.grid)
        kernel_store_snapshot = self.source_module(kernel_template_store_snapshot.substitute(
                                                   REAL=config.sim_config.dtypes['C_float_or_double'],
                                                   NX_SNAPS=Snapshot.nx_max,
                                                   NY_SNAPS=Snapshot.ny_max,
                                                   NZ_SNAPS=Snapshot.nz_max,
                                                   NX_FIELDS=self.grid.nx + 1,
                                                   NY_FIELDS=self.grid.ny + 1,
                                                   NZ_FIELDS=self.grid.nz + 1),
                                                   options=config.sim_config.cuda['nvcc_opts'])
        self.store_snapshot_gpu = kernel_store_snapshot.get_function("store_snapshot")

    def _copy_mat_coeffs(self, kernelE, kernelH):
        """Copy material coefficient arrays to constant memory of GPU
            (must be <64KB).

        Args:
            kernelE (kernel): electric field kernel.
            kernelH (kernel): magnetic field kernel.
        """

        # Check if coefficient arrays will fit on constant memory of GPU
        if (self.grid.updatecoeffsE.nbytes + self.grid.updatecoeffsH.nbytes
            > config.get_model_config().cuda['gpu'].constmem):
            logger.exception(f"Too many materials in the model to fit onto constant memory of size {human_size(config.get_model_config().cuda['gpu'].constmem)} on {config.get_model_config().cuda['gpu'].deviceID} - {config.get_model_config().cuda['gpu'].name} GPU")
            raise ValueError

        updatecoeffsE = kernelE.get_global('updatecoeffsE')[0]
        updatecoeffsH = kernelH.get_global('updatecoeffsH')[0]
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
