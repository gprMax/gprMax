# Copyright (C) 2015-2019: The University of Edinburgh
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

class CPUUpdates:

    def __init__(self, G):
        self.G = G
        self.dispersive_update_a = None
        self.dispersive_update_b = None

    def store_outputs(self, iteration):
        # Store field component values for every receiver and transmission line
        store_outputs(iteration,
                      self.G.Ex,
                      self.G.Ey,
                      self.G.Ez,
                      self.G.Hx,
                      self.G.Hy,
                      self.G.Hz,
                      self.G)

    def store_snapshots(self, iteration):
        # Store any snapshots
        for snap in self.G.snapshots:
            if snap.time == iteration + 1:
                snap.store(self.G)

    def update_magnetic(self):
        # Update magnetic field components
        update_magnetic(self.G.nx,
                        self.G.ny,
                        self.G.nz,
                        config.hostinfo['ompthreads'],
                        self.G.updatecoeffsH,
                        self.G.ID,
                        self.G.Ex,
                        self.G.Ey,
                        self.G.Ez,
                        self.G.Hx,
                        self.G.Hy,
                        self.G.Hz)

    def update_magnetic_pml(self, iteration):
        # Update magnetic field components with the PML correction
        for pml in self.G.pmls:
            pml.update_magnetic(self.G)

    def update_magnetic_sources(self, iteration):
        # Update magnetic field components from sources
        for source in self.G.transmissionlines + self.G.magneticdipoles:
            source.update_magnetic(iteration,
                                   self.G.updatecoeffsH,
                                   self.G.ID,
                                   self.G.Hx,
                                   self.G.Hy,
                                   self.G.Hz,
                                   self.G)

    def update_electric_a(self):
        # Update electric field components
        # All materials are non-dispersive so do standard update
        if Material.maxpoles == 0:
            update_electric(self.G.nx,
                            self.G.ny,
                            self.G.nz,
                            config.hostinfo['ompthreads'],
                            self.G.updatecoeffsE,
                            self.G.ID,
                            self.G.Ex,
                            self.G.Ey,
                            self.G.Ez,
                            self.G.Hx,
                            self.G.Hy,
                            self.G.Hz)

        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        self.dispersive_update_a(self.G.nx,
                                   self.G.ny,
                                   self.G.nz,
                                   config.hostinfo['ompthreads'],
                                   self.G.updatecoeffsE,
                                   self.G.updatecoeffsdispersive,
                                   self.G.ID,
                                   self.G.Tx,
                                   self.G.Ty,
                                   self.G.Tz,
                                   self.G.Ex,
                                   self.G.Ey,
                                   self.G.Ez,
                                   self.G.Hx,
                                   self.G.Hy,
                                   self.G.Hz)

    def update_electric_pml(self):
        # Update electric field components with the PML correction
        for pml in self.G.pmls:
            pml.update_electric(self.G)

    def update_electric_sources(self, iteration):
        # Update electric field components from sources (update any Hertzian dipole sources last)
        for source in self.G.voltagesources + self.G.transmissionlines + self.G.hertziandipoles:
            source.update_electric(iteration, self.G.updatecoeffsE, self.G.ID, self.G.Ex, self.G.Ey, self.G.Ez, self.G)

    def update_electric_b(self):
        # If there are any dispersive materials do 2nd part of dispersive update
        # (it is split into two parts as it requires present and updated electric
        # field values). Therefore it can only be completely updated after the
        # electric field has been updated by the PML and source updates.
        update_e_dispersive_b(self.G.nx,
                            self.G.ny,
                            self.G.nz,
                            config.hostinfo['ompthreads'],
                            Material.maxpoles,
                            self.G.updatecoeffsdispersive,
                            self.G.ID,
                            self.G.Tx,
                            self.G.Ty,
                            self.G.Tz,
                            self.G.Ex,
                            self.G.Ey,
                            self.G.Ez)

    def set_dispersive_updates(self, model_config):
        """Function to set dispersive update functions based on model."""
        update_f = 'update_electric_dispersive_{}pole_{}_{}_{}'
        disp_a = update_f.format(model_config.poles, 'A', model_config.precision, model_config.dispersion_type)
        disp_b = update_f.format(model_config.poles, 'B', model_config.precision, model_config.dispersion_type)

        disp_a_f = getattr(import_module('.cython.fields_updates_dispersive'), disp_a)
        disp_b_f = getattr(import_module('.cython.fields_updates_dispersive'), disp_b)

        self.dispersive_update_a = disp_a_f
        self.dispersive_update_b = disp_b_f


class SubgridUpdates:
    pass

class GPUUpdates:
    pass

"""
def solve_gpu(currentmodelrun, modelend, G):
    """Solving using FDTD method on GPU. Implemented using Nvidia CUDA.

    Args:
        currentmodelrun (int): Current model run number.
        modelend (int): Number of last model to run.
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        tsolve (float): Time taken to execute solving (seconds)
        memsolve (int): memory usage on final iteration (bytes)
    """

    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    drv.init()

    # Suppress nvcc warnings on Windows
    if sys.platform == 'win32':
        compiler_opts = ['-w']
    else:
        compiler_opts = None

    # Create device handle and context on specifc GPU device (and make it current context)
    dev = drv.Device(config.cuda['gpus'].deviceID)
    ctx = dev.make_context()

    # Electric and magnetic field updates - prepare kernels, get kernel functions, and initialise arrays on GPU
    if config.materials['maxpoles'] == 0:
        kernel_fields = SourceModule(kernel_template_fields.substitute(REAL=config.dtypes['C_float_or_double'], REAL_OR_COMPLEX=config.dtypes['C_complex'], N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=1, NX_FIELDS=G.Ex.shape[0], NY_FIELDS=G.Ex.shape[1], NZ_FIELDS=G.Ex.shape[2], NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3], NX_T=1, NY_T=1, NZ_T=1), options=compiler_opts)

    else:   # # If there are any dispersive materials (updates are split into two parts as they require present and updated electric field values).
        kernel_fields = SourceModule(kernel_template_fields.substitute(REAL=config.dtypes['C_float_or_double'], REAL_OR_COMPLEX=config.dtypes['C_complex'], N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=G.updatecoeffsdispersive.shape[1], NX_FIELDS=G.Ex.shape[0], NY_FIELDS=G.Ex.shape[1], NZ_FIELDS=G.Ex.shape[2], NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3], NX_T=G.Tx.shape[1], NY_T=G.Tx.shape[2], NZ_T=G.Tx.shape[3]), options=compiler_opts)
        update_e_dispersive_A_gpu = kernel_fields.get_function("update_e_dispersive_A")
        update_e_dispersive_B_gpu = kernel_fields.get_function("update_e_dispersive_B")
        G.gpu_initialise_dispersive_arrays()
    update_e_gpu = kernel_fields.get_function("update_e")
    update_h_gpu = kernel_fields.get_function("update_h")

    # Copy material coefficient arrays to constant memory of GPU (must be <64KB) for fields kernels
    updatecoeffsE = kernel_fields.get_global('updatecoeffsE')[0]
    updatecoeffsH = kernel_fields.get_global('updatecoeffsH')[0]
    if G.updatecoeffsE.nbytes + G.updatecoeffsH.nbytes > config.cuda['gpus'].constmem:
        raise GeneralError('Too many materials in the model to fit onto constant memory of size {} on {} - {} GPU'.format(human_size(config.cuda['gpus'].constmem), config.cuda['gpus'].deviceID, config.cuda['gpus'].name))
    else:
        drv.memcpy_htod(updatecoeffsE, G.updatecoeffsE)
        drv.memcpy_htod(updatecoeffsH, G.updatecoeffsH)

    # Electric and magnetic field updates - set blocks per grid and initialise field arrays on GPU
    G.gpu_set_blocks_per_grid()
    G.gpu_initialise_arrays()

    # PML updates
    if G.pmls:
        # Prepare kernels
        kernelelectricfunc = getattr(import_module('gprMax.cuda.pml_updates_electric_' + G.pmlformulation), 'kernels_template_pml_electric_' + G.pmlformulation)
        kernelmagneticfunc = getattr(import_module('gprMax.cuda.pml_updates_magnetic_' + G.pmlformulation), 'kernels_template_pml_magnetic_' + G.pmlformulation)
        kernels_pml_electric = SourceModule(kernelelectricfunc.substitute(REAL=config.dtypes['C_float_or_double'], N_updatecoeffsE=G.updatecoeffsE.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NX_FIELDS=G.Ex.shape[0], NY_FIELDS=G.Ex.shape[1], NZ_FIELDS=G.Ex.shape[2], NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3]), options=compiler_opts)
        kernels_pml_magnetic = SourceModule(kernelmagneticfunc.substitute(REAL=config.dtypes['C_float_or_double'], N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsH.shape[1], NX_FIELDS=G.Hx.shape[0], NY_FIELDS=G.Hx.shape[1], NZ_FIELDS=G.Hx.shape[2], NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3]), options=compiler_opts)
        # Copy material coefficient arrays to constant memory of GPU (must be <64KB) for PML kernels
        updatecoeffsE = kernels_pml_electric.get_global('updatecoeffsE')[0]
        updatecoeffsH = kernels_pml_magnetic.get_global('updatecoeffsH')[0]
        drv.memcpy_htod(updatecoeffsE, G.updatecoeffsE)
        drv.memcpy_htod(updatecoeffsH, G.updatecoeffsH)
        # Set block per grid, initialise arrays on GPU, and get kernel functions
        for pml in G.pmls:
            pml.gpu_set_blocks_per_grid(G)
            pml.gpu_initialise_arrays()
            pml.gpu_get_update_funcs(kernels_pml_electric, kernels_pml_magnetic)

    # Receivers
    if G.rxs:
        # Initialise arrays on GPU
        rxcoords_gpu, rxs_gpu = gpu_initialise_rx_arrays(G)
        # Prepare kernel and get kernel function

        kernel_store_outputs = SourceModule(kernel_template_store_outputs.substitute(REAL=config.dtypes['C_float_or_double'], NY_RXCOORDS=3, NX_RXS=len(Rx.gpu_allowableoutputs), NY_RXS=G.iterations, NZ_RXS=len(G.rxs), NX_FIELDS=G.Ex.shape[0], NY_FIELDS=G.Ex.shape[1], NZ_FIELDS=G.Ex.shape[2]), options=compiler_opts)
        store_outputs_gpu = kernel_store_outputs.get_function("store_outputs")

    # Sources - initialise arrays on GPU, prepare kernel and get kernel functions
    if G.voltagesources + G.hertziandipoles + G.magneticdipoles:
        kernel_sources = SourceModule(kernel_template_sources.substitute(REAL=config.dtypes['C_float_or_double'], N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_SRCINFO=4, NY_SRCWAVES=G.iterations, NX_FIELDS=G.Ex.shape[0], NY_FIELDS=G.Ex.shape[1], NZ_FIELDS=G.Ex.shape[2], NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3]), options=compiler_opts)
        # Copy material coefficient arrays to constant memory of GPU (must be <64KB) for source kernels
        updatecoeffsE = kernel_sources.get_global('updatecoeffsE')[0]
        updatecoeffsH = kernel_sources.get_global('updatecoeffsH')[0]
        drv.memcpy_htod(updatecoeffsE, G.updatecoeffsE)
        drv.memcpy_htod(updatecoeffsH, G.updatecoeffsH)
        if G.hertziandipoles:
            srcinfo1_hertzian_gpu, srcinfo2_hertzian_gpu, srcwaves_hertzian_gpu = gpu_initialise_src_arrays(G.hertziandipoles, G)
            update_hertzian_dipole_gpu = kernel_sources.get_function("update_hertzian_dipole")
        if G.magneticdipoles:
            srcinfo1_magnetic_gpu, srcinfo2_magnetic_gpu, srcwaves_magnetic_gpu = gpu_initialise_src_arrays(G.magneticdipoles, G)
            update_magnetic_dipole_gpu = kernel_sources.get_function("update_magnetic_dipole")
        if G.voltagesources:
            srcinfo1_voltage_gpu, srcinfo2_voltage_gpu, srcwaves_voltage_gpu = gpu_initialise_src_arrays(G.voltagesources, G)
            update_voltage_source_gpu = kernel_sources.get_function("update_voltage_source")

    # Snapshots - initialise arrays on GPU, prepare kernel and get kernel functions
    if G.snapshots:
        # Initialise arrays on GPU
        snapEx_gpu, snapEy_gpu, snapEz_gpu, snapHx_gpu, snapHy_gpu, snapHz_gpu = gpu_initialise_snapshot_array(G)
        # Prepare kernel and get kernel function
        kernel_store_snapshot = SourceModule(kernel_template_store_snapshot.substitute(REAL=config.dtypes['C_float_or_double'], NX_SNAPS=Snapshot.nx_max, NY_SNAPS=Snapshot.ny_max, NZ_SNAPS=Snapshot.nz_max, NX_FIELDS=G.Ex.shape[0], NY_FIELDS=G.Ex.shape[1], NZ_FIELDS=G.Ex.shape[2]), options=compiler_opts)
        store_snapshot_gpu = kernel_store_snapshot.get_function("store_snapshot")

    # Iteration loop timer
    iterstart = drv.Event()
    iterend = drv.Event()
    iterstart.record()

    for iteration in tqdm(range(G.iterations), desc='Running simulation, model ' + str(currentmodelrun) + '/' + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars']):

        # Get GPU memory usage on final iteration
        if iteration == G.iterations - 1:
            memsolve = drv.mem_get_info()[1] - drv.mem_get_info()[0]

        # Store field component values for every receiver
        if G.rxs:
            store_outputs_gpu(np.int32(len(G.rxs)), np.int32(iteration),
                              rxcoords_gpu.gpudata, rxs_gpu.gpudata,
                              G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                              G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                              block=(1, 1, 1), grid=(round32(len(G.rxs)), 1, 1))

        # Store any snapshots
        for i, snap in enumerate(G.snapshots):
            if snap.time == iteration + 1:
                if not config.cuda['snapsgpu2cpu']:
                    store_snapshot_gpu(np.int32(i), np.int32(snap.xs),
                                       np.int32(snap.xf), np.int32(snap.ys),
                                       np.int32(snap.yf), np.int32(snap.zs),
                                       np.int32(snap.zf), np.int32(snap.dx),
                                       np.int32(snap.dy), np.int32(snap.dz),
                                       G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                       G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                       snapEx_gpu.gpudata, snapEy_gpu.gpudata, snapEz_gpu.gpudata,
                                       snapHx_gpu.gpudata, snapHy_gpu.gpudata, snapHz_gpu.gpudata,
                                       block=Snapshot.tpb, grid=Snapshot.bpg)
                else:
                    store_snapshot_gpu(np.int32(0), np.int32(snap.xs),
                                       np.int32(snap.xf), np.int32(snap.ys),
                                       np.int32(snap.yf), np.int32(snap.zs),
                                       np.int32(snap.zf), np.int32(snap.dx),
                                       np.int32(snap.dy), np.int32(snap.dz),
                                       G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                       G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                       snapEx_gpu.gpudata, snapEy_gpu.gpudata, snapEz_gpu.gpudata,
                                       snapHx_gpu.gpudata, snapHy_gpu.gpudata, snapHz_gpu.gpudata,
                                       block=Snapshot.tpb, grid=Snapshot.bpg)
                    gpu_get_snapshot_array(snapEx_gpu.get(), snapEy_gpu.get(), snapEz_gpu.get(),
                                           snapHx_gpu.get(), snapHy_gpu.get(), snapHz_gpu.get(), 0, snap)

        # Update magnetic field components
        update_h_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz),
                     G.ID_gpu.gpudata, G.Hx_gpu.gpudata, G.Hy_gpu.gpudata,
                     G.Hz_gpu.gpudata, G.Ex_gpu.gpudata, G.Ey_gpu.gpudata,
                     G.Ez_gpu.gpudata, block=config.cuda['gpus'].tpb, grid=config.cuda['gpus'].bpg)

        # Update magnetic field components with the PML correction
        for pml in G.pmls:
            pml.gpu_update_magnetic(G)

        # Update magnetic field components for magetic dipole sources
        if G.magneticdipoles:
            update_magnetic_dipole_gpu(np.int32(len(G.magneticdipoles)), np.int32(iteration),
                                       config.dtypes['float_or_double'](G.dx), config.dtypes['float_or_double'](G.dy), config.dtypes['float_or_double'](G.dz),
                                       srcinfo1_magnetic_gpu.gpudata, srcinfo2_magnetic_gpu.gpudata,
                                       srcwaves_magnetic_gpu.gpudata, G.ID_gpu.gpudata,
                                       G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                       block=(1, 1, 1), grid=(round32(len(G.magneticdipoles)), 1, 1))

        # Update electric field components
        # If all materials are non-dispersive do standard update
        if config.materials['maxpoles'] == 0:
            update_e_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz), G.ID_gpu.gpudata,
                         G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                         G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                         block=config.cuda['gpus'].tpb, grid=config.cuda['gpus'].bpg)
        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            update_e_dispersive_A_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz),
                                      np.int32(config.materials['maxpoles']), G.updatecoeffsdispersive_gpu.gpudata,
                                      G.Tx_gpu.gpudata, G.Ty_gpu.gpudata, G.Tz_gpu.gpudata, G.ID_gpu.gpudata,
                                      G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                      G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                      block=config.cuda['gpus'].tpb, grid=config.cuda['gpus'].bpg)

        # Update electric field components with the PML correction
        for pml in G.pmls:
            pml.gpu_update_electric(G)

        # Update electric field components for voltage sources
        if G.voltagesources:
            update_voltage_source_gpu(np.int32(len(G.voltagesources)), np.int32(iteration),
                                      config.dtypes['float_or_double'](G.dx), config.dtypes['float_or_double'](G.dy), config.dtypes['float_or_double'](G.dz),
                                      srcinfo1_voltage_gpu.gpudata, srcinfo2_voltage_gpu.gpudata,
                                      srcwaves_voltage_gpu.gpudata, G.ID_gpu.gpudata,
                                      G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                      block=(1, 1, 1), grid=(round32(len(G.voltagesources)), 1, 1))

        # Update electric field components for Hertzian dipole sources (update any Hertzian dipole sources last)
        if G.hertziandipoles:
            update_hertzian_dipole_gpu(np.int32(len(G.hertziandipoles)), np.int32(iteration),
                                       config.dtypes['float_or_double'](G.dx), config.dtypes['float_or_double'](G.dy), config.dtypes['float_or_double'](G.dz),
                                       srcinfo1_hertzian_gpu.gpudata, srcinfo2_hertzian_gpu.gpudata,
                                       srcwaves_hertzian_gpu.gpudata, G.ID_gpu.gpudata,
                                       G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                       block=(1, 1, 1), grid=(round32(len(G.hertziandipoles)), 1, 1))

        # If there are any dispersive materials do 2nd part of dispersive update (it is split into two parts as it requires present and updated electric field values). Therefore it can only be completely updated after the electric field has been updated by the PML and source updates.
        if config.materials['maxpoles'] > 0:
            update_e_dispersive_B_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz),
                                      np.int32(config.materials['maxpoles']), G.updatecoeffsdispersive_gpu.gpudata,
                                      G.Tx_gpu.gpudata, G.Ty_gpu.gpudata, G.Tz_gpu.gpudata, G.ID_gpu.gpudata,
                                      G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                      block=config.cuda['gpus'].tpb, grid=config.cuda['gpus'].bpg)

    # Copy output from receivers array back to correct receiver objects
    if G.rxs:
        gpu_get_rx_array(rxs_gpu.get(), rxcoords_gpu.get(), G)

    # Copy data from any snapshots back to correct snapshot objects
    if G.snapshots and not config.cuda['snapsgpu2cpu']:
        for i, snap in enumerate(G.snapshots):
            gpu_get_snapshot_array(snapEx_gpu.get(), snapEy_gpu.get(), snapEz_gpu.get(),
                                   snapHx_gpu.get(), snapHy_gpu.get(), snapHz_gpu.get(), i, snap)

    iterend.record()
    iterend.synchronize()
    tsolve = iterstart.time_till(iterend) * 1e-3

    # Remove context from top of stack and delete
    ctx.pop()
    del ctx

    return tsolve, memsolve
"""
