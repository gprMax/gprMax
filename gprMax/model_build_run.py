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

import datetime
from importlib import import_module
import itertools
import os
import psutil
import sys

from colorama import init
from colorama import Fore
from colorama import Style
init()
import cython
import numpy as np
from terminaltables import SingleTable
from tqdm import tqdm

import gprMax.config as config
from .cuda.fields_updates import kernel_template_fields
from .cuda.snapshots import kernel_template_store_snapshot
from .cuda.source_updates import kernel_template_sources
from .cython.yee_cell_build import build_electric_components
from .cython.yee_cell_build import build_magnetic_components
from .exceptions import GeneralError
from .fields_outputs import store_outputs
from .fields_outputs import kernel_template_store_outputs
from .fields_outputs import write_hdf5_outputfile
from .grid import FDTDGrid
from .grid import dispersion_analysis
from .input_cmds_geometry import process_geometrycmds
from .input_cmds_file import process_python_include_code
from .input_cmds_file import write_processed_file
from .input_cmds_file import check_cmd_names
from .input_cmds_singleuse import process_singlecmds
from .input_cmds_multiuse import process_multicmds
from .materials import Material
from .materials import process_materials
from .pml import CFS
from .pml import PML
from .pml import build_pml
from .pml import pml_information
from .receivers import gpu_initialise_rx_arrays
from .receivers import gpu_get_rx_array
from .receivers import Rx
from .snapshots import Snapshot
from .snapshots import gpu_initialise_snapshot_array
from .snapshots import gpu_get_snapshot_array
from .sources import gpu_initialise_src_arrays
from .utilities import get_terminal_width
from .utilities import human_size
from .utilities import open_path_file
from .utilities import round32
from .utilities import timer

class Printer():

    def __init__(self, sim_config):
        self.printing = sim_config.general['messages']

    def print(self, str):
        if self.printing:
            print(str)



def run_model(solver, sim_config, model_config):
    """Runs a model - processes the input file; builds the Yee cells; calculates update coefficients; runs main FDTD loop.

    Args:
        args (dict): Namespace with command line arguments
        currentmodelrun (int): Current model run number.
        modelend (int): Number of last model to run.
        numbermodelruns (int): Total number of model runs.
        inputfile (object): File object for the input file.
        usernamespace (dict): Namespace that can be accessed by user
                in any Python code blocks in input file.

    Returns:
        tsolve (int): Length of time (seconds) of main FDTD calculations
    """

    printer = Printer(sim_config)

    # Monitor memory usage
    p = psutil.Process()

    # Normal model reading/building process; bypassed if geometry information to be reused
    if not sim_config.geometry_fixed:

        printer.print(Fore.GREEN + '{} {}\n'.format(model_config.inputfilestr, '-' * (get_terminal_width() - 1 - len(model_config.inputfilestr))) + Style.RESET_ALL)

        G = solver.G

        # api for multiple scenes / model runs
        try:
            scene = model_config.scene
        # process using hashcommands
        except AttributeError:
            scene = Scene()
            # parse the input file into user objects and add them to the scene
            scene = parse_hash_commands(args, usernamespace, appendmodelnumber, G, scene)

        # Creates the internal simulation objects.
        scene.create_internal_objects(G)

        # print PML information
        printer.print(pml_information(G))

        # build the PMLS
        pbar = tqdm(total=sum(1 for value in G.pmlthickness.values() if value > 0), desc='Building PML boundaries', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])

        for pml_id, thickness in G.pmlthickness.items():
            build_pml(G, pml_id, thickness)
            pbar.update()
        pbar.close()

        # Build the model, i.e. set the material properties (ID) for every edge
        # of every Yee cell
        printer.print('')
        pbar = tqdm(total=2, desc='Building main grid', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
        build_electric_components(G.solid, G.rigidE, G.ID, G)
        pbar.update()
        build_magnetic_components(G.solid, G.rigidH, G.ID, G)
        pbar.update()
        pbar.close()

        # Add PEC boundaries to invariant direction in 2D modes
        # N.B. 2D modes are a single cell slice of 3D grid
        if '2D TMx' in config.mode:
            # Ey & Ez components
            G.ID[1, 0, :, :] = 0
            G.ID[1, 1, :, :] = 0
            G.ID[2, 0, :, :] = 0
            G.ID[2, 1, :, :] = 0
        elif '2D TMy' in config.mode:
            # Ex & Ez components
            G.ID[0, :, 0, :] = 0
            G.ID[0, :, 1, :] = 0
            G.ID[2, :, 0, :] = 0
            G.ID[2, :, 1, :] = 0
        elif '2D TMz' in config.mode:
            # Ex & Ey components
            G.ID[0, :, :, 0] = 0
            G.ID[0, :, :, 1] = 0
            G.ID[1, :, :, 0] = 0
            G.ID[1, :, :, 1] = 0

        # Process any voltage sources (that have resistance) to create a new
        # material at the source location
        for voltagesource in G.voltagesources:
            voltagesource.create_material(G)

        # Initialise arrays of update coefficients to pass to update functions
        G.initialise_std_update_coeff_arrays()

        # Set datatype for dispersive arrays if there are any dispersive materials.
        if config.materials['maxpoles'] != 0:
            drudelorentz = any([m for m in G.materials if 'drude' in m.type or 'lorentz' in m.type])
            if drudelorentz:
                config.materials['dispersivedtype'] = config.dtypes['complex']
                config.materials['dispersiveCdtype'] = config.dtypes['C_complex']
            else:
                config.materials['dispersivedtype'] = config.dtypes['float_or_double']
                config.materials['dispersiveCdtype'] = config.dtypes['C_float_or_double']

            # Update estimated memory (RAM) usage
            G.memoryusage += int(3 * config.materials['maxpoles'] * (G.nx + 1) * (G.ny + 1) * (G.nz + 1) * np.dtype(config.materials['dispersivedtype']).itemsize)
            G.memory_check()
            printer.print('\nMemory (RAM) required - updated (dispersive): ~{}\n'.format(human_size(G.memoryusage)))

            G.initialise_dispersive_arrays(config.materials['dispersivedtype'])

        # Check there is sufficient memory to store any snapshots
        if G.snapshots:
            snapsmemsize = 0
            for snap in G.snapshots:
                # 2 x required to account for electric and magnetic fields
                snapsmemsize += (2 * snap.datasizefield)
            G.memoryusage += int(snapsmemsize)
            G.memory_check(snapsmemsize=int(snapsmemsize))

            printer.print('\nMemory (RAM) required - updated (snapshots): ~{}\n'.format(human_size(G.memoryusage)))

        # Process complete list of materials - calculate update coefficients,
        # store in arrays, and build text list of materials/properties
        materialsdata = process_materials(G)
        materialstable = SingleTable(materialsdata)
        materialstable.outer_border = False
        materialstable.justify_columns[0] = 'right'

        printer.print('\nMaterials:')
        printer.print(materialstable.table)

        # Check to see if numerical dispersion might be a problem
        results = dispersion_analysis(G)
        if results['error']:
            printer.print(Fore.RED + "\nWARNING: Numerical dispersion analysis not carried out as {}".format(results['error']) + Style.RESET_ALL)
        elif results['N'] < config.numdispersion['mingridsampling']:
            raise GeneralError("Non-physical wave propagation: Material '{}' has wavelength sampled by {} cells, less than required minimum for physical wave propagation. Maximum significant frequency estimated as {:g}Hz".format(results['material'].ID, results['N'], results['maxfreq']))
        elif results['deltavp'] and np.abs(results['deltavp']) > config.numdispersion['maxnumericaldisp']:
            printer.print(Fore.RED + "\nWARNING: Potentially significant numerical dispersion. Estimated largest physical phase-velocity error is {:.2f}% in material '{}' whose wavelength sampled by {} cells. Maximum significant frequency estimated as {:g}Hz".format(results['deltavp'], results['material'].ID, results['N'], results['maxfreq']) + Style.RESET_ALL)
        elif results['deltavp']:
            printer.print("\nNumerical dispersion analysis: estimated largest physical phase-velocity error is {:.2f}% in material '{}' whose wavelength sampled by {} cells. Maximum significant frequency estimated as {:g}Hz".format(results['deltavp'], results['material'].ID, results['N'], results['maxfreq']))

        disp_a = update_f.format(model_config.poles, 'A', model_config.precision, model_config.dispersion_type)

        # set the dispersive update functions based on the model configuration
        solver.updates.set_dispersive_updates(model_config)


    # If geometry information to be reused between model runs
    else:
        inputfilestr = '\n--- Model {}/{}, input file (not re-processed, i.e. geometry fixed): {}'.format(currentmodelrun, modelend, inputfile.name)
        printer.print(Fore.GREEN + '{} {}\n'.format(inputfilestr, '-' * (get_terminal_width() - 1 - len(inputfilestr))) + Style.RESET_ALL)

        # Clear arrays for field components
        G.initialise_field_arrays()

        # Clear arrays for fields in PML
        for pml in G.pmls:
            pml.initialise_field_arrays()

    # Adjust position of simple sources and receivers if required
    if G.srcsteps[0] != 0 or G.srcsteps[1] != 0 or G.srcsteps[2] != 0:
        for source in itertools.chain(G.hertziandipoles, G.magneticdipoles):
            if currentmodelrun == 1:
                if source.xcoord + G.srcsteps[0] * modelend < 0 or source.xcoord + G.srcsteps[0] * modelend > G.nx or source.ycoord + G.srcsteps[1] * modelend < 0 or source.ycoord + G.srcsteps[1] * modelend > G.ny or source.zcoord + G.srcsteps[2] * modelend < 0 or source.zcoord + G.srcsteps[2] * modelend > G.nz:
                    raise GeneralError('Source(s) will be stepped to a position outside the domain.')
            source.xcoord = source.xcoordorigin + (currentmodelrun - 1) * G.srcsteps[0]
            source.ycoord = source.ycoordorigin + (currentmodelrun - 1) * G.srcsteps[1]
            source.zcoord = source.zcoordorigin + (currentmodelrun - 1) * G.srcsteps[2]
    if G.rxsteps[0] != 0 or G.rxsteps[1] != 0 or G.rxsteps[2] != 0:
        for receiver in G.rxs:
            if currentmodelrun == 1:
                if receiver.xcoord + G.rxsteps[0] * modelend < 0 or receiver.xcoord + G.rxsteps[0] * modelend > G.nx or receiver.ycoord + G.rxsteps[1] * modelend < 0 or receiver.ycoord + G.rxsteps[1] * modelend > G.ny or receiver.zcoord + G.rxsteps[2] * modelend < 0 or receiver.zcoord + G.rxsteps[2] * modelend > G.nz:
                    raise GeneralError('Receiver(s) will be stepped to a position outside the domain.')
            receiver.xcoord = receiver.xcoordorigin + (currentmodelrun - 1) * G.rxsteps[0]
            receiver.ycoord = receiver.ycoordorigin + (currentmodelrun - 1) * G.rxsteps[1]
            receiver.zcoord = receiver.zcoordorigin + (currentmodelrun - 1) * G.rxsteps[2]

    # Write files for any geometry views and geometry object outputs
    if not (G.geometryviews or G.geometryobjectswrite) and args.geometry_only and config.general['messages']:
        print(Fore.RED + '\nWARNING: No geometry views or geometry objects to output found.' + Style.RESET_ALL)
    if config.general['messages']: print()
    for i, geometryview in enumerate(G.geometryviews):
        geometryview.set_filename(appendmodelnumber)
        pbar = tqdm(total=geometryview.datawritesize, unit='byte', unit_scale=True, desc='Writing geometry view file {}/{}, {}'.format(i + 1, len(G.geometryviews), os.path.split(geometryview.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
        geometryview.write_vtk(G, pbar)
        pbar.close()
    for i, geometryobject in enumerate(G.geometryobjectswrite):
        pbar = tqdm(total=geometryobject.datawritesize, unit='byte', unit_scale=True, desc='Writing geometry object file {}/{}, {}'.format(i + 1, len(G.geometryobjectswrite), os.path.split(geometryobject.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
        geometryobject.write_hdf5(G, pbar)
        pbar.close()

    # If only writing geometry information
    if args.geometry_only:
        tsolve = 0

    # Run simulation
    else:
        # Check and set output directory and filename
        if not os.path.isdir(config.outputfilepath):
            os.mkdir(config.outputfilepath)
            printer.print('\nCreated output directory: {}'.format(config.outputfilepath))
        outputfile = os.path.join(config.outputfilepath, os.path.splitext(os.path.split(config.inputfilepath)[1])[0] + appendmodelnumber + '.out')
        printer.print('\nOutput file: {}\n'.format(outputfile))
        # Main FDTD solving functions for either CPU or GPU
        if config.cuda['gpus'] is None:
            tsolve = solve_cpu(currentmodelrun, modelend, G)
        else:
            tsolve, memsolve = solve_gpu(currentmodelrun, modelend, G)

        # Write an output file in HDF5 format
        write_hdf5_outputfile(outputfile, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G)

        # Write any snapshots to file
        if G.snapshots:
            # Create directory and construct filename from user-supplied name and model run number
            snapshotdir = os.path.splitext(config.inputfilepath)[0] + '_snaps' + appendmodelnumber
            if not os.path.exists(snapshotdir):
                os.mkdir(snapshotdir)

            if config.general['messages']: print()
            for i, snap in enumerate(G.snapshots):
                snap.filename = os.path.abspath(os.path.join(snapshotdir, snap.basefilename + '.vti'))
                pbar = tqdm(total=snap.vtkdatawritesize, leave=True, unit='byte', unit_scale=True, desc='Writing snapshot file {} of {}, {}'.format(i + 1, len(G.snapshots), os.path.split(snap.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
                snap.write_vtk_imagedata(pbar, G)
                pbar.close()
            if config.general['messages']: print()

        memGPU = ''
        if config.cuda['gpus']:
            memGPU = ' host + ~{} GPU'.format(human_size(memsolve))
        printer.print('\nMemory (RAM) used: ~{}{}'.format(human_size(p.memory_full_info().uss), memGPU))
        printer.print('Solving time [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=tsolve)))

    # If geometry information to be reused between model runs then FDTDGrid
    # class instance must be global so that it persists
    if not args.geometry_fixed:
        del G

    return tsolve


def solve_cpu(solver, sim_config, model_config):
    """
    Solving using FDTD method on CPU. Parallelised using Cython (OpenMP) for
    electric and magnetic field updates, and PML updates.

    Args:
        currentmodelrun (int): Current model run number.
        modelend (int): Number of last model to run.
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        tsolve (float): Time taken to execute solving (seconds)
    """
    tsolvestart = timer()

    solver.solve()

    return tsolve


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
