# Copyright (C) 2015-2023: The University of Edinburgh
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
import numpy as np
from terminaltables import AsciiTable
from tqdm import tqdm

from gprMax.constants import floattype
from gprMax.constants import complextype
from gprMax.constants import cudafloattype
from gprMax.constants import cudacomplextype
from gprMax.exceptions import GeneralError

from gprMax.fields_outputs import store_outputs
from gprMax.fields_outputs import kernel_template_store_outputs
from gprMax.fields_outputs import write_hdf5_outputfile

from gprMax.fields_updates_ext import update_electric
from gprMax.fields_updates_ext import update_magnetic
from gprMax.fields_updates_ext import update_electric_dispersive_multipole_A
from gprMax.fields_updates_ext import update_electric_dispersive_multipole_B
from gprMax.fields_updates_ext import update_electric_dispersive_1pole_A
from gprMax.fields_updates_ext import update_electric_dispersive_1pole_B
from gprMax.fields_updates_gpu import kernels_template_fields

from gprMax.grid import FDTDGrid
from gprMax.grid import dispersion_analysis

from gprMax.input_cmds_geometry import process_geometrycmds
from gprMax.input_cmds_file import process_python_include_code
from gprMax.input_cmds_file import write_processed_file
from gprMax.input_cmds_file import check_cmd_names
from gprMax.input_cmds_multiuse import process_multicmds
from gprMax.input_cmds_singleuse import process_singlecmds
from gprMax.materials import Material
from gprMax.materials import process_materials
from gprMax.pml import CFS
from gprMax.pml import PML
from gprMax.pml import build_pmls
from gprMax.receivers import gpu_initialise_rx_arrays
from gprMax.receivers import gpu_get_rx_array
from gprMax.snapshots import Snapshot
from gprMax.snapshots import gpu_initialise_snapshot_array
from gprMax.snapshots import gpu_get_snapshot_array
from gprMax.snapshots_gpu import kernel_template_store_snapshot
from gprMax.sources import gpu_initialise_src_arrays
from gprMax.source_updates_gpu import kernels_template_sources
from gprMax.utilities import get_host_info
from gprMax.utilities import get_terminal_width
from gprMax.utilities import human_size
from gprMax.utilities import open_path_file
from gprMax.utilities import round32
from gprMax.utilities import timer
from gprMax.yee_cell_build_ext import build_electric_components
from gprMax.yee_cell_build_ext import build_magnetic_components


def run_model(args, currentmodelrun, modelend, numbermodelruns, inputfile, usernamespace):
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

    # Monitor memory usage
    p = psutil.Process()

    # Declare variable to hold FDTDGrid class
    global G

    # Used for naming geometry and output files
    appendmodelnumber = '' if numbermodelruns == 1 and not args.task and not args.restart else str(currentmodelrun)

    # Normal model reading/building process; bypassed if geometry information to be reused
    if 'G' not in globals():

        # Initialise an instance of the FDTDGrid class
        G = FDTDGrid()

        # Get information about host machine
        # (need to save this info to FDTDGrid instance after it has been created)
        G.hostinfo = get_host_info()

        # Single GPU object
        if args.gpu:
            G.gpu = args.gpu

        G.inputfilename = os.path.split(inputfile.name)[1]
        G.inputdirectory = os.path.dirname(os.path.abspath(inputfile.name))
        inputfilestr = '\n--- Model {}/{}, input file: {}'.format(currentmodelrun, modelend, inputfile.name)
        if G.messages:
            print(Fore.GREEN + '{} {}\n'.format(inputfilestr, '-' * (get_terminal_width() - 1 - len(inputfilestr))) + Style.RESET_ALL)

        # Add the current model run to namespace that can be accessed by
        # user in any Python code blocks in input file
        usernamespace['current_model_run'] = currentmodelrun

        # Read input file and process any Python and include file commands
        processedlines = process_python_include_code(inputfile, usernamespace)

        # Print constants/variables in user-accessable namespace
        uservars = ''
        for key, value in sorted(usernamespace.items()):
            if key != '__builtins__':
                uservars += '{}: {}, '.format(key, value)
        if G.messages:
            print('Constants/variables used/available for Python scripting: {{{}}}\n'.format(uservars[:-2]))

        # Write a file containing the input commands after Python or include file commands have been processed
        if args.write_processed:
            write_processed_file(processedlines, appendmodelnumber, G)

        # Check validity of command names and that essential commands are present
        singlecmds, multicmds, geometry = check_cmd_names(processedlines)

        # Create built-in materials
        m = Material(0, 'pec')
        m.se = float('inf')
        m.type = 'builtin'
        m.averagable = False
        G.materials.append(m)
        m = Material(1, 'free_space')
        m.type = 'builtin'
        G.materials.append(m)

        # Process parameters for commands that can only occur once in the model
        process_singlecmds(singlecmds, G)

        # Process parameters for commands that can occur multiple times in the model
        if G.messages: print()
        process_multicmds(multicmds, G)

        # Estimate and check memory (RAM) usage
        G.memory_estimate_basic()
        G.memory_check()
        if G.messages:
            if G.gpu is None:
                print('\nMemory (RAM) required: ~{}\n'.format(human_size(G.memoryusage)))
            else:
                print('\nMemory (RAM) required: ~{} host + ~{} GPU\n'.format(human_size(G.memoryusage), human_size(G.memoryusage)))

        # Initialise an array for volumetric material IDs (solid), boolean
        # arrays for specifying materials not to be averaged (rigid),
        # an array for cell edge IDs (ID)
        G.initialise_geometry_arrays()

        # Initialise arrays for the field components
        if G.gpu is None:
            G.initialise_field_arrays()

        # Process geometry commands in the order they were given
        process_geometrycmds(geometry, G)

        # Build the PMLs and calculate initial coefficients
        if G.messages: print()
        if all(value == 0 for value in G.pmlthickness.values()):
            if G.messages:
                print('PML: switched off')
            pass  # If all the PMLs are switched off don't need to build anything
        else:
            # Set default CFS parameters for PML if not given
            if not G.cfs:
                G.cfs = [CFS()]
            if G.messages:
                if all(value == G.pmlthickness['x0'] for value in G.pmlthickness.values()):
                    pmlinfo = str(G.pmlthickness['x0'])
                else:
                    pmlinfo = ''
                    for key, value in G.pmlthickness.items():
                        pmlinfo += '{}: {}, '.format(key, value)
                    pmlinfo = pmlinfo[:-2] + ' cells'
                print('PML: formulation: {}, order: {}, thickness: {}'.format(G.pmlformulation, len(G.cfs), pmlinfo))
            pbar = tqdm(total=sum(1 for value in G.pmlthickness.values() if value > 0), desc='Building PML boundaries', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars)
            build_pmls(G, pbar)
            pbar.close()

        # Build the model, i.e. set the material properties (ID) for every edge
        # of every Yee cell
        if G.messages: print()
        pbar = tqdm(total=2, desc='Building main grid', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars)
        build_electric_components(G.solid, G.rigidE, G.ID, G)
        pbar.update()
        build_magnetic_components(G.solid, G.rigidH, G.ID, G)
        pbar.update()
        pbar.close()

        # Add PEC boundaries to invariant direction in 2D modes
        # N.B. 2D modes are a single cell slice of 3D grid
        if '2D TMx' in G.mode:
            # Ey & Ez components
            G.ID[1, 0, :, :] = 0
            G.ID[1, 1, :, :] = 0
            G.ID[2, 0, :, :] = 0
            G.ID[2, 1, :, :] = 0
        elif '2D TMy' in G.mode:
            # Ex & Ez components
            G.ID[0, :, 0, :] = 0
            G.ID[0, :, 1, :] = 0
            G.ID[2, :, 0, :] = 0
            G.ID[2, :, 1, :] = 0
        elif '2D TMz' in G.mode:
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

        # Initialise arrays of update coefficients and temporary values if
        # there are any dispersive materials
        if Material.maxpoles != 0:
            # Update estimated memory (RAM) usage
            G.memoryusage += int(3 * Material.maxpoles * (G.nx + 1) * (G.ny + 1) * (G.nz + 1) * np.dtype(complextype).itemsize)
            G.memory_check()
            if G.messages:
                print('\nMemory (RAM) required - updated (dispersive): ~{}\n'.format(human_size(G.memoryusage)))

            G.initialise_dispersive_arrays()

        # Check there is sufficient memory to store any snapshots
        if G.snapshots:
            snapsmemsize = 0
            for snap in G.snapshots:
                # 2 x required to account for electric and magnetic fields
                snapsmemsize += (2 * snap.datasizefield)
            G.memoryusage += int(snapsmemsize)
            G.memory_check(snapsmemsize=int(snapsmemsize))
            if G.messages:
                print('\nMemory (RAM) required - updated (snapshots): ~{}\n'.format(human_size(G.memoryusage)))

        # Process complete list of materials - calculate update coefficients,
        # store in arrays, and build text list of materials/properties
        materialsdata = process_materials(G)
        if G.messages:
            print('\nMaterials:')
            materialstable = AsciiTable(materialsdata)
            materialstable.outer_border = False
            materialstable.justify_columns[0] = 'right'
            print(materialstable.table)

        # Check to see if numerical dispersion might be a problem
        results = dispersion_analysis(G)
        if results['error'] and G.messages:
            print(Fore.RED + "\nWARNING: Numerical dispersion analysis not carried out as {}".format(results['error']) + Style.RESET_ALL)
        elif results['N'] < G.mingridsampling:
            raise GeneralError("Non-physical wave propagation: Material '{}' has wavelength sampled by {} cells, less than required minimum for physical wave propagation. Maximum significant frequency estimated as {:g}Hz".format(results['material'].ID, results['N'], results['maxfreq']))
        elif results['deltavp'] and np.abs(results['deltavp']) > G.maxnumericaldisp and G.messages:
            print(Fore.RED + "\nWARNING: Potentially significant numerical dispersion. Estimated largest physical phase-velocity error is {:.2f}% in material '{}' whose wavelength sampled by {} cells. Maximum significant frequency estimated as {:g}Hz".format(results['deltavp'], results['material'].ID, results['N'], results['maxfreq']) + Style.RESET_ALL)
        elif results['deltavp'] and G.messages:
            print("\nNumerical dispersion analysis: estimated largest physical phase-velocity error is {:.2f}% in material '{}' whose wavelength sampled by {} cells. Maximum significant frequency estimated as {:g}Hz".format(results['deltavp'], results['material'].ID, results['N'], results['maxfreq']))

    # If geometry information to be reused between model runs
    else:
        inputfilestr = '\n--- Model {}/{}, input file (not re-processed, i.e. geometry fixed): {}'.format(currentmodelrun, modelend, inputfile.name)
        if G.messages:
            print(Fore.GREEN + '{} {}\n'.format(inputfilestr, '-' * (get_terminal_width() - 1 - len(inputfilestr))) + Style.RESET_ALL)

        if G.gpu is None:
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
    if not (G.geometryviews or G.geometryobjectswrite) and args.geometry_only and G.messages:
        print(Fore.RED + '\nWARNING: No geometry views or geometry objects to output found.' + Style.RESET_ALL)
    if G.geometryviews:
        if G.messages: print()
        for i, geometryview in enumerate(G.geometryviews):
            geometryview.set_filename(appendmodelnumber, G)
            pbar = tqdm(total=geometryview.datawritesize, unit='byte', unit_scale=True, desc='Writing geometry view file {}/{}, {}'.format(i + 1, len(G.geometryviews), os.path.split(geometryview.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars)
            geometryview.write_vtk(G, pbar)
            pbar.close()
    if G.geometryobjectswrite:
        for i, geometryobject in enumerate(G.geometryobjectswrite):
            pbar = tqdm(total=geometryobject.datawritesize, unit='byte', unit_scale=True, desc='Writing geometry object file {}/{}, {}'.format(i + 1, len(G.geometryobjectswrite), os.path.split(geometryobject.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars)
            geometryobject.write_hdf5(G, pbar)
            pbar.close()

    # If only writing geometry information
    if args.geometry_only:
        tsolve = 0

    # Run simulation
    else:
        # Output filename
        inputdirectory, inputfilename = os.path.split(os.path.join(G.inputdirectory, G.inputfilename))
        if G.outputdirectory is None:
            outputdir = inputdirectory
        else:
            outputdir = G.outputdirectory
        # Save current directory
        curdir = os.getcwd()
        os.chdir(inputdirectory)
        outputdir = os.path.abspath(outputdir)
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)
            if G.messages:
                print('\nCreated output directory: {}'.format(outputdir))
        # Restore current directory
        os.chdir(curdir)
        basename, ext = os.path.splitext(inputfilename)
        outputfile = os.path.join(outputdir, basename + appendmodelnumber + '.out')
        if G.messages:
            print('\nOutput file: {}\n'.format(outputfile))

        # Main FDTD solving functions for either CPU or GPU
        if G.gpu is None:
            tsolve = solve_cpu(currentmodelrun, modelend, G)
        else:
            tsolve, memsolve = solve_gpu(currentmodelrun, modelend, G)

        # Write an output file in HDF5 format
        write_hdf5_outputfile(outputfile, G)

        # Write any snapshots to file
        if G.snapshots:
            # Create directory and construct filename from user-supplied name and model run number
            snapshotdir = os.path.join(G.inputdirectory, os.path.splitext(G.inputfilename)[0] + '_snaps' + appendmodelnumber)
            if not os.path.exists(snapshotdir):
                os.mkdir(snapshotdir)

            if G.messages: print()
            for i, snap in enumerate(G.snapshots):
                snap.filename = os.path.abspath(os.path.join(snapshotdir, snap.basefilename + '.vti'))
                pbar = tqdm(total=snap.vtkdatawritesize, leave=True, unit='byte', unit_scale=True, desc='Writing snapshot file {} of {}, {}'.format(i + 1, len(G.snapshots), os.path.split(snap.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars)
                snap.write_vtk_imagedata(pbar, G)
                pbar.close()
            if G.messages: print()

        if G.messages:
            if G.gpu is None:
                print('Memory (RAM) used: ~{}'.format(human_size(p.memory_info().rss)))
            else:
                print('Memory (RAM) used: ~{} host + ~{} GPU'.format(human_size(p.memory_info().rss), human_size(memsolve)))
            print('Solving time [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=tsolve)))

    # If geometry information to be reused between model runs then FDTDGrid
    # class instance must be global so that it persists
    if not args.geometry_fixed or currentmodelrun == modelend:
        del G

    return tsolve


def solve_cpu(currentmodelrun, modelend, G):
    """
    Solving using FDTD method on CPU. Parallelised using Cython (OpenMP) for
    electric and magnetic field updates, and PML updates.

    Args:
        currentmodelrun (int): Current model run number.
        modelend (int): Number of last model to run.
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        tsolve (float): Time taken to execute solving
    """

    tsolvestart = timer()

    for iteration in tqdm(range(G.iterations), desc='Running simulation, model ' + str(currentmodelrun) + '/' + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars):
        # Store field component values for every receiver and transmission line
        store_outputs(iteration, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G)

        # Store any snapshots
        for snap in G.snapshots:
            if snap.time == iteration + 1:
                snap.store(G)

        # Update magnetic field components
        update_magnetic(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsH, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

        # Update magnetic field components with the PML correction
        for pml in G.pmls:
            pml.update_magnetic(G)

        # Update magnetic field components from sources
        for source in G.transmissionlines + G.magneticdipoles:
            source.update_magnetic(iteration, G.updatecoeffsH, G.ID, G.Hx, G.Hy, G.Hz, G)

        # Update electric field components
        # All materials are non-dispersive so do standard update
        if Material.maxpoles == 0:
            update_electric(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)
        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        elif Material.maxpoles == 1:
            update_electric_dispersive_1pole_A(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)
        elif Material.maxpoles > 1:
            update_electric_dispersive_multipole_A(G.nx, G.ny, G.nz, G.nthreads, Material.maxpoles, G.updatecoeffsE, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

        # Update electric field components with the PML correction
        for pml in G.pmls:
            pml.update_electric(G)

        # Update electric field components from sources (update any Hertzian dipole sources last)
        for source in G.voltagesources + G.transmissionlines + G.hertziandipoles:
            source.update_electric(iteration, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G)

        # If there are any dispersive materials do 2nd part of dispersive update
        # (it is split into two parts as it requires present and updated electric
        # field values). Therefore it can only be completely updated after the
        # electric field has been updated by the PML and source updates.
        if Material.maxpoles == 1:
            update_electric_dispersive_1pole_B(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez)
        elif Material.maxpoles > 1:
            update_electric_dispersive_multipole_B(G.nx, G.ny, G.nz, G.nthreads, Material.maxpoles, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez)

    tsolve = timer() - tsolvestart

    return tsolve


def solve_gpu(currentmodelrun, modelend, G):
    """Solving using FDTD method on GPU. Implemented using Nvidia CUDA.

    Args:
        currentmodelrun (int): Current model run number.
        modelend (int): Number of last model to run.
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        tsolve (float): Time taken to execute solving
        memsolve (int): memory usage on final iteration in bytes
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
    dev = drv.Device(G.gpu.deviceID)
    ctx = dev.make_context()

    # Electric and magnetic field updates - prepare kernels, and get kernel functions
    if Material.maxpoles > 0:
        kernels_fields = SourceModule(kernels_template_fields.substitute(REAL=cudafloattype, COMPLEX=cudacomplextype, N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=G.updatecoeffsdispersive.shape[1], NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3], NX_T=G.Tx.shape[1], NY_T=G.Tx.shape[2], NZ_T=G.Tx.shape[3]), options=compiler_opts)
    else:   # Set to one any substitutions for dispersive materials
        kernels_fields = SourceModule(kernels_template_fields.substitute(REAL=cudafloattype, COMPLEX=cudacomplextype, N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_MATDISPCOEFFS=1, NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3], NX_T=1, NY_T=1, NZ_T=1), options=compiler_opts)
    update_e_gpu = kernels_fields.get_function("update_e")
    update_h_gpu = kernels_fields.get_function("update_h")

    # Copy material coefficient arrays to constant memory of GPU (must be <64KB) for fields kernels
    updatecoeffsE = kernels_fields.get_global('updatecoeffsE')[0]
    updatecoeffsH = kernels_fields.get_global('updatecoeffsH')[0]
    if G.updatecoeffsE.nbytes + G.updatecoeffsH.nbytes > G.gpu.constmem:
        raise GeneralError('Too many materials in the model to fit onto constant memory of size {} on {} - {} GPU'.format(human_size(G.gpu.constmem), G.gpu.deviceID, G.gpu.name))
    else:
        drv.memcpy_htod(updatecoeffsE, G.updatecoeffsE)
        drv.memcpy_htod(updatecoeffsH, G.updatecoeffsH)

    # Electric and magnetic field updates - dispersive materials - get kernel functions and initialise array on GPU
    if Material.maxpoles > 0:  # If there are any dispersive materials (updates are split into two parts as they require present and updated electric field values).
        update_e_dispersive_A_gpu = kernels_fields.get_function("update_e_dispersive_A")
        update_e_dispersive_B_gpu = kernels_fields.get_function("update_e_dispersive_B")
        G.gpu_initialise_dispersive_arrays()

    # Electric and magnetic field updates - set blocks per grid and initialise field arrays on GPU
    G.gpu_set_blocks_per_grid()
    G.gpu_initialise_arrays()

    # PML updates
    if G.pmls:
        # Prepare kernels
        pmlmodulelectric = 'gprMax.pml_updates.pml_updates_electric_' + G.pmlformulation + '_gpu'
        kernelelectricfunc = getattr(import_module(pmlmodulelectric), 'kernels_template_pml_electric_' + G.pmlformulation)
        pmlmodulemagnetic = 'gprMax.pml_updates.pml_updates_magnetic_' + G.pmlformulation + '_gpu'
        kernelmagneticfunc = getattr(import_module(pmlmodulemagnetic), 'kernels_template_pml_magnetic_' + G.pmlformulation)
        kernels_pml_electric = SourceModule(kernelelectricfunc.substitute(REAL=cudafloattype, N_updatecoeffsE=G.updatecoeffsE.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3]), options=compiler_opts)
        kernels_pml_magnetic = SourceModule(kernelmagneticfunc.substitute(REAL=cudafloattype, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsH.shape[1], NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3]), options=compiler_opts)
        # Copy material coefficient arrays to constant memory of GPU (must be <64KB) for PML kernels
        updatecoeffsE = kernels_pml_electric.get_global('updatecoeffsE')[0]
        updatecoeffsH = kernels_pml_magnetic.get_global('updatecoeffsH')[0]
        drv.memcpy_htod(updatecoeffsE, G.updatecoeffsE)
        drv.memcpy_htod(updatecoeffsH, G.updatecoeffsH)
        # Set block per grid, initialise arrays on GPU, and get kernel functions
        for pml in G.pmls:
            pml.gpu_initialise_arrays()
            pml.gpu_get_update_funcs(kernels_pml_electric, kernels_pml_magnetic)
            pml.gpu_set_blocks_per_grid(G)

    # Receivers
    if G.rxs:
        # Initialise arrays on GPU
        rxcoords_gpu, rxs_gpu = gpu_initialise_rx_arrays(G)
        # Prepare kernel and get kernel function
        kernel_store_outputs = SourceModule(kernel_template_store_outputs.substitute(REAL=cudafloattype, NY_RXCOORDS=3, NX_RXS=6, NY_RXS=G.iterations, NZ_RXS=len(G.rxs), NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1), options=compiler_opts)
        store_outputs_gpu = kernel_store_outputs.get_function("store_outputs")

    # Sources - initialise arrays on GPU, prepare kernel and get kernel functions
    if G.voltagesources + G.hertziandipoles + G.magneticdipoles:
        kernels_sources = SourceModule(kernels_template_sources.substitute(REAL=cudafloattype, N_updatecoeffsE=G.updatecoeffsE.size, N_updatecoeffsH=G.updatecoeffsH.size, NY_MATCOEFFS=G.updatecoeffsE.shape[1], NY_SRCINFO=4, NY_SRCWAVES=G.iterations, NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1, NX_ID=G.ID.shape[1], NY_ID=G.ID.shape[2], NZ_ID=G.ID.shape[3]), options=compiler_opts)
        # Copy material coefficient arrays to constant memory of GPU (must be <64KB) for source kernels
        updatecoeffsE = kernels_sources.get_global('updatecoeffsE')[0]
        updatecoeffsH = kernels_sources.get_global('updatecoeffsH')[0]
        drv.memcpy_htod(updatecoeffsE, G.updatecoeffsE)
        drv.memcpy_htod(updatecoeffsH, G.updatecoeffsH)
        if G.hertziandipoles:
            srcinfo1_hertzian_gpu, srcinfo2_hertzian_gpu, srcwaves_hertzian_gpu = gpu_initialise_src_arrays(G.hertziandipoles, G)
            update_hertzian_dipole_gpu = kernels_sources.get_function("update_hertzian_dipole")
        if G.magneticdipoles:
            srcinfo1_magnetic_gpu, srcinfo2_magnetic_gpu, srcwaves_magnetic_gpu = gpu_initialise_src_arrays(G.magneticdipoles, G)
            update_magnetic_dipole_gpu = kernels_sources.get_function("update_magnetic_dipole")
        if G.voltagesources:
            srcinfo1_voltage_gpu, srcinfo2_voltage_gpu, srcwaves_voltage_gpu = gpu_initialise_src_arrays(G.voltagesources, G)
            update_voltage_source_gpu = kernels_sources.get_function("update_voltage_source")

    # Snapshots - initialise arrays on GPU, prepare kernel and get kernel functions
    if G.snapshots:
        # Initialise arrays on GPU
        snapEx_gpu, snapEy_gpu, snapEz_gpu, snapHx_gpu, snapHy_gpu, snapHz_gpu = gpu_initialise_snapshot_array(G)
        # Prepare kernel and get kernel function
        kernel_store_snapshot = SourceModule(kernel_template_store_snapshot.substitute(REAL=cudafloattype, NX_SNAPS=Snapshot.nx_max, NY_SNAPS=Snapshot.ny_max, NZ_SNAPS=Snapshot.nz_max, NX_FIELDS=G.nx + 1, NY_FIELDS=G.ny + 1, NZ_FIELDS=G.nz + 1), options=compiler_opts)
        store_snapshot_gpu = kernel_store_snapshot.get_function("store_snapshot")

    # Iteration loop timer
    iterstart = drv.Event()
    iterend = drv.Event()
    iterstart.record()

    for iteration in tqdm(range(G.iterations), desc='Running simulation, model ' + str(currentmodelrun) + '/' + str(modelend), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not G.progressbars):

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
                if not G.snapsgpu2cpu:
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
                     G.Ez_gpu.gpudata, block=G.tpb, grid=G.bpg)

        # Update magnetic field components with the PML correction
        for pml in G.pmls:
            pml.gpu_update_magnetic(G)

        # Update magnetic field components for magetic dipole sources
        if G.magneticdipoles:
            update_magnetic_dipole_gpu(np.int32(len(G.magneticdipoles)), np.int32(iteration),
                                       floattype(G.dx), floattype(G.dy), floattype(G.dz),
                                       srcinfo1_magnetic_gpu.gpudata, srcinfo2_magnetic_gpu.gpudata,
                                       srcwaves_magnetic_gpu.gpudata, G.ID_gpu.gpudata,
                                       G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                       block=(1, 1, 1), grid=(round32(len(G.magneticdipoles)), 1, 1))

        # Update electric field components
        # If all materials are non-dispersive do standard update
        if Material.maxpoles == 0:
            update_e_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz), G.ID_gpu.gpudata,
                         G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                         G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                         block=G.tpb, grid=G.bpg)
        # If there are any dispersive materials do 1st part of dispersive update
        # (it is split into two parts as it requires present and updated electric field values).
        else:
            update_e_dispersive_A_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz),
                                      np.int32(Material.maxpoles), G.updatecoeffsdispersive_gpu.gpudata,
                                      G.Tx_gpu.gpudata, G.Ty_gpu.gpudata, G.Tz_gpu.gpudata, G.ID_gpu.gpudata,
                                      G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                      G.Hx_gpu.gpudata, G.Hy_gpu.gpudata, G.Hz_gpu.gpudata,
                                      block=G.tpb, grid=G.bpg)

        # Update electric field components with the PML correction
        for pml in G.pmls:
            pml.gpu_update_electric(G)

        # Update electric field components for voltage sources
        if G.voltagesources:
            update_voltage_source_gpu(np.int32(len(G.voltagesources)), np.int32(iteration),
                                      floattype(G.dx), floattype(G.dy), floattype(G.dz),
                                      srcinfo1_voltage_gpu.gpudata, srcinfo2_voltage_gpu.gpudata,
                                      srcwaves_voltage_gpu.gpudata, G.ID_gpu.gpudata,
                                      G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                      block=(1, 1, 1), grid=(round32(len(G.voltagesources)), 1, 1))

        # Update electric field components for Hertzian dipole sources (update any Hertzian dipole sources last)
        if G.hertziandipoles:
            update_hertzian_dipole_gpu(np.int32(len(G.hertziandipoles)), np.int32(iteration),
                                       floattype(G.dx), floattype(G.dy), floattype(G.dz),
                                       srcinfo1_hertzian_gpu.gpudata, srcinfo2_hertzian_gpu.gpudata,
                                       srcwaves_hertzian_gpu.gpudata, G.ID_gpu.gpudata,
                                       G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                       block=(1, 1, 1), grid=(round32(len(G.hertziandipoles)), 1, 1))

        # If there are any dispersive materials do 2nd part of dispersive update (it is split into two parts as it requires present and updated electric field values). Therefore it can only be completely updated after the electric field has been updated by the PML and source updates.
        if Material.maxpoles > 0:
            update_e_dispersive_B_gpu(np.int32(G.nx), np.int32(G.ny), np.int32(G.nz),
                                      np.int32(Material.maxpoles), G.updatecoeffsdispersive_gpu.gpudata,
                                      G.Tx_gpu.gpudata, G.Ty_gpu.gpudata, G.Tz_gpu.gpudata, G.ID_gpu.gpudata,
                                      G.Ex_gpu.gpudata, G.Ey_gpu.gpudata, G.Ez_gpu.gpudata,
                                      block=G.tpb, grid=G.bpg)

    # Copy output from receivers array back to correct receiver objects
    if G.rxs:
        gpu_get_rx_array(rxs_gpu.get(), rxcoords_gpu.get(), G)

    # Copy data from any snapshots back to correct snapshot objects
    if G.snapshots and not G.snapsgpu2cpu:
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
