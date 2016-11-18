# Copyright (C) 2015-2016: The University of Edinburgh
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

"""gprMax.gprMax: provides entry point main()."""

import argparse
import datetime
from enum import Enum
import itertools
import os
import psutil
import sys
from time import perf_counter

from colorama import init, Fore, Style
init()
import numpy as np
from terminaltables import AsciiTable
from tqdm import tqdm

from ._version import __version__
from .constants import c, e0, m0, z0
from .exceptions import GeneralError
from .fields_outputs import write_hdf5_outputfile
from .fields_update import update_electric, update_magnetic, update_electric_dispersive_multipole_A, update_electric_dispersive_multipole_B, update_electric_dispersive_1pole_A, update_electric_dispersive_1pole_B
from .grid import FDTDGrid, dispersion_analysis
from .input_cmds_geometry import process_geometrycmds
from .input_cmds_file import process_python_include_code, write_processed_file, check_cmd_names
from .input_cmds_multiuse import process_multicmds
from .input_cmds_singleuse import process_singlecmds
from .materials import Material, process_materials
from .pml import PML, build_pmls
from .receivers import store_outputs
from .utilities import logo, human_size, get_machine_cpu_os, get_terminal_width, round_value
from .yee_cell_build import build_electric_components, build_magnetic_components


def main():
    """This is the main function for gprMax."""

    # Print gprMax logo, version, and licencing/copyright information
    logo(__version__ + ' (Bowmore)')

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='gprMax')
    parser.add_argument('inputfile', help='path to, and name of inputfile')
    parser.add_argument('-n', default=1, type=int, help='number of times to run the input file, e.g. to create a B-scan')
    parser.add_argument('-mpi', action='store_true', default=False, help='flag to switch on MPI task farm')
    parser.add_argument('-taskid', type=int, help='task identifier for job array on Open Grid Scheduler/Grid Engine (http://gridscheduler.sourceforge.net/index.html)')
    parser.add_argument('-benchmark', action='store_true', default=False, help='flag to switch on benchmarking mode')
    parser.add_argument('--geometry-only', action='store_true', default=False, help='flag to only build model and produce geometry file(s)')
    parser.add_argument('--geometry-fixed', action='store_true', default=False, help='flag to not reprocess model geometry, e.g. for B-scans where the geometry is fixed')
    parser.add_argument('--write-processed', action='store_true', default=False, help='flag to write an input file after any Python code and include commands in the original input file have been processed')
    parser.add_argument('--opt-taguchi', action='store_true', default=False, help='flag to optimise parameters using the Taguchi optimisation method')
    args = parser.parse_args()

    run_main(args)


def api(inputfile, n=1, mpi=False, taskid=False, benchmark=False, geometry_only=False, geometry_fixed=False, write_processed=False, opt_taguchi=False):
    """If installed as a module this is the entry point."""

    class ImportArguments:
        pass

    args = ImportArguments()

    args.inputfile = inputfile
    args.n = n
    args.mpi = mpi
    args.taskid = taskid
    args.benchmark = benchmark
    args.geometry_only = geometry_only
    args.geometry_fixed = geometry_fixed
    args.write_processed = write_processed
    args.opt_taguchi = opt_taguchi

    run_main(args)


def run_main(args):
    """Top-level function that controls what mode of simulation (standard/optimsation/benchmark etc...) is run.

    Args:
        args (dict): Namespace with input arguments from command line or api.
    """

    numbermodelruns = args.n
    inputdirectory = os.path.dirname(os.path.abspath(args.inputfile))
    inputfile = os.path.abspath(os.path.join(inputdirectory, os.path.basename(args.inputfile)))

    # Create a separate namespace that users can access in any Python code blocks in the input file
    usernamespace = {'c': c, 'e0': e0, 'm0': m0, 'z0': z0, 'number_model_runs': numbermodelruns, 'input_directory': inputdirectory}

    # Process for Taguchi optimisation
    if args.opt_taguchi:
        if args.benchmark:
            raise GeneralError('Taguchi optimisation should not be used with benchmarking mode')
        from gprMax.optimisation_taguchi import run_opt_sim
        run_opt_sim(args, numbermodelruns, inputfile, usernamespace)

    # Process for benchmarking simulation
    elif args.benchmark:
        run_benchmark_sim(args, inputfile, usernamespace)

    # Process for standard simulation
    else:
        # Mixed mode MPI/OpenMP - MPI task farm for models with each model parallelised with OpenMP
        if args.mpi:
            if args.benchmark:
                raise GeneralError('MPI should not be used with benchmarking mode')
            if numbermodelruns == 1:
                raise GeneralError('MPI is not beneficial when there is only one model to run')
            run_mpi_sim(args, numbermodelruns, inputfile, usernamespace)
        # Standard behaviour - part of a job array on Open Grid Scheduler/Grid Engine with each model parallelised with OpenMP
        elif args.taskid:
            if args.benchmark:
                raise GeneralError('A job array should not be used with benchmarking mode')
            if numbermodelruns == 1:
                raise GeneralError('A job array is not beneficial when there is only one model to run')
            run_job_array_sim(args, numbermodelruns, inputfile, usernamespace)
        # Standard behaviour - models run serially with each model parallelised with OpenMP
        else:
            run_std_sim(args, numbermodelruns, inputfile, usernamespace)


def run_std_sim(args, numbermodelruns, inputfile, usernamespace, optparams=None):
    """Run standard simulation - models are run one after another and each model is parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        numbermodelruns (int): Total number of model runs.
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it provides the parameters to optimise and their values.
    """

    tsimstart = perf_counter()
    for modelrun in range(1, numbermodelruns + 1):
        if optparams:  # If Taguchi optimistaion, add specific value for each parameter to optimise for each experiment to user accessible namespace
            tmp = {}
            tmp.update((key, value[modelrun - 1]) for key, value in optparams.items())
            modelusernamespace = usernamespace.copy()
            modelusernamespace.update({'optparams': tmp})
        else:
            modelusernamespace = usernamespace
        run_model(args, modelrun, numbermodelruns, inputfile, modelusernamespace)
    tsimend = perf_counter()
    simcompletestr = '\n=== Simulation completed in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=int(tsimend - tsimstart)))
    print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))


def run_job_array_sim(args, numbermodelruns, inputfile, usernamespace, optparams=None):
    """Run standard simulation as part of a job array on Open Grid Scheduler/Grid Engine (http://gridscheduler.sourceforge.net/index.html) - each model is parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        numbermodelruns (int): Total number of model runs.
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it provides the parameters to optimise and their values.
    """

    modelrun = args.taskid

    tsimstart = perf_counter()
    if optparams:  # If Taguchi optimistaion, add specific value for each parameter to optimise for each experiment to user accessible namespace
        tmp = {}
        tmp.update((key, value[modelrun - 1]) for key, value in optparams.items())
        modelusernamespace = usernamespace.copy()
        modelusernamespace.update({'optparams': tmp})
    else:
        modelusernamespace = usernamespace
    run_model(args, modelrun, numbermodelruns, inputfile, modelusernamespace)
    tsimend = perf_counter()
    simcompletestr = '\n=== Simulation completed in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=int(tsimend - tsimstart)))
    print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))


def run_benchmark_sim(args, inputfile, usernamespace):
    """Run standard simulation in benchmarking mode - models are run one after another and each model is parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
    """

    # Number of threads to test - start from max physical CPU cores and divide in half until 1
    thread = psutil.cpu_count(logical=False)
    threads = [thread]
    while not thread % 2:
        thread /= 2
        threads.append(int(thread))

    benchtimes = np.zeros(len(threads))
    numbermodelruns = len(threads)
    usernamespace['number_model_runs'] = numbermodelruns

    for modelrun in range(1, numbermodelruns + 1):
        os.environ['OMP_NUM_THREADS'] = str(threads[modelrun - 1])
        tsolve = run_model(args, modelrun, numbermodelruns, inputfile, usernamespace)
        benchtimes[modelrun - 1] = tsolve

    # Save number of threads and benchmarking times to NumPy archive
    threads = np.array(threads)
    machineID, cpuID, osversion = get_machine_cpu_os()
    machineIDlong = machineID + '; ' + cpuID + '; ' + osversion
    np.savez(os.path.splitext(inputfile)[0], threads=threads, benchtimes=benchtimes, machineID=machineIDlong, version=__version__)

    simcompletestr = '\n=== Simulation completed'
    print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))


def run_mpi_sim(args, numbermodelruns, inputfile, usernamespace, optparams=None):
    """Run mixed mode MPI/OpenMP simulation - MPI task farm for models with each model parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        numbermodelruns (int): Total number of model runs.
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it provides the parameters to optimise and their values.
    """

    from mpi4py import MPI

    # Define MPI message tags
    tags = Enum('tags', {'READY': 0, 'DONE': 1, 'EXIT': 2, 'START': 3})

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.Get_size()  # total number of processes
    rank = comm.Get_rank()  # rank of this process
    status = MPI.Status()   # get MPI status object
    name = MPI.Get_processor_name()     # get name of processor/host

    tsimstart = perf_counter()

    # Master process
    if rank == 0:
        modelrun = 1
        numworkers = size - 1
        closedworkers = 0
        print('Master: PID {} on {} using {} workers.'.format(os.getpid(), name, numworkers))
        while closedworkers < numworkers:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY.value:  # Worker is ready, so send it a task
                if modelrun < numbermodelruns + 1:
                    comm.send(modelrun, dest=source, tag=tags.START.value)
                    print('Master: sending model {} to worker {}.'.format(modelrun, source))
                    modelrun += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT.value)

            elif tag == tags.DONE.value:
                print('Worker {}: completed.'.format(source))

            elif tag == tags.EXIT.value:
                print('Worker {}: exited.'.format(source))
                closedworkers += 1

    # Worker process
    else:
        print('Worker {}: PID {} on {}.'.format(rank, os.getpid(), name))
        while True:
            comm.send(None, dest=0, tag=tags.READY.value)
            modelrun = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)  #  Receive a model number to run from the master
            tag = status.Get_tag()

            # Run a model
            if tag == tags.START.value:
                if optparams:  # If Taguchi optimistaion, add specific value for each parameter to optimise for each experiment to user accessible namespace
                    tmp = {}
                    tmp.update((key, value[modelrun - 1]) for key, value in optparams.items())
                    modelusernamespace = usernamespace.copy()
                    modelusernamespace.update({'optparams': tmp})
                else:
                    modelusernamespace = usernamespace

                run_model(args, modelrun, numbermodelruns, inputfile, modelusernamespace)
                comm.send(None, dest=0, tag=tags.DONE.value)

            elif tag == tags.EXIT.value:
                break

        comm.send(None, dest=0, tag=tags.EXIT.value)

    tsimend = perf_counter()
    simcompletestr = '\n=== Simulation completed in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=int(tsimend - tsimstart)))
    print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))


def run_model(args, modelrun, numbermodelruns, inputfile, usernamespace):
    """Runs a model - processes the input file; builds the Yee cells; calculates update coefficients; runs main FDTD loop.

    Args:
        args (dict): Namespace with command line arguments
        modelrun (int): Current model run number.
        numbermodelruns (int): Total number of model runs.
        inputfile (str): Name of the input file to open.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.

    Returns:
        tsolve (int): Length of time (seconds) of main FDTD calculations
    """

    # Monitor memory usage
    p = psutil.Process()

    # Declare variable to hold FDTDGrid class
    global G

    # Normal model reading/building process; bypassed if geometry information to be reused
    if 'G' not in globals():
        inputfilestr = '\n--- Model {} of {}, input file: {}'.format(modelrun, numbermodelruns, inputfile)
        print(Fore.GREEN + '{} {}\n'.format(inputfilestr, '-' * (get_terminal_width() - 1 - len(inputfilestr))) + Style.RESET_ALL)

        # Add the current model run to namespace that can be accessed by user in any Python code blocks in input file
        usernamespace['current_model_run'] = modelrun

        # Read input file and process any Python or include commands
        processedlines = process_python_include_code(inputfile, usernamespace)

        # Print constants/variables in user-accessable namespace
        uservars = ''
        for key, value in sorted(usernamespace.items()):
            if key != '__builtins__':
                uservars += '{}: {}, '.format(key, value)
        print('Constants/variables used/available for Python scripting: {{{}}}\n'.format(uservars[:-2]))

        # Write a file containing the input commands after Python or include commands have been processed
        if args.write_processed:
            write_processed_file(inputfile, modelrun, numbermodelruns, processedlines)

        # Check validity of command names and that essential commands are present
        singlecmds, multicmds, geometry = check_cmd_names(processedlines)

        # Initialise an instance of the FDTDGrid class
        G = FDTDGrid()
        G.inputfilename = os.path.split(inputfile)[1]
        G.inputdirectory = os.path.dirname(os.path.abspath(inputfile))

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
        print()
        process_multicmds(multicmds, G)

        # Initialise an array for volumetric material IDs (solid), boolean arrays for specifying materials not to be averaged (rigid),
        # an array for cell edge IDs (ID)
        G.initialise_geometry_arrays()

        # Initialise arrays for the field components
        G.initialise_field_arrays()

        # Process geometry commands in the order they were given
        process_geometrycmds(geometry, G)

        # Build the PMLs and calculate initial coefficients
        print()
        if all(value == 0 for value in G.pmlthickness.values()):
            if G.messages:
                print('PML boundaries: switched off')
            pass  # If all the PMLs are switched off don't need to build anything
        else:
            if G.messages:
                if all(value == G.pmlthickness['xminus'] for value in G.pmlthickness.values()):
                    pmlinfo = G.pmlthickness['xminus']
                else:
                    pmlinfo = ''
                    for key, value in G.pmlthickness.items():
                        pmlinfo += '{}: {}, '.format(key, value)
                    pmlinfo = pmlinfo[:-2]
                print('PML boundaries: {} cells'.format(pmlinfo))
            pbar = tqdm(total=sum(1 for value in G.pmlthickness.values() if value > 0), desc='Building PML boundaries', ncols=get_terminal_width() - 1, file=sys.stdout, disable=G.tqdmdisable)
            build_pmls(G, pbar)
            pbar.close()

        # Build the model, i.e. set the material properties (ID) for every edge of every Yee cell
        print()
        pbar = tqdm(total=2, desc='Building main grid', ncols=get_terminal_width() - 1, file=sys.stdout, disable=G.tqdmdisable)
        build_electric_components(G.solid, G.rigidE, G.ID, G)
        pbar.update()
        build_magnetic_components(G.solid, G.rigidH, G.ID, G)
        pbar.update()
        pbar.close()

        # Process any voltage sources (that have resistance) to create a new material at the source location
        for voltagesource in G.voltagesources:
            voltagesource.create_material(G)

        # Initialise arrays of update coefficients to pass to update functions
        G.initialise_std_update_coeff_arrays()

        # Initialise arrays of update coefficients and temporary values if there are any dispersive materials
        if Material.maxpoles != 0:
            G.initialise_dispersive_arrays()

        # Process complete list of materials - calculate update coefficients, store in arrays, and build text list of materials/properties
        materialsdata = process_materials(G)
        if G.messages:
            materialstable = AsciiTable(materialsdata)
            materialstable.outer_border = False
            materialstable.justify_columns[0] = 'right'
            print(materialstable.table)

        # Check to see if numerical dispersion might be a problem
        results = dispersion_analysis(G)
        if results['deltavp'] and np.abs(results['deltavp']) > G.maxnumericaldisp:
            print(Fore.RED + "\nWARNING: Potentially significant numerical dispersion. Largest physical phase-velocity error is {:.2f}% in material '{}' with wavelength sampled by {} cells (maximum significant frequency {:g}Hz)".format(results['deltavp'], results['material'].ID, round_value(results['N']), results['maxfreq']) + Style.RESET_ALL)
        elif results['deltavp']:
            print("\nNumerical dispersion analysis: largest physical phase-velocity error is {:.2f}% in material '{}' with wavelength sampled by {} cells (maximum significant frequency {:g}Hz)".format(results['deltavp'], results['material'].ID, round_value(results['N']), results['maxfreq']))

    # If geometry information to be reused between model runs
    else:
        inputfilestr = '\n--- Model {} of {}, input file (not re-processed, i.e. geometry fixed): {}'.format(modelrun, numbermodelruns, inputfile)
        print(Fore.GREEN + '{} {}\n'.format(inputfilestr, '-' * (get_terminal_width() - 1 - len(inputfilestr))) + Style.RESET_ALL)

        # Clear arrays for field components
        G.initialise_field_arrays()

        # Clear arrays for fields in PML
        for pml in G.pmls:
            pml.initialise_field_arrays()

    # Adjust position of simple sources and receivers if required
    if G.srcsteps[0] > 0 or G.srcsteps[1] > 0 or G.srcsteps[2] > 0:
        for source in itertools.chain(G.hertziandipoles, G.magneticdipoles):
            if modelrun == 1:
                if source.xcoord + G.srcsteps[0] * (numbermodelruns - 1) > G.nx or source.ycoord + G.srcsteps[1] * (numbermodelruns - 1) > G.ny or source.zcoord + G.srcsteps[2] * (numbermodelruns - 1) > G.nz:
                    raise GeneralError('Source(s) will be stepped to a position outside the domain.')
            source.xcoord = source.xcoordorigin + (modelrun - 1) * G.srcsteps[0]
            source.ycoord = source.ycoordorigin + (modelrun - 1) * G.srcsteps[1]
            source.zcoord = source.zcoordorigin + (modelrun - 1) * G.srcsteps[2]
    if G.rxsteps[0] > 0 or G.rxsteps[1] > 0 or G.rxsteps[2] > 0:
        for receiver in G.rxs:
            if modelrun == 1:
                if receiver.xcoord + G.rxsteps[0] * (numbermodelruns - 1) > G.nx or receiver.ycoord + G.rxsteps[1] * (numbermodelruns - 1) > G.ny or receiver.zcoord + G.rxsteps[2] * (numbermodelruns - 1) > G.nz:
                    raise GeneralError('Receiver(s) will be stepped to a position outside the domain.')
            receiver.xcoord = receiver.xcoordorigin + (modelrun - 1) * G.rxsteps[0]
            receiver.ycoord = receiver.ycoordorigin + (modelrun - 1) * G.rxsteps[1]
            receiver.zcoord = receiver.zcoordorigin + (modelrun - 1) * G.rxsteps[2]

    # Write files for any geometry views and geometry object outputs
    if not (G.geometryviews or G.geometryobjectswrite) and args.geometry_only:
        print(Fore.RED + '\nWARNING: No geometry views or geometry objects to output found.' + Style.RESET_ALL)
    if G.geometryviews:
        print()
        for i, geometryview in enumerate(G.geometryviews):
            geometryview.set_filename(modelrun, numbermodelruns, G)
            pbar = tqdm(total=geometryview.datawritesize, unit='byte', unit_scale=True, desc='Writing geometry view file {} of {}, {}'.format(i + 1, len(G.geometryviews), os.path.split(geometryview.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=G.tqdmdisable)
            geometryview.write_vtk(modelrun, numbermodelruns, G, pbar)
            pbar.close()
    if G.geometryobjectswrite:

        for i, geometryobject in enumerate(G.geometryobjectswrite):
            pbar = tqdm(total=geometryobject.datawritesize, unit='byte', unit_scale=True, desc='Writing geometry object file {} of {}, {}'.format(i + 1, len(G.geometryobjectswrite), os.path.split(geometryobject.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=G.tqdmdisable)
            geometryobject.write_hdf5(G, pbar)
            pbar.close()

    # Run simulation (if not doing geometry only)
    if not args.geometry_only:

        # Prepare any snapshot files
        for snapshot in G.snapshots:
            snapshot.prepare_vtk_imagedata(modelrun, numbermodelruns, G)

        # Output filename
        inputfileparts = os.path.splitext(inputfile)
        if numbermodelruns == 1:
            outputfile = inputfileparts[0] + '.out'
        else:
            outputfile = inputfileparts[0] + str(modelrun) + '.out'
        print('\nOutput file: {}\n'.format(outputfile))

        ####################################
        #  Start - Main FDTD calculations  #
        ####################################
        tsolvestart = perf_counter()

        # Absolute time
        abstime = 0

        for timestep in tqdm(range(G.iterations), desc='Running simulation, model ' + str(modelrun) + ' of ' + str(numbermodelruns), ncols=get_terminal_width() - 1, file=sys.stdout, disable=G.tqdmdisable):
            # Store field component values for every receiver and transmission line
            store_outputs(timestep, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G)

            # Write any snapshots to file
            for i, snap in enumerate(G.snapshots):
                if snap.time == timestep + 1:
                    snapiters = 36 * (((snap.xf - snap.xs) / snap.dx) * ((snap.yf - snap.ys) / snap.dy) * ((snap.zf - snap.zs) / snap.dz))
                    pbar = tqdm(total=snapiters, leave=False, unit='byte', unit_scale=True, desc='  Writing snapshot file {} of {}, {}'.format(i + 1, len(G.snapshots), os.path.split(snap.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=G.tqdmdisable)
                    snap.write_vtk_imagedata(G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G, pbar)
                    pbar.close()

            # Update electric field components
            if Material.maxpoles == 0:  # All materials are non-dispersive so do standard update
                update_electric(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)
            elif Material.maxpoles == 1:  # If there are any dispersive materials do 1st part of dispersive update (it is split into two parts as it requires present and updated electric field values).
                update_electric_dispersive_1pole_A(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)
            elif Material.maxpoles > 1:
                update_electric_dispersive_multipole_A(G.nx, G.ny, G.nz, G.nthreads, Material.maxpoles, G.updatecoeffsE, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

            # Update electric field components with the PML correction
            for pml in G.pmls:
                pml.update_electric(G)

            # Update electric field components from sources (update any Hertzian dipole sources last)
            for source in G.voltagesources + G.transmissionlines + G.hertziandipoles:
                source.update_electric(abstime, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G)

            # If there are any dispersive materials do 2nd part of dispersive update (it is split into two parts as it requires present and updated electric field values). Therefore it can only be completely updated after the electric field has been updated by the PML and source updates.
            if Material.maxpoles == 1:
                update_electric_dispersive_1pole_B(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez)
            elif Material.maxpoles > 1:
                update_electric_dispersive_multipole_B(G.nx, G.ny, G.nz, G.nthreads, Material.maxpoles, G.updatecoeffsdispersive, G.ID, G.Tx, G.Ty, G.Tz, G.Ex, G.Ey, G.Ez)

            # Increment absolute time value
            abstime += 0.5 * G.dt

            # Update magnetic field components
            update_magnetic(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsH, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

            # Update magnetic field components with the PML correction
            for pml in G.pmls:
                pml.update_magnetic(G)

            # Update magnetic field components from sources
            for source in G.transmissionlines + G.magneticdipoles:
                source.update_magnetic(abstime, G.updatecoeffsH, G.ID, G.Hx, G.Hy, G.Hz, G)

            # Increment absolute time value
            abstime += 0.5 * G.dt

        tsolve = int(perf_counter() - tsolvestart)

        # Write an output file in HDF5 format
        write_hdf5_outputfile(outputfile, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G)

        ##################################
        #  End - Main FDTD calculations  #
        ##################################

    if G.messages:
        print('Memory (RAM) used: ~{}'.format(human_size(p.memory_info().rss)))

    # If geometry information to be reused between model runs then FDTDGrid class instance must be global so that it persists
    if not args.geometry_fixed:
        del G

    # Return time to complete solving if in benchmarking mode
    if args.benchmark:
        return tsolve
