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

"""gprMax.gprMax: provides entry point main()."""

import argparse
import datetime
import os
import sys

from enum import Enum
from time import perf_counter

import h5py
import numpy as np

from gprMax._version import __version__, codename
from gprMax.constants import c
from gprMax.constants import e0
from gprMax.constants import m0
from gprMax.constants import z0
from gprMax.exceptions import GeneralError
from gprMax.model_build_run import run_model
from gprMax.utilities import detect_gpus
from gprMax.utilities import get_host_info
from gprMax.utilities import get_terminal_width
from gprMax.utilities import human_size
from gprMax.utilities import logo
from gprMax.utilities import open_path_file


def main():
    """This is the main function for gprMax."""

    # Print gprMax logo, version, and licencing/copyright information
    logo(__version__ + ' (' + codename + ')')

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='gprMax', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile', help='path to, and name of inputfile or file object')
    parser.add_argument('-n', default=1, type=int, help='number of times to run the input file, e.g. to create a B-scan')
    parser.add_argument('-task', type=int, help='task identifier (model number) for job array on Open Grid Scheduler/Grid Engine (http://gridscheduler.sourceforge.net/index.html)')
    parser.add_argument('-restart', type=int, help='model number to restart from, e.g. when creating B-scan')
    parser.add_argument('-mpi', type=int, help='number of MPI tasks, i.e. master + workers')
    parser.add_argument('-mpialt', action='store_true', default=False, help='flag to switch on MPI task farm')
    parser.add_argument('--mpi-worker', action='store_true', default=False, help=argparse.SUPPRESS)
    parser.add_argument('-gpu', type=int, action='append', nargs='?', const=True, help='flag to use Nvidia GPU (option to give device ID)')
    parser.add_argument('-benchmark', action='store_true', default=False, help='flag to switch on benchmarking mode')
    parser.add_argument('--geometry-only', action='store_true', default=False, help='flag to only build model and produce geometry file(s)')
    parser.add_argument('--geometry-fixed', action='store_true', default=False, help='flag to not reprocess model geometry, e.g. for B-scans where the geometry is fixed')
    parser.add_argument('--write-processed', action='store_true', default=False, help='flag to write an input file after any Python code and include commands in the original input file have been processed')
    parser.add_argument('--opt-taguchi', action='store_true', default=False, help='flag to optimise parameters using the Taguchi optimisation method')
    args = parser.parse_args()

    run_main(args)


def api(
    inputfile,
    n=1,
    task=None,
    restart=None,
    mpi=False,
    mpialt=False,
    mpicomm=None,
    gpu=None,
    benchmark=False,
    geometry_only=False,
    geometry_fixed=False,
    write_processed=False,
    opt_taguchi=False
):
    """If installed as a module this is the entry point."""

    # Print gprMax logo, version, and licencing/copyright information
    logo(__version__ + ' (' + codename + ')')

    class ImportArguments:
        pass

    args = ImportArguments()

    args.inputfile = inputfile
    args.n = n
    args.task = task
    args.restart = restart
    args.mpi = mpi
    args.mpialt = mpialt
    args.mpicomm = mpicomm
    args.gpu = gpu
    args.benchmark = benchmark
    args.geometry_only = geometry_only
    args.geometry_fixed = geometry_fixed
    args.write_processed = write_processed
    args.opt_taguchi = opt_taguchi

    run_main(args)


def run_main(args):
    """
    Top-level function that controls what mode of simulation (standard/optimsation/benchmark etc...) is run.

    Args:
        args (dict): Namespace with input arguments from command line or api.
    """

    with open_path_file(args.inputfile) as inputfile:

        # Get information about host machine
        hostinfo = get_host_info()
        hyperthreading = ', {} cores with Hyper-Threading'.format(hostinfo['logicalcores']) if hostinfo['hyperthreading'] else ''
        print('\nHost: {} | {} | {} x {} ({} cores{}) | {} RAM | {}'.format(hostinfo['hostname'],
                                                                            hostinfo['machineID'], hostinfo['sockets'], hostinfo['cpuID'], hostinfo['physicalcores'],
                                                                            hyperthreading, human_size(hostinfo['ram'], a_kilobyte_is_1024_bytes=True), hostinfo['osversion']))

        # Get information/setup Nvidia GPU(s)
        if args.gpu is not None:
            # Extract first item of list, either True to automatically determine device ID,
            # or an integer to manually specify device ID
            args.gpu = args.gpu[0]
            gpus = detect_gpus()

            # If a device ID is specified check it is valid
            if not isinstance(args.gpu, bool):
                if args.gpu > len(gpus) - 1:
                    raise GeneralError('GPU with device ID {} does not exist'.format(args.gpu))
                # Set args.gpu to GPU object to access elsewhere
                args.gpu = next(gpu for gpu in gpus if gpu.deviceID == args.gpu)

            # If no device ID is specified
            else:
                # If in MPI mode then set args.gpu to list of available GPUs
                if args.mpi:
                    # if args.mpi - 1 > len(gpus):
                    #     raise GeneralError('Too many MPI tasks requested ({}). The number of MPI tasks requested can only be a maximum of the number of GPU(s) detected plus one, i.e. {} GPU worker tasks + 1 CPU master task'.format(args.mpi, len(gpus)))
                    args.gpu = gpus
                elif args.mpialt:
                    args.gpu = gpus
                # If benchmarking mode then set args.gpu to list of available GPUs
                elif args.benchmark:
                    args.gpu = gpus
                # Otherwise set args.gpu to GPU object with default device ID (0) to access elsewhere
                else:
                    args.gpu = next(gpu for gpu in gpus if gpu.deviceID == 0)

        # Create a separate namespace that users can access in any Python code blocks in the input file
        usernamespace = {'c': c, 'e0': e0, 'm0': m0, 'z0': z0, 'number_model_runs': args.n, 'inputfile': os.path.abspath(inputfile.name)}

        #######################################
        # Process for benchmarking simulation #
        #######################################
        if args.benchmark:
            if args.mpi or args.opt_taguchi or args.task or args.n > 1:
                raise GeneralError('Benchmarking mode cannot be combined with MPI, job array, or Taguchi optimisation modes, or multiple model runs.')
            run_benchmark_sim(args, inputfile, usernamespace)

        ####################################################
        # Process for simulation with Taguchi optimisation #
        ####################################################
        elif args.opt_taguchi:
            if args.mpi_worker:  # Special case for MPI spawned workers - they do not need to enter the Taguchi optimisation mode
                run_mpi_sim(args, inputfile, usernamespace)
            else:
                from gprMax.optimisation_taguchi import run_opt_sim
                run_opt_sim(args, inputfile, usernamespace)

        ################################################
        # Process for standard simulation (CPU or GPU) #
        ################################################
        else:
            # Mixed mode MPI with OpenMP or CUDA - MPI task farm for models with each model parallelised with OpenMP (CPU) or CUDA (GPU)
            if args.mpi:
                if args.n == 1:
                    raise GeneralError('MPI is not beneficial when there is only one model to run')
                if args.task:
                    raise GeneralError('MPI cannot be combined with job array mode')
                run_mpi_sim(args, inputfile, usernamespace)

            # TEST! TEST! TEST! Alternate MPI configuration
            elif args.mpialt:
                run_mpi_alt_sim(args, inputfile, usernamespace)

            # Standard behaviour - models run serially with each model parallelised with OpenMP (CPU) or CUDA (GPU)
            else:
                if args.task and args.restart:
                    raise GeneralError('Job array and restart modes cannot be used together')
                run_std_sim(args, inputfile, usernamespace)


def run_std_sim(args, inputfile, usernamespace, optparams=None):
    """
    Run standard simulation - models are run one after another and each model
    is parallelised using either OpenMP (CPU) or CUDA (GPU)

    Args:
        args (dict): Namespace with command line arguments
        inputfile (object): File object for the input file.
        usernamespace (dict): Namespace that can be accessed by user in any
                Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it
                provides the parameters to optimise and their values.
    """

    # Set range for number of models to run
    if args.task:
        # Job array feeds args.n number of single tasks
        modelstart = args.task
        modelend = args.task + 1
    elif args.restart:
        modelstart = args.restart
        modelend = modelstart + args.n
    else:
        modelstart = 1
        modelend = modelstart + args.n
    numbermodelruns = args.n

    tsimstart = perf_counter()
    for currentmodelrun in range(modelstart, modelend):
        # If Taguchi optimistaion, add specific value for each parameter to
        # optimise for each experiment to user accessible namespace
        if optparams:
            tmp = {}
            tmp.update((key, value[currentmodelrun - 1]) for key, value in optparams.items())
            modelusernamespace = usernamespace.copy()
            modelusernamespace.update({'optparams': tmp})
        else:
            modelusernamespace = usernamespace
        run_model(args, currentmodelrun, modelend - 1, numbermodelruns, inputfile, modelusernamespace)
    tsimend = perf_counter()
    simcompletestr = '\n=== Simulation completed in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=tsimend - tsimstart))
    print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))


def run_benchmark_sim(args, inputfile, usernamespace):
    """
    Run standard simulation in benchmarking mode - models are run one
    after another and each model is parallelised using either OpenMP (CPU)
    or CUDA (GPU)

    Args:
        args (dict): Namespace with command line arguments
        inputfile (object): File object for the input file.
        usernamespace (dict): Namespace that can be accessed by user in any
                Python code blocks in input file.
    """

    # Get information about host machine
    hostinfo = get_host_info()
    hyperthreading = ', {} cores with Hyper-Threading'.format(hostinfo['logicalcores']) if hostinfo['hyperthreading'] else ''
    machineIDlong = '{}; {} x {} ({} cores{}); {} RAM; {}'.format(hostinfo['machineID'], hostinfo['sockets'], hostinfo['cpuID'], hostinfo['physicalcores'], hyperthreading, human_size(hostinfo['ram'], a_kilobyte_is_1024_bytes=True), hostinfo['osversion'])

    # Initialise arrays to hold CPU thread info and times, and GPU info and times
    cputhreads = np.array([], dtype=np.int32)
    cputimes = np.array([])
    gpuIDs = []
    gputimes = np.array([])

    # CPU only benchmarking
    if args.gpu is None:
        # Number of CPU threads to benchmark - start from single thread and double threads until maximum number of physical cores
        threads = 1
        maxthreads = hostinfo['physicalcores']
        maxthreadspersocket = hostinfo['physicalcores'] / hostinfo['sockets']
        while threads < maxthreadspersocket:
            cputhreads = np.append(cputhreads, int(threads))
            threads *= 2
        # Check for system with only single thread
        if cputhreads.size == 0:
            cputhreads = np.append(cputhreads, threads)
        # Add maxthreadspersocket and maxthreads if necessary
        if cputhreads[-1] != maxthreadspersocket:
            cputhreads = np.append(cputhreads, int(maxthreadspersocket))
        if cputhreads[-1] != maxthreads:
            cputhreads = np.append(cputhreads, int(maxthreads))
        cputhreads = cputhreads[::-1]
        cputimes = np.zeros(len(cputhreads))

        numbermodelruns = len(cputhreads)

    # GPU only benchmarking
    else:
        # Set size of array to store GPU runtimes and number of runs of model required
        if isinstance(args.gpu, list):
            for gpu in args.gpu:
                gpuIDs.append(gpu.name)
            gputimes = np.zeros(len(args.gpu))
            numbermodelruns = len(args.gpu)
        else:
            gpuIDs.append(args.gpu.name)
            gputimes = np.zeros(1)
            numbermodelruns = 1
        # Store GPU information in a temp variable
        gpus = args.gpu

    usernamespace['number_model_runs'] = numbermodelruns
    modelend = numbermodelruns + 1

    for currentmodelrun in range(1, modelend):
        # Run CPU benchmark
        if args.gpu is None:
            os.environ['OMP_NUM_THREADS'] = str(cputhreads[currentmodelrun - 1])
            cputimes[currentmodelrun - 1] = run_model(args, currentmodelrun, modelend - 1, numbermodelruns, inputfile, usernamespace)
        # Run GPU benchmark
        else:
            if isinstance(gpus, list):
                args.gpu = gpus[(currentmodelrun - 1)]
            else:
                args.gpu = gpus
            os.environ['OMP_NUM_THREADS'] = str(hostinfo['physicalcores'])
            gputimes[(currentmodelrun - 1)] = run_model(args, currentmodelrun, modelend - 1, numbermodelruns, inputfile, usernamespace)

        # Get model size (in cells) and number of iterations
        if currentmodelrun == 1:
            if numbermodelruns == 1:
                outputfile = os.path.splitext(args.inputfile)[0] + '.out'
            else:
                outputfile = os.path.splitext(args.inputfile)[0] + str(currentmodelrun) + '.out'
            f = h5py.File(outputfile, 'r')
            iterations = f.attrs['Iterations']
            numcells = f.attrs['nx_ny_nz']

    # Save number of threads and benchmarking times to NumPy archive
    np.savez(os.path.splitext(inputfile.name)[0], machineID=machineIDlong, gpuIDs=gpuIDs, cputhreads=cputhreads, cputimes=cputimes, gputimes=gputimes, iterations=iterations, numcells=numcells, version=__version__)

    simcompletestr = '\n=== Simulation completed'
    print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))


def run_mpi_sim(args, inputfile, usernamespace, optparams=None):
    """
    Run mixed mode MPI/OpenMP simulation - MPI task farm for models with
    each model parallelised using either OpenMP (CPU) or CUDA (GPU)

    Args:
        args (dict): Namespace with command line arguments
        inputfile (object): File object for the input file.
        usernamespace (dict): Namespace that can be accessed by user in any
                Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it
                provides the parameters to optimise and their values.
    """
    print('run_mpi_sim')
    from mpi4py import MPI

    status = MPI.Status()
    hostname = MPI.Get_processor_name()

    # Set range for number of models to run
    modelstart = args.restart if args.restart else 1
    modelend = modelstart + args.n
    numbermodelruns = args.n

    # Command line flag used to indicate a spawned worker instance
    workerflag = '--mpi-worker'
    numworkers = args.mpi - 1

    ##################
    # Master process #
    ##################
    if workerflag not in sys.argv:
        # N.B Spawned worker flag (--mpi-worker) applied to sys.argv when MPI.Spawn is called

        # Get MPI communicator object either through argument or just get comm_world
        if hasattr(args, 'mpicomm'):
            comm = args.mpicomm
        else:
            comm = MPI.COMM_WORLD
        rank = comm.Get_rank()  # rank of this process
        tsimstart = perf_counter()
        print('MPI master ({}, rank {}) on {} using {} workers\n'.format(comm.name, rank, hostname, numworkers))

        # Assemble a sys.argv replacement to pass to spawned worker
        # N.B This is required as sys.argv not available when gprMax is called via api()
        # Ignore mpicomm object if it exists as only strings can be passed via spawn
        myargv = []
        for key, value in vars(args).items():
            if value:
                # Input file name always comes first
                if 'inputfile' in key:
                    myargv.append(value)
                elif 'gpu' in key:
                    myargv.append('-' + key)
                    # If specific GPU device ID is given then add it
                    if not isinstance(value, list):
                        myargv.append(str(value.deviceID))
                elif 'mpicomm' in key:
                    pass
                elif '_' in key:
                    key = key.replace('_', '-')
                    myargv.append('--' + key)
                else:
                    myargv.append('-' + key)
                    if value is not True:
                        myargv.append(str(value))

        # Create a list of work
        worklist = []
        for model in range(modelstart, modelend):
            workobj = dict()
            workobj['currentmodelrun'] = model
            if optparams:
                workobj['optparams'] = optparams
            worklist.append(workobj)
        # Add stop sentinels
        worklist += ([StopIteration] * numworkers)

        # Spawn workers
        newcomm = comm.Spawn(sys.executable, args=['-m', 'gprMax'] + myargv + [workerflag], maxprocs=numworkers)

        # Reply to whoever asks until done
        for work in worklist:
            newcomm.recv(source=MPI.ANY_SOURCE, status=status)
            newcomm.send(obj=work, dest=status.Get_source())

        # Shutdown communicators
        newcomm.Disconnect()

        tsimend = perf_counter()
        simcompletestr = '\n=== Simulation completed in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=tsimend - tsimstart))
        print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))

    ##################
    # Worker process #
    ##################
    elif workerflag in sys.argv:
        # Connect to parent to get communicator
        try:
            comm = MPI.Comm.Get_parent()
            rank = comm.Get_rank()
        except ValueError:
            raise ValueError('MPI worker could not connect to parent')

        # Ask for work until stop sentinel
        for work in iter(lambda: comm.sendrecv(0, dest=0), StopIteration):
            currentmodelrun = work['currentmodelrun']

            # Get info and setup device ID for GPU(s)
            gpuinfo = ''
            if args.gpu is not None:
                # Set device ID for multiple GPUs
                if isinstance(args.gpu, list):
                    deviceID = (rank - 1) % len(args.gpu)
                    args.gpu = next(gpu for gpu in args.gpu if gpu.deviceID == deviceID)
                gpuinfo = ' using {} - {}, {} RAM '.format(args.gpu.deviceID, args.gpu.name, human_size(args.gpu.totalmem, a_kilobyte_is_1024_bytes=True))

            # If Taguchi optimistaion, add specific value for each parameter to
            # optimise for each experiment to user accessible namespace
            if 'optparams' in work:
                tmp = {}
                tmp.update((key, value[currentmodelrun - 1]) for key, value in work['optparams'].items())
                modelusernamespace = usernamespace.copy()
                modelusernamespace.update({'optparams': tmp})
            else:
                modelusernamespace = usernamespace

            # Run the model
            print('MPI worker (rank {}) starting model {}/{}{} on {}\n'.format(rank, currentmodelrun, numbermodelruns, gpuinfo, hostname))
            run_model(args, currentmodelrun, modelend - 1, numbermodelruns, inputfile, modelusernamespace)

        # Shutdown
        comm.Disconnect()


def run_mpi_alt_sim(args, inputfile, usernamespace, optparams=None):
    """
    Alternate MPI implementation that avoids using the spawn mechanism.
    This implementation is designed to be used as
    e.g. 'mpirun -n 5 python -m gprMax user_models/mymodel.in -n 10 -mpialt'

    Run mixed mode MPI/OpenMP simulation - MPI task farm for models with
    each model parallelised using either OpenMP (CPU) or CUDA (GPU)

    Args:
        args (dict): Namespace with command line arguments
        inputfile (object): File object for the input file.
        usernamespace (dict): Namespace that can be accessed by user in any
                Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it
                provides the parameters to optimise and their values.
    """

    from mpi4py import MPI

    # Define MPI message tags
    tags = Enum('tags', {'READY': 0, 'DONE': 1, 'EXIT': 2, 'START': 3})

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # total number of processes
    rank = comm.Get_rank()  # rank of this process
    status = MPI.Status()   # get MPI status object
    hostname = MPI.Get_processor_name()     # get name of processor/host

    # Set range for number of models to run
    modelstart = args.restart if args.restart else 1
    modelend = modelstart + args.n
    numbermodelruns = args.n
    currentmodelrun = modelstart  # can use -task argument to start numbering from something other than 1
    numworkers = size - 1

    ##################
    # Master process #
    ##################
    if rank == 0:
        tsimstart = perf_counter()
        print('MPI master ({}, rank {}) on {} using {} workers\n'.format(comm.name, rank, hostname, numworkers))

        closedworkers = 0
        while closedworkers < numworkers:
            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            # Worker is ready, so send it a task
            if tag == tags.READY.value:
                if currentmodelrun < modelend:
                    comm.send(currentmodelrun, dest=source, tag=tags.START.value)
                    currentmodelrun += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT.value)

            # Worker has completed a task
            elif tag == tags.DONE.value:
                pass

            # Worker has completed all tasks
            elif tag == tags.EXIT.value:
                closedworkers += 1

        tsimend = perf_counter()
        simcompletestr = '\n=== Simulation completed in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=tsimend - tsimstart))
        print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))

    ##################
    # Worker process #
    ##################
    else:
        while True:
            comm.send(None, dest=0, tag=tags.READY.value)
            # Receive a model number to run from the master
            currentmodelrun = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            # Run a model
            if tag == tags.START.value:

                # Get info and setup device ID for GPU(s)
                gpuinfo = ''
                if args.gpu is not None:
                    # Set device ID for multiple GPUs
                    if isinstance(args.gpu, list):
                        deviceID = (rank - 1) % len(args.gpu)
                        args.gpu = next(gpu for gpu in args.gpu if gpu.deviceID == deviceID)
                    gpuinfo = ' using {} - {}, {}'.format(args.gpu.deviceID, args.gpu.name, human_size(args.gpu.totalmem, a_kilobyte_is_1024_bytes=True))

                # If Taguchi optimistaion, add specific value for each parameter
                # to optimise for each experiment to user accessible namespace
                if optparams:
                    tmp = {}
                    tmp.update((key, value[currentmodelrun - 1]) for key, value in optparams.items())
                    modelusernamespace = usernamespace.copy()
                    modelusernamespace.update({'optparams': tmp})
                else:
                    modelusernamespace = usernamespace

                # Run the model
                print('MPI worker (rank {}) starting model {}/{}{} on {}\n'.format(rank, currentmodelrun, numbermodelruns, gpuinfo, hostname))
                run_model(args, currentmodelrun, modelend - 1, numbermodelruns, inputfile, modelusernamespace)
                comm.send(None, dest=0, tag=tags.DONE.value)

            # Break out of loop when work receives exit message
            elif tag == tags.EXIT.value:
                break

        comm.send(None, dest=0, tag=tags.EXIT.value)
