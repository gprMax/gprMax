# Copyright (C) 2015-2017: The University of Edinburgh
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
import os
from time import perf_counter
import sys

import h5py
import numpy as np

from gprMax._version import __version__, codename
from gprMax.constants import c, e0, m0, z0
from gprMax.exceptions import GeneralError
from gprMax.model_build_run import run_model
from gprMax.utilities import get_host_info, get_terminal_width, human_size, logo, open_path_file


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
    parser.add_argument('--mpi-worker', action='store_true', default=False, help=argparse.SUPPRESS)
    parser.add_argument('-benchmark', action='store_true', default=False, help='flag to switch on benchmarking mode')
    parser.add_argument('--geometry-only', action='store_true', default=False, help='flag to only build model and produce geometry file(s)')
    parser.add_argument('--geometry-fixed', action='store_true', default=False, help='flag to not reprocess model geometry, e.g. for B-scans where the geometry is fixed')
    parser.add_argument('--write-processed', action='store_true', default=False, help='flag to write an input file after any Python code and include commands in the original input file have been processed')
    parser.add_argument('--opt-taguchi', action='store_true', default=False, help='flag to optimise parameters using the Taguchi optimisation method')
    args = parser.parse_args()

    run_main(args)


def api(inputfile, n=1, task=None, restart=None, mpi=False, benchmark=False, geometry_only=False, geometry_fixed=False, write_processed=False, opt_taguchi=False):
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
    
    with open_path_file(args.inputfile) as inputfile:

        # Get information about host machine
        hostinfo = get_host_info()
        hyperthreading = ', {} cores with Hyper-Threading'.format(hostinfo['logicalcores']) if hostinfo['hyperthreading'] else ''
        print('\nHost: {}; {} x {} ({} cores{}); {} RAM; {}'.format(hostinfo['machineID'], hostinfo['sockets'], hostinfo['cpuID'], hostinfo['physicalcores'], hyperthreading, human_size(hostinfo['ram'], a_kilobyte_is_1024_bytes=True), hostinfo['osversion']))

        # Create a separate namespace that users can access in any Python code blocks in the input file
        usernamespace = {'c': c, 'e0': e0, 'm0': m0, 'z0': z0, 'number_model_runs': args.n, 'input_directory': os.path.dirname(os.path.abspath(inputfile.name))}

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
            if args.mpi_worker: # Special case for MPI spawned workers - they do not need to enter the Taguchi optimisation mode
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
        
            # Standard behaviour - models run serially with each model parallelised with OpenMP (CPU) or CUDA (GPU)
            else:
                if args.task and args.restart:
                    raise GeneralError('Job array and restart modes cannot be used together')
                run_std_sim(args, inputfile, usernamespace)


def run_std_sim(args, inputfile, usernamespace, optparams=None):
    """Run standard simulation - models are run one after another and each model is parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        inputfile (object): File object for the input file.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it provides the parameters to optimise and their values.
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
        if optparams:  # If Taguchi optimistaion, add specific value for each parameter to optimise for each experiment to user accessible namespace
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
    """Run standard simulation in benchmarking mode - models are run one after another and each model is parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        inputfile (object): File object for the input file.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
    """

    # Get information about host machine
    hostinfo = get_host_info()
    hyperthreading = ', {} cores with Hyper-Threading'.format(hostinfo['logicalcores']) if hostinfo['hyperthreading'] else ''
    machineIDlong = '{}; {} x {} ({} cores{}); {} RAM; {}'.format(hostinfo['machineID'], hostinfo['sockets'], hostinfo['cpuID'], hostinfo['physicalcores'], hyperthreading, human_size(hostinfo['ram'], a_kilobyte_is_1024_bytes=True), hostinfo['osversion'])

    # Number of CPU threads to benchmark - start from single thread and double threads until maximum number of physical cores
    threads = 1
    maxthreads = hostinfo['physicalcores']
    maxthreadspersocket = hostinfo['physicalcores'] / hostinfo['sockets']
    cputhreads = np.array([], dtype=np.int32)
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
    modelend = numbermodelruns + 1
                
    usernamespace['number_model_runs'] = numbermodelruns

    for currentmodelrun in range(1, modelend):
        os.environ['OMP_NUM_THREADS'] = str(cputhreads[currentmodelrun - 1])
        cputimes[currentmodelrun - 1] = run_model(args, currentmodelrun, modelend - 1, numbermodelruns, inputfile, usernamespace)

        # Get model size (in cells) and number of iterations
        if currentmodelrun == 1:
            if numbermodelruns == 1:
                outputfile = os.path.splitext(args.inputfile)[0] + '.out'
            else:
                outputfile = os.path.splitext(args.inputfile)[0] + str(currentmodelrun) + '.out'
            f = h5py.File(outputfile, 'r')
            iterations = f.attrs['Iterations']
            numcells = f.attrs['nx, ny, nz']

    # Save number of threads and benchmarking times to NumPy archive
    np.savez(os.path.splitext(inputfile.name)[0], machineID=machineIDlong, gpuIDs=[], cputhreads=cputhreads, cputimes=cputimes, gputimes=[], iterations=iterations, numcells=numcells, version=__version__)

    simcompletestr = '\n=== Simulation completed'
    print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))


def run_mpi_sim(args, inputfile, usernamespace, optparams=None):
    """Run mixed mode MPI/OpenMP simulation - MPI task farm for models with each model parallelised with OpenMP

    Args:
        args (dict): Namespace with command line arguments
        inputfile (object): File object for the input file.
        usernamespace (dict): Namespace that can be accessed by user in any Python code blocks in input file.
        optparams (dict): Optional argument. For Taguchi optimisation it provides the parameters to optimise and their values.
    """

    from mpi4py import MPI
    
    # Get name of processor/host
    name = MPI.Get_processor_name()

    # Set range for number of models to run
    modelstart = args.restart if args.restart else 1
    modelend = modelstart + args.n
    numbermodelruns = args.n
    
    # Number of workers and command line flag to indicate a spawned worker
    worker = '--mpi-worker'
    numberworkers = args.mpi - 1

    # Master process
    if worker not in sys.argv:
        
        tsimstart = perf_counter()
        
        print('MPI master rank (PID {}) on {} using {} workers'.format(os.getpid(), name, numberworkers))
        
        # Create a list of work
        worklist = []
        for model in range(modelstart, modelend):
            workobj = dict()
            workobj['currentmodelrun'] = model
            if optparams:
                workobj['optparams'] = optparams
            worklist.append(workobj)
        # Add stop sentinels
        worklist += ([StopIteration] * numberworkers)

        # Spawn workers
        comm = MPI.COMM_WORLD.Spawn(sys.executable, args=['-m', 'gprMax', '-n', str(args.n)] + sys.argv[1::] + [worker], maxprocs=numberworkers)

        # Reply to whoever asks until done
        status = MPI.Status()
        for work in worklist:
            comm.recv(source=MPI.ANY_SOURCE, status=status)
            comm.send(obj=work, dest=status.Get_source())

        # Shutdown
        comm.Disconnect()

        tsimend = perf_counter()
        simcompletestr = '\n=== Simulation completed in [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=tsimend - tsimstart))
        print('{} {}\n'.format(simcompletestr, '=' * (get_terminal_width() - 1 - len(simcompletestr))))

    # Worker process
    elif worker in sys.argv:

        # Connect to parent
        try:
            comm = MPI.Comm.Get_parent() # get MPI communicator object
            rank = comm.Get_rank()  # rank of this process
        except:
            raise ValueError('Could not connect to parent')

        # Ask for work until stop sentinel
        for work in iter(lambda: comm.sendrecv(0, dest=0), StopIteration):
            currentmodelrun = work['currentmodelrun']
            
            gpuinfo = ''
            print('MPI worker rank {} (PID {}) starting model {}/{}{} on {}'.format(rank, os.getpid(), currentmodelrun, numbermodelruns, gpuinfo, name))
            
            # If Taguchi optimistaion, add specific value for each parameter to optimise for each experiment to user accessible namespace
            if 'optparams' in work:
                tmp = {}
                tmp.update((key, value[currentmodelrun - 1]) for key, value in work['optparams'].items())
                modelusernamespace = usernamespace.copy()
                modelusernamespace.update({'optparams': tmp})
            else:
                modelusernamespace = usernamespace

            # Run the model
            run_model(args, currentmodelrun, modelend - 1, numbermodelruns, inputfile, modelusernamespace)

        # Shutdown
        comm.Disconnect()
