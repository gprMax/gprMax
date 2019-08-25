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

from .config import create_simulation_config
from .contexts import create_context
from .solvers import create_solver

import argparse

def run(
    scenes=None,
    inputfile=None,
    outputfile=None,
    n=1,
    task=None,
    restart=None,
    mpi=False,
    mpi_no_spawn=False,
    mpicomm=None,
    gpu=None,
    subgrid=None,
    benchmark=False,
    geometry_only=False,
    geometry_fixed=False,
    write_processed=False,
):
    """Run the simulation for the given list of scenes.

    :param scenes: List of the scenes to run the model. Multiple scene objects can given in order to run multiple simulation runs. Each scene must contain the essential simulation objects
    :type scenes: list, optional
    :param inputfile:  Input file path. Can also run simulation by providing an input file.
    :type inputfile: str, optional
    :param outputfile: File path to the output data file.
    :type outputfile: str, non-optional
    :param n: Number of required simulation runs.
    :type n: int, non-optional
    :param task: task identifier (model number) when running simulation as a job array on open grid scheduler/grid engine. for further details see the parallel performance section of the user guide
    :type task: int, optional
    :param restart: model number to start/restart simulation from. It would typically be used to restart a series of models from a specific model number, with the n argument, e.g. to restart from A-scan 45 when creating a B-scan with 60 traces
    :type restart: int, optional
    :param mpi: number of Message Passing Interface (MPI) tasks, i.e. master + workers, for MPI task farm. This option is most usefully combined with n to allow individual models to be farmed out using a MPI task farm, e.g. to create a B-scan with 60 traces and use MPI to farm out each trace1. For further details see the parallel performance section of the User Guide
    :type mpi: int, optional
    :param mpi_no_spawn: use MPI task farm without spawn mechanism. For further details see the parallel performance section of the User Guide.
    :type mpi_no_spawn: bool, optional
    :param gpu: Flag to use NVIDIA GPU or list of NVIDIA GPU device ID(s) for specific GPU card(s)
    :type gpu: list or bool, optional
    :param subgrid: Use sub-gridding.
    :type subgrid: bool, optional
    :param benchmark: Switch on benchmarking mode. This can be used to benchmark the threading (parallel) performance of gprMax on different hardware. For further details see the benchmarking section of the User Guide
    :type benchmark: bool, optional
    :param geometry_only: build a model and produce any geometry views but do not run the simulation.
    :type geometry_only: bool, optional
    :param geometry_fixed: build a model and produce any geometry views but do not run the simulation.
    :type geometry_fixed: bool, optional

    """
    class ImportArguments:
        pass

    args = ImportArguments()

    args.scenes = scenes
    args.inputfile = inputfile
    args.outputfile = outputfile
    args.n = n
    args.task = task
    args.restart = restart
    args.mpi = mpi
    args.mpi_no_spawn = mpi_no_spawn
    args.mpicomm = mpicomm
    args.gpu = gpu
    args.subgrid=subgrid
    args.benchmark = benchmark
    args.geometry_only = geometry_only
    args.geometry_fixed = geometry_fixed
    args.write_processed = write_processed

    run_main(args)


def main():
    """This is the main function for gprMax."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='gprMax', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile', help='path to, and name of inputfile or file object')
    parser.add_argument('-n', default=1, type=int, help='number of times to run the input file, e.g. to create a B-scan')
    parser.add_argument('-task', type=int, help='task identifier (model number) for job array on Open Grid Scheduler/Grid Engine (http://gridscheduler.sourceforge.net/index.html)')
    parser.add_argument('-restart', type=int, help='model number to restart from, e.g. when creating B-scan')
    parser.add_argument('-mpi', type=int, help='number of MPI tasks, i.e. master + workers')
    parser.add_argument('--mpi-no-spawn', action='store_true', default=False, help='flag to use MPI without spawn mechanism')
    parser.add_argument('--mpi-worker', action='store_true', default=False, help=argparse.SUPPRESS)
    parser.add_argument('-gpu', type=int, action='append', nargs='*', help='flag to use Nvidia GPU or option to give list of device ID(s)')
    parser.add_argument('-benchmark', action='store_true', default=False, help='flag to switch on benchmarking mode')
    parser.add_argument('--geometry-only', action='store_true', default=False, help='flag to only build model and produce geometry file(s)')
    parser.add_argument('--geometry-fixed', action='store_true', default=False, help='flag to not reprocess model geometry, e.g. for B-scans where the geometry is fixed')
    parser.add_argument('--write-processed', action='store_true', default=False, help='flag to write an input file after any Python code and include commands in the original input file have been processed')
    args = parser.parse_args()

    run_main(args)

def run_main(args):

    sim_config = create_simulation_config(args)
    context = create_context(sim_config)
    context.run()
