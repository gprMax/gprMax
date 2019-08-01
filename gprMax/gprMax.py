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

def api(
    scenes=None,
    inputfile=None,
    n=1,
    task=None,
    restart=None,
    mpi=False,
    mpi_no_spawn=False,
    mpicomm=None,
    gpu=None,
    benchmark=False,
    geometry_only=False,
    geometry_fixed=False,
    write_processed=False,
):
    """If installed as a module this is the entry point."""

    class ImportArguments:
        pass

    args = ImportArguments()

    args.scenes = scenes
    args.inputfile = inputfile
    args.n = n
    args.task = task
    args.restart = restart
    args.mpi = mpi
    args.mpi_no_spawn = mpi_no_spawn
    args.mpicomm = mpicomm
    args.gpu = gpu
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
    solver = create_solver(sim_config)
    context = create_context(sim_config, solver)
    context.run()
