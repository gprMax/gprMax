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

import argparse
import logging

from .config_parser import write_simulation_config
from .contexts import create_context
from .utilities import setup_logging

logger = logging.getLogger(__name__)


def run(
    scenes=None,
    inputfile=None,
    outputfile=None,
    n=1,
    task=None,
    restart=None,
    mpi=False,
    gpu=None,
    subgrid=None,
    autotranslate=False,
    geometry_only=False,
    geometry_fixed=False,
    write_processed=False,
):
    """This is the main function for gprMax when entering using application
        programming interface (API). Run the simulation for the
        given list of scenes.

    :param scenes: list of the scenes to run the model. Multiple scene objects
                    can given in order to run multiple simulation runs. Each
                    scene must contain the essential simulation objects
    :type scenes: list, optional

    :param inputfile:  input file path. Can also run simulation by providing an
                        input file.
    :type inputfile: str, optional

    :param outputfile: file path to the output data file.
    :type outputfile: str, non-optional

    :param n: number of required simulation runs.
    :type n: int, non-optional

    :param task: task identifier (model number) when running simulation as a
                    job array on open grid scheduler/grid engine. For further
                    details see the parallel performance section of the User Guide.
    :type task: int, optional

    :param restart: model number to start/restart simulation from. It would
                    typically be used to restart a series of models from a
                    specific model number, with the n argument, e.g. to restart
                    from A-scan 45 when creating a B-scan with 60 traces.
    :type restart: int, optional

    :param mpi: flag to use Message Passing Interface (MPI) task farm. This
                option is most usefully combined with n to allow individual
                models to be farmed out using a MPI task farm, e.g. to create a
                B-scan with 60 traces and use MPI to farm out each trace.
                For further details see the parallel performance section of the
                User Guide.
    :type mpi: bool, optional

    :param gpu: flag to use NVIDIA GPU or list of NVIDIA GPU device ID(s) for
                specific GPU card(s).
    :type gpu: list or bool, optional

    :param subgrid: flag to use sub-gridding.
    :type subgrid: bool, optional

    :param autotranslate: for sub-gridding - auto translate objects with main grid
                            coordinates to their equivalent local grid coordinate
                            within the subgrid. If this option is off users must
                            specify sub-grid object point within the global
                            subgrid space.
    :type autotranslate: bool, optional

    :param geometry_only: build a model and produce any geometry views but do
                            not run the simulation.
    :type geometry_only: bool, optional

    :param geometry_fixed: run a series of models where the geometry does not
                            change between models.
    :type geometry_fixed: bool, optional

    :param write_processed: write another input file after any Python code and
                            in the original input file has been processed.
    :type write_processed: bool, optional
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
    args.gpu = gpu
    args.subgrid = subgrid
    args.autotranslate = autotranslate
    args.geometry_only = geometry_only
    args.geometry_fixed = geometry_fixed
    args.write_processed = write_processed

    try:
        run_main(args)
    except Exception:
        logger.exception('Error from main API function', exc_info=True)


def main():
    """Main function for gprMax when entering using the command line interface (CLI)."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='gprMax', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile',
                        help='relative or absolute path to inputfile')
    parser.add_argument('-n', default=1, type=int,
                        help='number of times to run the input file, e.g. to create a B-scan')
    parser.add_argument('-task', type=int,
                        help='task identifier (model number) for job array on '
                        'Open Grid Scheduler/Grid Engine (http://gridscheduler.sourceforge.net/index.html)')
    parser.add_argument('-r', '--restart', type=int,
                        help='model number to restart from, e.g. when creating B-scan')
    parser.add_argument('-mpi', action='store_true', default=False,
                        help='flag to enable MPI task farming')
    parser.add_argument('-gpu', type=int, action='append', nargs='*',
                        help='flag to use Nvidia GPU or option to give list of device ID(s)')
    parser.add_argument('--geometry-only', action='store_true', default=False,
                        help='flag to only build model and produce geometry file(s)')
    parser.add_argument('--geometry-fixed', action='store_true', default=False,
                        help='flag to not reprocess model geometry, e.g. for B-scans where the geometry is fixed')
    parser.add_argument('--write-processed', action='store_true', default=False,
                        help='flag to write an input file after any Python code and include commands '
                        'in the original input file have been processed')
    parser.add_argument('-l', '--logfile', action='store_true', default=False,
                        help='flag to enable writing to a log file')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="flag to increase output")
    args = parser.parse_args()

    setup_logging()

    try:
        run_main(args)
    except Exception:
        logger.exception('Error from main CLI function', exc_info=True)


def run_main(args):
    """Called by either run (API) or main (CLI).

    Args:
        args (Namespace): arguments from either API or CLI.
    """

    write_simulation_config(args)
    context = create_context()
    context.run()
