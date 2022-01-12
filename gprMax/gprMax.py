# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

import gprMax.config as config

from .contexts import Context, MPIContext, SPOTPYContext
from .utilities.logging import logging_config

# Arguments (used for API) and their default values (used for API and CLI)
args_defaults = {'scenes': None,
                 'inputfile': None,
                 'outputfile': None,
                 'n': 1,
                 'task': None,
                 'restart': None,
                 'mpi': False,
                 'gpu': None,
                 'subgrid': False,
                 'autotranslate': False,
                 'geometry_only': False,
                 'geometry_fixed': False,
                 'write_processed': False,
                 'log_level': 20,
                 'log_file': False}

# Argument help messages (used for CLI argparse)
help_msg = {'scenes': '(list, opt): List of the scenes to run the model. '
                      'Multiple scene objects can given in order to run multiple '
                      'simulation runs. Each scene must contain the essential '
                      'simulation objects',
            'inputfile': '(str, opt): Input file path. Can also run simulation '
                         'by providing an input file.',
            'outputfile': '(str, opt): File path to the output data file.',
            'n': '(int, req): Number of required simulation runs.',
            'task': '(int, opt): Task identifier (model number) when running '
                    'simulation as a job array on Open Grid Scheduler/Grid '
                    'Engine (http://gridscheduler.sourceforge.net/index.html). '
                    'For further details see the parallel performance '
                    'section of the User Guide.',
            'restart': '(int, opt): Model number to start/restart simulation '
                       'from. It would typically be used to restart a series of '
                       'models from a specific model number, with the n argument, '
                       'e.g. to restart from A-scan 45 when creating a B-scan '
                       'with 60 traces.',
            'mpi': '(bool, opt): Flag to use Message Passing Interface (MPI) '
                   'task farm. This option is most usefully combined with n to '
                   'allow individual models to be farmed out using a MPI task '
                   'farm, e.g. to create a B-scan with 60 traces and use MPI to '
                   'farm out each trace. For further details see the parallel '
                   'performance section of the User Guide.',
            'gpu': '(list/bool, opt): Flag to use NVIDIA GPU or list of NVIDIA '
                   'GPU device ID(s) for specific GPU card(s).',
            'subgrid': '(bool, opt): Flag to use sub-gridding.',
            'autotranslate': '(bool, opt): For sub-gridding - auto translate '
                             'objects with main grid coordinates to their '
                             'equivalent local grid coordinate within the '
                             'subgrid. If this option is off users must specify '
                             'sub-grid object point within the global subgrid space.',
            'geometry_only': '(bool, opt): Build a model and produce any '
                             'geometry views but do not run the simulation.',
            'geometry_fixed': '(bool, opt): Run a series of models where the '
                              'geometry does not change between models.',
            'write_processed': '(bool, opt): Writes another input file after '
                               'any Python code (#python blocks) and in the '
                               'original input file has been processed.',
            'log_level': '(int, opt): Level of logging to use.',
            'log_file': '(bool, opt): Write logging information to file.'}


def run(scenes=args_defaults['scenes'],
        inputfile=args_defaults['inputfile'],
        outputfile=args_defaults['outputfile'],
        n=args_defaults['n'],
        task=args_defaults['task'],
        restart=args_defaults['restart'],
        mpi=args_defaults['mpi'],
        gpu=args_defaults['gpu'],
        subgrid=args_defaults['subgrid'],
        autotranslate=args_defaults['autotranslate'],
        geometry_only=args_defaults['geometry_only'],
        geometry_fixed=args_defaults['geometry_fixed'],
        write_processed=args_defaults['write_processed'],
        log_level=args_defaults['log_level'],
        log_file=args_defaults['log_file']):
    """This is the main function for gprMax when entering using application
        programming interface (API). Run the simulation for the given list of 
        scenes.     
    """

    args = argparse.Namespace(**{'scenes': scenes,
                                 'inputfile': inputfile,
                                 'outputfile': outputfile,
                                 'n': n,
                                 'task': task,
                                 'restart': restart,
                                 'mpi': mpi,
                                 'gpu': gpu,
                                 'subgrid': subgrid,
                                 'autotranslate': autotranslate,
                                 'geometry_only': geometry_only,
                                 'geometry_fixed': geometry_fixed,
                                 'write_processed': write_processed,
                                 'log_level': log_level,
                                 'log_file': log_file})

    run_main(args)


def cli():
    """Main function for gprMax when entering using the command line interface 
        (CLI).
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='gprMax', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile', help=help_msg['inputfile'])
    parser.add_argument('-n', default=args_defaults['n'], type=int, help=help_msg['n'])
    parser.add_argument('-task', type=int, help=help_msg['task'])
    parser.add_argument('-r', '--restart', type=int, help=help_msg['restart'])
    parser.add_argument('-mpi', action='store_true', default=args_defaults['mpi'],
                        help=help_msg['mpi'])
    parser.add_argument('-gpu', type=int, action='append', nargs='*',
                        help=help_msg['gpu'])
    parser.add_argument('--geometry-only', action='store_true', 
                        default=args_defaults['geometry_only'],
                        help=help_msg['geometry_only'])
    parser.add_argument('--geometry-fixed', action='store_true', 
                        default=args_defaults['geometry_fixed'],
                        help=help_msg['geometry_fixed'])
    parser.add_argument('--write-processed', action='store_true', 
                        default=args_defaults['write_processed'],
                        help=help_msg['write_processed'])
    parser.add_argument('--log-level', type=int, 
                        default=args_defaults['log_level'],
                        help=help_msg['log_level'])
    parser.add_argument('--log-file', action='store_true', 
                        default=args_defaults['log_file'],
                        help=help_msg['log_file'])
    args = parser.parse_args()

    run_main(args)


def run_main(args):
    """Called by either run (API) or main (CLI).

    Args:
        args (Namespace): arguments from either API or CLI.
    """

    logging_config(level=args.log_level, log_file=args.log_file)

    config.sim_config = config.SimulationConfig(args)
    
    # If integrating with SPOTPY (https://github.com/thouska/spotpy) - extra 
    # 'spotpy' attribute is added to args when called by SPOTPY
    if hasattr(args, 'spotpy'):
        if args.spotpy:
            context = SPOTPYContext()
            context.run(args.i)
    # MPI running with (OpenMP/CUDA)
    elif config.sim_config.args.mpi:
        context = MPIContext()
        context.run()
    # Standard running (OpenMP/CUDA)
    else:
        context = Context()
        context.run()
