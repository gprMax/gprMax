# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
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

from .contexts import Context, MPIContext
from .utilities.logging import logging_config

# Arguments (used for API) and their default values (used for API and CLI)
args_defaults = {
    "scenes": [],
    "inputfile": None,
    "outputfile": None,
    "n": 1,
    "i": None,
    "mpi": False,
    "gpu": None,
    "opencl": None,
    "subgrid": False,
    "autotranslate": False,
    "geometry_only": False,
    "geometry_fixed": False,
    "write_processed": False,
    "log_level": 20,  # Level DEBUG = 10; INFO = 20; BASIC = 25
    "log_file": False,
}

# Argument help messages (used for CLI argparse)
help_msg = {
    "scenes": "(list, req): Scenes to run the model. Multiple scene objects "
    "can given in order to run multiple simulation runs. Each scene "
    "must contain the essential simulation objects",
    "inputfile": "(str, opt): Input file path. Can also run simulation by " "providing an input file.",
    "outputfile": "(str, req): File path to the output data file.",
    "n": "(int, req): Number of required simulation runs.",
    "i": "(int, opt): Model number to start/restart simulation from. It would "
    "typically be used to restart a series of models from a specific "
    "model number, with the n argument, e.g. to restart from A-scan 45 "
    "when creating a B-scan with 60 traces.",
    "mpi": "(bool, opt): Flag to use Message Passing Interface (MPI) task farm. "
    "This option is most usefully combined with n to allow individual "
    "models to be farmed out using a MPI task farm, e.g. to create a "
    "B-scan with 60 traces and use MPI to farm out each trace. For "
    "further details see the performance section of the User Guide.",
    "gpu": "(list/bool, opt): Flag to use NVIDIA GPU or list of NVIDIA GPU " "device ID(s) for specific GPU card(s).",
    "opencl": "(list/bool, opt): Flag to use OpenCL or list of OpenCL device " "ID(s) for specific compute device(s).",
    "subgrid": "(bool, opt): Flag to use sub-gridding.",
    "autotranslate": "(bool, opt): For sub-gridding - auto translate objects "
    "with main grid coordinates to their equivalent local "
    "grid coordinate within the subgrid. If this option is "
    "off users must specify sub-grid object point within the "
    "global subgrid space.",
    "geometry_only": "(bool, opt): Build a model and produce any geometry " "views but do not run the simulation.",
    "geometry_fixed": "(bool, opt): Run a series of models where the geometry " "does not change between models.",
    "write_processed": "(bool, opt): Writes another input file after any "
    "Python code (#python blocks) and in the original input "
    "file has been processed.",
    "log_level": "(int, opt): Level of logging to use.",
    "log_file": "(bool, opt): Write logging information to file.",
}


def run(
    scenes=args_defaults["scenes"],
    inputfile=args_defaults["inputfile"],
    outputfile=args_defaults["outputfile"],
    n=args_defaults["n"],
    i=args_defaults["i"],
    mpi=args_defaults["mpi"],
    gpu=args_defaults["gpu"],
    opencl=args_defaults["opencl"],
    subgrid=args_defaults["subgrid"],
    autotranslate=args_defaults["autotranslate"],
    geometry_only=args_defaults["geometry_only"],
    geometry_fixed=args_defaults["geometry_fixed"],
    write_processed=args_defaults["write_processed"],
    log_level=args_defaults["log_level"],
    log_file=args_defaults["log_file"],
):
    """Entry point for application programming interface (API). Runs the
        simulation for the given list of scenes.

    Args:
        scenes: list of the scenes to run the model. Multiple scene objects can
                be given in order to run multiple simulation runs. Each scene
                must contain the essential simulation objects.
        inputfile: optional string for input file path. Can also run simulation
                    by providing an input file.
        outputfile: string for file path to the output data file
        n: optional int for number of required simulation runs.
        i: optional int for model number to start/restart simulation from.
            It would typically be used to restart a series of models from a
            specific model number, with the n argument, e.g. to restart from
            A-scan 45 when creating a B-scan with 60 traces.
        mpi: optional boolean flag to use Message Passing Interface (MPI) task
                farm. This option is most usefully combined with n to allow
                individual models to be farmed out using a MPI task farm,
                e.g. to create a B-scan with 60 traces and use MPI to farm out
                each trace. For further details see the performance section of
                the User Guide
        gpu: optional list/boolean to use NVIDIA GPU or list of NVIDIA GPU device
                ID(s) for specific GPU card(s).
        opencl: optional list/boolean to use OpenCL or list of OpenCL device ID(s)
                for specific compute device(s).
        subgrid: optional boolean to use sub-gridding.
        autotranslate: optional boolean for sub-gridding to auto translate
                        objects with main grid coordinates to their equivalent
                        local grid coordinate within the subgrid. If this option
                        is off users must specify sub-grid object point within
                        the global subgrid space.
        geometry_only: optional boolean to build a model and produce any
                        geometry views but do not run the simulation.
        geometry_fixed: optional boolean to run a series of models where the
                        geometry does not change between models.
        write_processed: optional boolean to write another input file after any
                            #python blocks (which are deprecated) in the
                            original input file has been processed.
        log_level: optional int for level of logging to use.
        log_file: optional boolean to write logging information to file.
    """

    args = argparse.Namespace(
        **{
            "scenes": scenes,
            "inputfile": inputfile,
            "outputfile": outputfile,
            "n": n,
            "i": i,
            "mpi": mpi,
            "gpu": gpu,
            "opencl": opencl,
            "subgrid": subgrid,
            "autotranslate": autotranslate,
            "geometry_only": geometry_only,
            "geometry_fixed": geometry_fixed,
            "write_processed": write_processed,
            "log_level": log_level,
            "log_file": log_file,
        }
    )

    run_main(args)


def cli():
    """Entry point for command line interface (CLI)."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog="gprMax", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputfile", help=help_msg["inputfile"])
    parser.add_argument("-n", default=args_defaults["n"], type=int, help=help_msg["n"])
    parser.add_argument("-i", type=int, help=help_msg["i"])
    parser.add_argument("-mpi", action="store_true", default=args_defaults["mpi"], help=help_msg["mpi"])
    parser.add_argument("-gpu", type=int, action="append", nargs="*", help=help_msg["gpu"])
    parser.add_argument("-opencl", type=int, action="append", nargs="*", help=help_msg["opencl"])
    parser.add_argument(
        "--geometry-only", action="store_true", default=args_defaults["geometry_only"], help=help_msg["geometry_only"]
    )
    parser.add_argument(
        "--geometry-fixed",
        action="store_true",
        default=args_defaults["geometry_fixed"],
        help=help_msg["geometry_fixed"],
    )
    parser.add_argument(
        "--write-processed",
        action="store_true",
        default=args_defaults["write_processed"],
        help=help_msg["write_processed"],
    )
    parser.add_argument("--log-level", type=int, default=args_defaults["log_level"], help=help_msg["log_level"])
    parser.add_argument("--log-file", action="store_true", default=args_defaults["log_file"], help=help_msg["log_file"])
    args = parser.parse_args()

    results = run_main(args)

    return results


def run_main(args):
    """Runs simulation contexts. Called by either API or CLI.

    Args:
        args: namespace with arguments from either API or CLI.

    Returns:
        results: dict that can contain useful results/data from simulation.
                    Enables these to be propagated to calling script.
    """

    results = {}
    logging_config(level=args.log_level, log_file=args.log_file)
    config.sim_config = config.SimulationConfig(args)

    # MPI running with (OpenMP/CUDA/OpenCL)
    if config.sim_config.args.mpi:
        context = MPIContext()
    # Standard running (OpenMP/CUDA/OpenCL)
    else:
        context = Context()

    results = context.run()

    return results
