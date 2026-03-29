# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
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
from .contexts import Context, MPIContext, TaskfarmContext
from .utilities.logging import logging_config

# Arguments (used for API) and their default values (used for API and CLI)
args_defaults = {
    "scenes": None,
    "inputfile": None,
    "outputfile": None,
    "n": 1,
    "i": None,
    "taskfarm": False,
    "mpi": None,
    "gpu": None,
    "opencl": None,
    "metal": None,
    "subgrid": False,
    "autotranslate": False,
    "geometry_only": False,
    "geometry_fixed": False,
    "write_processed": False,
    "show_progress_bars": False,
    "hide_progress_bars": False,
    "log_level": 20,
    "log_file": False,
    "log_all_ranks": False,
    "interactive": False,  # Your GSoC addition
}

# Argument help messages
help_msg = {
    "inputfile": "(str, req): Input file path.",
    "outputfile": "(str, opt): File path to the output data file.",
    "n": "(int, opt): Number of required simulation runs.",
    "i": "(int, opt): Model number to start/restart simulation from.",
    "taskfarm": "(bool, opt): Flag to use MPI task farm.",
    "mpi": "(list, opt): Flag to use MPI to divide the model.",
    "gpu": "(list/bool, opt): Flag to use NVIDIA GPU.",
    "opencl": "(list/bool, opt): Flag to use OpenCL.",
    "metal": "(list/bool, opt): Flag to use Apple Metal.",
    "subgrid": "(bool, opt): Flag to use sub-gridding.",
    "autotranslate": "(bool, opt): For sub-gridding - auto translate objects.",
    "geometry_only": "(bool, opt): Build model but do not run simulation.",
    "geometry_fixed": "(bool, opt): Geometry does not change between models.",
    "write_processed": "(bool, opt): Writes processed input file.",
    "show_progress_bars": "(bool, opt): Forces progress bars to be displayed.",
    "hide_progress_bars": "(bool, opt): Forces progress bars to be hidden.",
    "log_level": "(int, opt): Level of logging to use.",
    "log_file": "(bool, opt): Write logging information to file.",
    "log_all_ranks": "(bool, opt): Write logging information from all MPI ranks.",
    "interactive": "(bool, opt): Flag to launch the Interactive B-Scan Viewer.",
}

def run(
    scenes=args_defaults["scenes"],
    inputfile=args_defaults["inputfile"],
    outputfile=args_defaults["outputfile"],
    n=args_defaults["n"],
    i=args_defaults["i"],
    taskfarm=args_defaults["taskfarm"],
    mpi=args_defaults["mpi"],
    gpu=args_defaults["gpu"],
    opencl=args_defaults["opencl"],
    metal=args_defaults["metal"],
    subgrid=args_defaults["subgrid"],
    autotranslate=args_defaults["autotranslate"],
    geometry_only=args_defaults["geometry_only"],
    geometry_fixed=args_defaults["geometry_fixed"],
    write_processed=args_defaults["write_processed"],
    show_progress_bars=args_defaults["show_progress_bars"],
    hide_progress_bars=args_defaults["hide_progress_bars"],
    log_level=args_defaults["log_level"],
    log_file=args_defaults["log_file"],
    log_all_ranks=args_defaults["log_all_ranks"],
    interactive=args_defaults["interactive"], # Your GSoC addition
):
    args = argparse.Namespace(
        **{
            "scenes": scenes,
            "inputfile": inputfile,
            "outputfile": outputfile,
            "n": n,
            "i": i,
            "taskfarm": taskfarm,
            "mpi": mpi,
            "gpu": gpu,
            "opencl": opencl,
            "metal": metal,
            "subgrid": subgrid,
            "autotranslate": autotranslate,
            "geometry_only": geometry_only,
            "geometry_fixed": geometry_fixed,
            "write_processed": write_processed,
            "show_progress_bars": show_progress_bars,
            "hide_progress_bars": hide_progress_bars,
            "log_level": log_level,
            "log_file": log_file,
            "log_all_ranks": log_all_ranks,
            "interactive": interactive, # Your GSoC addition
        }
    )
    return run_main(args)

def cli():
    parser = argparse.ArgumentParser(
        prog="gprMax", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("inputfile", help=help_msg["inputfile"])
    parser.add_argument("-outputfile", "-o", help=help_msg["outputfile"])
    parser.add_argument("-n", default=args_defaults["n"], type=int, help=help_msg["n"])
    parser.add_argument("-i", type=int, help=help_msg["i"])
    parser.add_argument("--taskfarm", "-t", action="store_true", default=args_defaults["taskfarm"], help=help_msg["taskfarm"])
    parser.add_argument("--mpi", type=int, action="store", nargs=3, default=args_defaults["mpi"], help=help_msg["mpi"])
    parser.add_argument("-gpu", type=int, action="append", nargs="*", help=help_msg["gpu"])
    parser.add_argument("-opencl", type=int, action="append", nargs="*", help=help_msg["opencl"])
    parser.add_argument("-metal", type=int, action="append", nargs="*", help=help_msg["metal"])
    parser.add_argument("--geometry-only", action="store_true", default=args_defaults["geometry_only"], help=help_msg["geometry_only"])
    parser.add_argument("--geometry-fixed", action="store_true", default=args_defaults["geometry_fixed"], help=help_msg["geometry_fixed"])
    parser.add_argument("--write-processed", action="store_true", default=args_defaults["write_processed"], help=help_msg["write_processed"])
    parser.add_argument("--show-progress-bars", action="store_true", default=args_defaults["show_progress_bars"], help=help_msg["show_progress_bars"])
    parser.add_argument("--hide-progress-bars", action="store_true", default=args_defaults["hide_progress_bars"], help=help_msg["hide_progress_bars"])
    parser.add_argument("--log-level", type=int, default=args_defaults["log_level"], help=help_msg["log_level"])
    parser.add_argument("--log-file", action="store_true", default=args_defaults["log_file"], help=help_msg["log_file"])
    parser.add_argument("--log-all-ranks", action="store_true", default=args_defaults["log_all_ranks"], help=help_msg["log_all_ranks"])
    
    # Your GSoC Addition
    parser.add_argument("--interactive", action="store_true", default=args_defaults["interactive"], help=help_msg["interactive"])

    args = parser.parse_args()
    return run_main(args)

def run_main(args):
    logging_config(
        level=args.log_level,
        log_file=args.log_file,
        mpi_logger=args.mpi is not None,
        log_all_ranks=args.log_all_ranks,
    )
    config.sim_config = config.SimulationConfig(args)

    if config.sim_config.args.taskfarm:
        context = TaskfarmContext()
    elif config.sim_config.args.mpi is not None:
        context = MPIContext()
    else:
        context = Context()

    results = context.run()

    # --- Viewer Logic ---
    if args.interactive:
        # We import here to avoid dependency issues if the viewer isn't needed
        from gprMax.utilities.bscan_viewer import InteractiveBScanViewer 
        viewer = InteractiveBScanViewer(args.inputfile)
        viewer.show()
    # --- End of Viewer Logic ---

    return results