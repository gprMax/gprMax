# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
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

import datetime
import gc
import logging
import sys
from typing import Any, Dict, List, Optional

import humanize
import numpy as np
from colorama import Fore, Style, init

from gprMax.hash_cmds_file import parse_hash_commands
from gprMax.mpi_model import MPIModel
from gprMax.scene import Scene

init()

import gprMax.config as config
from gprMax.config import ModelConfig

from ._version import __version__, codename
from .model import Model
from .solvers import create_solver
from .utilities.host_info import print_cuda_info, print_host_info, print_opencl_info
from .utilities.utilities import get_terminal_width, logo, timer

logger = logging.getLogger(__name__)


class Context:
    """Standard context for building and running models.

    Models are run one after another and each model can exploit
    parallelisation using either OpenMP (CPU), CUDA (GPU), or OpenCL
    (CPU/GPU).
    """

    def __init__(self):
        self.model_range = range(config.sim_config.model_start, config.sim_config.model_end)
        self.sim_start_time = 0
        self.sim_end_time = 0

    def _start_simulation(self) -> None:
        """Run pre-simulation steps

        Start simulation timer. Output copyright notice and host info.
        """
        self.sim_start_time = timer()
        self.print_logo_copyright()
        print_host_info(config.sim_config.hostinfo)
        if config.sim_config.general["solver"] == "cuda":
            print_cuda_info(config.sim_config.devices["devs"])
        elif config.sim_config.general["solver"] == "opencl":
            print_opencl_info(config.sim_config.devices["devs"])

    def _end_simulation(self) -> None:
        """Run post-simulation steps

        Stop simulation timer. Output timing information.
        """
        self.sim_end_time = timer()
        self.print_sim_time_taken()

    def run(self) -> Dict:
        """Run the simulation in the correct context.

        Returns:
            results: dict that can contain useful results/data from simulation.
        """

        self._start_simulation()

        for i in self.model_range:
            self._run_model(i)

        self._end_simulation()

        return {}

    def _run_model(self, model_num: int) -> None:
        """Process for running a single model.

        Args:
            model_num: index of model to be run
        """

        config.sim_config.set_current_model(model_num)
        model_config = self._create_model_config(model_num)
        config.sim_config.set_model_config(model_config)

        if not model_config.reuse_geometry():
            scene = self._get_scene(model_num)
            model = self._create_model()
            scene.create_internal_objects(model)

        model.build()

        if not config.sim_config.geometry_only:
            solver = create_solver(model)
            model.solve(solver)
            del solver

        if not config.sim_config.geometry_fixed:
            # Manual garbage collection required to stop memory leak on GPUs
            # when using pycuda
            del model.G, model

        gc.collect()

    def _create_model_config(self, model_num: int) -> ModelConfig:
        """Create model config and save to global config."""
        return ModelConfig(model_num)

    def _get_scene(self, model_num: int) -> Scene:
        # API for multiple scenes / model runs
        scene = config.sim_config.get_scene(model_num)

        # If there is no scene, process the hash commands
        if scene is None:
            scene = Scene()
            config.sim_config.set_scene(scene, model_num)
            # Parse the input file into user objects and add them to the scene
            scene = parse_hash_commands(scene)

        return scene

    def _create_model(self) -> Model:
        return Model()

    def print_logo_copyright(self) -> None:
        """Prints gprMax logo, version, and copyright/licencing information."""
        logo_copyright = logo(f"{__version__} ({codename})")
        logger.basic(logo_copyright)

    def print_sim_time_taken(self) -> None:
        """Prints the total simulation time based on context."""
        s = (
            "=== Simulation completed in "
            f"{humanize.precisedelta(datetime.timedelta(seconds=self.sim_end_time - self.sim_start_time), format='%0.4f')}"
        )
        logger.basic(f"{s} {'=' * (get_terminal_width() - 1 - len(s))}\n")


class MPIContext(Context):
    def __init__(self):
        super().__init__()
        from mpi4py import MPI

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank

        requested_mpi_size = np.prod(config.sim_config.mpi)
        if self.comm.size < requested_mpi_size:
            raise ValueError(
                f"MPI_COMM_WORLD size of {self.comm.size} is too small for requested dimensions of"
                f" {config.sim_config.mpi}. {requested_mpi_size} ranks are required."
            )

        if self.rank >= requested_mpi_size:
            logger.warning(
                f"Rank {self.rank}: Only {requested_mpi_size} MPI ranks required for the"
                " dimensions specified. Either increase your MPI dimension size, or request"
                " fewer MPI tasks."
            )
            exit()

    def _create_model(self) -> MPIModel:
        return MPIModel()

    def run(self) -> Dict:
        try:
            return super().run()
        except:
            logger.exception(f"Rank {self.rank} encountered an error. Aborting...")
            self.comm.Abort()

    def _run_model(self, model_num: int) -> None:
        """Process for running a single model.

        Args:
            model_num: index of model to be run
        """

        config.sim_config.set_current_model(model_num)
        model_config = self._create_model_config(model_num)
        config.sim_config.set_model_config(model_config)

        if not model_config.reuse_geometry():
            model = self._create_model()
            scene = self._get_scene(model_num)
            scene.create_internal_objects(model)

        model.build()

        if not config.sim_config.geometry_only:
            solver = create_solver(model)
            model.solve(solver)
            del solver

        if not config.sim_config.geometry_fixed:
            # Manual garbage collection required to stop memory leak on GPUs
            # when using pycuda
            del model.G, model

        gc.collect()


class TaskfarmContext(Context):
    """Mixed mode MPI/OpenMP/CUDA context - MPI task farm is used to distribute
    models, and each model parallelised using either OpenMP (CPU),
    CUDA (GPU), or OpenCL (CPU/GPU).
    """

    def __init__(self):
        super().__init__()
        from mpi4py import MPI

        from gprMax.taskfarm import TaskfarmExecutor

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.TaskfarmExecutor = TaskfarmExecutor

    def _create_model_config(self, model_num: int) -> ModelConfig:
        """Create model config and save to global config.

        Set device in model config according to MPI rank.
        """
        model_config = super()._create_model_config(model_num)
        # Set GPU deviceID according to worker rank
        if config.sim_config.general["solver"] == "cuda":
            model_config.device = {
                "dev": config.sim_config.devices["devs"][self.rank - 1],
                "deviceID": self.rank - 1,
                "snapsgpu2cpu": False,
            }
        return model_config

    def _run_model(self, **work) -> None:
        """Process for running a single model.

        Args:
            work: dict of any additional information that is passed to MPI
                workers. By default only model number (i) is used.
        """
        return super()._run_model(work["i"])

    def run(self) -> Optional[List[Optional[Dict]]]:
        """Specialise how the models are run.

        Returns:
            results: dict that can contain useful results/data from simulation.
        """

        if self.rank == 0:
            self._start_simulation()

            s = f"\n--- Input file: {config.sim_config.input_file_path}"
            logger.basic(
                Fore.GREEN + f"{s} {'-' * (get_terminal_width() - 1 - len(s))}\n" + Style.RESET_ALL
            )

            sys.stdout.flush()

        # Contruct TaskfarmExecutor
        executor = self.TaskfarmExecutor(self._run_model, comm=self.comm)

        # Check GPU resources versus number of MPI tasks
        if (
            executor.is_master()
            and config.sim_config.general["solver"] == "cuda"
            and executor.size - 1 > len(config.sim_config.devices["devs"])
        ):
            logger.error(
                "Not enough GPU resources for number of "
                "MPI tasks requested. Number of MPI tasks "
                "should be equal to number of GPUs + 1."
            )
            raise ValueError

        jobs = [{"i": i} for i in self.model_range]
        # Send the workers to their work loop
        executor.start()
        if executor.is_master():
            results = executor.submit(jobs)

        # Make the workers exit their work loop and join the main loop again
        executor.join()

        if executor.is_master():
            self._end_simulation()
            return results
