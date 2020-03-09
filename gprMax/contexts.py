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

import datetime
import logging

import gprMax.config as config
from ._version import __version__, codename
from .model_build_run import ModelBuildRun
from .solvers import create_solver
from .solvers import create_G
from .utilities import get_terminal_width
from .utilities import human_size
from .utilities import logo
from .utilities import timer

logger = logging.getLogger(__name__)


class Context:
    """Standard context - models are run one after another and each model
        can exploit parallelisation using either OpenMP (CPU) or CUDA (GPU).
    """

    def __init__(self):
        self.model_range = range(config.sim_config.model_start, config.sim_config.model_end)
        self.tsimend = None
        self.tsimstart = None

    def run(self):
        """Run the simulation in the correct context."""
        self.print_logo_copyright()
        self.print_host_info()
        if config.sim_config.general['cuda']:
            self.print_gpu_info()
        self.tsimstart = timer()

        # Clear list of model configs. It can be retained when gprMax is
        # called in a loop, and want to avoid this.
        config.model_configs = []

        for i in self.model_range:
            config.model_num = i
            model_config = config.ModelConfig()
            config.model_configs.append(model_config)

            # Always create a grid for the first model. The next model to run
            # only gets a new grid if the geometry is not re-used.
            if i != 0 and config.sim_config.args.geometry_fixed:
                config.get_model_config().reuse_geometry = True
            else:
                G = create_G()

            model = ModelBuildRun(G)
            model.build()
            solver = create_solver(G)

            if not config.sim_config.args.geometry_only:
                model.solve(solver)

        self.tsimend = timer()
        self.print_time_report()

    def print_logo_copyright(self):
        """Print gprMax logo, version, and copyright/licencing information."""
        logo(__version__ + ' (' + codename + ')')

    def print_host_info(self):
        """Print information about the host machine."""
        hyperthreadingstr = f", {config.sim_config.hostinfo['logicalcores']} cores with Hyper-Threading" if config.sim_config.hostinfo['hyperthreading'] else ''
        logger.basic(f"\nHost: {config.sim_config.hostinfo['hostname']} | {config.sim_config.hostinfo['machineID']} | {config.sim_config.hostinfo['sockets']} x {config.sim_config.hostinfo['cpuID']} ({config.sim_config.hostinfo['physicalcores']} cores{hyperthreadingstr}) | {human_size(config.sim_config.hostinfo['ram'], a_kilobyte_is_1024_bytes=True)} RAM | {config.sim_config.hostinfo['osversion']}")

    def print_gpu_info(self):
        """Print information about any NVIDIA CUDA GPUs detected."""
        gpus_info = []
        for gpu in config.sim_config.cuda['gpus']:
            gpus_info.append(f'{gpu.deviceID} - {gpu.name}, {human_size(gpu.totalmem, a_kilobyte_is_1024_bytes=True)}')
        logger.basic(f" with GPU(s): {' | '.join(gpus_info)}")

    def print_time_report(self):
        """Print the total simulation time based on context."""
        s = f"\n=== Simulation on {config.sim_config.hostinfo['hostname']} completed in [HH:MM:SS]: {datetime.timedelta(seconds=self.tsimend - self.tsimstart)}"
        logger.basic(f"{s} {'=' * (get_terminal_width() - 1 - len(s))}\n")


class MPIContext(Context):
    """Mixed mode MPI/OpenMP/CUDA context - MPI task farm is used to distribute
        models, and each model parallelised using either OpenMP (CPU)
        or CUDA (GPU).
    """

    def __init__(self):
        super().__init__()
        from mpi4py import MPI
        from gprMax.mpi import MPIExecutor

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.MPIExecutor = MPIExecutor

    def _run_model(self, i, GPUdeviceID):
        """Process for running a single model."""

        config.model_num = i
        model_config = config.ModelConfig()
        if config.sim_config.general['cuda']:
            config.sim_config.set_model_gpu(GPUdeviceID)
        config.model_configs = model_config

        G = create_G()
        model = ModelBuildRun(G)
        model.build()
        solver = create_solver(G)

        if not config.sim_config.args.geometry_only:
            model.solve(solver)

    def run(self):
        """Specialise how the models are run."""

        self.tsimstart = timer()

        # Contruct MPIExecutor
        executor = self.MPIExecutor(self._run_model, comm=self.comm)

        # Compile jobs
        jobs = []
        for i in self.model_range:
            jobs.append({'i': i,
                         'GPUdeviceID': 0})

        # if executor.is_master():
        #     self.print_logo_copyright()
        # self.print_host_info()
        # if config.sim_config.general['cuda']:
        #     self.print_gpu_info()

        # Send the workers to their work loop
        executor.start()
        if executor.is_master():
            results = executor.submit(jobs)

        # Make the workers exit their work loop and join the main loop again
        executor.join()

        # with self.MPIExecutor(self._run_model, comm=self.comm) as executor:
        #     if executor is not None:
        #         results = executor.submit(jobs)
        #         logger.info('Results: %s' % str(results))
        # logger.basic('Finished.')

        self.tsimend = timer()
        if executor.is_master():
            self.print_time_report()

    def print_time_report(self):
        """Print the total simulation time based on context."""
        s = f"\n=== Simulation on {config.sim_config.hostinfo['hostname']} completed in [HH:MM:SS]: {datetime.timedelta(seconds=self.tsimend - self.tsimstart)}"
        logger.basic(f"{s} {'=' * (get_terminal_width() - 1 - len(s))}\n")


def create_context():
    """Create a context in which to run the simulation. i.e MPI.

    Returns:
        context (Context): Context for the model to run in.
    """

    if config.sim_config.args.mpi:
        context = MPIContext()
    else:
        context = Context()

    return context
