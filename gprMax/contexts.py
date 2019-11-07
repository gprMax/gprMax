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

import datetime
import logging

import gprMax.config as config
from ._version import __version__, codename
from .config_parser import write_model_config
from .model_build_run import ModelBuildRun
from .solvers import create_solver
from .solvers import create_G
from .utilities import get_terminal_width
from .utilities import human_size
from .utilities import logo
from .utilities import timer

log = logging.getLogger(__name__)


class Context:
    """Generic context for the model to run in. Sub-class with specific contexts
    e.g. an MPI context.
    """

    def __init__(self):
        self.model_range = range(config.sim_config.model_start, config.sim_config.model_end)
        self.tsimend = 0
        self.tsimstart = 1

    def run(self):
        """Run the simulation in the correct context."""
        self.print_logo_copyright()
        self.print_host_info()
        if config.sim_config.general['cuda']:
            self.print_gpu_info()
        self.tsimstart = timer()
        self._run()
        self.tsimend = timer()
        self.print_time_report()

    def print_logo_copyright(self):
        """Print gprMax logo, version, and copyright/licencing information."""
        logo(__version__ + ' (' + codename + ')')

    def print_host_info(self):
        """Print information about the host machine."""
        hyperthreadingstr = f", {config.sim_config.hostinfo['logicalcores']} cores with Hyper-Threading" if config.sim_config.hostinfo['hyperthreading'] else ''
        log.info(f"\nHost: {config.sim_config.hostinfo['hostname']} | {config.sim_config.hostinfo['machineID']} | {config.sim_config.hostinfo['sockets']} x {config.sim_config.hostinfo['cpuID']} ({config.sim_config.hostinfo['physicalcores']} cores{hyperthreadingstr}) | {human_size(config.sim_config.hostinfo['ram'], a_kilobyte_is_1024_bytes=True)} RAM | {config.sim_config.hostinfo['osversion']}")

    def print_gpu_info(self):
        """Print information about any NVIDIA CUDA GPUs detected."""
        gpus_info = []
        for gpu in config.sim_config.cuda['gpus']:
            gpus_info.append(f'{gpu.deviceID} - {gpu.name}, {human_size(gpu.totalmem, a_kilobyte_is_1024_bytes=True)}')
        log.info(f" with GPU(s): {' | '.join(gpus_info)}")

    def print_time_report(self):
        """Print the total simulation time based on context."""
        s = self.make_time_report()
        log.info(s)

    def make_time_report(self):
        """Generate a string for the total simulation time."""
        s = f"\n=== Simulation on {config.sim_config.hostinfo['hostname']} completed in [HH:MM:SS]: {datetime.timedelta(seconds=self.tsimend - self.tsimstart)}"
        return f"{s} {'=' * (get_terminal_width() - 1 - len(s))}\n"


class NoMPIContext(Context):
    """Standard context - models are run one after another and each model
        is parallelised using either OpenMP (CPU) or CUDA (GPU).
    """

    def _run(self):
        """Specialise how the models are farmed out."""

        for i in self.model_range:
            write_model_config(i)
            # Always create a solver for the first model.
            # The next model to run only gets a new solver if the
            # geometry is not re-used.
            if i != 0 and config.sim_config.args.geometry_fixed:
                config.model_configs[i].reuse_geometry = True
                # Ensure re-used G is associated correctly with model
                G.model_num = i
            else:
                G = create_G(i)

            model = ModelBuildRun(G)
            model.build()
            solver = create_solver(G)

            if not config.sim_config.args.geometry_only:
                model.solve(solver)


class MPIContext(Context):
    """Mixed mode MPI/OpenMP/CUDA context - MPI task farm is used to distribute
        models, and each model parallelised using either OpenMP (CPU)
        or CUDA (GPU).
    """

    def __init__(self):
        super().__init__()
        from mpi4py import MPI

    def _run(self):
        pass


class MPINoSpawnContext(Context):

    def __init__(self):
        super().__init__()
        from mpi4py import MPI

    def _run(self):
        pass


def create_context():
    """Create a context in which to run the simulation. i.e MPI.

    Returns:
        context (Context): Context for the model to run in.
    """

    if config.sim_config.args.mpi:
        context = MPIContext()
    elif config.sim_config.args.mpi_no_spawn:
        context = MPINoSpawnContext()
    else:
        context = NoMPIContext()

    return context
