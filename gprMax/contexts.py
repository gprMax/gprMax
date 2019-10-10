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

from ._version import __version__, codename
from .config import create_model_config
from .model_build_run import ModelBuildRun
from .solvers import create_solver
from .solvers import create_G
from .utilities import get_terminal_width
from .utilities import logo
from .utilities import timer


class Context:
    """Generic context for the model to run in. Sub-class with specific contexts
    e.g. an MPI context.
    """

    def __init__(self, sim_config):
        """
        Args:
            sim_config (SimConfig): Simulation level configuration object.
        """

        self.sim_config = sim_config
        self.model_range = range(sim_config.model_start, sim_config.model_end)
        self.tsimend = 0
        self.tsimstart = 1

    def run(self):
        """Run the simulation in the correct context."""
        self.print_logo_copyright()
        self.tsimstart = timer()
        self._run()
        self.tsimend = timer()

    def print_logo_copyright(self):
        """Print gprMax logo, version, and copyright/licencing information."""
        logo(__version__ + ' (' + codename + ')')

    def print_time_report(self):
        """Print the total simulation time based on context."""
        s = self.make_time_report()
        log.info(s)

    def make_time_report(self):
        """Generate a string for the total simulation time."""
        pass


class NoMPIContext(Context):
    """Standard context - models are run one after another and each model
        is parallelised using either OpenMP (CPU) or CUDA (GPU).
    """

    def __init__(self):
        super().__init__()

    def _run(self):
        """Specialise how the models are farmed out."""

        for i in self.model_range:
            model_config = create_model_config(self.sim_config, i)

            # Always create a solver for the first model.
            # The next model to run only gets a new solver if the
            # geometry is not re-used.
            if i != 0 and self.sim_config.geometry_fixed:
                model_config.reuse_geometry = True
            else:
                G = create_G(self.sim_config)

            model = ModelBuildRun(G, self.sim_config, model_config)
            model.build()

            solver = create_solver(G, self.sim_config)

            if not self.sim_config.geometry_only:
                model.solve(solver)

    def make_time_report(self):
        """Function to specialise the time reporting for the standard Simulation
            context.
        """

        sim_time = datetime.timedelta(seconds=self.tsimend - self.tsimstart)
        s = f'\n=== Simulation on {self.simconfig.hostinfo['hostname']} completed in [HH:MM:SS]: {sim_time}'
        return f'{s} {'=' * (get_terminal_width() - 1 - len(s))}\n'


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

    def make_time_report(self):
        pass


class MPINoSpawnContext(Context):

    def __init__(self):
        super().__init__()

    def _run(self):
        pass

    def make_time_report(self):
        pass


def create_context(sim_config):
    """Create a context in which to run the simulation. i.e MPI.

    Args:
        sim_config (SimConfig): Simulation level configuration object.

    Returns:
        context (Context): Context for the model to run in.
    """

    if sim_config.args.mpi:
        context = MPIContext(sim_config)
    elif sim_config.args.mpi_no_spawn:
        context = MPINoSpawnContext(sim_config)
    else:
        context = NoMPIContext(sim_config)

    return context
