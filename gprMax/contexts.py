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

import gprMax.config as config
from ._version import __version__, codename
from .config_parser import write_model_config
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

    def __init__(self):
        self.model_range = range(config.sim_config.model_start, config.sim_config.model_end)
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

    def _run(self):
        """Specialise how the models are farmed out."""

        for i in self.model_range:
            write_model_config(i)

            # Always create a solver for the first model.
            # The next model to run only gets a new solver if the
            # geometry is not re-used.
            if i != 0 and config.sim_config.args.geometry_fixed:
                config.model_config[i].reuse_geometry = True
            else:
                G = create_G()

            model = ModelBuildRun(i, G)
            model.build()
            solver = create_solver(G)

            if not config.sim_config.args.geometry_only:
                model.solve(solver)

    def make_time_report(self):
        """Function to specialise the time reporting for the standard simulation
            context.
        """

        sim_time = datetime.timedelta(seconds=self.tsimend - self.tsimstart)
        s = f"\n=== Simulation on {self.simconfig.hostinfo['hostname']} completed in [HH:MM:SS]: {sim_time}"
        return f"{s} {'=' * (get_terminal_width() - 1 - len(s))}\n"


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
        from mpi4py import MPI

    def _run(self):
        pass

    def make_time_report(self):
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
