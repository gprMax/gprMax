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
from .utilities import get_terminal_width
import datetime

class Context():

    def __init__(self, sim_config, solver):
        """Context for the model to run in. Sub-class this with contexts
        i.e. an MPI context.

        Args:
            sim_config (SimConfig): Simulation level configuration object.
            solver (Solver): FDTD general solver object.
        """
        self.sim_config = sim_config
        self.solver = solver
        self.model_range = range(self.sim_conf.model_start,
                                 self.sim_conf.model_end)
        self.tsimend = 0
        self.tsimstart = 0

    def run(self):
        """Function to run the simulation in the correct context."""
        self.tsimstart = timer()
        self._run()
        self.tsimend = timer()

    def print_time_report(self):
        """Function to print the total simulation time based on context."""
        s = self.make_time_report(sim_time)
        print(s)

    def make_time_report(self):
        """Function to generate a string for the total simulation time bas"""
        pass


class NoMPIContext(Context):

    def _run(self):
        """Specialise how the models are farmed out."""
        for currentmodelrun in self.model_range:
            # create the model configuration
            model_config = object()
            run_model(solver, sim_config, model_config)

    def make_time_report(self):
        """Function to specialise the time reporting for the standard Simulation
        context."""
        sim_time = datetime.timedelta(seconds=self.tsimend - self.tsimstart)
        s = '\n=== Simulation on {} completed in [HH:MM:SS]: {}'
        s = s.format(self.simconfig.hostinfo['hostname'], sim_time)
        return '{} {}\n'.format(s, '=' * (get_terminal_width() - 1 - len(s)))


class MPIContext(Context):

    def _run(self):
        pass

    def make_time_report(self):
        pass

class MPINoSpawnContext(Context):

    def _run(self):
        pass

    def make_time_report(self):
        pass


def create_context(sim_config, solver):
    """Create a context in which to run the simulation. i.e MPI."""
    if sim_config.mpi_no_spawn:
        context = MPIContext(sim_config, solver)
    elif sim_config.mpi:
        context = MPINoSpawnContext(sim_config, solver)
    else:
        context = NoMPIContext(sim_conf, solver)

    return context
