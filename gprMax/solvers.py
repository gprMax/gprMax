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
from gprMax.updates import CPUUpdates
from gprMax.updates import GPUUpdates
from gprMax.utilities import timer
from .grid import FDTDGrid
from .grid import GPUGrid
import gprMax.config as config
from .subgrids.solver import create_updates as create_subgrid_updates
from .subgrids.solver import SubGridSolver


def create_G(sim_config):
    """Returns the configured solver."""
    if sim_config.gpu:
        G = GPUGrid()
    elif sim_config.subgrid:
        G = FDTDGrid()
    else:
        G = FDTDGrid()

    return G


def create_solver(G, sim_config):
    """Returns the configured solver."""
    if sim_config.gpu:
        updates = GPUUpdates(G)
        solver = Solver(updates)
    elif sim_config.subgrid:
        updates = create_subgrid_updates(G)
        solver = SubGridSolver(G, updates)
    else:
        updates = CPUUpdates(G)
        solver = Solver(updates)

    return solver

    # a large range of function exist to advance the time step for dispersive
    # materials. The correct function is set here  based on the
    # the required numerical precision and dispersive material type.
    props = updates.adapt_dispersive_config(config)
    updates.set_dispersive_updates(props)


class Solver:

    """Generic solver for Update objects"""

    def __init__(self, updates):
        """Context for the model to run in. Sub-class this with contexts
        i.e. an MPI context.

        Args:
            updates (Updates): updates contains methods to run FDTD algorithm
            iterator (iterator): can be range() or tqdm()
        """
        self.updates = updates

    def get_G(self):
        return self.updates.G

    def solve(self, iterator):
        """Time step the FDTD model."""
        tsolvestart = timer()
        for iteration in iterator:
            self.updates.store_outputs(iteration)
            self.updates.store_snapshots(iteration)
            self.updates.update_magnetic()
            self.updates.update_magnetic_pml()
            self.updates.update_magnetic_sources(iteration)
            self.updates.update_electric_a()
            self.updates.update_electric_pml()
            self.updates.update_electric_sources(iteration)
            self.updates.update_electric_b()

        tsolve = timer() - tsolvestart
        return tsolve
