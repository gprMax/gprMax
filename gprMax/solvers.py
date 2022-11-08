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

import gprMax.config as config

from .grid import CUDAGrid, FDTDGrid, OpenCLGrid
from .subgrids.updates import create_updates as create_subgrid_updates
from .updates import CPUUpdates, CUDAUpdates, OpenCLUpdates


def create_G():
    """Create grid object according to solver.

    Returns:
        G: FDTDGrid class describing a grid in a model.
    """

    if config.sim_config.general['solver'] == 'cpu':
        G = FDTDGrid()
    elif config.sim_config.general['solver'] == 'cuda':
        G = CUDAGrid()
    elif config.sim_config.general['solver'] == 'opencl':
        G = OpenCLGrid()

    return G


def create_solver(G):
    """Create configured solver object.

    Args:
        G: FDTDGrid class describing a grid in a model.

    Returns:
        solver: Solver object.
    """

    if config.sim_config.general['subgrid']:
        updates = create_subgrid_updates(G)
        solver = Solver(updates, hsg=True)
        # A large range of different functions exist to advance the time step for
        # dispersive materials. The correct function is set here based on the
        # the required numerical precision and dispersive material type.
        props = updates.adapt_dispersive_config()
        updates.set_dispersive_updates(props)
    elif config.sim_config.general['solver'] == 'cpu':
        updates = CPUUpdates(G)
        solver = Solver(updates)
        props = updates.adapt_dispersive_config()
        updates.set_dispersive_updates(props)
    elif config.sim_config.general['solver'] == 'cuda':
        updates = CUDAUpdates(G)
        solver = Solver(updates)
    elif config.sim_config.general['solver'] == 'opencl':
        updates = OpenCLUpdates(G)
        solver = Solver(updates)

    return solver


class Solver:
    """Generic solver for Update objects"""

    def __init__(self, updates, hsg=False):
        """
        Args:
            updates: Updates contains methods to run FDTD algorithm.
            hsg: boolean to use sub-gridding.
        """

        self.updates = updates
        self.hsg = hsg

    def solve(self, iterator):
        """Time step the FDTD model.

        Args:
            iterator: can be range() or tqdm()

        Returns:
            tsolve: float for time taken to execute solving (seconds).
            memsolve: float for memory (RAM) used.
        """

        memsolve = 0
        self.updates.time_start()

        for iteration in iterator:
            self.updates.store_outputs()
            self.updates.store_snapshots(iteration)
            self.updates.update_magnetic()
            self.updates.update_magnetic_pml()
            self.updates.update_magnetic_sources()
            if self.hsg:
                self.updates.hsg_2()
            self.updates.update_electric_a()
            self.updates.update_electric_pml()
            self.updates.update_electric_sources()
            if self.hsg:
                self.updates.hsg_1()
            self.updates.update_electric_b()
            if config.sim_config.general['solver'] == 'cuda':
                memsolve = self.updates.calculate_memsolve(iteration)  

        self.updates.finalise()
        tsolve = self.updates.calculate_tsolve()
        self.updates.cleanup()

        return tsolve, memsolve
