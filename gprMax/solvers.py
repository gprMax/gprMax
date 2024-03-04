# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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

from .grid.cuda_grid import CUDAGrid
from .grid.fdtd_grid import FDTDGrid
from .grid.opencl_grid import OpenCLGrid
from .subgrids.updates import SubgridUpdates
from .subgrids.updates import create_updates as create_subgrid_updates
from .updates.cpu_updates import CPUUpdates
from .updates.cuda_updates import CUDAUpdates
from .updates.opencl_updates import OpenCLUpdates
from .updates.updates import Updates


def create_G() -> FDTDGrid:
    """Create grid object according to solver.

    Returns:
        G: FDTDGrid class describing a grid in a model.
    """

    if config.sim_config.general["solver"] == "cpu":
        G = FDTDGrid()
    elif config.sim_config.general["solver"] == "cuda":
        G = CUDAGrid()
    elif config.sim_config.general["solver"] == "opencl":
        G = OpenCLGrid()

    return G


class Solver:
    """Generic solver for Update objects"""

    def __init__(self, updates: Updates, hsg=False):
        """
        Args:
            updates: Updates contains methods to run FDTD algorithm.
            hsg: boolean to use sub-gridding.
        """

        self.updates = updates
        self.hsg = hsg
        self.solvetime = 0
        self.memused = 0

    def solve(self, iterator):
        """Time step the FDTD model.

        Args:
            iterator: can be range() or tqdm()
        """

        self.updates.time_start()

        for iteration in iterator:
            self.updates.store_outputs()
            self.updates.store_snapshots(iteration)
            self.updates.update_magnetic()
            self.updates.update_magnetic_pml()
            self.updates.update_magnetic_sources()
            if isinstance(self.updates, SubgridUpdates):
                self.updates.hsg_2()
            self.updates.update_electric_a()
            self.updates.update_electric_pml()
            self.updates.update_electric_sources()
            if isinstance(self.updates, SubgridUpdates):
                self.updates.hsg_1()
            self.updates.update_electric_b()
            if isinstance(self.updates, CUDAUpdates):
                self.memused = self.updates.calculate_memory_used(iteration)

        self.updates.finalise()
        self.solvetime = self.updates.calculate_solve_time()
        self.updates.cleanup()


def create_solver(G: FDTDGrid) -> Solver:
    """Create configured solver object.

    N.B. A large range of different functions exist to advance the time step for
            dispersive materials. The correct function is set by the
            set_dispersive_updates method, based on the required numerical
            precision and dispersive material type.
            This is done for solvers running on CPU, i.e. where Cython is used.
            CUDA and OpenCL dispersive material functions are handled through
            templating and substitution at runtime.

    Args:
        G: FDTDGrid class describing a grid in a model.

    Returns:
        solver: Solver object.
    """

    if config.sim_config.general["subgrid"]:
        updates = create_subgrid_updates(G)
        if config.get_model_config().materials["maxpoles"] != 0:
            # Set dispersive update functions for both SubgridUpdates and
            # SubgridUpdaters subclasses
            updates.set_dispersive_updates()
            for u in updates.updaters:
                u.set_dispersive_updates()
        solver = Solver(updates, hsg=True)
    elif config.sim_config.general["solver"] == "cpu":
        updates = CPUUpdates(G)
        if config.get_model_config().materials["maxpoles"] != 0:
            updates.set_dispersive_updates()
        solver = Solver(updates)
    elif config.sim_config.general["solver"] == "cuda":
        updates = CUDAUpdates(G)
        solver = Solver(updates)
    elif config.sim_config.general["solver"] == "opencl":
        updates = OpenCLUpdates(G)
        solver = Solver(updates)

    return solver
