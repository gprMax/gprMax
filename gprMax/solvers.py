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

import logging

import gprMax.config as config
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.model import Model
from gprMax.updates.mpi_updates import MPIUpdates

from .grid.cuda_grid import CUDAGrid
from .grid.hip_grid import HIPGrid
from .grid.fdtd_grid import FDTDGrid
from .grid.opencl_grid import OpenCLGrid
from .subgrids.updates import SubgridUpdates
from .subgrids.updates import create_updates as create_subgrid_updates
from .updates.cpu_updates import CPUUpdates
from .updates.cuda_updates import CUDAUpdates
from .updates.hip_updates import HIPUpdates
from .updates.opencl_updates import OpenCLUpdates
from .updates.updates import Updates

logger = logging.getLogger(__name__)


class Solver:
    """Generic solver for Update objects"""

    def __init__(self, updates: Updates):
        """
        Args:
            updates: Updates contains methods to run FDTD algorithm.
            hsg: boolean to use sub-gridding.
        """

        self.updates = updates
        self.solvetime = 0
        self.memused = 0

    def solve(self, iterator):
        """Time step the FDTD model.

        Args:
            iterator: can be range() or tqdm()
        """

        self.updates.time_start()

        for iteration in iterator:
            self.updates.store_outputs(iteration)
            self.updates.store_snapshots(iteration)
            self.updates.update_magnetic()
            self.updates.update_magnetic_pml()
            self.updates.update_magnetic_sources(iteration)
            # self.updates.update_plane_waves()
            if isinstance(self.updates, MPIUpdates):
                self.updates.halo_swap_magnetic()
            if isinstance(self.updates, SubgridUpdates):
                self.updates.hsg_2()
            self.updates.update_electric_a()
            self.updates.update_electric_pml()
            self.updates.update_electric_sources(iteration)
            # TODO: Increment iteration here if add Model to Solver
            if isinstance(self.updates, SubgridUpdates):
                self.updates.hsg_1()
            self.updates.update_electric_b()
            if isinstance(self.updates, MPIUpdates):
                self.updates.halo_swap_electric()
            if isinstance(self.updates, CUDAUpdates):
                self.memused = self.updates.calculate_memory_used(iteration)
            if isinstance(self.updates, HIPUpdates):
                self.memused = self.updates.calculate_memory_used(iteration)

        self.updates.finalise()
        self.solvetime = self.updates.calculate_solve_time()
        self.updates.cleanup()


def create_solver(model: Model) -> Solver:
    """Create configured solver object.

    N.B. A large range of different functions exist to advance the time
    step for dispersive materials. The correct function is set by the
    set_dispersive_updates method, based on the required numerical
    precision and dispersive material type. This is done for solvers
    running on CPU, i.e. where Cython is used. CUDA and OpenCL
    dispersive material functions are handled through templating and
    substitution at runtime.

    Args:
        model: model containing the main grid and subgrids.

    Returns:
        solver: Solver object.
    """
    grid = model.G
    if config.sim_config.general["subgrid"]:
        updates = create_subgrid_updates(model)
        if config.get_model_config().materials["maxpoles"] != 0:
            # Set dispersive update functions for both SubgridUpdates and
            # SubgridUpdaters subclasses
            updates.set_dispersive_updates()
            for u in updates.updaters:
                u.set_dispersive_updates()
    elif type(grid) is FDTDGrid:
        updates = CPUUpdates(grid)
        if config.get_model_config().materials["maxpoles"] != 0:
            updates.set_dispersive_updates()
    elif type(grid) is MPIGrid:
        updates = MPIUpdates(grid)
        if config.get_model_config().materials["maxpoles"] != 0:
            updates.set_dispersive_updates()
    elif type(grid) is CUDAGrid:
        updates = CUDAUpdates(grid)
    elif type(grid) is OpenCLGrid:
        updates = OpenCLUpdates(grid)
    elif type(grid) is HIPGrid:
        updates = HIPUpdates(grid)
        
    else:
        logger.error("Cannot create Solver: Unknown grid type")
        raise ValueError

    solver = Solver(updates)

    return solver
