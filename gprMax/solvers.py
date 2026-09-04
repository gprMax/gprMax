# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import logging

import gprMax.config as config
from gprMax.model import Model

from .grid.cuda_grid import CUDAGrid
from .grid.fdtd_grid import FDTDGrid
from .grid.metal_grid import MetalGrid
from .grid.opencl_grid import OpenCLGrid
from .subgrids.updates import SubgridUpdates
from .subgrids.updates import create_updates as create_subgrid_updates
from .updates.cpu_updates import CPUUpdates
from .updates.cuda_updates import CUDAUpdates
from .updates.metal_updates import MetalUpdates
from .updates.opencl_updates import OpenCLUpdates
from .updates.updates import Updates

logger = logging.getLogger(__name__)


class Solver:
    """Generic solver for Update objects"""

    def __init__(self, updates: Updates):
        """
        Args:
            updates: Updates contains methods to run FDTD algorithm.
        """

        if not isinstance(updates, Updates):
            raise TypeError(
                f"updates must implement the Updates interface, got {type(updates).__name__}"
            )

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
            # time loop at this point is at n
            self.updates.store_outputs(iteration)
            self.updates.store_snapshots(iteration)
            self.updates.observe_ntff_electric(iteration)
            self.updates.observe_sar_electric(iteration)

            # time loop at this point is working at fields updated to be at n+1/2
            self.updates.update_magnetic()
            self.updates.update_magnetic_pml()
            self.updates.update_magnetic_sources(iteration)
            self.updates.update_eigenmode_sources_magnetic(iteration)
            self.updates.update_plane_waves_magnetic(iteration)

            if getattr(self.updates, "is_distributed", False) is True:
                self.updates.halo_swap_magnetic()
                self.updates.update_magnetic_edge_devices(iteration)
                # Modal H projections interpolate across transverse Yee
                # edges. Observe only after the current H halos are available.
                self.updates.observe_eigenmode_ports(iteration)
            else:
                self.updates.observe_eigenmode_ports(iteration)

            if isinstance(self.updates, SubgridUpdates):
                self.updates.hsg_2()

            self.updates.observe_ntff_magnetic(iteration)

            # time loop at this point is still at working on fields updated to be at n+1
            self.updates.update_electric_a()
            # Apply the PMC ghost-image correction on the active local
            # backend. MPI ranks dispatch only their owned global faces.
            self.updates.update_symmetry_boundaries_electric()

            self.updates.update_electric_pml()
            self.updates.update_electric_sources(iteration)
            self.updates.update_eigenmode_sources_electric(iteration)
            self.updates.update_plane_waves_electric(iteration)

            # TODO: Increment iteration here if add Model to Solver
            if isinstance(self.updates, SubgridUpdates):
                self.updates.hsg_1()

            # Complete the dispersive PMC correction after PML and sources,
            # mirroring the bulk dispersive update's A/B split.
            self.updates.update_symmetry_boundaries_electric_b()

            self.updates.update_electric_b()
            self.updates.update_impedance_surfaces()
            self.updates.update_network_terminals(iteration)

            if getattr(self.updates, "is_distributed", False) is True:
                self.updates.halo_swap_electric()
            if isinstance(self.updates, CUDAUpdates):
                self.memused = self.updates.calculate_memory_used(iteration)

        self.updates.finalise()
        self.solvetime = self.updates.calculate_solve_time()
        self.updates.cleanup()


def create_solver(model: Model) -> Solver:
    """Create configured solver object.

    N.B. A large range of different functions exist to advance the time
    step for dispersive materials. The correct function is set by the
    set_dispersive_updates method, based on the required numerical
    precision and dispersive material type.This is done for solvers
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
        if updates.grid.maxpoles != 0:
            updates.set_dispersive_updates()
        for u in updates.updaters:
            if u.grid.maxpoles != 0:
                u.set_dispersive_updates()
    elif type(grid) is FDTDGrid:
        updates = CPUUpdates(grid)
        if grid.maxpoles != 0:
            updates.set_dispersive_updates()
    elif getattr(grid, "is_distributed", False) is True:
        from gprMax.updates.mpi_updates import MPIUpdates

        updates = MPIUpdates(grid)
        if grid.maxpoles != 0:
            updates.set_dispersive_updates()
    elif type(grid) is CUDAGrid:
        updates = CUDAUpdates(grid)
    elif type(grid) is OpenCLGrid:
        updates = OpenCLUpdates(grid)
    elif type(grid) is MetalGrid:
        updates = MetalUpdates(grid)
    else:
        logger.error("Cannot create Solver: Unknown grid type")
        raise ValueError

    solver = Solver(updates)

    return solver
