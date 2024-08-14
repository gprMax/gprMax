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

from .grid import CUDAGrid, FDTDGrid, OpenCLGrid
from .subgrids.updates import create_updates as create_subgrid_updates
from .updates import CPUUpdates, CUDAUpdates, OpenCLUpdates
import pybind11_xpu_solver

def create_G():
    """Create grid object according to solver.

    Returns:
        G: FDTDGrid class describing a grid in a model.
    """

    if config.sim_config.general["solver"] == "cpu" or config.sim_config.general["solver"] == "xpu":
        G = FDTDGrid()
    elif config.sim_config.general["solver"] == "cuda":
        G = CUDAGrid()
    elif config.sim_config.general["solver"] == "opencl":
        G = OpenCLGrid()

    return G


def create_solver(G):
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
    elif config.sim_config.general["solver"] == "xpu":
        solver = XPUSolver(G)
    elif config.sim_config.general["solver"] == "cuda":
        updates = CUDAUpdates(G)
        solver = Solver(updates)
    elif config.sim_config.general["solver"] == "opencl":
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
            if self.hsg:
                self.updates.hsg_2()
            self.updates.update_electric_a()
            self.updates.update_electric_pml()
            self.updates.update_electric_sources()
            if self.hsg:
                self.updates.hsg_1()
            self.updates.update_electric_b()
            if config.sim_config.general["solver"] == "cuda":
                self.memused = self.updates.calculate_memory_used(iteration)

        self.updates.finalise()
        self.solvetime = self.updates.calculate_solve_time()
        self.updates.cleanup()

class XPUSolver:
    def __init__(self, grid):

        self.solvetime = 0

        self.grid=grid
        self.BLT=2
        self.BLX=6
        self.BLY=6
        self.BLZ=6
        self.xmin=0
        self.xmax=self.grid.nx
        self.ymin=0
        self.ymax=self.grid.ny
        self.zmin=0
        self.zmax=self.grid.nz

        self.tx_tiling_type="p"
        self.ty_tiling_type="p"
        self.tz_tiling_type="p"
        self.max_phase=1
        self.TX_Tile_Shapes=["p"]
        self.TY_Tile_Shapes=["p"]
        self.TZ_Tile_Shapes=["p"]

        # self.tx_tiling_type="d"
        # self.ty_tiling_type="p"
        # self.tz_tiling_type="p"
        # self.max_phase=2
        # self.TX_Tile_Shapes=["m","v"]
        # self.TY_Tile_Shapes=["p","p"]
        # self.TZ_Tile_Shapes=["p","p"]

        # self.tx_tiling_type="d"
        # self.ty_tiling_type="d"
        # self.tz_tiling_type="p"
        # self.max_phase=4
        # self.TX_Tile_Shapes=["m","v","m","v"]
        # self.TY_Tile_Shapes=["m","m","v","v"]
        # self.TZ_Tile_Shapes=["p","p","p","p"]

        # self.tx_tiling_type="d"
        # self.ty_tiling_type="d"
        # self.tz_tiling_type="d"
        # self.max_phase=8
        # self.TX_Tile_Shapes=["m","v","m","m","v","v","m","v"]
        # self.TY_Tile_Shapes=["m","m","v","m","v","m","v","v"]
        # self.TZ_Tile_Shapes=["m","m","m","v","m","v","v","v"]

        self.x_ntiles=self.GetNumOfTiles(self.tx_tiling_type, self.BLT, self.BLX, self.xmin, self.xmax)
        self.y_ntiles=self.GetNumOfTiles(self.ty_tiling_type, self.BLT, self.BLY, self.ymin, self.ymax)
        self.z_ntiles=self.GetNumOfTiles(self.tz_tiling_type, self.BLT, self.BLZ, self.zmin, self.zmax)
        source = self.grid.hertziandipoles[0]
        componentID = f"E{source.polarisation}"
        source_id = self.grid.IDlookup[componentID]
        self.cpp_solver=pybind11_xpu_solver.xpu_solver(
            self.grid.Ex,
            self.grid.Ey,
            self.grid.Ez,
            self.grid.Hx,
            self.grid.Hy,
            self.grid.Hz,
            self.grid.updatecoeffsE,
            self.grid.updatecoeffsH,
            self.grid.ID,
            self.BLT, self.BLX, self.BLY, self.BLZ,
            self.x_ntiles, self.y_ntiles, self.z_ntiles,
            self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax,
            self.max_phase,
            self.tx_tiling_type, self.ty_tiling_type, self.tz_tiling_type,
            self.TX_Tile_Shapes, self.TY_Tile_Shapes, self.TZ_Tile_Shapes,
            source.xcoord, source.ycoord, source.zcoord, source.start, source.stop,
            source.waveformvalues_halfdt,
            source.dl, source_id,
            self.grid.dt, self.grid.dx, self.grid.dy, self.grid.dz,
        )
    
    def GetNumOfTiles(self, tiling_type, time_block_size, space_block_size, start, end):
        if(tiling_type=="d"):
            num=(end-start)//(2*space_block_size+2*time_block_size-1)
            num_left=(end-start)%(2*space_block_size+2*time_block_size-1)
            if(num_left<=2*space_block_size+time_block_size):
                num+=1
            else:
                num+=2
            return num
        elif(tiling_type=="p"):
            num=(end-start)//space_block_size
            while True:
                start_idx=space_block_size*num+1
                if(start_idx-time_block_size<=(end-start)):
                    num+=1
                else:
                    return num
    
    def solve(self, iterator):
        for tt in range(0, iterator.total, self.BLT):
            self.cpp_solver.update(tt)
            iterator.update(self.BLT)

