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

import itertools
from enum import IntEnum, unique
from typing import List, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from numpy import ndarray

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.receivers import Rx
from gprMax.sources import Source

CoordType = TypeVar("CoordType", bound=Union[Rx, Source])


@unique
class Dim(IntEnum):
    X = 0
    Y = 1
    Z = 2


@unique
class Dir(IntEnum):
    NEG = 0
    POS = 1


class MPIGrid(FDTDGrid):
    HALO_SIZE = 1
    COORDINATOR_RANK = 0

    def __init__(self, comm: MPI.Cartcomm):
        self.size = np.zeros(3, dtype=int)

        super().__init__()

        self.comm = comm

        self.mpi_tasks = np.array(self.comm.dims)

        self.lower_extent: npt.NDArray[np.intc] = np.zeros(3, dtype=int)
        self.upper_extent: npt.NDArray[np.intc] = np.zeros(3, dtype=int)
        self.global_size: npt.NDArray[np.intc] = np.zeros(3, dtype=int)

        self.neighbours = np.full((3, 2), -1, dtype=int)
        self.neighbours[Dim.X] = self.comm.Shift(direction=Dim.X, disp=1)
        self.neighbours[Dim.Y] = self.comm.Shift(direction=Dim.Y, disp=1)
        self.neighbours[Dim.Z] = self.comm.Shift(direction=Dim.Z, disp=1)

        self.send_halo_map = np.empty((3, 2), dtype=MPI.Datatype)
        self.recv_halo_map = np.empty((3, 2), dtype=MPI.Datatype)

    @property
    def rank(self) -> int:
        return self.comm.Get_rank()

    @property
    def coords(self) -> List[int]:
        return self.comm.coords

    @property
    def nx(self) -> int:
        return self.size[Dim.X]

    @nx.setter
    def nx(self, value: int):
        self.size[Dim.X] = value

    @property
    def ny(self) -> int:
        return self.size[Dim.Y]

    @ny.setter
    def ny(self, value: int):
        self.size[Dim.Y] = value

    @property
    def nz(self) -> int:
        return self.size[Dim.Z]

    @nz.setter
    def nz(self, value: int):
        self.size[Dim.Z] = value

    def is_coordinator(self) -> bool:
        return self.rank == self.COORDINATOR_RANK

    def get_rank_from_coordinate(self, coord: npt.NDArray) -> int:
        step_size = self.global_size // self.mpi_tasks
        overflow = self.global_size % self.mpi_tasks
        grid_coord = np.where(
            (step_size + 1) * overflow > coord,
            coord // (step_size + 1),
            (coord - overflow) // step_size,
        )
        return self.comm.Get_cart_rank(grid_coord.tolist())

    def global_to_local_coordinate(self, global_coord: npt.NDArray) -> npt.NDArray:
        return global_coord - self.lower_extent

    def local_to_global_coordinate(self, local_coord: npt.NDArray) -> npt.NDArray:
        return local_coord + self.lower_extent

    def scatter_coord_objects(self, objects: List[CoordType]) -> List[CoordType]:
        if self.is_coordinator():
            objects_by_rank: List[List[CoordType]] = [[] for _ in range(self.comm.size)]
            for o in objects:
                objects_by_rank[self.get_rank_from_coordinate(o.coord)].append(o)
        else:
            objects_by_rank = None

        objects = self.comm.scatter(objects_by_rank, root=self.COORDINATOR_RANK)

        for o in objects:
            o.coord = self.global_to_local_coordinate(o.coord)

        return objects

    def gather_coord_objects(self, objects: List[CoordType]) -> List[CoordType]:
        for o in objects:
            o.coord = self.local_to_global_coordinate(o.coord)
        gathered_objects: Optional[List[List[CoordType]]] = self.comm.gather(
            objects, root=self.COORDINATOR_RANK
        )

        if gathered_objects is not None:
            return list(itertools.chain(*gathered_objects))
        else:
            return objects

    def scatter_3d_array(self, array: npt.NDArray) -> npt.NDArray:
        self.comm.Bcast(array, root=self.COORDINATOR_RANK)

        return array[
            self.lower_extent[Dim.X] : self.upper_extent[Dim.X],
            self.lower_extent[Dim.Y] : self.upper_extent[Dim.Y],
            self.lower_extent[Dim.Z] : self.upper_extent[Dim.Z],
        ].copy(order="C")

    def scatter_4d_array(self, array: npt.NDArray) -> npt.NDArray:
        self.comm.Bcast(array, root=self.COORDINATOR_RANK)

        return array[
            :,
            self.lower_extent[Dim.X] : self.upper_extent[Dim.X],
            self.lower_extent[Dim.Y] : self.upper_extent[Dim.Y],
            self.lower_extent[Dim.Z] : self.upper_extent[Dim.Z],
        ].copy(order="C")

    def scatter_grid(self):
        self.materials = self.comm.bcast(self.materials, root=self.COORDINATOR_RANK)
        self.rxs = self.scatter_coord_objects(self.rxs)
        self.voltagesources = self.scatter_coord_objects(self.voltagesources)
        self.magneticdipoles = self.scatter_coord_objects(self.magneticdipoles)
        self.hertziandipoles = self.scatter_coord_objects(self.hertziandipoles)
        self.transmissionlines = self.scatter_coord_objects(self.transmissionlines)

        self.pmls = self.comm.bcast(self.pmls, root=self.COORDINATOR_RANK)
        if self.coords[Dim.X] != 0:
            self.pmls["thickness"]["x0"] = 0
        if self.coords[Dim.X] != self.mpi_tasks[Dim.X] - 1:
            self.pmls["thickness"]["xmax"] = 0
        if self.coords[Dim.Y] != 0:
            self.pmls["thickness"]["y0"] = 0
        if self.coords[Dim.Y] != self.mpi_tasks[Dim.Y] - 1:
            self.pmls["thickness"]["ymax"] = 0
        if self.coords[Dim.Z] != 0:
            self.pmls["thickness"]["z0"] = 0
        if self.coords[Dim.Z] != self.mpi_tasks[Dim.Z] - 1:
            self.pmls["thickness"]["zmax"] = 0

        old_size = self.size
        self.size = self.global_size
        if not self.is_coordinator():
            self.initialise_geometry_arrays()
        self.size = old_size

        self.solid = self.scatter_3d_array(self.solid)
        self.rigidE = self.scatter_4d_array(self.rigidE)
        self.rigidH = self.scatter_4d_array(self.rigidH)

    def gather_grid_objects(self):
        self.rxs = self.gather_coord_objects(self.rxs)
        self.voltagesources = self.gather_coord_objects(self.voltagesources)
        self.magneticdipoles = self.gather_coord_objects(self.magneticdipoles)
        self.hertziandipoles = self.gather_coord_objects(self.hertziandipoles)
        self.transmissionlines = self.gather_coord_objects(self.transmissionlines)

    def _halo_swap(self, array: ndarray, dim: Dim, dir: Dir):
        neighbour = self.neighbours[dim][dir]
        if neighbour != -1:
            self.comm.Sendrecv(
                [array, self.send_halo_map[dim][dir]],
                neighbour,
                0,
                [array, self.recv_halo_map[dim][dir]],
                neighbour,
                0,
                None,
            )

    def _halo_swap_by_dimension(self, array: ndarray, dim: Dim):
        if self.coords[dim] % 2 == 0:
            self._halo_swap(array, dim, Dir.NEG)
            self._halo_swap(array, dim, Dir.POS)
        else:
            self._halo_swap(array, dim, Dir.POS)
            self._halo_swap(array, dim, Dir.NEG)

    def _halo_swap_array(self, array: ndarray):
        self._halo_swap_by_dimension(array, Dim.X)
        self._halo_swap_by_dimension(array, Dim.Y)
        self._halo_swap_by_dimension(array, Dim.Z)

    def halo_swap_electric(self):
        self._halo_swap_array(self.Ex)
        self._halo_swap_array(self.Ey)
        self._halo_swap_array(self.Ez)

    def halo_swap_magnetic(self):
        self._halo_swap_array(self.Hx)
        self._halo_swap_array(self.Hy)
        self._halo_swap_array(self.Hz)

    def build(self):
        self.calculate_local_extents()
        self.set_halo_map()
        self.scatter_grid()
        super().build()

    def has_neighbour(self, dim: Dim, dir: Dir) -> bool:
        return self.neighbours[dim][dir] != -1

    def set_halo_map(self):
        print(f"[Rank {self.rank}] Size = {self.size}")
        size = (self.size + 1).tolist()

        for dim in Dim:
            halo_size = (self.size + 1 - np.sum(self.neighbours >= 0, axis=1)).tolist()
            halo_size[dim] = 1
            start = [1 if self.has_neighbour(dim, Dir.NEG) else 0 for dim in Dim]

            if self.has_neighbour(dim, Dir.NEG):
                start[dim] = 1
                print(
                    f"[Rank {self.rank}, Dim {dim}, Dir {Dir.NEG}] Grid of size {size}, creating halo map of size {halo_size} at start {start}"
                )
                self.send_halo_map[dim][Dir.NEG] = MPI.FLOAT.Create_subarray(size, halo_size, start)
                start[dim] = 0
                print(
                    f"[Rank {self.rank}, Dim {dim}, Dir {Dir.NEG}] Grid of size {size}, creating halo map of size {halo_size} at start {start}"
                )
                self.recv_halo_map[dim][Dir.NEG] = MPI.FLOAT.Create_subarray(size, halo_size, start)

                self.send_halo_map[dim][Dir.NEG].Commit()
                self.recv_halo_map[dim][Dir.NEG].Commit()

            if self.has_neighbour(dim, Dir.POS):
                start[dim] = size[dim] - 2
                print(
                    f"[Rank {self.rank}, Dim {dim}, Dir {Dir.POS}] Grid of size {size}, creating halo map of size {halo_size} at start {start}"
                )
                self.send_halo_map[dim][Dir.POS] = MPI.FLOAT.Create_subarray(size, halo_size, start)
                start[dim] = size[dim] - 1
                print(
                    f"[Rank {self.rank}, Dim {dim}, Dir {Dir.POS}] Grid of size {size}, creating halo map of size {halo_size} at start {start}"
                )
                self.recv_halo_map[dim][Dir.POS] = MPI.FLOAT.Create_subarray(size, halo_size, start)

                self.send_halo_map[dim][Dir.POS].Commit()
                self.recv_halo_map[dim][Dir.POS].Commit()

    def calculate_local_extents(self):
        print(f"[Rank {self.rank}] Global size = {self.global_size}")
        self.size = self.global_size // self.mpi_tasks
        print(f"[Rank {self.rank}] Initial size = {self.size}")

        self.lower_extent = self.size * self.coords + np.minimum(
            self.coords, self.global_size % self.mpi_tasks
        )

        print(f"[Rank {self.rank}] Lower extent = {self.lower_extent}")

        self.size += self.coords < self.global_size % self.mpi_tasks

        print(f"[Rank {self.rank}] Expanded size = {self.size}")

        # at_end = (self.mpi_tasks - self.coords) <= 1
        # self.size += at_end * self.global_size % self.mpi_tasks

        # Account for negative halo
        # Field arrays are created with dimensions size + 1 so space for
        # a positive halo will always exist. Grids not needing a
        # positive halo, still need the extra size as that makes the
        # global grid on the whole one larger than the user dimensions.
        self.size += self.neighbours[:, 0] >= 0
        self.lower_extent -= self.neighbours[:, 0] >= 0
        self.upper_extent = self.lower_extent + self.size

        print(f"[Rank {self.rank}] With positive halo size = {self.size}")
