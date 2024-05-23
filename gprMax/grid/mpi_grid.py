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

from enum import IntEnum, unique
from typing import List

import numpy as np
import numpy.typing as npt
from matplotlib.sankey import UP
from mpi4py import MPI
from numpy import ndarray

from gprMax.grid.fdtd_grid import FDTDGrid


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
        print(f"[Rank {self.rank}] Neighbours = {self.neighbours}")

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

    def broadcast_grid(self):
        self.calculate_local_extents()
        print(
            f"[Rank {self.rank}] - Global size: {self.global_size}, Local size: {self.size}, Cart comm dims: {self.mpi_tasks}, Local coordinates: {self.coords}, Lower extent = {self.lower_extent}, Upper extent = {self.upper_extent}"
        )

    def _get_halo(self, array: ndarray, dim: Dim, direction: Dir) -> ndarray:
        if direction == Dir.NEG:
            index = 0
        else:  # Direction.UP
            index = -1

        if dim == Dim.X:
            halo = array[:, index, index]
        elif dim == Dim.Y:
            halo = array[index, :, index]
        else:  # Dim.Z
            halo = array[index, index, :]

        return halo

    def _set_halo(self, array: ndarray, halo: ndarray, dim: Dim, direction: Dir):
        if direction == Dir.NEG:
            index = 0
        else:  # Direction.UP
            index = -1

        if dim == Dim.X:
            array[:, index, index] = halo
        elif dim == Dim.Y:
            array[index, :, index] = halo
        else:  # Dim.Z
            array[index, index, :] = halo

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

    def _halo_swap_dimension(self, array: ndarray, dim: Dim):
        if self.coords[dim] % 2 == 0:
            self._halo_swap(array, dim, Dir.NEG)
            self._halo_swap(array, dim, Dir.POS)
        else:
            self._halo_swap(array, dim, Dir.POS)
            self._halo_swap(array, dim, Dir.NEG)

    def halo_swap(self, array: ndarray):
        self._halo_swap_dimension(array, Dim.X)
        self._halo_swap_dimension(array, Dim.Y)
        self._halo_swap_dimension(array, Dim.Z)

    def build(self):
        self.calculate_local_extents()
        self.set_halo_map()
        super().build()
        self.halo_swap(self.Ex)

    def set_halo_map(self):
        size = self.size.tolist()

        for dim in Dim:
            halo_size = (self.size - 2).tolist()
            halo_size[dim] = 1

            start = [1, 1, 1]
            self.send_halo_map[dim][Dir.NEG] = MPI.DOUBLE.Create_subarray(size, halo_size, start)
            start[dim] = size[dim] - 2
            self.send_halo_map[dim][Dir.POS] = MPI.DOUBLE.Create_subarray(size, halo_size, start)

            start[dim] = 0
            self.recv_halo_map[dim][Dir.NEG] = MPI.DOUBLE.Create_subarray(size, halo_size, start)
            start[dim] = size[dim] - 1
            self.recv_halo_map[dim][Dir.POS] = MPI.DOUBLE.Create_subarray(size, halo_size, start)

            self.send_halo_map[dim][Dir.NEG].Commit()
            self.send_halo_map[dim][Dir.POS].Commit()
            self.recv_halo_map[dim][Dir.NEG].Commit()
            self.recv_halo_map[dim][Dir.POS].Commit()

    def calculate_local_extents(self):
        self.size = self.global_size // self.mpi_tasks

        self.lower_extent = self.size * self.coords

        at_end = (self.mpi_tasks - self.coords) <= 1
        self.size += at_end * self.global_size % self.mpi_tasks

        self.upper_extent = self.lower_extent + self.size

        # Account for halo
        self.size += 2

    def initialise_field_arrays(self):
        super().initialise_field_arrays()
