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
import logging
from enum import IntEnum, unique
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from numpy import ndarray

from gprMax import config
from gprMax.cython.pml_build import pml_sum_er_mr
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.pml import MPIPML, PML
from gprMax.receivers import Rx
from gprMax.snapshots import MPISnapshot, Snapshot
from gprMax.sources import Source

logger = logging.getLogger(__name__)

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
        self.size = np.zeros(3, dtype=np.intc)

        super().__init__()

        self.comm = comm
        self.x_comm = comm.Sub([False, True, True])
        self.y_comm = comm.Sub([True, False, True])
        self.z_comm = comm.Sub([True, True, False])
        self.pml_comm = MPI.COMM_NULL

        self.mpi_tasks = np.array(self.comm.dims, dtype=np.intc)

        self.lower_extent = np.zeros(3, dtype=np.intc)
        self.upper_extent = np.zeros(3, dtype=np.intc)
        self.negative_halo_offset = np.zeros(3, dtype=np.bool_)
        self.global_size = np.zeros(3, dtype=np.intc)

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

    def get_grid_coord_from_coordinate(self, coord: npt.NDArray) -> npt.NDArray[np.intc]:
        step_size = self.global_size // self.mpi_tasks
        overflow = self.global_size % self.mpi_tasks
        return np.where(
            (step_size + 1) * overflow >= coord,
            coord // (step_size + 1),
            np.minimum((coord - overflow) // np.maximum(step_size, 1), self.mpi_tasks - 1),
        )

    def get_rank_from_coordinate(self, coord: npt.NDArray) -> int:
        grid_coord = self.get_grid_coord_from_coordinate(coord)
        return self.comm.Get_cart_rank(grid_coord.tolist())

    def get_ranks_between_coordinates(
        self, start_coord: npt.NDArray, stop_coord: npt.NDArray
    ) -> List[int]:
        start = self.get_grid_coord_from_coordinate(start_coord)
        stop = self.get_grid_coord_from_coordinate(stop_coord) + 1
        coord_to_rank = lambda c: self.comm.Get_cart_rank((start + c).tolist())
        return [coord_to_rank(coord) for coord in np.ndindex(*(stop - start))]

    def global_to_local_coordinate(
        self, global_coord: npt.NDArray[np.intc]
    ) -> npt.NDArray[np.intc]:
        return global_coord - self.lower_extent

    def local_to_global_coordinate(self, local_coord: npt.NDArray[np.intc]) -> npt.NDArray[np.intc]:
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

    def scatter_snapshots(self):
        if self.is_coordinator():
            snapshots_by_rank: List[List[Optional[Snapshot]]] = [[] for _ in range(self.comm.size)]
            for s in self.snapshots:
                ranks = self.get_ranks_between_coordinates(s.start, s.stop + s.step)
                for rank in range(
                    self.comm.size
                ):  # TODO: Loop over ranks in snapshot, not all ranks
                    if rank in ranks:
                        snapshots_by_rank[rank].append(s)
                    else:
                        snapshots_by_rank[rank].append(None)
        else:
            snapshots_by_rank = None

        snapshots: List[Optional[MPISnapshot]] = self.comm.scatter(
            snapshots_by_rank, root=self.COORDINATOR_RANK
        )

        for s in snapshots:
            if s is None:
                self.comm.Split(MPI.UNDEFINED)
            else:
                comm = self.comm.Split()
                assert isinstance(comm, MPI.Intracomm)
                start = self.get_grid_coord_from_coordinate(s.start)
                stop = self.get_grid_coord_from_coordinate(s.stop + s.step) + 1
                s.comm = comm.Create_cart((stop - start).tolist())

                s.start = self.global_to_local_coordinate(s.start)
                # Calculate number of steps needed to bring the start
                # into the local grid (and not in the negative halo)
                s.offset = np.where(
                    s.start < self.negative_halo_offset,
                    np.abs((s.start - self.negative_halo_offset) // s.step),
                    s.offset,
                )
                s.start += s.step * s.offset

                s.stop = self.global_to_local_coordinate(s.stop)
                s.stop = np.where(s.stop > self.size, self.size, s.stop)

        self.snapshots = [s for s in snapshots if s is not None]

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

    def scatter_4d_array_with_positive_halo(self, array: npt.NDArray) -> npt.NDArray:
        self.comm.Bcast(array, root=self.COORDINATOR_RANK)

        return array[
            :,
            self.lower_extent[Dim.X] : self.upper_extent[Dim.X] + 1,
            self.lower_extent[Dim.Y] : self.upper_extent[Dim.Y] + 1,
            self.lower_extent[Dim.Z] : self.upper_extent[Dim.Z] + 1,
        ].copy(order="C")

    def scatter_grid(self):
        self.materials = self.comm.bcast(self.materials, root=self.COORDINATOR_RANK)
        self.rxs = self.scatter_coord_objects(self.rxs)
        self.voltagesources = self.scatter_coord_objects(self.voltagesources)
        self.magneticdipoles = self.scatter_coord_objects(self.magneticdipoles)
        self.hertziandipoles = self.scatter_coord_objects(self.hertziandipoles)
        self.transmissionlines = self.scatter_coord_objects(self.transmissionlines)

        self.scatter_snapshots()

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

        if not self.is_coordinator():
            # TODO: When scatter arrays properly, should initialise these to the local grid size
            self.initialise_geometry_arrays()

        self.ID = self.scatter_4d_array_with_positive_halo(self.ID)
        self.solid = self.scatter_3d_array(self.solid)
        self.rigidE = self.scatter_4d_array(self.rigidE)
        self.rigidH = self.scatter_4d_array(self.rigidH)

    def gather_grid_objects(self):
        self.rxs = self.gather_coord_objects(self.rxs)
        self.voltagesources = self.gather_coord_objects(self.voltagesources)
        self.magneticdipoles = self.gather_coord_objects(self.magneticdipoles)
        self.hertziandipoles = self.gather_coord_objects(self.hertziandipoles)
        self.transmissionlines = self.gather_coord_objects(self.transmissionlines)

    def initialise_geometry_arrays(self, use_local_size=False):
        if use_local_size:
            super().initialise_geometry_arrays()
        else:
            self.solid = np.ones(self.global_size, dtype=np.uint32)
            self.rigidE = np.zeros((12, *self.global_size), dtype=np.int8)
            self.rigidH = np.zeros((6, *self.global_size), dtype=np.int8)
            self.ID = np.ones((6, *(self.global_size + 1)), dtype=np.uint32)

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

    def _construct_pml(self, pml_ID: str, thickness: int) -> MPIPML:
        pml = super()._construct_pml(pml_ID, thickness, MPIPML)
        if pml.ID[0] == "x":
            pml.comm = self.x_comm
        elif pml.ID[0] == "y":
            pml.comm = self.y_comm
        elif pml.ID[0] == "z":
            pml.comm = self.z_comm
        pml.global_comm = self.pml_comm

        return pml

    def _calculate_average_pml_material_properties(self, pml: MPIPML) -> Tuple[float, float]:
        # Arrays to hold values of permittivity and permeability (avoids
        # accessing Material class in Cython.)
        ers = np.zeros(len(self.materials))
        mrs = np.zeros(len(self.materials))

        for i, m in enumerate(self.materials):
            ers[i] = m.er
            mrs[i] = m.mr

        # Need to account for the negative halo (remove it) to avoid
        # double counting. The solid array does not have a positive halo
        # so we don't need to consider that.
        if pml.ID[0] == "x":
            o1 = self.negative_halo_offset[1]
            o2 = self.negative_halo_offset[2]
            n1 = self.ny - o1
            n2 = self.nz - o2
            solid = self.solid[pml.xs, o1:, o2:]
        elif pml.ID[0] == "y":
            o1 = self.negative_halo_offset[0]
            o2 = self.negative_halo_offset[2]
            n1 = self.nx - o1
            n2 = self.nz - o2
            solid = self.solid[o1:, pml.ys, o2:]
        elif pml.ID[0] == "z":
            o1 = self.negative_halo_offset[0]
            o2 = self.negative_halo_offset[1]
            n1 = self.nx - o1
            n2 = self.ny - o2
            solid = self.solid[o1:, o2:, pml.zs]
        else:
            raise ValueError(f"Unknown PML ID '{pml.ID}'")

        sumer, summr = pml_sum_er_mr(n1, n2, config.get_model_config().ompthreads, solid, ers, mrs)
        n = pml.comm.allreduce(n1 * n2, MPI.SUM)
        sumer = pml.comm.allreduce(sumer, MPI.SUM)
        summr = pml.comm.allreduce(summr, MPI.SUM)
        averageer = sumer / n
        averagemr = summr / n

        return averageer, averagemr

    def build(self):
        if any(self.global_size + 1 < self.mpi_tasks):
            logger.error(
                f"Too many MPI tasks requested ({self.mpi_tasks}) for grid of size {self.global_size + 1}. Make sure the number of MPI tasks in each dimension is less than the size of the grid."
            )
            raise ValueError

        self.calculate_local_extents()
        self.set_halo_map()
        self.scatter_grid()

        # TODO: Check PML is not thicker than the grid size

        # Get PMLs present in this grid
        pmls = [
            PML.boundaryIDs.index(key) for key, value in self.pmls["thickness"].items() if value > 0
        ]
        if len(pmls) > 0:
            # Use PML ID as the key to ensure rank 0 is always the same
            # PML. This is needed to ensure the CFS sigma.max parameter
            # is calculated using the first PML present.
            self.pml_comm = self.comm.Split(0, pmls[0])
        else:
            self.pml_comm = self.comm.Split(MPI.UNDEFINED)

        super().build()

    def has_neighbour(self, dim: Dim, dir: Dir) -> bool:
        return self.neighbours[dim][dir] != -1

    def set_halo_map(self):
        size = (self.size + 1).tolist()

        for dim in Dim:
            halo_size = (self.size + 1 - np.sum(self.neighbours >= 0, axis=1)).tolist()
            halo_size[dim] = 1
            start = [1 if self.has_neighbour(dim, Dir.NEG) else 0 for dim in Dim]

            if self.has_neighbour(dim, Dir.NEG):
                start[dim] = 1
                self.send_halo_map[dim][Dir.NEG] = MPI.FLOAT.Create_subarray(size, halo_size, start)
                start[dim] = 0
                self.recv_halo_map[dim][Dir.NEG] = MPI.FLOAT.Create_subarray(size, halo_size, start)

                self.send_halo_map[dim][Dir.NEG].Commit()
                self.recv_halo_map[dim][Dir.NEG].Commit()

            if self.has_neighbour(dim, Dir.POS):
                start[dim] = size[dim] - 2
                self.send_halo_map[dim][Dir.POS] = MPI.FLOAT.Create_subarray(size, halo_size, start)
                start[dim] = size[dim] - 1
                self.recv_halo_map[dim][Dir.POS] = MPI.FLOAT.Create_subarray(size, halo_size, start)

                self.send_halo_map[dim][Dir.POS].Commit()
                self.recv_halo_map[dim][Dir.POS].Commit()

    def calculate_local_extents(self):
        self.size = self.global_size // self.mpi_tasks
        overflow = self.global_size % self.mpi_tasks

        # Ranks with coordinates less than the overflow have their size
        # increased by one. Account for this by adding the overflow or
        # this rank's coordinates, whichever is smaller.
        self.lower_extent = self.size * self.coords + np.minimum(self.coords, overflow)

        # For each coordinate, if it is less than the overflow, add 1
        self.size += self.coords < overflow

        # Account for a negative halo
        # Field arrays are created with dimensions size + 1 so space for
        # a positive halo will always exist. Grids not needing a
        # positive halo, still need the extra size as that makes the
        # global grid on the whole one larger than the user dimensions.
        self.negative_halo_offset = self.neighbours[:, 0] >= 0
        self.size += self.negative_halo_offset
        self.lower_extent -= self.negative_halo_offset
        self.upper_extent = self.lower_extent + self.size

        logger.debug(
            f"Grid size: {self.size}, Lower extent: {self.lower_extent}, Upper extent: {self.upper_extent}"
        )
