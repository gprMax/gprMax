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

import itertools
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from numpy import ndarray

from gprMax import config
from gprMax.cython.pml_build import pml_sum_er_mr
from gprMax.fractals.fractal_surface import MPIFractalSurface
from gprMax.fractals.fractal_volume import MPIFractalVolume
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.pml import MPIPML, PML, InternalPMLSpec
from gprMax.receivers import Rx
from gprMax.sources import Source
from gprMax.utilities.mpi import Dim, Dir, mpi_datatype_for_dtype

logger = logging.getLogger(__name__)

CoordType = TypeVar("CoordType", bound=Union[Rx, Source])
ObjectType = TypeVar("ObjectType")


class MPIGrid(FDTDGrid):
    HALO_SIZE = 1
    COORDINATOR_RANK = 0
    pml_type = MPIPML

    def __init__(self, comm: MPI.Cartcomm):
        self.comm = comm
        self.x_comm = comm.Sub([False, True, True])
        self.y_comm = comm.Sub([True, False, True])
        self.z_comm = comm.Sub([True, True, False])
        self.x_line_comm = comm.Sub([True, False, False])
        self.y_line_comm = comm.Sub([False, True, False])
        self.z_line_comm = comm.Sub([False, False, True])
        self.pml_comm = MPI.COMM_NULL

        self.mpi_tasks = np.array(self.comm.dims, dtype=np.int32)

        self.lower_extent = np.zeros(3, dtype=np.int32)
        self.upper_extent = np.zeros(3, dtype=np.int32)
        self.negative_halo_offset = np.zeros(3, dtype=np.bool_)
        self.global_size = np.zeros(3, dtype=np.int32)

        self.neighbours = np.full((3, 2), -1, dtype=np.int32)
        self.neighbours[Dim.X] = self.comm.Shift(direction=Dim.X, disp=1)
        self.neighbours[Dim.Y] = self.comm.Shift(direction=Dim.Y, disp=1)
        self.neighbours[Dim.Z] = self.comm.Shift(direction=Dim.Z, disp=1)

        self.send_halo_map = np.empty((3, 2), dtype=MPI.Datatype)
        self.recv_halo_map = np.empty((3, 2), dtype=MPI.Datatype)
        self.send_halo_map.fill(MPI.DATATYPE_NULL)
        self.recv_halo_map.fill(MPI.DATATYPE_NULL)
        self._halo_maps_initialised = False
        self._halo_maps_freed = False
        self.send_requests: List[MPI.Request] = []
        self.recv_requests: List[MPI.Request] = []

        super().__init__()

    @property
    def rank(self) -> int:
        return self.comm.Get_rank()

    @property
    def coords(self) -> List[int]:
        return self.comm.coords

    @property
    def gx(self) -> int:
        return self.global_size[Dim.X]

    @gx.setter
    def gx(self, value: int):
        self.global_size[Dim.X] = value

    @property
    def gy(self) -> int:
        return self.global_size[Dim.Y]

    @gy.setter
    def gy(self, value: int):
        self.global_size[Dim.Y] = value

    @property
    def gz(self) -> int:
        return self.global_size[Dim.Z]

    @gz.setter
    def gz(self, value: int):
        self.global_size[Dim.Z] = value

    def set_pml_thickness(self, thickness: Union[int, Tuple[int, int, int, int, int, int]]):
        super().set_pml_thickness(thickness)

        # Set PML thickness to zero if not at the edge of the domain
        if self.has_neighbour(Dim.X, Dir.NEG):
            self.pmls["thickness"]["x0"] = 0
        if self.has_neighbour(Dim.X, Dir.POS):
            self.pmls["thickness"]["xmax"] = 0
        if self.has_neighbour(Dim.Y, Dir.NEG):
            self.pmls["thickness"]["y0"] = 0
        if self.has_neighbour(Dim.Y, Dir.POS):
            self.pmls["thickness"]["ymax"] = 0
        if self.has_neighbour(Dim.Z, Dir.NEG):
            self.pmls["thickness"]["z0"] = 0
        if self.has_neighbour(Dim.Z, Dir.POS):
            self.pmls["thickness"]["zmax"] = 0

    def add_fractal_volume(
        self,
        xs: int,
        xf: int,
        ys: int,
        yf: int,
        zs: int,
        zf: int,
        frac_dim: float,
        seed: Optional[int],
    ) -> MPIFractalVolume:
        volume = MPIFractalVolume(xs, xf, ys, yf, zs, zf, frac_dim, seed, self.comm, self.size)
        self.fractalvolumes.append(volume)
        return volume

    def create_fractal_surface(
        self,
        xs: int,
        xf: int,
        ys: int,
        yf: int,
        zs: int,
        zf: int,
        frac_dim: float,
        seed: Optional[int],
    ) -> MPIFractalSurface:
        return MPIFractalSurface(xs, xf, ys, yf, zs, zf, frac_dim, seed, self.comm, self.size)

    def is_coordinator(self) -> bool:
        """Test if the current rank is the coordinator.

        Returns:
            is_coordinator: True if `self.rank` equals
                `self.COORDINATOR_RANK`.
        """
        return self.rank == self.COORDINATOR_RANK

    def create_sub_communicator(
        self, local_start: npt.NDArray[np.int32], local_stop: npt.NDArray[np.int32]
    ) -> Optional[MPI.Cartcomm]:
        if self.local_bounds_overlap_grid(local_start, local_stop):
            comm = self.comm.Split()
            assert isinstance(comm, MPI.Intracomm)
            start_grid_coord = self.get_grid_coord_from_local_coordinate(local_start)
            # Subtract 1 from local_stop as the upper extent is
            # exclusive meaning the last coordinate included in the sub
            # communicator is actually (local_stop - 1).
            stop_grid_coord = self.get_grid_coord_from_local_coordinate(local_stop - 1) + 1
            comm = comm.Create_cart((stop_grid_coord - start_grid_coord).tolist())
            return comm
        else:
            self.comm.Split(MPI.UNDEFINED)
            return None

    def get_grid_coord_from_local_coordinate(
        self, local_coord: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.int32]:
        """Get the MPI grid coordinate for a local grid coordinate.

        Args:
            local_coord: Local grid coordinate.

        Returns:
            grid_coord: Coordinate of the MPI rank containing the local
                grid coordinate.
        """
        coord = self.local_to_global_coordinate(local_coord)
        return self.get_grid_coord_from_coordinate(coord)

    def get_grid_coord_from_coordinate(self, coord: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        """Get the MPI grid coordinate for a global grid coordinate.

        Args:
            coord: Global grid coordinate.

        Returns:
            grid_coord: Coordinate of the MPI rank containing the global
                grid coordinate.
        """
        step_size = self.global_size // self.mpi_tasks
        overflow = self.global_size % self.mpi_tasks

        # The first n MPI ranks where n is the overflow, will have size
        # step_size + 1. Additionally, step_size may be zero in some
        # dimensions (e.g. in the 2D case) so we need to avoid division
        # by zero.
        return np.where(
            (step_size + 1) * overflow >= coord,
            coord // (step_size + 1),
            np.minimum((coord - overflow) // np.maximum(step_size, 1), self.mpi_tasks - 1),
        )

    def get_rank_from_coordinate(self, coord: npt.NDArray) -> int:
        """Get the MPI rank for a global grid coordinate.

        A coordinate only exists on a single rank (halos are ignored).

        Args:
            coord: Global grid coordinate.

        Returns:
            rank: MPI rank containing the global grid coordinate.
        """
        grid_coord = self.get_grid_coord_from_coordinate(coord)
        return self.comm.Get_cart_rank(grid_coord.tolist())

    def get_ranks_between_coordinates(
        self, start_coord: npt.NDArray, stop_coord: npt.NDArray
    ) -> List[int]:
        """Get the MPI ranks for between two global grid coordinates.

        `stop_coord` must not be less than `start_coord` in any
        dimension, however it can be equal. The returned ranks will
        contain coordinates inclusive of both `start_coord` and
        `stop_coord`.

        Args:
            start_coord: Starting global grid coordinate.
            stop_coord: End global grid coordinate.

        Returns:
            ranks: List of MPI ranks
        """
        start = self.get_grid_coord_from_coordinate(start_coord)
        stop = self.get_grid_coord_from_coordinate(stop_coord) + 1
        coord_to_rank = lambda c: self.comm.Get_cart_rank((start + c).tolist())
        return [coord_to_rank(coord) for coord in np.ndindex(*(stop - start))]

    def global_to_local_coordinate(
        self, global_coord: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.int32]:
        """Convert a global grid coordinate to a local grid coordinate.

        The returned coordinate will be relative to the current MPI
        rank's local grid. It may be negative, or greater than the size
        of the local grid if the point lies outside the local grid.

        Args:
            global_coord: Global grid coordinate.

        Returns:
            local_coord: Local grid coordinate.
        """
        return global_coord - self.lower_extent

    def local_to_global_coordinate(
        self, local_coord: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.int32]:
        """Convert a local grid coordinate to a global grid coordinate.

        Args:
            local_coord: Local grid coordinate.

        Returns:
            global_coord: Global grid coordinate.
        """
        return local_coord + self.lower_extent

    def global_coord_inside_grid(
        self, global_coord: npt.NDArray[np.int32], allow_inside_halo: bool = False
    ) -> bool:
        """Check if a global coordinate falls with in the local grid.

        Args:
            global_coord: Global grid coordinate.
            allow_inside_halo: If True, the function returns True when
                the coordinate is inside the grid halo. Otherwise, it
                will return False when the coordinate is inside the grid
                halo. (defaults to False)

        Returns:
            is_inside_grid: True if the global coordinate falls inside
                the local grid bounds.
        """
        if allow_inside_halo:
            lower_bound = self.lower_extent
            upper_bound = self.upper_extent + 1
        else:
            lower_bound = self.lower_extent + self.negative_halo_offset
            upper_bound = self.upper_extent

        return all(global_coord >= lower_bound) and all(global_coord <= upper_bound)

    def local_bounds_overlap_grid(
        self, local_start: npt.NDArray[np.int32], local_stop: npt.NDArray[np.int32]
    ) -> bool:
        """Check if local bounds overlap with the grid.

        The bounds overlap if any of the 3D box as defined by the lower
        and upper bounds overlaps with the local grid (excluding the
        halo).

        Args:
            local_start: Lower bound in the local grid coordinate space.
            local_stop: Upper bound in the local grid coordinate space.

        Returns:
            overlaps_grid: True if the box generated by the lower and
                upper bound overlaps with the local grid.
        """
        return all(local_start < self.size) and all(local_stop > self.negative_halo_offset)

    def gather_coord_objects(self, objects: List[CoordType]) -> List[CoordType]:
        """Gather coord objects on the coordinator MPI rank.

        The sending MPI rank converts the object locations to the global
        grid. The coord objects (sources and receivers) are all sent to
        the coordinatoor rank.

        Args:
            objects: Coord objects to be gathered.

        Returns:
            gathered_objects: List of gathered coord objects if the
                current rank is the coordinator. Otherwise, the original
                list of objects is returned.
        """
        for o in objects:
            o.coord = self.local_to_global_coordinate(o.coord)

        gathered_objects: Optional[List[List[CoordType]]] = self.comm.gather(
            objects, root=self.COORDINATOR_RANK
        )

        if gathered_objects is not None:
            return list(itertools.chain(*gathered_objects))
        else:
            return objects

    def gather_objects(self, objects: List[ObjectType]) -> List[ObjectType]:
        """Gather non-coordinate runtime objects on the coordinator rank."""

        gathered_objects: Optional[List[List[ObjectType]]] = self.comm.gather(
            objects, root=self.COORDINATOR_RANK
        )
        if gathered_objects is not None:
            return list(itertools.chain(*gathered_objects))
        return objects

    def gather_grid_objects(self):
        """Gather sources and receivers."""

        self.rxs = self.gather_coord_objects(self.rxs)
        self.voltagesources = self.gather_coord_objects(self.voltagesources)
        self.magneticdipoles = self.gather_coord_objects(self.magneticdipoles)
        self.hertziandipoles = self.gather_coord_objects(self.hertziandipoles)
        self.transmissionlines = self.gather_coord_objects(self.transmissionlines)
        primary_frills = [
            source for source in self.magneticfrillsources if getattr(source, "mpi_primary", True)
        ]
        self.magneticfrillsources = self.gather_coord_objects(primary_frills)
        if self.is_coordinator():
            self.magneticfrillsources.sort(
                key=lambda source: getattr(source, "mpi_global_index", 0)
            )
        self.networkterminals = self.gather_coord_objects(self.networkterminals)
        self.port_monitors = self.gather_objects(self.port_monitors)

    @contextmanager
    def gathered_output_state(self) -> Iterator[None]:
        """Temporarily expose globally gathered objects for output writing.

        ``gather_grid_objects`` converts coordinates to the global frame and,
        on the coordinator, replaces each rank-local list with the gathered
        list. Output writing also temporarily presents the global grid size.
        Those mutations are harmless when a grid is immediately discarded,
        but corrupt the next solve when ``geometry_fixed`` reuses the grid.

        Preserve the rank-local runtime state and restore it after output has
        been written. The ``finally`` block also protects reuse if output
        finalisation or HDF5 writing raises an exception.
        """

        coordinate_list_names = (
            "rxs",
            "voltagesources",
            "magneticdipoles",
            "hertziandipoles",
            "transmissionlines",
            "magneticfrillsources",
            "networkterminals",
        )
        object_list_names = coordinate_list_names + ("port_monitors",)
        local_lists = {name: getattr(self, name) for name in object_list_names}
        local_coordinates = [
            (item, item.coord) for name in coordinate_list_names for item in local_lists[name]
        ]
        local_size = self.size

        try:
            self.gather_grid_objects()
            yield
        finally:
            self.size = local_size
            for item, coordinate in local_coordinates:
                item.coord = coordinate
            for name, objects in local_lists.items():
                setattr(self, name, objects)

    def reduce_eigenmode_ports(self):
        """Sum distributed modal DFT projections onto the coordinator.

        Eigenmode bases and Gram matrices are identical on every rank, while
        each rank accumulates only the cells it owns on a modal plane. Reduce
        the two complex overlap histories once after time stepping; no modal
        collective is needed inside the FDTD iteration.
        """

        signatures = tuple(
            (port.port_index, port.output_id, tuple(port.mode_indices))
            for port in self.eigenmodeports
        )
        all_signatures = self.comm.allgather(signatures)
        if any(value != signatures for value in all_signatures):
            raise RuntimeError("MPI ranks constructed inconsistent eigenmode-port monitors.")

        for port in self.eigenmodeports:
            electric = np.empty_like(port.electric_dft) if self.is_coordinator() else None
            magnetic = np.empty_like(port.magnetic_dft) if self.is_coordinator() else None
            self.comm.Reduce(port.electric_dft, electric, op=MPI.SUM, root=self.COORDINATOR_RANK)
            self.comm.Reduce(port.magnetic_dft, magnetic, op=MPI.SUM, root=self.COORDINATOR_RANK)
            if self.is_coordinator():
                port.electric_dft = electric
                port.magnetic_dft = magnetic

    def gather_sar_payloads(self):
        """Gather owned-cell SAR payloads once after time stepping.

        Electric-field DFTs remain rank-local during the solve. Each monitor
        contributes its owned semantic cells and uniquely owned sparse Yee-edge
        DFTs. The coordinator performs global cell-centre collocation before
        source/port normalisation and mass averaging, after the complete model
        output state has been gathered.
        """

        if not self.sar_monitors:
            return None

        signatures = tuple(
            (
                monitor.output_id,
                tuple(float(value) for value in monitor.frequencies),
                monitor.tag_names,
            )
            for monitor in self.sar_monitors
        )
        all_signatures = self.comm.allgather(signatures)
        if any(value != signatures for value in all_signatures):
            raise RuntimeError("MPI ranks constructed inconsistent SAR monitors.")

        local_payloads = [monitor.local_payload() for monitor in self.sar_monitors]
        gathered = self.comm.gather(local_payloads, root=self.COORDINATOR_RANK)
        if gathered is None:
            return None
        return [
            [rank_payloads[index] for rank_payloads in gathered]
            for index in range(len(self.sar_monitors))
        ]

    def _halo_swap(self, array: ndarray, dim: Dim, dir: Dir):
        """Perform a halo swap in the specifed dimension and direction.

        If no neighbour exists for the current rank in the specifed
        dimension and direction, the halo swap is skipped.

        Args:
            array: Array to perform the halo swap with.
            dim: Dimension of halo to swap.
            dir: Direction of halo to swap.
        """
        neighbour = self.neighbours[dim][dir]
        if neighbour >= 0:
            send_request = self.comm.Isend([array, self.send_halo_map[dim][dir]], neighbour)
            recv_request = self.comm.Irecv([array, self.recv_halo_map[dim][dir]], neighbour)
            self.send_requests.append(send_request)
            self.recv_requests.append(recv_request)

    def _halo_swap_by_dimension(self, array: ndarray, dim: Dim):
        """Perform halo swaps in the specifed dimension.

        Perform a halo swaps in the positive and negative direction for
        the specified dimension. The order of the swaps is determined by
        the current rank's MPI grid coordinate to prevent deadlock.

        Args:
            array: Array to perform the halo swaps with.
            dim: Dimension of halos to swap.
        """
        if self.coords[dim] % 2 == 0:
            self._halo_swap(array, dim, Dir.NEG)
            self._halo_swap(array, dim, Dir.POS)
        else:
            self._halo_swap(array, dim, Dir.POS)
            self._halo_swap(array, dim, Dir.NEG)

    def _halo_swap_array(self, array: ndarray):
        """Perform halo swaps for the specified array.

        Args:
            array: Array to perform the halo swaps with.
        """
        self._halo_swap_by_dimension(array, Dim.X)
        self._halo_swap_by_dimension(array, Dim.Y)
        self._halo_swap_by_dimension(array, Dim.Z)

    def halo_swap_electric(self):
        """Perform halo swaps for electric field arrays."""

        # Ensure send requests for the magnetic field have completed
        # The magnetic field arrays may change after this halo swap in
        # the magnetic update step
        if len(self.send_requests) > 0:
            self.send_requests[0].Waitall(self.send_requests)
            self.send_requests = []

        self._halo_swap_array(self.Ex)
        self._halo_swap_array(self.Ey)
        self._halo_swap_array(self.Ez)

        # Wait for all receive requests to complete
        # Don't need to wait for send requests yet as the electric
        # field arrays won't be changed during the magnetic update steps
        if len(self.recv_requests) > 0:
            self.recv_requests[0].Waitall(self.recv_requests)
            self.recv_requests = []

    def halo_swap_magnetic(self):
        """Perform halo swaps for magnetic field arrays."""

        # Ensure send requests for the electric field have completed
        # The electric field arrays will change after this halo swap in
        # the electric update step
        if len(self.send_requests) > 0:
            self.send_requests[0].Waitall(self.send_requests)
            self.send_requests = []

        self._halo_swap_array(self.Hx)
        self._halo_swap_array(self.Hy)
        self._halo_swap_array(self.Hz)

        # Wait for all receive requests to complete
        # Don't need to wait for send requests yet as the magnetic
        # field arrays won't be changed during the electric update steps
        if len(self.recv_requests) > 0:
            self.recv_requests[0].Waitall(self.recv_requests)
            self.recv_requests = []

    def complete_halo_swaps(self) -> None:
        """Complete requests left by the final nonblocking halo exchange.

        A normal timestep waits for the previous field sends at the start of
        the opposite-field halo exchange. The final electric exchange has no
        following magnetic exchange, so its sends must be completed explicitly
        before MPI finalisation.
        """

        if self.send_requests:
            MPI.Request.Waitall(self.send_requests)
            self.send_requests = []
        if self.recv_requests:
            MPI.Request.Waitall(self.recv_requests)
            self.recv_requests = []

    def free_halo_maps(self) -> None:
        """Release committed MPI datatypes when this grid is retired.

        The derived datatypes describe non-contiguous field-array halo faces
        and must remain valid for the complete lifetime of the grid. In
        particular, a geometry-fixed series reuses them on every model run,
        so solver-level cleanup would free them too early. The owning
        ``MPIContext`` calls this method only before discarding an ordinary
        grid or after the final geometry-fixed run.

        Cleanup is idempotent so a future error-handling path can safely call
        it without risking a second ``MPI_Type_free`` operation.
        """

        if not self._halo_maps_initialised or self._halo_maps_freed:
            return

        self.complete_halo_swaps()
        for dim in Dim:
            for direction in Dir:
                for halo_maps in (self.send_halo_map, self.recv_halo_map):
                    datatype = halo_maps[dim][direction]
                    if datatype != MPI.DATATYPE_NULL:
                        datatype.Free()
                        halo_maps[dim][direction] = MPI.DATATYPE_NULL

        self._halo_maps_freed = True

    def _prepare_pml(self, pml: MPIPML) -> MPIPML:
        """Attach the communicator associated with the slab normal."""
        if pml.direction[0] == "x":
            pml.comm = self.x_comm
        elif pml.direction[0] == "y":
            pml.comm = self.y_comm
        elif pml.direction[0] == "z":
            pml.comm = self.z_comm
        pml.global_comm = self.pml_comm

        return pml

    @staticmethod
    def _internal_pml_bounds(spec: InternalPMLSpec) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray((spec.xs, spec.ys, spec.zs), dtype=np.int32),
            np.asarray((spec.xf, spec.yf, spec.zf), dtype=np.int32),
        )

    def _localise_internal_pml_spec(
        self, spec: InternalPMLSpec
    ) -> Tuple[InternalPMLSpec, int, int, bool] | None:
        """Clip a global internal-PML declaration to this rank's array.

        The local array includes its negative halo. Retaining that one-cell
        overlap is intentional: it lets the rank on the positive side of a
        partition own an internal slab's terminal electric plane, exactly as
        it owns the ordinary Yee electric field on that partition.
        """

        global_lower, global_upper = self._internal_pml_bounds(spec)
        local_global_lower = np.maximum(global_lower, self.lower_extent)
        local_global_upper = np.minimum(global_upper, self.upper_extent)
        if np.any(local_global_lower >= local_global_upper):
            return None

        local_lower = local_global_lower - self.lower_extent
        local_upper = local_global_upper - self.lower_extent
        local_spec = replace(
            spec,
            xs=int(local_lower[0]),
            xf=int(local_upper[0]),
            ys=int(local_lower[1]),
            yf=int(local_upper[1]),
            zs=int(local_lower[2]),
            zf=int(local_upper[2]),
        )

        axis = "xyz".index(spec.maximum_face[0])
        global_thickness = int(global_upper[axis] - global_lower[axis])
        if spec.direction.endswith("plus"):
            profile_offset = int(local_global_lower[axis] - global_lower[axis])
        else:
            profile_offset = int(global_upper[axis] - local_global_upper[axis])

        maximum_coordinate = (
            global_lower[axis] if spec.maximum_face.endswith("0") else global_upper[axis]
        )
        profile_endpoint = maximum_coordinate not in (0, self.global_size[axis])
        return local_spec, profile_offset, global_thickness, profile_endpoint

    def _construct_local_internal_pml(
        self,
        global_spec: InternalPMLSpec,
        local_spec: InternalPMLSpec,
        profile_offset: int,
        profile_thickness: int,
        profile_endpoint: bool,
    ) -> MPIPML:
        formulation, cfs = self._resolve_internal_pml_profile(global_spec)
        return self._new_pml(
            ID=global_spec.ID,
            direction=global_spec.direction,
            xs=local_spec.xs,
            xf=local_spec.xf,
            ys=local_spec.ys,
            yf=local_spec.yf,
            zs=local_spec.zs,
            zf=local_spec.zf,
            internal=True,
            maximum_face=global_spec.maximum_face,
            formulation=formulation,
            cfs=cfs,
            profile_id=global_spec.profile_id,
            profile_offset=profile_offset,
            profile_thickness=profile_thickness,
            profile_updates_terminal_e_plane=profile_endpoint,
        )

    def _calculate_average_internal_pml_material_properties(
        self, spec: InternalPMLSpec
    ) -> Tuple[float, float]:
        """Collectively average a global slab's zero-stretch entrance face."""

        ers = np.asarray([material.er for material in self.materials])
        mrs = np.asarray([material.mr for material in self.materials])
        lower, upper = self._internal_pml_bounds(spec)
        axis = "xyz".index(spec.maximum_face[0])
        entrance = upper[axis] - 1 if spec.direction.endswith("minus") else lower[axis]

        owned_lower = self.lower_extent + self.negative_halo_offset.astype(np.int32)
        owned_upper = self.upper_extent
        transverse_lower = np.maximum(lower, owned_lower)
        transverse_upper = np.minimum(upper, owned_upper)

        local_count = 0
        local_er = 0.0
        local_mr = 0.0
        if (
            owned_lower[axis] <= entrance < owned_upper[axis]
            and all(
                transverse_lower[dimension] < transverse_upper[dimension]
                for dimension in Dim
                if dimension != axis
            )
        ):
            local_lower = transverse_lower - self.lower_extent
            local_upper = transverse_upper - self.lower_extent
            entrance_local = int(entrance - self.lower_extent[axis])
            if axis == Dim.X:
                solid = self.solid[
                    entrance_local,
                    local_lower[Dim.Y] : local_upper[Dim.Y],
                    local_lower[Dim.Z] : local_upper[Dim.Z],
                ]
            elif axis == Dim.Y:
                solid = self.solid[
                    local_lower[Dim.X] : local_upper[Dim.X],
                    entrance_local,
                    local_lower[Dim.Z] : local_upper[Dim.Z],
                ]
            else:
                solid = self.solid[
                    local_lower[Dim.X] : local_upper[Dim.X],
                    local_lower[Dim.Y] : local_upper[Dim.Y],
                    entrance_local,
                ]

            n1, n2 = solid.shape
            local_count = n1 * n2
            local_er, local_mr = pml_sum_er_mr(
                n1,
                n2,
                config.get_model_config().ompthreads,
                solid,
                ers,
                mrs,
            )

        count = self.comm.allreduce(local_count, MPI.SUM)
        total_er = self.comm.allreduce(local_er, MPI.SUM)
        total_mr = self.comm.allreduce(local_mr, MPI.SUM)
        if count == 0:
            raise RuntimeError(f"Internal PML slab '{spec.ID}' has no distributed entrance cells.")
        return total_er / count, total_mr / count

    def _build_pmls(self) -> None:
        """Build native PMLs, then rank-local shards of global internal slabs."""

        global_specs = self.pmls["internal_specs"]
        self.pmls["internal_specs"] = []
        try:
            super()._build_pmls()
        finally:
            self.pmls["internal_specs"] = global_specs

        for global_spec in global_specs:
            averageer, averagemr = self._calculate_average_internal_pml_material_properties(
                global_spec
            )
            localised = self._localise_internal_pml_spec(global_spec)
            if localised is None:
                continue
            local_spec, offset, thickness, endpoint = localised
            pml = self._construct_local_internal_pml(
                global_spec,
                local_spec,
                offset,
                thickness,
                endpoint,
            )
            pml.calculate_update_coeffs(averageer, averagemr)
            self.pmls["slabs"].append(pml)

    def _terminate_pmls_with_pec(self) -> None:
        """Apply PML terminal PEC planes only on their owning global ranks."""

        pml_faces = [face for face, thickness in self.pmls["thickness"].items() if thickness]
        for spec in self.pmls["internal_specs"]:
            axis = "xyz".index(spec.maximum_face[0])
            lower, upper = self._internal_pml_bounds(spec)
            maximum_coordinate = (
                lower[axis] if spec.maximum_face.endswith("0") else upper[axis]
            )
            if maximum_coordinate in (0, self.global_size[axis]) and self.touches_global_face(
                spec.maximum_face
            ):
                pml_faces.append(spec.maximum_face)

        if not pml_faces:
            return
        pec_numid = next(material.numID for material in self.materials if material.ID == "pec")
        for face in pml_faces:
            self._force_pec_tangential_e(face, pec_numid)

    def _internal_pml_line_comm(self, axis: int) -> MPI.Cartcomm:
        return (self.x_line_comm, self.y_line_comm, self.z_line_comm)[axis]

    def _validate_internal_pml_material_extrusion(self, spec: InternalPMLSpec) -> None:
        """Collectively require a constant material cross-section along a slab."""

        lower, upper = self._internal_pml_bounds(spec)
        axis = "xyz".index(spec.maximum_face[0])
        owned_lower = self.lower_extent + self.negative_halo_offset.astype(np.int32)
        owned_upper = self.upper_extent
        overlap_lower = np.maximum(lower, owned_lower)
        overlap_upper = np.minimum(upper, owned_upper)
        transverse_axes = tuple(dimension for dimension in Dim if dimension != axis)
        cross_shape = tuple(
            max(0, int(overlap_upper[dimension] - overlap_lower[dimension]))
            for dimension in transverse_axes
        )

        local_min = np.full(cross_shape, np.iinfo(np.uint32).max, dtype=np.uint32)
        local_max = np.zeros(cross_shape, dtype=np.uint32)
        if (
            all(size > 0 for size in cross_shape)
            and overlap_lower[axis] < overlap_upper[axis]
        ):
            local_lower = overlap_lower - self.lower_extent
            local_upper = overlap_upper - self.lower_extent
            volume = self.solid[
                local_lower[Dim.X] : local_upper[Dim.X],
                local_lower[Dim.Y] : local_upper[Dim.Y],
                local_lower[Dim.Z] : local_upper[Dim.Z],
            ]
            local_min = np.min(volume, axis=axis)
            local_max = np.max(volume, axis=axis)

        mismatch = False
        if local_min.size:
            global_min = np.empty_like(local_min)
            global_max = np.empty_like(local_max)
            line_comm = self._internal_pml_line_comm(axis)
            line_comm.Allreduce(local_min, global_min, op=MPI.MIN)
            line_comm.Allreduce(local_max, global_max, op=MPI.MAX)
            mismatch = bool(np.any(global_min != global_max))

        if self.comm.allreduce(mismatch, op=MPI.LOR):
            raise ValueError(
                f"Internal PML slab '{spec.ID}' requires a material cross-section that is "
                "constant along its absorption direction."
            )

    def _distributed_pml_surface_is_pec(
        self,
        spec: InternalPMLSpec,
        normal: str,
        coordinate: int,
        pec_numids: np.ndarray,
    ) -> bool:
        """Collectively test the owned pieces of one global slab surface."""

        local_ok = True
        localised = self._localise_internal_pml_spec(spec)
        if localised is not None:
            local_spec = localised[0]
            dimension = "xyz".index(normal)
            local_coordinate = int(coordinate - self.lower_extent[dimension])
            owns_coordinate = 0 <= local_coordinate < self.size[dimension] or (
                local_coordinate == self.size[dimension]
                and not self.has_neighbour(Dim(dimension), Dir.POS)
            )
            if owns_coordinate:
                local_ok = self._pml_surface_is_pec(
                    local_spec,
                    normal,
                    local_coordinate,
                    pec_numids,
                )
        return bool(self.comm.allreduce(local_ok, op=MPI.LAND))

    def _validate_internal_pmls(self) -> None:
        """Collectively validate global internal slabs after local geometry build."""

        specs = self.pmls["internal_specs"]
        if not specs:
            return

        pec_numids = np.asarray(
            [material.numID for material in self.materials if material.is_pec],
            dtype=np.uint32,
        )
        limits = dict(zip("xyz", self.global_size))
        boundary_thickness = {
            face: self.comm.allreduce(thickness, op=MPI.MAX)
            for face, thickness in self.pmls["thickness"].items()
        }

        for index, spec in enumerate(specs):
            axis = spec.maximum_face[0]
            axis_index = "xyz".index(axis)
            lower_bounds, upper_bounds = self._internal_pml_bounds(spec)
            lower = lower_bounds[axis_index]
            upper = upper_bounds[axis_index]

            for other in specs[:index]:
                if other.maximum_face[0] == axis and self._pml_boxes_overlap(spec, other):
                    raise ValueError(
                        f"Internal PML slabs '{spec.ID}' and '{other.ID}' overlap along "
                        f"their common {axis}-axis absorption direction."
                    )

            for face in (f"{axis}0", f"{axis}max"):
                thickness = boundary_thickness[face]
                if not thickness:
                    continue
                native_lower = 0 if face.endswith("0") else limits[axis] - thickness
                native_upper = thickness if face.endswith("0") else limits[axis]
                if lower < native_upper and upper > native_lower:
                    raise ValueError(
                        f"Internal PML slab '{spec.ID}' overlaps the native {face} PML. "
                        f"Set that boundary's #pml_cells value to zero or move the slab."
                    )

            self._validate_internal_pml_material_extrusion(spec)

            lateral_faces = {
                "x": (("y", spec.ys), ("y", spec.yf), ("z", spec.zs), ("z", spec.zf)),
                "y": (("x", spec.xs), ("x", spec.xf), ("z", spec.zs), ("z", spec.zf)),
                "z": (("x", spec.xs), ("x", spec.xf), ("y", spec.ys), ("y", spec.yf)),
            }[axis]
            enclosure_warnings = []
            for normal, coordinate in lateral_faces:
                if coordinate in (0, limits[normal]):
                    continue
                if not self._distributed_pml_surface_is_pec(
                    spec, normal, coordinate, pec_numids
                ):
                    message = (
                        f"Internal PML slab '{spec.ID}' has an exposed transverse {normal}="
                        f"{coordinate} face. Enclose it with a continuous Yee-aligned PEC wall "
                        "or extend that face to the model boundary; exposed transverse "
                        "truncation can be numerically unstable."
                    )
                    enclosure_warnings.append(message)
                    logger.warning(message)

            maximum_coordinate = lower if spec.maximum_face.endswith("0") else upper
            maximum_on_boundary = maximum_coordinate in (0, limits[axis])
            if not maximum_on_boundary and not self._distributed_pml_surface_is_pec(
                spec, axis, maximum_coordinate, pec_numids
            ):
                message = (
                    f"Internal PML slab '{spec.ID}' must have a PEC backing on its "
                    f"maximum-stretch face ({spec.maximum_face}) or end at the model boundary; "
                    "open maximum-stretch terminations can be numerically unstable."
                )
                enclosure_warnings.append(message)
                logger.warning(message)

            transverse_full = {
                "x": spec.ys == 0
                and spec.yf == self.global_size[Dim.Y]
                and spec.zs == 0
                and spec.zf == self.global_size[Dim.Z],
                "y": spec.xs == 0
                and spec.xf == self.global_size[Dim.X]
                and spec.zs == 0
                and spec.zf == self.global_size[Dim.Z],
                "z": spec.xs == 0
                and spec.xf == self.global_size[Dim.X]
                and spec.ys == 0
                and spec.yf == self.global_size[Dim.Y],
            }[axis]
            classification = (
                "boundary-replacement"
                if maximum_on_boundary and transverse_full
                else "internal-absorber"
            )
            formulation, cfs = self._resolve_internal_pml_profile(spec)
            record = self.pmls["internal_registry"][spec.ID]
            record.update(
                classification=classification,
                formulation=formulation,
                order=len(cfs),
                enclosure_complete=not enclosure_warnings,
                enclosure_warnings=tuple(enclosure_warnings),
            )
            logger.info(
                f"Internal PML slab '{spec.ID}' validated as {classification} "
                f"({formulation}, order {len(cfs)})."
            )

    def _calculate_average_pml_material_properties(self, pml: MPIPML) -> Tuple[float, float]:
        """Calculate average material properties for the provided PML.

        Args:
            pml: PML to calculate the properties of.

        Returns:
            averageer, averagemr: Average permittivity and permeability
                in the PML slab.
        """
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

        # Cast from bool to int to avoid issue with numpy>2.3
        self.negative_halo_offset = self.negative_halo_offset.astype(int)

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
        """Set local properties and objects, then build the grid."""

        if any(self.global_size + 1 < self.mpi_tasks):
            logger.error(
                f"Too many MPI tasks requested ({self.mpi_tasks}) for grid of size"
                f" {self.global_size + 1}. Make sure the number of MPI tasks in each dimension is"
                " less than the size of the grid."
            )
            raise ValueError

        self.set_halo_map()

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

    def update_sources_and_recievers(self):
        """Update position of sources and receivers.

        If any sources or receivers have stepped outside of the local
        grid, they will be moved to the correct MPI rank.
        """
        super().update_sources_and_recievers()

        # Check it is possible for sources and receivers to have moved
        model_num = config.sim_config.current_model
        if (all(self.srcsteps == 0) and all(self.rxsteps == 0)) or model_num == 0:
            return

        # Get items that are outside the local bounds of the grid
        items_to_send = list(
            itertools.filterfalse(
                lambda x: self.within_bounds(x.coord),
                itertools.chain(
                    self.voltagesources,
                    self.hertziandipoles,
                    self.magneticdipoles,
                    self.transmissionlines,
                    self.rxs,
                ),
            )
        )

        # Map items being sent to the global coordinate space
        for item in items_to_send:
            item.coord = self.local_to_global_coordinate(item.coord)

        send_count_by_rank = np.zeros(self.comm.size, dtype=np.int32)

        # Send items to correct rank
        for rank, items in itertools.groupby(
            items_to_send, lambda x: self.get_rank_from_coordinate(x.coord)
        ):
            self.comm.isend(list(items), rank)
            send_count_by_rank[rank] += 1

        # Communicate the number of messages sent to each rank
        if self.is_coordinator():
            self.comm.Reduce(MPI.IN_PLACE, [send_count_by_rank, MPI.INT32_T], op=MPI.SUM)
        else:
            self.comm.Reduce([send_count_by_rank, MPI.INT32_T], None, op=MPI.SUM)

        # Get number of messages this rank will receive
        messages_to_receive = np.zeros(1, dtype=np.int32)
        if self.is_coordinator():
            self.comm.Scatter([send_count_by_rank, MPI.INT32_T], [messages_to_receive, MPI.INT32_T])
        else:
            self.comm.Scatter(None, [messages_to_receive, MPI.INT32_T])

        # Receive new items for this rank
        for _ in range(messages_to_receive[0]):
            new_items = self.comm.recv(None, MPI.ANY_SOURCE)
            for item in new_items:
                item.coord = self.global_to_local_coordinate(item.coord)
                if isinstance(item, Rx):
                    self.add_receiver(item)
                else:
                    self.add_source(item)

        # If this rank sent any items, remove them from our source and
        # receiver lists
        if len(items_to_send) > 0:
            # Map items sent back to the local coordinate space
            for item in items_to_send:
                item.coord = self.global_to_local_coordinate(item.coord)

            filter_items = lambda items: list(
                filter(lambda item: self.within_bounds(item.coord), items)
            )

            self.voltagesources = filter_items(self.voltagesources)
            self.hertziandipoles = filter_items(self.hertziandipoles)
            self.magneticdipoles = filter_items(self.magneticdipoles)
            self.transmissionlines = filter_items(self.transmissionlines)
            self.rxs = filter_items(self.rxs)

    def has_neighbour(self, dim: Dim, dir: Dir) -> bool:
        """Test if the current rank has a specified neighbour.

        Args:
            dim: Dimension of neighbour.
            dir: Direction of neighbour.
        Returns:
            has_neighbour: True if the current rank has a neighbour in
                the specified dimension and direction.
        """
        return self.neighbours[dim][dir] >= 0

    @staticmethod
    def _symmetry_face_dimension_direction(face: str) -> Tuple[Dim, Dir]:
        """Map a domain-face name to its Cartesian dimension and side."""

        dimensions = {"x": Dim.X, "y": Dim.Y, "z": Dim.Z}
        try:
            dimension = dimensions[face[0]]
        except (IndexError, KeyError) as exc:
            raise ValueError(f"Unknown symmetry boundary face '{face}'") from exc
        direction = Dir.NEG if face.endswith("0") else Dir.POS
        return dimension, direction

    def touches_global_face(self, face: str) -> bool:
        """Return whether this rank touches a named global domain face."""

        dimension, direction = self._symmetry_face_dimension_direction(face)
        return not self.has_neighbour(dimension, direction)

    def get_local_symmetry_boundaries(self) -> dict:
        """Return declared symmetry faces owned by this MPI rank.

        ``symmetry_boundaries`` remains the complete global declaration on
        every rank because source and NTFF validation use that semantic
        information. Only construction and per-iteration field dispatch use
        this rank-local subset.
        """

        return {
            face: boundary_type
            for face, boundary_type in self.symmetry_boundaries.items()
            if self.touches_global_face(face)
        }

    def set_halo_map(self):
        """Create MPI DataTypes for field array halo exchanges."""

        if self._halo_maps_initialised and not self._halo_maps_freed:
            self.free_halo_maps()
        self.send_halo_map.fill(MPI.DATATYPE_NULL)
        self.recv_halo_map.fill(MPI.DATATYPE_NULL)
        self._halo_maps_initialised = True
        self._halo_maps_freed = False
        size = (self.size + 1).tolist()
        field_mpi_dtype = mpi_datatype_for_dtype(config.sim_config.dtypes["float_or_double"])

        for dim in Dim:
            halo_size = (self.size + 1 - np.sum(self.neighbours >= 0, axis=1)).tolist()
            halo_size[dim] = 1
            start = [1 if self.has_neighbour(dim, Dir.NEG) else 0 for dim in Dim]

            if self.has_neighbour(dim, Dir.NEG):
                start[dim] = 1
                self.send_halo_map[dim][Dir.NEG] = field_mpi_dtype.Create_subarray(
                    size, halo_size, start
                )
                start[dim] = 0
                self.recv_halo_map[dim][Dir.NEG] = field_mpi_dtype.Create_subarray(
                    size, halo_size, start
                )

                self.send_halo_map[dim][Dir.NEG].Commit()
                self.recv_halo_map[dim][Dir.NEG].Commit()

            if self.has_neighbour(dim, Dir.POS):
                start[dim] = size[dim] - 2
                self.send_halo_map[dim][Dir.POS] = field_mpi_dtype.Create_subarray(
                    size, halo_size, start
                )
                start[dim] = size[dim] - 1
                self.recv_halo_map[dim][Dir.POS] = field_mpi_dtype.Create_subarray(
                    size, halo_size, start
                )

                self.send_halo_map[dim][Dir.POS].Commit()
                self.recv_halo_map[dim][Dir.POS].Commit()

    def calculate_local_extents(self):
        """Calculate size and extents of the local grid"""

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
            f"Global grid size: {self.global_size}, Local grid size: {self.size}, Lower extent:"
            f" {self.lower_extent}, Upper extent: {self.upper_extent}"
        )

    def within_bounds(self, local_point: npt.NDArray[np.int32]) -> bool:
        """Check a local point is within the grid.

        Args:
            local_point: Point to check.

        Returns:
            within_bounds: True if the point is within the local grid
                (i.e. this rank's grid). False otherwise.

        Raises:
            ValueError: Raised if the point is outside the global grid.
        """

        gx, gy, gz = self.local_to_global_coordinate(local_point)

        if gx < 0 or gx > self.gx:
            raise ValueError("x")
        if gy < 0 or gy > self.gy:
            raise ValueError("y")
        if gz < 0 or gz > self.gz:
            raise ValueError("z")

        return all(local_point >= self.negative_halo_offset) and all(local_point < self.size)

    def within_pml(self, local_point: npt.NDArray[np.int32]) -> bool:
        """Check if the provided point is within a PML.

        Args:
            local_point: Point to check. This must use this grid's
                coordinate system.

        Returns:
            within_pml: True if the point is within a PML.
        """
        if not (
            all(local_point >= self.negative_halo_offset) and all(local_point <= self.size)
        ):
            return False

        within_boundary_pml = (
            local_point[Dim.X] < self.pmls["thickness"]["x0"]
            or local_point[Dim.X] > self.nx - self.pmls["thickness"]["xmax"]
            or local_point[Dim.Y] < self.pmls["thickness"]["y0"]
            or local_point[Dim.Y] > self.ny - self.pmls["thickness"]["ymax"]
            or local_point[Dim.Z] < self.pmls["thickness"]["z0"]
            or local_point[Dim.Z] > self.nz - self.pmls["thickness"]["zmax"]
        )
        if within_boundary_pml:
            return True

        global_point = self.local_to_global_coordinate(local_point)
        return any(
            spec.xs <= global_point[Dim.X] <= spec.xf
            and spec.ys <= global_point[Dim.Y] <= spec.yf
            and spec.zs <= global_point[Dim.Z] <= spec.zf
            for spec in self.pmls["internal_specs"]
        )
