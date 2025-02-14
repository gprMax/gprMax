import logging
from itertools import chain
from typing import Generic

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from typing_extensions import TypeVar

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.materials import Material

logger = logging.getLogger(__name__)

GridType = TypeVar("GridType", bound=FDTDGrid)


class GridView(Generic[GridType]):
    def __init__(
        self,
        grid: GridType,
        xs: int,
        ys: int,
        zs: int,
        xf: int,
        yf: int,
        zf: int,
        dx: int = 1,
        dy: int = 1,
        dz: int = 1,
    ):
        """Create a new GridView.

        A grid view provides an interface to allow easy access to a
        subsection of an FDTDGrid.

        Args:
            grid: Grid to create a view of.
            xs: Start x coordinate of the grid view.
            ys: Start y coordinate of the grid view.
            zs: Start z coordinate of the grid view.
            xf: End x coordinate of the grid view.
            yf: End y coordinate of the grid view.
            zf: End z coordinate of the grid view.
            dx: Optional step size of the grid view in the x dimension. Defaults to 1.
            dy: Optional step size of the grid view in the y dimension. Defaults to 1.
            dz: Optional step size of the grid view in the z dimension. Defaults to 1.
        """

        self.start = np.array([xs, ys, zs], dtype=np.int32)
        self.stop = np.array([xf, yf, zf], dtype=np.int32)
        self.step = np.array([dx, dy, dz], dtype=np.int32)
        self.size = np.ceil((self.stop - self.start) / self.step).astype(np.int32)

        self.grid = grid

        self._ID = None

        logger.debug(
            f"Created GridView for grid '{self.grid.name}' (start={self.start}, stop={self.stop},"
            f" step={self.step}, size={self.size})"
        )

    def get_slice(self, dimension: int, upper_bound_exclusive: bool = True) -> slice:
        """Create a slice object for the specified dimension.

        Args:
            dimension: Dimension to create the slice object for. Values
                0, 1, and 2 map to the x, y, and z dimensions
                respectively.
            upper_bound_exclusive: Optionally specify if the upper bound
                of the slice should be exclusive or inclusive. Defaults
                to True.

        Returns:
            slice: Slice object
        """
        if upper_bound_exclusive:
            stop = self.stop[dimension]
        else:
            stop = self.stop[dimension] + self.step[dimension]

        return slice(self.start[dimension], stop, self.step[dimension])

    def slice_array(self, array: npt.NDArray, upper_bound_exclusive: bool = True) -> npt.NDArray:
        """Slice an array according to the dimensions of the grid view.

        It is assumed the last 3 dimensions of the provided array
        represent the x, y, z spacial information. Other dimensions will
        not be sliced.

        E.g. For an array of shape (10, 100, 50, 50) this function would
        return an array of shape (10, x, y, z) where x, y, and z are
        specified by the size/shape of the grid view.

        Args:
            array: Array to slice. Must have at least 3 dimensions.
            upper_bound_exclusive: Optionally specify if the upper bound
                of the slice should be exclusive or inclusive. Defaults
                to True.

        Returns:
            array: Sliced array
        """
        return np.ascontiguousarray(
            array[
                ...,
                self.get_slice(0, upper_bound_exclusive),
                self.get_slice(1, upper_bound_exclusive),
                self.get_slice(2, upper_bound_exclusive),
            ]
        )

    def initialise_materials(self):
        """Create a new ID map for materials in the grid view.

        Rather than using the default material IDs (as per the main grid
        object), we may want to create a new index for materials inside
        this grid view. Unlike using the default material IDs, this new
        index will be continuous from 0 - number of materials for the
        materials in the grid view.

        This function should be called before calling the
        map_to_view_materials() function.
        """
        # Get unique materials in the grid view
        ID = self.get_ID(force_refresh=True)
        materials_in_grid_view = np.unique(ID)

        # Get actual Material objects and sort
        self.materials = np.array(self.grid.materials, dtype=Material)[materials_in_grid_view]
        self.materials.sort()

        # Create map from material ID to 0 - number of materials
        materials_map = {material.numID: index for index, material in enumerate(self.materials)}
        self.map_materials_func = np.vectorize(lambda id: materials_map[id])

    NDArrayType = TypeVar("NDArrayType", bound=npt.NDArray)

    def map_to_view_materials(self, array: NDArrayType) -> NDArrayType:
        """Map from the main grid material IDs to the grid view IDs.

        Ensure initialise_materials() has been called before using this
        function.

        Args:
            array: Array to map.

        Returns:
            array: Mapped array.
        """
        return self.map_materials_func(array)

    def get_ID(self, force_refresh=False) -> npt.NDArray[np.uint32]:
        """Get a view of the ID array.

        By default, the slice of the ID array is cached to prevent
        unnecessary reconstruction of the view on repeat calls. E.g.
        from the initialise_materials() function as well as a user call
        to get_ID().

        Args:
            force_refresh: Optionally force reloading the ID array from
                the main grid object. Defaults to False.

        Returns:
            ID: View of the ID array.
        """
        if self._ID is None or force_refresh:
            self._ID = self.slice_array(self.grid.ID, upper_bound_exclusive=False)
        return self._ID

    def get_solid(self) -> npt.NDArray[np.uint32]:
        """Get a view of the solid array.

        Returns:
            solid: View of the solid array
        """
        return self.slice_array(self.grid.solid)

    def get_rigidE(self) -> npt.NDArray[np.int8]:
        """Get a view of the rigidE array.

        Returns:
            rigidE: View of the rigidE array
        """
        return self.slice_array(self.grid.rigidE)

    def get_rigidH(self) -> npt.NDArray[np.int8]:
        """Get a view of the rigidH array.

        Returns:
            rigidH: View of the rigidH array
        """
        return self.slice_array(self.grid.rigidH)

    def get_Ex(self) -> npt.NDArray[np.float32]:
        """Get a view of the Ex array.

        Returns:
            Ex: View of the Ex array
        """
        return self.slice_array(self.grid.Ex, upper_bound_exclusive=False)

    def get_Ey(self) -> npt.NDArray[np.float32]:
        """Get a view of the Ey array.

        Returns:
            Ey: View of the Ey array
        """
        return self.slice_array(self.grid.Ey, upper_bound_exclusive=False)

    def get_Ez(self) -> npt.NDArray[np.float32]:
        """Get a view of the Ez array.

        Returns:
            Ez: View of the Ez array
        """
        return self.slice_array(self.grid.Ez, upper_bound_exclusive=False)

    def get_Hx(self) -> npt.NDArray[np.float32]:
        """Get a view of the Hx array.

        Returns:
            Hx: View of the Hx array
        """
        return self.slice_array(self.grid.Hx, upper_bound_exclusive=False)

    def get_Hy(self) -> npt.NDArray[np.float32]:
        """Get a view of the Hy array.

        Returns:
            Hy: View of the Hy array
        """
        return self.slice_array(self.grid.Hy, upper_bound_exclusive=False)

    def get_Hz(self) -> npt.NDArray[np.float32]:
        """Get a view of the Hz array.

        Returns:
            Hz: View of the Hz array
        """
        return self.slice_array(self.grid.Hz, upper_bound_exclusive=False)


class MPIGridView(GridView[MPIGrid]):
    def __init__(
        self,
        grid: MPIGrid,
        xs: int,
        ys: int,
        zs: int,
        xf: int,
        yf: int,
        zf: int,
        dx: int = 1,
        dy: int = 1,
        dz: int = 1,
    ):
        """Create a new MPIGridView.

        An MPI grid view provides an interface to allow easy access to a
        subsection of an MPIGrid.

        Args:
            grid: MPI grid to create a view of.
            xs: Start x coordinate of the grid view.
            ys: Start y coordinate of the grid view.
            zs: Start z coordinate of the grid view.
            xf: End x coordinate of the grid view.
            yf: End y coordinate of the grid view.
            zf: End z coordinate of the grid view.
            dx: Optional step size of the grid view in the x dimension. Defaults to 1.
            dy: Optional step size of the grid view in the y dimension. Defaults to 1.
            dz: Optional step size of the grid view in the z dimension. Defaults to 1.
        """
        super().__init__(grid, xs, ys, zs, xf, yf, zf, dx, dy, dz)

        self.global_size = self.size

        # Calculate start for the local grid
        self.global_start = self.grid.local_to_global_coordinate(self.start)

        self.has_negative_neighbour = self.start < self.grid.negative_halo_offset

        # Bring start into the local grid (and not in the negative halo)
        # start must still be aligned with the provided step.
        self.start = np.where(
            self.has_negative_neighbour,
            self.grid.negative_halo_offset
            + ((self.start - self.grid.negative_halo_offset) % self.step),
            self.start,
        )

        # Calculate stop for the local grid
        self.global_stop = self.grid.local_to_global_coordinate(self.stop)

        self.has_positive_neighbour = self.stop > self.grid.size

        # Limit stop such that it is at most one step beyond the max
        # index of the grid. As stop is the upper bound, it is
        # exclusive, meaning when used to slice an array (with the
        # provided step), the last element accessed will one step below
        # stop.
        # Note: using self.grid.size as an index in any dimension would
        # fall in the positive halo (this counts as outside the local
        # grid).
        self.stop = np.where(
            self.has_positive_neighbour,
            self.grid.size + ((self.stop - self.grid.size) % self.step),
            self.stop,
        )

        # Calculate offset for the local grid view
        self.offset = self.grid.local_to_global_coordinate(self.start) - self.global_start

        # Update local size
        self.size = self.stop - self.start

        logger.debug(
            f"Created MPIGridView for grid '{self.grid.name}' (global_start={self.global_start},"
            f" global_stop={self.global_stop}, global_size={self.global_size}, start={self.start},"
            f" stop={self.stop}, step={self.step}, size={self.size}, offset={self.offset})"
        )

    def get_slice(self, dimension: int, upper_bound_exclusive: bool = True) -> slice:
        """Create a slice object for the specified dimension.

        Args:
            dimension: Dimension to create the slice object for. Values
                0, 1, and 2 map to the x, y, and z dimensions
                respectively.
            upper_bound_exclusive: Optionally specify if the upper bound
                of the slice should be exclusive or inclusive. Defaults
                to True.

        Returns:
            slice: Slice object
        """
        if upper_bound_exclusive or self.has_positive_neighbour[dimension]:
            stop = self.stop[dimension]
        else:
            # Make slice of array one step larger if this rank does not
            # have a positive neighbour
            stop = self.stop[dimension] + self.step[dimension]

        return slice(self.start[dimension], stop, self.step[dimension])

    def get_output_slice(self, dimension: int, upper_bound_exclusive: bool = True) -> slice:
        if upper_bound_exclusive or self.has_positive_neighbour[dimension]:
            size = self.size[dimension]
        else:
            # Make slice of array one step larger if this rank does not
            # have a positive neighbour
            size = self.size[dimension] + 1

        offset = self.offset[dimension]

        return slice(offset, offset + size)

    def initialise_materials(self, comm: MPI.Cartcomm):
        """Create a new ID map for materials in the grid view.

        Rather than using the default material IDs (as per the main grid
        object), we may want to create a new index for materials inside
        this grid view. Unlike using the default material IDs, this new
        index will be continuous from 0 - number of materials for the
        materials in the grid view.

        This function should only be called if required as it needs MPI
        communication to construct the new map. It should also be called
        before the map_to_view_materials() function.
        """
        ID = self.get_ID(force_refresh=True)

        local_material_ids = np.unique(ID)
        local_materials = np.array(self.grid.materials, dtype=Material)[local_material_ids]
        local_materials.sort()
        local_material_ids = [m.numID for m in local_materials]

        # Send all materials to the coordinating rank
        materials_by_rank = comm.gather(local_materials, root=0)

        if materials_by_rank is not None:
            # Filter out duplicate materials and sort by material ID
            all_materials = np.fromiter(chain.from_iterable(materials_by_rank), dtype=Material)
            self.materials = np.unique(all_materials)

            # The new material IDs corespond to each material's index in
            # the sorted self.materials array. For each rank, get the
            # new IDs of each material it sent to send back
            for rank in range(1, comm.size):
                new_material_ids = np.where(np.isin(self.materials, materials_by_rank[rank]))[0]
                comm.Isend([new_material_ids.astype(np.int32), MPI.INT], rank)

            new_material_ids = np.where(np.isin(self.materials, materials_by_rank[0]))[0]
            new_material_ids = new_material_ids.astype(np.int32)
        else:
            self.materials = None

            # Get list of global IDs for this rank's local materials
            new_material_ids = np.empty(len(local_materials), dtype=np.int32)
            comm.Recv([new_material_ids, MPI.INT], 0)

        # Create map from local material ID to global material ID
        materials_map = {
            local_material_ids[index]: new_id for index, new_id in enumerate(new_material_ids)
        }

        # Create map from material ID to 0 - number of materials
        self.map_materials_func = np.vectorize(lambda id: materials_map[id])
