from typing import Generic

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from typing_extensions import TypeVar

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.materials import Material

GridType = TypeVar("GridType", bound=FDTDGrid, default="FDTDGrid")
GridViewType = TypeVar("GridViewType", bound="GridView[GridType]", default="GridView")


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
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for extent of the volume in cells.
            filename: string for filename.
        """

        self.start = np.array([xs, ys, zs], dtype=np.int32)
        self.stop = np.array([xf, yf, zf], dtype=np.int32)
        self.step = np.array([dx, dy, dz], dtype=np.int32)
        self.size = np.ceil((self.stop - self.start) / self.step).astype(np.int32)

        self.grid = grid

        self._ID = None

    # Properties for backwards compatibility
    @property
    def xs(self) -> int:
        return self.start[0]

    @property
    def ys(self) -> int:
        return self.start[1]

    @property
    def zs(self) -> int:
        return self.start[2]

    @property
    def xf(self) -> int:
        return self.stop[0]

    @property
    def yf(self) -> int:
        return self.stop[1]

    @property
    def zf(self) -> int:
        return self.stop[2]

    @property
    def dx(self) -> int:
        return self.step[0]

    @property
    def dy(self) -> int:
        return self.step[1]

    @property
    def dz(self) -> int:
        return self.step[2]

    @property
    def nx(self) -> int:
        return self.size[0]

    @property
    def ny(self) -> int:
        return self.size[1]

    @property
    def nz(self) -> int:
        return self.size[2]

    def get_slice(self, index: int, upper_bound_exclusive: bool = True):
        if upper_bound_exclusive:
            stop = self.stop[index]
        else:
            stop = self.stop[index] + self.step[index]

        return slice(self.start[index], stop, self.step[index])

    def slice_array(self, array: npt.NDArray, upper_bound_exclusive: bool = True):
        return np.ascontiguousarray(
            array[
                ...,
                self.get_slice(0, upper_bound_exclusive),
                self.get_slice(1, upper_bound_exclusive),
                self.get_slice(2, upper_bound_exclusive),
            ]
        )

    def initialise_materials(self):
        ID = self.get_ID(force_refresh=True)

        materials_in_grid_view = np.unique(ID)

        self.materials = np.array(self.grid.materials, dtype=Material)[materials_in_grid_view]
        self.materials.sort()

        # Create map from material ID to 0 - number of materials
        materials_map = {material.numID: index for index, material in enumerate(self.materials)}
        self.map_materials_func = np.vectorize(lambda id: materials_map[id])

    def map_to_view_materials(self, array: npt.NDArray):
        return self.map_materials_func(array)

    def get_ID(self, force_refresh=False):
        if self._ID is None or force_refresh:
            self._ID = self.slice_array(self.grid.ID, upper_bound_exclusive=False)
        return self._ID

    def get_solid(self):
        return self.slice_array(self.grid.solid)

    def get_rigidE(self):
        return self.slice_array(self.grid.rigidE)

    def get_rigidH(self):
        return self.slice_array(self.grid.rigidH)

    def get_Ex(self):
        return self.slice_array(self.grid.Ex, upper_bound_exclusive=False)

    def get_Ey(self):
        return self.slice_array(self.grid.Ey, upper_bound_exclusive=False)

    def get_Ez(self):
        return self.slice_array(self.grid.Ez, upper_bound_exclusive=False)

    def get_Hx(self):
        return self.slice_array(self.grid.Hx, upper_bound_exclusive=False)

    def get_Hy(self):
        return self.slice_array(self.grid.Hy, upper_bound_exclusive=False)

    def get_Hz(self):
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
        super().__init__(grid, xs, ys, zs, xf, yf, zf, dx, dy, dz)

        self.global_size = self.size

        # Calculate start for the local grid
        self.global_start = self.start
        self.start = self.grid.global_to_local_coordinate(self.start)

        # Bring start into the local grid (and not in the negative halo)
        # local_start must still be aligned with the provided step.
        self.start = np.where(
            self.start < self.grid.negative_halo_offset,
            self.grid.negative_halo_offset
            + ((self.start - self.grid.negative_halo_offset) % self.step),
            self.start,
        )

        # Calculate stop for the local grid
        self.stop = self.grid.global_to_local_coordinate(self.stop)

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

    @property
    def gx(self) -> int:
        return self.global_size[0]

    @property
    def gy(self) -> int:
        return self.global_size[1]

    @property
    def gz(self) -> int:
        return self.global_size[2]

    def get_slice(self, index: int, upper_bound_exclusive: bool = True):
        if upper_bound_exclusive:
            stop = self.stop[index]
        else:
            # Make slice of array one step larger if this rank does not
            # have a positive neighbour
            stop = np.where(
                self.has_positive_neighbour,
                self.stop,
                self.stop + self.step,
            )

        return slice(self.start[index], stop, self.step[index])

    def initialise_materials(self, comm: MPI.Comm):
        ID = self.get_ID(force_refresh=True)

        local_material_ids = np.unique(ID)
        local_materials = np.array(self.grid.materials, dtype=Material)[local_material_ids]

        self.materials, material_id_map = self.grid.remap_material_ids(
            local_materials.tolist(), comm
        )

        # Create map from material ID to 0 - number of materials
        self.map_materials_func = np.vectorize(lambda id: material_id_map(id))
