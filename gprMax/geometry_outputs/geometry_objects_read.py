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

from contextlib import AbstractContextManager
from os import PathLike
from types import TracebackType
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from gprMax.geometry_outputs.grid_view import GridView, MPIGridView
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid


class ReadGeometryObject(AbstractContextManager):
    def __init__(
        self,
        filename: PathLike,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        material_id_map: npt.NDArray[np.int32],
        invariant_axis: Optional[int] = None,
        target_invariant_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            material_id_map: array mapping a file-local material index (the
                integer values found in the file's /data and /ID arrays) to
                the numID that material has in `grid`. Built by matching
                materials by ID name rather than assuming a fixed count/
                order of built-in materials - see GeometryObjectsRead.build().
            invariant_axis: 0, 1 or 2 for x, y or z if `grid` is in an
                active 2D TM/TE mode, else None. A 2D model's geometry has
                the same physical intention regardless of whether it was
                exported from a 1-cell TM reduction or a 2-cell TE one, so
                if the file's own thickness on this axis doesn't match
                `target_invariant_size`, it is broadcast (1 -> N) or
                reduced to its canonical layer (N -> 1) automatically - see
                _resize_cell_axis()/_resize_edge_axis().
            target_invariant_size: 1 (TM) or 2 (TE), required if
                `invariant_axis` is given.
        """
        self.file_handler = h5py.File(filename)

        data = self.file_handler["/data"]
        assert isinstance(data, h5py.Dataset)
        file_shape = np.array(data.shape, dtype=np.int32)
        stop = start + file_shape

        self.invariant_axis = invariant_axis
        self.file_invariant_size = (
            int(file_shape[invariant_axis]) if invariant_axis is not None else None
        )
        self.target_invariant_size = target_invariant_size

        if (
            invariant_axis is not None
            and self.file_invariant_size != target_invariant_size
        ):
            # Decouple the *write* region's size on this axis from the
            # file's own (read) size - the actual resizing of the read
            # arrays happens in _resize_cell_axis()/_resize_edge_axis(),
            # called from each read_*()/get_data() method below.
            stop = stop.copy()
            stop[invariant_axis] = start[invariant_axis] + target_invariant_size

        if isinstance(grid, MPIGrid):
            if grid.local_bounds_overlap_grid(start, stop):
                self.grid_view = MPIGridView(
                    grid, start[0], start[1], start[2], stop[0], stop[1], stop[2]
                )
            else:
                # The MPIGridView will create a new communicator using
                # MPI_Split. Calling this here prevents deadlock if not
                # all ranks need to read the geometry object.
                grid.comm.Split(MPI.UNDEFINED)
                self.grid_view = None

        else:
            self.grid_view = GridView(grid, start[0], start[1], start[2], stop[0], stop[1], stop[2])

        self.material_id_map = material_id_map

    def _resize_cell_axis(self, array: npt.NDArray, spatial_axis_offset: int) -> npt.NDArray:
        """Broadcasts (1 -> N) or reduces (N -> 1, taking the first layer)
        `array` along the invariant axis to match `self.target_invariant_size`,
        for a cell-based array (solid, rigidE, rigidH - sized nx/ny/nz, no
        +1 edge padding). No-op if there's no mismatch (including the pure
        3D case, `self.invariant_axis is None`).

        Args:
            array: array to resize, whose spatial dims start at
                `spatial_axis_offset` (0 for solid, 1 for rigidE/rigidH,
                which have a leading component axis).
        """
        if self.invariant_axis is None or self.file_invariant_size == self.target_invariant_size:
            return array

        axis = self.invariant_axis + spatial_axis_offset
        if self.file_invariant_size == 1:
            reps = [1] * array.ndim
            reps[axis] = self.target_invariant_size
            return np.tile(array, reps)
        else:
            # file_invariant_size > 1, target == 1: every 2D-mode geometry
            # command in this codebase already keeps the invariant axis's
            # cells identical (see FractalVolume/FractalSurface/AddGrass),
            # so any one layer is an equally valid canonical choice - use
            # the first.
            return np.take(array, [0], axis=axis)

    def _resize_edge_axis(self, array: npt.NDArray, spatial_axis_offset: int) -> npt.NDArray:
        """Same as _resize_cell_axis(), but for the ID array, which is
        edge-based (sized nx+1/ny+1/nz+1) - the invariant axis has 2 edges
        for TM (0, 1 - both equally valid, no wall/interior distinction for
        a 1-cell reduction) and 3 for TE (0, 1, 2 - only the interior edge,
        index 1, is genuinely live; 0 and 2 are outer walls forced pec/pmc
        afterwards by tex()/tey()/tez(), regardless of what's read here).
        The canonical edge is therefore TM's edge 0 or TE's edge 1 -
        whichever the file has - broadcast/reduced to fill the target's
        edge count.
        """
        if self.invariant_axis is None or self.file_invariant_size == self.target_invariant_size:
            return array

        axis = self.invariant_axis + spatial_axis_offset
        canonical_edge = 1 if self.file_invariant_size == 2 else 0
        canonical = np.take(array, [canonical_edge], axis=axis)
        reps = [1] * array.ndim
        reps[axis] = self.target_invariant_size + 1
        return np.tile(canonical, reps)

    def _remap(
        self, data: npt.NDArray[np.int16], existing: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.int32]:
        """Maps file-local material indices in `data` to numIDs in the
        target grid, via `self.material_id_map`. A value of -1 means "don't
        build anything here, leave whatever's already in the grid" (per the
        #geometry_objects_read documentation) - i.e. it takes its value
        from `existing` (the grid's current content at that location)
        rather than the material map, since -1 isn't a valid index into it
        and, in a model with prior geometry (e.g. a fractal soil built
        before an imported target), isn't the same thing as free_space.
        """
        safe_indices = np.where(data < 0, 0, data)
        mapped = self.material_id_map[safe_indices]
        return np.where(data < 0, existing, mapped)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Close the file when the context is exited.

        The parameters describe the exception that caused the context to
        be exited. If the context was exited without an exception, all
        three arguments will be None. Any exception will be
        processed normally upon exit from this method.

        Returns:
            suppress_exception (optional): Returns True if the exception
                should be suppressed (i.e. not propagated). Otherwise,
                the exception will be processed normally upon exit from
                this method.
        """
        self.close()

    def close(self) -> None:
        """Close the file handler"""
        self.file_handler.close()

    def has_valid_discritisation(self) -> bool:
        if self.grid_view is None:
            return True

        dx_dy_dz = self.file_handler.attrs["dx_dy_dz"]
        return not isinstance(dx_dy_dz, h5py.Empty) and all(dx_dy_dz == self.grid_view.grid.dl)

    def has_ID_array(self) -> bool:
        ID_class = self.file_handler.get("ID", getclass=True)
        return ID_class == h5py.Dataset

    def has_rigid_arrays(self) -> bool:
        rigidE_class = self.file_handler.get("rigidE", getclass=True)
        rigidH_class = self.file_handler.get("rigidH", getclass=True)
        return rigidE_class == h5py.Dataset and rigidH_class == h5py.Dataset

    def read_data(self):
        if self.grid_view is None:
            return

        data = self.file_handler["/data"]
        assert isinstance(data, h5py.Dataset)
        # A full, native-shape read - there is no partial-region reading
        # support in this class (the read region always equals the file's
        # own extent), so this is equivalent to the previous grid-view-
        # sliced read whenever there's no invariant-axis mismatch, and
        # correctly generalises when there is one (_resize_cell_axis()
        # below then adapts it to the target's own size).
        data = data[:]
        data = self._resize_cell_axis(data, spatial_axis_offset=0)

        # Should be int16 to allow for -1 which indicates background, i.e.
        # don't build anything, but AustinMan/Woman maybe uint16
        if data.dtype != "int16":
            data = data.astype("int16")

        existing = self.grid_view.get_solid()
        self.grid_view.set_solid(self._remap(data, existing))

    def get_data(self) -> Optional[npt.NDArray[np.int16]]:
        """Returns the file's material-index array with valid (>=0) entries
        already remapped to numIDs in the target grid. -1 is left as -1
        (rather than substituted, as read_data()/read_ID() do via _remap()),
        since the caller (build_voxels_from_array) already implements "-1
        means leave this cell alone" itself by skipping negative values.
        """
        if self.grid_view is None:
            return None

        data = self.file_handler["/data"]
        assert isinstance(data, h5py.Dataset)
        data = data[:]
        data = self._resize_cell_axis(data, spatial_axis_offset=0)

        # Should be int16 to allow for -1 which indicates background, i.e.
        # don't build anything, but AustinMan/Woman maybe uint16
        if data.dtype != "int16":
            data = data.astype("int16")

        safe_indices = np.where(data < 0, 0, data)
        mapped = self.material_id_map[safe_indices].astype(data.dtype)
        return np.where(data < 0, data, mapped)

    def read_rigidE(self):
        if self.grid_view is None:
            return

        rigidE = self.file_handler["/rigidE"]
        assert isinstance(rigidE, h5py.Dataset)

        rigidE = self._resize_cell_axis(rigidE[:], spatial_axis_offset=1)
        self.grid_view.set_rigidE(rigidE)

    def read_rigidH(self):
        if self.grid_view is None:
            return

        rigidH = self.file_handler["/rigidH"]
        assert isinstance(rigidH, h5py.Dataset)

        rigidH = self._resize_cell_axis(rigidH[:], spatial_axis_offset=1)
        self.grid_view.set_rigidH(rigidH)

    def read_ID(self):
        if self.grid_view is None:
            return

        ID = self.file_handler["/ID"]
        assert isinstance(ID, h5py.Dataset)

        data = self._resize_edge_axis(ID[:], spatial_axis_offset=1)
        existing = self.grid_view.get_ID(force_refresh=True)
        self.grid_view.set_ID(self._remap(data, existing))
