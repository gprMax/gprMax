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

from contextlib import AbstractContextManager
from os import PathLike
from types import TracebackType
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt

from gprMax.geometry_outputs.grid_view import GridView, MPIGridView
from gprMax.geometry_tags import UNTAGGED_NAME, validate_geometry_tag, validate_geometry_tag_ids
from gprMax.grid.fdtd_grid import FDTDGrid


def read_geometry_tag_names(geometry: h5py.File) -> tuple[str, ...]:
    """Validate tag metadata before map allocation or rank-local slicing.

    Legacy untagged files omit both datasets. A tagged file must pair a
    cell-shaped integer array with a one-dimensional, unique name catalogue.
    This only reads the small catalogue; ID bounds are checked on each local
    tag slice before remapping. Invariant-axis resizing happens afterwards
    and must not disguise a malformed on-disk shape by broadcasting it.
    """
    has_data = "tag_data" in geometry
    has_names = "tag_names" in geometry
    if not has_data and not has_names:
        return ()
    if not has_data or not has_names:
        raise ValueError("Geometry tag metadata requires both /tag_data and /tag_names")
    data, names = geometry["tag_data"], geometry["tag_names"]
    if not isinstance(data, h5py.Dataset) or not isinstance(names, h5py.Dataset):
        raise ValueError("Geometry tag metadata must contain datasets")
    if data.ndim != 3 or data.shape != geometry["data"].shape:
        raise ValueError("Geometry /tag_data must have the same cell shape as /data")
    if data.dtype.kind not in "iu":
        raise ValueError("Geometry tag IDs must be integers")
    if names.ndim != 1 or h5py.check_string_dtype(names.dtype) is None:
        raise ValueError("Geometry /tag_names must be a one-dimensional string catalogue")
    catalogue = tuple(names.asstr(encoding="utf-8")[:])
    if not catalogue or catalogue[0] != UNTAGGED_NAME:
        raise ValueError("Geometry /tag_names must start with 'untagged' at ID 0")
    if len(set(catalogue)) != len(catalogue):
        raise ValueError("Geometry /tag_names must contain unique names")
    for name in catalogue[1:]:
        validate_geometry_tag(name)
    return catalogue


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
        material_id_map = np.asarray(material_id_map)
        if material_id_map.ndim != 1 or material_id_map.dtype.kind not in "iu":
            raise ValueError("Geometry material ID map must be a one-dimensional integer array")
        if material_id_map.size and (
            int(material_id_map.min()) < 0 or int(material_id_map.max()) > np.iinfo(np.int32).max
        ):
            raise ValueError("Geometry material ID map entries must fit non-negative int32 IDs")
        self.material_id_map = material_id_map.astype(np.int32, copy=False)

        self.file_handler = h5py.File(filename)
        try:
            self.tag_names = read_geometry_tag_names(self.file_handler)
        except Exception:
            self.file_handler.close()
            raise

        data = self.file_handler["/data"]
        assert isinstance(data, h5py.Dataset)
        file_shape = np.array(data.shape, dtype=np.int32)
        stop = start + file_shape

        self.invariant_axis = invariant_axis
        self.file_invariant_size = (
            int(file_shape[invariant_axis]) if invariant_axis is not None else None
        )
        self.target_invariant_size = target_invariant_size

        if invariant_axis is not None and self.file_invariant_size != target_invariant_size:
            # Decouple the *write* region's size on this axis from the
            # file's own (read) size - the actual resizing of the read
            # arrays happens in _resize_cell_axis()/_resize_edge_axis(),
            # called from each read_*()/get_data() method below.
            stop = stop.copy()
            stop[invariant_axis] = start[invariant_axis] + target_invariant_size

        if getattr(grid, "is_distributed", False) is True:
            if grid.local_bounds_overlap_grid(start, stop):
                self.grid_view = MPIGridView(
                    grid, start[0], start[1], start[2], stop[0], stop[1], stop[2]
                )
            else:
                from gprMax.mpi_support import require_mpi

                # The MPIGridView will create a new communicator using
                # MPI_Split. Calling this here prevents deadlock if not
                # all ranks need to read the geometry object.
                MPI = require_mpi("distributed geometry-object input")
                grid.comm.Split(MPI.UNDEFINED)
                self.grid_view = None

        else:
            self.grid_view = GridView(grid, start[0], start[1], start[2], stop[0], stop[1], stop[2])

    def _resize_cell_axis(self, array: npt.NDArray, spatial_axis_offset: int) -> npt.NDArray:
        """Broadcasts (1 -> N) or reduces (N -> target, taking the first layer)
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

        # file_invariant_size > 1: geometry commands in this codebase
        # keep the invariant-axis cells identical (see
        # FractalVolume/FractalSurface/AddGrass), so any layer is a
        # valid canonical choice. Use the first layer and repeat it
        # when the target domain requires two cells for TE mode.
        canonical = np.take(array, [0], axis=axis)
        reps = [1] * array.ndim
        reps[axis] = self.target_invariant_size
        return np.tile(canonical, reps)

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

    def _check_material_coverage(self, data: npt.NDArray) -> None:
        """Raises a clear error if `data` references a file-local material
        index this file's materials file never declared, rather than
        letting numpy fancy-indexing fail with a bare IndexError. This
        happens whenever the materials file supplied to
        #geometry_objects_read omits a material that was actually present
        (and so given an index) when the geometry file was originally
        written - most commonly the implicit background material (e.g.
        free_space) of the written region, if the user only listed the
        material(s) they specifically cared about.

        Validation precedes any signed conversion: an unsigned positive code
        is still a file-local index, never an encoded negative sentinel.
        Only -1 is transparent; every non-negative index must address the map.
        """
        if data.dtype.kind not in "iu":
            raise ValueError("Geometry material indices must be integers")
        if data.size and int(data.min()) < -1:
            raise ValueError("Geometry material indices must be -1 (transparent) or non-negative")
        max_index = int(data.max()) if data.size else -1
        n_declared = len(self.material_id_map)
        if max_index >= n_declared:
            raise ValueError(
                f"'{self.file_handler.filename}' references material index "
                f"{max_index}, but the accompanying materials file only "
                f"declares {n_declared} "
                "material(s). The materials file must list every material "
                "present in the written region (including any implicit "
                "background material, e.g. free_space) in the same order "
                "they appeared when the geometry object file was written, "
                "not just the ones of interest."
            )

    def _remap(self, data: npt.NDArray, existing: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        """Maps file-local material indices in `data` to numIDs in the
        target grid, via `self.material_id_map`. A value of -1 means "don't
        build anything here, leave whatever's already in the grid" (per the
        #geometry_objects_read documentation) - i.e. it takes its value
        from `existing` (the grid's current content at that location)
        rather than the material map, since -1 isn't a valid index into it
        and, in a model with prior geometry (e.g. a fractal soil built
        before an imported target), isn't the same thing as free_space.
        """
        self._check_material_coverage(data)
        present = data >= 0
        result = existing.copy()
        result[present] = self.material_id_map[data[present]]
        return result

    def _read_spatial_dataset(
        self,
        dataset: h5py.Dataset,
        *,
        component_axis: bool = False,
        edge_based: bool = False,
    ) -> npt.NDArray:
        """Read exactly the portion of a geometry dataset owned by this rank.

        MPI grid arrays include the negative interface halo when values are
        assigned, so ``get_3d_read_slice`` is deliberately used rather than
        the non-overlapping output slice. For 2D TM/TE conversion the array
        must first be resized globally, after which the same rank-local slice
        is applied.
        """
        mismatch = (
            self.invariant_axis is not None
            and self.file_invariant_size != self.target_invariant_size
        )
        if mismatch:
            array = dataset[:]
            if edge_based:
                array = self._resize_edge_axis(array, int(component_axis))
            else:
                array = self._resize_cell_axis(array, int(component_axis))
            if isinstance(self.grid_view, MPIGridView):
                spatial = self.grid_view.get_3d_read_slice(upper_bound_exclusive=not edge_based)
                return np.ascontiguousarray(array[(..., *spatial)])
            return array

        if isinstance(self.grid_view, MPIGridView):
            spatial = self.grid_view.get_3d_read_slice(upper_bound_exclusive=not edge_based)
            return np.ascontiguousarray(dataset[(..., *spatial)])
        return dataset[:]

    def _get_assignment_region(
        self, array: npt.NDArray, *, edge_based: bool = False
    ) -> npt.NDArray:
        """Return existing values for the exact region set by ``GridView``.

        This differs from ``get_solid``/``get_ID`` on a non-leading MPI rank:
        setters include its negative interface halo, which is also present in
        the geometry file read slice.
        """
        assert self.grid_view is not None
        spatial = tuple(
            self.grid_view.setter_slice(axis, upper_bound_exclusive=not edge_based)
            for axis in range(3)
        )
        return np.ascontiguousarray(array[(..., *spatial)])

    def get_local_data_start(self) -> Optional[npt.NDArray[np.int32]]:
        """Return the local cell coordinate corresponding to ``get_data()[0,0,0]``.

        Legacy/external geometry files without rigid and component-ID arrays
        are rebuilt voxel by voxel. On MPI ranks, the returned data may begin
        in a negative interface halo rather than at the object's original
        local start, so the builder must use this matching coordinate.
        """
        if self.grid_view is None:
            return None
        return np.array(
            [self.grid_view.setter_slice(axis).start for axis in range(3)],
            dtype=np.int32,
        )

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

    def has_tag_data(self) -> bool:
        return bool(self.tag_names)

    def read_tags(self) -> None:
        """Import or clear semantic tags wherever the geometry writes cells."""

        if self.grid_view is None:
            return
        if self.grid_view.grid.geometry_tag_map is None:
            # An all-untagged catalogue does not allocate a map, but its
            # on-disk labels still must be valid IDs rather than sentinels.
            if self.has_tag_data():
                validate_geometry_tag_ids(
                    self._read_spatial_dataset(self.file_handler["/tag_data"]),
                    len(self.tag_names),
                )
            return

        raw_data = self.file_handler["/data"]
        assert isinstance(raw_data, h5py.Dataset)
        raw_data = self._read_spatial_dataset(raw_data)
        # MPI setters include the negative interface halo, matching the
        # rank-local read slice above. ``get_geometry_tags()`` deliberately
        # excludes that halo because it is intended for output, so use the
        # exact assignment region here just as ``read_data()`` does.
        existing = self._get_assignment_region(self.grid_view.grid.geometry_tag_map.data)

        if self.has_tag_data():
            tag_data = self.file_handler["/tag_data"]
            assert isinstance(tag_data, h5py.Dataset)
            tag_data = self._read_spatial_dataset(tag_data)
            imported = self.grid_view.grid.geometry_tag_map.remap_file_ids(tag_data, self.tag_names)
        else:
            imported = np.zeros(raw_data.shape, dtype=existing.dtype)

        self.grid_view.set_geometry_tags(np.where(raw_data < 0, existing, imported))

    def read_data(self):
        if self.grid_view is None:
            return

        data = self.file_handler["/data"]
        assert isinstance(data, h5py.Dataset)
        data = self._read_spatial_dataset(data)

        existing = self._get_assignment_region(self.grid_view.grid.solid)
        self.grid_view.set_solid(self._remap(data, existing))

    def get_data(self) -> Optional[npt.NDArray[np.int32]]:
        """Returns the file's material-index array with valid (>=0) entries
        already remapped to numIDs in the target grid. -1 is left as -1
        (rather than substituted, as read_data()/read_ID() do via _remap()),
        since the caller (build_voxels_from_array) already implements "-1
        means leave this cell alone" itself by skipping negative values.

        The returned signed int32 array may start in a rank's negative halo;
        pair it with ``get_local_data_start()``, not the original placement
        coordinate. Entries are already target-grid material IDs, so the voxel
        builder's additional material offset must be zero for this path.
        """
        if self.grid_view is None:
            return None

        data = self.file_handler["/data"]
        assert isinstance(data, h5py.Dataset)
        data = self._read_spatial_dataset(data)

        self._check_material_coverage(data)
        # The on-disk indices are compact, but the target catalogue may
        # already contain more than 32768 materials. Keep target-grid IDs wide
        # and signed so -1 remains distinct from every valid material ID.
        present = data >= 0
        mapped = np.full(data.shape, -1, dtype=np.int32)
        mapped[present] = self.material_id_map[data[present]]
        return mapped

    def _read_rigid(self, family: str) -> None:
        """Import cell-owned rigidity only for non-transparent components.

        Solid/tag transparency is defined by /data; each rigidity bit instead
        follows its own /ID position. An explicit edge can therefore be
        imported even when its surrounding cells are transparent. The offsets
        mirror get_rigid_Ex/Ey/Ez/Hx/Hy/Hz in yee_cell_setget_rigid.pyx.
        """
        if self.grid_view is None:
            return

        dataset = self.file_handler[f"/rigid{family}"]
        assert isinstance(dataset, h5py.Dataset)
        rigid = self._read_spatial_dataset(dataset, component_axis=True)
        if self.has_ID_array():
            component_ids = self._read_spatial_dataset(
                self.file_handler["/ID"], component_axis=True, edge_based=True
            )
            self._check_material_coverage(component_ids)
            existing = self._get_assignment_region(getattr(self.grid_view.grid, f"rigid{family}"))
            # Each tuple is (ID component, x offset, y offset, z offset),
            # with offsets in cells relative to the cell owning this bit.
            # Component order is Ex/Ey/Ez/Hx/Hy/Hz; tuple order is rigidity-bit
            # order, so a shared component can protect several cells' claims.
            if family == "E":
                positions = (
                    (0, 0, 0, 0),
                    (0, 0, 1, 0),
                    (0, 0, 1, 1),
                    (0, 0, 0, 1),
                    (1, 0, 0, 0),
                    (1, 0, 0, 1),
                    (1, 1, 0, 1),
                    (1, 1, 0, 0),
                    (2, 0, 0, 0),
                    (2, 1, 0, 0),
                    (2, 1, 1, 0),
                    (2, 0, 1, 0),
                )
            else:
                positions = (
                    (3, 0, 0, 0),
                    (3, 1, 0, 0),
                    (4, 0, 0, 0),
                    (4, 0, 1, 0),
                    (5, 0, 0, 0),
                    (5, 0, 0, 1),
                )
            for bit, (component, *offset) in enumerate(positions):
                spatial = tuple(
                    slice(start, start + size) for start, size in zip(offset, rigid.shape[1:])
                )
                transparent = component_ids[(component, *spatial)] == -1
                rigid[bit] = np.where(transparent, existing[bit], rigid[bit])
        getattr(self.grid_view, f"set_rigid{family}")(rigid)

    def read_rigidE(self):
        self._read_rigid("E")

    def read_rigidH(self):
        self._read_rigid("H")

    def read_ID(self):
        if self.grid_view is None:
            return

        ID = self.file_handler["/ID"]
        assert isinstance(ID, h5py.Dataset)

        data = self._read_spatial_dataset(ID, component_axis=True, edge_based=True)
        existing = self._get_assignment_region(self.grid_view.grid.ID, edge_based=True)
        self.grid_view.set_ID(self._remap(data, existing))
