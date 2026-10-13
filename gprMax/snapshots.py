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

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Generic, List

import h5py
import numpy as np
from tqdm import tqdm

import gprMax.config as config
from gprMax.geometry_outputs.grid_view import GridType, GridView, MPIGridView
from gprMax.grid.axes import Dim, Dir
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.vtkhdf_filehandlers.vtk_image_data import VtkImageData

if TYPE_CHECKING:
    from mpi4py import MPI

    from gprMax.grid.mpi_grid import MPIGrid

from ._version import __version__
from .cython.snapshots import calculate_snapshot_fields
from .mode2d import mode2d_geometry
from .utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)

# Native Yee locations, in half-cell units.
YEE_OFFSETS = {
    "Ex": (1, 0, 0),
    "Ey": (0, 1, 0),
    "Ez": (0, 0, 1),
    "Hx": (0, 1, 1),
    "Hy": (1, 0, 1),
    "Hz": (1, 1, 0),
}


def validate_snapshot_sampling(grid, start, stop, step):
    """Reject regular output centres requiring samples outside the native grid.

    A non-dividing interior ROI keeps its final regular cell. Its centre may
    extend beyond the requested ROI, but its interpolation stencil must remain
    within physical Yee support; padded half-cell components are not samples.
    """
    start, stop, step = (np.asarray(value, dtype=np.int64) for value in (start, stop, step))
    if np.any(step < 1) or np.any(stop <= start):
        raise ValueError("Snapshot requires positive extents and sampling steps.")
    if hasattr(grid, "global_size"):
        start = grid.local_to_global_coordinate(start)
        stop = grid.local_to_global_coordinate(stop)
        shape = np.asarray(grid.global_size)
    else:
        shape = np.asarray(grid.size)
    count = (stop - start + step - 1) // step
    last = start + (count - 1) * step
    geometry = mode2d_geometry(config.get_model_config().mode)
    axes = [axis for axis in range(3) if geometry is None or axis != geometry.invariant_axis]
    for component, offsets in YEE_OFFSETS.items():
        offsets = np.asarray(offsets)
        lower = start + (step - offsets) // 2
        upper = last + (step - offsets + 1) // 2
        if np.any(lower[axes] < 0) or np.any(upper[axes] > (shape - offsets)[axes]):
            raise ValueError(
                f"Snapshot coarse-cell centre requires {component} samples outside native Yee "
                "support. Reduce the extent or sampling step; partial final cells are not clipped."
            )


def save_snapshots(snapshots: List["Snapshot"]):
    """Saves snapshots to file(s).

    Args:
        grid: FDTDGrid class describing a grid in a model.
    """

    # Create directory for snapshots
    snapshotdir = config.get_model_config().set_snapshots_dir()
    snapshotdir.mkdir(exist_ok=True)
    logger.info("")
    logger.info(f"Snapshot directory: {snapshotdir.resolve()}")

    for i, snap in enumerate(snapshots):
        # Re-derive from just the basename, not the previous value of
        # snap.filename directly: under geometry_fixed reuse this function
        # runs again on the same Snapshot object, whose filename is already
        # an absolute path from the prior run. Path.__truediv__ discards the
        # left side entirely when the right side is already absolute, so
        # naively joining would silently collapse back to the previous
        # run's directory, overwriting its file instead of writing this
        # run's own.
        snap.filename = snapshotdir / Path(snap.filename).name
        pbar = tqdm(
            total=snap.nbytes,
            leave=True,
            unit="byte",
            unit_scale=True,
            desc=f"Writing snapshot file {i + 1} of {len(snapshots)}, {snap.filename.name}",
            ncols=get_terminal_width() - 1,
            file=sys.stdout,
            disable=not config.sim_config.general["progressbars"],
        )
        snap.write_file(pbar)
        pbar.close()
        # Free the full-size field buffers once written. Safe except under
        # geometry_fixed reuse (build(), and so initialise_snapfields(),
        # only runs once - a later reused run's store() needs these arrays
        # to still exist) - see #389.
        if not (config.sim_config.geometry_fixed and config.sim_config.number_of_models > 1):
            snap.snapfields = {}
    logger.info("")


def _snapshot_axis_strides():
    """Neighbour-offset strides (sx, sy, sz) for calculate_snapshot_fields().

    1 (genuine +1-neighbour averaging) on every axis for 3D mode and for
    2D TM mode - TM's only surviving components (Ez, Hx, Hy) never
    reference the invariant axis in their averaging formula at all, so a
    stride of 1 there is inert either way.

    For 2D TE mode, the invariant axis's stride is 0 instead: that axis
    has only ever one genuine field value (flanked by forced-zero
    PEC/PMC wall padding from FDTDGrid.tex()/tey()/tez() on either side,
    not a second real value) - averaging against the +1 neighbour there
    was unconditionally blending the one real value with a forced zero,
    an exact 50% reduction on every 2D TE-mode snapshot regardless of
    proximity to any actual PEC/PMC boundary in the model. A stride of 0
    makes both terms of any pair on that axis resolve to the same
    (genuine) index, i.e. (X + X) / n = X - no averaging, just the real
    value - instead of a special-cased branch per component.
    """
    geometry = mode2d_geometry(config.get_model_config().mode)
    if geometry is None:
        return 1, 1, 1
    return geometry.collocation_strides


class Snapshot(Generic[GridType]):
    """Snapshot fields at one zero-based electric-field time level.

    At iteration ``n``, electric fields are at ``n*dt`` and magnetic fields
    retain their native Yee staggering at ``(n-1/2)*dt``. Storage performs no
    temporal averaging; the field calculation below only collocates components
    spatially within each output cell.
    """

    allowableoutputs = {
        "Ex": None,
        "Ey": None,
        "Ez": None,
        "Hx": None,
        "Hy": None,
        "Hz": None,
    }

    # Snapshots can be output as VTK ImageData (.vtkhdf) format or
    # HDF5 format (.h5) files
    fileexts = [".vtkhdf", ".h5"]

    # Dimensions of largest requested snapshot
    nx_max = 0
    ny_max = 0
    nz_max = 0

    # GPU - threads per block
    tpb = (1, 1, 1)
    # GPU - blocks per grid - set according to largest requested snapshot
    bpg = None

    @property
    def GRID_VIEW_TYPE(self) -> type[GridView]:
        return GridView

    def __init__(
        self,
        xs: int,
        ys: int,
        zs: int,
        xf: int,
        yf: int,
        zf: int,
        dx: int,
        dy: int,
        dz: int,
        iteration: int,
        filename: str,
        fileext: str,
        outputs: Dict[str, bool],
        grid: GridType,
    ):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for the extent of the volume in cells.
            dx, dy, dz: ints for the spatial discretisation in cells.
            iteration: zero-based electric-field time level at which to take
                the snapshot.
            filename: string for the filename to save to.
            fileext: string for the file extension.
            outputs: dict of booleans for fields to use for snapshot.
        """

        self.fileext = fileext
        self.filename = Path(filename)
        if self.filename.suffix in self.fileexts:
            self.filename = self.filename.with_suffix(fileext)
        else:
            self.filename = self.filename.with_name(self.filename.name + fileext)
        self.iteration = int(iteration)
        self.outputs = outputs
        validate_snapshot_sampling(grid, (xs, ys, zs), (xf, yf, zf), (dx, dy, dz))
        self.grid_view = self.GRID_VIEW_TYPE(grid, xs, ys, zs, xf, yf, zf, dx, dy, dz)

        self.nbytes = 0

        # Create arrays to hold the field data for snapshot
        self.snapfields = {}

    @property
    def grid(self) -> GridType:
        return self.grid_view.grid

    @property
    def electric_time(self) -> float:
        """Physical time of the electric fields in the snapshot."""

        return self.iteration * self.grid.dt

    @property
    def magnetic_time(self) -> float:
        """Physical time of the native, half-step-staggered magnetic fields."""

        return (self.iteration - 0.5) * self.grid.dt

    # Properties for backwards compatibility
    @property
    def xs(self) -> int:
        return self.grid_view.xs

    @property
    def ys(self) -> int:
        return self.grid_view.ys

    @property
    def zs(self) -> int:
        return self.grid_view.zs

    @property
    def xf(self) -> int:
        return self.grid_view.xf

    @property
    def yf(self) -> int:
        return self.grid_view.yf

    @property
    def zf(self) -> int:
        return self.grid_view.zf

    @property
    def nx(self) -> int:
        return self.grid_view.nx

    @property
    def ny(self) -> int:
        return self.grid_view.ny

    @property
    def nz(self) -> int:
        return self.grid_view.nz

    @property
    def dx(self) -> int:
        return self.grid_view.dx

    @property
    def dy(self) -> int:
        return self.grid_view.dy

    @property
    def dz(self) -> int:
        return self.grid_view.dz

    def initialise_snapfields(self):
        for k, v in self.outputs.items():
            if v:
                self.snapfields[k] = np.zeros(
                    self.grid_view.size,
                    dtype=config.sim_config.dtypes["float_or_double"],
                )
                self.nbytes += self.snapfields[k].nbytes
            else:
                # If output is not required for snapshot just use a mimimal
                # size of array - still required to pass to Cython function
                self.snapfields[k] = np.zeros(
                    (1, 1, 1), dtype=config.sim_config.dtypes["float_or_double"]
                )

    def store(self):
        """Store spatially collocated fields without changing their time levels."""
        self._store_native(self.grid_view.size)

    def _store_native(self, size):
        """Collocate a native-data-backed prefix into existing output buffers.

        ``size`` counts output cells along each axis, not native grid cells.
        It may be smaller than the allocated buffers when an MPI snapshot
        handles the remaining cells through remote sampling. ``grid_view.step``
        gives output-cell widths in native cells; the input views must retain
        the intervening native samples, rather than stride over them first.
        """

        # Interpolate native samples, not already-strided coarse corners.
        end = self.grid_view.start + (size - 1) * self.grid_view.step
        end += (self.grid_view.step + 1) // 2 + 1
        native_slice = tuple(slice(int(a), int(b)) for a, b in zip(self.grid_view.start, end))
        Exslice, Eyslice, Ezslice, Hxslice, Hyslice, Hzslice = (
            getattr(self.grid, name)[native_slice] for name in YEE_OFFSETS
        )

        # Spatially collocate field components in snapshot cells. The E(n)
        # and H(n-1/2) time levels are stored without temporal averaging.
        sx, sy, sz = _snapshot_axis_strides()
        calculate_snapshot_fields(
            int(size[0]),
            int(size[1]),
            int(size[2]),
            config.get_model_config().ompthreads,
            self.outputs["Ex"],
            self.outputs["Ey"],
            self.outputs["Ez"],
            self.outputs["Hx"],
            self.outputs["Hy"],
            self.outputs["Hz"],
            Exslice,
            Eyslice,
            Ezslice,
            Hxslice,
            Hyslice,
            Hzslice,
            self.snapfields["Ex"],
            self.snapfields["Ey"],
            self.snapfields["Ez"],
            self.snapfields["Hx"],
            self.snapfields["Hy"],
            self.snapfields["Hz"],
            sx,
            sy,
            sz,
            self.dx,
            self.dy,
            self.dz,
        )

    def write_file(self, pbar: tqdm):
        """Writes snapshot file either as VTK ImageData (.vtkhdf) format
            or HDF5 format (.h5) files

        Args:
            pbar: Progress bar class instance.
            G: FDTDGrid class describing a grid in a model.
        """

        if self.fileext == ".vtkhdf":
            self.write_vtk(pbar)
        elif self.fileext == ".h5":
            self.write_hdf5(pbar)

    def write_vtk(self, pbar: tqdm):
        """Writes snapshot file in VTK ImageData (.vtkhdf) format.

        Args:
            pbar: Progress bar class instance.
        """

        origin = self._physical_origin()
        spacing = self.grid_view.step * self.grid.dl

        with VtkImageData(self.filename, self.grid_view.size, origin, spacing) as f:
            for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
                if self.outputs[key]:
                    f.add_cell_data(key, self.snapfields[key])
                    pbar.update(n=self.snapfields[key].nbytes)

    def write_hdf5(self, pbar: tqdm):
        """Writes snapshot file in HDF5 (.h5) format.

        Args:
            pbar: Progress bar class instance.
        """

        with h5py.File(self.filename, "w") as f:
            f.attrs["gprMax"] = __version__
            # TODO: Output model name (title) and grid name? in snapshot output
            # f.attrs["Title"] = G.title
            f.attrs["nx_ny_nz"] = tuple(self.grid_view.size)
            f.attrs["dx_dy_dz"] = self.grid_view.step * self.grid.dl
            f.attrs["origin"] = self._physical_origin()
            f.attrs["iteration"] = self.iteration
            # ``time`` remains the electric-field time for backwards
            # compatibility with existing snapshot readers.
            f.attrs["time"] = self.electric_time
            f.attrs["magnetic_time"] = self.magnetic_time

            for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
                if self.outputs[key]:
                    f[key] = self.snapfields[key]
                    pbar.update(n=self.snapfields[key].nbytes)

    def _physical_origin(self):
        """Return the snapshot origin in the model's global coordinate frame."""

        if isinstance(self.grid, SubGridBaseGrid):
            return self.grid.local_to_global(self.grid_view.start)
        origin = np.asarray(self.grid_view.start * self.grid.dl, dtype=np.float64)
        geometry = mode2d_geometry(config.get_model_config().mode)
        if geometry is not None and geometry.polarisation == "TE":
            # VTK/HDF5 store cell data, so their origin is the lower vertex of
            # the output cell.  The genuine TE fields live on the central
            # invariant plane at index one; move the one-cell output extent
            # back by half a cell so its cell centre lies on that plane.
            origin[geometry.invariant_axis] -= 0.5 * self.grid.dl[geometry.invariant_axis]
        return origin


class SnapshotMPIGridView(MPIGridView):
    """Output-only decomposition with all ranks available for sparse samples.

    Coarse lower corners retain ownership of output cells. A rank with no
    output cells can still own native samples needed by another rank. This
    does not change the general geometry-view decomposition or file format.

    Offsets/sizes count coarse output cells; starts/stops are local native
    grid indices. The communicator is the full grid communicator, not a
    subset selected by ownership of coarse output cells.
    """

    def __init__(self, grid, xs, ys, zs, xf, yf, zf, dx=1, dy=1, dz=1):
        GridView.__init__(self, grid, xs, ys, zs, xf, yf, zf, dx, dy, dz)
        self.comm = grid.comm
        self.global_start = grid.local_to_global_coordinate(self.start)
        self.global_stop = grid.local_to_global_coordinate(self.stop)
        self.global_size = self.size.copy()
        owned_lower = grid.lower_extent + grid.negative_halo_offset
        owned_upper = grid.upper_extent
        self.offset = np.clip(
            (owned_lower - self.global_start + self.step - 1) // self.step, 0, self.global_size
        ).astype(np.int32)
        end = np.clip(
            (owned_upper - self.global_start + self.step - 1) // self.step, 0, self.global_size
        ).astype(np.int32)
        self.size = np.maximum(end - self.offset, 0)
        self.start = grid.global_to_local_coordinate(self.global_start + self.offset * self.step)
        self.stop = self.start + self.size * self.step
        self.has_positive_neighbour = end < self.global_size
        self.has_negative_neighbour = self.offset > 0


class MPISnapshot(Snapshot["MPIGrid"]):
    _SAMPLE_BATCH_SIZE = 65536
    H_TAG = 0
    EX_TAG = 1
    EY_TAG = 2
    EZ_TAG = 3

    @property
    def GRID_VIEW_TYPE(self) -> type[MPIGridView]:
        return SnapshotMPIGridView

    def __init__(
        self,
        xs: int,
        ys: int,
        zs: int,
        xf: int,
        yf: int,
        zf: int,
        dx: int,
        dy: int,
        dz: int,
        iteration: int,
        filename: str,
        fileext: str,
        outputs: Dict[str, bool],
        grid: MPIGrid,
    ):
        super().__init__(
            xs, ys, zs, xf, yf, zf, dx, dy, dz, iteration, filename, fileext, outputs, grid
        )

        assert isinstance(self.grid_view, self.GRID_VIEW_TYPE)
        self.comm = self.grid_view.comm

        # Get neighbours
        self.neighbours = np.full((3, 2), -1, dtype=int)
        self.neighbours[Dim.X] = self.comm.Shift(direction=Dim.X, disp=1)
        self.neighbours[Dim.Y] = self.comm.Shift(direction=Dim.Y, disp=1)
        self.neighbours[Dim.Z] = self.comm.Shift(direction=Dim.Z, disp=1)

    def has_neighbour(self, dimension: Dim, direction: Dir) -> bool:
        return self.neighbours[dimension][direction] >= 0

    def store(self):
        """Collectively collocate native samples at the current E/H time levels.

        Every grid rank must enter with the same snapshot definition and
        selected components, including ranks owning no output cells. A local
        prefix uses only uniquely owned native samples, not potentially stale
        halos. Remaining samples are requested from their owning ranks in
        batches of output cells; all ranks perform the same collective batch
        count. This is spatial interpolation, not temporal E/H averaging.
        """
        logger.debug(f"Saving snapshot for iteration: {self.iteration}")
        view = self.grid_view
        shape = tuple(int(value) for value in view.size)
        local_count = int(np.prod(view.size))
        selected_indices = None
        strides = np.asarray(_snapshot_axis_strides(), dtype=np.int32)
        # Collocate the largest rectangular prefix whose entire requested
        # native stencil is uniquely owned here. Positive halos/corners are
        # deliberately excluded, even when allocated, because they may be stale.
        # The full output buffers remain contiguous; Cython's loop sizes can be
        # smaller than their allocated shape without copying or changing strides.
        shifts = [
            (strides * (view.step - np.asarray(offsets, dtype=np.int32)) + 1) // 2
            for name, offsets in YEE_OFFSETS.items()
            if self.outputs[name]
        ]
        max_shift = np.max(shifts, axis=0) if shifts else np.zeros(3, dtype=np.int32)
        owned_last = self.grid.size - (self.grid.neighbours[:, Dir.POS] >= 0)
        for name in YEE_OFFSETS:
            if self.outputs[name]:
                owned_last = np.minimum(owned_last, np.asarray(getattr(self.grid, name).shape) - 1)
        prefix = np.minimum(
            view.size, np.maximum(0, (owned_last - view.start - max_shift) // view.step + 1)
        )
        if local_count and np.all(view.start >= 0) and np.all(prefix > 0):
            self._store_native(prefix)
            complement = []
            # Disjoint slabs form the complement of the prefix. Each later
            # slab restricts earlier axes to the prefix, avoiding duplicates.
            for axis in range(3):
                if prefix[axis] < view.size[axis]:
                    slab_shape = view.size.copy()
                    slab_shape[:axis] = prefix[:axis]
                    slab_shape[axis] -= prefix[axis]
                    slab = np.indices(tuple(slab_shape), dtype=np.int32).reshape(3, -1)
                    slab[axis] += prefix[axis]
                    complement.append(np.ravel_multi_index(tuple(slab), shape))
            selected_indices = (
                np.concatenate(complement) if complement else np.empty(0, dtype=np.intp)
            )
            local_count = len(selected_indices)
        largest_count = max(self.comm.allgather(local_count))
        rank_lookup = np.empty(tuple(self.grid.mpi_tasks), dtype=np.int32)
        for rank_coordinates in np.ndindex(rank_lookup.shape):
            rank_lookup[rank_coordinates] = self.comm.Get_cart_rank(rank_coordinates)
        # Bound coordinate/ownership scratch space independently of output size.
        # All ranks execute the same batch count, including empty-output ranks.
        for first in range(0, largest_count, self._SAMPLE_BATCH_SIZE):
            last = min(first + self._SAMPLE_BATCH_SIZE, local_count)
            first_local = min(first, local_count)
            output_indices = (
                np.arange(first_local, last)
                if selected_indices is None
                else selected_indices[first_local:last]
            )
            indices = (
                np.asarray(np.unravel_index(output_indices, shape), dtype=np.int32).T
                if last > first_local
                else np.empty((0, 3), dtype=np.int32)
            )
            anchors = view.global_start + (indices + view.offset) * view.step
            for component, offsets in YEE_OFFSETS.items():
                if not self.outputs[component]:
                    continue
                q = strides * (view.step - np.asarray(offsets, dtype=np.int32))
                if np.all(view.step == 1):
                    # Match the original Cython arithmetic order, including TE's
                    # duplicate invariant-plane terms, exactly.
                    legacy = {
                        "Ex": ((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)),
                        "Ey": ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)),
                        "Ez": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
                        "Hx": ((0, 0, 0), (1, 0, 0)),
                        "Hy": ((0, 0, 0), (0, 1, 0)),
                        "Hz": ((0, 0, 0), (0, 0, 1)),
                    }
                    stencil = np.asarray(legacy[component], dtype=np.int32) * strides
                else:
                    stencil = np.asarray(
                        list(itertools.product(*(range(1 + int(v % 2)) for v in q))), dtype=np.int32
                    )
                    stencil += q // 2
                # Process one component at a time: numeric coordinate blocks only,
                # no Python object per sample and no communication for local data.
                coordinates = (anchors[:, None, :] + stencil[None, :, :]).reshape(-1, 3)
                rank_coordinates = self.grid.get_grid_coord_from_coordinate(coordinates)
                owners = rank_lookup[tuple(rank_coordinates.T)]
                field = getattr(self.grid, component)
                samples = np.empty(len(coordinates), dtype=field.dtype)
                local = owners == self.comm.rank
                local_coordinates = coordinates[local] - self.grid.lower_extent
                samples[local] = field[tuple(local_coordinates.T)]
                requests = []
                slots = []
                for rank in range(self.comm.size):
                    selected = (
                        np.flatnonzero(owners == rank)
                        if rank != self.comm.rank
                        else np.empty(0, dtype=np.intp)
                    )
                    slots.append(selected)
                    requests.append(np.ascontiguousarray(coordinates[selected], dtype=np.int32))
                incoming = self.comm.alltoall(requests)
                replies = [
                    np.ascontiguousarray(field[tuple((block - self.grid.lower_extent).T)])
                    for block in incoming
                ]
                received = self.comm.alltoall(replies)
                for selected, values in zip(slots, received):
                    samples[selected] = values
                values = samples.reshape(len(anchors), len(stencil))
                # Left-associated additions match both native CPU/GPU paths.
                result = values[:, 0].copy()
                for column in range(1, values.shape[1]):
                    result += values[:, column]
                result /= values.shape[1]
                self.snapfields[component].flat[output_indices] = result

    def write_vtk(self, pbar: tqdm):
        """Writes snapshot file in VTK ImageData (.vtkhdf) format.

        Args:
            pbar: Progress bar class instance.
        """
        assert isinstance(self.grid_view, self.GRID_VIEW_TYPE)

        origin = self._physical_origin()
        spacing = self.grid_view.step * self.grid.dl

        with VtkImageData(
            self.filename, self.grid_view.global_size, origin, spacing, comm=self.comm
        ) as f:
            for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
                if self.outputs.get(key):
                    f.add_cell_data(key, self.snapfields[key], self.grid_view.offset)
                    pbar.update(n=self.snapfields[key].nbytes)

    def write_hdf5(self, pbar: tqdm):
        """Writes snapshot file in HDF5 (.h5) format.

        Args:
            pbar: Progress bar class instance.
        """
        assert isinstance(self.grid_view, self.GRID_VIEW_TYPE)

        with h5py.File(self.filename, "w", driver="mpio", comm=self.comm) as f:
            f.attrs["gprMax"] = __version__
            # TODO: Output model name (title) and grid name? in snapshot output
            # f.attrs["Title"] = G.title
            f.attrs["nx_ny_nz"] = self.grid_view.global_size
            f.attrs["dx_dy_dz"] = self.grid_view.step * self.grid.dl
            f.attrs["origin"] = self._physical_origin()
            f.attrs["iteration"] = self.iteration
            # ``time`` remains the electric-field time for backwards
            # compatibility with existing snapshot readers.
            f.attrs["time"] = self.electric_time
            f.attrs["magnetic_time"] = self.magnetic_time

            dset_slice = self.grid_view.get_3d_output_slice()

            for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
                if self.outputs[key]:
                    dset = f.create_dataset(
                        key, self.grid_view.global_size, dtype=self.snapfields[key].dtype
                    )
                    dset[dset_slice] = self.snapfields[key]
                    pbar.update(n=self.snapfields[key].nbytes)

    def _physical_origin(self):
        origin = np.asarray(self.grid_view.global_start * self.grid.dl, dtype=np.float64)
        geometry = mode2d_geometry(config.get_model_config().mode)
        if geometry is not None and geometry.polarisation == "TE":
            origin[geometry.invariant_axis] -= 0.5 * self.grid.dl[geometry.invariant_axis]
        return origin


def update_snapshot_max_dims(snapshots: List["Snapshot"]):
    """Replace allocation maxima with those of this model, never earlier models.

    Kernel dimensions must agree with these allocation dimensions. Updaters
    retain their own shape so another grid cannot change their dispatch.
    """
    shape = tuple(
        max((getattr(snap, axis) for snap in snapshots), default=0) for axis in ("nx", "ny", "nz")
    )
    Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max = shape
    return shape


def htod_snapshot_array(snapshots: List[Snapshot], queue=None):
    """Initialises arrays on compute device to store field data for snapshots.

    Args:
        G: FDTDGrid class describing a grid in a model.
        queue: pyopencl queue.

    Returns:
        snapE_dev, snapH_dev: float arrays of snapshot data on compute device.
    """

    solver = config.sim_config.general.get("solver")
    if solver not in ("cuda", "opencl", "metal"):
        raise ValueError(f"Unknown device solver {solver!r} for snapshot arrays.")

    # Get dimensions of largest requested snapshot
    shape = update_snapshot_max_dims(snapshots)

    if config.sim_config.general["solver"] == "cuda":
        # Blocks per grid - according to largest requested snapshot
        Snapshot.bpg = (
            int(
                np.ceil(
                    ((Snapshot.nx_max) * (Snapshot.ny_max) * (Snapshot.nz_max)) / Snapshot.tpb[0]
                )
            ),
            1,
            1,
        )
    elif config.sim_config.general["solver"] == "opencl":
        # Workgroup size - according to largest requested snapshot
        Snapshot.wgs = (
            int(np.ceil(((Snapshot.nx_max) * (Snapshot.ny_max) * (Snapshot.nz_max)))),
            1,
            1,
        )

    # 4D arrays to store snapshots on GPU, e.g. snapEx(time, x, y, z);
    # if snapshots are not being stored on the GPU during the simulation then
    # they are copied back to the host after each iteration, hence numsnaps = 1
    numsnaps = 1 if config.get_model_config().device["snapsgpu2cpu"] else len(snapshots)
    snapEx = np.zeros(
        (numsnaps, *shape),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapEy = np.zeros(
        (numsnaps, *shape),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapEz = np.zeros(
        (numsnaps, *shape),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapHx = np.zeros(
        (numsnaps, *shape),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapHy = np.zeros(
        (numsnaps, *shape),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapHz = np.zeros(
        (numsnaps, *shape),
        dtype=config.sim_config.dtypes["float_or_double"],
    )

    # Copy arrays to compute device
    if config.sim_config.general["solver"] == "cuda":
        import pycuda.gpuarray as gpuarray

        snapEx_dev = gpuarray.to_gpu(snapEx)
        snapEy_dev = gpuarray.to_gpu(snapEy)
        snapEz_dev = gpuarray.to_gpu(snapEz)
        snapHx_dev = gpuarray.to_gpu(snapHx)
        snapHy_dev = gpuarray.to_gpu(snapHy)
        snapHz_dev = gpuarray.to_gpu(snapHz)

    elif config.sim_config.general["solver"] == "opencl":
        import pyopencl.array as clarray

        snapEx_dev = clarray.to_device(queue, snapEx)
        snapEy_dev = clarray.to_device(queue, snapEy)
        snapEz_dev = clarray.to_device(queue, snapEz)
        snapHx_dev = clarray.to_device(queue, snapHx)
        snapHy_dev = clarray.to_device(queue, snapHy)
        snapHz_dev = clarray.to_device(queue, snapHz)

    elif config.sim_config.general["solver"] == "metal":
        # Metal doesn't use a queue parameter, need to get device from config
        dev = config.get_model_config().device["dev"]

        snapEx_dev = dev.newBufferWithBytes_length_options_(snapEx, snapEx.nbytes, 0)
        snapEy_dev = dev.newBufferWithBytes_length_options_(snapEy, snapEy.nbytes, 0)
        snapEz_dev = dev.newBufferWithBytes_length_options_(snapEz, snapEz.nbytes, 0)
        snapHx_dev = dev.newBufferWithBytes_length_options_(snapHx, snapHx.nbytes, 0)
        snapHy_dev = dev.newBufferWithBytes_length_options_(snapHy, snapHy.nbytes, 0)
        snapHz_dev = dev.newBufferWithBytes_length_options_(snapHz, snapHz.nbytes, 0)

    return snapEx_dev, snapEy_dev, snapEz_dev, snapHx_dev, snapHy_dev, snapHz_dev


def dtoh_snapshot_array(
    snapEx_dev, snapEy_dev, snapEz_dev, snapHx_dev, snapHy_dev, snapHz_dev, i, snap
):
    """Copies snapshot array used on compute device back to snapshot objects and
        store in format for Paraview.

    Args:
        snapE_dev, snapH_dev: float arrays of snapshot data from compute device.
        i: int for index of snapshot data on compute device array.
        snap: Snapshot class instance
    """

    # snap*_dev's own (x, y, z) axes are already local/0-based (0..nx-1 etc,
    # where nx/ny/nz is this snapshot's own sample count) - store_snapshot()
    # (knl_snapshots.py) writes into them using that same local indexing.
    # They are NOT indexed by the absolute grid coordinates snap.xs/snap.ys/
    # snap.zs/etc - slicing with those (the original, buggy version of this
    # function) only happened to work for a snapshot starting exactly at the
    # grid origin; any other position silently truncated/misaligned the
    # copied-back data (confirmed empirically: an origin snapshot worked,
    # a non-origin one produced a wrongly-shaped result).
    snap.snapfields["Ex"] = snapEx_dev[i, : snap.nx, : snap.ny, : snap.nz]
    snap.snapfields["Ey"] = snapEy_dev[i, : snap.nx, : snap.ny, : snap.nz]
    snap.snapfields["Ez"] = snapEz_dev[i, : snap.nx, : snap.ny, : snap.nz]
    snap.snapfields["Hx"] = snapHx_dev[i, : snap.nx, : snap.ny, : snap.nz]
    snap.snapfields["Hy"] = snapHy_dev[i, : snap.nx, : snap.ny, : snap.nz]
    snap.snapfields["Hz"] = snapHz_dev[i, : snap.nx, : snap.ny, : snap.nz]
