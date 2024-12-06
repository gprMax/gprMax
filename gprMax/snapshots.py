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

import logging
import sys
from enum import IntEnum, unique
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import numpy.typing as npt
from evtk.hl import imageToVTK
from mpi4py import MPI
from tqdm import tqdm

import gprMax.config as config

from ._version import __version__
from .cython.snapshots import calculate_snapshot_fields
from .utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


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
        fn = snapshotdir / Path(snap.filename)
        snap.filename = fn.with_suffix(snap.fileext)
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
    logger.info("")


class Snapshot:
    """Snapshots of the electric and magnetic field values."""

    allowableoutputs = {
        "Ex": None,
        "Ey": None,
        "Ez": None,
        "Hx": None,
        "Hy": None,
        "Hz": None,
    }

    # Snapshots can be output as VTK ImageData (.vti) format or
    # HDF5 format (.h5) files
    fileexts = [".vti", ".h5"]

    # Dimensions of largest requested snapshot
    nx_max = 0
    ny_max = 0
    nz_max = 0

    # GPU - threads per block
    tpb = (1, 1, 1)
    # GPU - blocks per grid - set according to largest requested snapshot
    bpg = None

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
        time: int,
        filename: str,
        fileext: str,
        outputs: Dict[str, bool],
        grid_dl: npt.NDArray[np.float32],
        grid_dt: float,
    ):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for the extent of the volume in cells.
            dx, dy, dz: ints for the spatial discretisation in cells.
            time: int for the iteration number to take the snapshot on.
            filename: string for the filename to save to.
            fileext: optional string for the file extension.
            outputs: optional dict of booleans for fields to use for snapshot.
        """

        self.fileext = fileext
        self.filename = filename
        self.time = time
        self.outputs = outputs
        self.grid_dl = grid_dl
        self.grid_dt = grid_dt

        self.start = np.array([xs, ys, zs], dtype=np.intc)
        self.stop = np.array([xf, yf, zf], dtype=np.intc)
        self.step = np.array([dx, dy, dz], dtype=np.intc)
        self.size = np.ceil((self.stop - self.start) / self.step).astype(np.intc)
        self.slice: list[slice] = list(map(slice, self.start, self.stop + self.step, self.step))

        self.nbytes = 0

        # Create arrays to hold the field data for snapshot
        self.snapfields = {}

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

    @property
    def sx(self) -> slice:
        return self.slice[0]

    @property
    def sy(self) -> slice:
        return self.slice[1]

    @property
    def sz(self) -> slice:
        return self.slice[2]

    def initialise_snapfields(self):
        for k, v in self.outputs.items():
            if v:
                self.snapfields[k] = np.zeros(
                    (self.nx, self.ny, self.nz),
                    dtype=config.sim_config.dtypes["float_or_double"],
                )
                self.nbytes += self.snapfields[k].nbytes
            else:
                # If output is not required for snapshot just use a mimimal
                # size of array - still required to pass to Cython function
                self.snapfields[k] = np.zeros(
                    (1, 1, 1), dtype=config.sim_config.dtypes["float_or_double"]
                )

    def store(self, G):
        """Store (in memory) electric and magnetic field values for snapshot.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Memory views of field arrays to dimensions required for the snapshot
        Exslice = np.ascontiguousarray(G.Ex[self.sx, self.sy, self.sz])
        Eyslice = np.ascontiguousarray(G.Ey[self.sx, self.sy, self.sz])
        Ezslice = np.ascontiguousarray(G.Ez[self.sx, self.sy, self.sz])
        Hxslice = np.ascontiguousarray(G.Hx[self.sx, self.sy, self.sz])
        Hyslice = np.ascontiguousarray(G.Hy[self.sx, self.sy, self.sz])
        Hzslice = np.ascontiguousarray(G.Hz[self.sx, self.sy, self.sz])

        # Calculate field values at points (comes from averaging field
        # components in cells)
        calculate_snapshot_fields(
            self.nx,
            self.ny,
            self.nz,
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
        )

    def write_file(self, pbar: tqdm):
        """Writes snapshot file either as VTK ImageData (.vti) format
            or HDF5 format (.h5) files

        Args:
            pbar: Progress bar class instance.
            G: FDTDGrid class describing a grid in a model.
        """

        if self.fileext == ".vti":
            self.write_vtk(pbar)
        elif self.fileext == ".h5":
            self.write_hdf5(pbar)

    def write_vtk(self, pbar: tqdm):
        """Writes snapshot file in VTK ImageData (.vti) format.

        Args:
            pbar: Progress bar class instance.
        """

        celldata = {
            k: self.snapfields[k]
            for k in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
            if self.outputs.get(k)
        }

        imageToVTK(
            str(self.filename.with_suffix("")),
            origin=tuple(self.start * self.step * self.grid_dl),
            spacing=tuple(self.step * self.grid_dl),
            cellData=celldata,
        )

        pbar.update(
            n=len(celldata)
            * self.nx
            * self.ny
            * self.nz
            * np.dtype(config.sim_config.dtypes["float_or_double"]).itemsize
        )

    def write_hdf5(self, pbar: tqdm):
        """Writes snapshot file in HDF5 (.h5) format.

        Args:
            pbar: Progress bar class instance.
        """

        f = h5py.File(self.filename, "w")
        f.attrs["gprMax"] = __version__
        # TODO: Output model name (title) and grid name? in snapshot output
        # f.attrs["Title"] = G.title
        f.attrs["nx_ny_nz"] = (self.nx, self.ny, self.nz)
        f.attrs["dx_dy_dz"] = self.step * self.grid_dl
        f.attrs["time"] = self.time * self.grid_dt

        for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            if self.outputs[key]:
                f[key] = self.snapfields[key]
                pbar.update(n=self.snapfields[key].nbytes)

        f.close()


@unique
class Dim(IntEnum):
    X = 0
    Y = 1
    Z = 2


@unique
class Dir(IntEnum):
    NEG = 0
    POS = 1


class MPISnapshot(Snapshot):
    H_TAG = 0
    EX_TAG = 1
    EY_TAG = 2
    EZ_TAG = 3

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
        time: int,
        filename: str,
        fileext: str,
        outputs: Dict[str, bool],
        grid_dl: npt.NDArray[np.float32],
        grid_dt: float,
    ):
        super().__init__(
            xs, ys, zs, xf, yf, zf, dx, dy, dz, time, filename, fileext, outputs, grid_dl, grid_dt
        )

        self.offset = np.zeros(3, dtype=np.intc)
        self.global_size = self.size.copy()

        self.comm: MPI.Cartcomm = None  # type: ignore

    def initialise_snapfields(self):
        # Start and stop may have changed since initialisation
        self.size = np.ceil((self.stop - self.start) / self.step).astype(np.intc)
        return super().initialise_snapfields()

    def has_neighbour(self, dimension: Dim, direction: Dir) -> bool:
        return self.neighbours[dimension][direction] != -1

    def store(self, G):
        """Store (in memory) electric and magnetic field values for snapshot.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        logger.debug(f"Saving snapshot for iteration: {self.time}")

        # Get neighbours
        self.neighbours = np.full((3, 2), -1, dtype=int)
        self.neighbours[Dim.X] = self.comm.Shift(direction=Dim.X, disp=1)
        self.neighbours[Dim.Y] = self.comm.Shift(direction=Dim.Y, disp=1)
        self.neighbours[Dim.Z] = self.comm.Shift(direction=Dim.Z, disp=1)

        # If we do not have a positive neighbour, add an extra step to
        # make the upper bound inclusive. Otherwise the additional step
        # will be provided by the received halo.
        slice_stop = np.where(
            self.neighbours[:, Dir.POS] == -1,
            self.stop + self.step,
            self.stop,
        )
        self.slice = list(map(slice, self.start, slice_stop, self.step))

        # Memory views of field arrays to dimensions required for the snapshot
        Exslice = np.ascontiguousarray(G.Ex[self.sx, self.sy, self.sz])
        Eyslice = np.ascontiguousarray(G.Ey[self.sx, self.sy, self.sz])
        Ezslice = np.ascontiguousarray(G.Ez[self.sx, self.sy, self.sz])
        Hxslice = np.ascontiguousarray(G.Hx[self.sx, self.sy, self.sz])
        Hyslice = np.ascontiguousarray(G.Hy[self.sx, self.sy, self.sz])
        Hzslice = np.ascontiguousarray(G.Hz[self.sx, self.sy, self.sz])

        """
        Exslice - y + z halo
        Eyslice - x + z halo
        Ezslice - x + y halo
        Hxslice - x halo
        Hyslice - y halo
        Hzslice - z halo
        """

        # Shape and dtype should be the same for all field array slices
        shape = Hxslice.shape
        dtype = Hxslice.dtype

        Hxhalo = np.empty((1, shape[Dim.Y], shape[Dim.Z]), dtype=dtype)
        Hyhalo = np.empty((shape[Dim.X], 1, shape[Dim.Z]), dtype=dtype)
        Hzhalo = np.empty((shape[Dim.X], shape[Dim.Y], 1), dtype=dtype)

        Exyhalo = np.empty((shape[Dim.X], 1, shape[Dim.Z]), dtype=dtype)
        Eyzhalo = np.empty((shape[Dim.X], shape[Dim.Y], 1), dtype=dtype)
        Ezxhalo = np.empty((1, shape[Dim.Y], shape[Dim.Z]), dtype=dtype)

        x_offset = self.has_neighbour(Dim.X, Dir.POS)
        y_offset = self.has_neighbour(Dim.Y, Dir.POS)
        z_offset = self.has_neighbour(Dim.Z, Dir.POS)
        Exzhalo = np.empty((shape[Dim.X], shape[Dim.Y] + y_offset, 1), dtype=dtype)
        Eyxhalo = np.empty((1, shape[Dim.Y], shape[Dim.Z] + z_offset), dtype=dtype)
        Ezyhalo = np.empty((shape[Dim.X] + x_offset, 1, shape[Dim.Z]), dtype=dtype)

        blocking_requests: List[MPI.Request] = []
        requests: List[MPI.Request] = []

        if self.has_neighbour(Dim.X, Dir.NEG):
            requests += [
                self.comm.Isend(Hxslice[0, :, :], self.neighbours[Dim.X][Dir.NEG], self.H_TAG),
                self.comm.Isend(Ezslice[0, :, :], self.neighbours[Dim.X][Dir.NEG], self.EZ_TAG),
            ]
        if self.has_neighbour(Dim.X, Dir.POS):
            blocking_requests.append(
                self.comm.Irecv(Ezxhalo, self.neighbours[Dim.X][Dir.POS], self.EZ_TAG),
            )
            requests += [
                self.comm.Irecv(Hxhalo, self.neighbours[Dim.X][Dir.POS], self.H_TAG),
                self.comm.Irecv(Eyxhalo, self.neighbours[Dim.X][Dir.POS], self.EY_TAG),
            ]
        if self.has_neighbour(Dim.Y, Dir.NEG):
            requests += [
                self.comm.Isend(
                    np.ascontiguousarray(Hyslice[:, 0, :]),
                    self.neighbours[Dim.Y][Dir.NEG],
                    self.H_TAG,
                ),
                self.comm.Isend(
                    np.ascontiguousarray(Exslice[:, 0, :]),
                    self.neighbours[Dim.Y][Dir.NEG],
                    self.EX_TAG,
                ),
            ]
        if self.has_neighbour(Dim.Y, Dir.POS):
            blocking_requests.append(
                self.comm.Irecv(Exyhalo, self.neighbours[Dim.Y][Dir.POS], self.EX_TAG),
            )
            requests += [
                self.comm.Irecv(Hyhalo, self.neighbours[Dim.Y][Dir.POS], self.H_TAG),
                self.comm.Irecv(Ezyhalo, self.neighbours[Dim.Y][Dir.POS], self.EZ_TAG),
            ]
        if self.has_neighbour(Dim.Z, Dir.NEG):
            requests += [
                self.comm.Isend(
                    np.ascontiguousarray(Hzslice[:, :, 0]),
                    self.neighbours[Dim.Z][Dir.NEG],
                    self.H_TAG,
                ),
                self.comm.Isend(
                    np.ascontiguousarray(Eyslice[:, :, 0]),
                    self.neighbours[Dim.Z][Dir.NEG],
                    self.EY_TAG,
                ),
            ]
        if self.has_neighbour(Dim.Z, Dir.POS):
            blocking_requests.append(
                self.comm.Irecv(Eyzhalo, self.neighbours[Dim.Z][Dir.POS], self.EY_TAG),
            )
            requests += [
                self.comm.Irecv(Hzhalo, self.neighbours[Dim.Z][Dir.POS], self.H_TAG),
                self.comm.Irecv(Exzhalo, self.neighbours[Dim.Z][Dir.POS], self.EX_TAG),
            ]

        if len(blocking_requests) > 0:
            blocking_requests[0].Waitall(blocking_requests)

        logger.debug(f"Initial halo exchanges complete")

        if self.has_neighbour(Dim.X, Dir.POS):
            Ezslice = np.concatenate((Ezslice, Ezxhalo), axis=Dim.X)
        if self.has_neighbour(Dim.Y, Dir.POS):
            Exslice = np.concatenate((Exslice, Exyhalo), axis=Dim.Y)
        if self.has_neighbour(Dim.Z, Dir.POS):
            Eyslice = np.concatenate((Eyslice, Eyzhalo), axis=Dim.Z)

        if self.has_neighbour(Dim.X, Dir.NEG):
            requests.append(
                self.comm.Isend(Eyslice[0, :, :], self.neighbours[Dim.X][Dir.NEG], self.EY_TAG),
            )
        if self.has_neighbour(Dim.Y, Dir.NEG):
            requests.append(
                self.comm.Isend(
                    np.ascontiguousarray(Ezslice[:, 0, :]),
                    self.neighbours[Dim.Y][Dir.NEG],
                    self.EZ_TAG,
                ),
            )
        if self.has_neighbour(Dim.Z, Dir.NEG):
            requests.append(
                self.comm.Isend(
                    np.ascontiguousarray(Exslice[:, :, 0]),
                    self.neighbours[Dim.Z][Dir.NEG],
                    self.EX_TAG,
                ),
            )

        if len(requests) > 0:
            requests[0].Waitall(requests)

        logger.debug(f"All halo exchanges complete")

        if self.has_neighbour(Dim.X, Dir.POS):
            Eyslice = np.concatenate((Eyslice, Eyxhalo), axis=Dim.X)
            Hxslice = np.concatenate((Hxslice, Hxhalo), axis=Dim.X)
        if self.has_neighbour(Dim.Y, Dir.POS):
            Ezslice = np.concatenate((Ezslice, Ezyhalo), axis=Dim.Y)
            Hyslice = np.concatenate((Hyslice, Hyhalo), axis=Dim.Y)
        if self.has_neighbour(Dim.Z, Dir.POS):
            Exslice = np.concatenate((Exslice, Exzhalo), axis=Dim.Z)
            Hzslice = np.concatenate((Hzslice, Hzhalo), axis=Dim.Z)

        # Calculate field values at points (comes from averaging field
        # components in cells)
        calculate_snapshot_fields(
            self.nx,
            self.ny,
            self.nz,
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
        )

    def write_hdf5(self, pbar: tqdm):
        """Writes snapshot file in HDF5 (.h5) format.

        Args:
            pbar: Progress bar class instance.
        """

        f = h5py.File(self.filename, "w", driver="mpio", comm=self.comm)

        f.attrs["gprMax"] = __version__
        # TODO: Output model name (title) and grid name? in snapshot output
        # f.attrs["Title"] = G.title
        f.attrs["nx_ny_nz"] = self.global_size
        f.attrs["dx_dy_dz"] = self.step * self.grid_dl
        f.attrs["time"] = self.time * self.grid_dt

        for key in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            if self.outputs[key]:
                dset = f.create_dataset(key, self.global_size)
                # TODO: Is there a better way to do this slice?
                start = self.offset
                stop = start + self.size
                dset[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = self.snapfields[
                    key
                ]
                pbar.update(n=self.snapfields[key].nbytes)

        f.close()


def htod_snapshot_array(snapshots: List[Snapshot], queue=None):
    """Initialises arrays on compute device to store field data for snapshots.

    Args:
        G: FDTDGrid class describing a grid in a model.
        queue: pyopencl queue.

    Returns:
        snapE_dev, snapH_dev: float arrays of snapshot data on compute device.
    """

    # Get dimensions of largest requested snapshot
    for snap in snapshots:
        if snap.nx > Snapshot.nx_max:
            Snapshot.nx_max = snap.nx
        if snap.ny > Snapshot.ny_max:
            Snapshot.ny_max = snap.ny
        if snap.nz > Snapshot.nz_max:
            Snapshot.nz_max = snap.nz

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
        (numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapEy = np.zeros(
        (numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapEz = np.zeros(
        (numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapHx = np.zeros(
        (numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapHy = np.zeros(
        (numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
        dtype=config.sim_config.dtypes["float_or_double"],
    )
    snapHz = np.zeros(
        (numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
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

    snap.snapfields["Ex"] = snapEx_dev[i, snap.xs : snap.xf, snap.ys : snap.yf, snap.zs : snap.zf]
    snap.snapfields["Ey"] = snapEy_dev[i, snap.xs : snap.xf, snap.ys : snap.yf, snap.zs : snap.zf]
    snap.snapfields["Ez"] = snapEz_dev[i, snap.xs : snap.xf, snap.ys : snap.yf, snap.zs : snap.zf]
    snap.snapfields["Hx"] = snapHx_dev[i, snap.xs : snap.xf, snap.ys : snap.yf, snap.zs : snap.zf]
    snap.snapfields["Hy"] = snapHy_dev[i, snap.xs : snap.xf, snap.ys : snap.yf, snap.zs : snap.zf]
    snap.snapfields["Hz"] = snapHz_dev[i, snap.xs : snap.xf, snap.ys : snap.yf, snap.zs : snap.zf]
