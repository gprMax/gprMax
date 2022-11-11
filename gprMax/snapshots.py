# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
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
from pathlib import Path
import sys

import h5py
import numpy as np
from evtk.hl import imageToVTK
from tqdm import tqdm

import gprMax.config as config

from ._version import __version__
from .cython.snapshots import calculate_snapshot_fields
from .utilities.utilities import get_terminal_width, round_value

logger = logging.getLogger(__name__)


def save_snapshots(grid):
    """Saves snapshots to file(s).
    
    Args:
        grid: FDTDGrid class describing a grid in a model.
    """

    # Create directory for snapshots
    snapshotdir = config.get_model_config().set_snapshots_dir()
    snapshotdir.mkdir(exist_ok=True)
    logger.info('')
    logger.info(f'Snapshot directory: {snapshotdir.resolve()}')

    for i, snap in enumerate(grid.snapshots):
        fn = snapshotdir / Path(snap.filename)
        snap.filename = fn.with_suffix(snap.fileext)
        pbar = tqdm(total=snap.nbytes, leave=True, unit='byte',
                    unit_scale=True, desc=f'Writing snapshot file {i + 1} '
                                            f'of {len(grid.snapshots)}, '
                                            f'{snap.filename.name}', 
                    ncols=get_terminal_width() - 1, file=sys.stdout, 
                    disable=not config.sim_config.general['progressbars'])
        snap.write_file(pbar, grid)
        pbar.close()
    logger.info('')

class Snapshot:
    """Snapshots of the electric and magnetic field values."""

    allowableoutputs = {'Ex': None, 'Ey': None, 'Ez': None, 
                        'Hx': None, 'Hy': None, 'Hz': None}

    # Snapshots can be output as VTK ImageData (.vti) format or 
    # HDF5 format (.h5) files
    fileexts = ['.vti', '.h5']

    # Dimensions of largest requested snapshot
    nx_max = 0
    ny_max = 0
    nz_max = 0

    # GPU - threads per block
    tpb = (1, 1, 1)
    # GPU - blocks per grid - set according to largest requested snapshot
    bpg = None

    def __init__(self, xs=None, ys=None, zs=None, xf=None, yf=None, zf=None,
                 dx=None, dy=None, dz=None, time=None, filename=None, 
                 fileext=None, outputs=None):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for the extent of the volume in cells.
            dx, dy, dz: ints for the spatial discretisation in cells.
            time: int for the iteration number to take the snapshot on.
            filename: string for the filename to save to.
            fileext: string for the file extension.
            outputs: optional list of outputs for receiver. It can be any
                        selection from Ex, Ey, Ez, Hx, Hy, or Hz.
        """

        self.fileext = fileext
        self.filename = filename
        self.time = time
        self.outputs = outputs
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.xf = xf
        self.yf = yf
        self.zf = zf
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = round_value((self.xf - self.xs) / self.dx)
        self.ny = round_value((self.yf - self.ys) / self.dy)
        self.nz = round_value((self.zf - self.zs) / self.dz)
        self.sx = slice(self.xs, self.xf + self.dx, self.dx)
        self.sy = slice(self.ys, self.yf + self.dy, self.dy)
        self.sz = slice(self.zs, self.zf + self.dz, self.dz)
        self.nbytes = (6 * self.nx * self.ny * self.nz * 
                       np.dtype(config.sim_config.dtypes['float_or_double']).itemsize)

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

        # Create arrays to hold the field data for snapshot
        self.Exsnap = np.zeros((self.nx, self.ny, self.nz), 
                                dtype=config.sim_config.dtypes['float_or_double'])
        self.Eysnap = np.zeros((self.nx, self.ny, self.nz), 
                                dtype=config.sim_config.dtypes['float_or_double'])
        self.Ezsnap = np.zeros((self.nx, self.ny, self.nz), 
                                dtype=config.sim_config.dtypes['float_or_double'])
        self.Hxsnap = np.zeros((self.nx, self.ny, self.nz), 
                                dtype=config.sim_config.dtypes['float_or_double'])
        self.Hysnap = np.zeros((self.nx, self.ny, self.nz), 
                                dtype=config.sim_config.dtypes['float_or_double'])
        self.Hzsnap = np.zeros((self.nx, self.ny, self.nz), 
                                dtype=config.sim_config.dtypes['float_or_double'])

        # Calculate field values at points (comes from averaging field components in cells)
        calculate_snapshot_fields(
            self.nx,
            self.ny,
            self.nz,
            config.get_model_config().ompthreads,
            self.outputs['Ex'],
            self.outputs['Ey'],
            self.outputs['Ez'],
            self.outputs['Hx'],
            self.outputs['Hy'],
            self.outputs['Hz'],
            Exslice,
            Eyslice,
            Ezslice,
            Hxslice,
            Hyslice,
            Hzslice,
            self.Exsnap,
            self.Eysnap,
            self.Ezsnap,
            self.Hxsnap,
            self.Hysnap,
            self.Hzsnap
        )

    def write_file(self, pbar, G):
        """Writes snapshot file either as VTK ImageData (.vti) format 
            or HDF5 format (.h5) files

        Args:
            pbar: Progress bar class instance.
            G: FDTDGrid class describing a grid in a model.
        """

        if self.fileext == '.vti':
            self.write_vtk(pbar, G)
        elif self.fileext == '.h5':
            self.write_hdf5(pbar, G)

    def write_vtk(self, pbar, G):
        """Writes snapshot file in VTK ImageData (.vti) format.

        Args:
            pbar: Progress bar class instance.
            G: FDTDGrid class describing a grid in a model.
        """

        celldata = {}

        for k, v in self.outputs.items():
            if v:
                if k == 'Ex':
                    celldata[k] = self.Exsnap
                if k == 'Ey':
                    celldata[k] = self.Eysnap
                if k == 'Ez':
                    celldata[k] = self.Ezsnap
                if k == 'Hx':
                    celldata[k] = self.Hxsnap
                if k == 'Hy':
                    celldata[k] = self.Hysnap
                if k == 'Hz':
                    celldata[k] = self.Hzsnap

        imageToVTK(str(self.filename.with_suffix('')), 
                   origin=((self.xs * self.dx * G.dx), 
                           (self.ys * self.dy * G.dy), 
                           (self.zs * self.dz * G.dz)), 
                   spacing=((self.dx * G.dx),
                            (self.dy * G.dy),
                            (self.dz * G.dz)), 
                   cellData=celldata)
        
        pbar.update(n=len(celldata) * self.nx * self.ny * self.nz * 
                    np.dtype(config.sim_config.dtypes['float_or_double']).itemsize)


    def write_hdf5(self, pbar, G):
        """Writes snapshot file in HDF5 (.h5) format.

        Args:
            pbar: Progress bar class instance.
            G: FDTDGrid class describing a grid in a model.
        """

        f = h5py.File(self.filename, 'w')
        f.attrs['gprMax'] = __version__
        f.attrs['Title'] = G.title
        f.attrs['nx_ny_nz'] = (self.nx, self.ny, self.nz)
        f.attrs['dx_dy_dz'] = (self.dx * G.dx, self.dy * G.dy, self.dz * G.dz)
        f.attrs['time'] = self.time * G.dt

        if self.outputs['Ex']:
            f['Ex'] = self.Exsnap
            pbar.update(n=self.Exsnap.nbytes)
        if self.outputs['Ey']:
            f['Ey'] = self.Eysnap
            pbar.update(n=self.Eysnap.nbytes)
        if self.outputs['Ez']:
            f['Ez'] = self.Ezsnap
            pbar.update(n=self.Ezsnap.nbytes)
        if self.outputs['Hx']:
            f['Hx'] = self.Hxsnap
            pbar.update(n=self.Hxsnap.nbytes)
        if self.outputs['Hy']:
            f['Hy'] = self.Hysnap
            pbar.update(n=self.Hysnap.nbytes)
        if self.outputs['Hz']:
            f['Hz'] = self.Hzsnap
            pbar.update(n=self.Hzsnap.nbytes)

        f.close()


def htod_snapshot_array(G, queue=None):
    """Initialises arrays on compute device to store field data for snapshots.

    Args:
        G: FDTDGrid class describing a grid in a model.
        queue: pyopencl queue.

    Returns:
        snapE_dev, snapH_dev: float arrays of snapshot data on compute device.
    """

    # Get dimensions of largest requested snapshot
    for snap in G.snapshots:
        if snap.nx > Snapshot.nx_max:
            Snapshot.nx_max = snap.nx
        if snap.ny > Snapshot.ny_max:
            Snapshot.ny_max = snap.ny
        if snap.nz > Snapshot.nz_max:
            Snapshot.nz_max = snap.nz

    if config.sim_config.general['solver'] == 'cuda':
        # Blocks per grid - according to largest requested snapshot
        Snapshot.bpg = (int(np.ceil(((Snapshot.nx_max) *
                                     (Snapshot.ny_max) *
                                     (Snapshot.nz_max)) / Snapshot.tpb[0])), 1, 1)
    elif config.sim_config.general['solver'] == 'opencl':
        # Workgroup size - according to largest requested snapshot
        Snapshot.wgs = (int(np.ceil(((Snapshot.nx_max) * 
                                     (Snapshot.ny_max) * 
                                     (Snapshot.nz_max)))), 1, 1)

    # 4D arrays to store snapshots on GPU, e.g. snapEx(time, x, y, z);
    # if snapshots are not being stored on the GPU during the simulation then
    # they are copied back to the host after each iteration, hence numsnaps = 1
    numsnaps = 1 if config.get_model_config().device['snapsgpu2cpu'] else len(G.snapshots)
    snapEx = np.zeros((numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
                      dtype=config.sim_config.dtypes['float_or_double'])
    snapEy = np.zeros((numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
                      dtype=config.sim_config.dtypes['float_or_double'])
    snapEz = np.zeros((numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
                      dtype=config.sim_config.dtypes['float_or_double'])
    snapHx = np.zeros((numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
                      dtype=config.sim_config.dtypes['float_or_double'])
    snapHy = np.zeros((numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
                      dtype=config.sim_config.dtypes['float_or_double'])
    snapHz = np.zeros((numsnaps, Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max),
                      dtype=config.sim_config.dtypes['float_or_double'])

    # Copy arrays to compute device
    if config.sim_config.general['solver'] == 'cuda':
        import pycuda.gpuarray as gpuarray
        snapEx_dev = gpuarray.to_gpu(snapEx)
        snapEy_dev = gpuarray.to_gpu(snapEy)
        snapEz_dev = gpuarray.to_gpu(snapEz)
        snapHx_dev = gpuarray.to_gpu(snapHx)
        snapHy_dev = gpuarray.to_gpu(snapHy)
        snapHz_dev = gpuarray.to_gpu(snapHz)

    elif config.sim_config.general['solver'] == 'opencl':
        import pyopencl.array as clarray
        snapEx_dev = clarray.to_device(queue, snapEx)
        snapEy_dev = clarray.to_device(queue, snapEy)
        snapEz_dev = clarray.to_device(queue, snapEz)
        snapHx_dev = clarray.to_device(queue, snapHx)
        snapHy_dev = clarray.to_device(queue, snapHy)
        snapHz_dev = clarray.to_device(queue, snapHz)

    return snapEx_dev, snapEy_dev, snapEz_dev, snapHx_dev, snapHy_dev, snapHz_dev


def dtoh_snapshot_array(snapEx_dev, snapEy_dev, snapEz_dev, snapHx_dev, snapHy_dev, snapHz_dev, i, snap):
    """Copies snapshot array used on compute device back to snapshot objects and 
        store in format for Paraview.

    Args:
        snapE_dev, snapH_dev: float arrays of snapshot data from compute device.
        i: int for index of snapshot data on compute device array.
        snap: Snapshot class instance
    """

    snap.Exsnap = snapEx_dev[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Eysnap = snapEy_dev[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Ezsnap = snapEz_dev[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Hxsnap = snapHx_dev[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Hysnap = snapHy_dev[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Hzsnap = snapHz_dev[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
