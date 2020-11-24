# Copyright (C) 2015-2020: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

import sys
from struct import pack

import h5py
import numpy as np

import gprMax.config as config

from ._version import __version__
from .cython.snapshots import calculate_snapshot_fields
from .utilities import round_value


class Snapshot:
    """Snapshots of the electric and magnetic field values."""

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
                 dx=None, dy=None, dz=None, time=None, filename=None, fileext=None):
        """
        Args:
            xs, xf, ys, yf, zs, zf (int): Extent of the volume in cells.
            dx, dy, dz (int): Spatial discretisation in cells.
            time (int): Iteration number to take the snapshot on.
            filename (str): Filename to save to.
            fileext (str): File extension.
        """

        self.fileext = fileext
        self.filename = filename
        self.time = time
        # Select a set of field outputs - electric (Ex, Ey, Ez) 
        # and/or magnetic (Hx, Hy, Hz). Only affects field outputs written to
        # file, i.e. ALL field outputs are still stored in memory
        self.fieldoutputs = {'electric': True, 'magnetic': True}
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
        self.ncells = self.nx * self.ny * self.nz
        self.datasizefield = (3 * np.dtype(config.sim_config.dtypes['float_or_double']).itemsize
                                * self.ncells)
        self.vtkdatawritesize = ((self.fieldoutputs['electric'] +
                                  self.fieldoutputs['magnetic']) *
                                  self.datasizefield + (self.fieldoutputs['electric'] +
                                  self.fieldoutputs['magnetic']) * np.dtype(np.uint32).itemsize)

    def store(self, G):
        """Store (in memory) electric and magnetic field values for snapshot.

        Args:
            G (FDTDGrid): Parameters describing a grid in a model.
        """

        # Memory views of field arrays to dimensions required for the snapshot
        Exslice = np.ascontiguousarray(G.Ex[self.sx, self.sy, self.sz])
        Eyslice = np.ascontiguousarray(G.Ey[self.sx, self.sy, self.sz])
        Ezslice = np.ascontiguousarray(G.Ez[self.sx, self.sy, self.sz])
        Hxslice = np.ascontiguousarray(G.Hx[self.sx, self.sy, self.sz])
        Hyslice = np.ascontiguousarray(G.Hy[self.sx, self.sy, self.sz])
        Hzslice = np.ascontiguousarray(G.Hz[self.sx, self.sy, self.sz])

        # Create arrays to hold the field data for snapshot
        self.Exsnap = np.zeros((self.nx, self.ny, self.nz), dtype=config.sim_config.dtypes['float_or_double'])
        self.Eysnap = np.zeros((self.nx, self.ny, self.nz), dtype=config.sim_config.dtypes['float_or_double'])
        self.Ezsnap = np.zeros((self.nx, self.ny, self.nz), dtype=config.sim_config.dtypes['float_or_double'])
        self.Hxsnap = np.zeros((self.nx, self.ny, self.nz), dtype=config.sim_config.dtypes['float_or_double'])
        self.Hysnap = np.zeros((self.nx, self.ny, self.nz), dtype=config.sim_config.dtypes['float_or_double'])
        self.Hzsnap = np.zeros((self.nx, self.ny, self.nz), dtype=config.sim_config.dtypes['float_or_double'])

        # Calculate field values at points (comes from averaging field components in cells)
        calculate_snapshot_fields(
            self.nx,
            self.ny,
            self.nz,
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
            self.Hzsnap)

    def write_file(self, pbar, G):
        """Write snapshot file either as VTK ImageData (.vti) format 
            or HDF5 format (.h5) files

        Args:
            pbar (class): Progress bar class instance.
            G (FDTDGrid): Parameters describing a grid in a model.
        """

        if self.fileext == '.vti':
            self.write_vtk_imagedata(pbar, G)
        elif self.fileext == '.h5':
            self.write_hdf5(pbar, G)

    def write_vtk_imagedata(self, pbar, G):
        """Write snapshot file in VTK ImageData (.vti) format.

            N.B. No Python 3 support for VTK at time of writing (03/2015)

        Args:
            pbar (class): Progress bar class instance.
            G (FDTDGrid): Parameters describing a grid in a model.
        """

        hfield_offset = (3 * np.dtype(config.sim_config.dtypes['float_or_double']).itemsize
                         * self.ncells + np.dtype(np.uint32).itemsize)

        f = open(self.filename, 'wb')
        f.write('<?xml version="1.0"?>\n'.encode('utf-8'))
        f.write(f'<VTKFile type="ImageData" version="1.0" byte_order="{config.sim_config.vtk_byteorder}">\n'.encode('utf-8'))
        f.write(f'<ImageData WholeExtent="{self.xs} {round_value(self.xf / self.dx)} {self.ys} {round_value(self.yf / self.dy)} {self.zs} {round_value(self.zf / self.dz)}" Origin="0 0 0" Spacing="{self.dx * G.dx:.3} {self.dy * G.dy:.3} {self.dz * G.dz:.3}">\n'.encode('utf-8'))
        f.write(f'<Piece Extent="{self.xs} {round_value(self.xf / self.dx)} {self.ys} {round_value(self.yf / self.dy)} {self.zs} {round_value(self.zf / self.dz)}">\n'.encode('utf-8'))

        if self.fieldoutputs['electric'] and self.fieldoutputs['magnetic']:
            f.write('<CellData Vectors="E-field H-field">\n'.encode('utf-8'))
            f.write(f"""<DataArray type="{config.sim_config.dtypes['vtk_float']}" Name="E-field" NumberOfComponents="3" format="appended" offset="0" />\n""".encode('utf-8'))
            f.write(f"""<DataArray type="{config.sim_config.dtypes['vtk_float']}" Name="H-field" NumberOfComponents="3" format="appended" offset="{hfield_offset}" />\n""".encode('utf-8'))
        elif self.fieldoutputs['electric']:
            f.write('<CellData Vectors="E-field">\n'.encode('utf-8'))
            f.write(f"""<DataArray type="{config.sim_config.dtypes['vtk_float']}" Name="E-field" NumberOfComponents="3" format="appended" offset="0" />\n""".encode('utf-8'))
        elif self.fieldoutputs['magnetic']:
            f.write('<CellData Vectors="H-field">\n'.encode('utf-8'))
            f.write(f"""<DataArray type="{config.sim_config.dtypes['vtk_float']}" Name="H-field" NumberOfComponents="3" format="appended" offset="0" />\n""".encode('utf-8'))

        f.write('</CellData>\n</Piece>\n</ImageData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))

        if self.fieldoutputs['electric']:
            # Write number of bytes of appended data as UInt32
            f.write(pack('I', self.datasizefield))
            pbar.update(n=4)
            # Convert to format for Paraview
            electric = np.stack((self.Exsnap, self.Eysnap, self.Ezsnap)).reshape(-1, order='F')
            electric.tofile(f)
            pbar.update(n=self.datasizefield)

        if self.fieldoutputs['magnetic']:
            # Write number of bytes of appended data as UInt32
            f.write(pack('I', self.datasizefield))
            pbar.update(n=4)
            magnetic = np.stack((self.Hxsnap, self.Hysnap, self.Hzsnap)).reshape(-1, order='F')
            magnetic.tofile(f)
            pbar.update(n=self.datasizefield)

        f.write('\n</AppendedData>\n</VTKFile>'.encode('utf-8'))
        f.close()

    def write_hdf5(self, pbar, G):
        """Write snapshot file in HDF5 (.h5) format.

        Args:
            pbar (class): Progress bar class instance.
            G (FDTDGrid): Parameters describing a grid in a model.
        """

        f = h5py.File(self.filename, 'w')
        f.attrs['gprMax'] = __version__
        f.attrs['Title'] = G.title
        f.attrs['nx_ny_nz'] = (self.nx, self.ny, self.nz)
        f.attrs['dx_dy_dz'] = (self.dx * G.dx, self.dy * G.dy, self.dz * G.dz)
        f.attrs['time'] = self.time * G.dt

        if self.fieldoutputs['electric']:
            f['Ex'] = self.Exsnap
            f['Ey'] = self.Eysnap
            f['Ez'] = self.Ezsnap
            pbar.update(n=self.datasizefield)

        if self.fieldoutputs['magnetic']:
            f['Hx'] = self.Hxsnap
            f['Hy'] = self.Hysnap
            f['Hz'] = self.Hzsnap
            pbar.update(n=self.datasizefield)

        f.close()


def htod_snapshot_array(G):
    """Initialise array on GPU for to store field data for snapshots.

    Args:
        G (FDTDGrid): Parameters describing a grid in a model.

    Returns:
        snapE_gpu, snapH_gpu (float): numpy arrays of snapshot data on GPU.
    """

    import pycuda.gpuarray as gpuarray

    # Get dimensions of largest requested snapshot
    for snap in G.snapshots:
        if snap.nx > Snapshot.nx_max:
            Snapshot.nx_max = snap.nx
        if snap.ny > Snapshot.ny_max:
            Snapshot.ny_max = snap.ny
        if snap.nz > Snapshot.nz_max:
            Snapshot.nz_max = snap.nz

    # GPU - blocks per grid - according to largest requested snapshot
    Snapshot.bpg = (int(np.ceil(((Snapshot.nx_max) *
                                 (Snapshot.ny_max) *
                                 (Snapshot.nz_max)) / Snapshot.tpb[0])), 1, 1)

    # 4D arrays to store snapshots on GPU, e.g. snapEx(time, x, y, z);
    # if snapshots are not being stored on the GPU during the simulation then
    # they are copied back to the host after each iteration, hence numsnaps = 1
    numsnaps = 1 if config.get_model_config().cuda['snapsgpu2cpu'] else len(G.snapshots)
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

    # Copy arrays to GPU
    snapEx_gpu = gpuarray.to_gpu(snapEx)
    snapEy_gpu = gpuarray.to_gpu(snapEy)
    snapEz_gpu = gpuarray.to_gpu(snapEz)
    snapHx_gpu = gpuarray.to_gpu(snapHx)
    snapHy_gpu = gpuarray.to_gpu(snapHy)
    snapHz_gpu = gpuarray.to_gpu(snapHz)

    return snapEx_gpu, snapEy_gpu, snapEz_gpu, snapHx_gpu, snapHy_gpu, snapHz_gpu


def dtoh_snapshot_array(snapEx_gpu, snapEy_gpu, snapEz_gpu, snapHx_gpu, snapHy_gpu, snapHz_gpu, i, snap):
    """Copy snapshot array used on GPU back to snapshot objects and store in format for Paraview.

    Args:
        snapE_gpu, snapH_gpu (float): numpy arrays of snapshot data from GPU.
        i (int): index for snapshot data on GPU array.
        snap (class): Snapshot class instance
    """

    snap.Exsnap = snapEx_gpu[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Eysnap = snapEy_gpu[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Ezsnap = snapEz_gpu[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Hxsnap = snapHx_gpu[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Hysnap = snapHy_gpu[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
    snap.Hzsnap = snapHz_gpu[i, snap.xs:snap.xf, snap.ys:snap.yf, snap.zs:snap.zf]
