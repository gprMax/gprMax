# Copyright (C) 2015-2018: The University of Edinburgh
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

import os
import sys
from struct import pack

import numpy as np

from gprMax.constants import floattype
from gprMax.grid import Ix
from gprMax.grid import Iy
from gprMax.grid import Iz
from gprMax.utilities import round_value


class Snapshot(object):
    """Snapshots of the electric and magnetic field values."""

    # Set string for byte order
    if sys.byteorder == 'little':
        byteorder = 'LittleEndian'
    else:
        byteorder = 'BigEndian'

    # Set format text and string depending on float type
    if np.dtype(floattype).name == 'float32':
        floatname = 'Float32'
        floatstring = 'f'
    elif np.dtype(floattype).name == 'float64':
        floatname = 'Float64'
        floatstring = 'd'

    def __init__(self, xs=None, ys=None, zs=None, xf=None, yf=None, zf=None, dx=None, dy=None, dz=None, time=None, filename=None):
        """
        Args:
            xs, xf, ys, yf, zs, zf (float): Extent of the volume.
            dx, dy, dz (float): Spatial discretisation.
            time (int): Iteration number to take the snapshot on.
            filename (str): Filename to save to.
        """

        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.xf = xf
        self.yf = yf
        self.zf = zf
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.time = time
        self.basefilename = filename

    def prepare_vtk_imagedata(self, appendmodelnumber, G):
        """Prepares a VTK ImageData (.vti) file for a snapshot.

        Args:
            appendmodelnumber (str): Text to append to filename.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        # No Python 3 support for VTK at time of writing (03/2015)
        self.vtk_nx = self.xf - self.xs
        self.vtk_ny = self.yf - self.ys
        self.vtk_nz = self.zf - self.zs

        # Create directory and construct filename from user-supplied name and model run number
        snapshotdir = os.path.join(G.inputdirectory, os.path.splitext(G.inputfilename)[0] + '_snaps' + appendmodelnumber)
        if not os.path.exists(snapshotdir):
            os.mkdir(snapshotdir)
        self.filename = os.path.abspath(os.path.join(snapshotdir, self.basefilename + '.vti'))

        # Calculate number of cells according to requested sampling
        self.vtk_xscells = round_value(self.xs / self.dx)
        self.vtk_xfcells = round_value(self.xf / self.dx)
        self.vtk_yscells = round_value(self.ys / self.dy)
        self.vtk_yfcells = round_value(self.yf / self.dy)
        self.vtk_zscells = round_value(self.zs / self.dz)
        self.vtk_zfcells = round_value(self.zf / self.dz)
        vtk_hfield_offset = 3 * np.dtype(floattype).itemsize * (self.vtk_xfcells - self.vtk_xscells) * (self.vtk_yfcells - self.vtk_yscells) * (self.vtk_zfcells - self.vtk_zscells) + np.dtype(np.uint32).itemsize
        vtk_current_offset = 2 * vtk_hfield_offset

        self.filehandle = open(self.filename, 'wb')
        self.filehandle.write('<?xml version="1.0"?>\n'.encode('utf-8'))
        self.filehandle.write('<VTKFile type="ImageData" version="1.0" byte_order="{}">\n'.format(Snapshot.byteorder).encode('utf-8'))
        self.filehandle.write('<ImageData WholeExtent="{} {} {} {} {} {}" Origin="0 0 0" Spacing="{:.3} {:.3} {:.3}">\n'.format(self.vtk_xscells, self.vtk_xfcells, self.vtk_yscells, self.vtk_yfcells, self.vtk_zscells, self.vtk_zfcells, self.dx * G.dx, self.dy * G.dy, self.dz * G.dz).encode('utf-8'))
        self.filehandle.write('<Piece Extent="{} {} {} {} {} {}">\n'.format(self.vtk_xscells, self.vtk_xfcells, self.vtk_yscells, self.vtk_yfcells, self.vtk_zscells, self.vtk_zfcells).encode('utf-8'))
        self.filehandle.write('<CellData Vectors="E-field H-field Current">\n'.encode('utf-8'))
        self.filehandle.write('<DataArray type="{}" Name="E-field" NumberOfComponents="3" format="appended" offset="0" />\n'.format(Snapshot.floatname).encode('utf-8'))
        self.filehandle.write('<DataArray type="{}" Name="H-field" NumberOfComponents="3" format="appended" offset="{}" />\n'.format(Snapshot.floatname, vtk_hfield_offset).encode('utf-8'))
        self.filehandle.write('<DataArray type="{}" Name="Current" NumberOfComponents="3" format="appended" offset="{}" />\n'.format(Snapshot.floatname, vtk_current_offset).encode('utf-8'))
        self.filehandle.write('</CellData>\n</Piece>\n</ImageData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))
        self.filehandle.close()

    def write_vtk_imagedata(self, Ex, Ey, Ez, Hx, Hy, Hz, G, pbar):
        """Writes electric and magnetic field values to VTK ImageData (.vti) file.

        Args:
            Ex, Ey, Ez, Hx, Hy, Hz (memory view): Electric and magnetic field values.
            G (class): Grid class instance - holds essential parameters describing the model.
            pbar (class): Progress bar class instance.
        """

        self.filehandle = open(self.filename, 'ab')

        datasize = 3 * np.dtype(floattype).itemsize * (self.vtk_xfcells - self.vtk_xscells) * (self.vtk_yfcells - self.vtk_yscells) * (self.vtk_zfcells - self.vtk_zscells)
        # Write number of bytes of appended data as UInt32
        self.filehandle.write(pack('I', datasize))

        # The electric field component value at a point comes from average of the 4 electric field component values in that cell
        # Because of how this is calculated, need to eliminate the last value in each dimension
        Ex = (Ex[:-1,:-1,:-1] + Ex[:-1,1:,:-1] + Ex[:-1,:-1,1:] +Ex[:-1,1:,1:])/4
        Ey = (Ey[:-1,:-1,:-1] + Ey[1:,:-1,:-1] + Ey[:-1,:-1,1:] +Ey[1:,:-1,1:])/4
        Ez = (Ez[:-1,:-1,:-1] + Ez[1:,:-1,:-1] + Ez[:-1,1:,:-1] +Ez[1:,1:,:-1])/4
        E = np.stack((Ex.transpose(2,1,0),Ey.transpose(2,1,0),Ez.transpose(2,1,0)),axis=-1)
        self.filehandle.write(E.tobytes()) # send the data to the file

        # Have to generate the I values before the H values as I depends on H
        # If the code stayed in the same order, it would require making a copy of H values
        # Note that the I values are save later, making them in the same location in the file as before
        Ixx = np.zeros((Hx.shape[0]-1,Hx.shape[1]-1,Hx.shape[2]-1),dtype=Hx.dtype)
        Iyy = np.zeros((Hy.shape[0]-1,Hy.shape[1]-1,Hy.shape[2]-1),dtype=Hy.dtype)
        Izz = np.zeros((Hz.shape[0]-1,Hz.shape[1]-1,Hz.shape[2]-1),dtype=Hz.dtype)
        Ixx = (G.dy * (Hy[1:, 1:, :-1] - Hy[1:,:1,:1]) + G.dz * (Hz[1:,:1,:1] - Hz[1:, :-1, 1:]))
        Iyy = (G.dx * (Hx[1:, 1:, 1:] - Hx[1:, 1:, :-1]) + G.dz * (Hz[:-1, 1:, 1:] - Hz[1:, 1:, 1:]))
        Izz = (G.dx * (Hx[1:, :-1, 1:] - Hx[1:, 1:, 1:]) + G.dy * (Hy[1:, 1:, 1:] - Hy[:-1, 1:, 1:]))
        I = np.stack((Ixx.transpose(2,1,0),Iyy.transpose(2,1,0),Izz.transpose(2,1,0)),axis=-1)

        # The magnetic field component value at a point comes from average
        # of 2 magnetic field component values in that cell and the following cell
        self.filehandle.write(pack('I', datasize))
        Hx = (Hx[:-1, :-1, :-1] + Hx[1:, :-1, :-1]) / 2
        Hy = (Hy[:-1, :-1, :-1] + Hy[:-1, 1:, :-1]) / 2
        Hz = (Hz[:-1, :-1, :-1] + Hz[:-1, :-1, 1:]) / 2
        H = np.stack((Hx.transpose(2,1,0),Hy.transpose(2,1,0),Hz.transpose(2,1,0)),axis=-1)
        self.filehandle.write(H.tobytes()) # send the data to the file

        # Save the I values
        self.filehandle.write(pack('I', datasize))       
        self.filehandle.write(I.tobytes()) # send the data to the file

        self.filehandle.write('\n</AppendedData>\n</VTKFile>'.encode('utf-8'))
        self.filehandle.close()
