# Copyright (C) 2015: The University of Edinburgh
#            Authors: Craig Warren and Antonis Giannopoulos
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
import numpy as np
from struct import pack

from gprMax.constants import floattype
from gprMax.utilities import roundvalue


class Snapshot:
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
        self.filename = filename

    def prepare_file(self, modelrun, numbermodelruns, G):
        """Prepares a VTK ImageData (.vti) file for a snapshot.
            
        Args:
            modelrun (int): Current model run number.
            numbermodelruns (int): Total number of model runs.
            G (class): Grid class instance - holds essential parameters describing the model.
        """
        
        # No Python 3 support for VTK at time of writing (03/2015)
        self.vtk_nx = self.xf - self.xs
        self.vtk_ny = self.yf - self.ys
        self.vtk_nz = self.zf - self.zs
        
        # Construct filename from user-supplied name and model run number
        if numbermodelruns == 1:
            self.filename = G.inputdirectory + self.filename + '.vti'
        else:
            self.filename = G.inputdirectory + self.filename + '_' + str(modelrun) + '.vti'
        
        # Calculate number of cells according to requested sampling
        self.vtk_xscells = roundvalue(self.xs / self.dx)
        self.vtk_xfcells = roundvalue(self.xf / self.dx)
        self.vtk_yscells = roundvalue(self.ys / self.dy)
        self.vtk_yfcells = roundvalue(self.yf / self.dz)
        self.vtk_zscells = roundvalue(self.zs / self.dz)
        self.vtk_zfcells = roundvalue(self.zf / self.dz)
        vtk_hfield_offset = 3 * np.dtype(floattype).itemsize * (self.vtk_xfcells - self.vtk_xscells) * (self.vtk_yfcells - self.vtk_yscells) * (self.vtk_zfcells - self.vtk_zscells) + np.dtype(np.uint32).itemsize
#        vtk_current_offset = 2 * vtk_hfield_offset

        self.filehandle = open(self.filename, 'wb')
        self.filehandle.write('<?xml version="1.0"?>\n'.encode('utf-8'))
        self.filehandle.write('<VTKFile type="ImageData" version="1.0" byte_order="{}">\n'.format(Snapshot.byteorder).encode('utf-8'))
        self.filehandle.write('<ImageData WholeExtent="{} {} {} {} {} {}" Origin="0 0 0" Spacing="{:.3} {:.3} {:.3}">\n'.format(self.vtk_xscells, self.vtk_xfcells, self.vtk_yscells, self.vtk_yfcells, self.vtk_zscells, self.vtk_zfcells, self.dx * G.dx, self.dy * G.dy, self.dz * G.dz).encode('utf-8'))
        self.filehandle.write('<Piece Extent="{} {} {} {} {} {}">\n'.format(self.vtk_xscells, self.vtk_xfcells, self.vtk_yscells, self.vtk_yfcells, self.vtk_zscells, self.vtk_zfcells).encode('utf-8'))
        self.filehandle.write('<CellData Vectors="E-field H-field">\n'.encode('utf-8'))
#        self.filehandle.write('<CellData Vectors="E-field H-field Current">\n'.encode('utf-8'))
        self.filehandle.write('<DataArray type="{}" Name="E-field" NumberOfComponents="3" format="appended" offset="0" />\n'.format(Snapshot.floatname).encode('utf-8'))
        self.filehandle.write('<DataArray type="{}" Name="H-field" NumberOfComponents="3" format="appended" offset="{}" />\n'.format(Snapshot.floatname, vtk_hfield_offset).encode('utf-8'))
#        self.filehandle.write('<DataArray type="{}" Name="Current" NumberOfComponents="3" format="appended" offset="{}" />\n'.format(Snapshot.floatname, vtk_current_offset).encode('utf-8'))
        self.filehandle.write('</CellData>\n</Piece>\n</ImageData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))

    def write_snapshot(self, Ex, Ey, Ez, Hx, Hy, Hz, G):
        """Writes electric and magnetic field values to VTK ImageData (.vti) file.
            
        Args:
            Ex, Ey, Ez, Hx, Hy, Hz (memory view): Electric and magnetic field values.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        datasize = 3 * np.dtype(floattype).itemsize * (self.vtk_xfcells - self.vtk_xscells) * (self.vtk_yfcells - self.vtk_yscells) * (self.vtk_zfcells - self.vtk_zscells)
        # Write number of bytes of appended data as UInt32
        self.filehandle.write(pack('I', datasize))
        for k in range(self.zs, self.zf, self.dz):
            for j in range(self.ys, self.yf, self.dy):
                for i in range(self.xs, self.xf, self.dx):
                    # The electric field component value at a point comes from average of the 4 electric field component values in that cell
                    self.filehandle.write(pack(Snapshot.floatstring, (Ex[i, j, k] + Ex[i, j + 1, k] + Ex[i, j, k + 1] + Ex[i, j + 1, k + 1]) / 4))
                    self.filehandle.write(pack(Snapshot.floatstring, (Ey[i, j, k] + Ey[i + 1, j, k] + Ey[i, j, k + 1] + Ey[i + 1, j, k + 1]) / 4))
                    self.filehandle.write(pack(Snapshot.floatstring, (Ez[i, j, k] + Ez[i + 1, j, k] + Ez[i, j + 1, k] + Ez[i + 1, j + 1, k]) / 4))

        self.filehandle.write(pack('I', datasize))
        for k in range(self.zs, self.zf, self.dz):
            for j in range(self.ys, self.yf, self.dy):
                for i in range(self.xs, self.xf, self.dx):
                    # The magnetic field component value at a point comes from average of 2 magnetic field component values in that cell and the following cell
                    self.filehandle.write(pack(Snapshot.floatstring, (Hx[i, j, k] + Hx[i + 1, j, k]) / 2))
                    self.filehandle.write(pack(Snapshot.floatstring, (Hy[i, j, k] + Hy[i, j + 1, k]) / 2))
                    self.filehandle.write(pack(Snapshot.floatstring, (Hz[i, j, k] + Hz[i, j, k + 1]) / 2))

    #    self.filehandle.write(pack('I', datasize))
    #    for k in range(self.zs, self.zf, self.dz):
    #        for j in range(self.ys, self.yf, self.dy):
    #            for i in range(self.xs, self.xf, self.dx):
    #                self.filehandle.write(pack(Snapshot.floatstring, Ix[i, j, k]))
    #                self.filehandle.write(pack(Snapshot.floatstring, Iy[i, j, k]))
    #                self.filehandle.write(pack(Snapshot.floatstring, Iz[i, j, k]))

        self.filehandle.write('\n</AppendedData>\n</VTKFile>'.encode('utf-8'))
        self.filehandle.close()

