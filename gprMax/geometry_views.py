# Copyright (C) 2015-2016: The University of Edinburgh
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
import numpy as np
from struct import pack

from gprMax.utilities import roundvalue


class GeometryView:
    """Views of the geometry of the model."""
    
    if sys.byteorder == 'little':
        byteorder = 'LittleEndian'
    else:
        byteorder = 'BigEndian'

    def __init__(self, xs=None, ys=None, zs=None, xf=None, yf=None, zf=None, dx=None, dy=None, dz=None, filename=None, type=None):
        """
        Args:
            xs, xf, ys, yf, zs, zf (float): Extent of the volume.
            dx, dy, dz (float): Spatial discretisation.
            filename (str): Filename to save to.
            type (str): Either 'n' for a per cell geometry view, or 'f' for a per cell edge geometry view.
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
        self.filename = filename
        self.type = type

    def write_file(self, modelrun, numbermodelruns, G):
        """Writes the geometry information to a VTK file. Either ImageData (.vti) for a per cell geometry view, or PolygonalData (.vtp) for a per cell edge geometry view.
            
        Args:
            modelrun (int): Current model run number.
            numbermodelruns (int): Total number of model runs.
            G (class): Grid class instance - holds essential parameters describing the model.
        """
        
        # Construct filename from user-supplied name and model run number
        if numbermodelruns == 1:
            self.filename = G.inputdirectory + self.filename
        else:
            self.filename = G.inputdirectory + self.filename + '_' + str(modelrun)
        
        # No Python 3 support for VTK at time of writing (03/2015)
        self.vtk_nx = self.xf - self.xs
        self.vtk_ny = self.yf - self.ys
        self.vtk_nz = self.zf - self.zs
        
        if self.type == 'n':
            self.filename += '.vti'
            
            # Calculate number of cells according to requested sampling
            self.vtk_xscells = roundvalue(self.xs / self.dx)
            self.vtk_xfcells = roundvalue(self.xf / self.dx)
            self.vtk_yscells = roundvalue(self.ys / self.dy)
            self.vtk_yfcells = roundvalue(self.yf / self.dy)
            self.vtk_zscells = roundvalue(self.zs / self.dz)
            self.vtk_zfcells = roundvalue(self.zf / self.dz)
            with open(self.filename, 'wb') as f:
                f.write('<?xml version="1.0"?>\n'.encode('utf-8'))
                f.write('<VTKFile type="ImageData" version="1.0" byte_order="{}">\n'.format(GeometryView.byteorder).encode('utf-8'))
                f.write('<ImageData WholeExtent="{} {} {} {} {} {}" Origin="0 0 0" Spacing="{:.3} {:.3} {:.3}">\n'.format(self.vtk_xscells, self.vtk_xfcells, self.vtk_yscells, self.vtk_yfcells, self.vtk_zscells, self.vtk_zfcells, self.dx * G.dx, self.dy * G.dy, self.dz * G.dz).encode('utf-8'))
                f.write('<Piece Extent="{} {} {} {} {} {}">\n'.format(self.vtk_xscells, self.vtk_xfcells, self.vtk_yscells, self.vtk_yfcells, self.vtk_zscells, self.vtk_zfcells).encode('utf-8'))
                f.write('<CellData Scalars="Material">\n'.encode('utf-8'))
                f.write('<DataArray type="UInt32" Name="Material" format="appended" offset="0" />\n'.encode('utf-8'))
                f.write('</CellData>\n</Piece>\n</ImageData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))
                
                # Calculate number of bytes of appended data section
                datasize = roundvalue(np.dtype(np.uint32).itemsize * (self.vtk_nx / self.dx) * (self.vtk_ny / self.dy) * (self.vtk_nz / self.dz))
                # Write number of bytes of appended data as UInt32
                f.write(pack('I', datasize))
                for k in range(self.zs, self.zf, self.dz):
                    for j in range(self.ys, self.yf, self.dy):
                        for i in range(self.xs, self.xf, self.dx):
                            f.write(pack('I', G.solid[i, j, k]))
                f.write('\n</AppendedData>\n</VTKFile>'.encode('utf-8'))
                
                # Write gprMax specific information which relates material name to material numeric identifier
                f.write('\n\n<gprMax>\n'.encode('utf-8'))
                for material in G.materials:
                    f.write('<Material name="{}">{}</Material>\n'.format(material.ID, material.numID).encode('utf-8'))
                f.write('</gprMax>\n'.encode('utf-8'))

        elif self.type == 'f':
            self.filename += '.vtp'
            
            vtk_numpoints = (self.vtk_nx + 1) * (self.vtk_ny + 1) * (self.vtk_nz + 1)
            vtk_numpoint_components = 3
            vtk_numlines = 2 * self.vtk_nx * self.vtk_ny + 2 * self.vtk_ny * self.vtk_nz + 2 * self.vtk_nx * self.vtk_nz + 3 * self.vtk_nx * self.vtk_ny * self.vtk_nz + self.vtk_nx + self.vtk_ny + self.vtk_nz
            vtk_numline_components  = 2;
            vtk_connectivity_offset = (vtk_numpoints * vtk_numpoint_components * np.dtype(np.float32).itemsize) + np.dtype(np.uint32).itemsize
            vtk_offsets_offset = vtk_connectivity_offset + (vtk_numlines * vtk_numline_components * np.dtype(np.uint32).itemsize) + np.dtype(np.uint32).itemsize
            vtk_id_offset = vtk_offsets_offset + (vtk_numlines * np.dtype(np.uint32).itemsize) + np.dtype(np.uint32).itemsize
            vtk_offsets_size = vtk_numlines
            
            with open(self.filename, 'wb') as f:
                f.write('<?xml version="1.0"?>\n'.encode('utf-8'))
                f.write('<VTKFile type="PolyData" version="1.0" byte_order="{}">\n'.format(GeometryView.byteorder).encode('utf-8'))
                f.write('<PolyData>\n<Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="{}" NumberOfStrips="0" NumberOfPolys="0">\n'.format(vtk_numpoints, vtk_numlines).encode('utf-8'))
                f.write('<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="0" />\n</Points>\n'.encode('utf-8'))
                f.write('<Lines>\n<DataArray type="UInt32" Name="connectivity" format="appended" offset="{}" />\n'.format(vtk_connectivity_offset).encode('utf-8'))
                f.write('<DataArray type="UInt32" Name="offsets" format="appended" offset="{}" />\n</Lines>\n'.format(vtk_offsets_offset).encode('utf-8'))
                f.write('<CellData Scalars="Material">\n<DataArray type="UInt32" Name="Material" format="appended" offset="{}" />\n</CellData>\n'.format(vtk_id_offset).encode('utf-8'))
                f.write('</Piece>\n</PolyData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))

                # Write points
                datasize = np.dtype(np.float32).itemsize * vtk_numpoints * vtk_numpoint_components
                f.write(pack('I', datasize))
                for i in range(self.xs, self.xf + 1):
                    for j in range(self.ys, self.yf + 1):
                        for k in range(self.zs, self.zf + 1):
                                f.write(pack('fff', i * G.dx, j * G.dy, k * G.dz))

                # Write cell type (line) connectivity for x components
                datasize = np.dtype(np.uint32).itemsize * vtk_numlines * vtk_numline_components
                f.write(pack('I', datasize))
                vtk_x2 = (self.vtk_ny + 1) * (self.vtk_nz + 1)
                for vtk_x1 in range(self.vtk_nx * (self.vtk_ny + 1) * (self.vtk_nz + 1)):
                    f.write(pack('II', vtk_x1, vtk_x2))
                    # print('x {} {}'.format(vtk_x1, vtk_x2))
                    vtk_x2 += 1

                # Write cell type (line) connectivity for y components
                vtk_ycnt1 = 1
                vtk_ycnt2 = 0
                for vtk_y1 in range((self.vtk_nx + 1) * (self.vtk_ny + 1) * (self.vtk_nz + 1)):
                    if vtk_y1 >= (vtk_ycnt1 * (self.vtk_ny + 1) * (self.vtk_nz + 1)) - (self.vtk_nz + 1) and vtk_y1 < vtk_ycnt1 * (self.vtk_ny + 1) * (self.vtk_nz + 1):
                        vtk_ycnt2 += 1
                    else:
                        vtk_y2 = vtk_y1 + self.vtk_nz + 1
                        f.write(pack('II', vtk_y1, vtk_y2))
                        # print('y {} {}'.format(vtk_y1, vtk_y2))
                    if vtk_ycnt2 == self.vtk_nz + 1:
                        vtk_ycnt1 += 1
                        vtk_ycnt2 = 0

                # Write cell type (line) connectivity for z components
                vtk_zcnt = self.vtk_nz
                for vtk_z1 in range((self.vtk_nx + 1) * (self.vtk_ny + 1) * self.vtk_nz + (self.vtk_nx + 1) * (self.vtk_ny + 1)):
                    if vtk_z1 != vtk_zcnt:
                        vtk_z2 = vtk_z1 + 1
                        f.write(pack('II', vtk_z1, vtk_z2))
                        # print('z {} {}'.format(vtk_z1, vtk_z2))
                    else:
                        vtk_zcnt += self.vtk_nz + 1

                # Write cell type (line) offsets
                vtk_cell_pts = 2
                datasize = np.dtype(np.uint32).itemsize * vtk_offsets_size
                f.write(pack('I', datasize))
                for vtk_offsets in range(vtk_cell_pts, (vtk_numline_components * vtk_numlines) + vtk_cell_pts, vtk_cell_pts):
                    f.write(pack('I', vtk_offsets))

                # Write Ex, Ey, Ez values from ID array
                datasize = np.dtype(np.uint32).itemsize * vtk_numlines
                f.write(pack('I', datasize))
                for i in range(self.xs, self.xf):
                    for j in range(self.ys, self.yf + 1):
                        for k in range(self.zs, self.zf + 1):
                            f.write(pack('I', G.ID[0, i, j, k]))
                
                for i in range(self.xs, self.xf + 1):
                    for j in range(self.ys, self.yf):
                        for k in range(self.zs, self.zf + 1):
                            f.write(pack('I', G.ID[1, i, j, k]))

                for i in range(self.xs, self.xf + 1):
                    for j in range(self.ys, self.yf + 1):
                        for k in range(self.zs, self.zf):
                            f.write(pack('I', G.ID[2, i, j, k]))

                f.write('\n</AppendedData>\n</VTKFile>'.encode('utf-8'))

                # Write gprMax specific information which relates material name to material numeric identifier
                f.write('\n\n<gprMax>\n'.encode('utf-8'))
                for material in G.materials:
                    f.write('<Material name="{}">{}</Material>\n'.format(material.ID, material.numID).encode('utf-8'))
                f.write('</gprMax>\n'.encode('utf-8'))


