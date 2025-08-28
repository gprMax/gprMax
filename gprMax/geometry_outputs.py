# Copyright (C) 2015-2023: The University of Edinburgh
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

import h5py
import numpy as np
from struct import pack

from gprMax._version import __version__
from gprMax.geometry_outputs_ext import define_normal_geometry
from gprMax.geometry_outputs_ext import define_fine_geometry
from gprMax.utilities import round_value


class GeometryView(object):
    """Views of the geometry of the model."""

    if sys.byteorder == 'little':
        byteorder = 'LittleEndian'
    else:
        byteorder = 'BigEndian'

    def __init__(self, xs=None, ys=None, zs=None, xf=None, yf=None, zf=None, dx=None, dy=None, dz=None, filename=None, fileext=None):
        """
        Args:
            xs, xf, ys, yf, zs, zf (int): Extent of the volume in cells.
            dx, dy, dz (int): Spatial discretisation in cells.
            filename (str): Filename to save to.
            fileext (str): File extension of VTK file - either '.vti' for a per cell
                    geometry view, or '.vtp' for a per cell edge geometry view.
        """

        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.xf = xf
        self.yf = yf
        self.zf = zf
        self.nx = self.xf - self.xs
        self.ny = self.yf - self.ys
        self.nz = self.zf - self.zs
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.basefilename = filename
        self.fileext = fileext

        if self.fileext == '.vti':
            # Calculate number of cells according to requested sampling for geometry view
            self.vtk_xscells = round_value(self.xs / self.dx)
            self.vtk_xfcells = round_value(self.xf / self.dx)
            self.vtk_yscells = round_value(self.ys / self.dy)
            self.vtk_yfcells = round_value(self.yf / self.dy)
            self.vtk_zscells = round_value(self.zs / self.dz)
            self.vtk_zfcells = round_value(self.zf / self.dz)
            self.vtk_nxcells = round_value(self.nx / self.dx)
            self.vtk_nycells = round_value(self.ny / self.dy)
            self.vtk_nzcells = round_value(self.nz / self.dz)
            self.vtk_ncells = self.vtk_nxcells * self.vtk_nycells * self.vtk_nzcells
            self.datawritesize = (np.dtype(np.uint32).itemsize * self.vtk_ncells
                                  + 2 * np.dtype(np.int8).itemsize * self.vtk_ncells
                                  + 3 * np.dtype(np.uint32).itemsize)

        elif self.fileext == '.vtp':
            self.vtk_numpoints = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
            self.vtk_numpoint_components = 3
            self.vtk_numline_components = 2
            self.vtk_nxlines = self.nx * (self.ny + 1) * (self.nz + 1)
            self.vtk_nylines = self.ny * (self.nx + 1) * (self.nz + 1)
            self.vtk_nzlines = self.nz * (self.nx + 1) * (self.ny + 1)
            self.vtk_numlines = self.vtk_nxlines + self.vtk_nylines + self.vtk_nzlines
            self.vtk_connectivity_offset = round_value((self.vtk_numpoints
                                                        * self.vtk_numpoint_components
                                                        * np.dtype(np.float32).itemsize)
                                                       + np.dtype(np.uint32).itemsize)
            self.vtk_offsets_offset = round_value(self.vtk_connectivity_offset
                                                  + (self.vtk_numlines * self.vtk_numline_components * np.dtype(np.uint32).itemsize)
                                                  + np.dtype(np.uint32).itemsize)
            self.vtk_materials_offset = round_value(self.vtk_offsets_offset
                                                    + (self.vtk_numlines * np.dtype(np.uint32).itemsize)
                                                    + np.dtype(np.uint32).itemsize)
            vtk_cell_offsets = ((self.vtk_numline_components * self.vtk_numlines)
                                + self.vtk_numline_components - self.vtk_numline_components - 1) // self.vtk_numline_components + 1
            self.datawritesize = (np.dtype(np.float32).itemsize * self.vtk_numpoints * self.vtk_numpoint_components
                                  + np.dtype(np.uint32).itemsize * self.vtk_numlines * self.vtk_numline_components
                                  + np.dtype(np.uint32).itemsize * self.vtk_numlines
                                  + np.dtype(np.uint32).itemsize * vtk_cell_offsets
                                  + np.dtype(np.uint32).itemsize * 4)

    def set_filename(self, appendmodelnumber, G):
        """
        Construct filename from user-supplied name and model run number.

        Args:
            appendmodelnumber (str): Text to append to filename.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        self.filename = os.path.abspath(os.path.join(G.inputdirectory, self.basefilename + appendmodelnumber))
        self.filename += self.fileext

    def write_vtk(self, G, pbar):
        """
        Writes the geometry information to a VTK file. Either ImageData (.vti) for a
        per-cell geometry view, or PolygonalData (.vtp) for a per-cell-edge geometry view.

            N.B. No Python 3 support for VTK at time of writing (03/2015)

        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
            pbar (class): Progress bar class instance.
        """

        if self.fileext == '.vti':
            # Create arrays and add numeric IDs for PML, sources and receivers
            # (0 is not set, 1 is PML, srcs and rxs numbered thereafter)
            self.srcs_pml = np.zeros((G.nx + 1, G.ny + 1, G.nz + 1), dtype=np.int8)
            self.rxs = np.zeros((G.nx + 1, G.ny + 1, G.nz + 1), dtype=np.int8)
            for pml in G.pmls:
                self.srcs_pml[pml.xs:pml.xf, pml.ys:pml.yf, pml.zs:pml.zf] = 1
            for index, src in enumerate(G.hertziandipoles + G.magneticdipoles + G.voltagesources + G.transmissionlines):
                self.srcs_pml[src.xcoord, src.ycoord, src.zcoord] = index + 2
            for index, rx in enumerate(G.rxs):
                self.rxs[rx.xcoord, rx.ycoord, rx.zcoord] = index + 1

            vtk_srcs_pml_offset = round_value((np.dtype(np.uint32).itemsize * self.vtk_nxcells * self.vtk_nycells * self.vtk_nzcells) + np.dtype(np.uint32).itemsize)
            vtk_rxs_offset = round_value((np.dtype(np.uint32).itemsize * self.vtk_nxcells * self.vtk_nycells * self.vtk_nzcells) + np.dtype(np.uint32).itemsize + (np.dtype(np.int8).itemsize * self.vtk_nxcells * self.vtk_nycells * self.vtk_nzcells) + np.dtype(np.uint32).itemsize)

            with open(self.filename, 'wb') as f:
                f.write('<?xml version="1.0"?>\n'.encode('utf-8'))
                f.write('<VTKFile type="ImageData" version="1.0" byte_order="{}">\n'.format(GeometryView.byteorder).encode('utf-8'))
                f.write('<ImageData WholeExtent="{} {} {} {} {} {}" Origin="0 0 0" Spacing="{:.3} {:.3} {:.3}">\n'.format(self.vtk_xscells, self.vtk_xfcells, self.vtk_yscells, self.vtk_yfcells, self.vtk_zscells, self.vtk_zfcells, self.dx * G.dx, self.dy * G.dy, self.dz * G.dz).encode('utf-8'))
                f.write('<Piece Extent="{} {} {} {} {} {}">\n'.format(self.vtk_xscells, self.vtk_xfcells, self.vtk_yscells, self.vtk_yfcells, self.vtk_zscells, self.vtk_zfcells).encode('utf-8'))
                f.write('<CellData Scalars="Material">\n'.encode('utf-8'))
                f.write('<DataArray type="UInt32" Name="Material" format="appended" offset="0" />\n'.encode('utf-8'))
                f.write('<DataArray type="Int8" Name="Sources_PML" format="appended" offset="{}" />\n'.format(vtk_srcs_pml_offset).encode('utf-8'))
                f.write('<DataArray type="Int8" Name="Receivers" format="appended" offset="{}" />\n'.format(vtk_rxs_offset).encode('utf-8'))
                f.write('</CellData>\n'.encode('utf-8'))
                f.write('</Piece>\n</ImageData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))

                solid_geometry = np.zeros((self.vtk_ncells), dtype=np.uint32)
                srcs_pml_geometry = np.zeros((self.vtk_ncells), dtype=np.int8)
                rxs_geometry = np.zeros((self.vtk_ncells), dtype=np.int8)

                define_normal_geometry(
                    self.xs,
                    self.xf,
                    self.ys,
                    self.yf,
                    self.zs,
                    self.zf,
                    self.dx,
                    self.dy,
                    self.dz,
                    G.solid,
                    self.srcs_pml,
                    self.rxs,
                    solid_geometry,
                    srcs_pml_geometry,
                    rxs_geometry)

                # Write number of bytes of appended data as UInt32
                f.write(pack('I', solid_geometry.nbytes))
                pbar.update(n=4)
                # Write solid array
                f.write(solid_geometry)
                pbar.update(n=solid_geometry.nbytes)

                # Write number of bytes of appended data as UInt32
                f.write(pack('I', srcs_pml_geometry.nbytes))
                pbar.update(n=4)
                # Write sources and PML positions
                f.write(srcs_pml_geometry)
                pbar.update(n=srcs_pml_geometry.nbytes)

                # Write number of bytes of appended data as UInt32
                f.write(pack('I', rxs_geometry.nbytes))
                pbar.update(n=4)
                # Write receiver positions
                f.write(rxs_geometry)
                pbar.update(n=rxs_geometry.nbytes)

                f.write('\n</AppendedData>\n</VTKFile>'.encode('utf-8'))

                self.write_gprmax_info(f, G)

        elif self.fileext == '.vtp':
            with open(self.filename, 'wb') as f:
                f.write('<?xml version="1.0"?>\n'.encode('utf-8'))
                f.write('<VTKFile type="PolyData" version="1.0" byte_order="{}">\n'.format(GeometryView.byteorder).encode('utf-8'))
                f.write('<PolyData>\n<Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="{}" NumberOfStrips="0" NumberOfPolys="0">\n'.format(self.vtk_numpoints, self.vtk_numlines).encode('utf-8'))

                f.write('<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="0" />\n</Points>\n'.encode('utf-8'))
                f.write('<Lines>\n<DataArray type="UInt32" Name="connectivity" format="appended" offset="{}" />\n'.format(self.vtk_connectivity_offset).encode('utf-8'))
                f.write('<DataArray type="UInt32" Name="offsets" format="appended" offset="{}" />\n</Lines>\n'.format(self.vtk_offsets_offset).encode('utf-8'))

                f.write('<CellData Scalars="Material">\n'.encode('utf-8'))
                f.write('<DataArray type="UInt32" Name="Material" format="appended" offset="{}" />\n'.format(self.vtk_materials_offset).encode('utf-8'))
                f.write('</CellData>\n'.encode('utf-8'))

                f.write('</Piece>\n</PolyData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))

                # Coordinates of each point
                points = np.zeros((self.vtk_numpoints, 3), dtype=np.float32)

                # Number of x components

                # Node connectivity. Each index contains a pair of connected x nodes
                x_lines = np.zeros((self.vtk_nxlines, 2), dtype=np.uint32)
                # Material at Ex location in Yee cell.
                x_materials = np.zeros((self.vtk_nxlines), dtype=np.uint32)

                y_lines = np.zeros((self.vtk_nylines, 2), dtype=np.uint32)
                y_materials = np.zeros((self.vtk_nylines), dtype=np.uint32)

                z_lines = np.zeros((self.vtk_nzlines, 2), dtype=np.uint32)
                z_materials = np.zeros((self.vtk_nzlines), dtype=np.uint32)

                define_fine_geometry(self.nx,
                                     self.ny,
                                     self.nz,
                                     self.xs,
                                     self.xf,
                                     self.ys,
                                     self.yf,
                                     self.zs,
                                     self.zf,
                                     G.dx,
                                     G.dy,
                                     G.dz,
                                     G.ID,
                                     points,
                                     x_lines,
                                     x_materials,
                                     y_lines,
                                     y_materials,
                                     z_lines,
                                     z_materials)

                # Write point data
                f.write(pack('I', points.nbytes))
                f.write(points)
                pbar.update(n=points.nbytes)

                # Write connectivity data
                f.write(pack('I', np.dtype(np.uint32).itemsize * self.vtk_numlines * self.vtk_numline_components))
                pbar.update(n=4)
                f.write(x_lines)
                pbar.update(n=x_lines.nbytes)
                f.write(y_lines)
                pbar.update(n=y_lines.nbytes)
                f.write(z_lines)
                pbar.update(n=z_lines.nbytes)

                # Write cell type (line) offsets
                f.write(pack('I', np.dtype(np.uint32).itemsize * self.vtk_numlines))
                pbar.update(n=4)
                for vtk_offsets in range(self.vtk_numline_components, (self.vtk_numline_components * self.vtk_numlines) + self.vtk_numline_components, self.vtk_numline_components):
                    f.write(pack('I', vtk_offsets))
                    pbar.update(n=4)

                # Write material IDs per-cell-edge, i.e. from ID array
                f.write(pack('I', np.dtype(np.uint32).itemsize * self.vtk_numlines))
                pbar.update(n=4)
                f.write(x_materials)
                pbar.update(n=x_materials.nbytes)
                f.write(y_materials)
                pbar.update(n=y_materials.nbytes)
                f.write(z_materials)
                pbar.update(n=z_materials.nbytes)

                f.write('\n</AppendedData>\n</VTKFile>'.encode('utf-8'))
                self.write_gprmax_info(f, G, materialsonly=True)

    def write_gprmax_info(self, f, G, materialsonly=False):
        """
        Writes gprMax specific information relating material, source,
        and receiver names to numeric identifiers.

        Args:
            f (filehandle): VTK file.
            G (class): Grid class instance - holds essential parameters describing the model.
            materialsonly (boolean): Only write information on materials
        """

        f.write('\n\n<gprMax>\n'.encode('utf-8'))
        for material in G.materials:
            f.write('<Material name="{}">{}</Material>\n'.format(material.ID, material.numID).encode('utf-8'))
        if not materialsonly:
            f.write('<PML name="PML boundary region">1</PML>\n'.encode('utf-8'))
            for index, src in enumerate(G.hertziandipoles + G.magneticdipoles + G.voltagesources + G.transmissionlines):
                f.write('<Sources name="{}">{}</Sources>\n'.format(src.ID, index + 2).encode('utf-8'))
            for index, rx in enumerate(G.rxs):
                f.write('<Receivers name="{}">{}</Receivers>\n'.format(rx.ID, index + 1).encode('utf-8'))
        f.write('</gprMax>\n'.encode('utf-8'))


class GeometryObjects(object):
    """Geometry objects to be written to file."""

    def __init__(self, xs=None, ys=None, zs=None, xf=None, yf=None, zf=None, basefilename=None):
        """
        Args:
            xs, xf, ys, yf, zs, zf (int): Extent of the volume in cells.
            filename (str): Filename to save to.
        """

        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.xf = xf
        self.yf = yf
        self.zf = zf
        self.nx = self.xf - self.xs
        self.ny = self.yf - self.ys
        self.nz = self.zf - self.zs
        self.filename = basefilename + '.h5'
        self.materialsfilename = basefilename + '_materials.txt'

        # Sizes of arrays to write necessary to update progress bar
        self.solidsize = self.nx * self.ny * self.nz * np.dtype(np.int16).itemsize
        self.rigidsize = 18 * self.nx * self.ny * self.nz * np.dtype(np.int8).itemsize
        self.IDsize = 6 * self.nx * self.ny * self.nz * np.dtype(np.uint32).itemsize
        self.datawritesize = self.solidsize + self.rigidsize + self.IDsize

    def write_hdf5(self, G, pbar):
        """Write a geometry objects file in HDF5 format.

        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
            pbar (class): Progress bar class instance.
        """

        # Write the geometry objects to a HDF5 file
        fdata = h5py.File(os.path.abspath(os.path.join(G.inputdirectory, self.filename)), 'w')
        fdata.attrs['gprMax'] = __version__
        fdata.attrs['Title'] = G.title
        fdata.attrs['dx_dy_dz'] = (G.dx, G.dy, G.dz)

        # Get minimum and maximum integers of materials in geometry objects volume
        minmat = np.amin(G.ID[:, self.xs:self.xf, self.ys:self.yf, self.zs:self.zf])
        maxmat = np.amax(G.ID[:, self.xs:self.xf, self.ys:self.yf, self.zs:self.zf])
        fdata['/data'] = G.solid[self.xs:self.xf, self.ys:self.yf, self.zs:self.zf].astype('int16') - minmat
        pbar.update(self.solidsize)
        fdata['/rigidE'] = G.rigidE[:, self.xs:self.xf, self.ys:self.yf, self.zs:self.zf]
        fdata['/rigidH'] = G.rigidH[:, self.xs:self.xf, self.ys:self.yf, self.zs:self.zf]
        pbar.update(self.rigidsize)
        fdata['/ID'] = G.ID[:, self.xs:self.xf, self.ys:self.yf, self.zs:self.zf] - minmat
        pbar.update(self.IDsize)

        # Write materials list to a text file
        # This includes all materials in range whether used in volume or not
        fmaterials = open(os.path.abspath(os.path.join(G.inputdirectory, self.materialsfilename)), 'w')
        for numID in range(minmat, maxmat + 1):
            for material in G.materials:
                if material.numID == numID:
                    fmaterials.write('#material: {:g} {:g} {:g} {:g} {}\n'.format(material.er, material.se, material.mr, material.sm, material.ID))
                    if material.poles > 0:
                        if 'debye' in material.type:
                            dispersionstr = '#add_dispersion_debye: {:g} '.format(material.poles)
                            for pole in range(material.poles):
                                dispersionstr += '{:g} {:g} '.format(material.deltaer[pole], material.tau[pole])
                        elif 'lorenz' in material.type:
                            dispersionstr = '#add_dispersion_lorenz: {:g} '.format(material.poles)
                            for pole in range(material.poles):
                                dispersionstr += '{:g} {:g} {:g} '.format(material.deltaer[pole], material.tau[pole], material.alpha[pole])
                        elif 'drude' in material.type:
                            dispersionstr = '#add_dispersion_drude: {:g} '.format(material.poles)
                            for pole in range(material.poles):
                                dispersionstr += '{:g} {:g} '.format(material.tau[pole], material.alpha[pole])
                        dispersionstr += material.ID
                        fmaterials.write(dispersionstr + '\n')
