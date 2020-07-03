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

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import logging
import os
from pathlib import Path
from struct import pack

import gprMax.config as config
import h5py
import numpy as np

from ._version import __version__
from .cython.geometry_outputs import (define_fine_geometry,
                                      define_normal_geometry)
from .utilities import pretty_xml, round_value

logger = logging.getLogger(__name__)


class GeometryView:
    """Views of the geometry of the model."""

    def __init__(self, xs=None, ys=None, zs=None, xf=None, yf=None, zf=None, dx=None, dy=None, dz=None, filename=None, fileext=None, G=None):
        """
        Args:
            xs, xf, ys, yf, zs, zf (int): Extent of the volume in cells.
            dx, dy, dz (int): Spatial discretisation in cells.
            filename (str): Filename to save to.
            fileext (str): File extension of VTK file - either '.vti' for a per cell
                    geometry view, or '.vtp' for a per cell edge geometry view.
            G (FDTDGrid): Parameters describing a grid in a model.
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
        self.filename = filename
        self.fileext = fileext
        self.set_filename_called = False
        self.G = G

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
            self.datawritesize = (np.dtype(np.uint32).itemsize * self.vtk_ncells +
                                  3 * np.dtype(np.uint32).itemsize)

        elif self.fileext == '.vtp':
            self.vtk_numpoints = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
            self.vtk_numpoint_components = 3
            self.vtk_numline_components = 2
            self.vtk_nxlines = self.nx * (self.ny + 1) * (self.nz + 1)
            self.vtk_nylines = self.ny * (self.nx + 1) * (self.nz + 1)
            self.vtk_nzlines = self.nz * (self.nx + 1) * (self.ny + 1)
            self.vtk_numlines = self.vtk_nxlines + self.vtk_nylines + self.vtk_nzlines
            self.vtk_connectivity_offset = ((self.vtk_numpoints *
                                           self.vtk_numpoint_components *
                                           np.dtype(np.float32).itemsize) +
                                           np.dtype(np.uint32).itemsize)
            self.vtk_offsets_offset = (self.vtk_connectivity_offset +
                                      (self.vtk_numlines *
                                      self.vtk_numline_components *
                                      np.dtype(np.uint32).itemsize) +
                                      np.dtype(np.uint32).itemsize)
            self.vtk_materials_offset = (self.vtk_offsets_offset +
                                        (self.vtk_numlines *
                                        np.dtype(np.uint32).itemsize) +
                                        np.dtype(np.uint32).itemsize)
            vtk_cell_offsets = (((self.vtk_numline_components * self.vtk_numlines) +
                               self.vtk_numline_components - self.vtk_numline_components - 1) //
                               self.vtk_numline_components + 1)
            self.datawritesize = (np.dtype(np.float32).itemsize * self.vtk_numpoints *
                                 self.vtk_numpoint_components + np.dtype(np.uint32).itemsize *
                                 self.vtk_numlines * self.vtk_numline_components +
                                 np.dtype(np.uint32).itemsize * self.vtk_numlines +
                                 np.dtype(np.uint32).itemsize * vtk_cell_offsets +
                                 np.dtype(np.uint32).itemsize * 4)
    
    def initialise(self):
        pass

    def set_filename(self):
        """Construct filename from user-supplied name and model run number."""
        if not self.set_filename_called:
            self.set_filename_called = True
            parts = config.get_model_config().output_file_path.parts
            self.filename = Path(*parts[:-1], self.filename + config.get_model_config().appendmodelnumber)
            self.filename = self.filename.with_suffix(self.fileext)

    def write_vtk(self, G, pbar):
        """Writes the geometry information to a VTK file.
            Either ImageData (.vti) for a per-cell geometry view, or
            PolygonalData (.vtp) for a per-cell-edge geometry view.

            N.B. No Python 3 support for VTK at time of writing (03/2015)

        Args:
            G (FDTDGrid): Parameters describing a grid in a model.
            pbar (class): Progress bar class instance.
        """

        if self.fileext == '.vti':
            with open(self.filename, 'wb') as f:
                f.write('<?xml version="1.0"?>\n'.encode('utf-8'))
                f.write(f'<VTKFile type="ImageData" version="1.0" byte_order="{config.sim_config.vtk_byteorder}">\n'.encode('utf-8'))
                extent = f'{self.vtk_xscells} {self.vtk_xfcells} {self.vtk_yscells} {self.vtk_yfcells} {self.vtk_zscells} {self.vtk_zfcells}'
                f.write(f'<ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="{self.dx * G.dx:.3} {self.dy * G.dy:.3} {self.dz * G.dz:.3}">\n'.encode('utf-8'))
                f.write(f'<Piece Extent="{extent}">\n'.encode('utf-8'))
                f.write('<CellData Scalars="Material">\n'.encode('utf-8'))
                f.write('<DataArray type="UInt32" Name="Material" format="appended" offset="0" />\n'.encode('utf-8'))
                f.write('</CellData>\n'.encode('utf-8'))
                f.write('</Piece>\n</ImageData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))

                solid_geometry = np.zeros((self.vtk_ncells), dtype=np.uint32)

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
                    solid_geometry)

                # Write number of bytes of appended data as UInt32
                f.write(pack('I', solid_geometry.nbytes))
                pbar.update(n=4)
                # Write solid array
                f.write(solid_geometry)
                pbar.update(n=solid_geometry.nbytes)

                f.write('\n</AppendedData>\n</VTKFile>\n\n'.encode('utf-8'))
                self.write_gprmax_info(f, G)

        elif self.fileext == '.vtp':
            with open(self.filename, 'wb') as f:
                f.write('<?xml version="1.0"?>\n'.encode('utf-8'))
                f.write(f'<VTKFile type="PolyData" version="1.0" byte_order="{config.sim_config.vtk_byteorder}">\n'.encode('utf-8'))
                f.write(f'<PolyData>\n<Piece NumberOfPoints="{self.vtk_numpoints}" NumberOfVerts="0" NumberOfLines="{self.vtk_numlines}" NumberOfStrips="0" NumberOfPolys="0">\n'.encode('utf-8'))
                f.write('<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="0" />\n</Points>\n'.encode('utf-8'))
                f.write(f'<Lines>\n<DataArray type="UInt32" Name="connectivity" format="appended" offset="{self.vtk_connectivity_offset}" />\n'.encode('utf-8'))
                f.write(f'<DataArray type="UInt32" Name="offsets" format="appended" offset="{self.vtk_offsets_offset}" />\n</Lines>\n'.encode('utf-8'))
                f.write('<CellData Scalars="Material">\n'.encode('utf-8'))
                f.write('<DataArray type="UInt32" Name="Material" format="appended" offset="{}" />\n'.format(self.vtk_materials_offset).encode('utf-8'))
                f.write('</CellData>\n'.encode('utf-8'))
                f.write('</Piece>\n</PolyData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))

                # Coordinates of each point
                points = np.zeros((self.vtk_numpoints, 3), dtype=np.float32)

                # Node connectivity. Each index contains a pair of connected x nodes
                x_lines = np.zeros((self.vtk_nxlines, 2), dtype=np.uint32)
                # Material at Ex location in Yee cell.
                x_materials = np.zeros((self.vtk_nxlines), dtype=np.uint32)
                # Node connectivity. Each index contains a pair of connected y nodes
                y_lines = np.zeros((self.vtk_nylines, 2), dtype=np.uint32)
                # Material at Ey location in Yee cell.
                y_materials = np.zeros((self.vtk_nylines), dtype=np.uint32)
                # Node connectivity. Each index contains a pair of connected z nodes
                z_lines = np.zeros((self.vtk_nzlines, 2), dtype=np.uint32)
                # Material at Ez location in Yee cell.
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
                f.write(pack('I', np.dtype(np.uint32).itemsize *
                                  self.vtk_numlines * self.vtk_numline_components))
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
                for vtk_offsets in range(self.vtk_numline_components,
                                        (self.vtk_numline_components * self.vtk_numlines) +
                                        self.vtk_numline_components, self.vtk_numline_components):
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

                f.write('\n</AppendedData>\n</VTKFile>\n\n'.encode('utf-8'))
                self.write_gprmax_info(f, G, materialsonly=True)

    def write_gprmax_info(self, f, G, materialsonly=False):
        """Writes gprMax specific information relating material, source,
            and receiver names to numeric identifiers.

        Args:
            f (filehandle): VTK file.
            G (FDTDGrid): Parameters describing a grid in a model.
            materialsonly (bool): Only write information on materials
        """

        root = ET.Element('gprMax')
        root.set('Version', __version__)
        root.set('dx_dy_dz', (G.dx, G.dy, G.dz))
        root.set('nx_ny_nz', (G.nx, G.ny, G.nz))

        # Write the name and numeric ID for each material
        mats_el = ET.SubElement(root, 'Materials')
        for material in G.materials:
            mat_el = ET.SubElement(mats_el, 'Material')
            mat_el.set('ID', material.ID)
            mat_el.set('numID', str(material.numID))

        # Write information on PMLs, sources, and receivers
        if not materialsonly:
            # Information on PML thickness
            if G.pmls:
                # Only render PMLs if they are in the geometry view
                pmlstorender = dict.fromkeys(G.pmlthickness, 0)
                xmax = G.nx - self.vtk_xfcells
                ymax = G.ny - self.vtk_yfcells
                zmax = G.nz - self.vtk_zfcells
                if G.pmlthickness['x0'] - self.vtk_xscells > 0:
                    pmlstorender['x0'] = G.pmlthickness['x0']
                if G.pmlthickness['y0'] - self.vtk_yscells > 0:
                    pmlstorender['y0'] = G.pmlthickness['y0']
                if G.pmlthickness['z0'] - self.vtk_zscells > 0:
                    pmlstorender['z0'] = G.pmlthickness['z0']
                if self.vtk_xfcells > G.nx - G.pmlthickness['xmax']:
                    pmlstorender['xmax'] = G.pmlthickness['xmax']
                if self.vtk_yfcells > G.ny - G.pmlthickness['ymax']:
                    pmlstorender['ymax'] = G.pmlthickness['ymax']
                if self.vtk_zfcells > G.nz - G.pmlthickness['zmax']:
                    pmlstorender['zmax'] = G.pmlthickness['zmax']
                root.set('PMLthickness', list(pmlstorender.values()))
            # Location of sources and receivers
            srcs = G.hertziandipoles + G.magneticdipoles + G.voltagesources + G.transmissionlines
            if srcs:
                srcs_el = ET.SubElement(root, 'Sources')
                for src in srcs:
                    src_el = ET.SubElement(srcs_el, 'Source')
                    src_el.set('name', src.ID)
                    src_el.set('position', (src.xcoord * G.dx,
                                            src.ycoord * G.dy,
                                            src.zcoord * G.dz))                          
            if G.rxs:
                rxs_el = ET.SubElement(root, 'Receivers')
                for rx in G.rxs:
                    rx_el = ET.SubElement(rxs_el, 'Receiver')
                    rx_el.set('name', rx.ID)
                    rx_el.set('position', (rx.xcoord * G.dx,
                                           rx.ycoord * G.dy,
                                           rx.zcoord * G.dz))

        xml_string = pretty_xml(ET.tostring(root))
        f.write(str.encode(xml_string))

    def write_vtk_pvd(self, geometryviews):
        """Write a Paraview data file (.pvd) - PVD file provides pointers to the 
            collection of data files, i.e. GeometryViews.

        Args:
            geometryviews (list): list of GeometryViews to collect together.
        """

        root = ET.Element('VTKFile')
        root.set('type', 'Collection')
        root.set('version', '0.1')
        root.set('byte_order', str(config.sim_config.vtk_byteorder))
        collection = ET.SubElement(root, 'Collection')
        for gv in geometryviews:
            gv.set_filename()
            dataset = ET.SubElement(collection, 'DataSet')
            dataset.set('timestep', '0')
            dataset.set('group', '')
            dataset.set('part', '0')
            dataset.set('file', str(gv.filename.name))

        xml_string = pretty_xml(ET.tostring(root))

        self.pvdfile = config.get_model_config().output_file_path.with_suffix('.pvd')
        with open(self.pvdfile, 'w') as f:
            f.write(xml_string)


class GeometryObjects:
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

        # Set filenames
        parts = config.sim_config.input_file_path.with_suffix('').parts
        self.filename_hdf5 = Path(*parts[:-1], basefilename)
        self.filename_hdf5 = self.filename_hdf5.with_suffix('.h5')
        self.filename_materials = Path(*parts[:-1], basefilename + '_materials')
        self.filename_materials = self.filename_materials.with_suffix('.txt')

        # Sizes of arrays to write necessary to update progress bar
        self.solidsize = (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * np.dtype(np.int16).itemsize
        self.rigidsize = 18 * (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * np.dtype(np.int8).itemsize
        self.IDsize = 6 * (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * np.dtype(np.uint32).itemsize
        self.datawritesize = self.solidsize + self.rigidsize + self.IDsize

    def write_hdf5(self, G, pbar):
        """Write a geometry objects file in HDF5 format.

        Args:
            G (FDTDGrid): Parameters describing a grid in a model.
            pbar (class): Progress bar class instance.
        """

        # Write the geometry objects to a HDF5 file
        fdata = h5py.File(self.filename_hdf5, 'w')
        fdata.attrs['gprMax'] = __version__
        fdata.attrs['Title'] = G.title
        fdata.attrs['dx_dy_dz'] = (G.dx, G.dy, G.dz)

        # Get minimum and maximum integers of materials in geometry objects volume
        minmat = np.amin(G.ID[:, self.xs:self.xf + 1, self.ys:self.yf + 1, self.zs:self.zf + 1])
        maxmat = np.amax(G.ID[:, self.xs:self.xf + 1, self.ys:self.yf + 1, self.zs:self.zf + 1])
        fdata['/data'] = G.solid[self.xs:self.xf + 1, self.ys:self.yf + 1, self.zs:self.zf + 1].astype('int16') - minmat
        pbar.update(self.solidsize)
        fdata['/rigidE'] = G.rigidE[:, self.xs:self.xf + 1, self.ys:self.yf + 1, self.zs:self.zf + 1]
        fdata['/rigidH'] = G.rigidH[:, self.xs:self.xf + 1, self.ys:self.yf + 1, self.zs:self.zf + 1]
        pbar.update(self.rigidsize)
        fdata['/ID'] = G.ID[:, self.xs:self.xf + 1, self.ys:self.yf + 1, self.zs:self.zf + 1] - minmat
        pbar.update(self.IDsize)

        # Write materials list to a text file
        # This includes all materials in range whether used in volume or not
        fmaterials = open(self.filename_materials, 'w')
        for numID in range(minmat, maxmat + 1):
            for material in G.materials:
                if material.numID == numID:
                    fmaterials.write(f'#material: {material.er:g} {material.se:g} {material.mr:g} {material.sm:g} {material.ID}\n')
                    if material.poles > 0:
                        if 'debye' in material.type:
                            dispersionstr = f'#add_dispersion_debye: {material.poles:g} '
                            for pole in range(material.poles):
                                dispersionstr += f'{material.deltaer[pole]:g} {material.tau[pole]:g} '
                        elif 'lorenz' in material.type:
                            dispersionstr = f'#add_dispersion_lorenz: {material.poles:g} '
                            for pole in range(material.poles):
                                dispersionstr += f'{material.deltaer[pole]:g} {material.tau[pole]:g} {material.alpha[pole]:g} '
                        elif 'drude' in material.type:
                            dispersionstr = f'#add_dispersion_drude: {material.poles:g} '
                            for pole in range(material.poles):
                                dispersionstr += f'{material.tau[pole]:g} {material.alpha[pole]:g} '
                        dispersionstr += material.ID
                        fmaterials.write(dispersionstr + '\n')


class GeometryViewFineMultiGrid:
    """Geometry view for all grids in the simulation.

        Slicing is not supported by this class :( - only the full extent of the grids
        are output. The subgrids are output without the non-working regions.
        If you require domain slicing, GeometryView seperately for each grid you
        require and view them at once in Paraview.
    """

    def __init__(self, xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, fileext, G):
        """
        Args:
            xs, xf, ys, yf, zs, zf (int): Extent of the volume in cells.
            dx, dy, dz (int): Spatial discretisation in cells.
            filename (str): Filename to save to.
            fileext (str): File extension of VTK file - either '.vti' for a per cell
                    geometry view, or '.vtp' for a per cell edge geometry view.
            G (FDTDGrid): Parameters describing a grid in a model.
        """

        self.G = G
        self.nx = G.nx
        self.ny = G.ny
        self.nz = G.nz
        self.filename = filename
        self.fileext = '.vtp'
        self.sg_views = []

        self.additional_lines = 0
        self.additional_points = 0

    def set_filename(self):
        """Construct filename from user-supplied name and model run number."""
        parts = config.get_model_config().output_file_path.parts
        self.filename = Path(*parts[:-1], self.filename + config.get_model_config().appendmodelnumber)
        self.filename = self.filename.with_suffix(self.fileext)
    
    def initialise(self):

        G = self.G
    
        for sg in G.subgrids:
            # create an object to contain data relevant to the geometry processing
            sg_gv = SubgridGeometryView(sg)
            self.sg_views.append(sg_gv)
            # total additional lines required for subgrid
            self.additional_lines += sg_gv.n_total_lines
            # total additional points required for subgrid
            self.additional_points += sg_gv.n_total_points

        self.vtk_numpoints = self.additional_points + (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        self.vtk_numpoint_components = 3
        self.vtk_numlines = self.additional_lines + 2 * self.nx * self.ny + 2 * self.ny * self.nz + 2 * self.nx * self.nz + 3 * self.nx * self.ny * self.nz + self.nx + self.ny + self.nz
        self.vtk_numline_components = 2
        self.vtk_connectivity_offset = round_value(int((self.vtk_numpoints * self.vtk_numpoint_components * np.dtype(np.float32).itemsize) + np.dtype(np.uint32).itemsize))
        self.vtk_offsets_offset = round_value(int(self.vtk_connectivity_offset + (self.vtk_numlines * self.vtk_numline_components * np.dtype(np.uint32).itemsize) + np.dtype(np.uint32).itemsize))
        self.vtk_materials_offset = round_value(int(self.vtk_offsets_offset + (self.vtk_numlines * np.dtype(np.uint32).itemsize) + np.dtype(np.uint32).itemsize))
        self.datawritesize = np.dtype(np.float32).itemsize * self.vtk_numpoints * self.vtk_numpoint_components + np.dtype(np.uint32).itemsize * self.vtk_numlines * self.vtk_numline_components + np.dtype(np.uint32).itemsize * self.vtk_numlines + np.dtype(np.uint32).itemsize * self.vtk_numlines


    def write_vtk(self, *args):
        """Writes the geometry information to a VTK file.
            Either ImageData (.vti) for a per-cell geometry view, or
            PolygonalData (.vtp) for a per-cell-edge geometry view.

            N.B. No Python 3 support for VTK at time of writing (03/2015)
        """

        G = self.G

        with open(self.filename, 'wb') as f:
            # refine parameters for subgrid
            f.write('<?xml version="1.0"?>\n'.encode('utf-8'))
            f.write(f"""<VTKFile type="PolyData" version="1.0" byte_order="{config.sim_config.vtk_byteorder}">\n""".encode('utf-8'))
            f.write(f"""<PolyData>\n<Piece NumberOfPoints="{self.vtk_numpoints}" NumberOfVerts="0" NumberOfLines="{self.vtk_numlines}" NumberOfStrips="0" NumberOfPolys="0">\n""".encode('utf-8'))

            f.write('<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="0" />\n</Points>\n'.encode('utf-8'))
            f.write(f"""<Lines>\n<DataArray type="UInt32" Name="connectivity" format="appended" offset="{self.vtk_connectivity_offset}" />\n""".encode('utf-8'))
            f.write(f"""<DataArray type="UInt32" Name="offsets" format="appended" offset="{self.vtk_offsets_offset}" />\n</Lines>\n""".encode('utf-8'))
            f.write('<CellData Scalars="Material">\n'.encode('utf-8'))
            f.write(f"""<DataArray type="UInt32" Name="Material" format="appended" offset="{self.vtk_materials_offset}" />\n""".encode('utf-8'))
            f.write('</CellData>\n'.encode('utf-8'))

            f.write('</Piece>\n</PolyData>\n<AppendedData encoding="raw">\n_'.encode('utf-8'))

            # Write points
            logger.info('\nWriting points main grid')
            datasize = np.dtype(np.float32).itemsize * self.vtk_numpoints * self.vtk_numpoint_components
            f.write(pack('I', datasize))
            for i in range(0, G.nx + 1):
                for j in range(0, G.ny + 1):
                    for k in range(0, G.nz + 1):
                        f.write(pack('fff', i * G.dx, j * G.dy, k * G.dz))

            for sg_v in self.sg_views:
                logger.info('Writing points subgrid')
                sg_v.write_points(f, G)

            n_x_lines = self.nx * (self.ny + 1) * (self.nz + 1)
            x_lines = np.zeros((n_x_lines, 2), dtype=np.uint32)
            x_materials = np.zeros((n_x_lines), dtype=np.uint32)

            n_y_lines = self.ny * (self.nx + 1) * (self.nz + 1)
            y_lines = np.zeros((n_y_lines, 2), dtype=np.uint32)
            y_materials = np.zeros((n_y_lines), dtype=np.uint32)

            n_z_lines = self.nz * (self.nx + 1) * (self.ny + 1)
            z_lines = np.zeros((n_z_lines, 2), dtype=np.uint32)
            z_materials = np.zeros((n_z_lines), dtype=np.uint32)

            logger.info('Calculate connectivity main grid')
            label = 0
            counter_x = 0
            counter_y = 0
            counter_z = 0
            for i in range(self.nx + 1):
                for j in range(self.ny + 1):
                    for k in range(self.nz + 1):

                        if i < self.nx:
                            # x connectivity
                            label_x = label + (self.ny + 1) * (self.nz + 1)
                            x_lines[counter_x][0] = label
                            x_lines[counter_x][1] = label_x
                            # material for the line
                            x_materials[counter_x] = G.ID[0, i, j, k]
                            counter_x += 1
                        if j < self.ny:
                            label_y = label + self.nz + 1
                            y_lines[counter_y][0] = label
                            y_lines[counter_y][1] = label_y
                            y_materials[counter_y] = G.ID[1, i, j, k]
                            counter_y += 1
                        if k < self.nz:
                            label_z = label + 1
                            z_lines[counter_z][0] = label
                            z_lines[counter_z][1] = label_z
                            z_materials[counter_z] = G.ID[2, i, j, k]
                            counter_z += 1

                        label = label + 1

            logger.info('Calculate connectivity subgrids')
            for sg_v in self.sg_views:
                sg_v.populate_connectivity_and_materials(label)
                # use the last subgrids label for the next view
                label = sg_v.label

            datasize = np.dtype(np.uint32).itemsize * self.vtk_numlines * self.vtk_numline_components
            f.write(pack('I', datasize))

            f.write(x_lines.tostring())
            for sg_v in self.sg_views:
                f.write(sg_v.x_s_lines.tostring())
            f.write(y_lines.tostring())
            for sg_v in self.sg_views:
                f.write(sg_v.y_s_lines.tostring())
            f.write(z_lines.tostring())
            for sg_v in self.sg_views:
                f.write(sg_v.z_s_lines.tostring())

            # Write cell type (line) offsets
            vtk_cell_pts = 2
            datasize = np.dtype(np.uint32).itemsize * self.vtk_numlines
            f.write(pack('I', datasize))
            for vtk_offsets in range(vtk_cell_pts, (self.vtk_numline_components * self.vtk_numlines) + vtk_cell_pts, vtk_cell_pts):
                f.write(pack('I', vtk_offsets))

            datasize = np.dtype(np.uint32).itemsize * self.vtk_numlines
            f.write(pack('I', datasize))

            f.write(x_materials.tostring())
            for sg_v in self.sg_views:
                f.write(sg_v.x_s_materials.tostring())
            f.write(y_materials.tostring())
            for sg_v in self.sg_views:
                f.write(sg_v.y_s_materials.tostring())
            f.write(z_materials.tostring())
            for sg_v in self.sg_views:
                f.write(sg_v.z_s_materials.tostring())

            f.write('\n</AppendedData>\n</VTKFile>'.encode('utf-8'))
            self.write_gprmax_info(f, G, materialsonly=True)
    
    def write_gprmax_info(self, f, G, materialsonly=False):
        """Writes gprMax specific information relating material, source,
            and receiver names to numeric identifiers.

        Args:
            f (filehandle): VTK file.
            G (FDTDGrid): Parameters describing a grid in a model.
            materialsonly (bool): Only write information on materials
        """

        root = ET.Element('gprMax')
        root.set('Version', __version__)
        root.set('dx_dy_dz', (G.dx, G.dy, G.dz))
        root.set('nx_ny_nz', (G.nx, G.ny, G.nz))

        # Write the name and numeric ID for each material
        mats_el = ET.SubElement(root, 'Materials')
        for material in G.materials:
            mat_el = ET.SubElement(mats_el, 'Material')
            mat_el.set('ID', material.ID)
            mat_el.set('numID', str(material.numID))

        # Write information on PMLs, sources, and receivers
        if not materialsonly:
            # Information on PML thickness
            if G.pmls:
                # Only render PMLs if they are in the geometry view
                pmlstorender = dict.fromkeys(G.pmlthickness, 0)
                xmax = G.nx - self.vtk_xfcells
                ymax = G.ny - self.vtk_yfcells
                zmax = G.nz - self.vtk_zfcells
                if G.pmlthickness['x0'] - self.vtk_xscells > 0:
                    pmlstorender['x0'] = G.pmlthickness['x0']
                if G.pmlthickness['y0'] - self.vtk_yscells > 0:
                    pmlstorender['y0'] = G.pmlthickness['y0']
                if G.pmlthickness['z0'] - self.vtk_zscells > 0:
                    pmlstorender['z0'] = G.pmlthickness['z0']
                if self.vtk_xfcells > G.nx - G.pmlthickness['xmax']:
                    pmlstorender['xmax'] = G.pmlthickness['xmax']
                if self.vtk_yfcells > G.ny - G.pmlthickness['ymax']:
                    pmlstorender['ymax'] = G.pmlthickness['ymax']
                if self.vtk_zfcells > G.nz - G.pmlthickness['zmax']:
                    pmlstorender['zmax'] = G.pmlthickness['zmax']
                root.set('PMLthickness', list(pmlstorender.values()))
            # Location of sources and receivers
            srcs = G.hertziandipoles + G.magneticdipoles + G.voltagesources + G.transmissionlines
            if srcs:
                srcs_el = ET.SubElement(root, 'Sources')
                for src in srcs:
                    src_el = ET.SubElement(srcs_el, 'Source')
                    src_el.set('name', src.ID)
                    src_el.set('position', (src.xcoord * G.dx,
                                            src.ycoord * G.dy,
                                            src.zcoord * G.dz))                          
            if G.rxs:
                rxs_el = ET.SubElement(root, 'Receivers')
                for rx in G.rxs:
                    rx_el = ET.SubElement(rxs_el, 'Receiver')
                    rx_el.set('name', rx.ID)
                    rx_el.set('position', (rx.xcoord * G.dx,
                                           rx.ycoord * G.dy,
                                           rx.zcoord * G.dz))

        xml_string = pretty_xml(ET.tostring(root))
        f.write(str.encode(xml_string))


class SubgridGeometryView:

    def __init__(self, sg):

        self.sg = sg
        # n component lines in each direction required for subgrid in the working region
        n_sx_lines = sg.nwx * (sg.nwy + 1) * (sg.nwz + 1)
        n_sy_lines = sg.nwy * (sg.nwx + 1) * (sg.nwz + 1)
        n_sz_lines = sg.nwz * (sg.nwx + 1) * (sg.nwy + 1)

        n_total_lines = n_sx_lines + n_sy_lines + n_sz_lines
        self.n_total_lines = n_total_lines.astype(np.int32)

        # n points in the the working region
        n_total_points = (sg.nwx + 1) * (sg.nwy + 1) * (sg.nwz + 1)
        self.n_total_points = n_total_points.astype(np.int32)

        # connectivity array. 2 labels form an x component connection
        self.x_s_lines = np.zeros((n_sx_lines, 2), dtype=np.uint32)
        # material array. Each index contains a material index
        self.x_s_materials = np.zeros((n_sx_lines), dtype=np.uint32)

        self.y_s_lines = np.zeros((n_sy_lines, 2), dtype=np.uint32)
        self.y_s_materials = np.zeros((n_sy_lines), dtype=np.uint32)

        self.z_s_lines = np.zeros((n_sz_lines, 2), dtype=np.uint32)
        self.z_s_materials = np.zeros((n_sz_lines), dtype=np.uint32)

        self.label = 0

    def write_points(self, f, G):
        sg = self.sg
        for i in range(sg.i0, sg.i0 + sg.nwx + 1):
            for j in range(sg.j0, sg.j0 + sg.nwy + 1):
                for k in range(sg.k0, sg.k0 + sg.nwz + 1):
                    p_x = (sg.i0 * G.dx) + ((i - sg.i0) * sg.dx)
                    p_y = (sg.j0 * G.dy) + ((j - sg.j0) * sg.dy)
                    p_z = (sg.k0 * G.dz) + ((k - sg.k0) * sg.dz)
                    f.write(pack('fff', p_x, p_y, p_z))

    def populate_connectivity_and_materials(self, label):
        """Label is the starting label. 0 if no other grids are present but
            +1 the last label used for a multigrid view.
        """
        sg = self.sg
        self.label = label

        # counters to to index numpy arrays
        counter_x = 0
        counter_y = 0
        counter_z = 0

        for i in range(sg.nwx + 1):
            for j in range(sg.nwy + 1):
                for k in range(sg.nwz + 1):
                    i_s = i + sg.n_boundary_cells_x
                    j_s = j + sg.n_boundary_cells_y
                    k_s = k + sg.n_boundary_cells_z
                    if i < sg.nwx:
                        # x connectivity
                        label_x = self.label + (sg.nwy + 1) * (sg.nwz + 1)
                        self.x_s_lines[counter_x][0] = self.label
                        self.x_s_lines[counter_x][1] = label_x
                        # material for the line
                        self.x_s_materials[counter_x] = sg.ID[0, i_s, j_s, k_s]
                        counter_x += 1
                    if j < sg.nwy:
                        label_y = self.label + sg.nwz + 1
                        self.y_s_lines[counter_y][0] = self.label
                        self.y_s_lines[counter_y][1] = label_y
                        self.y_s_materials[counter_y] = sg.ID[1, i_s, j_s, k_s]
                        counter_y += 1
                    if k < sg.nwz:
                        label_z = self.label + 1
                        self.z_s_lines[counter_z][0] = self.label
                        self.z_s_lines[counter_z][1] = label_z
                        self.z_s_materials[counter_z] = sg.ID[2, i_s, j_s, k_s]
                        counter_z += 1

                    self.label = self.label + 1
