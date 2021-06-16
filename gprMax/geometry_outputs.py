# Copyright (C) 2015-2021: The University of Edinburgh
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


# TODO
# Get progress bar working
# subgrids geometry export
# have a look in evtk package for filesize estimation.
# cythonise line connectivity code
# user specifies subgrid in main grid coordinates
# write get size function
# if grid.pmlthickness['x0'] - self.geoview.vtk_xscells > 0:
#   if dx not 1 this above will break - fix
# can we get self.extension from evtk

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import json
import logging
from pathlib import Path
from struct import pack

import gprMax.config as config
import h5py
import numpy as np
from evtk.hl import rectilinearToVTK
from evtk.hl import linesToVTK

from ._version import __version__
from .cython.geometry_outputs import (define_fine_geometry,
                                      define_normal_geometry)
from .utilities.utilities import pretty_xml, round_value, numeric_list_to_int_list, numeric_list_to_float_list

logger = logging.getLogger(__name__)


class GeometryView():
    def __init__(self, xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, grid):
        """
        Args:
            xs, xf, ys, yf, zs, zf (int): Extent of the volume in cells.
            dx, dy, dz (int): Spatial discretisation in cells.
            filename (str): Filename to save to.
            grid (FDTDGrid): Parameters describing a grid in a model.
        """
        # indices start
        self.xs = xs
        self.ys = ys
        self.zs = zs
        # indices stop
        self.xf = xf
        self.yf = yf
        self.zf = zf

        self.nx = self.xf - self.xs
        self.ny = self.yf - self.ys
        self.nz = self.zf - self.zs
        # sampling
        self.dx = dx
        self.dy = dy
        self.dz = dz

        # 
        self.vtk_xscells = round_value(self.xs / self.dx)
        self.vtk_xfcells = round_value(self.xf / self.dx)
        self.vtk_yscells = round_value(self.ys / self.dy)
        self.vtk_yfcells = round_value(self.yf / self.dy)
        self.vtk_zscells = round_value(self.zs / self.dz)
        self.vtk_zfcells = round_value(self.zf / self.dz)

        self.filename = filename
        self.set_filename_called = False
        self.grid = grid

    def format_filename_evtk(self, filename):
        return str(filename)

    def set_filename(self):
        """Construct filename from user-supplied name and model run number."""
        if not self.set_filename_called:
            self.set_filename_called = True
            parts = config.get_model_config().output_file_path.parts
            self.filename = Path(
                *parts[:-1], self.filename + config.get_model_config().appendmodelnumber)

    
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
            dataset.set('file', str(gv.filename.name) + self.extension)

        xml_string = pretty_xml(ET.tostring(root))

        self.pvdfile = config.get_model_config().output_file_path.with_suffix('.pvd')
        with open(self.pvdfile, 'w') as f:
            f.write(xml_string)


    def get_size(self):
        return 100


class GeometryViewLines(GeometryView):

    def __init__(self, *args):
        super().__init__(*args)
        self.output_type = 'fine'
        self.extension = '.vtu'
    
    def sample_ID(self):
        """Function to sub sample the ID array."""                
        # only create a new array if subsampling is required.
        if (self.grid.solid.shape != (self.xf, self.yf, self.zf) or
            (self.dx, self.dy, self.dz) != (1, 1, 1) or
                (self.xs, self.ys, self.zs) != (0, 0, 0)):
            # require contiguous for evtk library
            ID = np.ascontiguousarray(self.grid.ID[:, self.xs:self.xf:self.dx,
                                                    self.ys:self.yf:self.dy, self.zs:self.zf:self.dz])
        else:
            # this array is contiguous by design.
            ID = self.grid.ID
        
        return ID

    def write_vtk(self, pbar):
        """Writes the geometry information to a VTK file.
            Unstructured edge (.vtu) for a per-cell geometry view

        Args:
            pbar (class): Progress bar class instance.
        """

        # sample self.grid.id at the user specified region and discretisation
        id = self.sample_ID()

        # line counter
        lc = 0
        # point counter
        pc = 0

        n_x_lines = self.nx * (self.ny + 1) * (self.nz + 1)
        n_y_lines = self.ny * (self.nx + 1) * (self.nz + 1)
        n_z_lines = self.nz * (self.nx + 1) * (self.ny + 1)

        n_lines = n_x_lines + n_y_lines + n_z_lines

        # a line is defined by 2 points. Each line must have 2 new points
        n_points = 2 * n_lines

        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.zeros(n_points)

        l = np.zeros(n_lines)

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):

                    # x yee cell edge
                    # line the line property to the relevent material

                    l[lc] = id[0][i, j, k]
                    # set the starting point position of the edge
                    x[pc], y[pc], z[pc] = i * self.dx, j * self.dy, k * self.dz
                    # next point
                    pc += 1
                    # set the end point position of the edge
                    x[pc], y[pc], z[pc] = (
                        i + 1) * self.dx, j * self.dy, k * self.dz
                    # next point
                    pc += 1

                    # next line
                    lc += 1

                    # y yee cell edge
                    l[lc] = id[1, i, j, k]
                    x[pc], y[pc], z[pc] = i * self.dx, j * self.dy, k * self.dz
                    pc += 1
                    x[pc], y[pc], z[pc] = i * \
                        self.dx, (j + 1) * self.dy, k * self.dz
                    pc += 1
                    lc += 1

                    # z yee cell edge
                    l[lc] = id[2, i, j, k]
                    x[pc], y[pc], z[pc] = i * self.dx, j * self.dy, k * self.dz
                    pc += 1
                    x[pc], y[pc], z[pc] = i * self.dx, j * \
                        self.dy, (k + 1) * self.dz
                    pc += 1
                    lc += 1

        # Get information about pml, sources, receivers
        comments = Comments(self.grid, self)
        comments.averaged_materials = True
        comments.materials_only = True
        info = comments.get_gprmax_info()
        comments = json.dumps(info)

        linesToVTK(self.format_filename_evtk(self.filename),
                   x, y, z, cellData={"Material": l}, comments=[comments])


class GeometryViewVoxels(GeometryView):
    """Views of the geometry of the model."""

    def __init__(self, *args):
        super().__init__(*args)
        self.output_type = 'normal'
        self.extension = '.vtr'
    
    def sample_solid(self):
        """Function to sub sample the solid array"""
        
        # only create a new array if subsampling is required.
        if (self.grid.solid.shape != (self.xf, self.yf, self.zf) or
            (self.dx, self.dy, self.dz) != (1, 1, 1) or
                (self.xs, self.ys, self.zs) != (0, 0, 0)):
            # require contiguous for evtk library
            solid = np.ascontiguousarray(self.grid.solid[self.xs:self.xf:self.dx,
                                                    self.ys:self.yf:self.dy, self.zs:self.zf:self.dz])
        else:
            # this array is contiguous by design.
            solid = self.grid.solid
        return solid
    
    def get_coordinates(self, solid):
        # (length is number of vertices in each direction) * (size of each block [m]) + (starting offset)
        x = np.arange(
            0, solid.shape[0] + 1) * (self.grid.dx * self.dx) + (self.xs * self.grid.dx)
        y = np.arange(
            0, solid.shape[1] + 1) * (self.grid.dy * self.dy) + (self.ys * self.grid.dy)
        z = np.arange(
            0, solid.shape[2] + 1) * (self.grid.dz * self.dz) + (self.zs * self.grid.dz)
        return x, y, z

    def write_vtk(self, pbar):
        """Writes the geometry information to a VTK file.
            Rectilinear (.vtr) for a per-cell geometry view, or
        Args:
            pbar (class): Progress bar class instance.
        """
        grid = self.grid

        solid = self.sample_solid()

        # coordinates of vertices (rectilinear)
        x, y, z = self.get_coordinates(solid)

        # Get information about pml, sources, receivers
        comments = Comments(grid, self)
        info = comments.get_gprmax_info()
        comments = json.dumps(info)

        # Write the VTK file .vtr
        rectilinearToVTK(self.format_filename_evtk(self.filename), x, y, z, cellData={
                         "Material": solid}, comments=[comments])


class GeometryViewSubgridVoxels(GeometryViewVoxels):
    """Views of the geometry of the model."""

    def __init__(self, *args):
        # for sub-grid we are only going to export the entire grid. temporary fix.
        xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, grid = args
        xs, ys, zs = 0, 0, 0
        xf, yf, zf = grid.nx, grid.ny, grid.nz
        dx, dy, dz = 1, 1, 1
        args = xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, grid
        super().__init__(*args)
        self.output_type = 'normal'
    """
    def get_coordinates(self, solid):
        # (length is number of vertices in each direction) * (size of each block [m]) + (starting offset) + grid offset
        x = np.arange(
            0, solid.shape[0] + 1) * (self.grid.dx * self.dx) + ((self.xs - self.grid.n_boundary_cells_x) * self.grid.dx)# + self.grid.x1
        y = np.arange(
            0, solid.shape[1] + 1) * (self.grid.dy * self.dy) + ((self.ys - self.grid.n_boundary_cells_y) * self.grid.dy)# + self.grid.y1
        z = np.arange(
            0, solid.shape[2] + 1) * (self.grid.dz * self.dz) + ((self.zs - self.grid.n_boundary_cells_z) * self.grid.dz)# + self.grid.z1
        return x, y, z
    """

class Comments():

    def __init__(self, grid, geoview):
        self.geoview = geoview
        self.grid = grid
        self.averaged_materials = False
        self.materials_only = False

    def get_gprmax_info(self):
        """Returns gprMax specific information relating material, source,
            and receiver names to numeric identifiers.

        Args:
            grid (FDTDGrid): Parameters describing a grid in a model.
            materialsonly (bool): Only write information on materials
        """

        # list containing comments for paraview macro
        comments = {}

        comments['Version'] = __version__
        comments['dx_dy_dz'] = self.dx_dy_dz_comment()
        comments['nx_ny_nz'] = self.nx_ny_nz_comment()
        # Write the name and numeric ID for each material
        comments['Materials'] = self.materials_comment()

        # Write information on PMLs, sources, and receivers
        if not self.materials_only:
            # Information on PML thickness
            if self.grid.pmls:
                comments['PMLthickness'] = self.pml_geoview_comment()
            srcs = self.grid.get_srcs()
            if srcs:
                comments['Sources'] = self.srcs_rx_geoview_comment(srcs)
            if self.grid.rxs:
                comments['Receivers'] = self.srcs_rx_geoview_comment(
                    self.grid.rxs)

        return comments

    def pml_geoview_comment(self):

        grid = self.grid
        # Only render PMLs if they are in the geometry view
        pmlstorender = dict.fromkeys(grid.pmlthickness, 0)

        if grid.pmlthickness['x0'] - self.geoview.vtk_xscells > 0:
            pmlstorender['x0'] = grid.pmlthickness['x0']
        if grid.pmlthickness['y0'] - self.geoview.vtk_yscells > 0:
            pmlstorender['y0'] = grid.pmlthickness['y0']
        if grid.pmlthickness['z0'] - self.geoview.vtk_zscells > 0:
            pmlstorender['z0'] = grid.pmlthickness['z0']
        if self.geoview.vtk_xfcells > grid.nx - grid.pmlthickness['xmax']:
            pmlstorender['xmax'] = grid.pmlthickness['xmax']
        if self.geoview.vtk_yfcells > grid.ny - grid.pmlthickness['ymax']:
            pmlstorender['ymax'] = grid.pmlthickness['ymax']
        if self.geoview.vtk_zfcells > grid.nz - grid.pmlthickness['zmax']:
            pmlstorender['zmax'] = grid.pmlthickness['zmax']

        return list(pmlstorender.values())

    def srcs_rx_geoview_comment(self, srcs):

        sc = []
        for src in srcs:
            p = (src.xcoord * self.grid.dx,
                 src.ycoord * self.grid.dy,
                 src.zcoord * self.grid.dz)
            p = numeric_list_to_float_list(p)

            s = {'name': src.ID,
                 'position': p
                 }
            sc.append(s)
        return sc

    def dx_dy_dz_comment(self):
        return numeric_list_to_float_list([self.grid.dx, self.grid.dy, self.grid.dz])

    def nx_ny_nz_comment(self):
        return numeric_list_to_int_list([self.grid.nx, self.grid.ny, self.grid.nz])

    def materials_comment(self):
        if not self.averaged_materials:
            return [m.ID for m in self.grid.materials if '+' not in m.ID]
        else:
            return [m.ID for m in self.grid.materials]


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
        self.filename_materials = Path(
            *parts[:-1], basefilename + '_materials')
        self.filename_materials = self.filename_materials.with_suffix('.txt')

        # Sizes of arrays to write necessary to update progress bar
        self.solidsize = (self.nx + 1) * (self.ny + 1) * \
            (self.nz + 1) * n.itemsize
        self.rigidsize = 18 * (self.nx + 1) * (self.ny + 1) * \
            (self.nz + 1) * np.dtype(np.int8).itemsize
        self.IDsize = 6 * (self.nx + 1) * (self.ny + 1) * \
            (self.nz + 1) * np.dtype(np.uint32).itemsize
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
        minmat = np.amin(G.ID[:, self.xs:self.xf + 1,
                         self.ys:self.yf + 1, self.zs:self.zf + 1])
        maxmat = np.amax(G.ID[:, self.xs:self.xf + 1,
                         self.ys:self.yf + 1, self.zs:self.zf + 1])
        fdata['/data'] = G.solid[self.xs:self.xf + 1, self.ys:self.yf +
                                 1, self.zs:self.zf + 1].astype('int16') - minmat
        pbar.update(self.solidsize)
        fdata['/rigidE'] = G.rigidE[:, self.xs:self.xf +
                                    1, self.ys:self.yf + 1, self.zs:self.zf + 1]
        fdata['/rigidH'] = G.rigidH[:, self.xs:self.xf +
                                    1, self.ys:self.yf + 1, self.zs:self.zf + 1]
        pbar.update(self.rigidsize)
        fdata['/ID'] = G.ID[:, self.xs:self.xf + 1,
                            self.ys:self.yf + 1, self.zs:self.zf + 1] - minmat
        pbar.update(self.IDsize)

        # Write materials list to a text file
        # This includes all materials in range whether used in volume or not
        fmaterials = open(self.filename_materials, 'w')
        for numID in range(minmat, maxmat + 1):
            for material in G.materials:
                if material.numID == numID:
                    fmaterials.write(
                        f'#material: {material.er:g} {material.se:g} {material.mr:g} {material.sm:g} {material.ID}\n')
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
