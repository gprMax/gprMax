# Copyright (C) 2015-2022: The University of Edinburgh
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
# subgrids geometry export
# user specifies subgrid in main grid coordinates
# if grid.pmlthickness['x0'] - self.geoview.vtk_xscells > 0:
#   if dx not 1 this above will break - fix

import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from evtk.hl import imageToVTK, linesToVTK
from evtk.vtk import VtkGroup, VtkImageData, VtkUnstructuredGrid
from tqdm import tqdm

import gprMax.config as config

from ._version import __version__
from .cython.geometry_outputs import write_lines
from .utilities.utilities import (get_terminal_width,
                                  numeric_list_to_float_list,
                                  numeric_list_to_int_list)

logger = logging.getLogger(__name__)


def write_vtk_pvd(gvs):
    """Write a Paraview data file (.pvd) - provides pointers to a collection of 
        data files, i.e. GeometryViews.

    Args:
        gvs (list): list of all GeometryViews.
    """

    filename = config.get_model_config().output_file_path
    pvd = VtkGroup(str(filename))

    # Add filenames of all GeometryViews to group
    for gv in gvs:
        sim_time = 0
        pvd.addFile(str(gv.filename) + gv.vtkfiletype.ext, sim_time, 
                    group = "", part = "0")
    pvd.save()
    
    logger.info(f'Written wrapper for geometry files: {filename.name}.pvd')


def save_geometry_views(gvs):
    """Create and save the geometryviews.
    
    Args:
        gvs (list): list of all GeometryViews.
    """
    
    logger.info('')
    for i, gv in enumerate(gvs):
        gv.set_filename()
        vtk_data = gv.prep_vtk()
        pbar = tqdm(total=vtk_data['nbytes'], unit='byte', unit_scale=True,
                    desc=f'Writing geometry view file {i + 1}/{len(gvs)}, {gv.filename.name}{gv.vtkfiletype.ext}',
                    ncols=get_terminal_width() - 1, file=sys.stdout,
                    disable=not config.sim_config.general['progressbars'])
        gv.write_vtk(vtk_data)
        pbar.update(vtk_data['nbytes'])
        pbar.close()

    # Write a Paraview data file (.pvd) if there is more than one GeometryView
    if len(gvs) > 1:
        write_vtk_pvd(gvs)
    
    logger.info('')
        

class GeometryView():
    """Base class for Geometry Views."""

    def __init__(self, xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, grid):
        """
        Args:
            xs, xf, ys, yf, zs, zf (int): Extent of the volume in cells
            dx, dy, dz (int): Spatial discretisation of geometry view in cells
            filename (str): Filename to save to
            grid (FDTDGrid): Parameters describing a grid in a model
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
        self.grid = grid

    def set_filename(self):
        """Construct filename from user-supplied name and model run number."""
        parts = config.get_model_config().output_file_path.parts
        self.filename = Path(*parts[:-1], 
                             self.filename + config.get_model_config().appendmodelnumber)


class GeometryViewLines(GeometryView):
    """Unstructured grid (.vtu) for a per-cell-edge geometry view."""

    def __init__(self, *args):
        super().__init__(*args)
        self.vtkfiletype = VtkUnstructuredGrid
    
    def prep_vtk(self):
        """Prepare data for writing to VTK file.
        
        Returns:
            vtk_data (dict): coordinates, data, and comments for VTK file
        """

        # Sample ID array according to geometry view spatial discretisation                
        # Only create a new array if subsampling is required
        if (self.grid.ID.shape != (self.xf, self.yf, self.zf) or
            (self.dx, self.dy, self.dz) != (1, 1, 1) or
                (self.xs, self.ys, self.zs) != (0, 0, 0)):
            # Require contiguous for evtk library
            ID = np.ascontiguousarray(self.grid.ID[:, self.xs:self.xf:self.dx,
                                                      self.ys:self.yf:self.dy, 
                                                      self.zs:self.zf:self.dz])
        else:
            # This array is contiguous by design
            ID = self.grid.ID

        x, y, z, lines = write_lines((self.xs * self.grid.dx),
                                     (self.ys * self.grid.dy),
                                     (self.zs * self.grid.dz),
                                     self.nx, self.ny, self.nz,
                                     (self.dx * self.grid.dx), 
                                     (self.dy * self.grid.dy),
                                     (self.dz * self.grid.dz), ID)

        # Write information about any PMLs, sources, receivers
        comments = Comments(self.grid, self)
        comments.averaged_materials = True
        comments.materials_only = True
        info = comments.get_gprmax_info()
        comments = json.dumps(info)

        # Number of bytes of data to be written to file
        offsets_size = np.arange(start = 2, step = 2, stop = len(x) + 1, 
                                 dtype = 'int32').nbytes
        connect_size = len(x) * np.dtype('int32').itemsize
        cell_type_size = len(x) * np.dtype('uint8').itemsize
        nbytes = (x.nbytes + y.nbytes + z.nbytes + lines.nbytes + offsets_size
                  + connect_size + cell_type_size)
        
        vtk_data = {'x': x, 'y': y, 'z': z, 'data': lines, 'comments': comments, 'nbytes': nbytes}

        return vtk_data

    def write_vtk(self, vtk_data):
        """Write geometry information to a VTK file.
        
        Args:
            vtk_data (dict): coordinates, data, and comments for VTK file
        """

         # Write the VTK file .vtu
        linesToVTK(str(self.filename), vtk_data['x'], vtk_data['y'], vtk_data['z'], cellData={"Material": vtk_data['data']}, 
                   comments=[vtk_data['comments']])


class GeometryViewVoxels(GeometryView):
    """Rectilinear grid (.vtr) for a per-cell geometry view."""

    def __init__(self, *args):
        super().__init__(*args)
        self.vtkfiletype = VtkImageData
    
    def prep_vtk(self):
        """Prepare data for writing to VTK file.
        
        Returns:
            vtk_data (dict): data and comments for VTK file
        """

        # Sample solid array according to geometry view spatial discretisation
        # Only create a new array if subsampling is required
        if (self.grid.solid.shape != (self.xf, self.yf, self.zf) or
            (self.dx, self.dy, self.dz) != (1, 1, 1) or
                (self.xs, self.ys, self.zs) != (0, 0, 0)):
            # Require contiguous for evtk library
            solid = np.ascontiguousarray(self.grid.solid[self.xs:self.xf:self.dx,
                                                         self.ys:self.yf:self.dy, 
                                                         self.zs:self.zf:self.dz])
        else:
            # This array is contiguous by design
            solid = self.grid.solid

        # Get information about pml, sources, receivers
        comments = Comments(self.grid, self)
        info = comments.get_gprmax_info()
        comments = json.dumps(info)

        vtk_data = {'data': solid, 'comments': comments, 'nbytes': solid.nbytes}

        return vtk_data

    def write_vtk(self, vtk_data):
        """Write geometry information to a VTK file.
        
        Args:
            vtk_data (dict): data and comments for VTK file
        """

        # Write the VTK file .vti
        imageToVTK(str(self.filename), 
                   origin=((self.xs * self.grid.dx),
                           (self.ys * self.grid.dy),
                           (self.zs * self.grid.dz)), 
                   spacing=((self.dx * self.grid.dx),
                            (self.dy * self.grid.dy),
                            (self.dz * self.grid.dz)), 
                   cellData={"Material": vtk_data['data']}, 
                   comments=[vtk_data['comments']])


class GeometryViewSubgridVoxels(GeometryViewVoxels):
    """Rectilinear grid (.vtr) for a per-cell geometry view for sub-grids."""

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
    """Comments can be strings included in the header of XML VTK file, and are
        used to hold extra (gprMax) information about the VTK data.
    """

    def __init__(self, grid, gv):
        self.grid = grid
        self.gv = gv
        self.averaged_materials = False
        self.materials_only = False

    def get_gprmax_info(self):
        """Returns gprMax specific information relating material, source,
            and receiver names to numeric identifiers.
        """

        # Comments for Paraview macro
        comments = {}

        comments['gprMax_version'] = __version__
        comments['dx_dy_dz'] = self.dx_dy_dz_comment()
        comments['nx_ny_nz'] = self.nx_ny_nz_comment()

        # Write the name and numeric ID for each material
        comments['Materials'] = self.materials_comment()

        # Write information on PMLs, sources, and receivers
        if not self.materials_only:
            # Information on PML thickness
            if self.grid.pmls:
                comments['PMLthickness'] = self.pml_gv_comment()
            srcs = self.grid.get_srcs()
            if srcs:
                comments['Sources'] = self.srcs_rx_gv_comment(srcs)
            if self.grid.rxs:
                comments['Receivers'] = self.srcs_rx_gv_comment(self.grid.rxs)

        return comments

    def pml_gv_comment(self):

        grid = self.grid

        # Only render PMLs if they are in the geometry view
        pmlstorender = dict.fromkeys(grid.pmlthickness, 0)

        # Casting to int required as json does not handle numpy types
        if grid.pmlthickness['x0'] - self.gv.xs > 0:
            pmlstorender['x0'] = int(grid.pmlthickness['x0'] - self.gv.xs)
        if grid.pmlthickness['y0'] - self.gv.ys > 0:
            pmlstorender['y0'] = int(grid.pmlthickness['y0'] - self.gv.ys)
        if grid.pmlthickness['z0'] - self.gv.zs > 0:
            pmlstorender['z0'] = int(grid.pmlthickness['z0'] - self.gv.zs)
        if self.gv.xf > grid.nx - grid.pmlthickness['xmax']:
            pmlstorender['xmax'] = int(self.gv.xf - (grid.nx - grid.pmlthickness['xmax']))
        if self.gv.yf > grid.ny - grid.pmlthickness['ymax']:
            pmlstorender['ymax'] = int(self.gv.yf - (grid.ny - grid.pmlthickness['ymax']))
        if self.gv.zf > grid.nz - grid.pmlthickness['zmax']:
            pmlstorender['zmax'] = int(self.gv.zf - (grid.nz - grid.pmlthickness['zmax']))

        return list(pmlstorender.values())

    def srcs_rx_gv_comment(self, srcs):
        """Used to name sources and/or receivers."""
        sc = []
        for src in srcs:
            p = (src.xcoord * self.grid.dx,
                 src.ycoord * self.grid.dy,
                 src.zcoord * self.grid.dz)
            p = numeric_list_to_float_list(p)

            s = {'name': src.ID,
                 'position': p}
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
        self.basefilename = basefilename

        # Set filenames
        parts = config.sim_config.input_file_path.with_suffix('').parts
        self.filename_hdf5 = Path(*parts[:-1], self.basefilename)
        self.filename_hdf5 = self.filename_hdf5.with_suffix('.h5')
        self.filename_materials = Path(
            *parts[:-1], self.basefilename + '_materials')
        self.filename_materials = self.filename_materials.with_suffix('.txt')

        # Sizes of arrays to write necessary to update progress bar
        self.solidsize = (self.nx + 1) * (self.ny + 1) * \
            (self.nz + 1) * np.dtype(np.uint32).itemsize
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
                    if hasattr(material, 'poles'):
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
