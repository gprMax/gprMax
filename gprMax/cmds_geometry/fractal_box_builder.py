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

import numpy as np

import gprMax.config as config
from .cmds_geometry import UserObjectGeometry
from ..cython.geometry_primitives import build_voxels_from_array
from ..cython.geometry_primitives import build_voxels_from_array_mask
from ..exceptions import CmdInputError


class FractalBoxBuilder(UserObjectGeometry):
    """Internal class for fractal box modifications. This class should be used
    internally only when surface modification have been made to a fractal box"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash = '#fractal_box_modifications'

    def create(self, grid, uip):
        for volume in grid.fractalvolumes:
            if volume.fractalsurfaces:
                volume.originalxs = volume.xs
                volume.originalxf = volume.xf
                volume.originalys = volume.ys
                volume.originalyf = volume.yf
                volume.originalzs = volume.zs
                volume.originalzf = volume.zf

                # Extend the volume to accomodate any rough surfaces, grass, or roots
                for surface in volume.fractalsurfaces:
                    if surface.surfaceID == 'xminus':
                        if surface.fractalrange[0] < volume.xs:
                            volume.nx += volume.xs - surface.fractalrange[0]
                            volume.xs = surface.fractalrange[0]
                    elif surface.surfaceID == 'xplus':
                        if surface.fractalrange[1] > volume.xf:
                            volume.nx += surface.fractalrange[1] - volume.xf
                            volume.xf = surface.fractalrange[1]
                    elif surface.surfaceID == 'yminus':
                        if surface.fractalrange[0] < volume.ys:
                            volume.ny += volume.ys - surface.fractalrange[0]
                            volume.ys = surface.fractalrange[0]
                    elif surface.surfaceID == 'yplus':
                        if surface.fractalrange[1] > volume.yf:
                            volume.ny += surface.fractalrange[1] - volume.yf
                            volume.yf = surface.fractalrange[1]
                    elif surface.surfaceID == 'zminus':
                        if surface.fractalrange[0] < volume.zs:
                            volume.nz += volume.zs - surface.fractalrange[0]
                            volume.zs = surface.fractalrange[0]
                    elif surface.surfaceID == 'zplus':
                        if surface.fractalrange[1] > volume.zf:
                            volume.nz += surface.fractalrange[1] - volume.zf
                            volume.zf = surface.fractalrange[1]

                # If there is only 1 bin then a normal material is being used, otherwise a mixing model
                if volume.nbins == 1:
                    volume.fractalvolume = np.ones((volume.nx, volume.ny, volume.nz),
                                                    dtype=config.sim_config.dtypes['float_or_double'])
                    materialnumID = next(x.numID for x in grid.materials if x.ID == volume.operatingonID)
                    volume.fractalvolume *= materialnumID
                else:
                    volume.generate_fractal_volume(grid)
                    volume.fractalvolume += volume.mixingmodel.startmaterialnum

                volume.generate_volume_mask()

                # Apply any rough surfaces and add any surface water to the 3D mask array
                for surface in volume.fractalsurfaces:
                    if surface.surfaceID == 'xminus':
                        for i in range(surface.fractalrange[0], surface.fractalrange[1]):
                            for j in range(surface.ys, surface.yf):
                                for k in range(surface.zs, surface.zf):
                                    if i > surface.fractalsurface[j - surface.ys, k - surface.zs]:
                                        volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 1
                                    elif surface.filldepth > 0 and i > surface.filldepth:
                                        volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 2
                                    else:
                                        volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 0

                    elif surface.surfaceID == 'xplus':
                        if not surface.ID:
                            for i in range(surface.fractalrange[0], surface.fractalrange[1]):
                                for j in range(surface.ys, surface.yf):
                                    for k in range(surface.zs, surface.zf):
                                        if i < surface.fractalsurface[j - surface.ys, k - surface.zs]:
                                            volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 1
                                        elif surface.filldepth > 0 and i < surface.filldepth:
                                            volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 2
                                        else:
                                            volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 0
                        elif surface.ID == 'grass':
                            g = surface.grass[0]
                            # Build the blades of the grass
                            blade = 0
                            for j in range(surface.ys, surface.yf):
                                for k in range(surface.zs, surface.zf):
                                    if surface.fractalsurface[j - surface.ys, k - surface.zs] > 0:
                                        height = 0
                                        for i in range(volume.xs, surface.fractalrange[1]):
                                            if (i < surface.fractalsurface[j - surface.ys, k - surface.zs] and
                                                volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] != 1):
                                                y, z = g.calculate_blade_geometry(blade, height)
                                                # Add y, z coordinates to existing location
                                                yy = int(j - volume.ys + y)
                                                zz = int(k - volume.zs + z)
                                                # If these coordinates are outwith fractal volume stop building the blade, otherwise set the mask for grass
                                                if yy < 0 or yy >= volume.mask.shape[1] or zz < 0 or zz >= volume.mask.shape[2]:
                                                    break
                                                else:
                                                    volume.mask[i - volume.xs, yy, zz] = 3
                                                    height += 1
                                        blade += 1

                            # Build the roots of the grass
                            root = 0
                            for j in range(surface.ys, surface.yf):
                                for k in range(surface.zs, surface.zf):
                                    if surface.fractalsurface[j - surface.ys, k - surface.zs] > 0:
                                        depth = 0
                                        i = volume.xf - 1
                                        while i > volume.xs:
                                            if (i > volume.originalxf - (surface.fractalsurface[j - surface.ys, k - surface.zs] -
                                                volume.originalxf) and volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] == 1):
                                                y, z = g.calculate_root_geometry(root, depth)
                                                # Add y, z coordinates to existing location
                                                yy = int(j - volume.ys + y)
                                                zz = int(k - volume.zs + z)
                                                # If these coordinates are outwith the fractal volume stop building the root, otherwise set the mask for grass
                                                if yy < 0 or yy >= volume.mask.shape[1] or zz < 0 or zz >= volume.mask.shape[2]:
                                                    break
                                                else:
                                                    volume.mask[i - volume.xs, yy, zz] = 3
                                                    depth += 1
                                            i -= 1
                                        root += 1

                    elif surface.surfaceID == 'yminus':
                        for i in range(surface.xs, surface.xf):
                            for j in range(surface.fractalrange[0], surface.fractalrange[1]):
                                for k in range(surface.zs, surface.zf):
                                    if j > surface.fractalsurface[i - surface.xs, k - surface.zs]:
                                        volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 1
                                    elif surface.filldepth > 0 and j > surface.filldepth:
                                        volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 2
                                    else:
                                        volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 0

                    elif surface.surfaceID == 'yplus':
                        if not surface.ID:
                            for i in range(surface.xs, surface.xf):
                                for j in range(surface.fractalrange[0], surface.fractalrange[1]):
                                    for k in range(surface.zs, surface.zf):
                                        if j < surface.fractalsurface[i - surface.xs, k - surface.zs]:
                                            volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 1
                                        elif surface.filldepth > 0 and j < surface.filldepth:
                                            volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 2
                                        else:
                                            volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 0
                        elif surface.ID == 'grass':
                            g = surface.grass[0]
                            # Build the blades of the grass
                            blade = 0
                            for i in range(surface.xs, surface.xf):
                                for k in range(surface.zs, surface.zf):
                                    if surface.fractalsurface[i - surface.xs, k - surface.zs] > 0:
                                        height = 0
                                        for j in range(volume.ys, surface.fractalrange[1]):
                                            if (j < surface.fractalsurface[i - surface.xs, k - surface.zs] and
                                                volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] != 1):
                                                x, z = g.calculate_blade_geometry(blade, height)
                                                # Add x, z coordinates to existing location
                                                xx = int(i - volume.xs + x)
                                                zz = int(k - volume.zs + z)
                                                # If these coordinates are outwith fractal volume stop building the blade, otherwise set the mask for grass
                                                if xx < 0 or xx >= volume.mask.shape[0] or zz < 0 or zz >= volume.mask.shape[2]:
                                                    break
                                                else:
                                                    volume.mask[xx, j - volume.ys, zz] = 3
                                                    height += 1
                                        blade += 1

                            # Build the roots of the grass
                            root = 0
                            for i in range(surface.xs, surface.xf):
                                for k in range(surface.zs, surface.zf):
                                    if surface.fractalsurface[i - surface.xs, k - surface.zs] > 0:
                                        depth = 0
                                        j = volume.yf - 1
                                        while j > volume.ys:
                                            if (j > volume.originalyf - (surface.fractalsurface[i - surface.xs, k - surface.zs] -
                                                volume.originalyf) and volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] == 1):
                                                x, z = g.calculate_root_geometry(root, depth)
                                                # Add x, z coordinates to existing location
                                                xx = int(i - volume.xs + x)
                                                zz = int(k - volume.zs + z)
                                                # If these coordinates are outwith the fractal volume stop building the root, otherwise set the mask for grass
                                                if xx < 0 or xx >= volume.mask.shape[0] or zz < 0 or zz >= volume.mask.shape[2]:
                                                    break
                                                else:
                                                    volume.mask[xx, j - volume.ys, zz] = 3
                                                    depth += 1
                                            j -= 1
                                        root += 1

                    elif surface.surfaceID == 'zminus':
                        for i in range(surface.xs, surface.xf):
                            for j in range(surface.ys, surface.yf):
                                for k in range(surface.fractalrange[0], surface.fractalrange[1]):
                                    if k > surface.fractalsurface[i - surface.xs, j - surface.ys]:
                                        volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 1
                                    elif surface.filldepth > 0 and k > surface.filldepth:
                                        volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 2
                                    else:
                                        volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 0

                    elif surface.surfaceID == 'zplus':
                        if not surface.ID:
                            for i in range(surface.xs, surface.xf):
                                for j in range(surface.ys, surface.yf):
                                    for k in range(surface.fractalrange[0], surface.fractalrange[1]):
                                        if k < surface.fractalsurface[i - surface.xs, j - surface.ys]:
                                            volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 1
                                        elif surface.filldepth > 0 and k < surface.filldepth:
                                            volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 2
                                        else:
                                            volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] = 0
                        elif surface.ID == 'grass':
                            g = surface.grass[0]
                            # Build the blades of the grass
                            blade = 0
                            for i in range(surface.xs, surface.xf):
                                for j in range(surface.ys, surface.yf):
                                    if surface.fractalsurface[i - surface.xs, j - surface.ys] > 0:
                                        height = 0
                                        for k in range(volume.zs, surface.fractalrange[1]):
                                            if (k < surface.fractalsurface[i - surface.xs, j - surface.ys] and
                                                volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] != 1):
                                                x, y = g.calculate_blade_geometry(blade, height)
                                                # Add x, y coordinates to existing location
                                                xx = int(i - volume.xs + x)
                                                yy = int(j - volume.ys + y)
                                                # If these coordinates are outwith the fractal volume stop building the blade, otherwise set the mask for grass
                                                if xx < 0 or xx >= volume.mask.shape[0] or yy < 0 or yy >= volume.mask.shape[1]:
                                                    break
                                                else:
                                                    volume.mask[xx, yy, k - volume.zs] = 3
                                                    height += 1
                                        blade += 1

                            # Build the roots of the grass
                            root = 0
                            for i in range(surface.xs, surface.xf):
                                for j in range(surface.ys, surface.yf):
                                    if surface.fractalsurface[i - surface.xs, j - surface.ys] > 0:
                                        depth = 0
                                        k = volume.zf - 1
                                        while k > volume.zs:
                                            if (k > volume.originalzf - (surface.fractalsurface[i - surface.xs, j - surface.ys] -
                                                volume.originalzf) and volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] == 1):
                                                x, y = g.calculate_root_geometry(root, depth)
                                                # Add x, y coordinates to existing location
                                                xx = int(i - volume.xs + x)
                                                yy = int(j - volume.ys + y)
                                                # If these coordinates are outwith the fractal volume stop building the root, otherwise set the mask for grass
                                                if xx < 0 or xx >= volume.mask.shape[0] or yy < 0 or yy >= volume.mask.shape[1]:
                                                    break
                                                else:
                                                    volume.mask[xx, yy, k - volume.zs] = 3
                                                    depth += 1
                                            k -= 1
                                        root += 1

                # Build voxels from any true values of the 3D mask array
                waternumID = next((x.numID for x in grid.materials if x.ID == 'water'), 0)
                grassnumID = next((x.numID for x in grid.materials if x.ID == 'grass'), 0)
                data = volume.fractalvolume.astype('int16', order='C')
                mask = volume.mask.copy(order='C')
                build_voxels_from_array_mask(volume.xs, volume.ys, volume.zs,
                                             waternumID, grassnumID, volume.averaging,
                                             mask, data, grid.solid, grid.rigidE,
                                             grid.rigidH, grid.ID)

            else:
                if volume.nbins == 1:
                    raise CmdInputError(self.__str__() + ' is being used with a single material and no modifications, therefore please use a #box command instead.')
                else:
                    volume.generate_fractal_volume(grid)
                    volume.fractalvolume += volume.mixingmodel.startmaterialnum

                data = volume.fractalvolume.astype('int16', order='C')
                build_voxels_from_array(volume.xs, volume.ys, volume.zs, 0,
                                        volume.averaging, data, grid.solid,
                                        grid.rigidE, grid.rigidH, grid.ID)
