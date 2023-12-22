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
from tqdm import tqdm

from gprMax.constants import floattype
from gprMax.input_cmds_file import check_cmd_names
from gprMax.input_cmds_multiuse import process_multicmds
from gprMax.exceptions import CmdInputError
from gprMax.fractals import FractalSurface
from gprMax.fractals import FractalVolume
from gprMax.fractals import Grass
from gprMax.geometry_primitives_ext import build_edge_x
from gprMax.geometry_primitives_ext import build_edge_y
from gprMax.geometry_primitives_ext import build_edge_z
from gprMax.geometry_primitives_ext import build_face_yz
from gprMax.geometry_primitives_ext import build_face_xz
from gprMax.geometry_primitives_ext import build_face_xy
from gprMax.geometry_primitives_ext import build_triangle
from gprMax.geometry_primitives_ext import build_box
from gprMax.geometry_primitives_ext import build_cylinder
from gprMax.geometry_primitives_ext import build_cylindrical_sector
from gprMax.geometry_primitives_ext import build_sphere
from gprMax.geometry_primitives_ext import build_voxels_from_array
from gprMax.geometry_primitives_ext import build_voxels_from_array_mask
from gprMax.materials import Material
from gprMax.utilities import round_value
from gprMax.utilities import get_terminal_width


def process_geometrycmds(geometry, G):
    """
    This function checks the validity of command parameters, creates instances
    of classes of parameters, and calls functions to directly set arrays
    solid, rigid and ID.

    Args:
        geometry (list): Geometry commands in the model
    """

    # Disable progress bar if on Windows as it does not update properly
    # when messages are printed for geometry
    if sys.platform == 'win32':
        progressbars = False
    else:
        progressbars = not G.progressbars

    for object in tqdm(geometry, desc='Processing geometry related cmds', unit='cmds', ncols=get_terminal_width() - 1, file=sys.stdout, disable=progressbars):
        tmp = object.split()

        if tmp[0] == '#geometry_objects_read:':
            if len(tmp) != 6:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires exactly five parameters')

            xs = round_value(float(tmp[1]) / G.dx)
            ys = round_value(float(tmp[2]) / G.dy)
            zs = round_value(float(tmp[3]) / G.dz)
            geofile = tmp[4]
            matfile = tmp[5]

            # See if material file exists at specified path and if not try input
            # file directory
            if not os.path.isfile(matfile):
                matfile = os.path.abspath(os.path.join(G.inputdirectory, matfile))

            matstr = os.path.splitext(os.path.split(matfile)[1])[0]
            numexistmaterials = len(G.materials)

            # Read materials from file
            with open(matfile, 'r') as f:
                # Read any lines that begin with a hash. Strip out any newline
                # characters and comments that must begin with double hashes.
                materials = [line.rstrip() + '{' + matstr + '}\n' for line in f if(line.startswith('#') and not line.startswith('##') and line.rstrip('\n'))]

            # Check validity of command names
            singlecmdsimport, multicmdsimport, geometryimport = check_cmd_names(materials, checkessential=False)

            # Process parameters for commands that can occur multiple times in the model
            process_multicmds(multicmdsimport, G)

            # Update material type
            for material in G.materials:
                if material.numID >= numexistmaterials:
                    if material.type:
                        material.type += ',\nimported'
                    else:
                        material.type = 'imported'

            # See if geometry object file exists at specified path and if not try input file directory
            if not os.path.isfile(geofile):
                geofile = os.path.abspath(os.path.join(G.inputdirectory, geofile))

            # Open geometry object file and read/check spatial resolution attribute
            f = h5py.File(geofile, 'r')
            dx_dy_dz = f.attrs['dx_dy_dz']
            if round_value(dx_dy_dz[0] / G.dx) != 1 or round_value(dx_dy_dz[1] / G.dy) != 1 or round_value(dx_dy_dz[2] / G.dz) != 1:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires the spatial resolution of the geometry objects file to match the spatial resolution of the model')

            data = f['/data'][:]

            # Should be int16 to allow for -1 which indicates background, i.e.
            # don't build anything, but AustinMan/Woman maybe uint16
            if data.dtype != 'int16':
                data = data.astype('int16')

            # Check that there are no values in the data greater than the maximum index for the specified materials
            if np.amax(data) > len(materials) - 1:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' found data value(s) ({}) in the geometry objects file greater than the maximum index for the specified materials ({})'.format(np.amax(data), len(materials) - 1))

            # Look to see if rigid and ID arrays are present (these should be
            # present if the original geometry objects were written from gprMax)
            try:
                rigidE = f['/rigidE'][:]
                rigidH = f['/rigidH'][:]
                ID = f['/ID'][:]
                G.solid[xs:xs + data.shape[0], ys:ys + data.shape[1], zs:zs + data.shape[2]] = data + numexistmaterials
                G.rigidE[:, xs:xs + rigidE.shape[1], ys:ys + rigidE.shape[2], zs:zs + rigidE.shape[3]] = rigidE
                G.rigidH[:, xs:xs + rigidH.shape[1], ys:ys + rigidH.shape[2], zs:zs + rigidH.shape[3]] = rigidH
                G.ID[:, xs:xs + ID.shape[1], ys:ys + ID.shape[2], zs:zs + ID.shape[3]] = ID + numexistmaterials
                if G.messages:
                    tqdm.write('Geometry objects from file {} inserted at {:g}m, {:g}m, {:g}m, with corresponding materials file {}.'.format(geofile, xs * G.dx, ys * G.dy, zs * G.dz, matfile))
            except KeyError:
                averaging = False
                build_voxels_from_array(xs, ys, zs, numexistmaterials, averaging, data, G.solid, G.rigidE, G.rigidH, G.ID)
                if G.messages:
                    tqdm.write('Geometry objects from file (voxels only) {} inserted at {:g}m, {:g}m, {:g}m, with corresponding materials file {}.'.format(geofile, xs * G.dx, ys * G.dy, zs * G.dz, matfile))

        elif tmp[0] == '#edge:':
            if len(tmp) != 8:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires exactly seven parameters')

            xs = round_value(float(tmp[1]) / G.dx)
            xf = round_value(float(tmp[4]) / G.dx)
            ys = round_value(float(tmp[2]) / G.dy)
            yf = round_value(float(tmp[5]) / G.dy)
            zs = round_value(float(tmp[3]) / G.dz)
            zf = round_value(float(tmp[6]) / G.dz)

            if xs < 0 or xs > G.nx:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs * G.dx))
            if xf < 0 or xf > G.nx:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf * G.dx))
            if ys < 0 or ys > G.ny:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys * G.dy))
            if yf < 0 or yf > G.ny:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf * G.dy))
            if zs < 0 or zs > G.nz:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs * G.dz))
            if zf < 0 or zf > G.nz:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf * G.dz))
            if xs > xf or ys > yf or zs > zf:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')

            material = next((x for x in G.materials if x.ID == tmp[7]), None)

            if not material:
                raise CmdInputError('Material with ID {} does not exist'.format(tmp[7]))

            # Check for valid orientations
            # x-orientated wire
            if xs != xf:
                if ys != yf or zs != zf:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the edge is not specified correctly')
                else:
                    for i in range(xs, xf):
                        build_edge_x(i, ys, zs, material.numID, G.rigidE, G.rigidH, G.ID)

            # y-orientated wire
            elif ys != yf:
                if xs != xf or zs != zf:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the edge is not specified correctly')
                else:
                    for j in range(ys, yf):
                        build_edge_y(xs, j, zs, material.numID, G.rigidE, G.rigidH, G.ID)

            # z-orientated wire
            elif zs != zf:
                if xs != xf or ys != yf:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the edge is not specified correctly')
                else:
                    for k in range(zs, zf):
                        build_edge_z(xs, ys, k, material.numID, G.rigidE, G.rigidH, G.ID)

            if G.messages:
                tqdm.write('Edge from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m of material {} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, tmp[7]))

        elif tmp[0] == '#plate:':
            if len(tmp) < 8:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires at least seven parameters')

            # Isotropic case
            elif len(tmp) == 8:
                materialsrequested = [tmp[7]]

            # Anisotropic case
            elif len(tmp) == 9:
                materialsrequested = [tmp[7:]]

            else:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')

            xs = round_value(float(tmp[1]) / G.dx)
            xf = round_value(float(tmp[4]) / G.dx)
            ys = round_value(float(tmp[2]) / G.dy)
            yf = round_value(float(tmp[5]) / G.dy)
            zs = round_value(float(tmp[3]) / G.dz)
            zf = round_value(float(tmp[6]) / G.dz)

            if xs < 0 or xs > G.nx:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs * G.dx))
            if xf < 0 or xf > G.nx:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf * G.dx))
            if ys < 0 or ys > G.ny:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys * G.dy))
            if yf < 0 or yf > G.ny:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf * G.dy))
            if zs < 0 or zs > G.nz:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs * G.dz))
            if zf < 0 or zf > G.nz:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf * G.dz))
            if xs > xf or ys > yf or zs > zf:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')

            # Check for valid orientations
            if xs == xf:
                if ys == yf or zs == zf:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the plate is not specified correctly')

            elif ys == yf:
                if xs == xf or zs == zf:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the plate is not specified correctly')

            elif zs == zf:
                if xs == xf or ys == yf:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the plate is not specified correctly')

            else:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the plate is not specified correctly')

            # Look up requested materials in existing list of material instances
            materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

            if len(materials) != len(materialsrequested):
                notfound = [x for x in materialsrequested if x not in materials]
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' material(s) {} do not exist'.format(notfound))

            # yz-plane plate
            if xs == xf:
                # Isotropic case
                if len(materials) == 1:
                    numIDx = numIDy = numIDz = materials[0].numID

                # Uniaxial anisotropic case
                elif len(materials) == 2:
                    numIDy = materials[0].numID
                    numIDz = materials[1].numID

                for j in range(ys, yf):
                    for k in range(zs, zf):
                        build_face_yz(xs, j, k, numIDy, numIDz, G.rigidE, G.rigidH, G.ID)

            # xz-plane plate
            elif ys == yf:
                # Isotropic case
                if len(materials) == 1:
                    numIDx = numIDy = numIDz = materials[0].numID

                # Uniaxial anisotropic case
                elif len(materials) == 2:
                    numIDx = materials[0].numID
                    numIDz = materials[1].numID

                for i in range(xs, xf):
                    for k in range(zs, zf):
                        build_face_xz(i, ys, k, numIDx, numIDz, G.rigidE, G.rigidH, G.ID)

            # xy-plane plate
            elif zs == zf:
                # Isotropic case
                if len(materials) == 1:
                    numIDx = numIDy = numIDz = materials[0].numID

                # Uniaxial anisotropic case
                elif len(materials) == 2:
                    numIDx = materials[0].numID
                    numIDy = materials[1].numID

                for i in range(xs, xf):
                    for j in range(ys, yf):
                        build_face_xy(i, j, zs, numIDx, numIDy, G.rigidE, G.rigidH, G.ID)

            if G.messages:
                tqdm.write('Plate from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m of material(s) {} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, ', '.join(materialsrequested)))

        elif tmp[0] == '#triangle:':
            if len(tmp) < 12:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires at least eleven parameters')

            # Isotropic case with no user specified averaging
            elif len(tmp) == 12:
                materialsrequested = [tmp[11]]
                averagetriangularprism = G.averagevolumeobjects

            # Isotropic case with user specified averaging
            elif len(tmp) == 13:
                materialsrequested = [tmp[11]]
                if tmp[12].lower() == 'y':
                    averagetriangularprism = True
                elif tmp[12].lower() == 'n':
                    averagetriangularprism = False
                else:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires averaging to be either y or n')

            # Uniaxial anisotropic case
            elif len(tmp) == 14:
                materialsrequested = tmp[11:]

            else:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')

            x1 = round_value(float(tmp[1]) / G.dx) * G.dx
            y1 = round_value(float(tmp[2]) / G.dy) * G.dy
            z1 = round_value(float(tmp[3]) / G.dz) * G.dz
            x2 = round_value(float(tmp[4]) / G.dx) * G.dx
            y2 = round_value(float(tmp[5]) / G.dy) * G.dy
            z2 = round_value(float(tmp[6]) / G.dz) * G.dz
            x3 = round_value(float(tmp[7]) / G.dx) * G.dx
            y3 = round_value(float(tmp[8]) / G.dy) * G.dy
            z3 = round_value(float(tmp[9]) / G.dz) * G.dz
            thickness = float(tmp[10])

            if x1 < 0 or x2 < 0 or x3 < 0 or x1 > G.nx or x2 > G.nx or x3 > G.nx:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the one of the x-coordinates is not within the model domain')
            if y1 < 0 or y2 < 0 or y3 < 0 or y1 > G.ny or y2 > G.ny or y3 > G.ny:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the one of the y-coordinates is not within the model domain')
            if z1 < 0 or z2 < 0 or z3 < 0 or z1 > G.nz or z2 > G.nz or z3 > G.nz:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the one of the z-coordinates is not within the model domain')
            if thickness < 0:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for thickness')

            # Check for valid orientations
            # yz-plane triangle
            if x1 == x2 and x2 == x3:
                normal = 'x'
            # xz-plane triangle
            elif y1 == y2 and y2 == y3:
                normal = 'y'
            # xy-plane triangle
            elif z1 == z2 and z2 == z3:
                normal = 'z'
            else:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the triangle is not specified correctly')

            # Look up requested materials in existing list of material instances
            materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

            if len(materials) != len(materialsrequested):
                notfound = [x for x in materialsrequested if x not in materials]
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' material(s) {} do not exist'.format(notfound))

            if thickness > 0:
                # Isotropic case
                if len(materials) == 1:
                    averaging = materials[0].averagable and averagetriangularprism
                    numID = numIDx = numIDy = numIDz = materials[0].numID

                # Uniaxial anisotropic case
                elif len(materials) == 3:
                    averaging = False
                    numIDx = materials[0].numID
                    numIDy = materials[1].numID
                    numIDz = materials[2].numID
                    requiredID = materials[0].ID + '+' + materials[1].ID + '+' + materials[2].ID
                    averagedmaterial = [x for x in G.materials if x.ID == requiredID]
                    if averagedmaterial:
                        numID = averagedmaterial.numID
                    else:
                        numID = len(G.materials)
                        m = Material(numID, requiredID)
                        m.type = 'dielectric-smoothed'
                        # Create dielectric-smoothed constituents for material
                        m.er = np.mean((materials[0].er, materials[1].er, materials[2].er), axis=0)
                        m.se = np.mean((materials[0].se, materials[1].se, materials[2].se), axis=0)
                        m.mr = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)
                        m.sm = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)

                        # Append the new material object to the materials list
                        G.materials.append(m)
            else:
                averaging = False
                # Isotropic case
                if len(materials) == 1:
                    numID = numIDx = numIDy = numIDz = materials[0].numID

                # Uniaxial anisotropic case
                elif len(materials) == 3:
                    # numID requires a value but it will not be used
                    numID = None
                    numIDx = materials[0].numID
                    numIDy = materials[1].numID
                    numIDz = materials[2].numID

            build_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, normal, thickness, G.dx, G.dy, G.dz, numID, numIDx, numIDy, numIDz, averaging, G.solid, G.rigidE, G.rigidH, G.ID)

            if G.messages:
                if thickness > 0:
                    if averaging:
                        dielectricsmoothing = 'on'
                    else:
                        dielectricsmoothing = 'off'
                    tqdm.write('Triangle with coordinates {:g}m {:g}m {:g}m, {:g}m {:g}m {:g}m, {:g}m {:g}m {:g}m and thickness {:g}m of material(s) {} created, dielectric smoothing is {}.'.format(x1, y1, z1, x2, y2, z2, x3, y3, z3, thickness, ', '.join(materialsrequested), dielectricsmoothing))
                else:
                    tqdm.write('Triangle with coordinates {:g}m {:g}m {:g}m, {:g}m {:g}m {:g}m, {:g}m {:g}m {:g}m of material(s) {} created.'.format(x1, y1, z1, x2, y2, z2, x3, y3, z3, ', '.join(materialsrequested)))

        elif tmp[0] == '#box:':
            if len(tmp) < 8:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires at least seven parameters')

            # Isotropic case with no user specified averaging
            elif len(tmp) == 8:
                materialsrequested = [tmp[7]]
                averagebox = G.averagevolumeobjects

            # Isotropic case with user specified averaging
            elif len(tmp) == 9:
                materialsrequested = [tmp[7]]
                if tmp[8].lower() == 'y':
                    averagebox = True
                elif tmp[8].lower() == 'n':
                    averagebox = False
                else:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires averaging to be either y or n')

            # Uniaxial anisotropic case
            elif len(tmp) == 10:
                materialsrequested = tmp[7:]

            else:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')

            xs = round_value(float(tmp[1]) / G.dx)
            xf = round_value(float(tmp[4]) / G.dx)
            ys = round_value(float(tmp[2]) / G.dy)
            yf = round_value(float(tmp[5]) / G.dy)
            zs = round_value(float(tmp[3]) / G.dz)
            zf = round_value(float(tmp[6]) / G.dz)

            if xs < 0 or xs > G.nx:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs * G.dx))
            if xf < 0 or xf > G.nx:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf * G.dx))
            if ys < 0 or ys > G.ny:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys * G.dy))
            if yf < 0 or yf > G.ny:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf * G.dy))
            if zs < 0 or zs > G.nz:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs * G.dz))
            if zf < 0 or zf > G.nz:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf * G.dz))
            if xs >= xf or ys >= yf or zs >= zf:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')

            # Look up requested materials in existing list of material instances
            materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

            if len(materials) != len(materialsrequested):
                notfound = [x for x in materialsrequested if x not in materials]
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' material(s) {} do not exist'.format(notfound))

            # Isotropic case
            if len(materials) == 1:
                averaging = materials[0].averagable and averagebox
                numID = numIDx = numIDy = numIDz = materials[0].numID

            # Uniaxial anisotropic case
            elif len(materials) == 3:
                averaging = False
                numIDx = materials[0].numID
                numIDy = materials[1].numID
                numIDz = materials[2].numID
                requiredID = materials[0].ID + '+' + materials[1].ID + '+' + materials[2].ID
                averagedmaterial = [x for x in G.materials if x.ID == requiredID]
                if averagedmaterial:
                    numID = averagedmaterial.numID
                else:
                    numID = len(G.materials)
                    m = Material(numID, requiredID)
                    m.type = 'dielectric-smoothed'
                    # Create dielectric-smoothed constituents for material
                    m.er = np.mean((materials[0].er, materials[1].er, materials[2].er), axis=0)
                    m.se = np.mean((materials[0].se, materials[1].se, materials[2].se), axis=0)
                    m.mr = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)
                    m.sm = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)

                    # Append the new material object to the materials list
                    G.materials.append(m)

            build_box(xs, xf, ys, yf, zs, zf, numID, numIDx, numIDy, numIDz, averaging, G.solid, G.rigidE, G.rigidH, G.ID)

            if G.messages:
                if averaging:
                    dielectricsmoothing = 'on'
                else:
                    dielectricsmoothing = 'off'
                tqdm.write('Box from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m of material(s) {} created, dielectric smoothing is {}.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, ', '.join(materialsrequested), dielectricsmoothing))

        elif tmp[0] == '#cylinder:':
            if len(tmp) < 9:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires at least eight parameters')

            # Isotropic case with no user specified averaging
            elif len(tmp) == 9:
                materialsrequested = [tmp[8]]
                averagecylinder = G.averagevolumeobjects

            # Isotropic case with user specified averaging
            elif len(tmp) == 10:
                materialsrequested = [tmp[8]]
                if tmp[9].lower() == 'y':
                    averagecylinder = True
                elif tmp[9].lower() == 'n':
                    averagecylinder = False
                else:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires averaging to be either y or n')

            # Uniaxial anisotropic case
            elif len(tmp) == 11:
                materialsrequested = tmp[8:]

            else:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')

            x1 = round_value(float(tmp[1]) / G.dx) * G.dx
            y1 = round_value(float(tmp[2]) / G.dy) * G.dy
            z1 = round_value(float(tmp[3]) / G.dz) * G.dz
            x2 = round_value(float(tmp[4]) / G.dx) * G.dx
            y2 = round_value(float(tmp[5]) / G.dy) * G.dy
            z2 = round_value(float(tmp[6]) / G.dz) * G.dz
            r = float(tmp[7])

            if r <= 0:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the radius {:g} should be a positive value.'.format(r))

            # Look up requested materials in existing list of material instances
            materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

            if len(materials) != len(materialsrequested):
                notfound = [x for x in materialsrequested if x not in materials]
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' material(s) {} do not exist'.format(notfound))

            # Isotropic case
            if len(materials) == 1:
                averaging = materials[0].averagable and averagecylinder
                numID = numIDx = numIDy = numIDz = materials[0].numID

            # Uniaxial anisotropic case
            elif len(materials) == 3:
                averaging = False
                numIDx = materials[0].numID
                numIDy = materials[1].numID
                numIDz = materials[2].numID
                requiredID = materials[0].ID + '+' + materials[1].ID + '+' + materials[2].ID
                averagedmaterial = [x for x in G.materials if x.ID == requiredID]
                if averagedmaterial:
                    numID = averagedmaterial.numID
                else:
                    numID = len(G.materials)
                    m = Material(numID, requiredID)
                    m.type = 'dielectric-smoothed'
                    # Create dielectric-smoothed constituents for material
                    m.er = np.mean((materials[0].er, materials[1].er, materials[2].er), axis=0)
                    m.se = np.mean((materials[0].se, materials[1].se, materials[2].se), axis=0)
                    m.mr = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)
                    m.sm = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)

                    # Append the new material object to the materials list
                    G.materials.append(m)

            build_cylinder(x1, y1, z1, x2, y2, z2, r, G.dx, G.dy, G.dz, numID, numIDx, numIDy, numIDz, averaging, G.solid, G.rigidE, G.rigidH, G.ID)

            if G.messages:
                if averaging:
                    dielectricsmoothing = 'on'
                else:
                    dielectricsmoothing = 'off'
                tqdm.write('Cylinder with face centres {:g}m, {:g}m, {:g}m and {:g}m, {:g}m, {:g}m, with radius {:g}m, of material(s) {} created, dielectric smoothing is {}.'.format(x1, y1, z1, x2, y2, z2, r, ', '.join(materialsrequested), dielectricsmoothing))

        elif tmp[0] == '#cylindrical_sector:':
            if len(tmp) < 10:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires at least nine parameters')

            # Isotropic case with no user specified averaging
            elif len(tmp) == 10:
                materialsrequested = [tmp[9]]
                averagecylindricalsector = G.averagevolumeobjects

            # Isotropic case with user specified averaging
            elif len(tmp) == 11:
                materialsrequested = [tmp[9]]
                if tmp[10].lower() == 'y':
                    averagecylindricalsector = True
                elif tmp[10].lower() == 'n':
                    averagecylindricalsector = False
                else:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires averaging to be either y or n')

            # Uniaxial anisotropic case
            elif len(tmp) == 12:
                materialsrequested = tmp[9:]

            else:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')

            normal = tmp[1].lower()
            ctr1 = float(tmp[2])
            ctr2 = float(tmp[3])
            extent1 = float(tmp[4])
            extent2 = float(tmp[5])
            thickness = extent2 - extent1
            r = float(tmp[6])
            sectorstartangle = 2 * np.pi * (float(tmp[7]) / 360)
            sectorangle = 2 * np.pi * (float(tmp[8]) / 360)

            if normal != 'x' and normal != 'y' and normal != 'z':
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the normal direction must be either x, y or z.')
            if r <= 0:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the radius {:g} should be a positive value.'.format(r))
            if sectorstartangle < 0 or sectorangle <= 0:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the starting angle and sector angle should be a positive values.')
            if sectorstartangle >= 2 * np.pi or sectorangle >= 2 * np.pi:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the starting angle and sector angle must be less than 360 degrees.')

            # Look up requested materials in existing list of material instances
            materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

            if len(materials) != len(materialsrequested):
                notfound = [x for x in materialsrequested if x not in materials]
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' material(s) {} do not exist'.format(notfound))

            if thickness > 0:
                # Isotropic case
                if len(materials) == 1:
                    averaging = materials[0].averagable and averagecylindricalsector
                    numID = numIDx = numIDy = numIDz = materials[0].numID

                # Uniaxial anisotropic case
                elif len(materials) == 3:
                    averaging = False
                    numIDx = materials[0].numID
                    numIDy = materials[1].numID
                    numIDz = materials[2].numID
                    requiredID = materials[0].ID + '+' + materials[1].ID + '+' + materials[2].ID
                    averagedmaterial = [x for x in G.materials if x.ID == requiredID]
                    if averagedmaterial:
                        numID = averagedmaterial.numID
                    else:
                        numID = len(G.materials)
                        m = Material(numID, requiredID)
                        m.type = 'dielectric-smoothed'
                        # Create dielectric-smoothed constituents for material
                        m.er = np.mean((materials[0].er, materials[1].er, materials[2].er), axis=0)
                        m.se = np.mean((materials[0].se, materials[1].se, materials[2].se), axis=0)
                        m.mr = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)
                        m.sm = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)

                        # Append the new material object to the materials list
                        G.materials.append(m)
            else:
                averaging = False
                # Isotropic case
                if len(materials) == 1:
                    numID = numIDx = numIDy = numIDz = materials[0].numID

                # Uniaxial anisotropic case
                elif len(materials) == 3:
                    # numID requires a value but it will not be used
                    numID = None
                    numIDx = materials[0].numID
                    numIDy = materials[1].numID
                    numIDz = materials[2].numID

            # yz-plane cylindrical sector
            if normal == 'x':
                ctr1 = round_value(ctr1 / G.dy) * G.dy
                ctr2 = round_value(ctr2 / G.dz) * G.dz
                level = round_value(extent1 / G.dx)

            # xz-plane cylindrical sector
            elif normal == 'y':
                ctr1 = round_value(ctr1 / G.dx) * G.dx
                ctr2 = round_value(ctr2 / G.dz) * G.dz
                level = round_value(extent1 / G.dy)

            # xy-plane cylindrical sector
            elif normal == 'z':
                ctr1 = round_value(ctr1 / G.dx) * G.dx
                ctr2 = round_value(ctr2 / G.dy) * G.dy
                level = round_value(extent1 / G.dz)

            build_cylindrical_sector(ctr1, ctr2, level, sectorstartangle, sectorangle, r, normal, thickness, G.dx, G.dy, G.dz, numID, numIDx, numIDy, numIDz, averaging, G.solid, G.rigidE, G.rigidH, G.ID)

            if G.messages:
                if thickness > 0:
                    if averaging:
                        dielectricsmoothing = 'on'
                    else:
                        dielectricsmoothing = 'off'
                    tqdm.write('Cylindrical sector with centre {:g}m, {:g}m, radius {:g}m, starting angle {:.1f} degrees, sector angle {:.1f} degrees, thickness {:g}m, of material(s) {} created, dielectric smoothing is {}.'.format(ctr1, ctr2, r, (sectorstartangle / (2 * np.pi)) * 360, (sectorangle / (2 * np.pi)) * 360, thickness, ', '.join(materialsrequested), dielectricsmoothing))
                else:
                    tqdm.write('Cylindrical sector with centre {:g}m, {:g}m, radius {:g}m, starting angle {:.1f} degrees, sector angle {:.1f} degrees, of material(s) {} created.'.format(ctr1, ctr2, r, (sectorstartangle / (2 * np.pi)) * 360, (sectorangle / (2 * np.pi)) * 360, ', '.join(materialsrequested)))

        elif tmp[0] == '#sphere:':
            if len(tmp) < 6:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires at least five parameters')

            # Isotropic case with no user specified averaging
            elif len(tmp) == 6:
                materialsrequested = [tmp[5]]
                averagesphere = G.averagevolumeobjects

            # Isotropic case with user specified averaging
            elif len(tmp) == 7:
                materialsrequested = [tmp[5]]
                if tmp[6].lower() == 'y':
                    averagesphere = True
                elif tmp[6].lower() == 'n':
                    averagesphere = False
                else:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires averaging to be either y or n')

            # Uniaxial anisotropic case
            elif len(tmp) == 8:
                materialsrequested = tmp[5:]

            else:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')

            # Centre of sphere
            xc = round_value(float(tmp[1]) / G.dx)
            yc = round_value(float(tmp[2]) / G.dy)
            zc = round_value(float(tmp[3]) / G.dz)
            r = float(tmp[4])

            # Look up requested materials in existing list of material instances
            materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

            if len(materials) != len(materialsrequested):
                notfound = [x for x in materialsrequested if x not in materials]
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' material(s) {} do not exist'.format(notfound))

            # Isotropic case
            if len(materials) == 1:
                averaging = materials[0].averagable and averagesphere
                numID = numIDx = numIDy = numIDz = materials[0].numID

            # Uniaxial anisotropic case
            elif len(materials) == 3:
                averaging = False
                numIDx = materials[0].numID
                numIDy = materials[1].numID
                numIDz = materials[2].numID
                requiredID = materials[0].ID + '+' + materials[1].ID + '+' + materials[2].ID
                averagedmaterial = [x for x in G.materials if x.ID == requiredID]
                if averagedmaterial:
                    numID = averagedmaterial.numID
                else:
                    numID = len(G.materials)
                    m = Material(numID, requiredID)
                    m.type = 'dielectric-smoothed'
                    # Create dielectric-smoothed constituents for material
                    m.er = np.mean((materials[0].er, materials[1].er, materials[2].er), axis=0)
                    m.se = np.mean((materials[0].se, materials[1].se, materials[2].se), axis=0)
                    m.mr = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)
                    m.sm = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)

                    # Append the new material object to the materials list
                    G.materials.append(m)

            build_sphere(xc, yc, zc, r, G.dx, G.dy, G.dz, numID, numIDx, numIDy, numIDz, averaging, G.solid, G.rigidE, G.rigidH, G.ID)

            if G.messages:
                if averaging:
                    dielectricsmoothing = 'on'
                else:
                    dielectricsmoothing = 'off'
                tqdm.write('Sphere with centre {:g}m, {:g}m, {:g}m, radius {:g}m, of material(s) {} created, dielectric smoothing is {}.'.format(xc * G.dx, yc * G.dy, zc * G.dz, r, ', '.join(materialsrequested), dielectricsmoothing))

        elif tmp[0] == '#fractal_box:':
            # Default is no dielectric smoothing for a fractal box
            averagefractalbox = False

            if len(tmp) < 14:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires at least thirteen parameters')
            elif len(tmp) == 14:
                seed = None
            elif len(tmp) == 15:
                seed = int(tmp[14])
            elif len(tmp) == 16:
                seed = int(tmp[14])
                if tmp[15].lower() == 'y':
                    averagefractalbox = True
                elif tmp[15].lower() == 'n':
                    averagefractalbox = False
                else:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires averaging to be either y or n')
            else:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')

            xs = round_value(float(tmp[1]) / G.dx)
            xf = round_value(float(tmp[4]) / G.dx)
            ys = round_value(float(tmp[2]) / G.dy)
            yf = round_value(float(tmp[5]) / G.dy)
            zs = round_value(float(tmp[3]) / G.dz)
            zf = round_value(float(tmp[6]) / G.dz)

            if xs < 0 or xs > G.nx:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs * G.dx))
            if xf < 0 or xf > G.nx:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf * G.dx))
            if ys < 0 or ys > G.ny:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys * G.dy))
            if yf < 0 or yf > G.ny:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf * G.dy))
            if zs < 0 or zs > G.nz:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs * G.dz))
            if zf < 0 or zf > G.nz:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf * G.dz))
            if xs >= xf or ys >= yf or zs >= zf:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
            if float(tmp[7]) < 0:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the fractal dimension')
            if float(tmp[8]) < 0:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the fractal weighting in the x direction')
            if float(tmp[9]) < 0:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the fractal weighting in the y direction')
            if float(tmp[10]) < 0:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the fractal weighting in the z direction')
            if round_value(tmp[11]) < 0:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the number of bins')

            # Find materials to use to build fractal volume, either from mixing models or normal materials
            mixingmodel = next((x for x in G.mixingmodels if x.ID == tmp[12]), None)
            material = next((x for x in G.materials if x.ID == tmp[12]), None)
            nbins = round_value(tmp[11])

            if mixingmodel:
                if nbins == 1:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' must be used with more than one material from the mixing model.')
                # Create materials from mixing model as number of bins now known from fractal_box command
                mixingmodel.calculate_debye_properties(nbins, G, tmp[13])
            elif not material:
                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' mixing model or material with ID {} does not exist'.format(tmp[12]))

            volume = FractalVolume(xs, xf, ys, yf, zs, zf, float(tmp[7]), seed)
            volume.ID = tmp[13]
            volume.operatingonID = tmp[12]
            volume.nbins = nbins
            volume.weighting = np.array([float(tmp[8]), float(tmp[9]), float(tmp[10])])
            volume.averaging = averagefractalbox

            if G.messages:
                if volume.averaging:
                    dielectricsmoothing = 'on'
                else:
                    dielectricsmoothing = 'off'
                tqdm.write('Fractal box {} from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with {}, fractal dimension {:g}, fractal weightings {:g}, {:g}, {:g}, fractal seeding {}, with {} material(s) created, dielectric smoothing is {}.'.format(volume.ID, xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, volume.operatingonID, volume.dimension, volume.weighting[0], volume.weighting[1], volume.weighting[2], volume.seed, volume.nbins, dielectricsmoothing))

            G.fractalvolumes.append(volume)

            # Search and process any modifiers for the fractal box
            for object in geometry:
                tmp = object.split()

                if tmp[0] == '#add_surface_roughness:':
                    if len(tmp) < 13:
                        raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires at least twelve parameters')
                    elif len(tmp) == 13:
                        seed = None
                    elif len(tmp) == 14:
                        seed = int(tmp[13])
                    else:
                        raise CmdInputError("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')

                    # Only process rough surfaces for this fractal volume
                    if tmp[12] == volume.ID:
                        xs = round_value(float(tmp[1]) / G.dx)
                        xf = round_value(float(tmp[4]) / G.dx)
                        ys = round_value(float(tmp[2]) / G.dy)
                        yf = round_value(float(tmp[5]) / G.dy)
                        zs = round_value(float(tmp[3]) / G.dz)
                        zf = round_value(float(tmp[6]) / G.dz)

                        if xs < 0 or xs > G.nx:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs * G.dx))
                        if xf < 0 or xf > G.nx:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf * G.dx))
                        if ys < 0 or ys > G.ny:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys * G.dy))
                        if yf < 0 or yf > G.ny:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf * G.dy))
                        if zs < 0 or zs > G.nz:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs * G.dz))
                        if zf < 0 or zf > G.nz:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf * G.dz))
                        if xs > xf or ys > yf or zs > zf:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
                        if float(tmp[7]) < 0:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the fractal dimension')
                        if float(tmp[8]) < 0:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the fractal weighting in the first direction of the surface')
                        if float(tmp[9]) < 0:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the fractal weighting in the second direction of the surface')

                        # Check for valid orientations
                        if xs == xf:
                            if ys == yf or zs == zf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')
                            if xs != volume.xs and xs != volume.xf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' can only be used on the external surfaces of a fractal box')
                            fractalrange = (round_value(float(tmp[10]) / G.dx), round_value(float(tmp[11]) / G.dx))
                            # xminus surface
                            if xs == volume.xs:
                                if fractalrange[0] < 0 or fractalrange[1] > volume.xf:
                                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' cannot apply fractal surface to fractal box as it would exceed either the upper coordinates of the fractal box or the domain in the x direction')
                                requestedsurface = 'xminus'
                            # xplus surface
                            elif xf == volume.xf:
                                if fractalrange[0] < volume.xs or fractalrange[1] > G.nx:
                                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' cannot apply fractal surface to fractal box as it would exceed either the lower coordinates of the fractal box or the domain in the x direction')
                                requestedsurface = 'xplus'

                        elif ys == yf:
                            if xs == xf or zs == zf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')
                            if ys != volume.ys and ys != volume.yf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' can only be used on the external surfaces of a fractal box')
                            fractalrange = (round_value(float(tmp[10]) / G.dy), round_value(float(tmp[11]) / G.dy))
                            # yminus surface
                            if ys == volume.ys:
                                if fractalrange[0] < 0 or fractalrange[1] > volume.yf:
                                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' cannot apply fractal surface to fractal box as it would exceed either the upper coordinates of the fractal box or the domain in the y direction')
                                requestedsurface = 'yminus'
                            # yplus surface
                            elif yf == volume.yf:
                                if fractalrange[0] < volume.ys or fractalrange[1] > G.ny:
                                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' cannot apply fractal surface to fractal box as it would exceed either the lower coordinates of the fractal box or the domain in the y direction')
                                requestedsurface = 'yplus'

                        elif zs == zf:
                            if xs == xf or ys == yf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')
                            if zs != volume.zs and zs != volume.zf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' can only be used on the external surfaces of a fractal box')
                            fractalrange = (round_value(float(tmp[10]) / G.dz), round_value(float(tmp[11]) / G.dz))
                            # zminus surface
                            if zs == volume.zs:
                                if fractalrange[0] < 0 or fractalrange[1] > volume.zf:
                                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' cannot apply fractal surface to fractal box as it would exceed either the upper coordinates of the fractal box or the domain in the x direction')
                                requestedsurface = 'zminus'
                            # zplus surface
                            elif zf == volume.zf:
                                if fractalrange[0] < volume.zs or fractalrange[1] > G.nz:
                                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' cannot apply fractal surface to fractal box as it would exceed either the lower coordinates of the fractal box or the domain in the z direction')
                                requestedsurface = 'zplus'

                        else:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')

                        surface = FractalSurface(xs, xf, ys, yf, zs, zf, float(tmp[7]), seed)
                        surface.surfaceID = requestedsurface
                        surface.fractalrange = fractalrange
                        surface.operatingonID = volume.ID
                        surface.weighting = np.array([float(tmp[8]), float(tmp[9])])

                        # List of existing surfaces IDs
                        existingsurfaceIDs = [x.surfaceID for x in volume.fractalsurfaces]
                        if surface.surfaceID in existingsurfaceIDs:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' has already been used on the {} surface'.format(surface.surfaceID))

                        surface.generate_fractal_surface(G)
                        volume.fractalsurfaces.append(surface)

                        if G.messages:
                            tqdm.write('Fractal surface from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with fractal dimension {:g}, fractal weightings {:g}, {:g}, fractal seeding {}, and range {:g}m to {:g}m, added to {}.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, surface.dimension, surface.weighting[0], surface.weighting[1], surface.seed, float(tmp[10]), float(tmp[11]), surface.operatingonID))

                if tmp[0] == '#add_surface_water:':
                    if len(tmp) != 9:
                        raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires exactly eight parameters')

                    # Only process surfaces for this fractal volume
                    if tmp[8] == volume.ID:
                        xs = round_value(float(tmp[1]) / G.dx)
                        xf = round_value(float(tmp[4]) / G.dx)
                        ys = round_value(float(tmp[2]) / G.dy)
                        yf = round_value(float(tmp[5]) / G.dy)
                        zs = round_value(float(tmp[3]) / G.dz)
                        zf = round_value(float(tmp[6]) / G.dz)
                        depth = float(tmp[7])

                        if xs < 0 or xs > G.nx:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs * G.dx))
                        if xf < 0 or xf > G.nx:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf * G.dx))
                        if ys < 0 or ys > G.ny:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys * G.dy))
                        if yf < 0 or yf > G.ny:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf * G.dy))
                        if zs < 0 or zs > G.nz:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs * G.dz))
                        if zf < 0 or zf > G.nz:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf * G.dz))
                        if xs > xf or ys > yf or zs > zf:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
                        if depth <= 0:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the depth of water')

                        # Check for valid orientations
                        if xs == xf:
                            if ys == yf or zs == zf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')
                            if xs != volume.xs and xs != volume.xf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' can only be used on the external surfaces of a fractal box')
                            # xminus surface
                            if xs == volume.xs:
                                requestedsurface = 'xminus'
                            # xplus surface
                            elif xf == volume.xf:
                                requestedsurface = 'xplus'
                            filldepthcells = round_value(depth / G.dx)
                            filldepth = filldepthcells * G.dx

                        elif ys == yf:
                            if xs == xf or zs == zf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')
                            if ys != volume.ys and ys != volume.yf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' can only be used on the external surfaces of a fractal box')
                            # yminus surface
                            if ys == volume.ys:
                                requestedsurface = 'yminus'
                            # yplus surface
                            elif yf == volume.yf:
                                requestedsurface = 'yplus'
                            filldepthcells = round_value(depth / G.dy)
                            filldepth = filldepthcells * G.dy

                        elif zs == zf:
                            if xs == xf or ys == yf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')
                            if zs != volume.zs and zs != volume.zf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' can only be used on the external surfaces of a fractal box')
                            # zminus surface
                            if zs == volume.zs:
                                requestedsurface = 'zminus'
                            # zplus surface
                            elif zf == volume.zf:
                                requestedsurface = 'zplus'
                            filldepthcells = round_value(depth / G.dz)
                            filldepth = filldepthcells * G.dz

                        else:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')

                        surface = next((x for x in volume.fractalsurfaces if x.surfaceID == requestedsurface), None)
                        if not surface:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' specified surface {} does not have a rough surface applied'.format(requestedsurface))

                        surface.filldepth = filldepthcells

                        # Check that requested fill depth falls within range of surface roughness
                        if surface.filldepth < surface.fractalrange[0] or surface.filldepth > surface.fractalrange[1]:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a value for the depth of water that lies with the range of the requested surface roughness')

                        # Check to see if water has been already defined as a material
                        if not any(x.ID == 'water' for x in G.materials):
                            m = Material(len(G.materials), 'water')
                            m.averagable = False
                            m.type = 'builtin, debye'
                            m.er = Material.watereri
                            m.poles = 1
                            m.deltaer.append(Material.waterdeltaer)
                            m.tau.append(Material.watertau)
                            G.materials.append(m)
                            if Material.maxpoles == 0:
                                Material.maxpoles = 1

                        # Check if time step for model is suitable for using water
                        water = next((x for x in G.materials if x.ID == 'water'))
                        testwater = next((x for x in water.tau if x < G.dt), None)
                        if testwater:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires the time step for the model to be less than the relaxation time required to model water.')

                        if G.messages:
                            tqdm.write('Water on surface from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with depth {:g}m, added to {}.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, filldepth, surface.operatingonID))

                if tmp[0] == '#add_grass:':
                    if len(tmp) < 12:
                        raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires at least eleven parameters')
                    elif len(tmp) == 12:
                        seed = None
                    elif len(tmp) == 13:
                        seed = int(tmp[12])
                    else:
                        raise CmdInputError("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')

                    # Only process grass for this fractal volume
                    if tmp[11] == volume.ID:
                        xs = round_value(float(tmp[1]) / G.dx)
                        xf = round_value(float(tmp[4]) / G.dx)
                        ys = round_value(float(tmp[2]) / G.dy)
                        yf = round_value(float(tmp[5]) / G.dy)
                        zs = round_value(float(tmp[3]) / G.dz)
                        zf = round_value(float(tmp[6]) / G.dz)
                        numblades = int(tmp[10])

                        if xs < 0 or xs > G.nx:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs * G.dx))
                        if xf < 0 or xf > G.nx:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf * G.dx))
                        if ys < 0 or ys > G.ny:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys * G.dy))
                        if yf < 0 or yf > G.ny:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf * G.dy))
                        if zs < 0 or zs > G.nz:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs * G.dz))
                        if zf < 0 or zf > G.nz:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf * G.dz))
                        if xs > xf or ys > yf or zs > zf:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
                        if float(tmp[7]) < 0:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the fractal dimension')
                        if float(tmp[8]) < 0 or float(tmp[9]) < 0:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires a positive value for the minimum and maximum heights for grass blades')

                        # Check for valid orientations
                        if xs == xf:
                            if ys == yf or zs == zf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')
                            if xs != volume.xs and xs != volume.xf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' must specify external surfaces on a fractal box')
                            fractalrange = (round_value(float(tmp[8]) / G.dx), round_value(float(tmp[9]) / G.dx))
                            # xminus surface
                            if xs == volume.xs:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' grass can only be specified on surfaces in the positive axis direction')
                            # xplus surface
                            elif xf == volume.xf:
                                if fractalrange[1] > G.nx:
                                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' cannot apply grass to fractal box as it would exceed the domain size in the x direction')
                                requestedsurface = 'xplus'

                        elif ys == yf:
                            if xs == xf or zs == zf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')
                            if ys != volume.ys and ys != volume.yf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' must specify external surfaces on a fractal box')
                            fractalrange = (round_value(float(tmp[8]) / G.dy), round_value(float(tmp[9]) / G.dy))
                            # yminus surface
                            if ys == volume.ys:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' grass can only be specified on surfaces in the positive axis direction')
                            # yplus surface
                            elif yf == volume.yf:
                                if fractalrange[1] > G.ny:
                                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' cannot apply grass to fractal box as it would exceed the domain size in the y direction')
                                requestedsurface = 'yplus'

                        elif zs == zf:
                            if xs == xf or ys == yf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')
                            if zs != volume.zs and zs != volume.zf:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' must specify external surfaces on a fractal box')
                            fractalrange = (round_value(float(tmp[8]) / G.dz), round_value(float(tmp[9]) / G.dz))
                            # zminus surface
                            if zs == volume.zs:
                                raise CmdInputError("'" + ' '.join(tmp) + "'" + ' grass can only be specified on surfaces in the positive axis direction')
                            # zplus surface
                            elif zf == volume.zf:
                                if fractalrange[1] > G.nz:
                                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' cannot apply grass to fractal box as it would exceed the domain size in the z direction')
                                requestedsurface = 'zplus'

                        else:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' dimensions are not specified correctly')

                        surface = FractalSurface(xs, xf, ys, yf, zs, zf, float(tmp[7]), seed)
                        surface.ID = 'grass'
                        surface.surfaceID = requestedsurface

                        # Set the fractal range to scale the fractal distribution between zero and one
                        surface.fractalrange = (0, 1)
                        surface.operatingonID = volume.ID
                        surface.generate_fractal_surface(G)
                        if numblades > surface.fractalsurface.shape[0] * surface.fractalsurface.shape[1]:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' the specified surface is not large enough for the number of grass blades/roots specified')

                        # Scale the distribution so that the summation is equal to one, i.e. a probability distribution
                        surface.fractalsurface = surface.fractalsurface / np.sum(surface.fractalsurface)

                        # Set location of grass blades using probability distribution
                        # Create 1D vector of probability values from the 2D surface
                        probability1D = np.cumsum(np.ravel(surface.fractalsurface))

                        # Create random numbers between zero and one for the number of blades of grass
                        R = np.random.RandomState(surface.seed)
                        A = R.random_sample(numblades)

                        # Locate the random numbers in the bins created by the 1D vector of probability values, and convert the 1D index back into a x, y index for the original surface.
                        bladesindex = np.unravel_index(np.digitize(A, probability1D), (surface.fractalsurface.shape[0], surface.fractalsurface.shape[1]))

                        # Set the fractal range to minimum and maximum heights of the grass blades
                        surface.fractalrange = fractalrange

                        # Set the fractal surface using the pre-calculated spatial distribution and a random height
                        surface.fractalsurface = np.zeros((surface.fractalsurface.shape[0], surface.fractalsurface.shape[1]))
                        for i in range(len(bladesindex[0])):
                            surface.fractalsurface[bladesindex[0][i], bladesindex[1][i]] = R.randint(surface.fractalrange[0], surface.fractalrange[1], size=1)

                        # Create grass geometry parameters
                        g = Grass(numblades, surface.seed)
                        g.set_geometry_parameters()
                        surface.grass.append(g)

                        # Check to see if grass has been already defined as a material
                        if not any(x.ID == 'grass' for x in G.materials):
                            m = Material(len(G.materials), 'grass')
                            m.averagable = False
                            m.type = 'builtin, debye'
                            m.er = Material.grasseri
                            m.poles = 1
                            m.deltaer.append(Material.grassdeltaer)
                            m.tau.append(Material.grasstau)
                            G.materials.append(m)
                            if Material.maxpoles == 0:
                                Material.maxpoles = 1

                        # Check if time step for model is suitable for using grass
                        grass = next((x for x in G.materials if x.ID == 'grass'))
                        testgrass = next((x for x in grass.tau if x < G.dt), None)
                        if testgrass:
                            raise CmdInputError("'" + ' '.join(tmp) + "'" + ' requires the time step for the model to be less than the relaxation time required to model grass.')

                        volume.fractalsurfaces.append(surface)

                        if G.messages:
                            tqdm.write('{} blades of grass on surface from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with fractal dimension {:g}, fractal seeding {}, and range {:g}m to {:g}m, added to {}.'.format(numblades, xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, surface.dimension, surface.seed, float(tmp[8]), float(tmp[9]), surface.operatingonID))

            # Process any modifications to the original fractal box then generate it
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
                    volume.fractalvolume = np.ones((volume.nx, volume.ny, volume.nz), dtype=floattype)
                    materialnumID = next(x.numID for x in G.materials if x.ID == volume.operatingonID)
                    volume.fractalvolume *= materialnumID
                else:
                    volume.generate_fractal_volume(G)
                    volume.fractalvolume += mixingmodel.startmaterialnum

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
                                            if i < surface.fractalsurface[j - surface.ys, k - surface.zs] and volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] != 1:
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
                                            if i > volume.originalxf - (surface.fractalsurface[j - surface.ys, k - surface.zs] - volume.originalxf) and volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] == 1:
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
                                            if j < surface.fractalsurface[i - surface.xs, k - surface.zs] and volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] != 1:
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
                                            if j > volume.originalyf - (surface.fractalsurface[i - surface.xs, k - surface.zs] - volume.originalyf) and volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] == 1:
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
                                            if k < surface.fractalsurface[i - surface.xs, j - surface.ys] and volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] != 1:
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
                                            if k > volume.originalzf - (surface.fractalsurface[i - surface.xs, j - surface.ys] - volume.originalzf) and volume.mask[i - volume.xs, j - volume.ys, k - volume.zs] == 1:
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
                waternumID = next((x.numID for x in G.materials if x.ID == 'water'), 0)
                grassnumID = next((x.numID for x in G.materials if x.ID == 'grass'), 0)
                data = volume.fractalvolume.astype('int16', order='C')
                mask = volume.mask.copy(order='C')
                build_voxels_from_array_mask(volume.xs, volume.ys, volume.zs, waternumID, grassnumID, volume.averaging, mask, data, G.solid, G.rigidE, G.rigidH, G.ID)

            else:
                if volume.nbins == 1:
                    raise CmdInputError("'" + ' '.join(tmp) + "'" + ' is being used with a single material and no modifications, therefore please use a #box command instead.')
                else:
                    volume.generate_fractal_volume(G)
                    volume.fractalvolume += mixingmodel.startmaterialnum

                data = volume.fractalvolume.astype('int16', order='C')
                build_voxels_from_array(volume.xs, volume.ys, volume.zs, 0, volume.averaging, data, G.solid, G.rigidE, G.rigidH, G.ID)
