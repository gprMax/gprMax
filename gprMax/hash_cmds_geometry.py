# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
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

import logging
import numpy as np

from .cmds_geometry.add_grass import AddGrass
from .cmds_geometry.add_surface_roughness import AddSurfaceRoughness
from .cmds_geometry.add_surface_water import AddSurfaceWater
from .cmds_geometry.box import Box
from .cmds_geometry.cylinder import Cylinder
from .cmds_geometry.cylindrical_sector import CylindricalSector
from .cmds_geometry.edge import Edge
from .cmds_geometry.fractal_box import FractalBox
from .cmds_geometry.plate import Plate
from .cmds_geometry.sphere import Sphere
from .cmds_geometry.triangle import Triangle
from .utilities.utilities import round_value

logger = logging.getLogger(__name__)


def check_averaging(averaging):
    """Check and set material averaging value.

        Args:
            averaging (string): Input value from hash command - should be 'y'
                                    or 'n'

        Returns:
            averaging (bool): geometry object material averaging
        """

    if averaging == 'y':
        averaging = True
    elif averaging == 'n':
        averaging = False
    else:
        logger.exception(self.__str__() + f' requires averaging to be either y or n')

    return averaging


def process_geometrycmds(geometry):
    """
    This function checks the validity of command parameters, creates instances
    of classes of parameters, and calls functions to directly set arrays
    solid, rigid and ID.

    Args:
        geometry (list): Geometry commands in the model,

    Returns:
        scene_objects (list): Holds objects in scene.
    """

    scene_objects = []

    for object in geometry:
        tmp = object.split()

        if tmp[0] == '#geometry_objects_read:':
            from .cmds_geometry.geometry_objects_read import GeometryObjectsRead
            
            if len(tmp) != 6:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires exactly five parameters')
                raise ValueError

            p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))

            gor = GeometryObjectsRead(p1=p1, geofile=tmp[4], matfile=tmp[5])
            scene_objects.append(gor)

        elif tmp[0] == '#edge:':
            if len(tmp) != 8:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires exactly seven parameters')
                raise ValueError

            edge = Edge(p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                        p2=(float(tmp[4]), float(tmp[5]), float(tmp[6])),
                        material_id=tmp[7])

            scene_objects.append(edge)

        elif tmp[0] == '#plate:':
            if len(tmp) < 8:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least seven parameters')
                raise ValueError

            # Isotropic case
            if len(tmp) == 8:
                plate = Plate(p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                              p2=(float(tmp[4]), float(tmp[5]), float(tmp[6])),
                              material_id=tmp[7])

            # Anisotropic case
            elif len(tmp) == 9:
                plate = Plate(p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                              p2=(float(tmp[4]), float(tmp[5]), float(tmp[6])),
                              material_ids=tmp[7:])

            else:
                logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                raise ValueError

            scene_objects.append(plate)

        elif tmp[0] == '#triangle:':
            if len(tmp) < 12:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least eleven parameters')
                raise ValueError

            p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
            p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))
            p3 = (float(tmp[7]), float(tmp[8]), float(tmp[9]))
            thickness = float(tmp[10])

            # Isotropic case with no user specified averaging
            if len(tmp) == 12:
                triangle = Triangle(p1=p1, p2=p2, p3=p3, thickness=thickness, material_id=tmp[11])

            # Isotropic case with user specified averaging
            elif len(tmp) == 13:
                averaging = check_averaging(tmp[12].lower())
                triangle = Triangle(p1=p1, p2=p2, p3=p3, thickness=thickness, material_id=tmp[11], averaging=averaging)

            # Uniaxial anisotropic case
            elif len(tmp) == 14:
                triangle = Triangle(p1=p1, p2=p2, p3=p3, thickness=thickness, material_ids=tmp[11:])

            else:
                logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                raise ValueError

            scene_objects.append(triangle)

        elif tmp[0] == '#box:':
            if len(tmp) < 8:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least seven parameters')
                raise ValueError

            p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
            p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))

            # Isotropic case with no user specified averaging
            if len(tmp) == 8:
                box = Box(p1=p1, p2=p2, material_id=tmp[7])

            # Isotropic case with user specified averaging
            elif len(tmp) == 9:
                averaging = check_averaging(tmp[8].lower())
                box = Box(p1=p1, p2=p2, material_id=tmp[7], averaging=averaging)

            # Uniaxial anisotropic case
            elif len(tmp) == 10:
                box = Box(p1=p1, p2=p2, material_ids=tmp[7:])

            else:
                logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                raise ValueError

            scene_objects.append(box)

        elif tmp[0] == '#cylinder:':
            if len(tmp) < 9:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least eight parameters')
                raise ValueError

            p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
            p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))
            r = float(tmp[7])

            # Isotropic case with no user specified averaging
            if len(tmp) == 9:
                cylinder = Cylinder(p1=p1, p2=p2, r=r, material_id=tmp[8])

            # Isotropic case with user specified averaging
            elif len(tmp) == 10:
                averaging = check_averaging(tmp[9].lower())
                cylinder = Cylinder(p1=p1, p2=p2, r=r, material_id=tmp[8], averaging=averaging)

            # Uniaxial anisotropic case
            elif len(tmp) == 11:
                cylinder = Cylinder(p1=p1, p2=p2, r=r, material_ids=tmp[8:])
                
            else:
                logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                raise ValueError

            scene_objects.append(cylinder)

        elif tmp[0] == '#cylindrical_sector:':
            if len(tmp) < 10:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least nine parameters')
                raise ValueError

            normal = tmp[1].lower()
            ctr1 = float(tmp[2])
            ctr2 = float(tmp[3])
            extent1 = float(tmp[4])
            extent2 = float(tmp[5])
            r = float(tmp[6])
            start = float(tmp[7])
            end = float(tmp[8])

            # Isotropic case with no user specified averaging
            if len(tmp) == 10:
                cylindrical_sector = CylindricalSector(normal=normal, ctr1=ctr1, ctr2=ctr2, extent1=extent1,
                                  extent2=extent2, r=r, start=start, end=end, msterial_id=tmp[9])

            # Isotropic case with user specified averaging
            elif len(tmp) == 11:
                averaging = check_averaging(tmp[10].lower())
                cylindrical_sector = CylindricalSector(normal=normal, ctr1=ctr1, ctr2=ctr2, extent1=extent1, extent2=extent2,
                                  r=r, start=start, end=end, averaging=averaging, material_id=tmp[9])

            # Uniaxial anisotropic case
            elif len(tmp) == 12:
                cylindrical_sector = CylindricalSector(normal=normal, ctr1=ctr1, ctr2=ctr2, extent1=extent1,
                                  extent2=extent2, r=r, start=start, end=end, material_ids=tmp[9:])

            else:
                logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                raise ValueError

            scene_objects.append(cylindrical_sector)

        elif tmp[0] == '#sphere:':
            if len(tmp) < 6:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least five parameters')
                raise ValueError

            p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
            r = float(tmp[4])

            # Isotropic case with no user specified averaging
            if len(tmp) == 6:
                sphere = Sphere(p1=p1, r=r, material_id=tmp[5])

            # Isotropic case with user specified averaging
            elif len(tmp) == 7:
                averaging = check_averaging(tmp[6].lower())
                sphere = Sphere(p1=p1, r=r, material_id=tmp[5], averaging=averaging)

            # Uniaxial anisotropic case
            elif len(tmp) == 8:
                sphere = Sphere(p1=p1, r=r, material_id=tmp[5:])

            else:
                logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                raise ValueError

            scene_objects.append(sphere)

        elif tmp[0] == '#fractal_box:':
            # Default is no dielectric smoothing for a fractal box

            if len(tmp) < 14:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least thirteen parameters')
                raise ValueError

            p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
            p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))
            frac_dim = float(tmp[7])
            weighting = np.array([float(tmp[8]), float(tmp[9]), float(tmp[10])])
            n_materials = round_value(tmp[11])
            mixing_model_id = tmp[12]
            ID = tmp[13]

            if len(tmp) == 14:
                fb = FractalBox(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, mixing_model_id=mixing_model_id, id=ID, n_materials=n_materials)
            elif len(tmp) == 15:
                fb = FractalBox(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, mixing_model_id=mixing_model_id, id=ID, n_materials=n_materials, seed=tmp[14])
            elif len(tmp) == 16:
                fb = FractalBox(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, mixing_model_id=mixing_model_id, id=ID, n_materials=n_materials, seed=tmp[14], averaging=tmp[15].lower())
            else:
                logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                raise ValueError

            scene_objects.append(fb)

            # Search and process any modifiers for the fractal box
            for object in geometry:
                tmp = object.split()

                if tmp[0] == '#add_surface_roughness:':
                    if len(tmp) < 13:
                        logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least twelve parameters')
                        raise ValueError

                    p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
                    p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))
                    frac_dim = float(tmp[7])
                    weighting = np.array([float(tmp[8]), float(tmp[9])])
                    limits = [float(tmp[10]), float(tmp[11])]
                    fractal_box_id = tmp[12]

                    if len(tmp) == 13:
                        asr = AddSurfaceRoughness(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, limits=limits, fractal_box_id=fractal_box_id)
                    elif len(tmp) == 14:
                        asr = AddSurfaceRoughness(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, limits=limits, fractal_box_id=fractal_box_id, seed=int(tmp[13]))
                    else:
                        logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                        raise ValueError

                    scene_objects.append(asr)

                if tmp[0] == '#add_surface_water:':
                    if len(tmp) != 9:
                        logger.exception("'" + ' '.join(tmp) + "'" + ' requires exactly eight parameters')
                        raise ValueError

                    p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
                    p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))
                    depth = float(tmp[7])
                    fractal_box_id = tmp[8]

                    asf = AddSurfaceWater(p1=p1, p2=p2, depth=depth, fractal_box_id=fractal_box_id)
                    scene_objects.append(asf)

                if tmp[0] == '#add_grass:':
                    if len(tmp) < 12:
                        logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least eleven parameters')
                        raise ValueError

                    p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
                    p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))
                    frac_dim = float(tmp[7])
                    limits = [float(tmp[8]), float(tmp[9])]
                    n_blades = int(tmp[10])
                    fractal_box_id = tmp[11]

                    if len(tmp) == 12:
                        grass = AddGrass(p1=p1, p2=p2, frac_dim=frac_dim, limits=limits, n_blades=n_blades, fractal_box_id=fractal_box_id)
                    elif len(tmp) == 13:
                        grass = AddGrass(p1=p1, p2=p2, frac_dim=frac_dim, limits=limits, n_blades=n_blades, fractal_box_id=fractal_box_id, seed=int(tmp[12]))
                    else:
                        logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                        raise ValueError

                    scene_objects.append(grass)

    return scene_objects
