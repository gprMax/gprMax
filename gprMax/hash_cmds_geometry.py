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

import logging
import numpy as np
import sys
from .random_gen import (rand_param_create, check_upper_greater, make_data_label)

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


def process_geometrycmds(geometry, domain_bounds, hash_count_geometrycmds):
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
    rand_params = []
    data_labels = []

    for object in geometry:
        tmp = object.split()

        if tmp[0] == '#geometry_objects_read:':
            from .cmds_geometry.geometry_objects_read import GeometryObjectsRead
            
            if len(tmp) == 6:
                p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
                gor = GeometryObjectsRead(p1=p1, geofile=tmp[4], matfile=tmp[5])

            elif len(tmp) == 10:  
                distr = tmp[1].lower()
                p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[4]), float(tmp[5]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[6]), float(tmp[7]), (0, domain_bounds[2]), tmp[0]))
                rand_params.extend(p1)
                hash_count_geometrycmds[tmp[0][:-1]] += 1
                data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)'])
                
                gor = GeometryObjectsRead(p1=p1, geofile=tmp[8], matfile=tmp[9])

            else:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires exactly five parameters (if single values entered) or nine parameters (if range of values entered)')
                raise ValueError

            scene_objects.append(gor)

        elif tmp[0] == '#edge:':
            if len(tmp) == 8:
                edge = Edge(p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                            p2=(float(tmp[4]), float(tmp[5]), float(tmp[6])),
                            material_id=tmp[7])      

            elif len(tmp) == 15:
                distr = tmp[1].lower()
                p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[4]), float(tmp[5]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[6]), float(tmp[7]), (0, domain_bounds[2]), tmp[0]))
                p2 = (rand_param_create(distr, float(tmp[8]), float(tmp[9]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[10]), float(tmp[11]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[12]), float(tmp[13]), (0, domain_bounds[2]), tmp[0]))
                p2 = check_upper_greater(p1, p2, tmp[0])
                rand_params.extend(p1); rand_params.extend(p2)
                hash_count_geometrycmds[tmp[0][:-1]] += 1
                data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (p2)'])

                edge = Edge(p1=p1, p2=p2, material_id=tmp[14])

            else:
                logger.exception("'" + ' '.join(tmp) + "'" + ' requires exactly seven parameters (if single values entered) or fourteen parameters (if range of values entered)')
                raise ValueError

            scene_objects.append(edge)

        elif tmp[0] == '#plate:':
            if len(tmp) <= 9:
                if len(tmp) < 8:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least seven parameters (if single values entered)')
                    raise ValueError
                
                p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
                p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))

                # Isotropic case
                if len(tmp) == 8:
                    plate = Plate(p1=p1, p2=p2, material_id=tmp[7])

                # Anisotropic case
                elif len(tmp) == 9:
                    plate = Plate(p1=p1, p2=p2, material_ids=tmp[7:])

            elif len(tmp) > 9:
                if len(tmp) < 15:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least fourteen parameters (if range of values entered)')
                    raise ValueError

                distr = tmp[1].lower()
                p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[4]), float(tmp[5]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[6]), float(tmp[7]), (0, domain_bounds[2]), tmp[0]))
                p2 = (rand_param_create(distr, float(tmp[8]), float(tmp[9]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[10]), float(tmp[11]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[12]), float(tmp[13]), (0, domain_bounds[2]), tmp[0]))
                p2 = check_upper_greater(p1, p2, tmp[0])
                rand_params.extend(p1); rand_params.extend(p2)
                hash_count_geometrycmds[tmp[0][:-1]] += 1
                data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (p2)'])

                # Isotropic case
                if len(tmp) == 15:
                    plate = Plate(p1=p1, p2=p2, material_id=tmp[14])

                # Anisotropic case
                elif len(tmp) == 16:
                    plate = Plate(p1=p1, p2=p2, material_ids=tmp[14:])

                else:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                    raise ValueError

            scene_objects.append(plate)

        elif tmp[0] == '#triangle:':
            if len(tmp) <= 14:
                if len(tmp) < 12:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least eleven parameters (if single values entered)')
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
                    triangle = Triangle(p1=p1, p2=p2, p3=p3, thickness=thickness, material_id=tmp[11], averaging=tmp[12].lower())

                # Uniaxial anisotropic case
                elif len(tmp) == 14:
                    triangle = Triangle(p1=p1, p2=p2, p3=p3, thickness=thickness, material_ids=tmp[11:])

            elif len(tmp) > 14:
                if len(tmp) < 23:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least twenty two parameters (if range of values entered)')
                    raise ValueError

                distr = tmp[1].lower()
                p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[4]), float(tmp[5]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[6]), float(tmp[7]), (0, domain_bounds[2]), tmp[0]))
                p2 = (rand_param_create(distr, float(tmp[8]), float(tmp[9]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[10]), float(tmp[11]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[12]), float(tmp[13]), (0, domain_bounds[2]), tmp[0]))
                p3 = (rand_param_create(distr, float(tmp[14]), float(tmp[15]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[16]), float(tmp[17]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[18]), float(tmp[19]), (0, domain_bounds[2]), tmp[0]))
                thickness = rand_param_create(distr, float(tmp[20]), float(tmp[21]), (0, np.inf), tmp[0])
                
                rand_params.extend(p1); rand_params.extend(p2); rand_params.extend(p3); rand_params.extend([thickness])
                hash_count_geometrycmds[tmp[0][:-1]] += 1
                data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (p2)', ' (p3)', ' (thickness)'])

                # Isotropic case with no user specified averaging
                if len(tmp) == 23:
                    triangle = Triangle(p1=p1, p2=p2, p3=p3, thickness=thickness, material_id=tmp[22])

                # Isotropic case with user specified averaging
                elif len(tmp) == 24:
                    triangle = Triangle(p1=p1, p2=p2, p3=p3, thickness=thickness, material_id=tmp[22], averaging=tmp[23].lower())

                # Uniaxial anisotropic case
                elif len(tmp) == 25:
                    triangle = Triangle(p1=p1, p2=p2, p3=p3, thickness=thickness, material_ids=tmp[22:])

                else:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                    raise ValueError

            scene_objects.append(triangle)

        elif tmp[0] == '#box:':
            if len(tmp) <= 10:
                if len(tmp) < 8:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least seven parameters (if single values entered)')
                    raise ValueError

                p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
                p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))

                # Isotropic case with no user specified averaging
                if len(tmp) == 8:
                    box = Box(p1=p1, p2=p2, material_id=tmp[7])

                # Isotropic case with user specified averaging
                elif len(tmp) == 9:
                    box = Box(p1=p1, p2=p2, material_id=tmp[7], averaging=tmp[8])

                # Uniaxial anisotropic case
                elif len(tmp) == 10:
                    box = Box(p1=p1, p2=p2, material_ids=tmp[7:])

            elif len(tmp) > 10:
                if len(tmp) < 15:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least fourteen parameters (if range of values entered)')
                    raise ValueError

                distr = tmp[1].lower()
                p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[4]), float(tmp[5]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[6]), float(tmp[7]), (0, domain_bounds[2]), tmp[0]))
                p2 = (rand_param_create(distr, float(tmp[8]), float(tmp[9]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[10]), float(tmp[11]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[12]), float(tmp[13]), (0, domain_bounds[2]), tmp[0]))
                p2 = check_upper_greater(p1, p2, tmp[0])

                rand_params.extend(p1); rand_params.extend(p2)
                hash_count_geometrycmds[tmp[0][:-1]] += 1
                data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (p2)'])

                # Isotropic case with no user specified averaging
                if len(tmp) == 15:
                    box = Box(p1=p1, p2=p2, material_id=tmp[14])

                # Isotropic case with user specified averaging
                elif len(tmp) == 16:
                    box = Box(p1=p1, p2=p2, material_id=tmp[14], averaging=tmp[15])

                # Uniaxial anisotropic case
                elif len(tmp) == 17:
                    box = Box(p1=p1, p2=p2, material_ids=tmp[14:])
                
                else:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                    raise ValueError

            scene_objects.append(box)

        elif tmp[0] == '#cylinder:':
            if len(tmp) <= 11:
                if len(tmp) < 9:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least eight parameters (if single values entered)')
                    raise ValueError

                p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
                p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))
                r = float(tmp[7])

                # Isotropic case with no user specified averaging
                if len(tmp) == 9:
                    cylinder = Cylinder(p1=p1, p2=p2, r=r, material_id=tmp[8])

                # Isotropic case with user specified averaging
                elif len(tmp) == 10:
                    cylinder = Cylinder(p1=p1, p2=p2, r=r, material_id=tmp[8], averaging=tmp[9])

                # Uniaxial anisotropic case
                elif len(tmp) == 11:
                    cylinder = Cylinder(p1=p1, p2=p2, r=r, material_ids=tmp[8:])

            elif len(tmp) > 11:
                if len(tmp) < 17:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least sixteen parameters (if range of values entered)')
                    raise ValueError

                distr = tmp[1].lower()
                p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3])), 
                      rand_param_create(distr, float(tmp[4]), float(tmp[5])), 
                      rand_param_create(distr, float(tmp[6]), float(tmp[7])))
                p2 = (rand_param_create(distr, float(tmp[8]), float(tmp[9])), 
                      rand_param_create(distr, float(tmp[10]), float(tmp[11])), 
                      rand_param_create(distr, float(tmp[12]), float(tmp[13])))               
                r = rand_param_create(distr, float(tmp[14]), float(tmp[15]), (sys.float_info.epsilon, np.inf), tmp[0])

                rand_params.extend(p1); rand_params.extend(p2); rand_params.extend([r])
                hash_count_geometrycmds[tmp[0][:-1]] += 1
                data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (p2)', ' (r)'])

                # Isotropic case with no user specified averaging
                if len(tmp) == 17:
                    cylinder = Cylinder(p1=p1, p2=p2, r=r, material_id=tmp[16])

                # Isotropic case with user specified averaging
                elif len(tmp) == 18:
                    cylinder = Cylinder(p1=p1, p2=p2, r=r, material_id=tmp[16], averaging=tmp[17])

                # Uniaxial anisotropic case
                elif len(tmp) == 19:
                    cylinder = Cylinder(p1=p1, p2=p2, r=r, material_ids=tmp[16:])
                
                else:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                    raise ValueError

            scene_objects.append(cylinder)

        elif tmp[0] == '#cylindrical_sector:':
            if len(tmp) <= 12:
                if len(tmp) < 10:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least nine parameters (if single values entered)')
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
                    cylindrical_sector = CylindricalSector(normal=normal, ctr1=ctr1, ctr2=ctr2, extent1=extent1, extent2=extent2,
                                    r=r, start=start, end=end, averaging=tmp[10], material_id=tmp[9])

                # Uniaxial anisotropic case
                elif len(tmp) == 12:
                    cylindrical_sector = CylindricalSector(normal=normal, ctr1=ctr1, ctr2=ctr2, extent1=extent1,
                                    extent2=extent2, r=r, start=start, end=end, material_ids=tmp[9:])
                
            if len(tmp) > 12:
                if len(tmp) < 18:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least seventeen parameters (if range of values entered)')
                    raise ValueError

                distr = tmp[1].lower()
                normal = tmp[2].lower()
                ctr1 = rand_param_create(distr, float(tmp[3]), float(tmp[4]))
                ctr2 = rand_param_create(distr, float(tmp[5]), float(tmp[6]))
                extent1 = rand_param_create(distr, float(tmp[7]), float(tmp[8]))
                extent2 = rand_param_create(distr, float(tmp[9]), float(tmp[10]))
                r = rand_param_create(distr, float(tmp[11]), float(tmp[12]), (sys.float_info.epsilon, np.inf), tmp[0])
                start = rand_param_create(distr, float(tmp[13]), float(tmp[14]), (0, 2*np.pi - sys.float_info.epsilon), tmp[0])
                end = rand_param_create(distr, float(tmp[15]), float(tmp[16]), (sys.float_info.epsilon, 2*np.pi - sys.float_info.epsilon), tmp[0])

                rand_params.extend([ctr1]); rand_params.extend([ctr2]); rand_params.extend([extent1]); rand_params.extend([extent2])
                rand_params.extend([r]); rand_params.extend([start]); rand_params.extend([end])
                hash_count_geometrycmds[tmp[0][:-1]] += 1
                data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (ctr1)', ' (ctr2)', ' (extent1)', ' (extent2)', ' (r)', ' (start)', ' (end)'])

                # Isotropic case with no user specified averaging
                if len(tmp) == 18:
                    cylindrical_sector = CylindricalSector(normal=normal, ctr1=ctr1, ctr2=ctr2, extent1=extent1,
                                    extent2=extent2, r=r, start=start, end=end, msterial_id=tmp[17])

                # Isotropic case with user specified averaging
                elif len(tmp) == 19:
                    cylindrical_sector = CylindricalSector(normal=normal, ctr1=ctr1, ctr2=ctr2, extent1=extent1, extent2=extent2,
                                    r=r, start=start, end=end, averaging=tmp[18], material_id=tmp[17])

                # Uniaxial anisotropic case
                elif len(tmp) == 20:
                    cylindrical_sector = CylindricalSector(normal=normal, ctr1=ctr1, ctr2=ctr2, extent1=extent1,
                                    extent2=extent2, r=r, start=start, end=end, material_ids=tmp[17:])

                else:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                    raise ValueError

            scene_objects.append(cylindrical_sector)

        elif tmp[0] == '#sphere:':
            if len(tmp) <= 8:
                if len(tmp) < 6:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least five parameters (if single values entered)')
                    raise ValueError

                p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
                r = float(tmp[4])

                # Isotropic case with no user specified averaging
                if len(tmp) == 6:
                    sphere = Sphere(p1=p1, r=r, material_id=tmp[5])

                # Isotropic case with user specified averaging
                elif len(tmp) == 7:
                    sphere = Sphere(p1=p1, r=r, material_id=tmp[5], averaging=tmp[6])

                # Uniaxial anisotropic case
                elif len(tmp) == 8:
                    sphere = Sphere(p1=p1, r=r, material_id=tmp[5:])

            elif len(tmp) > 8:
                if len(tmp) < 11:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least ten parameters (if range of values entered)')
                    raise ValueError
                
                distr = tmp[1].lower()
                p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3])), 
                      rand_param_create(distr, float(tmp[4]), float(tmp[5])), 
                      rand_param_create(distr, float(tmp[6]), float(tmp[7])))
                r = rand_param_create(distr, float(tmp[8]), float(tmp[9]), (sys.float_info.epsilon, np.inf), tmp[0])

                rand_params.extend(p1); rand_params.extend([r])
                hash_count_geometrycmds[tmp[0][:-1]] += 1
                data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (r)'])

                # Isotropic case with no user specified averaging
                if len(tmp) == 11:
                    sphere = Sphere(p1=p1, r=r, material_id=tmp[10])

                # Isotropic case with user specified averaging
                elif len(tmp) == 12:
                    sphere = Sphere(p1=p1, r=r, material_id=tmp[10], averaging=tmp[11])

                # Uniaxial anisotropic case
                elif len(tmp) == 13:
                    sphere = Sphere(p1=p1, r=r, material_id=tmp[10:])

                else:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                    raise ValueError

            scene_objects.append(sphere)

        elif tmp[0] == '#fractal_box:':
            # Default is no dielectric smoothing for a fractal box
            if len(tmp) <= 16:
                if len(tmp) < 14:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least thirteen parameters (if single values entered)')
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
            
            elif len(tmp) > 16:
                if len(tmp) < 25:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least twenty four parameters (if range of values entered)')
                    raise ValueError

                distr = tmp[1].lower()
                p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[4]), float(tmp[5]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[6]), float(tmp[7]), (0, domain_bounds[2]), tmp[0]))
                p2 = (rand_param_create(distr, float(tmp[8]), float(tmp[9]), (0, domain_bounds[0]), tmp[0]), 
                      rand_param_create(distr, float(tmp[10]), float(tmp[11]), (0, domain_bounds[1]), tmp[0]), 
                      rand_param_create(distr, float(tmp[12]), float(tmp[13]), (0, domain_bounds[2]), tmp[0]))
                p2 = check_upper_greater(p1, p2, tmp[0]) 
                
                frac_dim = rand_param_create(distr, float(tmp[14]), float(tmp[15]), (0, 3), tmp[0])
                weighting = np.array([rand_param_create(distr, float(tmp[16]), float(tmp[17]), (0, np.inf), tmp[0]), 
                                      rand_param_create(distr, float(tmp[18]), float(tmp[19]), (0, np.inf), tmp[0]), 
                                      rand_param_create(distr, float(tmp[20]), float(tmp[21]), (0, np.inf), tmp[0])])
                n_materials = round_value(tmp[22])
                mixing_model_id = tmp[23]
                ID = tmp[24]

                rand_params.extend(p1); rand_params.extend(p2); rand_params.extend([frac_dim]); rand_params.extend(weighting)
                hash_count_geometrycmds[tmp[0][:-1]] += 1
                data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (p2)', ' (frac_dim)', ' (weighting)'])

                if len(tmp) == 25:
                    fb = FractalBox(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, mixing_model_id=mixing_model_id, id=ID, n_materials=n_materials)
                elif len(tmp) == 26:
                    fb = FractalBox(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, mixing_model_id=mixing_model_id, id=ID, n_materials=n_materials, seed=tmp[25])
                elif len(tmp) == 27:
                    fb = FractalBox(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, mixing_model_id=mixing_model_id, id=ID, n_materials=n_materials, seed=tmp[25], averaging=tmp[26].lower())
                else:
                    logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given ')
                    raise ValueError

            scene_objects.append(fb)

            # Search and process any modifiers for the fractal box
            for object in geometry:
                tmp = object.split()

                if tmp[0] == '#add_surface_roughness:':
                    if len(tmp) <= 14:
                        if len(tmp) < 13:
                            logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least twelve parameters (if single values entered)')
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
                    
                    elif len(tmp) > 14:
                        if len(tmp) < 25:
                            logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least twenty four parameters (if range of values entered)')
                            raise ValueError
                        
                        distr = tmp[1].lower()
                        p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3]), (0, domain_bounds[0]), tmp[0]), 
                              rand_param_create(distr, float(tmp[4]), float(tmp[5]), (0, domain_bounds[1]), tmp[0]), 
                              rand_param_create(distr, float(tmp[6]), float(tmp[7]), (0, domain_bounds[2]), tmp[0]))
                        p2 = (rand_param_create(distr, float(tmp[8]), float(tmp[9]), (0, domain_bounds[0]), tmp[0]), 
                              rand_param_create(distr, float(tmp[10]), float(tmp[11]), (0, domain_bounds[1]), tmp[0]), 
                              rand_param_create(distr, float(tmp[12]), float(tmp[13]), (0, domain_bounds[2]), tmp[0])) 
                        p2 = check_upper_greater(p1, p2, tmp[0]) 
                        frac_dim = rand_param_create(distr, float(tmp[14]), float(tmp[15]), (0, 3), tmp[0])
                        weighting = np.array([rand_param_create(distr, float(tmp[16]), float(tmp[17]), (0, np.inf), tmp[0]), 
                                              rand_param_create(distr, float(tmp[18]), float(tmp[19]), (0, np.inf), tmp[0])])
                        limits = [rand_param_create(distr, float(tmp[20]), float(tmp[21])), 
                                  rand_param_create(distr, float(tmp[22]), float(tmp[23]))]
                        fractal_box_id = tmp[24]

                        rand_params.extend(p1); rand_params.extend(p2); rand_params.extend([frac_dim]); rand_params.extend(weighting); rand_params.extend(limits)
                        hash_count_geometrycmds[tmp[0][:-1]] += 1
                        data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (p2)', ' (frac_dim)', ' (weighting)', ' (limits)'])   

                        if len(tmp) == 24:
                            asr = AddSurfaceRoughness(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, limits=limits, fractal_box_id=fractal_box_id)
                        elif len(tmp) == 25:
                            asr = AddSurfaceRoughness(p1=p1, p2=p2, frac_dim=frac_dim, weighting=weighting, limits=limits, fractal_box_id=fractal_box_id, seed=int(tmp[24]))
                        else:
                            logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                            raise ValueError

                    scene_objects.append(asr)

                if tmp[0] == '#add_surface_water:':
                    if len(tmp) == 9:
                        p1 = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
                        p2 = (float(tmp[4]), float(tmp[5]), float(tmp[6]))
                        depth = float(tmp[7])
                        fractal_box_id = tmp[8]

                        asf = AddSurfaceWater(p1=p1, p2=p2, depth=depth, fractal_box_id=fractal_box_id)

                    elif len(tmp) == 17:
                        distr = tmp[1].lower()
                        p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3]), (0, domain_bounds[0]), tmp[0]), 
                              rand_param_create(distr, float(tmp[4]), float(tmp[5]), (0, domain_bounds[1]), tmp[0]), 
                              rand_param_create(distr, float(tmp[6]), float(tmp[7]), (0, domain_bounds[2]), tmp[0]))
                        p2 = (rand_param_create(distr, float(tmp[8]), float(tmp[9]), (0, domain_bounds[0]), tmp[0]), 
                              rand_param_create(distr, float(tmp[10]), float(tmp[11]), (0, domain_bounds[1]), tmp[0]), 
                              rand_param_create(distr, float(tmp[12]), float(tmp[13]), (0, domain_bounds[2]), tmp[0])) 
                        p2 = check_upper_greater(p1, p2, tmp[0]) 
                        depth = rand_param_create(distr, float(tmp[14]), float(tmp[15]), (sys.float_info.epsilon, np.inf), tmp[0])
                        fractal_box_id = tmp[16]

                        rand_params.extend(p1); rand_params.extend(p2); rand_params.extend([depth])
                        hash_count_geometrycmds[tmp[0][:-1]] += 1
                        data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (p2)', ' (depth)'])  

                        asf = AddSurfaceWater(p1=p1, p2=p2, depth=depth, fractal_box_id=fractal_box_id)
                    
                    else:
                        logger.exception("'" + ' '.join(tmp) + "'" + ' requires exactly eight parameters (if single values entered) or sixteen parameters (if range of values entered)')
                        raise ValueError

                    scene_objects.append(asf)

                if tmp[0] == '#add_grass:':
                    if len(tmp) <= 13:
                        if len(tmp) < 12:
                            logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least eleven parameters (if single values entered)')
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
                    
                    elif len(tmp) > 13:
                        if len(tmp) < 22:
                            logger.exception("'" + ' '.join(tmp) + "'" + ' requires at least twenty one parameters (if range of values entered)')
                            raise ValueError

                        distr = tmp[1].lower()
                        p1 = (rand_param_create(distr, float(tmp[2]), float(tmp[3]), (0, domain_bounds[0]), tmp[0]), 
                              rand_param_create(distr, float(tmp[4]), float(tmp[5]), (0, domain_bounds[1]), tmp[0]), 
                              rand_param_create(distr, float(tmp[6]), float(tmp[7]), (0, domain_bounds[2]), tmp[0]))
                        p2 = (rand_param_create(distr, float(tmp[8]), float(tmp[9]), (0, domain_bounds[0]), tmp[0]), 
                              rand_param_create(distr, float(tmp[10]), float(tmp[11]), (0, domain_bounds[1]), tmp[0]), 
                              rand_param_create(distr, float(tmp[12]), float(tmp[13]), (0, domain_bounds[2]), tmp[0]))  
                        p2 = check_upper_greater(p1, p2, tmp[0])
                        frac_dim = rand_param_create(distr, float(tmp[14]), float(tmp[15]), (0, 3), tmp[0])
                        limits = [rand_param_create(distr, float(tmp[16]), float(tmp[17]), (0, np.inf), tmp[0]), 
                                  rand_param_create(distr, float(tmp[18]), float(tmp[19]), (0, np.inf), tmp[0])]
                        n_blades = int(tmp[20])
                        fractal_box_id = tmp[21]

                        rand_params.extend(p1); rand_params.extend(p2); rand_params.extend([frac_dim]); rand_params.extend(limits)
                        hash_count_geometrycmds[tmp[0][:-1]] += 1
                        data_labels = make_data_label(data_labels, tmp[0][1:-1] + (' #' + str(hash_count_geometrycmds[tmp[0][:-1]]) if hash_count_geometrycmds[tmp[0][:-1]]>1 else ''), [' (p1)', ' (p2)', ' (frac_dim)', ' (limits)'])  

                        if len(tmp) == 22:
                            grass = AddGrass(p1=p1, p2=p2, frac_dim=frac_dim, limits=limits, n_blades=n_blades, fractal_box_id=fractal_box_id)
                        elif len(tmp) == 23:
                            grass = AddGrass(p1=p1, p2=p2, frac_dim=frac_dim, limits=limits, n_blades=n_blades, fractal_box_id=fractal_box_id, seed=int(tmp[22]))
                        else:
                            logger.exception("'" + ' '.join(tmp) + "'" + ' too many parameters have been given')
                            raise ValueError

                    scene_objects.append(grass)

    return scene_objects, rand_params, data_labels
