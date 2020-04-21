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

import logging

import numpy as np

from ..cython.geometry_primitives import build_cylindrical_sector
from ..materials import Material
from .cmds_geometry import UserObjectGeometry

logger = logging.getLogger(__name__)


class CylindricalSector(UserObjectGeometry):
    """Allows you to introduce a cylindrical sector (shaped like a slice of pie) into the model.

    :param normal: Direction of the axis of the cylinder from which the sector is defined and can be x, y, or z..
    :type normal: str, non-optional
    :param ctr1: First coordinate of the centre of the cylindrical sector.
    :type ctr1: float, non-optional
    :param ctr2:  Second coordinate of the centre of the cylindrical sector.
    :type ctr2: Float, non-optional
    :param extent1: First thickness from the centre of the cylindrical sector.
    :type extent1: float, non-optional
    :param extent2: Second thickness from the centre of the cylindrical sector.
    :type extent2: float, non-optional
    :param r: is the radius of the cylindrical sector.
    :type r: float, non-optional
    :param start: The starting angle (in degrees) for the cylindrical sector.
    :type start: float, non-optional
    :param end: The angle (in degrees) swept by the cylindrical sector
    :type end: float, non-optional
    :param material_id: Material identifier that must correspond to material that has already been defined.
    :type material_id: str, non-optional
    :param material_ids:  Material identifiers in the x, y, z directions.
    :type material_ids: list, non-optional
    :param averaging:  y or n, used to switch on and off dielectric smoothing.
    :type averaging: str, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash = '#cylindrical_sector'

    def create(self, grid, uip):

        try:
            normal = self.kwargs['normal'].lower()
            ctr1 = self.kwargs['ctr1']
            ctr2 = self.kwargs['ctr2']
            extent1 = self.kwargs['extent1']
            extent2 = self.kwargs['extent2']
            start = self.kwargs['start']
            end = self.kwargs['end']
            r = self.kwargs['r']
            thickness = extent2 - extent1
        except KeyError:
            logger.exception(self.__str__())
            raise

        # check averaging
        try:
            # go with user specified averaging
            averagecylindricalsector = self.kwargs['averaging']
        except KeyError:
            # if they havent specfied - go with the grid default
            averagecylindricalsector = grid.averagevolumeobjects

        # check materials have been specified
        # isotropic case
        try:
            materialsrequested = [self.kwargs['material_id']]
        except KeyError:
            # Anisotropic case
            try:
                materialsrequested = self.kwargs['material_ids']
            except KeyError:
                logger.exception(self.__str__() + ' No materials have been specified')
                raise

        sectorstartangle = 2 * np.pi * (start / 360)
        sectorangle = 2 * np.pi * (end / 360)

        if normal != 'x' and normal != 'y' and normal != 'z':
            logger.exception(self.__str__() + ' the normal direction must be either x, y or z.')
            raise ValueError
        if r <= 0:
            logger.exception(self.__str__() + f' the radius {r:g} should be a positive value.')
        if sectorstartangle < 0 or sectorangle <= 0:
            logger.exception(self.__str__() + ' the starting angle and sector angle should be a positive values.')
            raise ValueError
        if sectorstartangle >= 2 * np.pi or sectorangle >= 2 * np.pi:
            logger.exception(self.__str__() + ' the starting angle and sector angle must be less than 360 degrees.')
            raise ValueError

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            logger.exception(self.__str__() + f' material(s) {notfound} do not exist')
            raise ValueError

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
                averagedmaterial = [x for x in grid.materials if x.ID == requiredID]
                if averagedmaterial:
                    numID = averagedmaterial.numID
                else:
                    numID = len(grid.materials)
                    m = Material(numID, requiredID)
                    m.type = 'dielectric-smoothed'
                    # Create dielectric-smoothed constituents for material
                    m.er = np.mean((materials[0].er, materials[1].er, materials[2].er), axis=0)
                    m.se = np.mean((materials[0].se, materials[1].se, materials[2].se), axis=0)
                    m.mr = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)
                    m.sm = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)

                    # Append the new material object to the materials list
                    grid.materials.append(m)
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
            level, ctr1, ctr2 = uip.round_to_grid((extent1, ctr1, ctr2))

        # xz-plane cylindrical sector
        elif normal == 'y':
            ctr1, level, ctr2 = uip.round_to_grid((ctr1, extent1, ctr2))

        # xy-plane cylindrical sector
        elif normal == 'z':
            ctr1, ctr2, level = uip.round_to_grid((ctr1, ctr2, extent1))

        build_cylindrical_sector(ctr1, ctr2, level, sectorstartangle, sectorangle, r, normal, thickness, grid.dx, grid.dy, grid.dz, numID, numIDx, numIDy, numIDz, averaging, grid.solid, grid.rigidE, grid.rigidH, grid.ID)

        if thickness > 0:
            dielectricsmoothing = 'on' if averaging else 'off'
            logger.info(f"Cylindrical sector with centre {ctr1:g}m, {ctr2:g}m, radius {r:g}m, starting angle {(sectorstartangle / (2 * np.pi)) * 360:.1f} degrees, sector angle {(sectorangle / (2 * np.pi)) * 360:.1f} degrees, thickness {thickness:g}m, of material(s) {', '.join(materialsrequested)} created, dielectric smoothing is {dielectricsmoothing}.")
        else:
            logger.info(f"Cylindrical sector with centre {ctr1:g}m, {ctr2:g}m, radius {r:g}m, starting angle {(sectorstartangle / (2 * np.pi)) * 360:.1f} degrees, sector angle {(sectorangle / (2 * np.pi)) * 360:.1f} degrees, of material(s) {', '.join(materialsrequested)} created.")
