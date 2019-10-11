# Copyright (C) 2015-2019: The University of Edinburgh
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

import gprMax.config as config
from .cmds_geometry import UserObjectGeometry
from ..cython.geometry_primitives import build_cylinder
from ..exceptions import CmdInputError
from ..materials import Material

log = logging.getLogger(__name__)


class Cylinder(UserObjectGeometry):
    """Allows you to introduce a circular cylinder into the model.

    :param p1: The coordinates (x,y,z) of the centre of the first face of the cylinder.
    :type p1: list, non-optional
    :param p2: The coordinates (x,y,z) of the centre of the second face of the cylinder.
    :type p2: list, non-optional
    :param r: is the radius of the cylinder.
    :type r: float, non-optional
    :param material_id: Material identifier that must correspond to material that has already been defined.
    :type material_id: str, non-optional
    :param material_ids:  Material identifiers in the x, y, z directions.
    :type material_ids: list, non-optional
    :param averaging:  y or n, used to switch on and off dielectric smoothing.
    :type averaging: str, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 6
        self.hash = '#cylinder'

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            r = self.kwargs['r']

        except KeyError:
            raise CmdInputError(self.__str__() + ' Please specify 2 points and a radius')

        # check averaging
        try:
            # go with user specified averaging
            averagecylinder = self.kwargs['averaging']
        except KeyError:
            # if they havent specfied - go with the grid default
            averagecylinder = grid.averagevolumeobjects

        # check materials have been specified
        # isotropic case
        try:
            materialsrequested = [self.kwargs['material_id']]
        except KeyError:
            # Anisotropic case
            try:
                materialsrequested = self.kwargs['material_ids']
            except KeyError:
                raise CmdInputError(self.__str__() + ' No materials have been specified')

        x1, y1, z1 = uip.round_to_grid(p1)
        x2, y2, z2 = uip.round_to_grid(p2)

        if r <= 0:
            raise CmdInputError(self.__str__() + ' the radius {:g} should be a positive value.'.format(r))

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

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

        build_cylinder(x1, y1, z1, x2, y2, z2, r, grid.dx, grid.dy, grid.dz, numID, numIDx, numIDy, numIDz, averaging, grid.solid, grid.rigidE, grid.rigidH, grid.ID)

        dielectricsmoothing = 'on' if averaging else 'off'
        log.info(f"Cylinder with face centres {x1:g}m, {y1:g}m, {z1:g}m and {x2:g}m, {y2:g}m, {z2:g}m, with radius {r:g}m, of material(s) {', '.join(materialsrequested)} created, dielectric smoothing is {dielectricsmoothing}.")
