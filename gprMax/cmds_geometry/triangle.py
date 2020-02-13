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

import gprMax.config as config
from .cmds_geometry import UserObjectGeometry
from ..cython.geometry_primitives import build_triangle
from ..exceptions import CmdInputError
from ..materials import Material

log = logging.getLogger(__name__)


class Triangle(UserObjectGeometry):
    """Allows you to introduce a triangular patch or a triangular prism with specific properties into the model.

    :param p1: the coordinates (x,y,z) of the first apex of the triangle.
    :type p1: list, non-optional
    :param p2: the coordinates (x,y,z) of the second apex of the triangle
    :type p2: list, non-optional
    :param p3: the coordinates (x,y,z) of the third apex of the triangle.
    :type p3: list, non-optional
    :param thickness: The thickness of the triangular prism. If the thickness is zero then a triangular patch is created.
    :type thickness: float, non-optional
    :param material_id: Material identifier that must correspond to material that has already been defined.
    :type material_id: str, non-optional
    :param material_ids:  Material identifiers in the x, y, z directions.
    :type material_ids: list, non-optional
    :param averaging:  y or n, used to switch on and off dielectric smoothing.
    :type averaging: str, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 4
        self.hash = '#triangle'

    def create(self, grid, uip):
        try:
            up1 = self.kwargs['p1']
            up2 = self.kwargs['p2']
            up3 = self.kwargs['p3']
            thickness = self.kwargs['thickness']
        except KeyError:
            raise CmdInputError(self.params_str() + ' Specify 3 points and a thickness')

        # check averaging
        try:
            # go with user specified averaging
            averagetriangularprism = self.kwargs['averaging']
        except KeyError:
            # if they havent specfied - go with the grid default
            averagetriangularprism = grid.averagevolumeobjects

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

        # Check whether points are valid against grid
        uip.check_tri_points(up1, up2, up3, object)
        # Convert points to metres
        x1, y1, z1 = uip.round_to_grid(up1)
        x2, y2, z2 = uip.round_to_grid(up2)
        x3, y3, z3 = uip.round_to_grid(up3)

        if thickness < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for thickness')

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
            raise CmdInputError(self.__str__() + ' the triangle is not specified correctly')

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

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

        build_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, normal, thickness, grid.dx, grid.dy, grid.dz, numID, numIDx, numIDy, numIDz, averaging, grid.solid, grid.rigidE, grid.rigidH, grid.ID)

        if thickness > 0:
            dielectricsmoothing = 'on' if averaging else 'off'
            log.info(f"Triangle with coordinates {x1:g}m {y1:g}m {z1:g}m, {x2:g}m {y2:g}m {z2:g}m, {x3:g}m {y3:g}m {z3:g}m and thickness {thickness:g}m of material(s) {', '.join(materialsrequested)} created, dielectric smoothing is {dielectricsmoothing}.")
        else:
            log.info(f"Triangle with coordinates {x1:g}m {y1:g}m {z1:g}m, {x2:g}m {y2:g}m {z2:g}m, {x3:g}m {y3:g}m {z3:g}m of material(s) {', '.join(materialsrequested)} created.")
