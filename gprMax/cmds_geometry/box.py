# Copyright (C) 2015-2021: The University of Edinburgh
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

from ..cython.geometry_primitives import build_box
from ..materials import Material
from .cmds_geometry import UserObjectGeometry, rotate_2point_object

logger = logging.getLogger(__name__)


class Box(UserObjectGeometry):
    """Allows you to introduce an orthogonal parallelepiped with specific properties into the model.

    :param p1: The lower left (x,y,z) coordinates of a the box.
    :type p1: list, non-optional
    :param p2: The lower left (x,y,z) coordinates of the box.
    :type p2: list, non-optional
    :param material_id: Material identifier that must correspond to material that has already been defined.
    :type material_id: str, non-optional
    :param material_ids:  Material identifiers in the x, y, z directions.
    :type material_ids: list, non-optional
    :param averaging:  y or n, used to switch on and off dielectric smoothing.
    :type averaging: str, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash = '#box'

    def rotate(self, axis, angle, origin=None):
        """Set parameters for rotation."""
        self.axis = axis
        self.angle = angle
        self.origin = origin
        self.dorotate = True

    def __dorotate(self):
        """Perform rotation."""
        pts = np.array([self.kwargs['p1'], self.kwargs['p2']])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.kwargs['p1'] = tuple(rot_pts[0, :])
        self.kwargs['p2'] = tuple(rot_pts[1, :])

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
        except KeyError:
            logger.exception(self.__str__() + ' Please specify two points.')
            raise

        if self.dorotate:
            self.__dorotate()

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

        # check averaging
        try:
            # go with user specified averaging
            averagebox = self.kwargs['averaging']
        except KeyError:
            # if they havent specfied - go with the grid default
            averagebox = grid.averagevolumeobjects

        p3, p4 = uip.check_box_points(p1, p2, self.__str__())
        # find nearest point on grid without translation
        p5 = uip.round_to_grid_static_point(p1)
        p6 = uip.round_to_grid_static_point(p2)
        xs, ys, zs = p3
        xf, yf, zf = p4

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            logger.exception(self.__str__() + f' material(s) {notfound} do not exist')
            raise ValueError

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

        build_box(xs, xf, ys, yf, zs, zf, numID, numIDx, numIDy, numIDz, averaging, grid.solid, grid.rigidE, grid.rigidH, grid.ID)

        dielectricsmoothing = 'on' if averaging else 'off'

        logger.info(self.grid_name(grid) + f"Box from {p5[0]:g}m, {p5[1]:g}m, {p5[2]:g}m, to {p6[0]:g}m, {p6[1]:g}m, {p6[2]:g}m of material(s) {', '.join(materialsrequested)} created, dielectric smoothing is {dielectricsmoothing}.")
