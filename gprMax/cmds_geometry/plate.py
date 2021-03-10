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

from ..cython.geometry_primitives import (build_face_xy, build_face_xz,
                                          build_face_yz)
from .cmds_geometry import UserObjectGeometry, rotate_2point_object

logger = logging.getLogger(__name__)


class Plate(UserObjectGeometry):
    """Allows you to introduce a plate with specific properties into the model.

    :param p1: The lower left (x,y,z) coordinates of the plate.
    :type p1: list, non-optional
    :param p2: The upper right (x,y,z) coordinates of the plate.
    :type p2: list, non-optional
    :param material_id: Material identifier that must correspond to material that has already been defined.
    :type material_id: str, non-optional
    :param material_ids:  Material identifiers in the x, y, z directions.
    :type material_ids: list, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash = '#plate'

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
            logger.exception(self.__str__() + ' 2 points must be specified')
            raise

        # isotropic
        try:
            materialsrequested = [self.kwargs['material_id']]
        except KeyError:
            # Anisotropic case
            try:
                materialsrequested = self.kwargs['material_ids']
            except KeyError:
                logger.exception(self.__str__() + ' No materials have been specified')
                raise

        if self.dorotate:
            self.__dorotate()

        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        # Check for valid orientations
        if xs == xf:
            if ys == yf or zs == zf:
                logger.exception(self.__str__() + ' the plate is not specified correctly')
                raise ValueError

        elif ys == yf:
            if xs == xf or zs == zf:
                logger.exception(self.__str__() + ' the plate is not specified correctly')
                raise ValueError

        elif zs == zf:
            if xs == xf or ys == yf:
                logger.exception(self.__str__() + ' the plate is not specified correctly')
                raise ValueError

        else:
            logger.exception(self.__str__() + ' the plate is not specified correctly')
            raise ValueError

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            logger.exception(self.__str__() + f' material(s) {notfound} do not exist')
            raise ValueError

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
                    build_face_yz(xs, j, k, numIDy, numIDz, grid.rigidE, grid.rigidH, grid.ID)

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
                    build_face_xz(i, ys, k, numIDx, numIDz, grid.rigidE, grid.rigidH, grid.ID)

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
                    build_face_xy(i, j, zs, numIDx, numIDy, grid.rigidE, grid.rigidH, grid.ID)

        logger.info(self.grid_name(grid) + f"Plate from {p3[0]:g}m, {p3[1]:g}m, {p3[2]:g}m, to {p4[0]:g}m, {p4[1]:g}m, {p4[2]:g}m of material(s) {', '.join(materialsrequested)} created.")
