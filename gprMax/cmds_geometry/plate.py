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

import gprMax.config as config
from .cmds_geometry import UserObjectGeometry
from ..cython.geometry_primitives import build_face_yz
from ..cython.geometry_primitives import build_face_xz
from ..cython.geometry_primitives import build_face_xy
from ..exceptions import CmdInputError

log = logging.getLogger(__name__)


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
        self.order = 3
        self.hash = '#plate'

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
        except KeyError:
            raise CmdInputError(self.__str__() + ' 2 points must be specified')

        # isotropic
        try:
            materialsrequested = [self.kwargs['material_id']]
        except KeyError:
            # Anisotropic case
            try:
                materialsrequested = self.kwargs['material_ids']
            except KeyError:
                raise CmdInputError(self.__str__() + ' No materials have been specified')

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        # Check for valid orientations
        if xs == xf:
            if ys == yf or zs == zf:
                raise CmdInputError(self.__str__() + ' the plate is not specified correctly')

        elif ys == yf:
            if xs == xf or zs == zf:
                raise CmdInputError(self.__str__() + ' the plate is not specified correctly')

        elif zs == zf:
            if xs == xf or ys == yf:
                raise CmdInputError(self.__str__() + ' the plate is not specified correctly')

        else:
            raise CmdInputError(self.__str__() + ' the plate is not specified correctly')

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

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

        log.info(f"Plate from {xs * grid.dx:g}m, {ys * grid.dy:g}m, {zs * grid.dz:g}m, to {xf * grid.dx:g}m, {yf * grid.dy:g}m, {zf * grid.dz:g}m of material(s) {', '.join(materialsrequested)} created.")
