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
from ..cython.geometry_primitives import build_edge_x
from ..cython.geometry_primitives import build_edge_y
from ..cython.geometry_primitives import build_edge_z
from ..exceptions import CmdInputError

log = logging.getLogger(__name__)


class Edge(UserObjectGeometry):
    """Allows you to introduce a wire with specific properties into the model.

    :param p1: Starting point of the edge.
    :type p1: list, non-optional
    :param p2: Ending point of the edge.
    :type p2: list, non-optional
    :param material_id: Material identifier that must correspond to material that has already been defined.
    :type material_id: str, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 2
        self.hash = '#edge'

    def create(self, grid, uip):
        """Create edge and add it to the grid."""
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            material_id = self.kwargs['material_id']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires exactly 3 parameters')

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        material = next((x for x in grid.materials if x.ID == material_id), None)

        if not material:
            raise CmdInputError('Material with ID {} does not exist'.format(material_id))

        # Check for valid orientations
        # x-orientated wire
        if xs != xf:
            if ys != yf or zs != zf:
                raise CmdInputError(self.__str__() + ' the edge is not specified correctly')
            else:
                for i in range(xs, xf):
                    build_edge_x(i, ys, zs, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        # y-orientated wire
        elif ys != yf:
            if xs != xf or zs != zf:
                raise CmdInputError(self.__str__() + ' the edge is not specified correctly')
            else:
                for j in range(ys, yf):
                    build_edge_y(xs, j, zs, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        # z-orientated wire
        elif zs != zf:
            if xs != xf or ys != yf:
                raise CmdInputError(self.__str__() + ' the edge is not specified correctly')
            else:
                for k in range(zs, zf):
                    build_edge_z(xs, ys, k, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        log.info(f'Edge from {xs * grid.dx:g}m, {ys * grid.dy:g}m, {zs * grid.dz:g}m, to {xf * grid.dx:g}m, {yf * grid.dy:g}m, {zf * grid.dz:g}m of material {material_id} created.')
