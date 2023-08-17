# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
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

from ..cython.geometry_primitives import build_edge_x, build_edge_y, build_edge_z
from .cmds_geometry import UserObjectGeometry, rotate_2point_object

logger = logging.getLogger(__name__)


class Edge(UserObjectGeometry):
    """Introduces a wire with specific properties into the model.

    Attributes:
        p1: list of the coordinates (x,y,z) of the starting point of the edge.
        p2: list of the coordinates (x,y,z) of the ending point of the edge.
        material_id: string for the material identifier that must correspond
                        to material that has already been defined.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash = "#edge"

    def rotate(self, axis, angle, origin=None):
        """Set parameters for rotation."""
        self.axis = axis
        self.angle = angle
        self.origin = origin
        self.do_rotate = True

    def _do_rotate(self):
        """Performs rotation."""
        pts = np.array([self.kwargs["p1"], self.kwargs["p2"]])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.kwargs["p1"] = tuple(rot_pts[0, :])
        self.kwargs["p2"] = tuple(rot_pts[1, :])

    def create(self, grid, uip):
        """Creates edge and adds it to the grid."""
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            material_id = self.kwargs["material_id"]
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly 3 parameters")
            raise

        if self.do_rotate:
            self._do_rotate()

        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        material = next((x for x in grid.materials if x.ID == material_id), None)

        if not material:
            logger.exception(f"Material with ID {material_id} does not exist")
            raise ValueError

        # Check for valid orientations
        # x-orientated edge
        if (
            (xs != xf and (ys != yf or zs != zf))
            or (ys != yf and (xs != xf or zs != zf))
            or (zs != zf and (xs != xf or ys != yf))
        ):
            logger.exception(f"{self.__str__()} the edge is not specified correctly")
            raise ValueError
        elif xs != xf:
            for i in range(xs, xf):
                build_edge_x(i, ys, zs, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        elif ys != yf:
            for j in range(ys, yf):
                build_edge_y(xs, j, zs, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        elif zs != zf:
            for k in range(zs, zf):
                build_edge_z(xs, ys, k, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        logger.info(
            f"{self.grid_name(grid)}Edge from {p3[0]:g}m, {p3[1]:g}m, "
            f"{p3[2]:g}m, to {p4[0]:g}m, {p4[1]:g}m, {p4[2]:g}m of "
            f"material {material_id} created."
        )
