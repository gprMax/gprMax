# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
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
import math

import numpy as np

import gprMax.config as config
from gprMax.cython.geometry_primitives import build_edge_x, build_edge_y, build_edge_z
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.user_objects.rotatable import RotatableMixin
from gprMax.user_objects.user_objects import GeometryUserObject

from .cmds_geometry import rotate_2point_object

logger = logging.getLogger(__name__)


class Edge(RotatableMixin, GeometryUserObject):
    """Introduces a wire with specific properties into the model.

    Attributes:
        p1: list of the coordinates (x,y,z) of the starting point of the edge.
        p2: list of the coordinates (x,y,z) of the ending point of the edge.
        material_id: string for the material identifier that must correspond
                        to material that has already been defined.
    """

    @property
    def hash(self):
        return "#edge"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        pts = np.array([self.kwargs["p1"], self.kwargs["p2"]])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.kwargs["p1"] = tuple(rot_pts[0, :])
        self.kwargs["p2"] = tuple(rot_pts[1, :])

    def build(self, grid: FDTDGrid):
        """Creates edge and adds it to the grid."""
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            material_id = self.kwargs["material_id"]
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly 3 parameters")
            raise

        if self.do_rotate:
            self._do_rotate(grid)

        uip = self._create_uip(grid)
        mode = config.get_model_config().mode
        invariant_axis = "xyz".index(mode[-1]) if mode.startswith("2D") else None

        orig_p1, orig_p2 = p1, p2
        p1 = uip.resolve_inf_point(p1, role="lower")
        p2 = uip.resolve_inf_point(p2, role="upper")

        # In TE mode, an edge is only physically meaningful running
        # along a NON-invariant axis (an edge along the invariant axis
        # would set the E-component that's forced to pec there, with no
        # effect) - so the invariant axis must be a flat coordinate,
        # constant at the interior reference layer, not a 0/axis-extent
        # span. Override it here regardless of the role-based
        # resolution above, for either endpoint that was `inf` on that
        # axis. (TM is the opposite case - the invariant axis is the
        # only viable run axis there, for which the role-based 0..1-cell
        # span above is already exactly right, no override needed.)
        if invariant_axis is not None and "TE" in mode:
            axis = invariant_axis
            if math.isinf(orig_p1[axis]) or math.isinf(orig_p2[axis]):
                reference = grid.dl[axis] * 1
                p1, p2 = list(p1), list(p2)
                if math.isinf(orig_p1[axis]):
                    p1[axis] = reference
                if math.isinf(orig_p2[axis]):
                    p2[axis] = reference
                p1, p2 = tuple(p1), tuple(p2)

        edge_within_grid, discretised_p1, discretised_p2 = uip.check_box_points(
            p1, p2, self.__str__()
        )

        # Exit early if none of the edge is in this grid as there is
        # nothing else to do.
        if not edge_within_grid:
            return

        xs, ys, zs = discretised_p1
        xf, yf, zf = discretised_p2

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
            or (xs == xf and ys == yf and zs == zf)
        ):
            logger.exception(f"{self.__str__()} the edge is not specified correctly")
            raise ValueError

        if invariant_axis is not None:
            run_axis = 0 if xs != xf else 1 if ys != yf else 2 if zs != zf else None
            if run_axis is not None:
                if "TM" in mode and run_axis != invariant_axis:
                    raise ValueError(
                        f"{self.__str__()} in 2D TM mode, an edge is only "
                        f"physically meaningful running along the invariant "
                        f"axis ('{mode[-1]}') - any other orientation sets an "
                        "E-component that is forced to zero there, with no "
                        "effect."
                    )
                if "TE" in mode and run_axis == invariant_axis:
                    raise ValueError(
                        f"{self.__str__()} in 2D TE mode, an edge running along "
                        f"the invariant axis ('{mode[-1]}') has no physical "
                        "effect - that E-component is forced to zero there. Use "
                        "an edge along one of the other two axes instead."
                    )

        if xs != xf:
            for i in range(xs, xf):
                build_edge_x(i, ys, zs, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        elif ys != yf:
            for j in range(ys, yf):
                build_edge_y(xs, j, zs, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        elif zs != zf:
            for k in range(zs, zf):
                build_edge_z(xs, ys, k, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        logger.info(
            f"{self.grid_name(grid)}Edge from {p3[0]:g}m, {p3[1]:g}m, "
            f"{p3[2]:g}m, to {p4[0]:g}m, {p4[1]:g}m, {p4[2]:g}m of "
            f"material {material_id} created."
        )
