# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import logging

import gprMax.config as config
from gprMax.cython.geometry_primitives import (
    build_magnetic_edge_x,
    build_magnetic_edge_y,
    build_magnetic_edge_z,
)
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.user_objects.user_objects import GeometryUserObject

from .cmds_geometry import resolve_geometry_materials

logger = logging.getLogger(__name__)


class MagneticEdge(GeometryUserObject):
    """Introduces a single magnetic (H) edge with specific properties into
        the model - the magnetic dual of #edge.

    Attributes:
        p1: list of the coordinates (x,y,z) of the starting point of the edge.
        p2: list of the coordinates (x,y,z) of the ending point of the edge.
        material_id: string for the material identifier that must correspond
                        to material that has already been defined.
    """

    @property
    def hash(self):
        return "#magnetic_edge"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        """Creates a magnetic edge and adds it to the grid."""
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            material_id = self.kwargs["material_id"]
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly 3 parameters")
            raise

        if config.get_model_config().mode.startswith("2D"):
            raise ValueError(
                f"{self.__str__()} is not yet supported in 2D mode - which axis "
                "is physically meaningful for a magnetic edge differs from "
                "#edge's own 2D rule and has not yet been verified."
            )

        uip = self._create_uip(grid)

        p1 = uip.resolve_inf_point(p1, role="lower")
        p2 = uip.resolve_inf_point(p2, role="upper")

        edge_within_grid, discretised_p1, discretised_p2 = uip.check_box_points(
            p1, p2, self.__str__()
        )

        # Exit early if none of the edge is in this grid as there is
        # nothing else to do.
        if not edge_within_grid:
            return

        xs, ys, zs = discretised_p1
        xf, yf, zf = discretised_p2

        material = resolve_geometry_materials(
            grid,
            [material_id],
            geometry=self.params_str(),
            cell_volume=False,
        )[0]

        # Check for valid orientations
        if (
            (xs != xf and (ys != yf or zs != zf))
            or (ys != yf and (xs != xf or zs != zf))
            or (zs != zf and (xs != xf or ys != yf))
            or (xs == xf and ys == yf and zs == zf)
        ):
            logger.exception(f"{self.__str__()} the edge is not specified correctly")
            raise ValueError

        if xs != xf:
            for i in range(xs, xf):
                build_magnetic_edge_x(i, ys, zs, material.numID, grid.rigidH, grid.ID)

        elif ys != yf:
            for j in range(ys, yf):
                build_magnetic_edge_y(xs, j, zs, material.numID, grid.rigidH, grid.ID)

        elif zs != zf:
            for k in range(zs, zf):
                build_magnetic_edge_z(xs, ys, k, material.numID, grid.rigidH, grid.ID)

        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        logger.info(
            f"{self.grid_name(grid)}Magnetic edge from {p3[0]:g}m, {p3[1]:g}m, "
            f"{p3[2]:g}m, to {p4[0]:g}m, {p4[1]:g}m, {p4[2]:g}m of "
            f"material {material_id} created."
        )
