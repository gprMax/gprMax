# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
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

"""Sub-cell thin-wire geometry for the Cartesian Yee grid.

A thin wire is represented by PEC electric edges and the improved projected-H
correction of Mäkinen, Juntunen, and Kivikoski. The electric geometry is
registered while scene geometry is parsed;
the component IDs are applied later by :meth:`FDTDGrid._build_thin_wires`,
after ordinary electric/magnetic material averaging has resolved the actual
background at every Yee component.
"""

import logging
import math

import numpy as np

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.user_objects.rotatable import RotatableMixin
from gprMax.user_objects.user_objects import GeometryUserObject

from .cmds_geometry import rotate_2point_object

logger = logging.getLogger(__name__)


class ThinWire(RotatableMixin, GeometryUserObject):
    """Introduces an axis-aligned PEC thin wire with physical radius ``a``.

    The logarithmic radius factor follows Umashankar, Taflove, and Beker
    (IEEE TAP, 1987, doi:10.1109/TAP.1987.1144000). The surrounding magnetic
    updates use the improved projected-field factors of Mäkinen, Juntunen,
    and Kivikoski (IEEE T-MTT, 2002). The charge-based open-end correction
    from the latter paper is not implemented.

    Args:
        p1: Coordinates of the start of the wire.
        p2: Coordinates of the end of the wire.
        radius: Physical wire radius in metres.
    """

    @property
    def hash(self):
        return "#thin_wire"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wire_axis = None
        self.start = None
        self.stop = None
        self.radius = None

    def _do_rotate(self, grid: FDTDGrid):
        """Rotate the endpoints before discretisation."""
        points = np.array([self.kwargs["p1"], self.kwargs["p2"]])
        rotated = rotate_2point_object(points, self.axis, self.angle, self.origin)
        self.kwargs["p1"] = tuple(rotated[0, :])
        self.kwargs["p2"] = tuple(rotated[1, :])

    def build(self, grid: FDTDGrid):
        """Validate and register the wire for post-component construction."""
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            radius = float(self.kwargs["radius"])
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly 3 parameters")
            raise

        if config.get_model_config().mode.startswith("2D"):
            raise ValueError(f"{self.__str__()} is not yet supported in 2D mode.")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"{self.__str__()} requires a finite radius greater than zero.")

        if self.do_rotate:
            self._do_rotate(grid)
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]

        uip = self._create_uip(grid)
        if hasattr(grid, "comm"):
            self._build_mpi(grid, uip, p1, p2, radius)
            return
        p1 = uip.resolve_inf_point(p1, role="lower")
        p2 = uip.resolve_inf_point(p2, role="upper")
        within_grid, start, stop = uip.check_box_points(p1, p2, self.__str__())
        if not within_grid:
            return

        changed = np.flatnonzero(start != stop)
        if changed.size != 1:
            raise ValueError(f"{self.__str__()} must define one non-zero, axis-aligned line.")

        axis_index = int(changed[0])
        transverse = [index for index in range(3) if index != axis_index]
        limiting_step = min(float(grid.dl[index]) for index in transverse)
        if radius >= 0.5 * limiting_step:
            raise ValueError(
                f"{self.__str__()} radius {radius:g}m must be smaller than half "
                f"the minimum transverse cell size ({0.5 * limiting_step:g}m)."
            )

        self.wire_axis = "xyz"[axis_index]
        self.start = np.asarray(start, dtype=np.int32)
        self.stop = np.asarray(stop, dtype=np.int32)
        self.radius = radius
        grid.thinwires.append(self)

        rounded_start = uip.round_to_grid_static_point(p1)
        rounded_stop = uip.round_to_grid_static_point(p2)
        logger.info(
            f"{self.grid_name(grid)}Thin wire from {rounded_start[0]:g}m, "
            f"{rounded_start[1]:g}m, {rounded_start[2]:g}m, to "
            f"{rounded_stop[0]:g}m, {rounded_stop[1]:g}m, "
            f"{rounded_stop[2]:g}m, radius {radius:g}m, created."
        )

    def _build_mpi(self, grid, uip, p1, p2, radius):
        """Register one global wire definition on every MPI rank.

        Component construction later assigns each electric and magnetic edge
        to exactly one owning rank. Keeping the uncut global endpoints here is
        essential because the surrounding four-H-edge stencil can straddle a
        decomposition boundary even though the wire E edge has one owner.
        """

        domain = np.asarray(grid.global_size, dtype=np.int32)
        p1 = uip.resolve_inf_point(p1, role="lower")
        p2 = uip.resolve_inf_point(p2, role="upper")
        start = np.asarray(uip.discretise_static_point(p1), dtype=np.int32)
        stop = np.asarray(uip.discretise_static_point(p2), dtype=np.int32)
        if np.any(start < 0) or np.any(stop > domain):
            raise ValueError(f"{self.__str__()} endpoints must lie inside the global MPI domain.")
        changed = np.flatnonzero(start != stop)
        if changed.size != 1 or np.any(start > stop):
            raise ValueError(f"{self.__str__()} must define one non-zero, axis-aligned line.")

        axis_index = int(changed[0])
        transverse = [index for index in range(3) if index != axis_index]
        limiting_step = min(float(grid.dl[index]) for index in transverse)
        if radius >= 0.5 * limiting_step:
            raise ValueError(
                f"{self.__str__()} radius {radius:g}m must be smaller than half "
                f"the minimum transverse cell size ({0.5 * limiting_step:g}m)."
            )

        self.wire_axis = "xyz"[axis_index]
        self.global_start = start
        self.global_stop = stop
        # Retain local aliases for diagnostics; MPI construction uses the
        # global endpoints above.
        self.start = grid.global_to_local_coordinate(start)
        self.stop = grid.global_to_local_coordinate(stop)
        self.radius = radius
        grid.thinwires.append(self)

        if grid.is_coordinator():
            rounded_start = start * grid.dl
            rounded_stop = stop * grid.dl
            logger.info(
                f"Thin wire from {rounded_start[0]:g}m, {rounded_start[1]:g}m, "
                f"{rounded_start[2]:g}m, to {rounded_stop[0]:g}m, "
                f"{rounded_stop[1]:g}m, {rounded_stop[2]:g}m, radius "
                f"{radius:g}m, created."
            )

    def cells(self):
        """Yield the electric Yee-edge coordinates occupied by the wire."""
        axis_index = "xyz".index(self.wire_axis)
        for position in range(int(self.start[axis_index]), int(self.stop[axis_index])):
            cell = self.start.copy()
            cell[axis_index] = position
            yield tuple(int(value) for value in cell)
