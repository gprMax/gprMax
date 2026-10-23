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

import numpy as np

import gprMax.config as config
from gprMax.cython.geometry_primitives import build_helix
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import create_directional_material
from gprMax.user_objects.cmds_geometry.cmds_geometry import (
    check_averaging,
    geometry_tag_args,
    resolve_geometry_materials,
    validate_geometry_rasterisation,
)
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


def _pec_yee_edge_components(grid, occupancy, edge_ids):
    """Count connected components in the PEC Yee-edge network."""
    offset = np.zeros(3, dtype=np.int64)
    if getattr(grid, "is_distributed", False) is True:
        offset = np.asarray(grid.lower_extent, dtype=np.int64)
    ox, oy, oz = (int(value) for value in offset)
    local_edges = set()
    for cell in np.argwhere(occupancy):
        i, j, k = (int(value) for value in cell)
        for dj in (0, 1):
            for dk in (0, 1):
                if grid.ID[0, i, j + dj, k + dk] == edge_ids[0]:
                    a = (i + ox, j + dj + oy, k + dk + oz)
                    local_edges.add((a, (a[0] + 1, a[1], a[2])))
        for di in (0, 1):
            for dk in (0, 1):
                if grid.ID[1, i + di, j, k + dk] == edge_ids[1]:
                    a = (i + di + ox, j + oy, k + dk + oz)
                    local_edges.add((a, (a[0], a[1] + 1, a[2])))
        for di in (0, 1):
            for dj in (0, 1):
                if grid.ID[2, i + di, j + dj, k] == edge_ids[2]:
                    a = (i + di + ox, j + dj + oy, k + oz)
                    local_edges.add((a, (a[0], a[1], a[2] + 1)))

    groups = [(int(np.count_nonzero(occupancy)), local_edges)]
    if getattr(grid, "is_distributed", False) is True:
        groups = grid.comm.allgather(groups[0])
    cells = sum(count for count, _ in groups)
    if cells == 0:
        return 0, 0

    edges = set().union(*(edge_set for _, edge_set in groups))
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    return len({find(node) for edge in edges for node in edge}), cells


class Helix(GeometryUserObject):
    """Helical wire of circular cross-section around the axis from p1 to p2.

    The centreline starts at ``p1`` and advances towards ``p2``; only the
    direction of ``p2`` matters. The axial length is ``pitch * turns``, and
    with ``pitch`` zero the wire closes into ``turns`` loops in the plane of
    the base.

    Attributes:
        p1: list of the coordinates (x,y,z) of the centre of the helix base.
        p2: list of the coordinates (x,y,z) of a point defining the helix axis.
        coil_radius: float of the radius of the helical centreline.
        pitch: float of the axial advance per complete turn.
        turns: float of the number of complete turns.
        wire_radius: float of the radius of the circular wire cross-section.
        handedness: string ``r`` or ``l`` for a right- or left-handed helix.
        material_id: string for the material identifier that must correspond
                        to material that has already been defined.
        material_ids: list of material identifiers in the x, y, z directions.
        averaging: string (y or n) used to switch on and off dielectric smoothing.
        tag: optional semantic geometry-tag string written to occupied cells.
    """

    @property
    def hash(self):
        return "#helix"

    _allowed_kwargs = frozenset({
        "averaging",
        "coil_radius",
        "handedness",
        "material_id",
        "material_ids",
        "p1",
        "p2",
        "pitch",
        "tag",
        "turns",
        "wire_radius",
    })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid) -> None:
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            coil_radius = self.kwargs["coil_radius"]
            pitch = self.kwargs["pitch"]
            turns = self.kwargs["turns"]
            wire_radius = self.kwargs["wire_radius"]
            handedness = self.kwargs["handedness"]
        except KeyError:
            logger.exception(
                f"{self.__str__()} please specify two points, coil radius, pitch, "
                "turns, wire radius and handedness"
            )
            raise

        if config.get_model_config().mode.startswith("2D"):
            raise ValueError(
                f"{self.__str__()} cannot be used in 2D mode - a helix is not "
                "invariant along either transverse axis."
            )

        if not all(np.isfinite(p1)) or not all(np.isfinite(p2)):
            message = f"{self.__str__()} helix points must have finite coordinates."
            logger.error(message)
            raise ValueError(message)

        if not np.isfinite(coil_radius) or coil_radius <= 0:
            message = (
                f"{self.__str__()} the coil radius {coil_radius:g} should be a "
                "positive value."
            )
            logger.error(message)
            raise ValueError(message)

        if not np.isfinite(wire_radius) or wire_radius <= 0:
            message = (
                f"{self.__str__()} the wire radius {wire_radius:g} should be a "
                "positive value."
            )
            logger.error(message)
            raise ValueError(message)

        if not np.isfinite(pitch) or pitch < 0:
            message = (
                f"{self.__str__()} the pitch {pitch:g} should be zero or a positive "
                "value."
            )
            logger.error(message)
            raise ValueError(message)

        if not np.isfinite(turns) or turns <= 0:
            message = (
                f"{self.__str__()} the number of turns {turns:g} should be a "
                "positive value."
            )
            logger.error(message)
            raise ValueError(message)

        if not isinstance(handedness, str) or handedness.lower() not in {"r", "l"}:
            message = f"{self.__str__()} handedness must be either 'r' or 'l'."
            logger.error(message)
            raise ValueError(message)
        handedness = handedness.lower()

        try:
            averagehelix = check_averaging(self.kwargs["averaging"])
        except KeyError:
            averagehelix = grid.averagevolumeobjects

        try:
            materialsrequested = [self.kwargs["material_id"]]
        except KeyError:
            try:
                materialsrequested = self.kwargs["material_ids"]
            except KeyError:
                logger.exception(f"{self.__str__()} no materials have been specified")
                raise

        uip = self._create_uip(grid)
        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)
        x1, y1, z1 = uip.round_to_grid(p1)
        x2, y2, z2 = uip.round_to_grid(p2)

        if (x1, y1, z1) == (x2, y2, z2):
            message = (
                f"{self.__str__()} the base and axis-defining points must occupy "
                "different grid points."
            )
            logger.error(message)
            raise ValueError(message)

        materials = resolve_geometry_materials(
            grid,
            materialsrequested,
            geometry=self.params_str(),
            directional="material_id" not in self.kwargs,
        )

        if len(materials) == 1:
            averaging = materials[0].averagable and averagehelix
            numID = numIDx = numIDy = numIDz = materials[0].numID
            pec_x = pec_y = pec_z = materials[0].is_pec

        elif len(materials) == 3:
            averaging = False
            numIDx = materials[0].numID
            numIDy = materials[1].numID
            numIDz = materials[2].numID
            pec_x = materials[0].is_pec
            pec_y = materials[1].is_pec
            pec_z = materials[2].is_pec
            numID = create_directional_material(grid, materials).numID

        fully_pec = pec_x and pec_y and pec_z
        cell_size = min(grid.dx, grid.dy, grid.dz)
        if not fully_pec and 2 * wire_radius < cell_size:
            message = (
                f"{self.__str__()} the wire diameter {2 * wire_radius:g}m is "
                f"smaller than the grid cell size {cell_size:g}m. Increase the "
                "wire radius or reduce the discretisation."
            )
            logger.error(message)
            raise ValueError(message)

        tag_data, tag_id = geometry_tag_args(grid, self.kwargs.get("tag"))
        connectivity_mask = None
        if fully_pec:
            connectivity_mask = np.zeros(grid.solid.shape, dtype=np.uint8)
        occupied = build_helix(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            coil_radius,
            pitch,
            turns,
            wire_radius,
            handedness == "r",
            grid.dx,
            grid.dy,
            grid.dz,
            numID,
            numIDx,
            numIDy,
            numIDz,
            averaging,
            pec_x,
            pec_y,
            pec_z,
            grid.solid,
            grid.rigidE,
            grid.rigidH,
            grid.ID,
            connectivity_mask if fully_pec else tag_data,
            1 if fully_pec else tag_id,
        )
        validate_geometry_rasterisation(grid, occupied, geometry=self.params_str())

        if fully_pec:
            if tag_data is not None:
                tag_data[connectivity_mask != 0] = tag_id
            components, cells = _pec_yee_edge_components(
                grid, connectivity_mask, (numIDx, numIDy, numIDz)
            )
            if cells > 0 and components != 1:
                detail = (
                    "no PEC Yee edges"
                    if components == 0
                    else f"{components} disconnected PEC Yee-edge components"
                )
                message = (
                    f"{self.__str__()} the requested wire radius {wire_radius:g}m "
                    f"is under-resolved: it produces {detail} on the "
                    f"({grid.dx:g}, {grid.dy:g}, {grid.dz:g})m grid. Increase the "
                    "wire radius or reduce the discretisation; the requested "
                    "radius is not changed automatically."
                )
                logger.error(message)
                raise ValueError(message)

        largest_cell = max(grid.dx, grid.dy, grid.dz)
        if 2 * wire_radius < largest_cell:
            logger.warning(
                f"{self.__str__()} the wire diameter {2 * wire_radius:g}m is "
                f"smaller than the largest grid cell size {largest_cell:g}m; the "
                "requested radius is preserved, but the rasterised wire may be "
                "under-resolved in one or more directions."
            )
        if (pec_x or pec_y or pec_z) and not fully_pec:
            logger.warning(
                f"{self.__str__()} PEC Yee-edge connectivity is not checked when "
                "only some material directions are PEC."
            )

        dielectricsmoothing = "on" if averaging else "off"
        logger.info(
            f"{self.grid_name(grid)}Helix with base centre {p3[0]:g}m, "
            f"{p3[1]:g}m, {p3[2]:g}m and axis towards {p4[0]:g}m, "
            f"{p4[1]:g}m, {p4[2]:g}m, coil radius {coil_radius:g}m, "
            f"pitch {pitch:g}m, {turns:g} turn(s), wire radius "
            f"{wire_radius:g}m and {handedness}-handedness, of material(s) "
            f"{', '.join(materialsrequested)} created, dielectric smoothing is "
            f"{dielectricsmoothing}."
        )
