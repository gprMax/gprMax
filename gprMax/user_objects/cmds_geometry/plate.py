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
from gprMax.cython.geometry_primitives import build_face_xy, build_face_xz, build_face_yz
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.user_objects.user_objects import GeometryUserObject

from .cmds_geometry import resolve_geometry_materials, validate_geometry_rasterisation

logger = logging.getLogger(__name__)


class Plate(GeometryUserObject):
    """Introduces a plate with specific properties into the model.

    Attributes:
        p1: list of the lower left (x,y,z) coordinates of the plate.
        p2: list of the upper right (x,y,z) coordinates of the plate.
        material_id: string for the material identifier that must correspond
                        to material that has already been defined.
        material_ids: two material identifiers for the plate's tangential
            directions: y/z for a yz-plane plate, x/z for an xz-plane plate,
            or x/y for an xy-plane plate.
    """

    @property
    def hash(self):
        return "#plate"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
        except KeyError:
            logger.exception(f"{self.__str__()} 2 points must be specified")
            raise

        mode = config.get_model_config().mode
        invariant_axis = "xyz".index(mode[-1]) if mode.startswith("2D") else None

        # isotropic
        try:
            materialsrequested = [self.kwargs["material_id"]]
        except KeyError:
            # Anisotropic case
            try:
                materialsrequested = self.kwargs["material_ids"]
            except KeyError:
                logger.exception(f"{self.__str__()} No materials have been specified")
                raise

        uip = self._create_uip(grid)
        # A plate must be exactly flat (zero thickness) on one axis. In 2D
        # mode, if that flat axis turns out to be the invariant one (see
        # the orientation check below), the plate is rejected outright - a
        # wall position on the invariant axis is already a forced PEC/PMC
        # boundary, so a material plate there is moot. For the other two
        # (extent) axes, `inf` resolves the same way as Box's corners
        # (role="lower"/"upper"), which is also exactly right for the
        # invariant axis when it's one of a plate's extent axes (a plate
        # standing normal to a transverse axis, spanning the full 1-cell
        # TM / 2-cell TE invariant thickness). If a user mistakenly puts
        # `inf` on the flat axis, it resolves to a real (non-flat) span
        # like any other axis, and the "not specified correctly" check
        # below catches it.
        p1 = uip.resolve_inf_point(p1, role="lower")
        p2 = uip.resolve_inf_point(p2, role="upper")
        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        plate_within_grid, p1, p2 = uip.check_box_points(p1, p2, self.__str__())

        # Exit early if none of the plate is in this grid as there is
        # nothing else to do.
        if not plate_within_grid:
            if getattr(grid, "is_distributed", False) is True:
                validate_geometry_rasterisation(grid, 0, geometry=self.params_str())
            return

        xs, ys, zs = p1
        xf, yf, zf = p2

        # Check for valid orientations
        if (
            (xs == xf and (ys == yf or zs == zf))
            or (ys == yf and (xs == xf or zs == zf))
            or (zs == zf and (xs == xf or ys == yf))
            or (xs != xf and ys != yf and zs != zf)
        ):
            raise ValueError(f"{self.__str__()} the plate is not specified correctly")

        if invariant_axis is not None:
            flat_axis = 0 if xs == xf else 1 if ys == yf else 2
            if flat_axis == invariant_axis:
                raise ValueError(
                    f"{self.__str__()} in 2D mode, a plate normal to the "
                    f"invariant axis ('{mode[-1]}') would lie exactly on the "
                    "domain wall already forced PEC/PMC there - a material "
                    "plate has no effect in that orientation. Use a plate "
                    "normal to one of the other two axes instead."
                )

        materials = resolve_geometry_materials(
            grid,
            materialsrequested,
            geometry=self.params_str(),
            cell_volume=False,
            directional="material_id" not in self.kwargs,
            directional_count=2,
        )

        occupied = 0

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
                    occupied += 1

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
                    occupied += 1

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
                    occupied += 1

        validate_geometry_rasterisation(grid, occupied, geometry=self.params_str())

        logger.info(
            f"{self.grid_name(grid)}Plate from {p3[0]:g}m, {p3[1]:g}m, "
            f"{p3[2]:g}m, to {p4[0]:g}m, {p4[1]:g}m, {p4[2]:g}m of "
            f"material(s) {', '.join(materialsrequested)} created."
        )
