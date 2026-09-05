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

from gprMax.cython.geometry_primitives import build_ellipsoid
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material
from gprMax.user_objects.cmds_geometry.cmds_geometry import (
    check_averaging,
    geometry_tag_args,
    resolve_geometry_materials,
    validate_geometry_rasterisation,
)
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


class Ellipsoid(GeometryUserObject):
    """Introduces an ellipsoidal object with specific parameters into the model.

    Attributes:
        p1: list of the coordinates (x,y,z) of the centre of the ellipsoid.
        xr: float for x-semiaxis of the ellipsoid.
        yr: float for y-semiaxis of the ellipsoid.
        zr: float for z-semiaxis of the ellipsoid.
        material_id: string for the material identifier that must correspond
                        to material that has already been defined.
        material_ids: list of material identifiers in the x, y, z directions.
        averaging: string (y or n) used to switch on and off dielectric smoothing.
        tag: optional semantic geometry-tag string written to occupied cells.
    """

    @property
    def hash(self):
        return "#ellipsoid"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            p1 = self.kwargs["p1"]
            xr = self.kwargs["xr"]
            yr = self.kwargs["yr"]
            zr = self.kwargs["zr"]

        except KeyError:
            logger.exception(f"{self.__str__()} please specify a point and the three semiaxes.")
            raise

        if not np.all(np.isfinite((xr, yr, zr))) or xr <= 0 or yr <= 0 or zr <= 0:
            message = (
                f"{self.__str__()} the semiaxes ({xr:g}, {yr:g}, {zr:g}) "
                "should all be positive values."
            )
            logger.error(message)
            raise ValueError(message)

        # Check averaging
        try:
            # Try user-specified averaging
            averageellipsoid = check_averaging(self.kwargs["averaging"])
        except KeyError:
            # Otherwise go with the grid default
            averageellipsoid = grid.averagevolumeobjects

        # Check materials have been specified
        # Isotropic case
        try:
            materialsrequested = [self.kwargs["material_id"]]
        except KeyError:
            # Anisotropic case
            try:
                materialsrequested = self.kwargs["material_ids"]
            except KeyError:
                logger.exception(f"{self.__str__()} no materials have been specified")
                raise

        # Centre of ellipsoid
        uip = self._create_uip(grid)
        p2 = uip.round_to_grid_static_point(p1)
        xc, yc, zc = uip.discretise_point(p1)

        materials = resolve_geometry_materials(
            grid,
            materialsrequested,
            geometry=self.params_str(),
            directional="material_id" not in self.kwargs,
        )

        # Isotropic case
        if len(materials) == 1:
            averaging = materials[0].averagable and averageellipsoid
            numID = numIDx = numIDy = numIDz = materials[0].numID
            pec_x = pec_y = pec_z = materials[0].is_pec

        # Uniaxial anisotropic case
        elif len(materials) == 3:
            averaging = False
            numIDx = materials[0].numID
            numIDy = materials[1].numID
            numIDz = materials[2].numID
            pec_x = materials[0].is_pec
            pec_y = materials[1].is_pec
            pec_z = materials[2].is_pec
            requiredID = Material.create_compound_id(materials[0], materials[1], materials[2])
            averagedmaterial = [x for x in grid.materials if x.ID == requiredID]
            if averagedmaterial:
                numID = averagedmaterial[0].numID
            else:
                numID = len(grid.materials)
                m = Material(numID, requiredID)
                m.type = "dielectric-smoothed"
                # Create dielectric-smoothed constituents for material
                m.er = np.mean((materials[0].er, materials[1].er, materials[2].er), axis=0)
                m.se = np.mean((materials[0].se, materials[1].se, materials[2].se), axis=0)
                m.mr = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)
                m.sm = np.mean((materials[0].sm, materials[1].sm, materials[2].sm), axis=0)

                # Append the new material object to the materials list
                grid.materials.append(m)

        tag_data, tag_id = geometry_tag_args(grid, self.kwargs.get("tag"))
        occupied = build_ellipsoid(
            xc,
            yc,
            zc,
            xr,
            yr,
            zr,
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
            tag_data,
            tag_id,
        )
        validate_geometry_rasterisation(grid, occupied, geometry=self.params_str())

        dielectricsmoothing = "on" if averaging else "off"
        logger.info(
            f"{self.grid_name(grid)}Ellipsoid with centre {p2[0]:g}m, "
            f"{p2[1]:g}m, {p2[2]:g}m, x-semiaxis {xr:g}m, "
            f"y-semiaxis {yr:g}m and z-semiaxis {zr:g}m of material(s) "
            f"{', '.join(materialsrequested)} created, dielectric "
            f"smoothing is {dielectricsmoothing}."
        )
