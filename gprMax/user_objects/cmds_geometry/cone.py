# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
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

from gprMax.cython.geometry_primitives import build_cone
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material
from gprMax.user_objects.cmds_geometry.cmds_geometry import check_averaging
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


class Cone(GeometryUserObject):
    """Introduces a circular cone into the model. The difference with the cylinder is that the faces of the cone
       can have different radii and one of them can be zero.

    Attributes:
        p1: list of the coordinates (x,y,z) of the centre of the first face
            of the cone.
        p2: list of the coordinates (x,y,z) of the centre of the second face
            of the cone.
        r1: float of the radius of the first face of the cone.
        r2: float of the radius of the second face of the cone.
        material_id: string for the material identifier that must correspond
                        to material that has already been defined.
        material_ids: list of material identifiers in the x, y, z directions.
        averaging: string (y or n) used to switch on and off dielectric smoothing.
    """

    @property
    def hash(self):
        return "#cone"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid) -> None:
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            r1 = self.kwargs["r1"]
            r2 = self.kwargs["r2"]
        except KeyError:
            logger.exception(f"{self.__str__()} please specify two points and two radii")
            raise

        # Check averaging
        try:
            # Try user-specified averaging
            averagecone = check_averaging(self.kwargs["averaging"])
        except KeyError:
            # Otherwise go with the grid default
            averagecone = grid.averagevolumeobjects

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

        uip = self._create_uip(grid)
        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        x1, y1, z1 = uip.round_to_grid(p1)
        x2, y2, z2 = uip.round_to_grid(p2)

        if r1 < 0:
            logger.exception(
                f"{self.__str__()} the radius of the first face {r1:g} should be a positive value."
            )
            raise ValueError

        if r2 < 0:
            logger.exception(
                f"{self.__str__()} the radius of the second face {r2:g} should be a positive value."
            )
            raise ValueError

        if r1 == 0 and r2 == 0:
            logger.exception(f"{self.__str__()} both radii cannot be zero.")
            raise ValueError

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            logger.exception(f"{self.__str__()} material(s) {notfound} do not exist")
            raise ValueError

        # Isotropic case
        if len(materials) == 1:
            averaging = materials[0].averagable and averagecone
            numID = numIDx = numIDy = numIDz = materials[0].numID

        # Uniaxial anisotropic case
        elif len(materials) == 3:
            averaging = False
            numIDx = materials[0].numID
            numIDy = materials[1].numID
            numIDz = materials[2].numID
            requiredID = Material.create_compound_id(materials[0], materials[1], materials[2])
            averagedmaterial = [x for x in grid.materials if x.ID == requiredID]
            if averagedmaterial:
                numID = averagedmaterial.numID
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

        build_cone(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            r1,
            r2,
            grid.dx,
            grid.dy,
            grid.dz,
            numID,
            numIDx,
            numIDy,
            numIDz,
            averaging,
            grid.solid,
            grid.rigidE,
            grid.rigidH,
            grid.ID,
        )

        dielectricsmoothing = "on" if averaging else "off"
        logger.info(
            f"{self.grid_name(grid)}Cone with face centres {p3[0]:g}m, "
            f"{p3[1]:g}m, {p3[2]:g}m and {p4[0]:g}m, {p4[1]:g}m, {p4[2]:g}m, "
            f"with radii {r1:g}m and {r2:g}, of material(s) {', '.join(materialsrequested)} "
            f"created, dielectric smoothing is {dielectricsmoothing}."
        )
