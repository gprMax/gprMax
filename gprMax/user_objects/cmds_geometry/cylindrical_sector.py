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
from gprMax.cython.geometry_primitives import build_cylindrical_sector
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


class CylindricalSector(GeometryUserObject):
    """Introduces a cylindrical sector (shaped like a slice of pie) into the model.

    Attributes:
        normal: string for the direction of the axis of the cylinder from which
                the sector is defined and can be x, y, or z.
        ctr1: float for the first coordinate of the centre of the cylindrical
                sector.
        ctr2: float for the second coordinate of the centre of the cylindrical
                sector.
        extent1: float for the first thickness from the centre of the
                    cylindrical sector.
        extent2: float for the second thickness from the centre of the
                    cylindrical sector.
        r: float for the radius of the cylindrical sector.
        start: float for the starting angle (in degrees) for the cylindrical
                sector.
        end: float for the angle (in degrees) swept by the cylindrical sector.
        material_id: string for the material identifier that must correspond
                        to material that has already been defined.
        material_ids: list of material identifiers in the x, y, z directions.
        averaging: string (y or n) used to switch on and off dielectric smoothing.
        tag: optional semantic geometry-tag string written to occupied cells.
    """

    @property
    def hash(self):
        return "#cylindrical_sector"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        try:
            normal = self.kwargs["normal"].lower()
            ctr1 = self.kwargs["ctr1"]
            ctr2 = self.kwargs["ctr2"]
            extent1 = self.kwargs["extent1"]
            extent2 = self.kwargs["extent2"]
            start = self.kwargs["start"]
            end = self.kwargs["end"]
            r = self.kwargs["r"]
        except KeyError:
            logger.exception(self.__str__())
            raise

        if normal not in ["x", "y", "z"]:
            logger.exception(f"{self.__str__()} the normal direction must be either x, y or z.")
            raise ValueError

        uip = self._create_uip(grid)

        # In 2D mode, a sector is only meaningful with its own (normal)
        # axis matching the invariant axis - like #cylinder, this lets it
        # span the full 1-cell (TM) or 2-cell (TE) thickness via `inf`,
        # wall-to-wall. Unlike #cylinder's p1/p2 (full 3D points), a
        # sector's cross-section coordinates (ctr1/ctr2) are two scalars
        # in the plane perpendicular to `normal` - a sector whose normal
        # is a *transverse* axis would need one of ctr1/ctr2 itself to
        # span the invariant thickness (the sector's footprint would need
        # to reach both TE cells), which isn't supported here, so that
        # orientation is rejected outright in 2D rather than silently
        # producing a cross-section that only exists on one cell.
        mode = config.get_model_config().mode
        if mode.startswith("2D"):
            invariant_letter = mode[-1]
            if normal != invariant_letter:
                raise ValueError(
                    f"{self.__str__()} in 2D mode, the normal axis must match the "
                    f"invariant axis ('{invariant_letter}') - a sector normal to a "
                    "transverse axis is not supported."
                )

        # extent1/extent2 are scalars along `normal` - resolve `inf` the
        # same way #cylinder resolves its own-axis p1/p2 coordinates
        # (role="lower"/"upper"), via a throwaway 3-tuple carrying the
        # value in the correct axis slot. resolve_inf_point() is a no-op
        # when there's no `inf` present, so this is safe to call
        # unconditionally in 3D too - it naturally raises the standard
        # "inf' is only allowed... in 2D mode" error there if `inf` is
        # used, rather than a raw crash further down.
        axis_index = "xyz".index(normal)
        lower_point = [0.0, 0.0, 0.0]
        lower_point[axis_index] = extent1
        upper_point = [0.0, 0.0, 0.0]
        upper_point[axis_index] = extent2
        extent1 = uip.resolve_inf_point(tuple(lower_point), role="lower")[axis_index]
        extent2 = uip.resolve_inf_point(tuple(upper_point), role="upper")[axis_index]

        if not np.all(np.isfinite((ctr1, ctr2, extent1, extent2, start, end, r))):
            raise ValueError(f"{self.__str__()} dimensions and angles must all be finite.")

        thickness = extent2 - extent1

        # Check thickness of the object first as may be able to exit
        # early if fully outside the grid.

        # yz-plane cylindrical sector
        if normal == "x":
            level, ctr1, ctr2 = uip.round_to_grid((extent1, ctr1, ctr2))

        # xz-plane cylindrical sector
        elif normal == "y":
            ctr1, level, ctr2 = uip.round_to_grid((ctr1, extent1, ctr2))

        # xy-plane cylindrical sector
        elif normal == "z":
            ctr1, ctr2, level = uip.round_to_grid((ctr1, ctr2, extent1))

        sector_within_grid, level, thickness = uip.check_thickness(
            normal, extent1, thickness, self.__str__()
        )

        # Exit early if none of the cylindrical sector is in this grid
        # as there is nothing else to do.
        if not sector_within_grid:
            if getattr(grid, "is_distributed", False) is True:
                validate_geometry_rasterisation(grid, 0, geometry=self.params_str())
            return

        # Check averaging
        try:
            # Try user-specified averaging
            averagecylindricalsector = check_averaging(self.kwargs["averaging"])
        except KeyError:
            # Otherwise go with the grid default
            averagecylindricalsector = grid.averagevolumeobjects

        # Check materials have been specified
        # Isotropic case
        try:
            materialsrequested = [self.kwargs["material_id"]]
        except KeyError:
            # Anisotropic case
            try:
                materialsrequested = self.kwargs["material_ids"]
            except KeyError:
                logger.exception(f"{self.__str__()} No materials have been specified")
                raise

        sectorstartangle = 2 * np.pi * (start / 360)
        sectorangle = 2 * np.pi * (end / 360)

        if r <= 0:
            message = f"{self.__str__()} the radius {r:g} should be a positive value."
            logger.error(message)
            raise ValueError(message)
        if sectorstartangle < 0 or sectorangle <= 0:
            logger.exception(
                f"{self.__str__()} the starting angle and sector angle should be a positive values."
            )
            raise ValueError
        if sectorstartangle >= 2 * np.pi or sectorangle >= 2 * np.pi:
            logger.exception(
                f"{self.__str__()} the starting angle and sector angle must be less than 360 degrees."
            )
            raise ValueError

        materials = resolve_geometry_materials(
            grid,
            materialsrequested,
            geometry=self.params_str(),
            cell_volume=thickness > 0,
            directional="material_id" not in self.kwargs,
        )

        if thickness > 0:
            # Isotropic case
            if len(materials) == 1:
                averaging = materials[0].averagable and averagecylindricalsector
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
        else:
            averaging = False
            # Isotropic case
            if len(materials) == 1:
                numID = numIDx = numIDy = numIDz = materials[0].numID
                pec_x = pec_y = pec_z = materials[0].is_pec

            # Uniaxial anisotropic case
            elif len(materials) == 3:
                # The typed volumetric ID is unused for a surface sector, but
                # the Python/Cython boundary still requires a valid integer.
                numID = materials[0].numID
                numIDx = materials[0].numID
                numIDy = materials[1].numID
                numIDz = materials[2].numID
                pec_x = materials[0].is_pec
                pec_y = materials[1].is_pec
                pec_z = materials[2].is_pec

        tag = self.kwargs.get("tag")
        if tag is not None and thickness <= 0:
            raise ValueError(f"{self.params_str()} a cell-centred tag requires a volumetric sector")
        tag_data, tag_id = geometry_tag_args(grid, tag)
        occupied = build_cylindrical_sector(
            ctr1,
            ctr2,
            level,
            sectorstartangle,
            sectorangle,
            r,
            normal,
            thickness,
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

        if thickness > 0:
            dielectricsmoothing = "on" if averaging else "off"
            logger.info(
                f"{self.grid_name(grid)}Cylindrical sector with centre "
                f"{ctr1:g}m, {ctr2:g}m, radius {r:g}m, starting angle "
                f"{(sectorstartangle / (2 * np.pi)) * 360:.1f} degrees, "
                f"sector angle {(sectorangle / (2 * np.pi)) * 360:.1f} degrees, "
                f"thickness {thickness:g}m, of material(s) {', '.join(materialsrequested)} "
                f"created, dielectric smoothing is {dielectricsmoothing}."
            )
        else:
            logger.info(
                f"{self.grid_name(grid)}Cylindrical sector with centre "
                f"{ctr1:g}m, {ctr2:g}m, radius {r:g}m, starting angle "
                f"{(sectorstartangle / (2 * np.pi)) * 360:.1f} degrees, "
                f"sector angle {(sectorangle / (2 * np.pi)) * 360:.1f} "
                f"degrees, of material(s) {', '.join(materialsrequested)} "
                f"created."
            )
