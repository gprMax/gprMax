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

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import create_water
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


class AddSurfaceWater(GeometryUserObject):
    """Adds surface water to a FractalBox class in the model.

    Attributes:
        p1: list of the lower left (x,y,z) coordinates of a surface on a
            FractalBox class.
        p2: list of the upper right (x,y,z) coordinates of a surface on a
            FractalBox class.
        depth: float that defines the depth of the water, which should be
                specified relative to the dimensions of the #fractal_box that
                the surface water is being applied.
        fractal_box_id: string identifier for the FractalBox class that the
                        surface water should be applied to.
    """

    @property
    def hash(self):
        return "#add_surface_water"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        """ "Create surface water on fractal box."""
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            fractal_box_id = self.kwargs["fractal_box_id"]
            depth = self.kwargs["depth"]
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly eight parameters")
            raise

        if volumes := [volume for volume in grid.fractalvolumes if volume.ID == fractal_box_id]:
            volume = volumes[0]
        else:
            raise ValueError(f"{self.__str__()} cannot find FractalBox {fractal_box_id}")

        uip = self._create_uip(grid)

        # p1/p2 have one flat (normal) axis, where the coordinates must be
        # equal, and two extent axes - same structure as
        # #add_surface_roughness/#add_grass. Resolve the flat axis sign-based
        # (role=None) so both points land on the same value, and the two
        # extent axes positionally (role="lower"/"upper"). No separate
        # Case-A (normal==invariant axis) guard is needed here: this
        # command requires an existing FractalSurface on the same face,
        # which #add_surface_roughness's own Case-A guard already prevents
        # from ever being created there.
        p1_arr = np.asarray(p1, dtype=np.float64)
        p2_arr = np.asarray(p2, dtype=np.float64)
        flat_axis = next(
            (
                axis
                for axis in range(3)
                if p1_arr[axis] == p2_arr[axis]
                and not (np.isinf(p1_arr[axis]) and np.isinf(p2_arr[axis]))
            ),
            None,
        )
        if flat_axis is None:
            flat_axis = next(
                (axis for axis in range(3) if np.isinf(p1_arr[axis]) and np.isinf(p2_arr[axis])),
                None,
            )
        p1_ranged = uip.resolve_inf_point(p1, role="lower")
        p2_ranged = uip.resolve_inf_point(p2, role="upper")
        if flat_axis is not None and (
            np.isinf(p1_arr[flat_axis]) or np.isinf(p2_arr[flat_axis])
        ):
            p1_single = uip.resolve_inf_point(p1, role=None)
            p2_single = uip.resolve_inf_point(p2, role=None)
            p1 = tuple(p1_single[a] if a == flat_axis else p1_ranged[a] for a in range(3))
            p2 = tuple(p2_single[a] if a == flat_axis else p2_ranged[a] for a in range(3))
        else:
            p1 = p1_ranged
            p2 = p2_ranged

        discretised_p1, discretised_p2 = uip.check_output_object_bounds(p1, p2, self.__str__())
        xs, ys, zs = discretised_p1
        xf, yf, zf = discretised_p2

        if depth <= 0:
            raise ValueError(f"{self.__str__()} requires a positive value for the depth of water")

        # Check for valid orientations
        if np.count_nonzero(discretised_p1 == discretised_p2) != 1:
            raise ValueError(f"{self.__str__()} dimensions are not specified correctly")

        if xs == xf:
            # xminus surface
            if xs == volume.xs:
                requestedsurface = "xminus"
            # xplus surface
            elif xf == volume.xf:
                requestedsurface = "xplus"
            else:
                raise ValueError(
                    f"{self.__str__()} can only be used on the external surfaces of a fractal box"
                )
            filldepthcells = uip.discretise_point((depth, 0, 0))[0]
            filldepth = uip.round_to_grid_static_point((depth, 0, 0))[0]

        elif ys == yf:
            # yminus surface
            if ys == volume.ys:
                requestedsurface = "yminus"
            # yplus surface
            elif yf == volume.yf:
                requestedsurface = "yplus"
            else:
                raise ValueError(
                    f"{self.__str__()} can only be used on the external surfaces of a fractal box"
                )
            filldepthcells = uip.discretise_point((0, depth, 0))[1]
            filldepth = uip.round_to_grid_static_point((0, depth, 0))[1]

        elif zs == zf:
            # zminus surface
            if zs == volume.zs:
                requestedsurface = "zminus"
            # zplus surface
            elif zf == volume.zf:
                requestedsurface = "zplus"
            else:
                raise ValueError(
                    f"{self.__str__()} can only be used on the external surfaces of a fractal box"
                )
            filldepthcells = uip.discretise_point((0, 0, depth))[2]
            filldepth = uip.round_to_grid_static_point((0, 0, depth))[2]

        else:
            raise ValueError(f"{self.__str__()} dimensions are not specified correctly")

        surface = next((x for x in volume.fractalsurfaces if x.surfaceID == requestedsurface), None)
        if not surface:
            raise ValueError(
                f"{self.__str__()} specified surface {requestedsurface} does not have a rough surface applied"
            )

        surface.filldepth = filldepthcells

        # Check that requested fill depth falls within range of surface roughness
        if (
            surface.filldepth < surface.fractalrange[0]
            or surface.filldepth > surface.fractalrange[1]
        ):
            raise ValueError(
                f"{self.__str__()} requires a value for the depth of water that lies with the "
                f"range of the requested surface roughness"
            )

        # Check to see if water has been already defined as a material
        if not any(x.ID == "water" for x in grid.materials):
            create_water(grid)

        # Check if time step for model is suitable for using water
        water = next((x for x in grid.materials if x.ID == "water"))
        if testwater := next((x for x in water.tau if x < grid.dt), None):
            raise ValueError(
                f"{self.__str__()} requires the time step for the model "
                f"to be less than the relaxation time required to model water."
            )

        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        logger.info(
            f"{self.grid_name(grid)}Water on surface from {p3[0]:g}m,"
            f" {p3[1]:g}m, {p3[2]:g}m, to {p4[0]:g}m, {p4[1]:g}m,"
            f" {p4[2]:g}m with depth {filldepth:g}m, added to"
            f" {surface.operatingonID}."
        )
