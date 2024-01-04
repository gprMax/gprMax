# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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

from ..materials import create_water
from ..utilities.utilities import round_value
from .cmds_geometry import UserObjectGeometry, rotate_2point_object

logger = logging.getLogger(__name__)


class AddSurfaceWater(UserObjectGeometry):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash = "#add_surface_water"

    def rotate(self, axis, angle, origin=None):
        """Set parameters for rotation."""
        self.axis = axis
        self.angle = angle
        self.origin = origin
        self.do_rotate = True

    def _do_rotate(self):
        """Perform rotation."""
        pts = np.array([self.kwargs["p1"], self.kwargs["p2"]])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.kwargs["p1"] = tuple(rot_pts[0, :])
        self.kwargs["p2"] = tuple(rot_pts[1, :])

    def build(self, grid, uip):
        """ "Create surface water on fractal box."""
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            fractal_box_id = self.kwargs["fractal_box_id"]
            depth = self.kwargs["depth"]
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly eight parameters")
            raise

        if self.do_rotate:
            self._do_rotate()

        if volumes := [volume for volume in grid.fractalvolumes if volume.ID == fractal_box_id]:
            volume = volumes[0]
        else:
            logger.exception(f"{self.__str__()} cannot find FractalBox {fractal_box_id}")
            raise ValueError

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        if depth <= 0:
            logger.exception(f"{self.__str__()} requires a positive value for the depth of water")
            raise ValueError

        # Check for valid orientations
        if xs == xf:
            if ys == yf or zs == zf:
                logger.exception(f"{self.__str__()} dimensions are not specified correctly")
                raise ValueError
            if xs not in [volume.xs, volume.xf]:
                logger.exception(f"{self.__str__()} can only be used on the external surfaces of a fractal box")
                raise ValueError
            # xminus surface
            if xs == volume.xs:
                requestedsurface = "xminus"
            # xplus surface
            elif xf == volume.xf:
                requestedsurface = "xplus"
            filldepthcells = round_value(depth / grid.dx)
            filldepth = filldepthcells * grid.dx

        elif ys == yf:
            if zs == zf:
                logger.exception(f"{self.__str__()} dimensions are not specified correctly")
                raise ValueError
            if ys not in [volume.ys, volume.yf]:
                logger.exception(f"{self.__str__()} can only be used on the external surfaces of a fractal box")
                raise ValueError
            # yminus surface
            if ys == volume.ys:
                requestedsurface = "yminus"
            # yplus surface
            elif yf == volume.yf:
                requestedsurface = "yplus"
            filldepthcells = round_value(depth / grid.dy)
            filldepth = filldepthcells * grid.dy

        elif zs == zf:
            if zs not in [volume.zs, volume.zf]:
                logger.exception(f"{self.__str__()} can only be used on the external surfaces of a fractal box")
                raise ValueError
            # zminus surface
            if zs == volume.zs:
                requestedsurface = "zminus"
            # zplus surface
            elif zf == volume.zf:
                requestedsurface = "zplus"
            filldepthcells = round_value(depth / grid.dz)
            filldepth = filldepthcells * grid.dz

        else:
            logger.exception(f"{self.__str__()} dimensions are not specified correctly")
            raise ValueError

        surface = next((x for x in volume.fractalsurfaces if x.surfaceID == requestedsurface), None)
        if not surface:
            logger.exception(
                f"{self.__str__()} specified surface {requestedsurface} does not have a rough surface applied"
            )
            raise ValueError

        surface.filldepth = filldepthcells

        # Check that requested fill depth falls within range of surface roughness
        if surface.filldepth < surface.fractalrange[0] or surface.filldepth > surface.fractalrange[1]:
            logger.exception(
                f"{self.__str__()} requires a value for the depth of water that lies with the "
                f"range of the requested surface roughness"
            )
            raise ValueError

        # Check to see if water has been already defined as a material
        if not any(x.ID == "water" for x in grid.materials):
            create_water(grid)

        # Check if time step for model is suitable for using water
        water = next((x for x in grid.materials if x.ID == "water"))
        if testwater := next((x for x in water.tau if x < grid.dt), None):
            logger.exception(
                f"{self.__str__()} requires the time step for the model "
                f"to be less than the relaxation time required to model water."
            )
            raise ValueError

        logger.info(
            f"{self.grid_name(grid)}Water on surface from {xs * grid.dx:g}m, "
            f"{ys * grid.dy:g}m, {zs * grid.dz:g}m, to {xf * grid.dx:g}m, "
            f"{yf * grid.dy:g}m, {zf * grid.dz:g}m with depth {filldepth:g}m, "
            f"added to {surface.operatingonID}."
        )
