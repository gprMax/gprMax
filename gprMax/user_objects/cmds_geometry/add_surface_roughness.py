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

import numpy as np

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.user_objects.rotatable import RotatableMixin
from gprMax.user_objects.user_objects import GeometryUserObject

from .cmds_geometry import rotate_2point_object

logger = logging.getLogger(__name__)


class AddSurfaceRoughness(RotatableMixin, GeometryUserObject):
    """Adds surface roughness to a FractalBox class in the model.

    Attributes:
        p1: list of the lower left (x,y,z) coordinates of a surface on a
            FractalBox class.
        p2: list of the upper right (x,y,z) coordinates of a surface on
            a FractalBox class.
        frac_dim: float for the fractal dimension which, for an
            orthogonal parallelepiped, should take values between zero
            and three.
        weighting: list with weightings in the first and second
            direction of the surface.
        limits: list to define lower and upper limits for a range over
            which the surface roughness can vary.
        fractal_box_id: string identifier for the FractalBox class that
            the surface roughness should be applied to.
        seed: (optional) float parameter which controls the seeding of
            the random number generator used to create the fractals.
    """

    @property
    def hash(self):
        return "#add_surface_roughness"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do_rotate(self, grid: FDTDGrid):
        """Perform rotation."""
        pts = np.array([self.kwargs["p1"], self.kwargs["p2"]])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.kwargs["p1"] = tuple(rot_pts[0, :])
        self.kwargs["p2"] = tuple(rot_pts[1, :])

    def build(self, grid: FDTDGrid):
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            frac_dim = self.kwargs["frac_dim"]
            weighting = np.array(self.kwargs["weighting"], dtype=np.float64)
            limits = np.array(self.kwargs["limits"])
            fractal_box_id = self.kwargs["fractal_box_id"]
        except KeyError:
            logger.exception(f"{self.__str__()} incorrect parameters")
            raise

        try:
            seed = int(self.kwargs["seed"])
        except KeyError:
            logger.warning(
                f"{self.__str__()} no value for seed detected. This "
                "means you will get a different fractal distribution "
                "every time the model runs."
            )
            seed = None

        if self.do_rotate:
            self._do_rotate(grid)

        # Get the correct fractal volume
        volumes = [volume for volume in grid.fractalvolumes if volume.ID == fractal_box_id]
        if volumes:
            volume = volumes[0]
        else:
            raise ValueError(f"{self.__str__()} cannot find FractalBox {fractal_box_id}")

        uip = self._create_uip(grid)
        discretised_p1, discretised_p2 = uip.check_output_object_bounds(p1, p2, self.__str__())
        xs, ys, zs = discretised_p1
        xf, yf, zf = discretised_p2

        if frac_dim < 0:
            raise ValueError(
                f"{self.__str__()} requires a positive value for the fractal dimension"
            )
        if weighting[0] < 0:
            raise ValueError(
                f"{self.__str__()} requires a positive value for the fractal weighting in the first"
                " direction of the surface"
            )
        if weighting[1] < 0:
            raise ValueError(
                f"{self.__str__()} requires a positive value for the fractal weighting in the"
                " second direction of the surface"
            )

        # Check for valid orientations
        if np.count_nonzero(discretised_p1 == discretised_p2) != 1:
            raise ValueError(f"{self.__str__()} dimensions are not specified correctly")

        if xs == xf:
            # xminus surface
            if xs == volume.xs:
                lower_bound = discretised_p1
                upper_bound = uip.discretise_point((limits[1], p2[1], p2[2]))
                grid_bound = uip.discretise_point((limits[0], p1[1], p1[2]))
                fractalrange = (grid_bound[0], upper_bound[0])
                requestedsurface = "xminus"
            # xplus surface
            elif xf == volume.xf:
                lower_bound = uip.discretise_point((limits[0], p1[1], p1[2]))
                upper_bound = discretised_p2
                grid_bound = uip.discretise_point((limits[1], p2[1], p2[2]))
                fractalrange = (lower_bound[0], grid_bound[0])
                requestedsurface = "xplus"
            else:
                raise ValueError(
                    f"{self.__str__()} can only be used on the external surfaces of a fractal box"
                )
        elif ys == yf:
            # yminus surface
            if ys == volume.ys:
                lower_bound = discretised_p1
                upper_bound = uip.discretise_point((p2[0], limits[1], p2[2]))
                grid_bound = uip.discretise_point((p1[0], limits[0], p1[2]))
                fractalrange = (grid_bound[1], upper_bound[1])
                requestedsurface = "yminus"
            # yplus surface
            elif yf == volume.yf:
                lower_bound = uip.discretise_point((p1[0], limits[0], p1[2]))
                upper_bound = discretised_p2
                grid_bound = uip.discretise_point((p2[0], limits[1], p2[2]))
                fractalrange = (lower_bound[1], grid_bound[1])
                requestedsurface = "yplus"
            else:
                raise ValueError(
                    f"{self.__str__()} can only be used on the external surfaces of a fractal box"
                )
        elif zs == zf:
            # zminus surface
            if zs == volume.zs:
                lower_bound = discretised_p1
                upper_bound = uip.discretise_point((p2[0], p2[1], limits[1]))
                grid_bound = uip.discretise_point((p1[0], p1[1], limits[0]))
                fractalrange = (grid_bound[2], upper_bound[2])
                requestedsurface = "zminus"
            # zplus surface
            elif zf == volume.zf:
                lower_bound = uip.discretise_point((p1[0], p1[1], limits[0]))
                upper_bound = discretised_p2
                grid_bound = uip.discretise_point((p2[0], p2[1], limits[1]))
                fractalrange = (lower_bound[2], grid_bound[2])
                requestedsurface = "zplus"
            else:
                raise ValueError(
                    f"{self.__str__()} can only be used on the external surfaces of a fractal box"
                )
        else:
            raise ValueError(f"{self.__str__()} dimensions are not specified correctly")

        if any(lower_bound < volume.start):
            raise ValueError(
                f"{self.__str__()} cannot apply fractal surface to"
                " fractal box as it would exceed the lower coordinates"
                " of the fractal box."
            )
        elif any(upper_bound > volume.stop):
            raise ValueError(
                f"{self.__str__()} cannot apply fractal surface to"
                " fractal box as it would exceed the upper coordinates"
                " of the fractal box."
            )

        # Check lower or upper extent of the fractal surface (depending
        # if the fractal surface has been applied in the minus or plus
        # direction).
        uip.point_within_bounds(grid_bound, f"{self.__str__()}")

        surface = grid.create_fractal_surface(xs, xf, ys, yf, zs, zf, frac_dim, seed)
        surface.surfaceID = requestedsurface
        surface.fractalrange = fractalrange
        surface.operatingonID = volume.ID
        surface.weighting = weighting

        # List of existing surfaces IDs
        existingsurfaceIDs = [x.surfaceID for x in volume.fractalsurfaces]
        if surface.surfaceID in existingsurfaceIDs:
            raise ValueError(
                f"{self.__str__()} has already been used on the {surface.surfaceID} surface"
            )

        surface.generate_fractal_surface()

        volume.fractalsurfaces.append(surface)

        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        logger.info(
            f"{self.grid_name(grid)}Fractal surface from {p3[0]:g}m, "
            f"{p3[1]:g}m, {p3[2]:g}m, to {p4[0]:g}m, "
            f"{p4[1]:g}m, {p4[2]:g}m with fractal dimension "
            f"{surface.dimension:g}, fractal weightings {surface.weighting[0]:g}, "
            f"{surface.weighting[1]:g}, fractal seeding {surface.seed}, "
            f"and range {limits[0]:g}m to {limits[1]:g}m, added to "
            f"{surface.operatingonID}."
        )
