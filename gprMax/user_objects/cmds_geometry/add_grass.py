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

from gprMax.fractals.fractal_surface import FractalSurface
from gprMax.fractals.grass import Grass
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import create_grass
from gprMax.user_objects.rotatable import RotatableMixin
from gprMax.user_objects.user_objects import GeometryUserObject
from gprMax.utilities.utilities import round_value

from .cmds_geometry import rotate_2point_object

logger = logging.getLogger(__name__)


class AddGrass(RotatableMixin, GeometryUserObject):
    """Adds grass with roots to a FractalBox class in the model.

    Attributes:
        p1: list of the lower left (x,y,z) coordinates of a surface on a
            FractalBox class.
        p2: list of the upper right (x,y,z) coordinates of a surface on a
            FractalBox class.
        frac_dim: float for the fractal dimension which, for an orthogonal
                    parallelepiped, should take values between zero and three.
        limits: list to define lower and upper limits for a range over which
                    the height of the blades of grass can vary.
        n_blades:int for the number of blades of grass that should be
                    applied to the surface area.
        fractal_box_id: string identifier for the FractalBox class that the
                        grass should be applied to.
    """

    @property
    def hash(self):
        return "#add_grass"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do_rotate(self, grid: FDTDGrid):
        """Perform rotation."""
        pts = np.array([self.kwargs["p1"], self.kwargs["p2"]])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.kwargs["p1"] = tuple(rot_pts[0, :])
        self.kwargs["p2"] = tuple(rot_pts[1, :])

    def build(self, grid: FDTDGrid):
        """Add Grass to fractal box."""
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            fractal_box_id = self.kwargs["fractal_box_id"]
            frac_dim = self.kwargs["frac_dim"]
            limits = self.kwargs["limits"]
            n_blades = self.kwargs["n_blades"]
        except KeyError:
            logger.exception(f"{self.__str__()} requires at least eleven parameters")
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
        if limits[0] < 0 or limits[1] < 0:
            raise ValueError(
                f"{self.__str__()} requires a positive value for the minimum and maximum heights for grass blades"
            )

        # Check for valid orientations
        if np.count_nonzero(discretised_p1 == discretised_p2) != 1:
            raise ValueError(f"{self.__str__()} dimensions are not specified correctly")

        if xs == xf:
            # xminus surface
            if xs == volume.xs:
                raise ValueError(
                    f"{self.__str__()} grass can only be specified on surfaces in the positive axis direction"
                )
            # xplus surface
            elif xf == volume.xf:
                lower_bound = uip.discretise_point((limits[0], 0, 0))
                upper_bound = uip.discretise_point((limits[1], p2[1], p2[2]))
                uip.point_within_bounds(upper_bound, self.__str__())
                fractalrange = (lower_bound[0], upper_bound[0])
                requestedsurface = "xplus"
            else:
                raise ValueError(
                    f"{self.__str__()} must specify external surfaces on a fractal box"
                )

        elif ys == yf:
            # yminus surface
            if ys == volume.ys:
                raise ValueError(
                    f"{self.__str__()} grass can only be specified on surfaces in the positive axis direction"
                )
            # yplus surface
            elif yf == volume.yf:
                lower_bound = uip.discretise_point((0, limits[0], 0))
                upper_bound = uip.discretise_point((p2[0], limits[1], p2[2]))
                uip.point_within_bounds(upper_bound, self.__str__())
                fractalrange = (lower_bound[1], upper_bound[1])
                requestedsurface = "yplus"
            else:
                raise ValueError(
                    f"{self.__str__()} must specify external surfaces on a fractal box"
                )

        elif zs == zf:
            # zminus surface
            if zs == volume.zs:
                raise ValueError(
                    f"{self.__str__()} grass can only be specified on surfaces in the positive axis direction"
                )
            # zplus surface
            elif zf == volume.zf:
                lower_bound = uip.discretise_point((0, 0, limits[0]))
                upper_bound = uip.discretise_point((p2[0], p2[1], limits[1]))
                uip.point_within_bounds(upper_bound, self.__str__())
                fractalrange = (lower_bound[2], upper_bound[2])
                requestedsurface = "zplus"
            else:
                raise ValueError(
                    f"{self.__str__()} must specify external surfaces on a fractal box"
                )
        else:
            raise ValueError(f"{self.__str__()} dimensions are not specified correctly")

        surface = FractalSurface(xs, xf, ys, yf, zs, zf, frac_dim, seed)
        surface.ID = "grass"
        surface.surfaceID = requestedsurface

        # Set the fractal range to scale the fractal distribution between zero and one
        surface.fractalrange = (0, 1)
        surface.operatingonID = volume.ID
        surface.generate_fractal_surface()
        if n_blades > surface.fractalsurface.shape[0] * surface.fractalsurface.shape[1]:
            raise ValueError(
                f"{self.__str__()} the specified surface is not large "
                "enough for the number of grass blades/roots specified"
            )

        # Scale the distribution so that the summation is equal to one,
        # i.e. a probability distribution
        surface.fractalsurface = surface.fractalsurface / np.sum(surface.fractalsurface)

        # Set location of grass blades using probability distribution
        # Create 1D vector of probability values from the 2D surface
        probability1D = np.cumsum(np.ravel(surface.fractalsurface))

        # Create random numbers between zero and one for the number of blades of grass
        R = np.random.RandomState(surface.seed)
        A = R.random_sample(n_blades)

        # Locate the random numbers in the bins created by the 1D vector of
        # probability values, and convert the 1D index back into a x, y index
        # for the original surface.
        bladesindex = np.unravel_index(
            np.digitize(A, probability1D),
            (surface.fractalsurface.shape[0], surface.fractalsurface.shape[1]),
        )

        # Set the fractal range to minimum and maximum heights of the grass blades
        surface.fractalrange = fractalrange

        # Set the fractal surface using the pre-calculated spatial distribution
        # and a random height
        surface.fractalsurface = np.zeros(
            (surface.fractalsurface.shape[0], surface.fractalsurface.shape[1])
        )
        for i in range(len(bladesindex[0])):
            surface.fractalsurface[bladesindex[0][i], bladesindex[1][i]] = R.randint(
                surface.fractalrange[0], surface.fractalrange[1], size=1
            )

        # Create grass geometry parameters
        g = Grass(n_blades, surface.seed)
        surface.grass.append(g)

        # Check to see if grass has been already defined as a material
        if not any(x.ID == "grass" for x in grid.materials):
            create_grass(grid)

        # Check if time step for model is suitable for using grass
        grass = next((x for x in grid.materials if x.ID == "grass"))
        testgrass = next((x for x in grass.tau if x < grid.dt), None)
        if testgrass:
            raise ValueError(
                f"{self.__str__()} requires the time step for the "
                "model to be less than the relaxation time required to model grass."
            )

        volume.fractalsurfaces.append(surface)

        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        logger.info(
            f"{self.grid_name(grid)}{n_blades} blades of grass on surface from "
            f"{p3[0]:g}m, {p3[1]:g}m, {p3[2]:g}m, "
            f"to {p4[0]:g}m, {p4[1]:g}m, {p4[2]:g}m "
            f"with fractal dimension {surface.dimension:g}, fractal seeding "
            f"{surface.seed}, and range {limits[0]:g}m to {limits[1]:g}m, "
            f"added to {surface.operatingonID}."
        )
