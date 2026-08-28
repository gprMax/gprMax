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
        seed: optional integer used to seed the random number generator. If
            omitted, a different distribution is generated on each run.
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

        # p1/p2 have one flat (normal) axis, where the coordinates must be
        # equal, and two extent axes - same structure as
        # #add_surface_roughness. Resolve the flat axis sign-based
        # (role=None) so both points land on the same value, and the two
        # extent axes positionally (role="lower"/"upper").
        # A literal (non-inf) coordinate match always wins over a "both
        # inf" match - the latter is ambiguous with a non-normal axis that
        # legitimately spans its full range via `inf` on both endpoints
        # (e.g. the model's own invariant axis, resolved positionally like
        # any other extent axis), and only means "flat" when no genuinely
        # flat axis exists elsewhere.
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

        mode = config.get_model_config().mode
        if mode.startswith("2D") and requestedsurface[0] == mode[-1]:
            raise ValueError(
                f"{self.__str__()} cannot be applied to the {requestedsurface} surface in 2D "
                "mode - its normal is the invariant axis, which has no meaningful depth for "
                "grass (the same restriction as #add_surface_roughness in 2D mode)."
            )

        surface = grid.create_fractal_surface(xs, xf, ys, yf, zs, zf, frac_dim, seed)
        surface.ID = "grass"
        surface.surfaceID = requestedsurface

        # Create grass geometry parameters
        # Add grass to the surface here as an MPIFractalSurface needs to
        # know if grass is present when generate_fractal_surface() is
        # called
        g = Grass(n_blades, surface.seed)
        surface.grass.append(g)

        # Set the fractal range to scale the fractal distribution between zero and one
        surface.fractalrange = (0, 1)
        surface.operatingonID = volume.ID
        if not surface.generate_fractal_surface():
            return

        # In 2D TE mode the invariant axis is 2 cells thick, and
        # generate_fractal_surface() has already made the underlying density
        # map identical on both (see FractalSurface). But blade position/
        # height sampling below is a *separate* random draw over the full
        # flattened array - sampling it directly would place different
        # blades on each of the two cells and would not reproduce what an
        # equivalent TM-mode (1-cell) grass surface with the same seed would
        # produce (the flattened probability vector, and hence every
        # downstream digitize() result, has a different length/order for a
        # 2-cell vs 1-cell array). So sample on a single reduced-thickness
        # slice, matching TM's own shape/computation exactly. The result is
        # placed at only one of the two cells below (not broadcast) - see
        # that comment for why, and how the other cell still ends up
        # correctly invariant.
        te_axis = None
        mode = config.get_model_config().mode
        if mode.startswith("2D TE"):
            invariant_axis = "xyz".index(mode[-1])
            if surface.size[invariant_axis] == 2:
                te_axis = surface._te_invariant_inplane_index(invariant_axis)

        density = surface.fractalsurface
        if te_axis == 0:
            density = density[:1, :]
        elif te_axis == 1:
            density = density[:, :1]

        if n_blades > density.shape[0] * density.shape[1]:
            raise ValueError(
                f"{self.__str__()} the specified surface is not large "
                "enough for the number of grass blades/roots specified"
            )

        # Scale the distribution so that the summation is equal to one,
        # i.e. a probability distribution
        density = density / np.sum(density)

        # Set location of grass blades using probability distribution
        # Create 1D vector of probability values from the 2D surface
        probability1D = np.cumsum(np.ravel(density))

        # Create random numbers between zero and one for the number of blades of grass
        R = np.random.RandomState(surface.seed)
        A = R.random_sample(n_blades)

        # Locate the random numbers in the bins created by the 1D vector of
        # probability values, and convert the 1D index back into a x, y index
        # for the original surface.
        bladesindex = np.unravel_index(
            np.digitize(A, probability1D),
            (density.shape[0], density.shape[1]),
        )

        # Set the fractal range to minimum and maximum heights of the grass blades
        surface.fractalrange = fractalrange

        # Set the fractal surface using the pre-calculated spatial distribution
        # and a random height
        heights = np.zeros((density.shape[0], density.shape[1]))
        for i in range(len(bladesindex[0])):
            heights[bladesindex[0][i], bladesindex[1][i]] = R.randint(
                surface.fractalrange[0], surface.fractalrange[1]
            )

        if te_axis is not None:
            # Deliberately do NOT broadcast heights to both invariant-axis
            # cells here (unlike the plain density map above). The
            # FractalBox.build() blade/root loop below increments a
            # sequential index into Grass's fixed-size geometryparams array
            # once per "blade found" grid point it visits - if both cells
            # showed height>0 for the same row, that loop would visit (and
            # allocate a geometry-parameter row for) the same logical blade
            # twice, overflowing geometryparams (sized for exactly n_blades)
            # and also giving the two cells different wobble geometry from
            # different array rows. Instead, leave the duplicate cell at
            # zero here so the loop only ever builds each blade once; the
            # post-hoc mask broadcast in FractalBox.build() then copies the
            # actual built voxels to the duplicate cell, achieving
            # invariance without touching that loop at all.
            full_shape = list(heights.shape)
            full_shape[te_axis] = 2
            full = np.zeros(tuple(full_shape))
            indexer = [slice(None), slice(None)]
            indexer[te_axis] = slice(0, 1)
            full[tuple(indexer)] = heights
            surface.fractalsurface = full
        else:
            surface.fractalsurface = heights

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
