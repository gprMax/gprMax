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

import gprMax.config as config
from gprMax.cython.geometry_primitives import build_voxels_from_array, build_voxels_from_array_mask
from gprMax.fractals import FractalVolume
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import ListMaterial
from gprMax.user_objects.cmds_geometry.cmds_geometry import rotate_2point_object
from gprMax.user_objects.rotatable import Rotatable
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


class FractalBox(GeometryUserObject, Rotatable):
    """Introduces an orthogonal parallelepiped with fractal distributed
        properties which are related to a mixing model or normal material into
        the model.

    Attributes:
        p1: list of the lower left (x,y,z) coordinates of the parallelepiped.
        p2: list of the upper right (x,y,z) coordinates of the parallelepiped.
        frac_dim: float for the fractal dimension which, for an orthogonal
                    parallelepiped, should take values between zero and three.
        weighting: list of the weightings in the x, y, z direction of the
                    parallelepiped.
        n_materials: int of the number of materials to use for the fractal
                        distribution (defined according to the associated
                        mixing model). This should be set to one if using a
                        normal material instead of a mixing model.
        mixing_model_id: string identifier for the associated mixing model or
                            material.
        id: string identifier for the fractal box itself.
        seed: (optional) float parameter which controls the seeding of the
                random number generator used to create the fractals.
        averaging: string (y or n) used to switch on and off dielectric smoothing.
    """

    @property
    def hash(self):
        return "#fractal_box"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_pre_build = True

    def _do_rotate(self, grid: FDTDGrid):
        """Performs rotation."""
        pts = np.array([self.kwargs["p1"], self.kwargs["p2"]])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.kwargs["p1"] = tuple(rot_pts[0, :])
        self.kwargs["p2"] = tuple(rot_pts[1, :])

    def pre_build(self, grid: FDTDGrid):
        try:
            p1 = self.kwargs["p1"]
            p2 = self.kwargs["p2"]
            frac_dim = self.kwargs["frac_dim"]
            weighting = np.array(self.kwargs["weighting"])
            n_materials = self.kwargs["n_materials"]
            mixing_model_id = self.kwargs["mixing_model_id"]
            ID = self.kwargs["id"]
        except KeyError:
            logger.exception(f"{self.__str__()} Incorrect parameters")
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

        # Check averaging
        try:
            # Go with user specified averaging
            averagefractalbox = self.kwargs["averaging"]
        except KeyError:
            # If they havent specified - default is no dielectric smoothing for
            # a fractal box.
            averagefractalbox = False

        uip = self._create_uip(grid)
        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        if frac_dim < 0:
            logger.exception(
                f"{self.__str__()} requires a positive value for the fractal dimension"
            )
            raise ValueError
        if weighting[0] < 0:
            logger.exception(
                f"{self.__str__()} requires a positive value for the fractal weighting in the x direction"
            )
            raise ValueError
        if weighting[1] < 0:
            logger.exception(
                f"{self.__str__()} requires a positive value for the fractal weighting in the y direction"
            )
            raise ValueError
        if weighting[2] < 0:
            logger.exception(
                f"{self.__str__()} requires a positive value for the fractal weighting in the z direction"
            )
        if n_materials < 0:
            logger.exception(f"{self.__str__()} requires a positive value for the number of bins")
            raise ValueError

        # Find materials to use to build fractal volume, either from mixing
        # models or normal materials.
        mixingmodel = next((x for x in grid.mixingmodels if x.ID == mixing_model_id), None)
        material = next((x for x in grid.materials if x.ID == mixing_model_id), None)
        nbins = n_materials

        if mixingmodel:
            if nbins == 1:
                logger.exception(
                    f"{self.__str__()} must be used with more than one material from the mixing model."
                )
                raise ValueError
            if isinstance(mixingmodel, ListMaterial) and nbins > len(mixingmodel.mat):
                logger.exception(
                    f"{self.__str__()} too many materials/bins "
                    "requested compared to materials in "
                    "mixing model."
                )
                raise ValueError
            # Create materials from mixing model as number of bins now known
            # from fractal_box command.
            mixingmodel.calculate_properties(nbins, grid)
        elif not material:
            logger.exception(
                f"{self.__str__()} mixing model or material with "
                + "ID {mixing_model_id} does not exist"
            )
            raise ValueError

        self.volume = FractalVolume(xs, xf, ys, yf, zs, zf, frac_dim, seed)
        self.volume.ID = ID
        self.volume.operatingonID = mixing_model_id
        self.volume.nbins = nbins
        self.volume.weighting = weighting
        self.volume.averaging = averagefractalbox
        self.volume.mixingmodel = mixingmodel

        dielectricsmoothing = "on" if self.volume.averaging else "off"
        logger.info(
            f"{self.grid_name(grid)}Fractal box {self.volume.ID} from "
            f"{p3[0]:g}m, {p3[1]:g}m, {p3[2]:g}m, to {p4[0]:g}m, "
            f"{p4[1]:g}m, {p4[2]:g}m with {self.volume.operatingonID}, "
            f"fractal dimension {self.volume.dimension:g}, fractal weightings "
            f"{self.volume.weighting[0]:g}, {self.volume.weighting[1]:g}, "
            f"{self.volume.weighting[2]:g}, fractal seeding {self.volume.seed}, "
            f"with {self.volume.nbins} material(s) created, dielectric smoothing "
            f"is {dielectricsmoothing}."
        )
        grid.fractalvolumes.append(self.volume)

    def build(self, grid: FDTDGrid):
        if self.do_pre_build:
            self.pre_build(grid)
            self.do_pre_build = False
        else:
            if self.volume.fractalsurfaces:
                self.volume.originalxs = self.volume.xs
                self.volume.originalxf = self.volume.xf
                self.volume.originalys = self.volume.ys
                self.volume.originalyf = self.volume.yf
                self.volume.originalzs = self.volume.zs
                self.volume.originalzf = self.volume.zf

                # Extend the volume to accomodate any rough surfaces, grass,
                # or roots
                for surface in self.volume.fractalsurfaces:
                    if surface.surfaceID == "xminus":
                        if surface.fractalrange[0] < self.volume.xs:
                            self.volume.nx += self.volume.xs - surface.fractalrange[0]
                            self.volume.xs = surface.fractalrange[0]
                    elif surface.surfaceID == "xplus":
                        if surface.fractalrange[1] > self.volume.xf:
                            self.volume.nx += surface.fractalrange[1] - self.volume.xf
                            self.volume.xf = surface.fractalrange[1]
                    elif surface.surfaceID == "yminus":
                        if surface.fractalrange[0] < self.volume.ys:
                            self.volume.ny += self.volume.ys - surface.fractalrange[0]
                            self.volume.ys = surface.fractalrange[0]
                    elif surface.surfaceID == "yplus":
                        if surface.fractalrange[1] > self.volume.yf:
                            self.volume.ny += surface.fractalrange[1] - self.volume.yf
                            self.volume.yf = surface.fractalrange[1]
                    elif surface.surfaceID == "zminus":
                        if surface.fractalrange[0] < self.volume.zs:
                            self.volume.nz += self.volume.zs - surface.fractalrange[0]
                            self.volume.zs = surface.fractalrange[0]
                    elif surface.surfaceID == "zplus":
                        if surface.fractalrange[1] > self.volume.zf:
                            self.volume.nz += surface.fractalrange[1] - self.volume.zf
                            self.volume.zf = surface.fractalrange[1]

                # If there is only 1 bin then a normal material is being used,
                # otherwise a mixing model
                if self.volume.nbins == 1:
                    self.volume.fractalvolume = np.ones(
                        (self.volume.nx, self.volume.ny, self.volume.nz),
                        dtype=config.sim_config.dtypes["float_or_double"],
                    )
                    materialnumID = next(
                        x.numID for x in grid.materials if x.ID == self.volume.operatingonID
                    )
                    self.volume.fractalvolume *= materialnumID
                else:
                    self.volume.generate_fractal_volume()
                    for i in range(0, self.volume.nx):
                        for j in range(0, self.volume.ny):
                            for k in range(0, self.volume.nz):
                                numberinbin = self.volume.fractalvolume[i, j, k]
                                self.volume.fractalvolume[i, j, k] = self.volume.mixingmodel.matID[
                                    int(numberinbin)
                                ]

                self.volume.generate_volume_mask()

                # Apply any rough surfaces and add any surface water to the
                # 3D mask array
                # TODO: Allow extract of rough surface profile (to print/file?)
                for surface in self.volume.fractalsurfaces:
                    if surface.surfaceID == "xminus":
                        for i in range(surface.fractalrange[0], surface.fractalrange[1]):
                            for j in range(surface.ys, surface.yf):
                                for k in range(surface.zs, surface.zf):
                                    if i > surface.fractalsurface[j - surface.ys, k - surface.zs]:
                                        self.volume.mask[
                                            i - self.volume.xs,
                                            j - self.volume.ys,
                                            k - self.volume.zs,
                                        ] = 1
                                    elif surface.filldepth > 0 and i > surface.filldepth:
                                        self.volume.mask[
                                            i - self.volume.xs,
                                            j - self.volume.ys,
                                            k - self.volume.zs,
                                        ] = 2
                                    else:
                                        self.volume.mask[
                                            i - self.volume.xs,
                                            j - self.volume.ys,
                                            k - self.volume.zs,
                                        ] = 0

                    elif surface.surfaceID == "xplus":
                        if not surface.ID:
                            for i in range(surface.fractalrange[0], surface.fractalrange[1]):
                                for j in range(surface.ys, surface.yf):
                                    for k in range(surface.zs, surface.zf):
                                        if (
                                            i
                                            < surface.fractalsurface[j - surface.ys, k - surface.zs]
                                        ):
                                            self.volume.mask[
                                                i - self.volume.xs,
                                                j - self.volume.ys,
                                                k - self.volume.zs,
                                            ] = 1
                                        elif surface.filldepth > 0 and i < surface.filldepth:
                                            self.volume.mask[
                                                i - self.volume.xs,
                                                j - self.volume.ys,
                                                k - self.volume.zs,
                                            ] = 2
                                        else:
                                            self.volume.mask[
                                                i - self.volume.xs,
                                                j - self.volume.ys,
                                                k - self.volume.zs,
                                            ] = 0
                        elif surface.ID == "grass":
                            g = surface.grass[0]
                            # Build the blades of the grass
                            blade = 0
                            for j in range(surface.ys, surface.yf):
                                for k in range(surface.zs, surface.zf):
                                    if surface.fractalsurface[j - surface.ys, k - surface.zs] > 0:
                                        height = 0
                                        for i in range(self.volume.xs, surface.fractalrange[1]):
                                            if (
                                                i
                                                < surface.fractalsurface[
                                                    j - surface.ys, k - surface.zs
                                                ]
                                                and self.volume.mask[
                                                    i - self.volume.xs,
                                                    j - self.volume.ys,
                                                    k - self.volume.zs,
                                                ]
                                                != 1
                                            ):
                                                y, z = g.calculate_blade_geometry(blade, height)
                                                # Add y, z coordinates to existing location
                                                yy = int(j - self.volume.ys + y)
                                                zz = int(k - self.volume.zs + z)
                                                # If these coordinates are outwith fractal volume stop building the blade,
                                                # otherwise set the mask for grass.
                                                if (
                                                    yy < 0
                                                    or yy >= self.volume.mask.shape[1]
                                                    or zz < 0
                                                    or zz >= self.volume.mask.shape[2]
                                                ):
                                                    break
                                                else:
                                                    self.volume.mask[i - self.volume.xs, yy, zz] = 3
                                                    height += 1
                                        blade += 1

                            # Build the roots of the grass
                            root = 0
                            for j in range(surface.ys, surface.yf):
                                for k in range(surface.zs, surface.zf):
                                    if surface.fractalsurface[j - surface.ys, k - surface.zs] > 0:
                                        depth = 0
                                        i = self.volume.xf - 1
                                        while i > self.volume.xs:
                                            if (
                                                i
                                                > self.volume.originalxf
                                                - (
                                                    surface.fractalsurface[
                                                        j - surface.ys, k - surface.zs
                                                    ]
                                                    - self.volume.originalxf
                                                )
                                                and self.volume.mask[
                                                    i - self.volume.xs,
                                                    j - self.volume.ys,
                                                    k - self.volume.zs,
                                                ]
                                                == 1
                                            ):
                                                y, z = g.calculate_root_geometry(root, depth)
                                                # Add y, z coordinates to existing location
                                                yy = int(j - self.volume.ys + y)
                                                zz = int(k - self.volume.zs + z)
                                                # If these coordinates are outwith the fractal volume stop building the root,
                                                # otherwise set the mask for grass.
                                                if (
                                                    yy < 0
                                                    or yy >= self.volume.mask.shape[1]
                                                    or zz < 0
                                                    or zz >= self.volume.mask.shape[2]
                                                ):
                                                    break
                                                else:
                                                    self.volume.mask[i - self.volume.xs, yy, zz] = 3
                                                    depth += 1
                                            i -= 1
                                        root += 1

                    elif surface.surfaceID == "yminus":
                        for i in range(surface.xs, surface.xf):
                            for j in range(surface.fractalrange[0], surface.fractalrange[1]):
                                for k in range(surface.zs, surface.zf):
                                    if j > surface.fractalsurface[i - surface.xs, k - surface.zs]:
                                        self.volume.mask[
                                            i - self.volume.xs,
                                            j - self.volume.ys,
                                            k - self.volume.zs,
                                        ] = 1
                                    elif surface.filldepth > 0 and j > surface.filldepth:
                                        self.volume.mask[
                                            i - self.volume.xs,
                                            j - self.volume.ys,
                                            k - self.volume.zs,
                                        ] = 2
                                    else:
                                        self.volume.mask[
                                            i - self.volume.xs,
                                            j - self.volume.ys,
                                            k - self.volume.zs,
                                        ] = 0

                    elif surface.surfaceID == "yplus":
                        if not surface.ID:
                            for i in range(surface.xs, surface.xf):
                                for j in range(surface.fractalrange[0], surface.fractalrange[1]):
                                    for k in range(surface.zs, surface.zf):
                                        if (
                                            j
                                            < surface.fractalsurface[i - surface.xs, k - surface.zs]
                                        ):
                                            self.volume.mask[
                                                i - self.volume.xs,
                                                j - self.volume.ys,
                                                k - self.volume.zs,
                                            ] = 1
                                        elif surface.filldepth > 0 and j < surface.filldepth:
                                            self.volume.mask[
                                                i - self.volume.xs,
                                                j - self.volume.ys,
                                                k - self.volume.zs,
                                            ] = 2
                                        else:
                                            self.volume.mask[
                                                i - self.volume.xs,
                                                j - self.volume.ys,
                                                k - self.volume.zs,
                                            ] = 0
                        elif surface.ID == "grass":
                            g = surface.grass[0]
                            # Build the blades of the grass
                            blade = 0
                            for i in range(surface.xs, surface.xf):
                                for k in range(surface.zs, surface.zf):
                                    if surface.fractalsurface[i - surface.xs, k - surface.zs] > 0:
                                        height = 0
                                        for j in range(self.volume.ys, surface.fractalrange[1]):
                                            if (
                                                j
                                                < surface.fractalsurface[
                                                    i - surface.xs, k - surface.zs
                                                ]
                                                and self.volume.mask[
                                                    i - self.volume.xs,
                                                    j - self.volume.ys,
                                                    k - self.volume.zs,
                                                ]
                                                != 1
                                            ):
                                                x, z = g.calculate_blade_geometry(blade, height)
                                                # Add x, z coordinates to existing location
                                                xx = int(i - self.volume.xs + x)
                                                zz = int(k - self.volume.zs + z)
                                                # If these coordinates are outwith fractal volume stop building the blade,
                                                # otherwise set the mask for grass.
                                                if (
                                                    xx < 0
                                                    or xx >= self.volume.mask.shape[0]
                                                    or zz < 0
                                                    or zz >= self.volume.mask.shape[2]
                                                ):
                                                    break
                                                else:
                                                    self.volume.mask[xx, j - self.volume.ys, zz] = 3
                                                    height += 1
                                        blade += 1

                            # Build the roots of the grass
                            root = 0
                            for i in range(surface.xs, surface.xf):
                                for k in range(surface.zs, surface.zf):
                                    if surface.fractalsurface[i - surface.xs, k - surface.zs] > 0:
                                        depth = 0
                                        j = self.volume.yf - 1
                                        while j > self.volume.ys:
                                            if (
                                                j
                                                > self.volume.originalyf
                                                - (
                                                    surface.fractalsurface[
                                                        i - surface.xs, k - surface.zs
                                                    ]
                                                    - self.volume.originalyf
                                                )
                                                and self.volume.mask[
                                                    i - self.volume.xs,
                                                    j - self.volume.ys,
                                                    k - self.volume.zs,
                                                ]
                                                == 1
                                            ):
                                                x, z = g.calculate_root_geometry(root, depth)
                                                # Add x, z coordinates to existing location
                                                xx = int(i - self.volume.xs + x)
                                                zz = int(k - self.volume.zs + z)
                                                # If these coordinates are outwith the fractal volume stop building the root,
                                                # otherwise set the mask for grass.
                                                if (
                                                    xx < 0
                                                    or xx >= self.volume.mask.shape[0]
                                                    or zz < 0
                                                    or zz >= self.volume.mask.shape[2]
                                                ):
                                                    break
                                                else:
                                                    self.volume.mask[xx, j - self.volume.ys, zz] = 3
                                                    depth += 1
                                            j -= 1
                                        root += 1

                    elif surface.surfaceID == "zminus":
                        for i in range(surface.xs, surface.xf):
                            for j in range(surface.ys, surface.yf):
                                for k in range(surface.fractalrange[0], surface.fractalrange[1]):
                                    if k > surface.fractalsurface[i - surface.xs, j - surface.ys]:
                                        self.volume.mask[
                                            i - self.volume.xs,
                                            j - self.volume.ys,
                                            k - self.volume.zs,
                                        ] = 1
                                    elif surface.filldepth > 0 and k > surface.filldepth:
                                        self.volume.mask[
                                            i - self.volume.xs,
                                            j - self.volume.ys,
                                            k - self.volume.zs,
                                        ] = 2
                                    else:
                                        self.volume.mask[
                                            i - self.volume.xs,
                                            j - self.volume.ys,
                                            k - self.volume.zs,
                                        ] = 0

                    elif surface.surfaceID == "zplus":
                        if not surface.ID:
                            for i in range(surface.xs, surface.xf):
                                for j in range(surface.ys, surface.yf):
                                    for k in range(
                                        surface.fractalrange[0], surface.fractalrange[1]
                                    ):
                                        if (
                                            k
                                            < surface.fractalsurface[i - surface.xs, j - surface.ys]
                                        ):
                                            self.volume.mask[
                                                i - self.volume.xs,
                                                j - self.volume.ys,
                                                k - self.volume.zs,
                                            ] = 1
                                        elif surface.filldepth > 0 and k < surface.filldepth:
                                            self.volume.mask[
                                                i - self.volume.xs,
                                                j - self.volume.ys,
                                                k - self.volume.zs,
                                            ] = 2
                                        else:
                                            self.volume.mask[
                                                i - self.volume.xs,
                                                j - self.volume.ys,
                                                k - self.volume.zs,
                                            ] = 0
                        elif surface.ID == "grass":
                            g = surface.grass[0]
                            # Build the blades of the grass
                            blade = 0
                            for i in range(surface.xs, surface.xf):
                                for j in range(surface.ys, surface.yf):
                                    if surface.fractalsurface[i - surface.xs, j - surface.ys] > 0:
                                        height = 0
                                        for k in range(self.volume.zs, surface.fractalrange[1]):
                                            if (
                                                k
                                                < surface.fractalsurface[
                                                    i - surface.xs, j - surface.ys
                                                ]
                                                and self.volume.mask[
                                                    i - self.volume.xs,
                                                    j - self.volume.ys,
                                                    k - self.volume.zs,
                                                ]
                                                != 1
                                            ):
                                                x, y = g.calculate_blade_geometry(blade, height)
                                                # Add x, y coordinates to existing location
                                                xx = int(i - self.volume.xs + x)
                                                yy = int(j - self.volume.ys + y)
                                                # If these coordinates are outwith the fractal volume stop building the blade,
                                                # otherwise set the mask for grass.
                                                if (
                                                    xx < 0
                                                    or xx >= self.volume.mask.shape[0]
                                                    or yy < 0
                                                    or yy >= self.volume.mask.shape[1]
                                                ):
                                                    break
                                                else:
                                                    self.volume.mask[xx, yy, k - self.volume.zs] = 3
                                                    height += 1
                                        blade += 1

                            # Build the roots of the grass
                            root = 0
                            for i in range(surface.xs, surface.xf):
                                for j in range(surface.ys, surface.yf):
                                    if surface.fractalsurface[i - surface.xs, j - surface.ys] > 0:
                                        depth = 0
                                        k = self.volume.zf - 1
                                        while k > self.volume.zs:
                                            if (
                                                k
                                                > self.volume.originalzf
                                                - (
                                                    surface.fractalsurface[
                                                        i - surface.xs, j - surface.ys
                                                    ]
                                                    - self.volume.originalzf
                                                )
                                                and self.volume.mask[
                                                    i - self.volume.xs,
                                                    j - self.volume.ys,
                                                    k - self.volume.zs,
                                                ]
                                                == 1
                                            ):
                                                x, y = g.calculate_root_geometry(root, depth)
                                                # Add x, y coordinates to existing location
                                                xx = int(i - self.volume.xs + x)
                                                yy = int(j - self.volume.ys + y)
                                                # If these coordinates are outwith the fractal volume stop building the root,
                                                # otherwise set the mask for grass.
                                                if (
                                                    xx < 0
                                                    or xx >= self.volume.mask.shape[0]
                                                    or yy < 0
                                                    or yy >= self.volume.mask.shape[1]
                                                ):
                                                    break
                                                else:
                                                    self.volume.mask[xx, yy, k - self.volume.zs] = 3
                                                    depth += 1
                                            k -= 1
                                        root += 1

                # Build voxels from any true values of the 3D mask array
                waternumID = next((x.numID for x in grid.materials if x.ID == "water"), 0)
                grassnumID = next((x.numID for x in grid.materials if x.ID == "grass"), 0)
                data = self.volume.fractalvolume.astype("int16", order="C")
                mask = self.volume.mask.copy(order="C")
                build_voxels_from_array_mask(
                    self.volume.xs,
                    self.volume.ys,
                    self.volume.zs,
                    config.get_model_config().ompthreads,
                    waternumID,
                    grassnumID,
                    self.volume.averaging,
                    mask,
                    data,
                    grid.solid,
                    grid.rigidE,
                    grid.rigidH,
                    grid.ID,
                )

            else:
                if self.volume.nbins == 1:
                    logger.exception(
                        f"{self.__str__()} is being used with a "
                        "single material and no modifications, "
                        "therefore please use a #box command instead."
                    )
                    raise ValueError
                else:
                    self.volume.generate_fractal_volume()
                    for i in range(0, self.volume.nx):
                        for j in range(0, self.volume.ny):
                            for k in range(0, self.volume.nz):
                                numberinbin = self.volume.fractalvolume[i, j, k]
                                self.volume.fractalvolume[i, j, k] = self.volume.mixingmodel.matID[
                                    int(numberinbin)
                                ]

                data = self.volume.fractalvolume.astype("int16", order="C")
                build_voxels_from_array(
                    self.volume.xs,
                    self.volume.ys,
                    self.volume.zs,
                    config.get_model_config().ompthreads,
                    0,
                    self.volume.averaging,
                    data,
                    grid.solid,
                    grid.rigidE,
                    grid.rigidH,
                    grid.ID,
                )
