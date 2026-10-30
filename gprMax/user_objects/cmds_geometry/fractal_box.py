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
from gprMax.cython.geometry_primitives import build_voxels_from_array, build_voxels_from_array_mask
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import ListMaterial
from gprMax.user_objects.cmds_geometry.cmds_geometry import (
    check_averaging,
    fractal_seed,
    geometry_tag_args,
    resolve_geometry_materials,
)
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


class FractalBox(GeometryUserObject):
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
                            material. Several boxes may share a mixing model;
                            each box uses its own n_materials bin mapping.
        id: string identifier for the fractal box itself.
        seed: (optional) whole number seeding the random number generator used
                to create the fractals: the same seed gives the same fractal. Without
                one the fractal differs from run to run.
        averaging: string (y or n) used to switch on and off dielectric smoothing.
        tag: optional semantic geometry-tag string written to the final cells
            occupied by the fractal volume.
    """

    @property
    def hash(self):
        return "#fractal_box"

    _allowed_kwargs = frozenset({
        "averaging", "frac_dim", "id", "mixing_model_id", "n_materials", "p1", "p2", "seed",
        "tag", "weighting",
    })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_pre_build = True

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
            seed = fractal_seed(self.kwargs["seed"], self.__str__())
        except KeyError:
            logger.warning(
                f"{self.__str__()} no value for seed detected. This "
                "means you will get a different fractal distribution "
                "every time the model runs."
            )
            seed = None

        # Check averaging
        try:
            # Go with user specified averaging
            averagefractalbox = check_averaging(self.kwargs["averaging"])
        except KeyError:
            # If they havent specified - default is no dielectric smoothing for
            # a fractal box.
            averagefractalbox = False

        uip = self._create_uip(grid)
        p1 = uip.resolve_inf_point(p1, role="lower")
        p2 = uip.resolve_inf_point(p2, role="upper")
        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        p1, p2 = uip.check_output_object_bounds(p1, p2, self.__str__())

        xs, ys, zs = p1
        xf, yf, zf = p2

        if frac_dim < 0:
            raise ValueError(
                f"{self.__str__()} requires a positive value for the fractal dimension"
            )
        if weighting[0] < 0:
            raise ValueError(
                f"{self.__str__()} requires a positive value for the fractal weighting in the x direction"
            )
        if weighting[1] < 0:
            raise ValueError(
                f"{self.__str__()} requires a positive value for the fractal weighting in the y direction"
            )
        if weighting[2] < 0:
            raise ValueError(
                f"{self.__str__()} requires a positive value for the fractal weighting in the z direction"
            )
        if n_materials <= 0:
            raise ValueError(f"{self.__str__()} requires a positive value for the number of bins")

        # Find materials to use to build fractal volume, either from mixing
        # models or normal materials.
        mixingmodel = next((x for x in grid.mixingmodels if x.ID == mixing_model_id), None)
        material = next((x for x in grid.materials if x.ID == mixing_model_id), None)
        surface_impedance = getattr(grid, "surface_impedance_models", {}).get(mixing_model_id)
        if surface_impedance is not None and mixingmodel is not None:
            raise ValueError(
                f"{self.params_str()} ID {mixing_model_id!r} is ambiguous between "
                "a mixing model and a surface impedance"
            )
        if mixingmodel is None and material is None and surface_impedance is not None:
            material = resolve_geometry_materials(
                grid,
                [mixing_model_id],
                geometry=self.params_str(),
            )[0]
        nbins = n_materials

        if mixingmodel:
            if nbins == 1:
                raise ValueError(
                    f"{self.__str__()} must be used with more than one material from the mixing model."
                )
            if isinstance(mixingmodel, ListMaterial) and nbins > len(mixingmodel.mat):
                raise ValueError(
                    f"{self.__str__()} too many materials/bins "
                    "requested compared to materials in "
                    "mixing model."
                )
            # Create materials from mixing model as number of bins now known
            # from fractal_box command.
            mixingmodel.calculate_properties(nbins, grid)
        elif not material:
            raise ValueError(
                f"{self.__str__()} mixing model or material with "
                + "ID {mixing_model_id} does not exist"
            )
        elif hasattr(material, "surface_impedance_id"):
            if nbins != 1:
                raise ValueError(f"{self.params_str()} a surface impedance requires n_materials=1")
            averagefractalbox = False

        if any(volume.ID == ID for volume in grid.fractalvolumes):
            raise ValueError(f"{self.__str__()} FractalBox ID {ID!r} is already in use")

        # grid.add_fractal_volume() already appends the new volume to
        # grid.fractalvolumes (and MPIGrid's override does the same) -
        # an extra `grid.fractalvolumes.append(self.volume)` used to sit
        # at the end of this method, registering the identical object a
        # second time. Harmless for add_grass/add_surface_roughness/
        # add_surface_water (they only ever take the first match) and
        # fractal generation itself (triggered directly via self.volume,
        # never by iterating grid.fractalvolumes), but FDTDGrid.
        # mem_est_fractals() sums memory per entry in that list, so every
        # fractal box's estimated memory was silently doubled in the
        # pre-run "Memory required" report and the host memory-sufficiency
        # check (gprMax/utilities/host_info.py) - not real wasted memory
        # (no second array was ever allocated), just an inflated estimate.
        self.volume = grid.add_fractal_volume(xs, xf, ys, yf, zs, zf, frac_dim, seed)
        self.volume.ID = ID
        self.volume.operatingonID = mixing_model_id
        self.volume.nbins = nbins
        self.volume.weighting = weighting
        self.volume.averaging = averagefractalbox
        self.volume.mixingmodel = mixingmodel
        if mixingmodel is not None:
            # Scene prepares every fractal box before rasterising any of them.
            # A later box can evaluate this model with a different bin count.
            self.volume.material_ids = tuple(mixingmodel.matID)

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

    def build(self, grid: FDTDGrid):
        # The two passes belong to a grid build, not to this reusable API
        # definition's lifetime. A preview, retry, backend change or another
        # Scene may build the same definition against a fresh grid. Its volume
        # and material-bin IDs must be registered on that grid before modifiers
        # run; never rasterise a previous grid's volume during the prepass.
        prepared_here = any(
            volume is getattr(self, "volume", None) for volume in grid.fractalvolumes
        )
        if self.do_pre_build or not prepared_here:
            self.pre_build(grid)
            self.do_pre_build = False
        else:
            if self.volume.fractalsurfaces:
                # Extend the volume to accomodate any rough surfaces, grass,
                # or roots
                for surface in self.volume.fractalsurfaces:
                    if surface.surfaceID == "xminus":
                        if surface.fractalrange[0] < self.volume.xs:
                            self.volume.xs = surface.fractalrange[0]
                    elif surface.surfaceID == "xplus":
                        if surface.fractalrange[1] > self.volume.xf:
                            self.volume.xf = surface.fractalrange[1]
                    elif surface.surfaceID == "yminus":
                        if surface.fractalrange[0] < self.volume.ys:
                            self.volume.ys = surface.fractalrange[0]
                    elif surface.surfaceID == "yplus":
                        if surface.fractalrange[1] > self.volume.yf:
                            self.volume.yf = surface.fractalrange[1]
                    elif surface.surfaceID == "zminus":
                        if surface.fractalrange[0] < self.volume.zs:
                            self.volume.zs = surface.fractalrange[0]
                    elif surface.surfaceID == "zplus":
                        if surface.fractalrange[1] > self.volume.zf:
                            self.volume.zf = surface.fractalrange[1]

                # MPI grass is stamped from a mask replicated on every rank,
                # since blades and roots can straddle rank boundaries. That
                # needs the extended global bounds, so capture start/stop
                # before generate_fractal_volume() clips them locally.
                global_grass = (
                    getattr(grid, "is_distributed", False)
                    and grid.comm.Get_size() > 1
                    and any(surface.ID == "grass" for surface in self.volume.fractalsurfaces)
                )
                if global_grass:
                    first_grass = next(
                        index
                        for index, surface in enumerate(self.volume.fractalsurfaces)
                        if surface.ID == "grass"
                    )
                    if any(
                        surface.ID != "grass"
                        for surface in self.volume.fractalsurfaces[first_grass:]
                    ):
                        raise ValueError(
                            f"{self.__str__()} in an MPI decomposition requires every rough "
                            "surface or surface water command to be defined before #add_grass."
                        )
                    preclip_start = self.volume.start.copy()
                    preclip_stop = self.volume.stop.copy()

                # If there is only 1 bin then a normal material is being used,
                # otherwise a mixing model
                volume_ready = True
                if self.volume.nbins == 1:
                    if global_grass:
                        # A single-material volume never goes through the
                        # distributed generator, which is where start/stop
                        # normally get clipped to this rank.
                        self.volume.start = np.maximum(self.volume.start, 0)
                        self.volume.stop = np.minimum(self.volume.stop, grid.size)
                        self.volume.stop = np.maximum(self.volume.stop, self.volume.start)
                    self.volume.fractalvolume = np.ones(
                        (self.volume.nx, self.volume.ny, self.volume.nz),
                        dtype=config.sim_config.dtypes["float_or_double"],
                    )
                    materialnumID = next(
                        x.numID
                        for x in grid.materials
                        if x.ID == self.volume.operatingonID
                        or getattr(x, "surface_impedance_id", None) == self.volume.operatingonID
                    )
                    self.volume.fractalvolume *= materialnumID
                else:
                    volume_ready = self.volume.generate_fractal_volume()
                    if not volume_ready and not global_grass:
                        return
                    if volume_ready:
                        for i in range(0, self.volume.nx):
                            for j in range(0, self.volume.ny):
                                for k in range(0, self.volume.nz):
                                    numberinbin = self.volume.fractalvolume[i, j, k]
                                    self.volume.fractalvolume[i, j, k] = self.volume.material_ids[
                                        int(numberinbin)
                                    ]

                if volume_ready:
                    self.volume.generate_volume_mask()

                # Apply any rough surfaces and add any surface water to the
                # 3D mask array
                # TODO: Allow extract of rough surface profile (to print/file?)
                for surface in self.volume.fractalsurfaces:
                    if global_grass and surface.ID == "grass":
                        # Stamped across every rank from the replicated mask
                        # after the local rough surface/water passes.
                        continue
                    if not volume_ready:
                        # Rank does not own any part of the extended volume.
                        continue
                    if surface.surfaceID == "xminus":
                        surface.fractalrange = (
                            max(surface.fractalrange[0], 0),
                            min(surface.fractalrange[1], grid.nx),
                        )
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
                        surface.fractalrange = (
                            max(surface.fractalrange[0], 0),
                            min(surface.fractalrange[1], grid.nx),
                        )
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
                            # In 2D TE mode the invariant axis is 2 cells
                            # thick (vs TM's 1). TM's own out-of-bounds check
                            # already stops blade/root growth the instant the
                            # wobble offset along that axis is nonzero (its
                            # mask is only 1 cell deep there, so *any*
                            # nonzero offset is out of bounds) - TE's mask is
                            # 2 cells deep, so the same nonzero offset would
                            # stay in-bounds and growth would continue
                            # differently. Treat a nonzero invariant-axis
                            # offset as out-of-bounds in TE too (even though
                            # it's technically a valid index), so growth
                            # stops at exactly the same (blade/root, height/
                            # depth) step TM's would, keeping the two
                            # reproducible and keeping growth confined to the
                            # surface's own genuinely 2D in-plane axis.
                            te_axis = None
                            mode = config.get_model_config().mode
                            if mode.startswith("2D TE"):
                                invariant_axis = "xyz".index(mode[-1])
                                if surface.size[invariant_axis] == 2:
                                    te_axis = surface._te_invariant_inplane_index(invariant_axis)
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
                                                    or (te_axis == 0 and y != 0)
                                                    or (te_axis == 1 and z != 0)
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
                                                    or (te_axis == 0 and y != 0)
                                                    or (te_axis == 1 and z != 0)
                                                ):
                                                    break
                                                else:
                                                    self.volume.mask[i - self.volume.xs, yy, zz] = 3
                                                    depth += 1
                                            i -= 1
                                        root += 1

                    elif surface.surfaceID == "yminus":
                        surface.fractalrange = (
                            max(surface.fractalrange[0], 0),
                            min(surface.fractalrange[1], grid.ny),
                        )
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
                        surface.fractalrange = (
                            max(surface.fractalrange[0], 0),
                            min(surface.fractalrange[1], grid.ny),
                        )
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
                            # See the equivalent xplus block above for why
                            # this is needed: forces the invariant-axis
                            # wobble component to zero for TM/TE-reproducible
                            # blade/root growth.
                            te_axis = None
                            mode = config.get_model_config().mode
                            if mode.startswith("2D TE"):
                                invariant_axis = "xyz".index(mode[-1])
                                if surface.size[invariant_axis] == 2:
                                    te_axis = surface._te_invariant_inplane_index(invariant_axis)
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
                                                    or (te_axis == 0 and x != 0)
                                                    or (te_axis == 1 and z != 0)
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
                                                    or (te_axis == 0 and x != 0)
                                                    or (te_axis == 1 and z != 0)
                                                ):
                                                    break
                                                else:
                                                    self.volume.mask[xx, j - self.volume.ys, zz] = 3
                                                    depth += 1
                                            j -= 1
                                        root += 1

                    elif surface.surfaceID == "zminus":
                        surface.fractalrange = (
                            max(surface.fractalrange[0], 0),
                            min(surface.fractalrange[1], grid.nz),
                        )
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
                        surface.fractalrange = (
                            max(surface.fractalrange[0], 0),
                            min(surface.fractalrange[1], grid.nz),
                        )
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
                            # See the equivalent xplus block above for why
                            # this is needed: forces the invariant-axis
                            # wobble component to zero for TM/TE-reproducible
                            # blade/root growth.
                            te_axis = None
                            mode = config.get_model_config().mode
                            if mode.startswith("2D TE"):
                                invariant_axis = "xyz".index(mode[-1])
                                if surface.size[invariant_axis] == 2:
                                    te_axis = surface._te_invariant_inplane_index(invariant_axis)
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
                                                    or (te_axis == 0 and x != 0)
                                                    or (te_axis == 1 and y != 0)
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
                                                    or (te_axis == 0 and x != 0)
                                                    or (te_axis == 1 and y != 0)
                                                ):
                                                    break
                                                else:
                                                    self.volume.mask[xx, yy, k - self.volume.zs] = 3
                                                    depth += 1
                                            k -= 1
                                        root += 1

                if global_grass:
                    self._stamp_grass_global(grid, preclip_start, preclip_stop)
                    if not volume_ready:
                        return

                # In 2D TE mode the invariant axis is 2 cells thick. The
                # rough-surface/water mask assignment above is already
                # invariant (it thresholds against fractalsurface, which
                # generate_fractal_surface() already made identical on both
                # cells), but grass's blade/root placement wobbles using a
                # per-(row,column) counter into Grass's geometry parameters
                # that isn't itself invariant-axis-aware, so it can build
                # different blade shapes on each cell even from an identical
                # underlying height map. Force the mask to match on both
                # cells here as a general safety net - a no-op for content
                # that's already invariant, a real fix for grass.
                mode = config.get_model_config().mode
                if mode.startswith("2D TE"):
                    invariant_axis = "xyz".index(mode[-1])
                    if self.volume.size[invariant_axis] == 2:
                        indexer0 = [slice(None), slice(None), slice(None)]
                        indexer0[invariant_axis] = 0
                        indexer1 = [slice(None), slice(None), slice(None)]
                        indexer1[invariant_axis] = 1
                        self.volume.mask[tuple(indexer1)] = self.volume.mask[tuple(indexer0)]

                # Build voxels from any true values of the 3D mask array
                waternumID = next((x.numID for x in grid.materials if x.ID == "water"), 0)
                grassnumID = next((x.numID for x in grid.materials if x.ID == "grass"), 0)
                # These are target-grid material IDs, not compact file indices.
                data = self.volume.fractalvolume.astype("int32", order="C")
                mask = self.volume.mask.copy(order="C")
                is_pec_lookup = np.array([m.is_pec for m in grid.materials], dtype=np.uint8)
                is_averagable_lookup = np.array(
                    [m.averagable for m in grid.materials], dtype=np.uint8
                )
                tag_data, tag_id = geometry_tag_args(grid, self.kwargs.get("tag"))
                build_voxels_from_array_mask(
                    self.volume.xs,
                    self.volume.ys,
                    self.volume.zs,
                    waternumID,
                    grassnumID,
                    self.volume.averaging,
                    is_pec_lookup,
                    is_averagable_lookup,
                    mask,
                    data,
                    grid.solid,
                    grid.rigidE,
                    grid.rigidH,
                    grid.ID,
                    tag_data,
                    tag_id,
                    0,
                    0,
                )

                # fractalvolume/mask are only needed to build the voxel data
                # above (already consumed into grid.solid/rigidE/rigidH/ID) -
                # nothing later in this grid's solve reads them again. A
                # fresh grid rebuild prepares a new volume; geometry_fixed
                # reuse skips geometry construction entirely. Freeing them
                # here matters for multi-scene
                # sweeps, where the caller's own scenes list keeps every
                # Scene - and therefore every FractalBox/FractalVolume -
                # alive for the whole run; without this, a large fractal
                # box's array (real memory, sized to its cell count) would
                # otherwise persist for as long as the caller holds that
                # scene, accumulating across scenes.
                self.volume.fractalvolume = None
                self.volume.mask = None

            else:
                if self.volume.nbins == 1:
                    raise ValueError(
                        f"{self.__str__()} is being used with a "
                        "single material and no modifications, "
                        "therefore please use a #box command instead."
                    )
                else:
                    if not self.volume.generate_fractal_volume():
                        return
                    for i in range(0, self.volume.nx):
                        for j in range(0, self.volume.ny):
                            for k in range(0, self.volume.nz):
                                numberinbin = self.volume.fractalvolume[i, j, k]
                                self.volume.fractalvolume[i, j, k] = self.volume.material_ids[
                                    int(numberinbin)
                                ]

                # These are target-grid material IDs, not compact file indices.
                data = self.volume.fractalvolume.astype("int32", order="C")
                is_pec_lookup = np.array([m.is_pec for m in grid.materials], dtype=np.uint8)
                is_averagable_lookup = np.array(
                    [m.averagable for m in grid.materials], dtype=np.uint8
                )
                tag_data, tag_id = geometry_tag_args(grid, self.kwargs.get("tag"))
                build_voxels_from_array(
                    self.volume.xs,
                    self.volume.ys,
                    self.volume.zs,
                    0,
                    self.volume.averaging,
                    is_pec_lookup,
                    is_averagable_lookup,
                    data,
                    grid.solid,
                    grid.rigidE,
                    grid.rigidH,
                    grid.ID,
                    tag_data,
                    tag_id,
                )

                # See the comment on the equivalent free in the
                # fractalsurfaces branch above - fractalvolume is fully
                # consumed by this point and never read again.
                self.volume.fractalvolume = None

    def _stamp_grass_global(self, grid: FDTDGrid, preclip_start, preclip_stop):
        """Stamps grass blades and roots across every MPI rank.

        Every rank holds the whole grass surface but only its own slice of
        the volume mask. The serial blade/root loops are replayed against a
        mask assembled from all ranks' slices, so the geometry is identical
        to a serial run; each rank then keeps only its own slice for local
        voxelisation.

        Args:
            grid: Grid being built.
            preclip_start, preclip_stop: Local extended volume bounds
                captured before generate_fractal_volume() clipped them.
        """

        lower_extent = grid.lower_extent
        local_mask = getattr(self.volume, "mask", None)
        local_start = self.volume.start + lower_extent
        local_stop = self.volume.stop + lower_extent
        # Clipped local bounds cannot be shifted by this rank's lower_extent
        # into a shared origin - an object crossing a rank's negative face
        # is clipped to zero there. Every item therefore carries its own
        # global bounds and the union origin is derived from them.
        payload = grid.comm.allgather(
            (
                local_mask,
                local_start,
                local_stop,
                preclip_start + lower_extent,
                preclip_stop + lower_extent,
                self.volume.unclipped_original_stop + lower_extent,
            )
        )

        global_start = np.min(np.stack([item[3] for item in payload]), axis=0)
        global_stop = np.max(np.stack([item[4] for item in payload]), axis=0)
        original_stop = np.max(np.stack([item[5] for item in payload]), axis=0)

        global_mask = np.zeros(global_stop - global_start, dtype=np.int8)
        for mask, start, stop, _, _, _ in payload:
            if mask is None or np.any(stop <= start):
                continue
            destination = start - global_start
            global_mask[
                destination[0] : destination[0] + stop[0] - start[0],
                destination[1] : destination[1] + stop[1] - start[1],
                destination[2] : destination[2] + stop[2] - start[2],
            ] = mask

        for surface in self.volume.fractalsurfaces:
            if surface.ID == "grass":
                self._stamp_grass_surface(
                    surface, grid, global_mask, global_start, global_stop, original_stop
                )

        if local_mask is not None and np.all(local_stop > local_start):
            destination = local_start - global_start
            local_mask[...] = global_mask[
                destination[0] : destination[0] + local_stop[0] - local_start[0],
                destination[1] : destination[1] + local_stop[1] - local_start[1],
                destination[2] : destination[2] + local_stop[2] - local_start[2],
            ]

    def _stamp_grass_surface(
        self, surface, grid: FDTDGrid, mask, volume_start, volume_stop, original_stop
    ):
        """Replays one grass surface's serial blade/root loops on a global
        mask. Coordinates and the mask are all in global indices.
        """

        lower_extent = grid.lower_extent
        normal = "xyz".index(surface.surfaceID[0])
        dims = [axis for axis in range(3) if axis != normal]
        first, second = dims
        fractalrange = (
            surface.fractalrange[0] + lower_extent[normal],
            surface.fractalrange[1] + lower_extent[normal],
        )
        g = surface.grass[0]

        te_axis = None
        mode = config.get_model_config().mode
        if mode.startswith("2D TE"):
            invariant_axis = "xyz".index(mode[-1])
            if surface.size[invariant_axis] == 2:
                te_axis = surface._te_invariant_inplane_index(invariant_axis)

        shape = np.array(mask.shape)
        index = np.zeros(3, dtype=np.int64)

        # Build the blades of the grass
        blade = 0
        for u in range(surface.start[first], surface.stop[first]):
            for v in range(surface.start[second], surface.stop[second]):
                # Height-map entries already hold normal-axis coordinates in
                # the global frame (see AddGrass.build).
                height_map = surface.fractalsurface[u - surface.start[first], v - surface.start[second]]
                if height_map <= 0:
                    continue
                height = 0
                for w in range(volume_start[normal], int(fractalrange[1])):
                    index[normal] = w - volume_start[normal]
                    index[first] = u + lower_extent[first] - volume_start[first]
                    index[second] = v + lower_extent[second] - volume_start[second]
                    if w < height_map and mask[tuple(index)] != 1:
                        offset_first, offset_second = g.calculate_blade_geometry(blade, height)
                        uu = int(u + lower_extent[first] - volume_start[first] + offset_first)
                        vv = int(v + lower_extent[second] - volume_start[second] + offset_second)
                        if (
                            uu < 0
                            or uu >= shape[first]
                            or vv < 0
                            or vv >= shape[second]
                            or (te_axis == 0 and offset_first != 0)
                            or (te_axis == 1 and offset_second != 0)
                        ):
                            break
                        index[first] = uu
                        index[second] = vv
                        mask[tuple(index)] = 3
                        height += 1
                blade += 1

        # Build the roots of the grass
        root = 0
        for u in range(surface.start[first], surface.stop[first]):
            for v in range(surface.start[second], surface.stop[second]):
                height_map = surface.fractalsurface[u - surface.start[first], v - surface.start[second]]
                if height_map <= 0:
                    continue
                depth = 0
                w = volume_stop[normal] - 1
                while w > volume_start[normal]:
                    index[normal] = w - volume_start[normal]
                    index[first] = u + lower_extent[first] - volume_start[first]
                    index[second] = v + lower_extent[second] - volume_start[second]
                    if (
                        w > original_stop[normal] - (height_map - original_stop[normal])
                        and mask[tuple(index)] == 1
                    ):
                        offset_first, offset_second = g.calculate_root_geometry(root, depth)
                        uu = int(u + lower_extent[first] - volume_start[first] + offset_first)
                        vv = int(v + lower_extent[second] - volume_start[second] + offset_second)
                        if (
                            uu < 0
                            or uu >= shape[first]
                            or vv < 0
                            or vv >= shape[second]
                            or (te_axis == 0 and offset_first != 0)
                            or (te_axis == 1 and offset_second != 0)
                        ):
                            break
                        index[first] = uu
                        index[second] = vv
                        mask[tuple(index)] = 3
                        depth += 1
                    w -= 1
                root += 1
