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
from typing import List, Sequence

import numpy as np

from gprMax.geometry_tags import GeometryTagMap, GeometryTagRegistry
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import create_built_in_materials
from gprMax.model import Model
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.subgrids.user_objects import SubGridBase as SubGridUserBase
from gprMax.user_objects.cmds_geometry.add_grass import AddGrass
from gprMax.user_objects.cmds_geometry.add_surface_roughness import AddSurfaceRoughness
from gprMax.user_objects.cmds_geometry.add_surface_water import AddSurfaceWater
from gprMax.user_objects.cmds_geometry.fractal_box import FractalBox
from gprMax.user_objects.cmds_singleuse import Discretisation, Domain, TimeWindow
from gprMax.user_objects.user_objects import (
    GeometryUserObject,
    GridUserObject,
    ModelUserObject,
    OutputUserObject,
    UserObject,
)

logger = logging.getLogger(__name__)


class Scene:
    """Scene stores all of the user created objects."""

    ESSENTIAL_CMDS = [Domain, TimeWindow, Discretisation]

    def __init__(self):
        self.single_use_objects: List[ModelUserObject] = []
        self.grid_objects: List[GridUserObject] = []
        self.geometry_objects: List[GeometryUserObject] = []
        self.output_objects: List[OutputUserObject] = []
        self.subgrid_objects: List[SubGridUserBase] = []

    def add(self, user_object: UserObject):
        """Add the user object to the scene.

        Args:
            user_object: user object to add to the scene. For example,
                `gprMax.Domain`
        """
        # Check for
        if isinstance(user_object, SubGridUserBase):
            self.subgrid_objects.append(user_object)
        elif isinstance(user_object, ModelUserObject):
            self.single_use_objects.append(user_object)
        elif isinstance(user_object, GeometryUserObject):
            self.geometry_objects.append(user_object)
        elif isinstance(user_object, GridUserObject):
            self.grid_objects.append(user_object)
        elif isinstance(user_object, OutputUserObject):
            self.output_objects.append(user_object)
        else:
            raise TypeError(f"Object of type '{type(user_object)}' is unknown to gprMax")

    def build_model_objects(self, objects: Sequence[ModelUserObject], model: Model):
        """Builds objects in models.

        Args:
            obj: user object
            model: Model being built
        """
        try:
            for model_user_object in sorted(objects):
                model_user_object.build(model)
        except ValueError:
            logger.exception(f"Error creating user object '{model_user_object}'")
            raise

    def build_grid_objects(self, objects: Sequence[GridUserObject], grid: FDTDGrid):
        """Builds objects in FDTDGrids.

        Args:
            objects: user object
            grid: FDTDGrid class describing a grid in a model.
        """
        try:
            for grid_user_object in sorted(objects):
                grid_user_object.build(grid)
        except ValueError:
            logger.exception(f"Error creating user object '{grid_user_object}'")
            raise

    def build_output_objects(
        self, objects: Sequence[OutputUserObject], model: Model, grid: FDTDGrid
    ):
        try:
            for output_user_object in sorted(objects):
                output_user_object.build(model, grid)
        except ValueError:
            logger.exception(f"Error creating user object '{output_user_object}'")
            raise

    def process_single_use_objects(self, model: Model):
        # Check for duplicate commands and warn user if they exist. Each
        # single-use command TYPE (Domain, Discretisation, TimeWindow,
        # PMLThickness, ...) is meant to appear at most once per model.
        # `set(self.single_use_objects)` cannot detect this: UserObject
        # has no value-based __eq__/__hash__, so it falls back to
        # identity, meaning two separate Domain(...) instances built from
        # two `#domain` lines are never considered equal/duplicate - the
        # old check never fired for real duplicate commands. Detect
        # duplicates by command type instead.
        seen_types = set()
        duplicate_types = set()
        for cmd in self.single_use_objects:
            cmd_type = type(cmd)
            if cmd_type in seen_types:
                duplicate_types.add(cmd_type)
            seen_types.add(cmd_type)

        if duplicate_types:
            names = ", ".join(sorted(t.__name__ for t in duplicate_types))
            logger.exception(f"Duplicate single-use commands exist in the input: {names}.")
            raise ValueError

        # Check essential commands and warn user if missing
        for cmd_type in self.ESSENTIAL_CMDS:
            d = any(isinstance(cmd, cmd_type) for cmd in self.single_use_objects)
            if not d:
                logger.exception(
                    "Your input file is missing essential commands "
                    + "required to run a model. Essential commands "
                    + "are: Domain, Discretisation, Time Window"
                )
                raise ValueError

        self.build_model_objects(self.single_use_objects, model)

    def process_multi_use_objects(self, model: Model):
        self.build_grid_objects(self.grid_objects, model.G)
        self.build_output_objects(self.output_objects, model, model.G)
        self.build_model_objects(self.subgrid_objects, model)

    def process_geometry_objects(self, geometry_objects: List[GeometryUserObject], grid: FDTDGrid):
        # Check for fractal boxes and modifications and pre-process them first
        # TODO: Can this be removed in favour of sorting geometry objects?
        objects_to_be_built: List[GeometryUserObject] = []
        for obj in geometry_objects:
            if isinstance(obj, (FractalBox, AddGrass, AddSurfaceRoughness, AddSurfaceWater)):
                self.build_grid_objects([obj], grid)
                if isinstance(obj, (FractalBox)):
                    objects_to_be_built.append(obj)
            else:
                objects_to_be_built.append(obj)

        # Process all geometry commands
        self.build_grid_objects(objects_to_be_built, grid)

    @staticmethod
    def _build_internal_pml_enclosures(grid: FDTDGrid):
        """Build requested PEC enclosures after all user geometry.

        The plates must be the final geometry operation so their rigid PEC
        edges cannot be overwritten by a later user object. They are still
        built before ``FDTDGrid.build()``, material averaging, and PML
        coefficient generation, using the normal :class:`Plate` path.
        """
        from gprMax.user_objects.cmds_geometry.plate import Plate

        limits = (
            tuple(grid.global_size)
            if getattr(grid, "is_distributed", False) is True
            else (grid.nx, grid.ny, grid.nz)
        )
        for spec in grid.pmls["internal_specs"]:
            if not spec.build_pec:
                continue

            lower = np.array((spec.xs, spec.ys, spec.zs), dtype=int)
            upper = np.array((spec.xf, spec.yf, spec.zf), dtype=int)
            axis = "xyz".index(spec.maximum_face[0])
            faces = []

            for normal in range(3):
                if normal == axis:
                    continue
                for coordinate in (lower[normal], upper[normal]):
                    if coordinate in (0, limits[normal]):
                        continue
                    p1 = lower.copy()
                    p2 = upper.copy()
                    p1[normal] = coordinate
                    p2[normal] = coordinate
                    faces.append((p1, p2))

            maximum_coordinate = lower[axis] if spec.maximum_face.endswith("0") else upper[axis]
            if maximum_coordinate not in (0, limits[axis]):
                p1 = lower.copy()
                p2 = upper.copy()
                p1[axis] = maximum_coordinate
                p2[axis] = maximum_coordinate
                faces.append((p1, p2))

            for p1, p2 in faces:
                if isinstance(grid, SubGridBaseGrid):
                    continuous_p1 = tuple(grid.local_to_global(p1))
                    continuous_p2 = tuple(grid.local_to_global(p2))
                else:
                    continuous_p1 = tuple(np.asarray(grid.dl) * p1)
                    continuous_p2 = tuple(np.asarray(grid.dl) * p2)
                Plate(
                    p1=continuous_p1,
                    p2=continuous_p2,
                    material_id="pec",
                ).build(grid)

            grid.pmls["internal_registry"][spec.ID]["generated_pec_faces"] = len(faces)
            logger.info(
                f"Internal PML slab '{spec.ID}' generated {len(faces)} PEC enclosure plates."
            )

    def process_subgrid_objects(self, model: Model):
        """Process all commands in any sub-grids."""
        # Iterate through the user command objects under the subgrid user object
        for subgrid_object in self.subgrid_objects:
            # When the subgrid is created its reference is attached to its user
            # object. This reference allows the multi and geo user objects
            # to build in the correct subgrid.
            subgrid = subgrid_object.subgrid
            self.build_grid_objects(subgrid_object.children_grid, subgrid)
            self.build_output_objects(subgrid_object.children_output, model, subgrid)
            self.process_geometry_objects(subgrid_object.children_geometry, subgrid)
            self._build_internal_pml_enclosures(subgrid)

    def initialise_geometry_tags(self, model: Model):
        """Create one model-wide registry and optional cell maps per grid."""

        registry = GeometryTagRegistry()
        grid_objects = [(model.G, self.geometry_objects)]
        grid_objects.extend(
            (subgrid_object.subgrid, subgrid_object.children_geometry)
            for subgrid_object in self.subgrid_objects
        )

        declared_by_grid = []
        for grid, objects in grid_objects:
            declared = tuple(
                tag for obj in objects for tag in obj.declared_geometry_tags()
            )
            registry.register_many(declared)
            declared_by_grid.append((grid, declared))

        registry.freeze()
        if not registry.has_tags:
            model.geometry_tag_registry = None
            for grid, _ in declared_by_grid:
                grid.geometry_tag_registry = None
                grid.geometry_tag_map = None
            return

        model.geometry_tag_registry = registry
        for grid, declared in declared_by_grid:
            grid.geometry_tag_registry = registry
            grid.geometry_tag_map = (
                GeometryTagMap(tuple(int(value) for value in grid.size), registry)
                if declared
                else None
            )

    def create_internal_objects(self, model: Model):
        """Calls the UserObject.build() function in the correct way - API
        presents the user with UserObjects in order to build the internal
        Rx(), Cylinder() etc... objects.
        """

        # Create pre-defined (built-in) materials
        create_built_in_materials(model.G)

        # Process commands that can only have a single instance
        self.process_single_use_objects(model)

        # Process multiple commands
        self.process_multi_use_objects(model)

        # Discover semantic tags, including catalogues stored in imported
        # geometry, before selecting the compact map dtype.
        self.initialise_geometry_tags(model)

        # Initialise geometry arrays for main and subgrids
        for grid in [model.G] + model.subgrids:
            grid.initialise_geometry_arrays()

        # Process the main grid geometry commands
        self.process_geometry_objects(self.geometry_objects, model.G)
        self._build_internal_pml_enclosures(model.G)

        # Process all the commands for subgrids
        self.process_subgrid_objects(model)
