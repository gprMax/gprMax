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
from typing import List, Sequence

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import create_built_in_materials
from gprMax.model import Model
from gprMax.mpi_model import MPIModel
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
                `gprMax.user_objects.cmds_singleuse.Domain`
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
        # Check for duplicate commands and warn user if they exist
        # TODO: Test this works
        unique_commands = list(set(self.single_use_objects))
        if len(unique_commands) != len(self.single_use_objects):
            logger.exception("Duplicate single-use commands exist in the input.")
            raise ValueError

        # Check essential commands and warn user if missing
        for cmd_type in self.ESSENTIAL_CMDS:
            d = any(isinstance(cmd, cmd_type) for cmd in unique_commands)
            if not d:
                logger.exception(
                    "Your input file is missing essential commands "
                    + "required to run a model. Essential commands "
                    + "are: Domain, Discretisation, Time Window"
                )
                raise ValueError

        self.build_model_objects(unique_commands, model)

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

    def create_internal_objects(self, model: Model):
        """Calls the UserObject.build() function in the correct way - API
        presents the user with UserObjects in order to build the internal
        Rx(), Cylinder() etc... objects.
        """

        # Create pre-defined (built-in) materials
        create_built_in_materials(model.G)

        # Process commands that can only have a single instance
        self.process_single_use_objects(model)

        if (
            isinstance(model, MPIModel)
            and model.is_coordinator()
            or not isinstance(model, MPIModel)
        ):
            # Process multiple commands
            self.process_multi_use_objects(model)

            # Initialise geometry arrays for main and subgrids
            for grid in [model.G] + model.subgrids:
                grid.initialise_geometry_arrays()

            # Process the main grid geometry commands
            self.process_geometry_objects(self.geometry_objects, model.G)

            # Process all the commands for subgrids
            self.process_subgrid_objects(model)
