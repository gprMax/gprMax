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
from typing import List, Optional, Union

from gprMax.cmds_geometry.add_grass import AddGrass
from gprMax.cmds_geometry.add_surface_roughness import AddSurfaceRoughness
from gprMax.cmds_geometry.add_surface_water import AddSurfaceWater
from gprMax.cmds_geometry.cmds_geometry import UserObjectGeometry
from gprMax.cmds_geometry.fractal_box import FractalBox
from gprMax.cmds_multiuse import UserObjectMulti
from gprMax.cmds_singleuse import Discretisation, Domain, TimeWindow, UserObjectSingle
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import create_built_in_materials
from gprMax.model import Model
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.subgrids.user_objects import SubGridBase as SubGridUserBase
from gprMax.user_inputs import create_user_input_points

logger = logging.getLogger(__name__)


class Scene:
    """Scene stores all of the user created objects."""

    def __init__(self):
        self.multiple_cmds: List[UserObjectMulti] = []
        self.single_cmds: List[UserObjectSingle] = []
        self.geometry_cmds: List[UserObjectGeometry] = []
        self.essential_cmds = [Domain, TimeWindow, Discretisation]

    def add(self, user_object):
        """Add the user object to the scene.

        Args:
            user_object: user object to add to the scene. For example,
                            :class:`gprMax.cmds_single_use.Domain`
        """
        if isinstance(user_object, UserObjectMulti):
            self.multiple_cmds.append(user_object)
        elif isinstance(user_object, UserObjectGeometry):
            self.geometry_cmds.append(user_object)
        elif isinstance(user_object, UserObjectSingle):
            self.single_cmds.append(user_object)
        else:
            logger.exception("This object is unknown to gprMax")
            raise ValueError

    def build_grid_obj(self, obj: Union[UserObjectMulti, UserObjectGeometry], grid: FDTDGrid):
        """Builds objects in FDTDGrids.

        Args:
            obj: user object
            grid: FDTDGrid class describing a grid in a model.
        """
        uip = create_user_input_points(grid, obj)
        try:
            obj.build(grid, uip)
        except ValueError:
            logger.exception("Error creating user input object")
            raise

    def build_model_obj(self, obj: UserObjectSingle, model: Model):
        """Builds objects in models.

        Args:
            obj: user object
            model: Model being built
        """
        uip = create_user_input_points(model.G, obj)
        try:
            obj.build(model, uip)
        except ValueError:
            logger.exception("Error creating user input object")
            raise

    def process_subgrid_cmds(self, model: Model):
        """Process all commands in any sub-grids."""

        # Subgrid user objects
        subgrid_cmds = [
            sg_cmd for sg_cmd in self.multiple_cmds if isinstance(sg_cmd, SubGridUserBase)
        ]

        # Iterate through the user command objects under the subgrid user object
        for sg_cmd in subgrid_cmds:
            # When the subgrid is created its reference is attached to its user
            # object. This reference allows the multi and geo user objects
            # to build in the correct subgrid.
            sg = sg_cmd.subgrid
            self.process_cmds(sg_cmd.children_multiple, model, sg)
            self.process_geocmds(sg_cmd.children_geometry, sg)

    def process_cmds(
        self,
        commands: Union[List[UserObjectMulti], List[UserObjectSingle]],
        model: Model,
        subgrid: Optional[SubGridBaseGrid] = None,
    ):
        """Process list of commands."""
        grid = model.G if subgrid is None else subgrid
        cmds_sorted = sorted(commands, key=lambda cmd: cmd.order)
        for obj in cmds_sorted:
            if isinstance(obj, UserObjectSingle):
                self.build_model_obj(obj, model)
            else:
                self.build_grid_obj(obj, grid)

        return self

    def process_geocmds(self, commands, grid):
        # Check for fractal boxes and modifications and pre-process them first
        proc_cmds = []
        for obj in commands:
            if isinstance(obj, (FractalBox, AddGrass, AddSurfaceRoughness, AddSurfaceWater)):
                self.build_grid_obj(obj, grid)
                if isinstance(obj, (FractalBox)):
                    proc_cmds.append(obj)
            else:
                proc_cmds.append(obj)

        # Process all geometry commands
        for obj in proc_cmds:
            self.build_grid_obj(obj, grid)

        return self

    def process_singlecmds(self, model: Model):
        # Check for duplicate commands and warn user if they exist
        cmds_unique = list(set(self.single_cmds))
        if len(cmds_unique) != len(self.single_cmds):
            logger.exception("Duplicate single-use commands exist in the input.")
            raise ValueError

        # Check essential commands and warn user if missing
        for cmd_type in self.essential_cmds:
            d = any(isinstance(cmd, cmd_type) for cmd in cmds_unique)
            if not d:
                logger.exception(
                    "Your input file is missing essential commands "
                    + "required to run a model. Essential commands "
                    + "are: Domain, Discretisation, Time Window"
                )
                raise ValueError

        self.process_cmds(cmds_unique, model)

    def create_internal_objects(self, model: Model):
        """Calls the UserObject.build() function in the correct way - API
        presents the user with UserObjects in order to build the internal
        Rx(), Cylinder() etc... objects.
        """

        G = model.G

        # Create pre-defined (built-in) materials
        create_built_in_materials(G)

        # Process commands that can only have a single instance
        self.process_singlecmds(model)

        # Process main grid multiple commands
        self.process_cmds(self.multiple_cmds, model)

        # Initialise geometry arrays for main and subgrids
        for grid in [G] + G.subgrids:
            grid.initialise_geometry_arrays()

        # Process the main grid geometry commands
        self.process_geocmds(self.geometry_cmds, G)

        # Process all the commands for subgrids
        self.process_subgrid_cmds(model)
