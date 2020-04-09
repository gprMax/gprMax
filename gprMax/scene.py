# Copyright (C) 2015-2020: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

from .cmds_geometry.cmds_geometry import UserObjectGeometry
from .cmds_geometry.fractal_box_builder import FractalBoxBuilder
from .cmds_multiple import UserObjectMulti
from .cmds_single_use import (Discretisation, Domain, TimeWindow,
                              UserObjectSingle)
from .materials import create_built_in_materials
from .subgrids.user_objects import SubGridBase as SubGridUserBase
from .user_inputs import create_user_input_points
from .utilities import human_size

logger = logging.getLogger(__name__)


class Scene:
    """Scene stores all of the user created objects."""

    def __init__(self):
        self.multiple_cmds = []
        self.single_cmds = []
        self.geometry_cmds = []
        self.essential_cmds = [Domain, TimeWindow, Discretisation]

    def add(self, user_object):
        """Add the user object to the scene.

        :param user_object: User object to add to the scene. For example, :class:`gprMax.cmds_single_use.Domain`
        :type user_object: UserObjectMulti/UserObjectGeometry/UserObjectSingle
        """
        if isinstance(user_object, UserObjectMulti):
            self.multiple_cmds.append(user_object)
        elif isinstance(user_object, UserObjectGeometry):
            self.geometry_cmds.append(user_object)
        elif isinstance(user_object, UserObjectSingle):
            self.single_cmds.append(user_object)
        else:
            logger.exception('This object is unknown to gprMax')
            raise ValueError

    def process_subgrid_commands(self, subgrids):
        # Check for subgrid user objects
        def func(obj):
            if isinstance(obj, SubGridUserBase):
                return True
            else:
                return False

        # Subgrid user objects
        subgrid_cmds = list(filter(func, self.multiple_cmds))

        # Iterate through the user command objects under the subgrid user object
        for sg_cmd in subgrid_cmds:
            # When the subgrid is created its reference is attached to its user
            # object. This reference allows the multi and geo user objects
            # to build in the correct subgrid.
            sg = sg_cmd.subgrid
            self.process_cmds(sg_cmd.children_multiple, sg)
            self.process_cmds(sg_cmd.children_geometry, sg, sort=False)

    def process_cmds(self, commands, grid, sort=True):
        if sort:
            cmds_sorted = sorted(commands, key=lambda cmd: cmd.order)
        else:
            cmds_sorted = commands

        for obj in cmds_sorted:
            # in the first level all objects belong to the main grid
            uip = create_user_input_points(grid, obj)
            # Create an instance to check the geometry points provided by the
            # user. The way the point are checked depends on which grid the
            # points belong to.
            try:
                obj.create(grid, uip)
            except ValueError:
                logger.exception('Error creating user input object')
                raise

        return self

    def process_singlecmds(self, G):
        # Check for duplicate commands and warn user if they exist
        cmds_unique = list(set(self.single_cmds))
        if len(cmds_unique) != len(self.single_cmds):
            logger.exception('Duplicate single-use commands exist in the input.')
            raise ValueError

        # Check essential commands and warn user if missing
        for cmd_type in self.essential_cmds:
            d = any([isinstance(cmd, cmd_type) for cmd in cmds_unique])
            if not d:
                logger.exception('Your input file is missing essential commands required to run a model. Essential commands are: Domain, Discretisation, Time Window')
                raise ValueError

        self.process_cmds(cmds_unique, G)

    def create_internal_objects(self, G):
        """API presents the user with UserObjects in order to build the internal
            Rx(), Cylinder() etc... objects. This function essentially calls the
            UserObject.create() function in the correct way.
        """

        # Fractal box commands have an additional nonuser object which
        # processes modifications
        fbb = FractalBoxBuilder()
        self.add(fbb)

        # Create pre-defined (built-in) materials
        create_built_in_materials(G)

        # Process commands that can only have a single instance
        self.process_singlecmds(G)

        # Process main grid multiple commands
        self.process_cmds(self.multiple_cmds, G)

        # Initialise geometry arrays for main and subgrids
        for grid in [G] + G.subgrids:
            grid.initialise_geometry_arrays()

        # Process the main grid geometry commands
        self.process_cmds(self.geometry_cmds, G, sort=False)

        # Process all the commands for the subgrid
        self.process_subgrid_commands(G.subgrids)

        return self
