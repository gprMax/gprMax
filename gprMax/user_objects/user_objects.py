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

from abc import ABC, abstractmethod
from typing import List, Union

from gprMax import config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.model import Model
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.user_inputs import MainGridUserInput, MPIUserInput, SubgridUserInput


class UserObject(ABC):
    """User defined object.

    Attributes:
        order (int): Specifies the order user objects should be
            constructed in.
        hash (str): gprMax hash command used to create the user object
            in an input file.
        kwargs (dict): Keyword arguments used to construct the user
            object.
        autotranslate (bool): TODO
        is_single_use (bool): True if the object can only appear once in a
            given model. False otherwise. Default True.
        is_geometry_object (bool): True if the object adds geometry to the
            model. False otherwise. Default False.
    """

    @property
    @abstractmethod
    def order(self) -> int:
        pass

    @property
    @abstractmethod
    def hash(self) -> str:
        pass

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.autotranslate = True

    def __lt__(self, obj: "UserObject"):
        return self.order < obj.order

    def __str__(self) -> str:
        """Readable user object as per hash commands."""
        args: List[str] = []
        for value in self.kwargs.values():
            if isinstance(value, (tuple, list)):
                for element in value:
                    args.append(str(element))
            elif value is not None:
                args.append(str(value))

        return f"{self.hash}: {' '.join(args)}"

    def params_str(self) -> str:
        """Readable string of parameters given to object."""
        return f"{self.hash}: {str(self.kwargs)}"

    def _create_uip(self, grid: FDTDGrid) -> MainGridUserInput:
        """Returns a point checker class based on the grid supplied.

        Args:
            grid: Grid to get a UserInput object for.

        Returns:
            uip: UserInput object for the grid provided.
        """

        # If autotranslate is set as True globally, local object
        # configuration trumps. I.e. User can turn off autotranslate for
        # specific objects.
        if (
            isinstance(grid, SubGridBaseGrid)
            and config.sim_config.args.autotranslate
            and self.autotranslate
        ):
            return SubgridUserInput(grid)
        elif isinstance(grid, MPIGrid):
            return MPIUserInput(grid)
        else:
            return MainGridUserInput(grid)


class ModelUserObject(UserObject):
    """User defined object to add to the model."""

    @abstractmethod
    def build(self, model: Model):
        """Build user object and set model properties.

        Args:
            model: Model to set the properties of.
        """
        pass


class GridUserObject(UserObject):
    """User defined object to add to a grid."""

    @abstractmethod
    def build(self, grid: FDTDGrid):
        pass

    def grid_name(self, grid: FDTDGrid) -> str:
        """Format grid name for use with logging info.

        Returns an empty string if the grid is the main grid.

        Args:
            grid: Grid to get the name of.

        Returns:
            grid_name: Formatted version of the grid name.
        """
        if isinstance(grid, SubGridBaseGrid):
            return f"[{grid.name}] "
        else:
            return ""


class OutputUserObject(UserObject):
    """User defined object that controls the output of data."""

    @abstractmethod
    def build(self, model: Model, grid: FDTDGrid):
        pass

    def grid_name(self, grid: FDTDGrid) -> str:
        """Format grid name for use with logging info.

        Returns an empty string if the grid is the main grid.

        Args:
            grid: Grid to get the name of.

        Returns:
            grid_name: Formatted version of the grid name.
        """
        if isinstance(grid, SubGridBaseGrid):
            return f"[{grid.name}] "
        else:
            return ""


class GeometryUserObject(GridUserObject):
    """User defined object that adds geometry to a grid."""

    @property
    def order(self):
        """Geometry Objects do not have an ordering.

        They should be built in the order they were added to the scene.
        """
        return 1
