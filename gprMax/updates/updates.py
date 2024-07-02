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


from abc import ABC, abstractmethod
from typing import Generic

from typing_extensions import TypeVar

from gprMax.grid.fdtd_grid import FDTDGrid

GridType = TypeVar("GridType", bound=FDTDGrid, default=FDTDGrid)


class Updates(Generic[GridType], ABC):
    """Defines update functions for a solver."""

    def __init__(self, G: GridType):
        """
        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        self.grid = G

    @abstractmethod
    def store_outputs(self, iteration: int) -> None:
        """Stores field component values for every receiver and transmission line."""
        pass

    @abstractmethod
    def store_snapshots(self, iteration: int) -> None:
        """Stores any snapshots.

        Args:
            iteration: int for iteration number.
        """
        pass

    @abstractmethod
    def update_magnetic(self) -> None:
        """Updates magnetic field components."""
        pass

    @abstractmethod
    def update_magnetic_pml(self) -> None:
        """Updates magnetic field components with the PML correction."""
        pass

    @abstractmethod
    def update_magnetic_sources(self, iteration: int) -> None:
        """Updates magnetic field components from sources."""
        pass

    @abstractmethod
    def update_electric_a(self) -> None:
        """Updates electric field components."""
        pass

    @abstractmethod
    def update_electric_pml(self) -> None:
        """Updates electric field components with the PML correction."""
        pass

    @abstractmethod
    def update_electric_sources(self, iteration: int) -> None:
        """Updates electric field components from sources -
        update any Hertzian dipole sources last.
        """
        pass

    @abstractmethod
    def update_electric_b(self) -> None:
        """If there are any dispersive materials do 2nd part of dispersive
        update - it is split into two parts as it requires present and
        updated electric field values. Therefore it can only be completely
        updated after the electric field has been updated by the PML and
        source updates.
        """
        pass

    @abstractmethod
    def time_start(self) -> None:
        """Starts timer used to calculate solving time for model."""
        pass

    @abstractmethod
    def calculate_solve_time(self) -> float:
        """Calculates solving time for model."""
        pass

    def finalise(self) -> None:
        pass

    def cleanup(self) -> None:
        pass
