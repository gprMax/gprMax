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
from typing import Optional, Tuple

from gprMax.grid.fdtd_grid import FDTDGrid


class RotatableMixin(ABC):
    """Stores parameters and defines an interface for rotatable objects.

    Attributes:
        axis (str): Defines the axis about which to perform the
            rotation. Must have value "x", "y", or "z". Default x.
        angle (int): Specifies the angle of rotation (degrees).
            Default 0.
        origin (tuple | None): Optional point about which to perform the
            rotation (x, y, z). Default None.
        do_rotate (bool): True if the object should be rotated. False
            otherwise. Default False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Forward all unused arguments
        self.axis = "x"
        self.angle = 0
        self.origin = None
        self.do_rotate = False

    def rotate(self, axis: str, angle: int, origin: Optional[Tuple[float, float, float]] = None):
        """Sets parameters for rotation.

        Args:
            axis: Defines the axis about which to perform the rotation.
                Must have value "x", "y", or "z".
            angle: Specifies the angle of rotation (degrees).
            origin: Optional point about which to perform the rotation
                (x, y, z). Default None.
        """
        self.axis = axis
        self.angle = angle
        self.origin = origin
        self.do_rotate = True

    @abstractmethod
    def _do_rotate(self, grid: FDTDGrid):
        """Performs the rotation."""
        pass
