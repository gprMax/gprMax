from abc import ABC, abstractmethod
from typing import Optional, Tuple

from gprMax.grid.fdtd_grid import FDTDGrid


class Rotatable(ABC):
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

    def __init__(self):
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
