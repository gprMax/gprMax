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
from __future__ import annotations

import logging
from typing import Generic, Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.subgrids.grid import SubGridBaseGrid

from .utilities.utilities import round_int

logger = logging.getLogger(__name__)


"""Module contains classes to handle points supplied by a user. The
    classes implement a common interface such that geometry building objects
    such as box or triangle do not need to have any knowledge which grid to
    which they are rounding continuous points or checking the point is within
    the grid. Additionally all logic related to rounding points etc is
    encapulsated here.
"""

GridType = TypeVar("GridType", bound=FDTDGrid, default=FDTDGrid)


class UserInput(Generic[GridType]):
    """Handles (x, y, z) points supplied by the user."""

    def __init__(self, grid: GridType):
        self.grid = grid

    def point_within_bounds(
        self, p: npt.NDArray[np.int32], cmd_str: str, name: str = "", ignore_error=False
    ) -> bool:
        try:
            return self.grid.within_bounds(p)
        except ValueError as err:
            if ignore_error:
                return False

            v = ["x", "y", "z"]
            # Discretisation
            dl = getattr(self.grid, f"d{err.args[0]}")
            # Incorrect index
            i = p[v.index(err.args[0])]
            if name:
                s = f"\n'{cmd_str}' {err.args[0]} {name}-coordinate {i * dl:g} is not within the model domain"
            else:
                s = f"\n'{cmd_str}' {err.args[0]}-coordinate {i * dl:g} is not within the model domain"
            logger.exception(s)
            raise

    def grid_upper_bound(self) -> list[int]:
        return [self.grid.nx, self.grid.ny, self.grid.nz]

    def discretise_point(self, p: Tuple[float, float, float]) -> npt.NDArray[np.int32]:
        """Gets the index of a continuous point with the grid."""
        rv = np.vectorize(round_int, otypes=[np.int32])
        return rv(p / self.grid.dl)

    def round_to_grid(self, p: Tuple[float, float, float]) -> npt.NDArray[np.float64]:
        """Gets the nearest continuous point on the grid from a continuous point
        in space.
        """
        return self.discretise_point(p) * self.grid.dl

    def discretised_to_continuous(self, p: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
        """Returns a point given as indices to a continuous point in the real space."""
        return p * self.grid.dl


class MainGridUserInput(UserInput[GridType]):
    """Handles (x, y, z) points supplied by the user in the main grid."""

    def __init__(self, grid):
        super().__init__(grid)

    def check_point(self, p, cmd_str, name=""):
        """Discretises point and check its within the domain"""
        p = self.discretise_point(p)
        self.point_within_bounds(p, cmd_str, name)
        return p

    def check_src_rx_point(self, p: npt.NDArray[np.int32], cmd_str: str, name: str = "") -> bool:
        within_grid = self.point_within_bounds(p, cmd_str, name)

        if self.grid.within_pml(p):
            logger.warning(
                f"'{cmd_str}' sources and receivers should not normally be positioned within the PML."
            )

        return within_grid

    def check_box_points(self, p1, p2, cmd_str):
        p1 = self.check_point(p1, cmd_str, name="lower")
        p2 = self.check_point(p2, cmd_str, name="upper")

        if np.greater(p1, p2).any():
            logger.exception(
                f"'{cmd_str}' the lower coordinates should be less than the upper coordinates."
            )
            raise ValueError

        return p1, p2

    def check_tri_points(self, p1, p2, p3, cmd_str):
        p1 = self.check_point(p1, cmd_str, name="vertex_1")
        p2 = self.check_point(p2, cmd_str, name="vertex_2")
        p3 = self.check_point(p3, cmd_str, name="vertex_3")

        return p1, p2, p3

    def discretise_static_point(self, p):
        """Gets the index of a continuous point regardless of the point of
        origin of the grid.
        """
        return super().discretise_point(p)

    def round_to_grid_static_point(self, p):
        """Gets the index of a continuous point regardless of the point of
        origin of the grid.
        """
        return super().discretise_point(p) * self.grid.dl


class SubgridUserInput(MainGridUserInput[SubGridBaseGrid]):
    """Handles (x, y, z) points supplied by the user in the subgrid.
    This class autotranslates points from main grid to subgrid equivalent
    (within IS). Useful if material traverse is not required.
    """

    def __init__(self, grid):
        super().__init__(grid)

        # Defines the region exposed to the user
        self.inner_bound = np.array(
            [grid.n_boundary_cells_x, grid.n_boundary_cells_y, grid.n_boundary_cells_z]
        )

        self.outer_bound = np.subtract([grid.nx, grid.ny, grid.nz], self.inner_bound)

    def translate_to_gap(self, p) -> npt.NDArray[np.int32]:
        """Translates the user input point to the real point in the subgrid."""

        p1 = (p[0] - self.grid.i0 * self.grid.ratio) + self.grid.n_boundary_cells_x
        p2 = (p[1] - self.grid.j0 * self.grid.ratio) + self.grid.n_boundary_cells_y
        p3 = (p[2] - self.grid.k0 * self.grid.ratio) + self.grid.n_boundary_cells_z

        return np.array([p1, p2, p3])

    def discretise_point(self, p) -> npt.NDArray[np.int32]:
        """Discretises a point. Does not provide any checks. The user enters
        coordinates relative to self.inner_bound. This function translate
        the user point to the correct index for building objects.
        """

        p = super().discretise_point(p)
        p_t = self.translate_to_gap(p)
        return p_t

    def round_to_grid(self, p):
        p_t = self.discretise_point(p)
        p_m = p_t * self.grid.dl
        return p_m

    def check_point(self, p, cmd_str, name=""):
        p_t = super().check_point(p, cmd_str, name)

        # Provide user within a warning if they have placed objects within
        # the OS non-working region.
        if np.less(p_t, self.inner_bound).any() or np.greater(p_t, self.outer_bound).any():
            logger.warning(
                f"'{cmd_str}' this object traverses the Outer Surface. This is an advanced feature."
            )
        return p_t

    def discretise_static_point(self, p):
        """Gets the index of a continuous point regardless of the point of
        origin of the grid."""
        return super().discretise_point(p)

    def round_to_grid_static_point(self, p):
        """Gets the index of a continuous point regardless of the point of
        origin of the grid."""
        return super().discretise_point(p) * self.grid.dl
