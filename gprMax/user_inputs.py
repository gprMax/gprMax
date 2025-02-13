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
from gprMax.grid.mpi_grid import MPIGrid
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

GridType = TypeVar("GridType", bound=FDTDGrid)


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

    def discretise_static_point(self, point: Tuple[float, float, float]) -> npt.NDArray[np.int32]:
        """Get the nearest grid index to a continuous static point.

        For a static point, the point of the origin of the grid is
        ignored. I.e. it is assumed to be at (0, 0, 0). There are no
        checks of the validity of the point such as bound checking.

        Args:
            point: x, y, z coordinates of the point in space.

        Returns:
            discretised_point: x, y, z indices of the point on the grid.
        """
        rv = np.vectorize(round_int, otypes=[np.int32])
        return rv(point / self.grid.dl)

    def round_to_grid_static_point(
        self, point: Tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Round a continuous static point to the nearest point on the grid.

        For a static point, the point of the origin of the grid is
        ignored. I.e. it is assumed to be at (0, 0, 0). There are no
        checks of the validity of the point such as bound checking.

        Args:
            point: x, y, z coordinates of the point in space.

        Returns:
            rounded_point: x, y, z coordinates of the nearest continuous
                point on the grid.
        """
        return self.discretise_static_point(point) * self.grid.dl

    def discretise_point(self, point: Tuple[float, float, float]) -> npt.NDArray[np.int32]:
        """Get the nearest grid index to a continuous static point.

        This function translates user points to the correct index for
        building objects. Points will be mapped from the user coordinate
        space to the local coordinate space of the grid. There are no
        checks of the validity of the point such as bound checking.

        Args:
            point: x, y, z coordinates of the point in space.

        Returns:
            discretised_point: x, y, z indices of the point on the grid.
        """
        return self.discretise_static_point(point)

    def round_to_grid(self, point: Tuple[float, float, float]) -> npt.NDArray[np.float64]:
        """Round a continuous static point to the nearest point on the grid.

        The point will be mapped from the user coordinate space to the
        local coordinate space of the grid. There are no checks of the
        validity of the point such as bound checking.

        Args:
            point: x, y, z coordinates of the point in space.

        Returns:
            rounded_point: x, y, z coordinates of the nearest continuous
                point on the grid.
        """
        return self.discretise_point(point) * self.grid.dl


class MainGridUserInput(UserInput[GridType]):
    """Handles (x, y, z) points supplied by the user in the main grid."""

    def __init__(self, grid):
        super().__init__(grid)

    def check_point(
        self, point: Tuple[float, float, float], cmd_str: str, name: str = ""
    ) -> Tuple[bool, npt.NDArray[np.int32]]:
        """Discretises point and check its within the domain"""
        discretised_point = self.discretise_point(point)
        within_bounds = self.point_within_bounds(discretised_point, cmd_str, name)
        return within_bounds, discretised_point

    def check_src_rx_point(
        self, point: Tuple[float, float, float], cmd_str: str, name: str = ""
    ) -> Tuple[bool, npt.NDArray[np.int32]]:
        within_bounds, discretised_point = self.check_point(point, cmd_str, name)

        if self.grid.within_pml(discretised_point):
            logger.warning(
                f"'{cmd_str}' sources and receivers should not normally be positioned within the PML."
            )

        return within_bounds, discretised_point

    def _check_2d_points(
        self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], cmd_str: str
    ) -> Tuple[bool, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        lower_within_grid, lower_point = self.check_point(p1, cmd_str, "lower")
        upper_within_grid, upper_point = self.check_point(p2, cmd_str, "upper")

        if np.greater(lower_point, upper_point).any():
            raise ValueError(
                f"'{cmd_str}' the lower coordinates should be less than the upper coordinates."
            )

        return lower_within_grid and upper_within_grid, lower_point, upper_point

    def check_output_object_bounds(
        self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], cmd_str: str
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        # We only care if the bounds are in the global grid (an error
        # will be thrown if that is not the case).
        _, lower_bound, upper_bound = self._check_2d_points(p1, p2, cmd_str)
        return lower_bound, upper_bound

    def check_box_points(
        self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], cmd_str: str
    ) -> Tuple[bool, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        return self._check_2d_points(p1, p2, cmd_str)

    def check_tri_points(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        p3: Tuple[float, float, float],
        cmd_str: str,
    ) -> Tuple[bool, npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        p1_within_grid, p1_checked = self.check_point(p1, cmd_str, name="vertex_1")
        p2_within_grid, p2_checked = self.check_point(p2, cmd_str, name="vertex_2")
        p3_within_grid, p3_checked = self.check_point(p3, cmd_str, name="vertex_3")

        return (
            p1_within_grid and p2_within_grid and p3_within_grid,
            p1_checked,
            p2_checked,
            p3_checked,
        )

    def check_thickness(
        self, dimension: str, start: float, thickness: float
    ) -> Tuple[bool, float, float]:
        """Check the thickness of an object in a specified dimension.

        Args:
            dimension: Dimension to check the thickness value for.
                This must have value x, y, or z.
            start: Start coordinate of the object in the specified
                dimension. This must be in the local grid coordinate
                system - i.e. previously translated using the
                round_to_grid function.
            thickness: Thickness of the object.

        Raises:
            ValueError: Raised if dimension has an invalid value.

        Returns:
            within_grid: True if part of the object is within the
                current grid. False otherwise.
            start: Start value limited to the bounds of the grid.
            thickness: Thickness value such that start + thickness is
                within the bounds of the grid.
        """
        if dimension == "x":
            grid_size = self.grid.nx * self.grid.dx
        elif dimension == "y":
            grid_size = self.grid.ny * self.grid.dy
        elif dimension == "z":
            grid_size = self.grid.nz * self.grid.dz
        else:
            raise ValueError("Dimension should have value x, y, or z")

        end = start + thickness

        if start > grid_size:
            return False, start, thickness
        elif end < 0:
            return False, start, thickness

        if start < 0:
            start = 0
        if end > grid_size:
            thickness = grid_size - start

        return True, start, thickness


class MPIUserInput(MainGridUserInput[MPIGrid]):
    """Handles (x, y, z) points supplied by the user for MPI grids.

    This class autotranslates points from the global coordinate system
    to the grid's local coordinate system.
    """

    def discretise_point(self, point: Tuple[float, float, float]) -> npt.NDArray[np.int32]:
        """Get the nearest grid index to a continuous static point.

        This function translates user points to the correct index for
        building objects. Points will be mapped from the global
        coordinate space to the local coordinate space of the grid.
        There are no checks of the validity of the point such as bound
        checking.

        Args:
            point: x, y, z coordinates of the point in space.

        Returns:
            discretised_point: x, y, z indices of the point on the grid.
        """
        discretised_point = super().discretise_point(point)
        return self.grid.global_to_local_coordinate(discretised_point)

    def check_box_points(
        self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], cmd_str: str
    ) -> Tuple[bool, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        _, lower_point, upper_point = super().check_box_points(p1, p2, cmd_str)

        # Restrict points to the bounds of the local grid
        lower_point = np.where(lower_point < 0, 0, lower_point)
        upper_point = np.where(upper_point > self.grid.size, self.grid.size, upper_point)

        return all(lower_point <= upper_point), lower_point, upper_point


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

    def discretise_point(self, point: Tuple[float, float, float]) -> npt.NDArray[np.int32]:
        """Get the nearest grid index to a continuous static point.

        This function translates user points to the correct index for
        building objects. The user enters coordinates relative to
        self.inner_bound which are mapped to the local coordinate space
        of the grid. There are no checks of the validity of the point
        such as bound checking.

        Args:
            point: x, y, z coordinates of the point in space relative to
                self.inner_bound.

        Returns:
            discretised_point: x, y, z indices of the point on the grid.
        """

        discretised_point = super().discretise_point(point)
        discretised_point = self.translate_to_gap(discretised_point)
        return discretised_point

    def check_point(self, p, cmd_str, name=""):
        p_t = super().check_point(p, cmd_str, name)

        # Provide user within a warning if they have placed objects within
        # the OS non-working region.
        if np.less(p_t, self.inner_bound).any() or np.greater(p_t, self.outer_bound).any():
            logger.warning(
                f"'{cmd_str}' this object traverses the Outer Surface. This is an advanced feature."
            )
        return p_t
