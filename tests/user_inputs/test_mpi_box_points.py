# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.user_inputs import MainGridUserInput, MPIUserInput
from gprMax.utilities.mpi import Dim, Dir


def _input(monkeypatch, lower, upper, positive_neighbours=()):
    lower = np.asarray(lower, dtype=np.int32)
    upper = np.asarray(upper, dtype=np.int32)

    monkeypatch.setattr(
        MainGridUserInput,
        "check_box_points",
        lambda self, p1, p2, cmd_str: (True, lower.copy(), upper.copy()),
    )

    neighbours = np.full((3, 2), -1, dtype=np.int32)
    for dimension in positive_neighbours:
        neighbours[dimension, Dir.POS] = 1

    grid = SimpleNamespace(
        size=np.asarray((7, 7, 7), dtype=np.int32),
        neighbours=neighbours,
    )
    grid.has_neighbour = lambda dimension, direction: (grid.neighbours[dimension, direction] >= 0)
    return MPIUserInput(grid)


@pytest.mark.parametrize("normal", list(Dim))
def test_global_max_zero_thickness_face_belongs_to_terminal_rank(monkeypatch, normal):
    lower = np.asarray((2, 2, 2), dtype=np.int32)
    upper = np.asarray((5, 5, 5), dtype=np.int32)
    lower[normal] = 7
    upper[normal] = 7
    user_input = _input(monkeypatch, lower, upper)

    within_grid, local_lower, local_upper = user_input.check_box_points((), (), "#plate")

    assert within_grid
    np.testing.assert_array_equal(local_lower, lower)
    np.testing.assert_array_equal(local_upper, upper)


@pytest.mark.parametrize("normal", list(Dim))
def test_internal_positive_zero_thickness_face_belongs_to_next_rank(monkeypatch, normal):
    lower = np.asarray((2, 2, 2), dtype=np.int32)
    upper = np.asarray((5, 5, 5), dtype=np.int32)
    lower[normal] = 7
    upper[normal] = 7
    user_input = _input(monkeypatch, lower, upper, positive_neighbours=(normal,))

    within_grid, _, _ = user_input.check_box_points((), (), "#plate")

    assert not within_grid


@pytest.mark.parametrize("run_axis", list(Dim))
def test_global_max_edge_belongs_to_terminal_rank(monkeypatch, run_axis):
    lower = np.full(3, 7, dtype=np.int32)
    upper = np.full(3, 7, dtype=np.int32)
    lower[run_axis] = 2
    upper[run_axis] = 5
    user_input = _input(monkeypatch, lower, upper)

    within_grid, local_lower, local_upper = user_input.check_box_points((), (), "#magnetic_edge")

    assert within_grid
    np.testing.assert_array_equal(local_lower, lower)
    np.testing.assert_array_equal(local_upper, upper)


@pytest.mark.parametrize("run_axis", list(Dim))
def test_global_max_edge_is_rejected_by_non_terminal_rank(monkeypatch, run_axis):
    lower = np.full(3, 7, dtype=np.int32)
    upper = np.full(3, 7, dtype=np.int32)
    lower[run_axis] = 2
    upper[run_axis] = 5
    transverse_axis = next(dimension for dimension in Dim if dimension != run_axis)
    user_input = _input(
        monkeypatch,
        lower,
        upper,
        positive_neighbours=(transverse_axis,),
    )

    within_grid, _, _ = user_input.check_box_points((), (), "#magnetic_edge")

    assert not within_grid


def test_non_degenerate_object_outside_local_grid_is_rejected(monkeypatch):
    user_input = _input(monkeypatch, (-4, 2, 2), (0, 5, 5))

    within_grid, _, _ = user_input.check_box_points((), (), "#box")

    assert not within_grid
