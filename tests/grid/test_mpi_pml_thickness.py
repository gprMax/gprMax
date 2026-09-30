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

"""Unit tests for MPI PML local thickness validation."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from mpi4py import MPI

from gprMax.grid.mpi_grid import MPIGrid

pytestmark = pytest.mark.unit


def _create_mock_mpi_grid(local_size, thickness_dict):
    grid = object.__new__(MPIGrid)
    grid.size = np.array(local_size)
    grid.global_size = np.array([local_size[0] * 3, local_size[1], local_size[2]])
    grid.mpi_tasks = np.array([3, 1, 1])
    grid.pmls = {"thickness": thickness_dict}
    grid.set_halo_map = MagicMock()
    grid.comm = MagicMock()
    return grid


def test_mpi_grid_build_rejects_pml_exceeding_local_dimension():
    grid = _create_mock_mpi_grid(
        [10, 50, 50],
        {"x0": 20, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0},
    )
    grid.comm.allreduce.return_value = 1

    with pytest.raises(ValueError, match="PML has too many cells for the domain size"):
        grid.build()

    grid.comm.allreduce.assert_called_once_with(1, op=MPI.MAX)


def test_mpi_grid_build_accepts_pml_fitting_local_dimension():
    grid = _create_mock_mpi_grid(
        [30, 50, 50],
        {"x0": 20, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0},
    )
    grid.comm.allreduce.return_value = 0

    with patch("gprMax.grid.fdtd_grid.FDTDGrid.build"):
        grid.build()

    grid.comm.allreduce.assert_called_once_with(0, op=MPI.MAX)


def test_pml_free_internal_rank_contributes_zero_to_allreduce():
    grid = _create_mock_mpi_grid(
        [20, 50, 50],
        {"x0": 0, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0},
    )
    grid.comm.allreduce.return_value = 0

    with patch("gprMax.grid.fdtd_grid.FDTDGrid.build"):
        grid.build()

    grid.comm.allreduce.assert_called_once_with(0, op=MPI.MAX)


def test_locally_valid_rank_raises_when_remote_rank_reports_error():
    grid = _create_mock_mpi_grid(
        [20, 50, 50],
        {"x0": 0, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0},
    )
    grid.comm.allreduce.return_value = 1

    with pytest.raises(ValueError, match="PML has too many cells for the domain size"):
        grid.build()

    grid.comm.allreduce.assert_called_once_with(0, op=MPI.MAX)


@pytest.mark.parametrize("axis", ("y", "z"))
def test_opposing_pmls_on_undecomposed_axis_use_shared_sum_check(axis):
    thickness = dict.fromkeys(("x0", "xmax", "y0", "ymax", "z0", "zmax"), 0)
    thickness[f"{axis}0"] = thickness[f"{axis}max"] = 10
    grid = _create_mock_mpi_grid([20, 20, 20], thickness)
    grid.comm.allreduce.return_value = 1

    with pytest.raises(ValueError, match="PML has too many cells for the domain size"):
        grid.build()

    # Neither slab individually fills the dimension; their sum does.
    grid.comm.allreduce.assert_called_once_with(1, op=MPI.MAX)


def test_failed_build_does_not_cache_successful_validation():
    grid = _create_mock_mpi_grid(
        [12, 36, 36],
        {"x0": 13, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0},
    )
    grid.comm.allreduce.side_effect = lambda value, op: value

    for _ in range(2):
        with pytest.raises(ValueError, match="PML has too many cells for the domain size"):
            grid.build()

    assert grid.comm.allreduce.call_args_list == [call(1, op=MPI.MAX)] * 2


def test_each_build_revalidates_changed_thickness():
    grid = _create_mock_mpi_grid(
        [12, 36, 36],
        {"x0": 4, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0},
    )
    grid.comm.allreduce.side_effect = lambda value, op: value

    with patch("gprMax.grid.fdtd_grid.FDTDGrid.build"):
        grid.build()
        grid.pmls["thickness"]["x0"] = 13
        with pytest.raises(ValueError, match="PML has too many cells for the domain size"):
            grid.build()

    assert grid.comm.allreduce.call_args_list == [call(0, op=MPI.MAX), call(1, op=MPI.MAX)]
