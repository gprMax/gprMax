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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gprMax.grid.mpi_grid import MPIGrid

pytestmark = pytest.mark.unit


def test_mpi_grid_build_rejects_pml_exceeding_local_dimension():
    grid = object.__new__(MPIGrid)
    grid.size = np.array([10, 50, 50])
    grid.global_size = np.array([100, 50, 50])
    grid.mpi_tasks = np.array([10, 1, 1])
    grid.pmls = {"thickness": {"x0": 20, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0}}
    grid.set_halo_map = MagicMock()
    grid.comm = MagicMock()
    grid.comm.allreduce.return_value = 1

    with pytest.raises(
        ValueError, match="cannot be greater than or equal to the local MPI grid dimension"
    ):
        grid.build()


def test_mpi_grid_build_accepts_pml_fitting_local_dimension():
    grid = object.__new__(MPIGrid)
    grid.size = np.array([30, 50, 50])
    grid.global_size = np.array([100, 50, 50])
    grid.mpi_tasks = np.array([3, 1, 1])
    grid.pmls = {"thickness": {"x0": 20, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0}}
    grid.set_halo_map = MagicMock()
    grid.comm = MagicMock()
    grid.comm.allreduce.return_value = 0

    with patch("gprMax.grid.fdtd_grid.FDTDGrid.build"):
        grid.build()
