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


def test_three_rank_decomposition_with_pmls_only_at_physical_ends():
    def run_three_ranks(rank0_x0_thickness):
        r0 = _create_mock_mpi_grid(
            [20, 50, 50],
            {"x0": rank0_x0_thickness, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0},
        )
        r1 = _create_mock_mpi_grid(
            [20, 50, 50],
            {"x0": 0, "xmax": 0, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0},
        )
        r2 = _create_mock_mpi_grid(
            [20, 50, 50],
            {"x0": 0, "xmax": 10, "y0": 0, "ymax": 0, "z0": 0, "zmax": 0},
        )

        ranks = [r0, r1, r2]
        payloads = {}

        def make_allreduce(rank_id):
            def _allreduce(sendbuf, op=MPI.MAX):
                payloads[rank_id] = sendbuf
                if len(payloads) == 3:
                    return max(payloads.values())
                return None

            return _allreduce

        for i, r in enumerate(ranks):
            r._local_validate = lambda rank_obj: (
                0
                if all(v == 0 for v in rank_obj.pmls["thickness"].values())
                else int(
                    rank_obj.pmls["thickness"]["x0"] + rank_obj.pmls["thickness"]["xmax"]
                    >= rank_obj.nx
                )
            )
            payloads[i] = r._local_validate(r)
            r.comm.allreduce.side_effect = lambda sendbuf, op=MPI.MAX: max(payloads.values())

        return ranks, payloads

    # Valid scenario: all ranks pass
    ranks_valid, payloads_valid = run_three_ranks(rank0_x0_thickness=10)
    assert payloads_valid == {0: 0, 1: 0, 2: 0}
    with patch("gprMax.grid.fdtd_grid.FDTDGrid.build"):
        for r in ranks_valid:
            r.build()

    # Invalid scenario: rank 0 has oversized PML, all 3 ranks reject
    ranks_invalid, payloads_invalid = run_three_ranks(rank0_x0_thickness=25)
    assert payloads_invalid == {0: 1, 1: 0, 2: 0}
    for r in ranks_invalid:
        with pytest.raises(ValueError, match="PML has too many cells for the domain size"):
            r.build()
