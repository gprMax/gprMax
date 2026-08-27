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
from mpi4py import MPI

from gprMax.contexts import Context, MPIContext
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.utilities.mpi import Dim, Dir


class _RecordingDatatype:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def Free(self):
        self.calls.append(f"free_{self.name}")


def _empty_halo_maps():
    maps = np.empty((3, 2), dtype=object)
    maps.fill(MPI.DATATYPE_NULL)
    return maps


def test_halo_datatype_cleanup_is_complete_and_idempotent():
    calls = []
    send = _empty_halo_maps()
    recv = _empty_halo_maps()
    send[Dim.X][Dir.NEG] = _RecordingDatatype("send", calls)
    recv[Dim.Y][Dir.POS] = _RecordingDatatype("recv", calls)
    grid = SimpleNamespace(
        send_halo_map=send,
        recv_halo_map=recv,
        _halo_maps_initialised=True,
        _halo_maps_freed=False,
        complete_halo_swaps=lambda: calls.append("complete_halo_swaps"),
    )

    MPIGrid.free_halo_maps(grid)
    MPIGrid.free_halo_maps(grid)

    assert calls == ["complete_halo_swaps", "free_send", "free_recv"]
    assert grid._halo_maps_freed
    assert all(datatype == MPI.DATATYPE_NULL for datatype in send.flat)
    assert all(datatype == MPI.DATATYPE_NULL for datatype in recv.flat)


def test_mpi_context_retires_geometry_fixed_grid_after_all_runs(monkeypatch):
    calls = []
    grid = SimpleNamespace(free_halo_maps=lambda: calls.append("free_halo_maps"))
    context = MPIContext.__new__(MPIContext)
    context.model = SimpleNamespace(G=grid)
    context.rank = 0
    context.comm = SimpleNamespace(Barrier=lambda: calls.append("barrier"))

    monkeypatch.setattr(Context, "run", lambda self: calls.append("run_all_models") or {})

    assert context.run() == {}
    assert calls == ["run_all_models", "free_halo_maps", "barrier"]


def test_gathered_output_state_restores_rank_local_geometry_after_error():
    receiver = SimpleNamespace(coord=np.array((2, 3, 4), dtype=np.int32))
    source = SimpleNamespace(coord=np.array((5, 6, 7), dtype=np.int32))
    grid = SimpleNamespace(
        size=np.array((8, 9, 10), dtype=np.int32),
        global_size=np.array((16, 9, 10), dtype=np.int32),
        rxs=[receiver],
        voltagesources=[],
        magneticdipoles=[],
        hertziandipoles=[source],
        transmissionlines=[],
        magneticfrillsources=[],
        networkterminals=[],
        port_monitors=[],
    )
    local_rxs = grid.rxs
    local_sources = grid.hertziandipoles
    local_receiver_coord = receiver.coord
    local_source_coord = source.coord
    local_size = grid.size

    def gather_grid_objects():
        receiver.coord = receiver.coord + 10
        source.coord = source.coord + 10
        grid.rxs = [SimpleNamespace(coord=np.array((12, 13, 14)))]

    grid.gather_grid_objects = gather_grid_objects

    with pytest.raises(RuntimeError, match="write failed"):
        with MPIGrid.gathered_output_state(grid):
            grid.size = grid.global_size
            raise RuntimeError("write failed")

    assert grid.rxs is local_rxs
    assert grid.hertziandipoles is local_sources
    assert receiver.coord is local_receiver_coord
    assert source.coord is local_source_coord
    assert grid.size is local_size
    np.testing.assert_array_equal(receiver.coord, (2, 3, 4))
    np.testing.assert_array_equal(source.coord, (5, 6, 7))
    np.testing.assert_array_equal(grid.size, (8, 9, 10))
