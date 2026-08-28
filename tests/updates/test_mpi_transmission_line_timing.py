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

from gprMax.solvers import Solver
from gprMax.updates.mpi_updates import MPIUpdates


class _RecordingSource:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def update_magnetic(self, *args):
        self.calls.append(self.name)

    def update_magnetic_mpi(self, *args):
        self.calls.append(self.name)


def test_mpi_separates_magnetic_writers_from_transmission_line_sampling():
    calls = []
    grid = SimpleNamespace(
        transmissionlines=[_RecordingSource("transmission_line", calls)],
        magneticdipoles=[_RecordingSource("magnetic_dipole", calls)],
        magneticfrillsources=[_RecordingSource("magnetic_frill", calls)],
        updatecoeffsH=None,
        ID=None,
        Hx=None,
        Hy=None,
        Hz=None,
    )
    updates = MPIUpdates.__new__(MPIUpdates)
    updates.grid = grid

    updates.update_magnetic_sources(3)
    assert calls == ["magnetic_dipole", "magnetic_frill"]

    updates.update_magnetic_edge_devices(3)
    assert calls == ["magnetic_dipole", "magnetic_frill", "transmission_line"]


def test_solver_samples_mpi_transmission_lines_after_magnetic_halo():
    calls = []
    updates = MPIUpdates.__new__(MPIUpdates)
    updates.grid = SimpleNamespace(iterations=1)

    def record(name):
        return lambda *args, **kwargs: calls.append(name)

    for name in (
        "time_start",
        "store_outputs",
        "store_snapshots",
        "observe_ntff_electric",
        "update_magnetic",
        "update_magnetic_pml",
        "update_magnetic_sources",
        "update_eigenmode_sources_magnetic",
        "update_plane_waves_magnetic",
        "observe_eigenmode_ports",
        "halo_swap_magnetic",
        "update_magnetic_edge_devices",
        "observe_ntff_magnetic",
        "update_electric_a",
        "update_symmetry_boundaries_electric",
        "update_electric_pml",
        "update_electric_sources",
        "update_eigenmode_sources_electric",
        "update_plane_waves_electric",
        "update_symmetry_boundaries_electric_b",
        "update_electric_b",
        "update_network_terminals",
        "halo_swap_electric",
        "finalise",
        "cleanup",
    ):
        setattr(updates, name, record(name))
    updates.calculate_solve_time = lambda: 0.0

    Solver(updates).solve(range(1))

    assert calls.index("update_magnetic_sources") < calls.index("halo_swap_magnetic")
    assert calls.index("halo_swap_magnetic") < calls.index("update_magnetic_edge_devices")
    assert calls.index("update_magnetic_edge_devices") < calls.index("update_electric_a")
    assert calls.index("update_electric_a") < calls.index("update_symmetry_boundaries_electric")
    assert calls.index("update_symmetry_boundaries_electric") < calls.index("update_electric_pml")
    assert calls.index("update_plane_waves_electric") < calls.index(
        "update_symmetry_boundaries_electric_b"
    )
    assert calls.index("update_symmetry_boundaries_electric_b") < calls.index("update_electric_b")


def test_mpi_finalise_completes_last_halo_exchange():
    calls = []
    updates = MPIUpdates.__new__(MPIUpdates)
    updates.grid = SimpleNamespace(
        complete_halo_swaps=lambda: calls.append("complete_halo_swaps"),
        ntff_monitors=[],
    )

    updates.finalise()

    assert calls == ["complete_halo_swaps"]
