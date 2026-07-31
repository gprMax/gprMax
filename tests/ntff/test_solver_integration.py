from types import SimpleNamespace

import numpy as np

from gprMax.solvers import Solver
from gprMax.updates.cpu_updates import CPUUpdates


def test_solver_calls_ntff_observers_at_completed_e_and_h_time_levels():
    calls = []
    updates = CPUUpdates.__new__(CPUUpdates)
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
        "observe_ntff_magnetic",
        "update_electric_a",
        "update_symmetry_boundaries_electric",
        "update_electric_pml",
        "update_electric_sources",
        "update_eigenmode_sources_electric",
        "update_plane_waves_electric",
        "update_symmetry_boundaries_electric_b",
        "update_electric_b",
        "finalise",
        "calculate_solve_time",
        "cleanup",
    ):
        setattr(updates, name, record(name))
    updates.calculate_solve_time = lambda: 0.0

    Solver(updates).solve(range(1))

    assert calls.index("observe_ntff_electric") > calls.index("store_snapshots")
    assert calls.index("observe_ntff_electric") < calls.index("update_magnetic")
    assert calls.index("update_magnetic_sources") < calls.index(
        "update_eigenmode_sources_magnetic"
    )
    assert calls.index("update_eigenmode_sources_magnetic") < calls.index(
        "update_plane_waves_magnetic"
    )
    assert calls.index("observe_ntff_magnetic") > calls.index("update_plane_waves_magnetic")
    assert calls.index("observe_ntff_magnetic") < calls.index("update_electric_a")
    assert calls.index("update_electric_sources") < calls.index(
        "update_eigenmode_sources_electric"
    )
    assert calls.index("update_eigenmode_sources_electric") < calls.index(
        "update_plane_waves_electric"
    )
    assert calls.index("finalise") > calls.index("update_electric_b")


def test_cpu_updates_forward_host_fields_to_ntff_monitor():
    calls = []

    class Monitor:
        def observe_electric(self, iteration, *fields):
            calls.append(("E", iteration, fields))

        def observe_magnetic(self, iteration, *fields):
            calls.append(("H", iteration, fields))

        def finalise(self):
            calls.append(("finalise",))

    arrays = [np.full((2, 2, 2), value) for value in range(6)]
    grid = SimpleNamespace(
        Ex=arrays[0],
        Ey=arrays[1],
        Ez=arrays[2],
        Hx=arrays[3],
        Hy=arrays[4],
        Hz=arrays[5],
        ntff_monitors=[Monitor()],
    )
    updates = CPUUpdates.__new__(CPUUpdates)
    updates.grid = grid

    updates.observe_ntff_electric(4)
    updates.observe_ntff_magnetic(4)
    updates.finalise()

    assert calls[0][:2] == ("E", 4)
    assert all(actual is expected for actual, expected in zip(calls[0][2], arrays[:3]))
    assert calls[1][:2] == ("H", 4)
    assert all(actual is expected for actual, expected in zip(calls[1][2], arrays[3:]))
    assert calls[2] == ("finalise",)
