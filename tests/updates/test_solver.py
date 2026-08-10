"""``Solver`` and ``create_solver`` — ``gprMax/solvers.py``.

This file holds the **running order of an FDTD timestep**, and it holds it
nowhere else. There is no "advance one step" method on ``CPUUpdates``; the
sequence exists only as eleven consecutive lines inside ``Solver.solve``.

That matters because the Yee scheme leapfrogs. Electric and magnetic fields
are staggered half a timestep apart and each is computed from the curl of the
other, so the order is the algorithm. Swap two calls and the simulation still
runs, still terminates, still writes a well-formed output file — with wrong
numbers. It is the archetype of the failure this project exists to catch, and
until now nothing asserted it.

Testing it needs neither a grid nor a kernel. ``Solver`` only ever calls
methods on the object it was handed, so a recorder that appends its own name
is a complete stand-in, and one iteration of the loop yields the sequence as
a plain list.

``create_solver`` is the other half: a six-way dispatch that decides which
backend a model gets. It uses ``type(grid) is FDTDGrid`` — **exact type
identity, not** ``isinstance`` — so a subclass falls through to a bare
``raise ValueError``.
"""

from types import SimpleNamespace

import pytest

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.solvers import Solver, create_solver
from gprMax.updates.cpu_updates import CPUUpdates
from gprMax.updates.updates import Updates

# The canonical per-iteration sequence for a plain CPU run, in the order
# ``Solver.solve`` issues it. Stated once, asserted several ways below.
CPU_ITERATION_ORDER = [
    "store_outputs",
    "store_snapshots",
    "observe_ntff_electric",
    "update_magnetic",
    "update_magnetic_pml",
    "update_magnetic_sources",
    "update_eigenmode_sources_magnetic",
    "update_plane_waves_magnetic",
    "observe_eigenmode_ports",
    "observe_ntff_magnetic",
    "update_electric_a",
    "update_symmetry_boundaries_electric",
    "update_electric_pml",
    "update_electric_sources",
    "update_eigenmode_sources_electric",
    "update_plane_waves_electric",
    "update_symmetry_boundaries_electric_b",
    "update_electric_b",
]

# Called once each, outside the loop.
PROLOGUE = ["time_start"]
EPILOGUE = ["finalise", "calculate_solve_time", "cleanup"]


class RecordingUpdates(CPUUpdates):
    """A ``CPUUpdates`` whose every step records its own name.

    Subclasses the real class rather than faking one, so the
    ``isinstance(self.updates, CPUUpdates)`` guards inside ``Solver.solve``
    take the branches a genuine CPU run would. Nothing is computed — every
    method is replaced by an append.
    """

    def __init__(self):
        super().__init__(SimpleNamespace(ntff_monitors=[]))
        self.calls = []

    def _record(self, name):
        self.calls.append(name)

    def store_outputs(self, iteration):
        self._record("store_outputs")

    def store_snapshots(self, iteration):
        self._record("store_snapshots")

    def update_magnetic(self):
        self._record("update_magnetic")

    def update_magnetic_pml(self):
        self._record("update_magnetic_pml")

    def update_magnetic_sources(self, iteration):
        self._record("update_magnetic_sources")

    def update_plane_waves_magnetic(self, iteration):
        self._record("update_plane_waves_magnetic")

    def update_electric_a(self):
        self._record("update_electric_a")

    def update_electric_pml(self):
        self._record("update_electric_pml")

    def update_electric_sources(self, iteration):
        self._record("update_electric_sources")

    def update_plane_waves_electric(self, iteration):
        self._record("update_plane_waves_electric")

    def observe_ntff_electric(self, iteration):
        self._record("observe_ntff_electric")

    def update_eigenmode_sources_magnetic(self, iteration):
        self._record("update_eigenmode_sources_magnetic")

    def observe_eigenmode_ports(self, iteration):
        self._record("observe_eigenmode_ports")

    def observe_ntff_magnetic(self, iteration):
        self._record("observe_ntff_magnetic")

    def update_symmetry_boundaries_electric(self):
        self._record("update_symmetry_boundaries_electric")

    def update_eigenmode_sources_electric(self, iteration):
        self._record("update_eigenmode_sources_electric")

    def update_symmetry_boundaries_electric_b(self):
        self._record("update_symmetry_boundaries_electric_b")

    def update_electric_b(self):
        self._record("update_electric_b")

    def time_start(self):
        self._record("time_start")

    def calculate_solve_time(self):
        self._record("calculate_solve_time")
        return 1.25

    def finalise(self):
        self._record("finalise")

    def cleanup(self):
        self._record("cleanup")


class MinimalRecordingUpdates(Updates):
    """A conforming backend that is *not* a ``CPUUpdates``.

    Used to show which steps are CPU-only: the two plane-wave calls are
    skipped entirely for a backend the ``isinstance`` guard does not match.
    """

    def __init__(self):
        super().__init__(SimpleNamespace())
        self.calls = []

    def _record(self, name):
        self.calls.append(name)

    def store_outputs(self, iteration):
        self._record("store_outputs")

    def store_snapshots(self, iteration):
        self._record("store_snapshots")

    def update_magnetic(self):
        self._record("update_magnetic")

    def update_magnetic_pml(self):
        self._record("update_magnetic_pml")

    def update_magnetic_sources(self, iteration):
        self._record("update_magnetic_sources")

    def update_electric_a(self):
        self._record("update_electric_a")

    def update_electric_pml(self):
        self._record("update_electric_pml")

    def update_electric_sources(self, iteration):
        self._record("update_electric_sources")

    def observe_ntff_electric(self, iteration):
        self._record("observe_ntff_electric")

    def observe_ntff_magnetic(self, iteration):
        self._record("observe_ntff_magnetic")

    def update_symmetry_boundaries_electric(self):
        self._record("update_symmetry_boundaries_electric")

    def update_symmetry_boundaries_electric_b(self):
        self._record("update_symmetry_boundaries_electric_b")

    def update_electric_b(self):
        self._record("update_electric_b")

    def time_start(self):
        self._record("time_start")

    def calculate_solve_time(self):
        self._record("calculate_solve_time")
        return 0.0

    # Overridden purely so the recorder can see them.
    # not override them inherits the base-class no-ops instead, which
    # ``test_a_backend_inheriting_the_default_hooks_still_solves`` covers.
    def finalise(self):
        self._record("finalise")

    def cleanup(self):
        self._record("cleanup")


@pytest.fixture
def recording_updates():
    """A CPU-flavoured recorder."""
    return RecordingUpdates()


class TestSolverConstruction:
    """``Solver(updates)`` — three attributes, no logic."""

    def test_stores_the_updates_object(self, recording_updates):
        assert Solver(recording_updates).updates is recording_updates

    def test_solve_time_starts_at_zero(self, recording_updates):
        assert Solver(recording_updates).solvetime == 0

    def test_memory_used_starts_at_zero(self, recording_updates):
        assert Solver(recording_updates).memused == 0

    def test_construction_sets_exactly_three_attributes(self, recording_updates):
        assert set(vars(Solver(recording_updates))) == {
            "updates",
            "solvetime",
            "memused",
        }

    def test_construction_does_not_call_the_updates_object(self, recording_updates):
        """Nothing runs until ``solve``."""
        Solver(recording_updates)
        assert recording_updates.calls == []


class TestIterationOrder:
    """The eleven beats, in order. The heart of this file."""

    def test_one_iteration_produces_the_canonical_sequence(self, recording_updates):
        """The full per-iteration order for a CPU run.

        If this test fails after a source change, the physics changed. There
        is no such thing as an incidental reordering here.
        """
        Solver(recording_updates).solve(range(1))

        body = [c for c in recording_updates.calls if c not in PROLOGUE + EPILOGUE]
        assert body == CPU_ITERATION_ORDER

    def test_magnetic_field_is_updated_before_the_electric_field(
        self, recording_updates
    ):
        """The leapfrog: H advances half a step, then E uses the new H."""
        Solver(recording_updates).solve(range(1))
        calls = recording_updates.calls

        assert calls.index("update_magnetic") < calls.index("update_electric_a")

    def test_outputs_are_stored_before_anything_moves(self, recording_updates):
        """Receivers record the state at the *top* of the iteration."""
        Solver(recording_updates).solve(range(1))
        calls = recording_updates.calls

        assert calls.index("store_outputs") < calls.index("update_magnetic")

    def test_snapshots_are_stored_before_anything_moves(self, recording_updates):
        Solver(recording_updates).solve(range(1))
        calls = recording_updates.calls

        assert calls.index("store_snapshots") < calls.index("update_magnetic")

    def test_outputs_are_stored_before_snapshots(self, recording_updates):
        Solver(recording_updates).solve(range(1))
        calls = recording_updates.calls

        assert calls.index("store_outputs") < calls.index("store_snapshots")

    @pytest.mark.parametrize(
        "field,pml,sources",
        [
            ("update_magnetic", "update_magnetic_pml", "update_magnetic_sources"),
            ("update_electric_a", "update_electric_pml", "update_electric_sources"),
        ],
    )
    def test_each_half_step_runs_field_then_pml_then_sources(
        self, recording_updates, field, pml, sources
    ):
        """The PML correction follows the bulk update, then sources inject.

        The PML has to see the field the bulk kernel just produced, and a
        source must be added after both — otherwise the absorbing layer eats
        the excitation on the step it is applied.
        """
        Solver(recording_updates).solve(range(1))
        calls = recording_updates.calls

        assert calls.index(field) < calls.index(pml) < calls.index(sources)

    def test_electric_b_is_the_last_step_of_the_iteration(self, recording_updates):
        """The dispersive closing half runs after everything else.

        It needs both the old and the new electric field, so it cannot run
        until the PML and the sources have finished with E.
        """
        Solver(recording_updates).solve(range(1))

        body = [c for c in recording_updates.calls if c not in PROLOGUE + EPILOGUE]
        assert body[-1] == "update_electric_b"

    def test_electric_b_follows_the_electric_sources(self, recording_updates):
        Solver(recording_updates).solve(range(1))
        calls = recording_updates.calls

        assert calls.index("update_electric_sources") < calls.index("update_electric_b")

    def test_plane_waves_follow_the_discrete_sources_in_each_half(
        self, recording_updates
    ):
        Solver(recording_updates).solve(range(1))
        calls = recording_updates.calls

        assert calls.index("update_magnetic_sources") < calls.index(
            "update_plane_waves_magnetic"
        )
        assert calls.index("update_electric_sources") < calls.index(
            "update_plane_waves_electric"
        )

    def test_there_are_eighteen_steps_in_an_iteration(self, recording_updates):
        Solver(recording_updates).solve(range(1))

        body = [c for c in recording_updates.calls if c not in PROLOGUE + EPILOGUE]
        assert len(body) == 18


class TestLoopBracketing:
    """What happens once, outside the loop."""

    def test_time_start_runs_before_the_first_iteration(self, recording_updates):
        Solver(recording_updates).solve(range(2))

        assert recording_updates.calls[0] == "time_start"

    def test_finalise_then_solve_time_then_cleanup(self, recording_updates):
        """The exact teardown order.

        ``calculate_solve_time`` sits between the two hooks, so ``finalise``
        can flush work that should be counted and ``cleanup`` can release
        resources that should not.
        """
        Solver(recording_updates).solve(range(1))

        assert recording_updates.calls[-3:] == EPILOGUE

    def test_time_start_is_called_exactly_once(self, recording_updates):
        Solver(recording_updates).solve(range(5))

        assert recording_updates.calls.count("time_start") == 1

    @pytest.mark.parametrize("name", EPILOGUE)
    def test_teardown_step_is_called_exactly_once(self, recording_updates, name):
        Solver(recording_updates).solve(range(5))

        assert recording_updates.calls.count(name) == 1

    def test_solve_time_is_stored_on_the_solver(self, recording_updates):
        """The return value of ``calculate_solve_time`` lands in ``solvetime``."""
        solver = Solver(recording_updates)

        solver.solve(range(1))

        assert solver.solvetime == pytest.approx(1.25)

    def test_solve_time_replaces_the_initial_zero(self, recording_updates):
        solver = Solver(recording_updates)
        assert solver.solvetime == 0

        solver.solve(range(1))

        assert solver.solvetime != 0


class TestIterationCount:
    """The loop runs once per item the iterator yields."""

    @pytest.mark.parametrize("iterations", [0, 1, 2, 5, 17])
    def test_body_repeats_once_per_iteration(self, recording_updates, iterations):
        Solver(recording_updates).solve(range(iterations))

        body = [c for c in recording_updates.calls if c not in PROLOGUE + EPILOGUE]
        assert body == CPU_ITERATION_ORDER * iterations

    def test_an_empty_iterator_still_brackets_the_run(self, recording_updates):
        """Zero iterations: the timer and the hooks still fire.

        A ``#time_window`` of zero produces this, and it must not crash.
        """
        Solver(recording_updates).solve(range(0))

        assert recording_updates.calls == PROLOGUE + EPILOGUE

    def test_the_iterator_may_be_any_iterable(self, recording_updates):
        """``solve`` takes ``range()`` or ``tqdm()``; it only iterates.

        The loop variable is passed straight through to ``store_outputs`` and
        the source updates, so a non-range iterable works identically.
        """
        Solver(recording_updates).solve(iter([0, 1, 2]))

        body = [c for c in recording_updates.calls if c not in PROLOGUE + EPILOGUE]
        assert len(body) == 3 * 18

    def test_iteration_values_are_passed_to_the_steps(self):
        """Whatever the iterator yields reaches the iteration-taking steps."""
        seen = []

        class IterationRecorder(RecordingUpdates):
            def store_outputs(self, iteration):
                seen.append(iteration)

        Solver(IterationRecorder()).solve([3, 9, 27])

        assert seen == [3, 9, 27]


class TestBackendSpecificSteps:
    """Which steps a backend gets depends on its type.

    ``Solver.solve`` guards four of its calls with ``isinstance``, because
    the methods are not on the ``Updates`` base class.
    """

    def test_a_non_cpu_backend_skips_the_plane_wave_steps(self):
        """The two plane-wave calls are ``CPUUpdates``-only."""
        updates = MinimalRecordingUpdates()

        Solver(updates).solve(range(1))

        assert "update_plane_waves_electric" not in updates.calls
        assert "update_plane_waves_magnetic" not in updates.calls

    def test_a_non_cpu_backend_still_runs_the_shared_steps(self):
        """Everything on the base class still happens, in the same order.
        CPU-only steps (plane-waves, eigenmode) are excluded."""
        updates = MinimalRecordingUpdates()

        Solver(updates).solve(range(1))

        cpu_only = {"update_plane_waves", "update_eigenmode_sources", "observe_eigenmode_ports"}
        body = [c for c in updates.calls if c not in PROLOGUE + EPILOGUE]
        assert body == [
            step
            for step in CPU_ITERATION_ORDER
            if not any(step.startswith(prefix) for prefix in cpu_only)
        ]

    def test_a_non_cpu_backend_is_still_bracketed(self):
        updates = MinimalRecordingUpdates()

        Solver(updates).solve(range(1))

        assert updates.calls[0] == "time_start"
        assert updates.calls[-3:] == EPILOGUE

    def test_a_backend_inheriting_the_default_hooks_still_solves(self):
        """``finalise`` and ``cleanup`` are optional for a backend.

        ``Solver.solve`` calls both unconditionally, so the base class's
        no-op defaults are what make them optional. A backend that overrides
        neither runs to completion.
        """

        # Restore the genuine base-class no-ops over the recording overrides.
        NoHooks = type(
            "NoHooks",
            (MinimalRecordingUpdates,),
            {"finalise": Updates.finalise, "cleanup": Updates.cleanup},
        )

        updates = NoHooks()
        solver = Solver(updates)
        solver.solve(range(1))

        assert "finalise" not in updates.calls
        assert "cleanup" not in updates.calls
        assert updates.calls[-1] == "calculate_solve_time"

    def test_the_cpu_guard_matches_subclasses(self, recording_updates):
        """``isinstance`` here, not exact type — unlike ``create_solver``.

        ``RecordingUpdates`` subclasses ``CPUUpdates`` and does get the
        plane-wave steps, which is what makes it a faithful stand-in.
        """
        assert isinstance(recording_updates, CPUUpdates)
        assert type(recording_updates) is not CPUUpdates

        Solver(recording_updates).solve(range(1))

        assert "update_plane_waves_electric" in recording_updates.calls


class TestCreateSolver:
    """``create_solver(model)`` — the backend dispatch."""

    @pytest.fixture
    def make_model(self, updates_config):
        """A stand-in ``Model`` exposing only the ``G`` attribute used."""

        def _make(grid, subgrid=False, maxpoles=0):
            updates_config.sim_config.general["subgrid"] = subgrid
            updates_config.model_config.materials["maxpoles"] = maxpoles
            return SimpleNamespace(G=grid)

        return _make

    def test_a_plain_grid_gets_cpu_updates(self, make_model):
        solver = create_solver(make_model(FDTDGrid()))

        assert type(solver.updates) is CPUUpdates

    def test_the_solver_wraps_the_updates_object(self, make_model):
        solver = create_solver(make_model(FDTDGrid()))

        assert isinstance(solver, Solver)
        assert solver.updates.grid is not None

    def test_the_updates_object_holds_the_models_grid(self, make_model):
        grid = FDTDGrid()

        solver = create_solver(make_model(grid))

        assert solver.updates.grid is grid

    def test_a_non_dispersive_model_skips_dispersive_setup(self, make_model):
        """``maxpoles == 0`` leaves the dispersive functions unbound."""
        solver = create_solver(make_model(FDTDGrid(), maxpoles=0))

        assert not hasattr(solver.updates, "dispersive_update_a")

    def test_a_dispersive_model_gets_dispersive_setup(self, make_model, updates_config):
        """``maxpoles != 0`` triggers ``set_dispersive_updates()``.

        This is the only place in production that call is made, which is why
        constructing ``CPUUpdates`` any other way leaves those attributes
        missing.
        """
        updates_config.model_config.materials["dispersivedtype"] = None

        solver = create_solver(make_model(FDTDGrid(), maxpoles=2))

        assert callable(solver.updates.dispersive_update_a)
        assert callable(solver.updates.dispersive_update_b)

    def test_an_unknown_grid_type_raises_value_error(self, make_model):
        """The terminal ``else`` logs and raises — with no message.

        ``raise ValueError`` bare, so the reason exists only in the log. The
        same pattern as ``check_kappamin`` in the PML, recorded in PR 10.
        """
        with pytest.raises(ValueError):
            create_solver(make_model(object()))

    def test_a_subclass_of_fdtd_grid_is_rejected(self, make_model):
        """``type(grid) is FDTDGrid``, so inheritance does not qualify.

        Every branch of the dispatch uses exact type identity. A user or a
        test that subclasses ``FDTDGrid`` to add behaviour gets a bare
        ``ValueError`` rather than the CPU backend, which is surprising —
        subclassing is otherwise how this codebase extends grids
        (``SubGridBaseGrid`` does exactly that).

        Written up in ``notes/bugs/create-solver-exact-type-dispatch.md``.
        """

        class DerivedGrid(FDTDGrid):
            pass

        with pytest.raises(ValueError):
            create_solver(make_model(DerivedGrid()))

    def test_the_error_is_logged_before_raising(self, make_model, caplog):
        """The message the bare ``ValueError`` omits."""
        with caplog.at_level("ERROR", logger="gprMax.solvers"):
            with pytest.raises(ValueError):
                create_solver(make_model(object()))

        assert "Unknown grid type" in caplog.text
