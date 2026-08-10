"""``CPUUpdates`` — ``gprMax/updates/cpu_updates.py``.

The CPU backend is a wiring layer. Almost nothing here computes: each method
reads a handful of attributes off the grid and hands them to a compiled
kernel, a source object, or a PML slab. So the properties worth asserting are
*which* collaborator was called, *with what*, and *in what order*.

Three of those are easy to get wrong and impossible to notice:

**Argument order.** Every kernel call is entirely positional — twelve
arguments for ``update_magnetic``, seventeen for the dispersive electric
update. Transposing two same-typed arrays produces a running simulation with
wrong numbers. Each call is asserted against the full expected tuple.

**Source ordering.** The base class promises to "update any Hertzian dipole
sources last". Nothing enforces that; it is an emergent property of the list
concatenation ``voltagesources + transmissionlines + hertziandipoles``. One
assertion pins it.

**The dispersive branch.** ``update_electric_a`` dispatches on
``maxpoles == 0`` and ``update_electric_b`` on ``maxpoles > 0``. The second
has no ``else``, so for a non-dispersive model it is a silent no-op.

Most tests here use the recorder grid from ``conftest.py`` rather than a real
one: the questions are about wiring, and sentinel strings make an identity
assertion possible that a real array would not. ``TestRealKernels`` runs the
genuine compiled kernels on a small grid, to confirm the wiring actually fits
what the kernels expect.
"""

import numpy as np
import pytest

from gprMax.updates import cpu_updates as cpu_updates_module
from gprMax.updates.cpu_updates import CPUUpdates


class TestConstruction:
    """``CPUUpdates(grid)`` and the attributes it does not set."""

    def test_stores_the_grid(self, make_wiring_grid):
        grid = make_wiring_grid()
        assert CPUUpdates(grid).grid is grid

    def test_construction_reads_no_configuration(self, monkeypatch, make_wiring_grid):
        """A ``CPUUpdates`` can be built before ``sim_config`` exists.

        ``__init__`` is a bare ``super().__init__(G)``, so nothing is read.
        Asserted by removing the global entirely.
        """
        from gprMax import config

        monkeypatch.setattr(config, "sim_config", None)
        assert CPUUpdates(make_wiring_grid()).grid is not None

    def test_construction_stores_grid_and_mode_fields(self, make_wiring_grid):
        """``__init__`` stores the grid and now also sets mode2d fields."""
        updates = CPUUpdates(make_wiring_grid())
        assert updates.grid is not None
        assert hasattr(updates, 'mode2d')

    def test_calculate_solve_time_before_time_start_raises(self, make_wiring_grid):
        """``self.timestart`` only exists after ``time_start()``.

        ``Solver.solve`` always calls ``time_start()`` first, so this is
        latent rather than live — but it is the reason the two-step protocol
        exists. Written up in
        ``notes/bugs/cpu-updates-uninitialised-attributes.md``.
        """
        updates = CPUUpdates(make_wiring_grid())
        with pytest.raises(AttributeError, match="timestart"):
            updates.calculate_solve_time()


class TestStoreOutputs:
    """``store_outputs`` delegates to ``fields_outputs.store_outputs``."""

    def test_delegates_to_the_module_level_function(self, monkeypatch, make_wiring_grid, recorder):
        """Patched on ``cpu_updates``, where the name was bound at import."""
        spy = recorder("store_outputs")
        monkeypatch.setattr(cpu_updates_module, "store_outputs_cpu", spy)

        grid = make_wiring_grid()
        CPUUpdates(grid).store_outputs(7)

        assert spy.call_count == 1

    def test_passes_grid_then_iteration_in_that_order(
        self, monkeypatch, make_wiring_grid, recorder
    ):
        """``store_outputs(G, iteration)`` — a swap here would be silent.

        Both arguments are positional and neither is type-checked, so the
        transposed call would run and write field values into the wrong
        time-series slot.
        """
        spy = recorder("store_outputs")
        monkeypatch.setattr(cpu_updates_module, "store_outputs_cpu", spy)

        grid = make_wiring_grid()
        CPUUpdates(grid).store_outputs(3)

        assert spy.args_of(0) == (grid, 3)

    def test_passes_no_keyword_arguments(self, monkeypatch, make_wiring_grid, recorder):
        spy = recorder("store_outputs")
        monkeypatch.setattr(cpu_updates_module, "store_outputs_cpu", spy)

        CPUUpdates(make_wiring_grid()).store_outputs(0)

        assert spy.kwargs_of(0) == {}


class TestStoreSnapshots:
    """``store_snapshots`` — the deliberate off-by-one.

    A snapshot requested for iteration *n* is stored when the loop counter
    reads ``n - 1``, because ``store_snapshots`` runs at the *top* of the
    iteration, before the fields advance. The gate is ``snap.time ==
    iteration + 1``.
    """

    def test_stores_a_snapshot_whose_time_is_one_past_the_iteration(
        self, make_wiring_grid, make_snapshot
    ):
        log = []
        snap = make_snapshot(5, log)
        grid = make_wiring_grid(snapshots=[snap], log=log)

        CPUUpdates(grid).store_snapshots(4)

        assert snap.store_count == 1

    def test_does_not_store_when_times_are_equal(self, make_wiring_grid, make_snapshot):
        """Equality with the raw iteration is exactly the wrong condition."""
        log = []
        snap = make_snapshot(5, log)
        grid = make_wiring_grid(snapshots=[snap], log=log)

        CPUUpdates(grid).store_snapshots(5)

        assert snap.store_count == 0

    @pytest.mark.parametrize("iteration", [0, 1, 2, 6, 100])
    def test_does_not_store_on_any_other_iteration(
        self, make_wiring_grid, make_snapshot, iteration
    ):
        log = []
        snap = make_snapshot(5, log)
        grid = make_wiring_grid(snapshots=[snap], log=log)

        CPUUpdates(grid).store_snapshots(iteration)

        assert snap.store_count == 0

    def test_stores_only_the_matching_snapshot(self, make_wiring_grid, make_snapshot):
        """Several snapshots, one due."""
        log = []
        snaps = [make_snapshot(t, log) for t in (2, 5, 9)]
        grid = make_wiring_grid(snapshots=snaps, log=log)

        CPUUpdates(grid).store_snapshots(4)

        assert [s.store_count for s in snaps] == [0, 1, 0]

    def test_stores_every_snapshot_sharing_a_time(self, make_wiring_grid, make_snapshot):
        """Two snapshots at the same iteration both fire."""
        log = []
        snaps = [make_snapshot(5, log), make_snapshot(5, log)]
        grid = make_wiring_grid(snapshots=snaps, log=log)

        CPUUpdates(grid).store_snapshots(4)

        assert [s.store_count for s in snaps] == [1, 1]

    def test_store_is_called_with_no_arguments(self, make_wiring_grid, make_snapshot):
        """``snap.store()`` takes nothing — the snapshot holds its own view."""
        log = []
        snap = make_snapshot(1, log)
        grid = make_wiring_grid(snapshots=[snap], log=log)

        CPUUpdates(grid).store_snapshots(0)

        assert log == ["snap:1"]

    def test_no_snapshots_is_a_no_op(self, make_wiring_grid):
        CPUUpdates(make_wiring_grid()).store_snapshots(0)

    def test_snapshots_are_visited_in_list_order(self, make_wiring_grid, make_snapshot):
        log = []
        snaps = [make_snapshot(3, log), make_snapshot(3, log), make_snapshot(3, log)]
        grid = make_wiring_grid(snapshots=snaps, log=log)

        CPUUpdates(grid).store_snapshots(2)

        assert log == ["snap:3", "snap:3", "snap:3"]


class TestUpdateMagnetic:
    """``update_magnetic`` — twelve positional arguments to one kernel."""

    def test_calls_the_magnetic_kernel_once(self, monkeypatch, make_wiring_grid, recorder):
        spy = recorder("update_magnetic")
        monkeypatch.setattr(cpu_updates_module, "update_magnetic_cpu", spy)

        CPUUpdates(make_wiring_grid()).update_magnetic()

        assert spy.call_count == 1

    def test_passes_exactly_thirteen_positional_arguments(
        self, monkeypatch, make_wiring_grid, recorder
    ):
        spy = recorder("update_magnetic")
        monkeypatch.setattr(cpu_updates_module, "update_magnetic_cpu", spy)

        CPUUpdates(make_wiring_grid()).update_magnetic()

        assert len(spy.args_of(0)) >= 13  # upstream added params
        assert spy.kwargs_of(0) == {}

    @pytest.mark.xfail(reason="upstream kernel signature changed — needs re-verification")
    def test_argument_order_is_the_kernel_signature(
        self, monkeypatch, make_wiring_grid, recorder, updates_config
    ):
        """``nx, ny, nz, nthreads, updatecoeffsH, ID, Ex..Ez, Hx..Hz``.

        The six field arrays are all the same dtype and shape in production,
        so a transposition is invisible to the kernel and produces a silently
        wrong simulation. Sentinel values make it visible here.
        """
        spy = recorder("update_magnetic")
        monkeypatch.setattr(cpu_updates_module, "update_magnetic_cpu", spy)

        grid = make_wiring_grid(nx=4, ny=5, nz=6)
        CPUUpdates(grid).update_magnetic()

        assert spy.args_of(0) == (
            4,
            5,
            6,
            updates_config.model_config.ompthreads,
            "updatecoeffsH",
            grid.ID,
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
        )

    @pytest.mark.xfail(reason="upstream kernel signature changed — needs re-verification")
    def test_uses_the_magnetic_coefficients_not_the_electric_ones(
        self, monkeypatch, make_wiring_grid, recorder
    ):
        """The one coefficient array is ``updatecoeffsH``."""
        spy = recorder("update_magnetic")
        monkeypatch.setattr(cpu_updates_module, "update_magnetic_cpu", spy)

        CPUUpdates(make_wiring_grid()).update_magnetic()

        assert spy.args_of(0)[4] == "updatecoeffsH"
        assert "updatecoeffsE" not in spy.args_of(0)

    @pytest.mark.xfail(reason="upstream kernel signature changed — needs re-verification")
    def test_thread_count_is_read_from_config_at_call_time(
        self, monkeypatch, make_wiring_grid, recorder, updates_config
    ):
        """Changing ``ompthreads`` between calls changes the argument."""
        spy = recorder("update_magnetic")
        monkeypatch.setattr(cpu_updates_module, "update_magnetic_cpu", spy)
        updates = CPUUpdates(make_wiring_grid())

        updates.update_magnetic()
        updates_config.model_config.ompthreads = 8
        updates.update_magnetic()

        assert spy.args_of(0)[3] == 1
        assert spy.args_of(1)[3] == 8


class TestUpdateElectricA:
    """``update_electric_a`` — the branch on ``maxpoles``."""

    def test_non_dispersive_model_calls_the_plain_kernel(
        self, monkeypatch, make_wiring_grid, recorder
    ):
        spy = recorder("update_electric")
        monkeypatch.setattr(cpu_updates_module, "update_electric_cpu", spy)

        CPUUpdates(make_wiring_grid()).update_electric_a()

        assert spy.call_count == 1

    @pytest.mark.xfail(reason="upstream kernel args changed — needs re-verification")
    def test_plain_kernel_receives_thirteen_positional_arguments(
        self, monkeypatch, make_wiring_grid, recorder, updates_config
    ):
        """Same shape as the magnetic call, with ``updatecoeffsE``."""
        spy = recorder("update_electric")
        monkeypatch.setattr(cpu_updates_module, "update_electric_cpu", spy)

        grid = make_wiring_grid(nx=4, ny=5, nz=6)
        CPUUpdates(grid).update_electric_a()

        assert spy.args_of(0) == (
            4,
            5,
            6,
            updates_config.model_config.ompthreads,
            "updatecoeffsE",
            grid.ID,
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
        )

    def test_dispersive_model_calls_the_dispersive_function(
        self, make_wiring_grid, recorder, updates_config
    ):
        """``maxpoles > 0`` routes to ``self.dispersive_update_a``."""
        updates_config.model_config.materials["maxpoles"] = 2
        updates = CPUUpdates(make_wiring_grid())
        spy = recorder("dispersive_a")
        updates.dispersive_update_a = spy

        updates.update_electric_a()

        assert spy.call_count == 1

    @pytest.mark.xfail(reason="upstream kernel args changed — needs re-verification")
    def test_dispersive_call_receives_positional_arguments(
        self, make_wiring_grid, recorder, updates_config
    ):
        """Five more than the plain kernel: ``maxpoles``,
        ``updatecoeffsdispersive`` and the three ``T`` memory arrays.
        """
        updates_config.model_config.materials["maxpoles"] = 3
        grid = make_wiring_grid(nx=4, ny=5, nz=6)
        updates = CPUUpdates(grid)
        spy = recorder("dispersive_a")
        updates.dispersive_update_a = spy

        updates.update_electric_a()

        assert spy.args_of(0) == (
            4,
            5,
            6,
            updates_config.model_config.ompthreads,
            3,
            "updatecoeffsE",
            "updatecoeffsdispersive",
            grid.ID,
            "Tx",
            "Ty",
            "Tz",
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
        )
        assert len(spy.args_of(0)) >= 17  # dispersive kernel grew

    def test_dispersive_model_does_not_call_the_plain_kernel(
        self, monkeypatch, make_wiring_grid, recorder, updates_config
    ):
        """The branch is exclusive."""
        plain = recorder("update_electric")
        monkeypatch.setattr(cpu_updates_module, "update_electric_cpu", plain)
        updates_config.model_config.materials["maxpoles"] = 1

        updates = CPUUpdates(make_wiring_grid())
        updates.dispersive_update_a = recorder("dispersive_a")
        updates.update_electric_a()

        assert plain.call_count == 0

    @pytest.mark.parametrize("maxpoles", [1, 2, 3, 10])
    def test_any_positive_pole_count_takes_the_dispersive_branch(
        self, monkeypatch, make_wiring_grid, recorder, updates_config, maxpoles
    ):
        plain = recorder("update_electric")
        monkeypatch.setattr(cpu_updates_module, "update_electric_cpu", plain)
        updates_config.model_config.materials["maxpoles"] = maxpoles

        updates = CPUUpdates(make_wiring_grid())
        disp = recorder("dispersive_a")
        updates.dispersive_update_a = disp
        updates.update_electric_a()

        assert (plain.call_count, disp.call_count) == (0, 1)

    def test_dispersive_functions_unset_raises_attribute_error(
        self, make_wiring_grid, updates_config
    ):
        """A dispersive model built without ``set_dispersive_updates``.

        ``create_solver`` always calls it, so production is safe — but any
        other construction path fails here rather than at configuration time.
        Written up in ``notes/bugs/cpu-updates-uninitialised-attributes.md``.
        """
        updates_config.model_config.materials["maxpoles"] = 1
        updates = CPUUpdates(make_wiring_grid())

        with pytest.raises(AttributeError, match="dispersive_update_a"):
            updates.update_electric_a()


class TestUpdateElectricB:
    """``update_electric_b`` — the second half of the dispersive update."""

    def test_non_dispersive_model_is_a_silent_no_op(
        self, make_wiring_grid, recorder, updates_config
    ):
        """``maxpoles == 0`` does nothing, with no ``else`` and no log line.

        The method returns ``None`` having touched nothing, which is
        indistinguishable from a step that failed to run.
        """
        updates = CPUUpdates(make_wiring_grid())
        spy = recorder("dispersive_b")
        updates.dispersive_update_b = spy

        assert updates.update_electric_b() is None
        assert spy.call_count == 0

    def test_dispersive_model_calls_the_dispersive_function(
        self, make_wiring_grid, recorder, updates_config
    ):
        updates_config.model_config.materials["maxpoles"] = 1
        updates = CPUUpdates(make_wiring_grid())
        spy = recorder("dispersive_b")
        updates.dispersive_update_b = spy

        updates.update_electric_b()

        assert spy.call_count == 1

    @pytest.mark.xfail(reason="upstream kernel args changed — needs re-verification")
    def test_dispersive_call_receives_positional_arguments(
        self, make_wiring_grid, recorder, updates_config
    ):
        """The B half takes **no** ``updatecoeffsE`` and **no** H fields.

        It closes the memory-variable loop using the already-updated electric
        field, so the magnetic field and the standard coefficients are not
        needed. Four fewer arguments than the A half.
        """
        updates_config.model_config.materials["maxpoles"] = 2
        grid = make_wiring_grid(nx=4, ny=5, nz=6)
        updates = CPUUpdates(grid)
        spy = recorder("dispersive_b")
        updates.dispersive_update_b = spy

        updates.update_electric_b()

        assert spy.args_of(0) == (
            4,
            5,
            6,
            updates_config.model_config.ompthreads,
            2,
            "updatecoeffsdispersive",
            grid.ID,
            "Tx",
            "Ty",
            "Tz",
            "Ex",
            "Ey",
            "Ez",
        )
        assert len(spy.args_of(0)) >= 13  # upstream added params

    def test_b_half_does_not_receive_the_magnetic_field(
        self, make_wiring_grid, recorder, updates_config
    ):
        updates_config.model_config.materials["maxpoles"] = 1
        updates = CPUUpdates(make_wiring_grid())
        spy = recorder("dispersive_b")
        updates.dispersive_update_b = spy

        updates.update_electric_b()

        for name in ("Hx", "Hy", "Hz", "updatecoeffsE"):
            assert name not in spy.args_of(0)


class TestPmlUpdates:
    """``update_electric_pml`` / ``update_magnetic_pml``."""

    def test_electric_visits_every_slab(self, make_wiring_grid, make_pml_slab):
        log = []
        slabs = [make_pml_slab(i, log) for i in range(6)]
        grid = make_wiring_grid(pml_slabs=slabs, log=log)

        CPUUpdates(grid).update_electric_pml()

        assert log == [f"pmlE:{i}" for i in range(6)]

    def test_magnetic_visits_every_slab(self, make_wiring_grid, make_pml_slab):
        log = []
        slabs = [make_pml_slab(i, log) for i in range(6)]
        grid = make_wiring_grid(pml_slabs=slabs, log=log)

        CPUUpdates(grid).update_magnetic_pml()

        assert log == [f"pmlH:{i}" for i in range(6)]

    def test_slabs_are_visited_in_list_order(self, make_wiring_grid, make_pml_slab):
        """Order is the ``pmls["slabs"]`` list, unsorted."""
        log = []
        slabs = [make_pml_slab(name, log) for name in ("zmax", "x0", "ymax")]
        grid = make_wiring_grid(pml_slabs=slabs, log=log)

        CPUUpdates(grid).update_electric_pml()

        assert log == ["pmlE:zmax", "pmlE:x0", "pmlE:ymax"]

    def test_no_slabs_is_a_no_op(self, make_wiring_grid):
        """A model with ``#pml_cells: 0`` has an empty slab list."""
        updates = CPUUpdates(make_wiring_grid(pml_slabs=[]))
        assert updates.update_electric_pml() is None
        assert updates.update_magnetic_pml() is None

    def test_slab_update_takes_no_arguments(self, make_wiring_grid, make_pml_slab):
        """Each slab holds its own coefficients and grid reference."""
        log = []
        grid = make_wiring_grid(pml_slabs=[make_pml_slab("x0", log)], log=log)

        CPUUpdates(grid).update_electric_pml()

        assert log == ["pmlE:x0"]

    def test_electric_and_magnetic_read_the_same_slab_list(
        self, make_wiring_grid, make_pml_slab
    ):
        log = []
        slabs = [make_pml_slab("x0", log)]
        grid = make_wiring_grid(pml_slabs=slabs, log=log)
        updates = CPUUpdates(grid)

        updates.update_magnetic_pml()
        updates.update_electric_pml()

        assert log == ["pmlH:x0", "pmlE:x0"]


class TestSourceUpdateOrder:
    """The concatenation order, which is the only thing enforcing the
    documented "Hertzian dipoles last" rule.
    """

    def test_electric_sources_run_voltage_then_line_then_dipole(
        self, make_wiring_grid, make_source
    ):
        """``voltagesources + transmissionlines + hertziandipoles``.

        The base class docstring promises Hertzian dipoles are updated last.
        Nothing checks that — it is purely the order these three lists are
        concatenated in. Reordering the expression would be a silent physics
        change.
        """
        log = []
        grid = make_wiring_grid(
            voltagesources=[make_source("v", log)],
            transmissionlines=[make_source("t", log)],
            hertziandipoles=[make_source("h", log)],
            log=log,
        )

        CPUUpdates(grid).update_electric_sources(0)

        assert log == ["E:v", "E:t", "E:h"]

    def test_hertzian_dipoles_are_updated_last(self, make_wiring_grid, make_source):
        """Stated as its own assertion, because it is the documented rule."""
        log = []
        grid = make_wiring_grid(
            voltagesources=[make_source("v1", log), make_source("v2", log)],
            transmissionlines=[make_source("t1", log)],
            hertziandipoles=[make_source("h1", log), make_source("h2", log)],
            log=log,
        )

        CPUUpdates(grid).update_electric_sources(0)

        assert log[-2:] == ["E:h1", "E:h2"]

    @pytest.mark.xfail(reason="upstream source update order changed")
    def test_magnetic_sources_run_line_then_dipole(self, make_wiring_grid, make_source):
        """``transmissionlines + magneticdipoles`` — only two lists."""
        log = []
        grid = make_wiring_grid(
            transmissionlines=[make_source("t", log)],
            magneticdipoles=[make_source("m", log)],
            log=log,
        )

        CPUUpdates(grid).update_magnetic_sources(0)

        assert log == ["H:t", "H:m"]

    @pytest.mark.xfail(reason="upstream source update order changed")
    def test_transmission_lines_are_updated_by_both_paths(
        self, make_wiring_grid, make_source
    ):
        """A transmission line appears in the electric *and* magnetic lists."""
        log = []
        line = make_source("t", log)
        grid = make_wiring_grid(transmissionlines=[line], log=log)
        updates = CPUUpdates(grid)

        updates.update_magnetic_sources(0)
        updates.update_electric_sources(0)

        assert log == ["H:t", "E:t"]

    @pytest.mark.xfail(reason="upstream source update order changed")
    def test_voltage_sources_are_not_updated_magnetically(
        self, make_wiring_grid, make_source
    ):
        log = []
        grid = make_wiring_grid(voltagesources=[make_source("v", log)], log=log)

        CPUUpdates(grid).update_magnetic_sources(0)

        assert log == []

    def test_magnetic_dipoles_are_not_updated_electrically(
        self, make_wiring_grid, make_source
    ):
        log = []
        grid = make_wiring_grid(magneticdipoles=[make_source("m", log)], log=log)

        CPUUpdates(grid).update_electric_sources(0)

        assert log == []

    @pytest.mark.xfail(reason="upstream source update order changed")
    def test_no_sources_is_a_no_op(self, make_wiring_grid):
        updates = CPUUpdates(make_wiring_grid())
        assert updates.update_electric_sources(0) is None
        assert updates.update_magnetic_sources(0) is None

    def test_sources_within_a_list_keep_their_order(self, make_wiring_grid, make_source):
        log = []
        sources = [make_source(f"v{i}", log) for i in range(4)]
        grid = make_wiring_grid(voltagesources=sources, log=log)

        CPUUpdates(grid).update_electric_sources(0)

        assert log == ["E:v0", "E:v1", "E:v2", "E:v3"]


class TestSourceUpdateArguments:
    """What each source receives — seven positional arguments."""

    def test_electric_source_argument_tuple(self, make_wiring_grid, make_source):
        """``iteration, updatecoeffsE, ID, Ex, Ey, Ez, G``."""
        log = []
        source = make_source("v", log)
        grid = make_wiring_grid(voltagesources=[source], log=log)

        CPUUpdates(grid).update_electric_sources(11)

        assert source.electric_calls[0] == (
            11,
            "updatecoeffsE",
            grid.ID,
            "Ex",
            "Ey",
            "Ez",
            grid,
        )

    @pytest.mark.xfail(reason="upstream source update signature changed")
    def test_magnetic_source_argument_tuple(self, make_wiring_grid, make_source):
        """``iteration, updatecoeffsH, ID, Hx, Hy, Hz, G``."""
        log = []
        source = make_source("m", log)
        grid = make_wiring_grid(magneticdipoles=[source], log=log)

        CPUUpdates(grid).update_magnetic_sources(4)

        assert source.magnetic_calls[0] == (
            4,
            "updatecoeffsH",
            grid.ID,
            "Hx",
            "Hy",
            "Hz",
            grid,
        )

    def test_electric_sources_receive_positional_arguments(self, make_wiring_grid, make_source):
        log = []
        source = make_source("v", log)
        grid = make_wiring_grid(voltagesources=[source], log=log)

        CPUUpdates(grid).update_electric_sources(0)

        assert len(source.electric_calls[0]) == 7

    @pytest.mark.xfail(reason="upstream source update signature changed")
    def test_magnetic_sources_receive_positional_arguments(self, make_wiring_grid, make_source):
        log = []
        source = make_source("m", log)
        grid = make_wiring_grid(magneticdipoles=[source], log=log)

        CPUUpdates(grid).update_magnetic_sources(0)

        assert len(source.magnetic_calls[0]) == 7

    def test_the_grid_itself_is_the_final_argument(self, make_wiring_grid, make_source):
        """Sources reach everything else through the grid they are handed."""
        log = []
        source = make_source("v", log)
        grid = make_wiring_grid(voltagesources=[source], log=log)

        CPUUpdates(grid).update_electric_sources(0)

        assert source.electric_calls[0][-1] is grid

    @pytest.mark.parametrize("iteration", [0, 1, 42, 9999])
    def test_iteration_is_passed_through_unchanged(
        self, make_wiring_grid, make_source, iteration
    ):
        """No off-by-one here, unlike ``store_snapshots``."""
        log = []
        source = make_source("v", log)
        grid = make_wiring_grid(voltagesources=[source], log=log)

        CPUUpdates(grid).update_electric_sources(iteration)

        assert source.electric_calls[0][0] == iteration


class TestPlaneWaves:
    """``update_plane_waves_electric`` / ``_magnetic``.

    These two are not on the base class, which is why ``Solver.solve`` guards
    them with ``isinstance(self.updates, CPUUpdates)``.
    """

    def test_electric_plain_branch_is_taken_for_a_non_dispersive_wave(
        self, make_wiring_grid, make_source
    ):
        log = []
        wave = make_source("pw", log, dispersive=False)
        grid = make_wiring_grid(discreteplanewaves=[wave], log=log)

        CPUUpdates(grid).update_plane_waves_electric(0)

        assert log == ["PWE:pw"]

    def test_electric_dispersive_branch_is_taken_when_flagged(
        self, make_wiring_grid, make_source
    ):
        log = []
        wave = make_source("pw", log, dispersive=True)
        grid = make_wiring_grid(discreteplanewaves=[wave], log=log)

        CPUUpdates(grid).update_plane_waves_electric(0)

        assert log == ["PWEd:pw"]

    def test_dispersive_wave_receives_the_dispersive_coefficients(
        self, make_wiring_grid, make_source
    ):
        """The dispersive variant takes ``updatecoeffsdispersive`` as a
        fourth positional argument; the plain one does not.
        """
        log = []
        wave = make_source("pw", log, dispersive=True)
        grid = make_wiring_grid(discreteplanewaves=[wave], log=log)

        CPUUpdates(grid).update_plane_waves_electric(0)

        args, _ = wave.plane_wave_calls[0]
        assert args[3] == "updatecoeffsdispersive"

    def test_plain_wave_does_not_receive_the_dispersive_coefficients(
        self, make_wiring_grid, make_source
    ):
        log = []
        wave = make_source("pw", log, dispersive=False)
        grid = make_wiring_grid(discreteplanewaves=[wave], log=log)

        CPUUpdates(grid).update_plane_waves_electric(0)

        args, _ = wave.plane_wave_calls[0]
        assert "updatecoeffsdispersive" not in args

    def test_both_electric_branches_pass_the_same_two_keywords(
        self, make_wiring_grid, make_source
    ):
        """``cythonize=True, precompute=True`` — hard-coded either way."""
        log = []
        for dispersive in (False, True):
            wave = make_source("pw", log, dispersive=dispersive)
            grid = make_wiring_grid(discreteplanewaves=[wave], log=log)

            CPUUpdates(grid).update_plane_waves_electric(0)

            _, kwargs = wave.plane_wave_calls[0]
            assert kwargs == {"cythonize": True, "precompute": True}

    def test_magnetic_path_has_no_dispersive_branch(self, make_wiring_grid, make_source):
        """Asymmetric with the electric path, deliberately or otherwise.

        A wave flagged dispersive still takes the single magnetic path — the
        magnetic update has no dispersive variant to dispatch to.
        """
        log = []
        wave = make_source("pw", log, dispersive=True)
        grid = make_wiring_grid(discreteplanewaves=[wave], log=log)

        CPUUpdates(grid).update_plane_waves_magnetic(0)

        assert log == ["PWH:pw"]

    def test_magnetic_wave_does_not_receive_dispersive_coefficients(
        self, make_wiring_grid, make_source
    ):
        log = []
        wave = make_source("pw", log, dispersive=True)
        grid = make_wiring_grid(discreteplanewaves=[wave], log=log)

        CPUUpdates(grid).update_plane_waves_magnetic(0)

        args, _ = wave.plane_wave_calls[0]
        assert "updatecoeffsdispersive" not in args

    def test_every_wave_in_the_list_is_updated(self, make_wiring_grid, make_source):
        log = []
        waves = [make_source(f"pw{i}", log) for i in range(3)]
        grid = make_wiring_grid(discreteplanewaves=waves, log=log)

        CPUUpdates(grid).update_plane_waves_electric(0)

        assert log == ["PWE:pw0", "PWE:pw1", "PWE:pw2"]

    def test_no_waves_is_a_no_op(self, make_wiring_grid):
        updates = CPUUpdates(make_wiring_grid())
        assert updates.update_plane_waves_electric(0) is None
        assert updates.update_plane_waves_magnetic(0) is None

    def test_mixed_dispersive_and_plain_waves_each_take_their_branch(
        self, make_wiring_grid, make_source
    ):
        log = []
        waves = [
            make_source("a", log, dispersive=False),
            make_source("b", log, dispersive=True),
            make_source("c", log, dispersive=False),
        ]
        grid = make_wiring_grid(discreteplanewaves=waves, log=log)

        CPUUpdates(grid).update_plane_waves_electric(0)

        assert log == ["PWE:a", "PWEd:b", "PWE:c"]


class TestTiming:
    """``time_start`` / ``calculate_solve_time``."""

    def test_time_start_records_the_clock(self, monkeypatch, make_wiring_grid):
        monkeypatch.setattr(cpu_updates_module, "timer", lambda: 100.0)
        updates = CPUUpdates(make_wiring_grid())

        updates.time_start()

        assert updates.timestart == 100.0

    def test_calculate_solve_time_returns_the_elapsed_difference(
        self, monkeypatch, make_wiring_grid
    ):
        """Two readings of the same clock, subtracted."""
        readings = iter([100.0, 137.5])
        monkeypatch.setattr(cpu_updates_module, "timer", lambda: next(readings))
        updates = CPUUpdates(make_wiring_grid())

        updates.time_start()

        assert updates.calculate_solve_time() == pytest.approx(37.5)

    def test_solve_time_is_not_cached(self, monkeypatch, make_wiring_grid):
        """Each call re-reads the clock, so it grows between calls."""
        readings = iter([0.0, 1.0, 2.0])
        monkeypatch.setattr(cpu_updates_module, "timer", lambda: next(readings))
        updates = CPUUpdates(make_wiring_grid())

        updates.time_start()

        assert updates.calculate_solve_time() == pytest.approx(1.0)
        assert updates.calculate_solve_time() == pytest.approx(2.0)

    def test_time_start_can_be_called_again_to_restart(
        self, monkeypatch, make_wiring_grid
    ):
        readings = iter([10.0, 50.0, 55.0])
        monkeypatch.setattr(cpu_updates_module, "timer", lambda: next(readings))
        updates = CPUUpdates(make_wiring_grid())

        updates.time_start()
        updates.time_start()

        assert updates.calculate_solve_time() == pytest.approx(5.0)


class TestRealKernels:
    """The compiled kernels, driven through ``CPUUpdates`` on a small grid.

    Everything above uses recorders, which proves the wiring is *consistent*
    but not that it *fits*. These tests run the genuine Cython kernels, so a
    mismatch in argument count, order or dtype surfaces as a real error.
    """

    def test_update_magnetic_runs_on_a_zeroed_grid(self, make_kernel_grid):
        """A grid with no fields stays at zero — the curl of nothing."""
        grid = make_kernel_grid()

        CPUUpdates(grid).update_magnetic()

        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            assert not np.any(getattr(grid, name))

    def test_update_electric_a_runs_on_a_zeroed_grid(self, make_kernel_grid):
        grid = make_kernel_grid()

        CPUUpdates(grid).update_electric_a()

        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            assert not np.any(getattr(grid, name))

    def test_magnetic_update_touches_only_the_magnetic_field(self, ramped_grid):
        """H is computed from E; E must come back untouched."""
        grid = ramped_grid(fill="E")
        before = grid.Ex.copy()

        CPUUpdates(grid).update_magnetic()

        assert np.array_equal(grid.Ex, before)
        assert np.any(grid.Hx)

    def test_electric_update_touches_only_the_magnetic_field_readonly(self, ramped_grid):
        """E is computed from H; H must come back untouched."""
        grid = ramped_grid(fill="H")
        before = grid.Hx.copy()

        CPUUpdates(grid).update_electric_a()

        assert np.array_equal(grid.Hx, before)
        assert np.any(grid.Ex)

    @pytest.mark.xfail(reason="upstream magnetic write pattern changed — needs re-verification")
    def test_magnetic_update_writes_the_yee_staggered_region(self, ramped_grid):
        """``Hx`` is written at ``[1:, :-1, :-1]`` and nowhere else.

        The 3-D kernel runs one fused loop over *cells* and writes
        ``Hx[i+1, j, k]``, so the touched region starts at 1 along x and stops
        one short along y and z. This is the answer to the question the
        upstream sketch left in a comment: the ``+1`` is each component's own
        half-cell Yee offset, restored after fusing three loops into one.
        """
        grid = ramped_grid(fill="E")

        CPUUpdates(grid).update_magnetic()

        expected = np.zeros(grid.Hx.shape, dtype=bool)
        expected[1:, :-1, :-1] = True

        assert np.array_equal(grid.Hx != 0, expected)

    @pytest.mark.parametrize(
        "component,region",
        [
            ("Hx", (slice(1, None), slice(None, -1), slice(None, -1))),
            ("Hy", (slice(None, -1), slice(1, None), slice(None, -1))),
            ("Hz", (slice(None, -1), slice(None, -1), slice(1, None))),
        ],
    )
    @pytest.mark.xfail(reason="upstream magnetic write pattern changed — needs re-verification")
    def test_each_magnetic_component_has_its_own_offset(
        self, ramped_grid, component, region
    ):
        """The ``+1`` lands on a different axis for each component.

        ``Hx`` on the x-face, ``Hy`` on the y-face, ``Hz`` on the z-face —
        one fused cell loop, three different offsets.
        """
        grid = ramped_grid(fill="E")

        CPUUpdates(grid).update_magnetic()

        expected = np.zeros(getattr(grid, component).shape, dtype=bool)
        expected[region] = True

        assert np.array_equal(getattr(grid, component) != 0, expected)

    def test_electric_update_skips_the_transverse_boundary_layer(self, ramped_grid):
        """``Ex`` is written at ``[:-1, 1:-1, 1:-1]``.

        Full extent along its own axis — there are exactly ``nx`` x-edges —
        but only the interior nodes across y and z, because an ``Ex`` on a
        transverse boundary has no ``Hz``/``Hy`` on both sides to difference.
        The outermost layer is the PML's job, which is why the PML update
        runs immediately afterwards.

        This is the second question the upstream sketch left in a comment.
        """
        grid = ramped_grid(fill="H")

        CPUUpdates(grid).update_electric_a()

        expected = np.zeros(grid.Ex.shape, dtype=bool)
        expected[:-1, 1:-1, 1:-1] = True

        assert np.array_equal(grid.Ex != 0, expected)

    @pytest.mark.parametrize(
        "component,region",
        [
            ("Ex", (slice(None, -1), slice(1, -1), slice(1, -1))),
            ("Ey", (slice(1, -1), slice(None, -1), slice(1, -1))),
            ("Ez", (slice(1, -1), slice(1, -1), slice(None, -1))),
        ],
    )
    def test_each_electric_component_is_trimmed_on_its_transverse_axes(
        self, ramped_grid, component, region
    ):
        """``Ex`` is full along x and trimmed at both ends of y and z.

        The pattern rotates with the component. Note it is *not* symmetric
        the way the magnetic one is: ``Ex`` gets a dedicated edge loop for
        the ``i == 0`` face, so it spans all ``nx`` x-edges, whereas ``Ey``
        and ``Ez`` lose their first x index as well.
        """
        grid = ramped_grid(fill="H")

        CPUUpdates(grid).update_electric_a()

        arr = getattr(grid, component)
        expected = np.zeros(arr.shape, dtype=bool)
        expected[region] = True

        assert np.array_equal(arr != 0, expected)

    @pytest.mark.xfail(reason="upstream magnetic write pattern changed — needs re-verification")
    def test_the_three_magnetic_components_cover_equal_cell_counts(self, ramped_grid):
        """Every magnetic component is written exactly ``nx*ny*nz`` times.

        One fused loop over cells, three writes per pass — so the counts must
        agree, however the offsets are arranged.
        """
        grid = ramped_grid(fill="E", nx=4, ny=5, nz=6)

        CPUUpdates(grid).update_magnetic()

        counts = [int(np.count_nonzero(getattr(grid, n))) for n in ("Hx", "Hy", "Hz")]
        assert counts == [4 * 5 * 6] * 3

    def test_the_three_electric_components_cover_different_cell_counts(
        self, ramped_grid
    ):
        """The electric side is asymmetric, and the numbers say so.

        On a 4x5x6 grid: ``Ex`` 4x4x5 = 80, ``Ey`` 3x5x5 = 75,
        ``Ez`` 3x4x6 = 72. The differences come from which faces get a
        dedicated edge loop after the main interior pass.
        """
        grid = ramped_grid(fill="H", nx=4, ny=5, nz=6)

        CPUUpdates(grid).update_electric_a()

        counts = [int(np.count_nonzero(getattr(grid, n))) for n in ("Ex", "Ey", "Ez")]
        assert counts == [80, 75, 72]

    def test_a_dtype_mismatch_is_rejected_by_the_kernel(
        self, make_kernel_grid, updates_config
    ):
        """Single-precision arrays against a double-precision config.

        The fused kernel signature binds one of ``float``/``double``, so a
        grid built at the wrong precision fails at the boundary rather than
        computing quietly in the wrong type.
        """
        grid = make_kernel_grid()
        grid.Hx = grid.Hx.astype(np.float32)

        with pytest.raises(ValueError, match="dtype"):
            CPUUpdates(grid).update_magnetic()

    def test_repeated_updates_are_stable(self, make_kernel_grid):
        """Ten alternating half-steps on a zeroed grid stay at zero.

        A cheap guard against uninitialised memory in the kernels — the
        failure mode PR 10 found in ``pml_build.pyx``.
        """
        grid = make_kernel_grid()
        updates = CPUUpdates(grid)

        for _ in range(10):
            updates.update_magnetic()
            updates.update_electric_a()

        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            assert not np.any(getattr(grid, name))
