"""The memory estimates gprMax makes before it allocates anything.

An FDTD grid is a dozen dense arrays sized ``(nx+1)(ny+1)(nz+1)``. A model
that will not fit fails with a ``MemoryError`` from inside NumPy, minutes into
a build, with no indication of which grid was too large. These functions exist
to say so first, in bytes the user recognises.

Three things are being tested, and they are different in kind:

**Arithmetic.** ``mem_check_run_all`` and ``mem_check_build_all`` accumulate
into two places at once — ``get_model_config().mem_use``, which is per *model*
and spans every grid including subgrids, and ``grid.mem_use``, which is per
grid. Getting one and not the other is the obvious failure, and it is silent:
the warning threshold is simply never reached.

**Conditionals.** Dispersive materials add coefficient arrays; snapshots add a
field copy each; fractal volumes add complex arrays during construction only.
Each is counted only when present, which is four branches whose absence costs
nothing and whose presence can double the estimate.

**A warning that must fire.** ``mem_check_host`` compares against the RAM
figure ``get_host_info`` found at startup. It is the last thing standing
between a user and an out-of-memory kill.

The grids here are ``SimpleNamespace`` stand-ins with four method attributes.
A real ``FDTDGrid`` would allocate the very arrays these functions are trying
to predict, which would make the tests both slow and circular.
"""

from gprMax.utilities.host_info import (
    mem_check_build_all,
    mem_check_host,
    mem_check_run_all,
)


class TestMemCheckHost:
    """``mem_check_host`` — one comparison, one warning."""

    def test_a_small_request_is_silent(self, install_host_config, caplog):
        caplog.set_level(1)
        install_host_config()

        mem_check_host(1024)

        assert caplog.records == []

    def test_an_oversized_request_warns(self, install_host_config, caplog):
        install_host_config()

        mem_check_host(32 * 1024**3)

        assert caplog.records[0].levelname == "WARNING"

    def test_the_requested_amount_appears_in_the_warning(
        self, install_host_config, caplog
    ):
        """Humanised, so the user can compare it with what they have."""
        install_host_config()

        mem_check_host(32 * 1024**3)

        assert "34.4 GB" in caplog.text

    def test_the_available_amount_appears_in_the_warning(
        self, install_host_config, caplog
    ):
        install_host_config()

        mem_check_host(32 * 1024**3)

        assert "16.0 GiB" in caplog.text

    def test_exactly_the_available_amount_does_not_warn(
        self, install_host_config, caplog
    ):
        """A strict ``>``; using every last byte is allowed."""
        caplog.set_level(1)
        install_host_config()

        mem_check_host(16 * 1024**3)

        assert caplog.records == []

    def test_one_byte_over_warns(self, install_host_config, caplog):
        install_host_config()

        mem_check_host(16 * 1024**3 + 1)

        assert len(caplog.records) == 1

    def test_it_only_warns_and_never_raises(self, install_host_config):
        """The user may know something the estimate does not — swap, or a
        machine whose RAM was mis-detected. The run is not blocked.
        """
        install_host_config()

        assert mem_check_host(10**15) is None

    def test_the_limit_comes_from_the_global_host_info(
        self, install_host_config, caplog
    ):
        """Not probed live, so a long run is judged against startup figures.

        Shown by lowering the recorded RAM to an absurd figure: a request of
        two kilobytes then warns, which no real machine would provoke.
        """
        install_host_config(ram=1024)

        mem_check_host(2048)

        assert "exceeds" in caplog.text


class TestMemCheckRunAll:
    """``mem_check_run_all`` — the estimate made just before solving."""

    def test_the_basic_estimate_is_added_to_the_model_total(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        model_config = install_model_config(mem_use=65_000_000)

        mem_check_run_all([make_grid(basic=1000)])

        assert model_config.mem_use == 65_001_000

    def test_the_basic_estimate_is_added_to_the_grid_total(
        self, install_host_config, install_model_config, make_grid
    ):
        """Both tallies are kept; the per-grid one is what the banner prints."""
        install_host_config()
        install_model_config()
        grid = make_grid(basic=1000)

        mem_check_run_all([grid])

        assert grid.mem_use == 1000

    def test_the_model_total_is_returned(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config(mem_use=65_000_000)

        total, _ = mem_check_run_all([make_grid(basic=1000)])

        assert total == 65_001_000

    def test_every_grid_contributes(
        self, install_host_config, install_model_config, make_grid
    ):
        """Subgrids are separate ``FDTDGrid`` objects in the same list."""
        install_host_config()
        model_config = install_model_config()

        mem_check_run_all([make_grid(basic=1000), make_grid(basic=2000)])

        assert model_config.mem_use == 3000

    def test_one_string_is_produced_per_grid(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config()

        _, strings = mem_check_run_all(
            [make_grid(name="main"), make_grid(name="sub")]
        )

        assert len(strings) == 2

    def test_each_string_names_its_grid(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config()

        _, strings = mem_check_run_all([make_grid(name="main")])

        assert "[main]" in strings[0]

    def test_each_string_is_humanised(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config()

        _, strings = mem_check_run_all([make_grid(basic=1_500_000)])

        assert "1.5 MB" in strings[0]

    def test_an_empty_grid_list_returns_the_existing_total(
        self, install_host_config, install_model_config
    ):
        """The 65 MB interpreter overhead ``ModelConfig`` starts with."""
        install_host_config()
        install_model_config(mem_use=65_000_000)

        total, strings = mem_check_run_all([])

        assert (total, strings) == (65_000_000, [])


class TestDispersiveMaterials:
    """The dispersive coefficient arrays, counted only when poles exist."""

    def test_no_poles_means_no_extra_memory(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        model_config = install_model_config(maxpoles=0)

        mem_check_run_all([make_grid(basic=1000, dispersive=9999)])

        assert model_config.mem_use == 1000

    def test_poles_add_the_dispersive_estimate(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        model_config = install_model_config(maxpoles=1)

        mem_check_run_all([make_grid(basic=1000, dispersive=2000)])

        assert model_config.mem_use == 3000

    def test_the_grid_tally_is_also_increased(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config(maxpoles=3)
        grid = make_grid(basic=1000, dispersive=2000)

        mem_check_run_all([grid])

        assert grid.mem_use == 3000

    def test_the_pole_count_is_read_once_per_model_not_per_grid(
        self, install_host_config, install_model_config, make_grid
    ):
        """``maxpoles`` is a model-wide property; subgrids share it."""
        install_host_config()
        model_config = install_model_config(maxpoles=2)

        mem_check_run_all(
            [make_grid(basic=0, dispersive=100), make_grid(basic=0, dispersive=200)]
        )

        assert model_config.mem_use == 300


class TestSnapshots:
    """Snapshots are copies of the field arrays, held until they are written."""

    @staticmethod
    def _snapshot(nbytes):
        from types import SimpleNamespace

        return SimpleNamespace(nbytes=nbytes)

    def test_no_snapshots_add_nothing(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        model_config = install_model_config()

        mem_check_run_all([make_grid(basic=1000, snapshots=[])])

        assert model_config.mem_use == 1000

    def test_each_snapshot_is_counted(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        model_config = install_model_config()
        snapshots = [self._snapshot(100), self._snapshot(200)]

        mem_check_run_all([make_grid(basic=1000, snapshots=snapshots)])

        assert model_config.mem_use == 1300

    def test_the_grid_tally_is_also_increased(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config()
        grid = make_grid(basic=1000, snapshots=[self._snapshot(500)])

        mem_check_run_all([grid])

        assert grid.mem_use == 1500

    def test_snapshots_across_grids_are_all_counted(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        model_config = install_model_config()

        mem_check_run_all(
            [
                make_grid(basic=0, snapshots=[self._snapshot(100)]),
                make_grid(basic=0, snapshots=[self._snapshot(200)]),
            ]
        )

        assert model_config.mem_use == 300

    def test_a_snapshot_heavy_model_can_trigger_the_host_warning(
        self, install_host_config, install_model_config, make_grid, caplog
    ):
        """The reason snapshots are counted at all.

        The field arrays fit; the hundred copies of them do not.
        """
        install_host_config()
        install_model_config()
        snapshots = [self._snapshot(1024**3) for _ in range(20)]

        mem_check_run_all([make_grid(basic=1000, snapshots=snapshots)])

        assert "exceeds" in caplog.text


class TestMemCheckBuildAll:
    """``mem_check_build_all`` — the estimate made before geometry is built."""

    def test_the_basic_estimate_is_counted(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config(mem_use=1000)

        total, _ = mem_check_build_all([make_grid(basic=2000)])

        assert total == 3000

    def test_fractal_volumes_add_their_estimate(
        self, install_host_config, install_model_config, make_grid
    ):
        """Fractal construction needs complex arrays the solve never sees."""
        install_host_config()
        install_model_config()

        total, _ = mem_check_build_all(
            [make_grid(basic=1000, fractals=5000, fractalvolumes=["a volume"])]
        )

        assert total == 6000

    def test_no_fractal_volumes_means_no_extra_memory(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config()

        total, _ = mem_check_build_all(
            [make_grid(basic=1000, fractals=5000, fractalvolumes=[])]
        )

        assert total == 1000

    def test_it_does_not_mutate_the_model_tally(
        self, install_host_config, install_model_config, make_grid
    ):
        """Unlike ``mem_check_run_all``, the build estimate is transient.

        Build-time memory is released before the solve; folding it into
        ``mem_use`` would double-count against the run-time check that
        follows.
        """
        install_host_config()
        model_config = install_model_config(mem_use=1000)

        mem_check_build_all([make_grid(basic=2000)])

        assert model_config.mem_use == 1000

    def test_it_does_not_mutate_the_grid_tally(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config()
        grid = make_grid(basic=2000)

        mem_check_build_all([grid])

        assert grid.mem_use == 0

    def test_dispersive_materials_are_not_counted(
        self, install_host_config, install_model_config, make_grid
    ):
        """They do not exist yet — materials are assigned after geometry."""
        install_host_config()
        install_model_config(maxpoles=3)

        total, _ = mem_check_build_all([make_grid(basic=1000, dispersive=9999)])

        assert total == 1000

    def test_one_string_is_produced_per_grid(
        self, install_host_config, install_model_config, make_grid
    ):
        install_host_config()
        install_model_config()

        _, strings = mem_check_build_all([make_grid(), make_grid()])

        assert len(strings) == 2

    def test_each_string_reports_only_its_own_grid(
        self, install_host_config, install_model_config, make_grid
    ):
        """A running total here would misattribute the second grid's size."""
        install_host_config()
        install_model_config()

        _, strings = mem_check_build_all(
            [make_grid(name="main", basic=1_000_000),
             make_grid(name="sub", basic=1_000_000)]
        )

        assert strings == ["~1.0 MB [main]", "~1.0 MB [sub]"]

    def test_an_oversized_build_warns(
        self, install_host_config, install_model_config, make_grid, caplog
    ):
        install_host_config()
        install_model_config()

        mem_check_build_all([make_grid(basic=32 * 1024**3)])

        assert "exceeds" in caplog.text

    def test_an_empty_grid_list_returns_the_existing_total(
        self, install_host_config, install_model_config
    ):
        install_host_config()
        install_model_config(mem_use=65_000_000)

        total, strings = mem_check_build_all([])

        assert (total, strings) == (65_000_000, [])
