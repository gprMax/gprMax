"""``ModelConfig`` — ``gprMax/config.py:43``.

One of these exists per model in a run. It carries the per-model mutable
state: the material summary the update kernels dispatch on, the numerical
dispersion thresholds, the memory tally, and the banner printed at the top of
each model.

The construction-order constraint is the thing to understand first.
``__init__`` reads the module-level ``sim_config`` three times — for
``model_end`` and ``input_file_path`` in the banner, and for ``args.n`` when
deciding whether to number the output file. So a ``ModelConfig`` cannot be
built before a ``SimulationConfig`` has been installed as the global. The
object under test depends on the global it is part of, which is why this
suite builds real objects rather than the stand-ins every other directory
uses.

Output-path construction is large enough for its own file; see
``test_output_paths.py``. The registry that decides *which* ``ModelConfig``
``get_model_config()`` returns is in ``test_model_registry.py``.
"""

import numpy as np
import pytest


class TestConstruction:
    """What ``ModelConfig(n)`` sets, and what it needs first."""

    def test_constructs_with_a_simulation_config_installed(self, make_model_config):
        assert make_model_config(0) is not None

    def test_requires_the_global_simulation_config(self, install_sim_config):
        """Without the global, construction fails on attribute access.

        ``ModelConfig.__init__`` reads ``sim_config.model_end`` while
        building the banner string, so a ``None`` global raises here rather
        than at first use.
        """
        from gprMax import config

        config.sim_config = None

        with pytest.raises(AttributeError):
            config.ModelConfig(0)

    def test_model_number_is_stored_as_given(self, make_model_config):
        """Zero-based internally; the banner adds one for display."""
        assert make_model_config(3).model_num == 3

    def test_mode_defaults_to_three_dimensional(self, make_model_config):
        """``2D`` is selected later, by the grid, if a dimension is one cell."""
        assert make_model_config(0).mode == "3D"

    def test_grids_starts_empty(self, make_model_config):
        assert make_model_config(0).grids == []

    def test_thread_count_starts_unset(self, make_model_config):
        """``set_omp_threads`` fills this in once the host is known.

        Until then it is ``None``, and passing ``None`` into a Cython
        ``int nthreads`` parameter raises ``TypeError`` — which is why every
        test suite that drives a kernel has to set it explicitly.
        """
        assert make_model_config(0).ompthreads is None

    def test_a_cpu_run_has_no_device_attribute(self, make_model_config):
        """``device`` is only created on the CUDA/OpenCL/Metal paths.

        Worth pinning because the stand-in config in the PR 10 outputs suite
        supplies a ``device`` key unconditionally — the real object does not
        have one on a CPU run.
        """
        assert not hasattr(make_model_config(0), "device")

    def test_an_accelerator_run_has_a_three_key_device_dictionary(
        self, make_model_config, monkeypatch, fake_device
    ):
        """``dev``, ``deviceID`` and ``snapsgpu2cpu``."""
        from gprMax import config

        monkeypatch.setattr(config, "detect_cuda_gpus", lambda: {0: fake_device()})

        model_config = make_model_config(0, gpu=[0])

        assert set(model_config.device) == {"dev", "deviceID", "snapsgpu2cpu"}

    def test_snapshot_transfer_starts_disabled(self, make_model_config, monkeypatch, fake_device):
        """Enabled later only if snapshots would not fit in device memory."""
        from gprMax import config

        monkeypatch.setattr(config, "detect_cuda_gpus", lambda: {0: fake_device()})

        assert make_model_config(0, gpu=[0]).device["snapsgpu2cpu"] is False


class TestMemoryTally:
    """``mem_overhead`` / ``mem_use`` — the running estimate."""

    def test_overhead_is_sixty_five_megabytes(self, make_model_config):
        """The comment above it says 50 MB; the value is 65e6."""
        assert make_model_config(0).mem_overhead == 65e6

    def test_usage_starts_at_the_overhead(self, make_model_config):
        """Estimates accumulate on top of a fixed baseline."""
        model_config = make_model_config(0)

        assert model_config.mem_use == model_config.mem_overhead

    def test_usage_is_a_mutable_running_total(self, make_model_config):
        """``mem_check_run_all`` adds to this in place."""
        model_config = make_model_config(0)

        model_config.mem_use += 1_000_000

        assert model_config.mem_use == 66e6


class TestNumericalDispersion:
    """``numdispersion`` — three thresholds for the dispersion analysis."""

    def test_has_exactly_three_keys(self, make_model_config):
        assert set(make_model_config(0).numdispersion) == {
            "highestfreqthres",
            "maxnumericaldisp",
            "mingridsampling",
        }

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("highestfreqthres", 40),
            ("maxnumericaldisp", 2),
            ("mingridsampling", 3),
        ],
    )
    def test_default_threshold(self, make_model_config, key, expected):
        """40 dB down from peak power, 2% phase error, 3 cells per wavelength."""
        assert make_model_config(0).numdispersion[key] == expected


class TestMaterials:
    """``materials`` — the summary the update dispatchers read."""

    def test_has_exactly_five_keys(self, make_model_config):
        assert set(make_model_config(0).materials) == {
            "maxpoles",
            "dispersivedtype",
            "dispersiveCdtype",
            "drudelorentz",
            "crealfunc",
        }

    def test_pole_count_starts_at_zero(self, make_model_config):
        """Which routes ``update_electric_a`` to the plain kernel."""
        assert make_model_config(0).materials["maxpoles"] == 0

    @pytest.mark.parametrize(
        "key", ["dispersivedtype", "dispersiveCdtype", "drudelorentz", "crealfunc"]
    )
    def test_derived_entry_starts_as_none(self, make_model_config, key):
        """All four are filled in by ``set_dispersive_material_types``.

        ``dispersivedtype`` starting as ``None`` is the reason
        ``set_dispersive_updates`` silently selects a *real* kernel when it
        runs first — see
        ``tests/unit/updates/test_dispersive_dispatch.py``.
        """
        assert make_model_config(0).materials[key] is None


class TestSetDispersiveMaterialTypes:
    """``set_dispersive_material_types`` — real or complex poles."""

    def test_debye_materials_get_the_real_dtype(self, make_model_config):
        """Debye poles are purely relaxational, so real arithmetic suffices."""
        model_config = make_model_config(0)
        model_config.materials["drudelorentz"] = False

        model_config.set_dispersive_material_types()

        assert model_config.materials["dispersivedtype"] is np.float32

    def test_drude_or_lorentz_materials_get_the_complex_dtype(self, make_model_config):
        """Those two have resonant poles, which need complex arithmetic."""
        model_config = make_model_config(0)
        model_config.materials["drudelorentz"] = True

        model_config.set_dispersive_material_types()

        assert model_config.materials["dispersivedtype"] is np.complex64

    def test_real_path_uses_an_empty_real_extraction(self, make_model_config):
        """``crealfunc`` is pasted into GPU kernel source.

        For a real dtype there is nothing to extract, so the substitution is
        the empty string rather than a no-op call.
        """
        model_config = make_model_config(0)
        model_config.materials["drudelorentz"] = False

        model_config.set_dispersive_material_types()

        assert model_config.materials["crealfunc"] == ""

    def test_complex_path_extracts_the_real_component(self, make_model_config):
        model_config = make_model_config(0)
        model_config.materials["drudelorentz"] = True

        model_config.set_dispersive_material_types()

        assert model_config.materials["crealfunc"] == ".real()"

    def test_the_dtype_matches_the_configured_precision(self, make_model_config):
        """Double-precision runs get ``complex128``, not ``complex64``."""
        model_config = make_model_config(0, subgrid=True)
        model_config.materials["drudelorentz"] = True

        model_config.set_dispersive_material_types()

        assert model_config.materials["dispersivedtype"] is np.complex128

    def test_the_c_dtype_is_set_alongside_the_numpy_one(self, make_model_config):
        model_config = make_model_config(0)
        model_config.materials["drudelorentz"] = False

        model_config.set_dispersive_material_types()

        assert model_config.materials["dispersiveCdtype"] == "float"

    def test_an_unset_drudelorentz_flag_takes_the_real_path(self, make_model_config):
        """``None`` is falsy, so the default takes the Debye branch.

        Benign — a model with no dispersive materials at all never reaches a
        dispersive kernel — but it means the flag has three states and only
        two behaviours.
        """
        model_config = make_model_config(0)
        assert model_config.materials["drudelorentz"] is None

        model_config.set_dispersive_material_types()

        assert model_config.materials["dispersivedtype"] is np.float32

    def test_the_result_agrees_with_the_kernel_dispatch(self, make_model_config):
        """The knot with ``set_dispersive_updates``.

        The dispatcher decides ``real`` versus ``complex`` by comparing
        ``dispersivedtype`` against ``sim_config.dtypes["complex"]``. That
        comparison is only meaningful if this method wrote one of exactly
        those two values, which it does.
        """
        from gprMax import config

        for drudelorentz, expected_key in ((True, "complex"), (False, "float_or_double")):
            model_config = make_model_config(0)
            model_config.materials["drudelorentz"] = drudelorentz
            model_config.set_dispersive_material_types()

            assert (
                model_config.materials["dispersivedtype"] is config.sim_config.dtypes[expected_key]
            )


class TestBanner:
    """``inputfilestr`` — the header printed before each model runs."""

    def test_contains_the_one_based_model_number(self, make_model_config):
        """Displayed as ``Model 3/5`` for internal index 2."""
        model_config = make_model_config(2, n=5)

        assert "Model 3/5" in model_config.inputfilestr

    def test_first_model_displays_as_one(self, make_model_config):
        assert "Model 1/1" in make_model_config(0).inputfilestr

    def test_contains_the_input_file_path(self, make_model_config):
        assert "model.in" in make_model_config(0).inputfilestr

    def test_is_padded_to_the_terminal_width(self, make_model_config):
        """The trailing rule fills the line.

        ``get_terminal_width`` is patched to a fixed value by the suite's
        autouse fixture — the real one differs between an interactive shell,
        pytest with no tty, and each CI runner, so an unpinned assertion
        here would be flaky across the three OSes.
        """
        from .conftest import TERMINAL_WIDTH

        banner = make_model_config(0).inputfilestr
        rule = [line for line in banner.splitlines() if "---" in line][0]

        # Strip the colour escapes the banner is wrapped in.
        visible = rule.replace("\x1b[32m", "").replace("\x1b[0m", "")
        assert len(visible) == TERMINAL_WIDTH - 1

    def test_is_wrapped_in_colour_codes(self, make_model_config):
        """Green, reset — colorama constants."""
        banner = make_model_config(0).inputfilestr

        assert banner.startswith("\x1b[32m")
        assert banner.endswith("\x1b[0m")


class TestGeometryReuse:
    """``reuse_geometry`` — skip rebuilding for later models."""

    def test_the_first_model_never_reuses(self, make_model_config):
        """There is nothing to reuse yet."""
        assert make_model_config(0, geometry_fixed=True).reuse_geometry() is False

    def test_a_later_model_reuses_when_the_flag_is_set(self, make_model_config):
        assert make_model_config(1, geometry_fixed=True).reuse_geometry() is True

    def test_a_later_model_does_not_reuse_by_default(self, make_model_config):
        assert make_model_config(1).reuse_geometry() is False

    @pytest.mark.parametrize("model_num", [1, 2, 7])
    def test_every_model_after_the_first_reuses(self, make_model_config, model_num):
        assert make_model_config(model_num, geometry_fixed=True).reuse_geometry() is True


class TestUserNamespace:
    """``get_usernamespace`` — the names visible to deprecated
    ``#python`` blocks."""

    def test_includes_the_electromagnetic_constants(self, make_model_config):
        namespace = make_model_config(0).get_usernamespace()

        assert {"c", "e0", "m0", "z0"} <= set(namespace)

    def test_includes_the_run_counters(self, make_model_config):
        namespace = make_model_config(1, n=4).get_usernamespace()

        assert namespace["number_model_runs"] == 4
        assert namespace["current_model_run"] == 2

    def test_model_run_number_is_one_based(self, make_model_config):
        assert make_model_config(0).get_usernamespace()["current_model_run"] == 1

    def test_input_file_is_absolute(self, make_model_config):
        """``resolve()`` is called, so the path is made absolute.

        This touches the filesystem — the only such call on the read path of
        a ``ModelConfig``.
        """
        namespace = make_model_config(0).get_usernamespace()

        assert namespace["inputfile"].is_absolute()

    def test_has_exactly_seven_names(self, make_model_config):
        namespace = make_model_config(0).get_usernamespace()

        assert set(namespace) == {
            "c",
            "e0",
            "m0",
            "z0",
            "number_model_runs",
            "current_model_run",
            "inputfile",
        }


pytestmark = pytest.mark.unit
