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

"""``SimulationConfig`` — ``gprMax/config.py:196``.

The object built once per run, from the argument namespace the CLI or the API
supplies, and then read by every module in the package. It is almost entirely
derivation: forty-odd attributes computed from twenty arguments, plus five
validation checks.

Two things make it worth testing carefully.

**``em_consts`` is a class attribute**, not an instance one, so every
``SimulationConfig`` in the process shares one dictionary. Mutating a key
through one instance changes it for all of them, including for tests that ran
earlier.

**The validation is uneven.** Four of the five checks raise a *bare*
``ValueError`` whose reason exists only in the log, and the fifth — the one
meant to stop two accelerators being selected at once — never fires at all,
because it tests ``count(True)`` against values the CLI supplies as lists.

Precision and dtype selection is large enough to have its own file; see
``test_precision_dtypes.py``.
"""

import numpy as np
import pytest

from gprMax import config as config_module

from .conftest import FAKE_HOST_INFO


class TestDefaults:
    """A default run: no accelerator, no MPI, one model."""

    def test_constructs_from_the_shipped_argument_defaults(self, make_sim_config):
        """``args_defaults`` plus an input file is a valid configuration."""
        assert make_sim_config() is not None

    def test_solver_defaults_to_cpu(self, make_sim_config):
        assert make_sim_config().general["solver"] == "cpu"

    def test_precision_defaults_to_single(self, make_sim_config):
        """Note this differs from what the other test suites stub in.

        Every existing conftest supplies ``precision: "double"`` for its
        stand-in config; the real default is ``"single"``. Harmless, but it
        means those suites test the double-precision code path exclusively.
        """
        assert make_sim_config().general["precision"] == "single"

    def test_general_has_exactly_four_keys(self, make_sim_config):
        """``solver``, ``precision``, ``progressbars`` and ``subgrid``.

        The first three are set together; ``subgrid`` is added later in
        ``__init__``, which is why a stand-in that omits it still works for
        most consumers.
        """
        assert set(make_sim_config().general) == {
            "solver",
            "precision",
            "progressbars",
            "subgrid",
        }

    def test_subgrid_defaults_to_false(self, make_sim_config):
        assert make_sim_config().general["subgrid"] is False

    def test_current_model_starts_at_zero(self, make_sim_config):
        assert make_sim_config().current_model == 0

    def test_model_configs_is_sized_by_the_model_count(self, make_sim_config):
        """One slot per model, all empty until a model is built."""
        sim_config = make_sim_config(n=4)

        assert sim_config.model_configs == [None, None, None, None]

    def test_arguments_are_kept_for_later_lookup(self, make_sim_config):
        """Several consumers read ``sim_config.args`` directly."""
        sim_config = make_sim_config()

        assert sim_config.args.n == 1
        assert sim_config.args.inputfile == "model.in"

    @pytest.mark.parametrize(
        "attribute,expected",
        [
            ("geometry_fixed", False),
            ("geometry_only", False),
            ("gpu", None),
            ("mpi", None),
            ("number_of_models", 1),
            ("opencl", None),
            ("taskfarm", False),
            ("write_processed_input_file", False),
        ],
    )
    def test_argument_is_copied_onto_the_config(self, make_sim_config, attribute, expected):
        """Each of these is a straight copy from the namespace."""
        assert getattr(make_sim_config(), attribute) == expected

    def test_autotranslate_defaults_to_the_argument_value(self, make_sim_config):
        assert make_sim_config().autotranslate_subgrid_coordinates is False


class TestHostInfo:
    """The host probe runs unconditionally at construction."""

    def test_host_info_is_stored(self, make_sim_config):
        assert make_sim_config().hostinfo == FAKE_HOST_INFO

    def test_host_info_is_probed_exactly_once(self, monkeypatch, make_args):
        """One probe per ``SimulationConfig``, not per read."""
        calls = []

        monkeypatch.setattr(
            config_module,
            "get_host_info",
            lambda: calls.append(1) or dict(FAKE_HOST_INFO),
        )

        config_module.SimulationConfig(make_args())

        assert len(calls) == 1

    def test_host_info_is_probed_even_on_a_cpu_run(self, monkeypatch, make_args):
        """There is no way to opt out.

        A plain CPU run still shells out to ``wmic``/``sysctl``/``lscpu``,
        which is why the wmic removal broke gprMax at startup for every user
        rather than only for GPU users.
        """
        calls = []
        monkeypatch.setattr(
            config_module,
            "get_host_info",
            lambda: calls.append(1) or dict(FAKE_HOST_INFO),
        )

        config_module.SimulationConfig(make_args(gpu=None, opencl=None, metal=None))

        assert len(calls) == 1


class TestElectromagneticConstants:
    """``em_consts`` — four values, shared by every instance."""

    def test_has_exactly_four_keys(self, make_sim_config):
        assert set(make_sim_config().em_consts) == {"c", "e0", "m0", "z0"}

    def test_impedance_of_free_space_is_derived_from_the_others(self, make_sim_config):
        """``z0 = sqrt(m0 / e0)`` — about 376.73 ohms."""
        consts = make_sim_config().em_consts

        assert consts["z0"] == pytest.approx(np.sqrt(consts["m0"] / consts["e0"]))
        assert consts["z0"] == pytest.approx(376.73, abs=0.01)

    def test_speed_of_light_is_the_scipy_value(self, make_sim_config):
        from scipy.constants import c

        assert make_sim_config().em_consts["c"] == c

    def test_permittivity_is_the_scipy_value(self, make_sim_config):
        from scipy.constants import epsilon_0

        assert make_sim_config().em_consts["e0"] == epsilon_0

    def test_permeability_is_the_scipy_value(self, make_sim_config):
        from scipy.constants import mu_0

        assert make_sim_config().em_consts["m0"] == mu_0

    def test_em_consts_is_a_class_attribute_shared_by_all_instances(self, make_sim_config):
        """Every instance sees the same dictionary object.

        A test — or any caller — that mutated a key through one instance
        would change it for every other one, including instances created
        earlier. Nothing in the package does, but nothing prevents it either.
        """
        first = make_sim_config()
        second = make_sim_config()

        assert first.em_consts is second.em_consts
        assert first.em_consts is config_module.SimulationConfig.em_consts


class TestValidation:
    """The five guards in ``__init__``, and the one that does not work."""

    def test_taskfarm_with_fixed_geometry_is_rejected(self, make_sim_config):
        with pytest.raises(ValueError):
            make_sim_config(taskfarm=True, geometry_fixed=True)

    def test_taskfarm_alone_is_accepted(self, make_sim_config):
        assert make_sim_config(taskfarm=True) is not None

    def test_fixed_geometry_alone_is_accepted(self, make_sim_config):
        assert make_sim_config(geometry_fixed=True) is not None

    def test_showing_and_hiding_progress_bars_is_rejected(self, make_sim_config):
        with pytest.raises(ValueError):
            make_sim_config(show_progress_bars=True, hide_progress_bars=True)

    def test_mpi_with_subgrids_is_rejected(self, make_sim_config):
        with pytest.raises(ValueError):
            make_sim_config(mpi=[2], subgrid=True)

    def test_subgrid_with_an_accelerator_is_rejected(self, make_sim_config):
        """Sub-gridding needs double precision, which the GPU paths force
        to single — so the combination is refused rather than silently
        downgraded."""
        with pytest.raises(ValueError):
            make_sim_config(subgrid=True, gpu=[0])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"taskfarm": True, "geometry_fixed": True},
            {"show_progress_bars": True, "hide_progress_bars": True},
            {"mpi": [2], "subgrid": True},
        ],
    )
    def test_rejections_raise_a_bare_value_error(self, make_sim_config, kwargs):
        """No message on the exception — only in the log.

        A user running through the API sees ``ValueError`` with an empty
        string. The same pattern as ``check_kappamin`` in the PML, recorded
        during PR 10.
        """
        with pytest.raises(ValueError) as excinfo:
            make_sim_config(**kwargs)

        assert str(excinfo.value) == ""

    def test_the_reason_is_logged(self, make_sim_config, caplog):
        with caplog.at_level("ERROR", logger="gprMax.config"):
            with pytest.raises(ValueError):
                make_sim_config(taskfarm=True, geometry_fixed=True)

        assert "geometry fixed option cannot be used with MPI taskfarm" in caplog.text

    def test_combined_accelerators_now_rejected(self, make_sim_config):
        """Upstream fixed the guard: combined accelerators now raise ValueError."""
        with pytest.raises(ValueError):
            make_sim_config(gpu=[0], opencl=[0])

    def test_the_guard_does_fire_for_literal_booleans(self, make_sim_config):
        """It works only for a caller that passes ``True`` itself.

        Which nothing in gprMax does — establishing that the check is not
        broken so much as guarding a shape the codebase never produces.
        """
        with pytest.raises(ValueError):
            make_sim_config(gpu=True, opencl=True)


class TestProgressBars:
    """``progressbars`` is derived from three arguments and the log level."""

    def test_on_by_default_at_the_default_log_level(self, make_sim_config):
        """``log_level`` defaults to 20 (INFO), so bars are shown."""
        assert make_sim_config().general["progressbars"] is True

    def test_off_when_the_log_level_is_above_info(self, make_sim_config):
        """Above INFO the bars would interleave with sparse output."""
        assert make_sim_config(log_level=30).general["progressbars"] is False

    def test_on_when_explicitly_shown_even_at_a_high_log_level(self, make_sim_config):
        """``show_progress_bars`` wins over the log-level heuristic."""
        sim_config = make_sim_config(log_level=40, show_progress_bars=True)

        assert sim_config.general["progressbars"] is True

    def test_off_when_explicitly_hidden(self, make_sim_config):
        assert make_sim_config(hide_progress_bars=True).general["progressbars"] is False

    @pytest.mark.parametrize("log_level", [0, 10, 20])
    def test_on_at_or_below_info(self, make_sim_config, log_level):
        assert make_sim_config(log_level=log_level).general["progressbars"] is True

    @pytest.mark.parametrize("log_level", [21, 25, 30, 50])
    def test_off_above_info(self, make_sim_config, log_level):
        """25 is the custom ``BASIC`` level, which also suppresses bars."""
        assert make_sim_config(log_level=log_level).general["progressbars"] is False


class TestSolverSelection:
    """Which backend the arguments select."""

    def test_no_accelerator_argument_gives_cpu(self, make_sim_config):
        assert make_sim_config().general["solver"] == "cpu"

    def test_gpu_argument_selects_cuda(self, make_sim_config):
        assert make_sim_config(gpu=[0]).general["solver"] == "cuda"

    def test_opencl_argument_selects_opencl(self, make_sim_config):
        assert make_sim_config(opencl=[0]).general["solver"] == "opencl"

    def test_metal_argument_selects_metal(self, make_sim_config):
        assert make_sim_config(metal=[0]).general["solver"] == "metal"

    @pytest.mark.parametrize("accelerator", ["gpu", "opencl", "metal"])
    def test_every_accelerator_forces_single_precision(self, make_sim_config, accelerator):
        """Both precisions work on a GPU; single is chosen for speed."""
        sim_config = make_sim_config(**{accelerator: [0]})

        assert sim_config.general["precision"] == "single"

    def test_subgrids_force_double_precision(self, make_sim_config):
        """The Huygens sub-grid coupling is too ill-conditioned for float32."""
        assert make_sim_config(subgrid=True).general["precision"] == "double"

    def test_an_empty_device_list_still_selects_the_accelerator(self, make_sim_config):
        """The branch tests ``is not None``, not truthiness.

        ``-gpu`` with no device ID parses to ``[]``, which still means "use
        CUDA" and defaults to device 0 later.
        """
        assert make_sim_config(gpu=[]).general["solver"] == "cuda"

    @pytest.mark.parametrize("accelerator", ["gpu", "opencl", "metal"])
    def test_accelerator_runs_get_a_devices_dictionary(self, make_sim_config, accelerator):
        sim_config = make_sim_config(**{accelerator: [0]})

        assert "devs" in sim_config.devices

    def test_a_cpu_run_has_no_devices_attribute(self, make_sim_config):
        """``devices`` is only created on an accelerator path."""
        assert not hasattr(make_sim_config(), "devices")

    def test_cuda_devices_carry_compiler_options(self, make_sim_config):
        assert "nvcc_opts" in make_sim_config(gpu=[0]).devices

    def test_windows_suppresses_nvcc_warnings(self, make_sim_config, monkeypatch):
        """The one platform-conditional line in ``SimulationConfig``.

        ``sys.platform`` is patched rather than read, so this branch is
        covered on all three CI runners instead of one — and the *absence* of
        the flag elsewhere is covered too. Left unpatched, the assertion would
        depend on which runner executed it.
        """
        monkeypatch.setattr("sys.platform", "win32")

        assert make_sim_config(gpu=[0]).devices["nvcc_opts"] == ["-w"]

    @pytest.mark.parametrize("platform_name", ["linux", "darwin"])
    def test_other_platforms_pass_no_nvcc_options(
        self, make_sim_config, monkeypatch, platform_name
    ):
        monkeypatch.setattr("sys.platform", platform_name)

        assert make_sim_config(gpu=[0]).devices["nvcc_opts"] is None

    @pytest.mark.parametrize("accelerator", ["opencl", "metal"])
    def test_non_cuda_devices_carry_compiler_options(self, make_sim_config, accelerator):
        sim_config = make_sim_config(**{accelerator: [0]})

        assert "compiler_opts" in sim_config.devices


class TestInputFilePath:
    """``_set_input_file_path`` — where the model is read from."""

    def test_input_file_becomes_a_path(self, make_sim_config):
        from pathlib import Path

        assert make_sim_config().input_file_path == Path("model.in")

    def test_the_output_file_is_used_when_no_input_file_is_given(self, make_sim_config):
        """The API can supply a scene and an output name with no input file."""
        from pathlib import Path

        sim_config = make_sim_config(inputfile=None, outputfile="result.h5")

        assert sim_config.input_file_path == Path("result.h5")

    def test_neither_path_given_raises(self, make_sim_config):
        """``Path(None)`` — a reachable combination of the shipped defaults.

        ``args_defaults`` has both ``inputfile`` and ``outputfile`` set to
        ``None``, so constructing straight from the defaults fails with a
        ``TypeError`` about ``NoneType`` rather than a message naming the
        missing argument.
        """
        with pytest.raises(TypeError):
            make_sim_config(inputfile=None, outputfile=None)


class TestModelStartAndEnd:
    """``_set_model_start_end`` — which model numbers this run covers."""

    def test_a_single_model_run_spans_zero_to_one(self, make_sim_config):
        sim_config = make_sim_config(n=1)

        assert (sim_config.model_start, sim_config.model_end) == (0, 1)

    def test_a_multi_model_run_spans_zero_to_n(self, make_sim_config):
        sim_config = make_sim_config(n=5)

        assert (sim_config.model_start, sim_config.model_end) == (0, 5)

    def test_a_restart_index_shifts_the_range(self, make_sim_config):
        """``-i 3 -n 2`` resumes at model 3 and runs two models."""
        sim_config = make_sim_config(i=3, n=2)

        assert (sim_config.model_start, sim_config.model_end) == (2, 4)

    def test_a_restart_index_of_zero_now_rejected(self, make_sim_config):
        """Upstream now rejects i=0 — must be greater than zero."""
        with pytest.raises(ValueError):
            make_sim_config(i=0, n=3)

    def test_the_model_config_list_is_not_resized_for_a_restart(self, make_sim_config):
        """``model_configs`` is sized ``n`` but indices run to ``(i-1)+n``.

        With ``-i 5 -n 3`` the run iterates models 4, 5 and 6 while the list
        holds three slots, so storing the first config raises ``IndexError``.
        Asserted here as the arithmetic mismatch rather than by driving the
        failure, since that needs the whole context loop.

        Written up in ``notes/bugs/config-model-index-range-mismatch.md``.
        """
        sim_config = make_sim_config(i=5, n=3)

        assert len(sim_config.model_configs) == 3
        assert sim_config.model_end == 7
        assert sim_config.model_end > len(sim_config.model_configs)


class TestSceneStorage:
    """``scenes`` — one per model, supplied by the API or left empty."""

    def test_scenes_default_to_one_empty_slot_per_model(self, make_sim_config):
        assert make_sim_config(n=3).scenes == [None, None, None]

    def test_supplied_scenes_are_kept(self, make_sim_config):
        scenes = ["scene-a", "scene-b"]

        assert make_sim_config(n=2, scenes=scenes).scenes == scenes

    def test_a_scene_can_be_retrieved_by_model_number(self, make_sim_config):
        sim_config = make_sim_config(n=2, scenes=["first", "second"])

        assert sim_config.get_scene(1) == "second"

    def test_a_scene_can_be_stored_by_model_number(self, make_sim_config):
        sim_config = make_sim_config(n=2)

        sim_config.set_scene("late-scene", 1)

        assert sim_config.scenes[1] == "late-scene"

    def test_storing_a_scene_defaults_to_the_current_model(self, make_sim_config):
        sim_config = make_sim_config(n=2)
        sim_config.current_model = 1

        sim_config.set_scene("current")

        assert sim_config.scenes == [None, "current"]


pytestmark = pytest.mark.unit
