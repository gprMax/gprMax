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

"""Tests for ``gprMax.hash_cmds_singleuse.process_singlecmds``.

``process_singlecmds`` is a single dispatch function. For every
once-per-model command it checks whether the corresponding entry in the
input dict is ``None``; if not it ``split()``-tokenises the value,
validates arity, and appends a fresh user-object to the result list.

These tests drive the dispatcher with hand-built dicts so each command's
branch is exercised in isolation — no file I/O, no globals.
"""

import pytest

from gprMax.hash_cmds_multiuse import process_multicmds
from gprMax.hash_cmds_singleuse import process_singlecmds
from gprMax.user_objects.cmds_singleuse import (
    Discretisation,
    Domain,
    OMPThreads,
    OutputDir,
    PMLFormulation,
    PMLThickness,
    RxSteps,
    SrcSteps,
    TimeStepStabilityFactor,
    TimeWindow,
    Title,
)


class TestNonePassthrough:
    """Every entry left as ``None`` must short-circuit — no user objects."""

    def test_all_none_yields_empty_list(self, singlecmds_template):
        # All keys default to None in the template fixture
        assert process_singlecmds(singlecmds_template) == []

    def test_single_command_does_not_create_others(self, singlecmds_template):
        singlecmds_template["#title"] = "demo"
        objs = process_singlecmds(singlecmds_template)
        # Exactly one Title; nothing else
        assert len(objs) == 1
        assert isinstance(objs[0], Title)


class TestTitle:
    def test_title_string_stored(self, singlecmds_template):
        singlecmds_template["#title"] = "my model"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], Title)
        assert objs[0].title == "my model"
        assert objs[0].kwargs["name"] == "my model"

    def test_title_cast_to_str(self, singlecmds_template):
        # Dispatcher wraps in ``str(...)`` so even non-string values survive
        singlecmds_template["#title"] = 12345
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], Title)
        assert objs[0].title == "12345"


class TestOutputDir:
    def test_output_dir_stored(self, singlecmds_template):
        singlecmds_template["#output_dir"] = "results/run1"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], OutputDir)
        # OutputDir kwarg is ``dir``
        assert objs[0].kwargs["dir"] == "results/run1"


class TestOMPThreads:
    def test_single_thread_count_accepted(self, singlecmds_template):
        singlecmds_template["#omp_threads"] = "4"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], OMPThreads)
        assert objs[0].omp_threads == 4

    def test_two_tokens_rejected(self, singlecmds_template):
        singlecmds_template["#omp_threads"] = "4 8"
        with pytest.raises(ValueError):
            process_singlecmds(singlecmds_template)

    def test_non_integer_token_rejected(self, singlecmds_template):
        singlecmds_template["#omp_threads"] = "abc"
        with pytest.raises(ValueError):
            process_singlecmds(singlecmds_template)


class TestDiscretisation:
    def test_three_floats_become_tuple(self, singlecmds_template):
        singlecmds_template["#dx_dy_dz"] = "0.001 0.002 0.004"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], Discretisation)
        assert objs[0].discretisation == (0.001, 0.002, 0.004)
        assert objs[0].kwargs["p1"] == (0.001, 0.002, 0.004)

    @pytest.mark.parametrize("payload", ["0.001 0.002", "0.001 0.002 0.003 0.004"])
    def test_wrong_arity_rejected(self, singlecmds_template, payload):
        singlecmds_template["#dx_dy_dz"] = payload
        with pytest.raises(ValueError):
            process_singlecmds(singlecmds_template)


class TestDomain:
    def test_three_floats_become_tuple(self, singlecmds_template):
        singlecmds_template["#domain"] = "0.2 0.3 0.4"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], Domain)
        assert objs[0].domain_size == (0.2, 0.3, 0.4)
        assert objs[0].kwargs["p1"] == (0.2, 0.3, 0.4)

    @pytest.mark.parametrize("payload", ["0.2 0.3", "0.2 0.3 0.4 0.5"])
    def test_wrong_arity_rejected(self, singlecmds_template, payload):
        singlecmds_template["#domain"] = payload
        with pytest.raises(ValueError):
            process_singlecmds(singlecmds_template)


class TestTimeStepStabilityFactor:
    def test_first_token_becomes_float_factor(self, singlecmds_template):
        # ``tmp[0]`` is used regardless of arity in this branch
        singlecmds_template["#time_step_stability_factor"] = "0.5"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], TimeStepStabilityFactor)
        assert objs[0].stability_factor == 0.5


class TestTimeWindow:
    """``#time_window`` does dual-mode dispatch on ``int(token)`` success."""

    def test_integer_token_routes_to_iterations(self, singlecmds_template):
        singlecmds_template["#time_window"] = "100"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], TimeWindow)
        assert objs[0].iterations == 100
        assert objs[0].time is None

    def test_float_token_routes_to_time(self, singlecmds_template):
        singlecmds_template["#time_window"] = "1e-9"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], TimeWindow)
        assert objs[0].time == 1e-9
        assert objs[0].iterations is None

    def test_decimal_token_routes_to_time(self, singlecmds_template):
        # ``int("5.0")`` raises ValueError so the except branch handles it
        singlecmds_template["#time_window"] = "5.0"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], TimeWindow)
        assert objs[0].time == 5.0
        assert objs[0].iterations is None

    def test_lowercase_normalisation_does_not_strip_sign(self, singlecmds_template):
        # ``.lower()`` is called on the token before the int/float try
        singlecmds_template["#time_window"] = "1E-9"
        objs = process_singlecmds(singlecmds_template)
        assert objs[0].time == 1e-9

    def test_multi_token_rejected(self, singlecmds_template):
        singlecmds_template["#time_window"] = "1e-9 extra"
        with pytest.raises(ValueError):
            process_singlecmds(singlecmds_template)

    def test_garbage_token_rejected(self, singlecmds_template):
        # Neither int() nor float() can parse — the float() retry re-raises
        singlecmds_template["#time_window"] = "abc"
        with pytest.raises(ValueError):
            process_singlecmds(singlecmds_template)


class TestPMLFormulation:
    """``#pml_formulation`` — moved to multi-use dispatcher upstream."""

    def test_formulation_string_stored(self, multicmds_template):
        multicmds_template["#pml_formulation"] = ["HORIPML"]
        objs = process_multicmds(multicmds_template)
        assert len(objs) == 1
        assert isinstance(objs[0], PMLFormulation)
        assert objs[0].formulation == "HORIPML"

    def test_multi_token_now_accepted(self, multicmds_template):
        """Multi-token ``#pml_formulation`` is now accepted by the
        multi-use dispatcher (extraneous tokens are silently stored)."""
        multicmds_template["#pml_formulation"] = ["HORIPML extra"]
        objs = process_multicmds(multicmds_template)
        assert isinstance(objs[0], PMLFormulation)


class TestPMLCells:
    """``#pml_cells`` accepts 1 (uniform) or 6 (per-face) tokens."""

    def test_uniform_thickness_single_token(self, singlecmds_template):
        singlecmds_template["#pml_cells"] = "10"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], PMLThickness)
        assert objs[0].thickness == 10

    @pytest.mark.parametrize("payload", ["10 10", "10 10 10", "10 10 10 10", "10 10 10 10 10"])
    def test_invalid_arity_rejected(self, singlecmds_template, payload):
        singlecmds_template["#pml_cells"] = payload
        with pytest.raises(ValueError):
            process_singlecmds(singlecmds_template)


class TestPMLCellsSixArgBranchBug:
    """Upstream fixed the 6-token ``#pml_cells`` branch.
    Previously raised ``TypeError`` (unknown kwargs); now dispatches correctly."""

    def test_six_token_form_now_works(self, singlecmds_template):
        singlecmds_template["#pml_cells"] = "10 10 10 10 10 10"
        objs = process_singlecmds(singlecmds_template)
        assert len(objs) == 1
        assert isinstance(objs[0], PMLThickness)


class TestSrcSteps:
    def test_three_floats_become_tuple(self, singlecmds_template):
        singlecmds_template["#src_steps"] = "0.01 0.02 0.03"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], SrcSteps)
        assert objs[0].kwargs["p1"] == (0.01, 0.02, 0.03)

    @pytest.mark.parametrize("payload", ["0.01 0.02", "0.01 0.02 0.03 0.04"])
    def test_wrong_arity_rejected(self, singlecmds_template, payload):
        singlecmds_template["#src_steps"] = payload
        with pytest.raises(ValueError):
            process_singlecmds(singlecmds_template)


class TestRxSteps:
    def test_three_floats_become_tuple(self, singlecmds_template):
        singlecmds_template["#rx_steps"] = "0.01 0.02 0.03"
        objs = process_singlecmds(singlecmds_template)
        assert isinstance(objs[0], RxSteps)
        assert objs[0].kwargs["p1"] == (0.01, 0.02, 0.03)

    @pytest.mark.parametrize("payload", ["0.01 0.02", "0.01 0.02 0.03 0.04"])
    def test_wrong_arity_rejected(self, singlecmds_template, payload):
        singlecmds_template["#rx_steps"] = payload
        with pytest.raises(ValueError):
            process_singlecmds(singlecmds_template)


class TestObjectOrder:
    """Dispatcher walks commands in a fixed source-defined order.

    This is observable behaviour (downstream code relies on ``Title``
    landing before ``Discretisation``). If someone re-orders the if-chain
    in ``process_singlecmds`` this test flags the change.
    """

    def test_title_then_output_dir_then_threads_then_grid(self, singlecmds_template):
        singlecmds_template["#title"] = "demo"
        singlecmds_template["#output_dir"] = "out"
        singlecmds_template["#omp_threads"] = "2"
        singlecmds_template["#dx_dy_dz"] = "0.001 0.001 0.001"
        singlecmds_template["#domain"] = "0.1 0.1 0.1"
        singlecmds_template["#time_window"] = "100"

        objs = process_singlecmds(singlecmds_template)
        types = [type(o) for o in objs]
        assert types == [
            Title,
            OutputDir,
            OMPThreads,
            Discretisation,
            Domain,
            TimeWindow,
        ]


pytestmark = pytest.mark.unit
