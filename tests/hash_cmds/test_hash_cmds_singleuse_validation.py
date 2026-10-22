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

import re
from pathlib import Path

import pytest

from gprMax.hash_cmds_singleuse import process_singlecmds
from gprMax.user_objects.cmds_singleuse import RxSteps, SrcSteps


def _parse(singlecmds, command, value):
    singlecmds[command] = value
    return process_singlecmds(singlecmds)


@pytest.mark.parametrize(
    "command,value",
    [
        ("#omp_threads", "1 2"),
        ("#num_threads", "1 2"),
        ("#dx_dy_dz", "0.1 0.1"),
        ("#domain_mode", "TM TE"),
        ("#magnetic_averaging", "harmonic arithmetic"),
        ("#dispersive_averaging", "y n"),
        ("#domain", "0.1 0.1"),
        ("#time_step_stability_factor", "0.5 0.6"),
        ("#time_window", "1e-9 2e-9"),
        ("#pml_cells", "1 2 3"),
        ("#src_steps", "0.1 0.2"),
        ("#rx_steps", "0.1 0.2"),
    ],
)
def test_arity_errors_name_the_command(singlecmds_template, command, value):
    with pytest.raises(ValueError, match=re.escape(command)) as excinfo:
        _parse(singlecmds_template, command, value)

    assert "parameter" in str(excinfo.value)


def test_arity_is_checked_before_numeric_conversion(singlecmds_template):
    with pytest.raises(ValueError, match=r"#dx_dy_dz requires exactly 3 parameters"):
        _parse(singlecmds_template, "#dx_dy_dz", "abc")


@pytest.mark.parametrize(
    "command,value,index",
    [
        ("#dx_dy_dz", "0.1 abc 0.1", 2),
        ("#domain", "0.1 abc 0.1", 2),
        ("#src_steps", "0.1 abc 0.1", 2),
        ("#rx_steps", "0.1 abc 0.1", 2),
        ("#time_step_stability_factor", "abc", 1),
    ],
)
def test_non_numeric_values_report_the_parameter(singlecmds_template, command, value, index):
    message = rf"{re.escape(command)} parameter {index} is not a number"
    with pytest.raises(ValueError, match=message):
        _parse(singlecmds_template, command, value)


@pytest.mark.parametrize(
    "command,value,index",
    [
        ("#omp_threads", "2.5", 1),
        ("#num_threads", "half", 1),
        ("#pml_cells", "10.0", 1),
        ("#pml_cells", "10 10 10 10 10 1.5", 6),
    ],
)
def test_non_integer_values_report_the_parameter(singlecmds_template, command, value, index):
    message = rf"{re.escape(command)} parameter {index} is not an integer"
    with pytest.raises(ValueError, match=message):
        _parse(singlecmds_template, command, value)


def test_time_window_rejects_non_numeric_value(singlecmds_template):
    with pytest.raises(ValueError, match="seconds or an iteration count"):
        _parse(singlecmds_template, "#time_window", "abc")


def test_time_window_rejects_multiple_values(singlecmds_template):
    with pytest.raises(ValueError, match=r"#time_window requires exactly one parameter"):
        _parse(singlecmds_template, "#time_window", "1e-9 2e-9")


def test_dispersive_averaging_rejects_unknown_choice(singlecmds_template):
    with pytest.raises(ValueError, match=r"#dispersive_averaging requires 'y' or 'n'"):
        _parse(singlecmds_template, "#dispersive_averaging", "maybe")


@pytest.mark.parametrize("command", ["#src_steps", "#rx_steps"])
@pytest.mark.parametrize("value", ["nan 0 0", "0 inf 0", "0 0 -inf"])
def test_step_commands_reject_non_finite_values(singlecmds_template, command, value):
    with pytest.raises(ValueError, match="must be finite"):
        _parse(singlecmds_template, command, value)


@pytest.mark.parametrize("command,object_type", [("#src_steps", SrcSteps), ("#rx_steps", RxSteps)])
@pytest.mark.parametrize("value", ["0 0 0", "-0.1 0.2 -0.3"])
def test_step_commands_accept_finite_offsets(singlecmds_template, command, object_type, value):
    objects = _parse(singlecmds_template, command, value)

    assert isinstance(objects[0], object_type)
    assert objects[0].step_size == tuple(float(token) for token in value.split())


def test_omp_threads_rejects_its_legacy_alias(singlecmds_template):
    singlecmds_template["#omp_threads"] = "2"
    singlecmds_template["#num_threads"] = "2"

    with pytest.raises(ValueError, match="cannot both be specified"):
        process_singlecmds(singlecmds_template)


def test_output_dir_is_relative_to_the_input_file(singlecmds_template, tmp_path):
    singlecmds_template["#output_dir"] = "results"

    objects = process_singlecmds(singlecmds_template, input_dir=tmp_path)

    assert Path(objects[0].output_dir) == tmp_path / "results"


def test_output_dir_preserves_absolute_paths(singlecmds_template, tmp_path):
    output_dir = tmp_path / "results"
    singlecmds_template["#output_dir"] = str(output_dir)

    objects = process_singlecmds(singlecmds_template, input_dir=tmp_path / "model")

    assert Path(objects[0].output_dir) == output_dir


pytestmark = pytest.mark.unit
