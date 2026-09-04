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

"""Tests for ``gprMax.hash_cmds_file``.

The "front door" of the hash parser: turns a `.in` text file (or any
``io.StringIO``) into the three command dicts the downstream dispatchers
consume.

Coverage targets:

* ``process_python_include_code`` — strips comments and blank lines,
  executes any ``#python:`` blocks and captures the commands they ``print``,
  preserves regular hash lines, then re-runs include-file resolution.
* ``process_include_files`` — replaces ``#include_file: path`` lines with
  the contents of the named file; falls back to ``input_file_path.parent``
  if the path isn't absolute and doesn't exist relative to CWD.
* ``check_cmd_names`` — routes each command line into the singleuse /
  multiuse / geometry buckets and validates command names, the
  command-name/parameters split, single-instance rule, and essentials
  presence.
* ``get_user_objects`` — end-to-end glue across the three dispatchers.
"""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gprMax.hash_cmds_file import (
    check_cmd_names,
    get_user_objects,
    process_include_files,
    process_python_include_code,
)
from gprMax.user_objects.cmds_singleuse import Discretisation, Domain, TimeWindow

# ---------------------------------------------------------------------------
# process_python_include_code
# ---------------------------------------------------------------------------


class TestProcessPythonIncludeCode:
    def test_double_hash_comments_stripped(self):
        # Lines beginning with ``##`` are comments; everything else stays
        text = io.StringIO(
            "## this is a header comment\n"
            "#title: demo\n"
            "## another comment\n"
            "#domain: 0.1 0.1 0.1\n"
        )
        out = process_python_include_code(text, {})
        assert out == ["#title: demo\n", "#domain: 0.1 0.1 0.1\n"]

    def test_blank_lines_stripped(self):
        text = io.StringIO("\n#title: demo\n\n\n#domain: 0.1 0.1 0.1\n")
        out = process_python_include_code(text, {})
        assert out == ["#title: demo\n", "#domain: 0.1 0.1 0.1\n"]

    def test_non_hash_lines_dropped(self):
        # Plain prose without leading ``#`` is silently skipped
        text = io.StringIO("just some text\n#title: demo\n")
        out = process_python_include_code(text, {})
        assert out == ["#title: demo\n"]

    def test_python_block_emits_printed_commands(self):
        # The block runs through ``exec``; its ``print`` output is sliced
        # into commands (lines starting with #) and other lines (echoed
        # back as info messages).
        text = io.StringIO(
            "#python:\n"
            "print('#title: from python')\n"
            "print('#domain: 0.2 0.2 0.2')\n"
            "#end_python:\n"
        )
        out = process_python_include_code(text, {})
        # Both commands are picked up, in order, each with a trailing \n
        assert out == ["#title: from python\n", "#domain: 0.2 0.2 0.2\n"]

    def test_python_block_namespace_passthrough(self):
        # ``usernamespace`` is exposed to the executed block
        text = io.StringIO("#python:\n" "print(f'#title: hello {NAME}')\n" "#end_python:\n")
        out = process_python_include_code(text, {"NAME": "world"})
        assert out == ["#title: hello world\n"]

    def test_missing_end_python_raises_syntax_error(self):
        text = io.StringIO("#python:\nprint('forgot to close')\n")
        with pytest.raises(SyntaxError):
            process_python_include_code(text, {})

    def test_stdout_reset_to_os_stdout_after_python_block(self):
        # The internal redirect to a StringIO is reverted with the
        # explicit ``sys.stdout = sys.__stdout__`` line — so after the
        # block, ``sys.stdout`` is the OS stdout (not whatever wrapper
        # the surrounding test runner had in place).
        text = io.StringIO("#python:\n" "print('#title: demo')\n" "#end_python:\n")
        process_python_include_code(text, {})
        assert sys.stdout is sys.__stdout__


# ---------------------------------------------------------------------------
# process_include_files
# ---------------------------------------------------------------------------


class TestProcessIncludeFiles:
    def test_no_include_lines_passes_through(self):
        cmds = ["#title: demo\n", "#domain: 0.1 0.1 0.1\n"]
        assert process_include_files(cmds) == cmds

    def test_include_file_inlines_contents(self, tmp_path):
        included = tmp_path / "extra.in"
        included.write_text("#title: from include\n#domain: 0.2 0.2 0.2\n")
        cmds = [f"#include_file: {included}\n"]

        out = process_include_files(cmds)
        # Both lines from the included file appear, each newline-terminated
        assert out == ["#title: from include\n", "#domain: 0.2 0.2 0.2\n"]

    def test_include_file_drops_comments_and_blanks(self, tmp_path):
        included = tmp_path / "with_comments.in"
        included.write_text("## this is a comment\n" "\n" "#title: kept\n" "## trailing comment\n")
        cmds = [f"#include_file: {included}\n"]
        out = process_include_files(cmds)
        assert out == ["#title: kept\n"]

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError):
            process_include_files(["#include_file: a b\n"])

    def test_relative_path_falls_back_to_input_file_parent(self, tmp_path, monkeypatch):
        # Place the included file next to a faked input file
        included = tmp_path / "sidecar.in"
        included.write_text("#title: sidecar wins\n")
        fake_input = tmp_path / "main.in"
        # Path the dispatcher reads when the requested path doesn't exist
        from gprMax import config

        fake_sim_config = SimpleNamespace(input_file_path=fake_input)
        monkeypatch.setattr(config, "sim_config", fake_sim_config)

        # Pass only the bare filename — first attempt (relative to CWD) misses,
        # then the fallback to ``input_file_path.parent`` resolves
        cmds = ["#include_file: sidecar.in\n"]
        out = process_include_files(cmds)
        assert out == ["#title: sidecar wins\n"]


# ---------------------------------------------------------------------------
# check_cmd_names
# ---------------------------------------------------------------------------


class TestCheckCmdNames:
    def _essentials(self):
        # Minimum set so ``checkessential=True`` is satisfied
        return [
            "#domain: 0.1 0.1 0.1\n",
            "#dx_dy_dz: 0.001 0.001 0.001\n",
            "#time_window: 100\n",
        ]

    def test_single_cmd_routed_to_single_dict(self):
        lines = self._essentials() + ["#title: my model\n"]
        single, multi, geometry = check_cmd_names(lines)
        assert single["#title"] == "my model"
        assert single["#domain"] == "0.1 0.1 0.1"
        # Empty / untouched keys remain at their defaults
        assert single["#omp_threads"] is None
        # Multi/geometry buckets remain empty
        assert all(v == [] for v in multi.values())
        assert geometry == []

    def test_colons_in_single_command_parameters_are_preserved(self):
        lines = self._essentials() + ["#title: survey: line 1\n"]
        single, _, _ = check_cmd_names(lines)
        assert single["#title"] == "survey: line 1"

    def test_windows_drive_colon_is_not_treated_as_another_separator(self):
        lines = self._essentials() + ["#output_dir: C:\\gprmax\\results\n"]
        single, _, _ = check_cmd_names(lines)
        assert single["#output_dir"] == "C:\\gprmax\\results"

    def test_multi_cmd_appended_to_multi_dict(self):
        lines = self._essentials() + [
            "#waveform: gaussian 1.0 1e9 wf1\n",
            "#waveform: ricker 2.0 5e8 wf2\n",
        ]
        _, multi, _ = check_cmd_names(lines)
        assert multi["#waveform"] == [
            "gaussian 1.0 1e9 wf1",
            "ricker 2.0 5e8 wf2",
        ]

    def test_geometry_cmd_appended_to_geometry_list(self):
        lines = self._essentials() + [
            "#box: 0 0 0 0.1 0.1 0.1 m1\n",
            "#sphere: 0.05 0.05 0.05 0.02 m1\n",
        ]
        _, _, geometry = check_cmd_names(lines)
        assert geometry == [
            "#box: 0 0 0 0.1 0.1 0.1 m1",
            "#sphere: 0.05 0.05 0.05 0.02 m1",
        ]

    def test_duplicate_single_cmd_rejected(self):
        lines = self._essentials() + [
            "#title: first\n",
            "#title: second\n",
        ]
        with pytest.raises(SyntaxError):
            check_cmd_names(lines)

    def test_unknown_command_rejected(self):
        lines = self._essentials() + ["#not_a_real_command: 1 2 3\n"]
        with pytest.raises(SyntaxError):
            check_cmd_names(lines)

    def test_missing_space_after_colon_rejected(self):
        # The dispatcher requires a space between the colon and the
        # parameters (or no parameters at all).
        lines = self._essentials() + ["#title:demo\n"]
        with pytest.raises(SyntaxError):
            check_cmd_names(lines)

    def test_missing_colon_exits(self):
        # No ``:`` in the line → ``exit(1)`` (raises SystemExit)
        lines = self._essentials() + ["#title is wrong\n"]
        with pytest.raises(SystemExit):
            check_cmd_names(lines)

    def test_missing_essentials_rejected_when_check_enabled(self):
        # Only one essential present (#domain) — should fail
        lines = ["#domain: 0.1 0.1 0.1\n"]
        with pytest.raises(SyntaxError):
            check_cmd_names(lines, checkessential=True)

    def test_missing_essentials_accepted_when_check_disabled(self):
        # Same input but ``checkessential=False`` lets it through
        lines = ["#domain: 0.1 0.1 0.1\n"]
        single, multi, geometry = check_cmd_names(lines, checkessential=False)
        assert single["#domain"] == "0.1 0.1 0.1"

    def test_returns_three_distinct_containers(self):
        single, multi, geometry = check_cmd_names(self._essentials())
        # Sanity: types match the dispatcher contract
        assert isinstance(single, dict)
        assert isinstance(multi, dict)
        assert isinstance(geometry, list)


# ---------------------------------------------------------------------------
# get_user_objects — end-to-end glue across the three dispatchers
# ---------------------------------------------------------------------------


class TestGetUserObjects:
    def test_essentials_produce_singleuse_objects(self):
        lines = [
            "#domain: 0.1 0.1 0.1\n",
            "#dx_dy_dz: 0.001 0.001 0.001\n",
            "#time_window: 100\n",
        ]
        objs = get_user_objects(lines)
        # Singleuse objects, in source-defined order:
        types = [type(o) for o in objs]
        assert types == [Discretisation, Domain, TimeWindow]

    def test_mixed_command_buckets_all_appear(self):
        lines = [
            "#domain: 0.1 0.1 0.1\n",
            "#dx_dy_dz: 0.001 0.001 0.001\n",
            "#time_window: 100\n",
            "#waveform: gaussian 1.0 1e9 wf1\n",
            "#box: 0 0 0 0.05 0.05 0.05 m1\n",
        ]
        objs = get_user_objects(lines)
        # Singleuse, then multiuse, then geometry — this is the order
        # ``get_user_objects`` concatenates the three dispatcher outputs.
        names = [type(o).__name__ for o in objs]
        assert "Discretisation" in names
        assert "Waveform" in names
        assert "Box" in names
        # Multiuse objects come after the singleuse triple
        assert names.index("Waveform") > names.index("TimeWindow")
        # Geometry objects come after the multiuse block
        assert names.index("Box") > names.index("Waveform")

    def test_skip_essential_check_allows_minimal_input(self):
        objs = get_user_objects(["#waveform: gaussian 1.0 1e9 wf1\n"], checkessential=False)
        assert len(objs) == 1


pytestmark = pytest.mark.unit
