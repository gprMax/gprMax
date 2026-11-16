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
* ``process_include_files`` — recursively expands includes relative to their
  containing files, independently of the working directory.
* ``check_cmd_names`` — routes each command line into the singleuse /
  multiuse / geometry buckets and validates command names, the
  command-name/parameters split, single-instance rule, and essentials
  presence.
* ``get_user_objects`` — end-to-end glue across the three dispatchers.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from gprMax.hash_cmds_file import (
    _read_input_file,
    check_cmd_names,
    get_user_objects,
    process_include_files,
    process_python_include_code,
)
from gprMax.user_objects.cmds_singleuse import Discretisation, Domain, TimeWindow

# ---------------------------------------------------------------------------
# _read_input_file
# ---------------------------------------------------------------------------


class TestReadInputFile:
    def test_a_byte_order_mark_is_not_part_of_the_first_command(self, tmp_path):
        path = tmp_path / "bom.in"
        path.write_bytes(b"\xef\xbb\xbf#title: demo\n#domain: 0.1 0.1 0.1\n")

        assert _read_input_file(path).startswith("#title: demo")

    def test_bom_file_parses_through_the_front_door(self, tmp_path):
        path = tmp_path / "bom.in"
        path.write_bytes(b"\xef\xbb\xbf#title: demo\n#domain: 0.1 0.1 0.1\n")

        with io.StringIO(_read_input_file(path)) as stream:
            commands = process_python_include_code(stream, {})

        assert commands == ["#title: demo\n", "#domain: 0.1 0.1 0.1\n"]

    def test_non_ascii_utf8_is_preserved(self, tmp_path):
        path = tmp_path / "utf8.in"
        path.write_text("#title: 中文模型\n#domain: 0.1 0.1 0.1\n", encoding="utf-8")

        assert "中文模型" in _read_input_file(path)

    def test_non_utf8_file_is_rejected_with_a_clear_message(self, tmp_path):
        path = tmp_path / "gbk.in"
        path.write_bytes("#title: 中文\n".encode("gbk"))

        with pytest.raises(ValueError, match="not valid UTF-8"):
            _read_input_file(path)


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

    def test_stdout_restored_to_callers_stream_after_python_block(self):
        text = io.StringIO("#python:\n" "print('#title: demo')\n" "#end_python:\n")
        caller = io.StringIO()
        with redirect_stdout(caller):
            process_python_include_code(text, {})
            assert sys.stdout is caller
            print("after")
        assert caller.getvalue() == "after\n"

    @pytest.mark.parametrize("exception", [RuntimeError, SystemExit, KeyboardInterrupt])
    def test_exception_restores_callers_stdout(self, exception):
        text = io.StringIO("#python:\nraise failure('test')\n#end_python:\n")
        caller = io.StringIO()
        with redirect_stdout(caller):
            with pytest.raises(exception):
                process_python_include_code(text, {"failure": exception})
            assert sys.stdout is caller

    def test_terminal_empty_python_block_reports_missing_end(self):
        with pytest.raises(SyntaxError, match="missing #end_python"):
            process_python_include_code(io.StringIO("#python:\n"), {})


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

    def test_relative_path_uses_input_file_parent(self, tmp_path, monkeypatch):
        # Place the included file next to a faked input file
        included = tmp_path / "sidecar.in"
        included.write_text("#title: sidecar wins\n")
        fake_input = tmp_path / "main.in"
        # The input file determines the include-search root.
        from gprMax import config

        fake_sim_config = SimpleNamespace(input_file_path=fake_input)
        monkeypatch.setattr(config, "sim_config", fake_sim_config)

        # Pass only the bare filename.
        cmds = ["#include_file: sidecar.in\n"]
        out = process_include_files(cmds)
        assert out == ["#title: sidecar wins\n"]

    def test_nested_paths_order_repetition_and_cwd_collision(self, tmp_path, monkeypatch):
        from gprMax import config

        model = tmp_path / "model"
        model.mkdir()
        nested = model / "nested"
        nested.mkdir()
        caller = tmp_path / "caller"
        caller.mkdir()
        (caller / "outer.in").write_text("#title: wrong CWD file\n")
        (model / "outer.in").write_text(
            "#box: before\n#include_file: nested/inner.in\n#box: after\n"
        )
        (nested / "inner.in").write_text("#box: inner\n")
        monkeypatch.chdir(caller)
        monkeypatch.setattr(
            config, "sim_config", SimpleNamespace(input_file_path=model / "main.in")
        )
        expected = ["#box: before\n", "#box: inner\n", "#box: after\n"]
        assert process_include_files(["#include_file: outer.in\n"] * 2) == expected * 2

    def test_missing_input_relative_file_never_uses_cwd_fallback(self, tmp_path, monkeypatch):
        from gprMax import config

        (tmp_path / "extra.in").write_text("#title: unrelated\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            config, "sim_config", SimpleNamespace(input_file_path=tmp_path / "model/main.in")
        )
        with pytest.raises(FileNotFoundError) as error:
            process_include_files(["#include_file: extra.in\n"])
        assert Path(error.value.filename) == tmp_path / "model" / "extra.in"

    @pytest.mark.parametrize("symlink", [False, True])
    def test_include_cycles_are_rejected_with_chain(self, tmp_path, monkeypatch, symlink):
        from gprMax import config

        root = tmp_path / "main.in"
        child = tmp_path / "child.in"
        root.write_text("#include_file: child.in\n")
        if symlink:
            alias = tmp_path / "alias.in"
            try:
                alias.symlink_to(root)
            except OSError as error:
                if getattr(error, "winerror", None) != 1314:
                    raise
                pytest.skip("Windows requires Developer Mode or symlink creation privilege")
        child.write_text(f"#include_file: {'alias.in' if symlink else 'main.in'}\n")
        monkeypatch.setattr(config, "sim_config", SimpleNamespace(input_file_path=root))
        with pytest.raises(ValueError, match="cycle detected:.*main.in.*child.in.*main.in"):
            process_include_files(["#include_file: child.in\n"])

    def test_nested_missing_file_raises(self, tmp_path):
        outer = tmp_path / "outer.in"
        outer.write_text("#include_file: missing.in\n")
        with pytest.raises(FileNotFoundError, match="missing.in"):
            process_include_files([f"#include_file: {outer}\n"])

    def test_python_blocks_in_include_are_explicitly_unsupported(self, tmp_path):
        outer = tmp_path / "outer.in"
        outer.write_text("#python:\nraise RuntimeError('must not execute')\n#end_python:\n")
        with pytest.raises(SyntaxError, match="Python blocks in included files are unsupported"):
            process_include_files([f"#include_file: {outer}\n"])

    def test_unexpanded_include_cannot_be_silently_dispatched(self):
        with pytest.raises(ValueError, match="must be expanded"):
            get_user_objects(["#include_file: unused.in\n"], checkessential=False)


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
