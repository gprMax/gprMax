# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Command-line interface regression tests."""

import sys

import pytest

from gprMax.gprMax import _existing_input_file, cli


@pytest.mark.unit
def test_existing_input_file_is_returned_as_a_path(tmp_path):
    inputfile = tmp_path / "model.in"
    inputfile.write_text("#title: model\n", encoding="utf-8")

    assert _existing_input_file(str(inputfile)) == inputfile


@pytest.mark.unit
def test_missing_input_file_is_reported_without_traceback(monkeypatch, capsys, tmp_path):
    missing = tmp_path / "missing.in"
    monkeypatch.setattr(sys, "argv", ["gprMax", str(missing)])

    with pytest.raises(SystemExit) as excinfo:
        cli()

    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    assert f"input file does not exist: {missing}" in error
    assert "Traceback" not in error
