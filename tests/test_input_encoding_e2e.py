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

"""End-to-end input encoding tests through the public ``gprMax.run`` entry point."""

import logging

import pytest

import gprMax

MODEL = (
    "#title: encoding test\n"
    "#domain_mode: TM\n"
    "#domain: 0.10 0.10 inf\n"
    "#dx_dy_dz: 0.002 0.002 0.002\n"
    "#time_window: 1e-9\n"
    "#waveform: ricker 1 1e9 w\n"
    "#hertzian_dipole: z 0.04 0.05 inf w\n"
    "#rx: 0.06 0.05 inf\n"
)


def _write(path, text, *, bom=False, encoding="utf-8"):
    payload = text.encode(encoding)
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + payload)


def _run(inputfile, outputfile):
    gprMax.run(
        inputfile=str(inputfile),
        outputfile=str(outputfile),
        hide_progress_bars=True,
        log_level=logging.WARNING,
    )


def test_bom_model_runs_end_to_end(tmp_path):
    model = tmp_path / "bom.in"
    _write(model, MODEL.replace("#title: encoding test", "#title: 中文模型"), bom=True)

    _run(model, tmp_path / "bom")

    assert (tmp_path / "bom.h5").is_file()


def test_invalid_utf8_reports_the_filename(tmp_path):
    model = tmp_path / "gbk.in"
    _write(model, MODEL.replace("#title: encoding test", "#title: 中文模型"), encoding="gbk")

    with pytest.raises(ValueError, match="not valid UTF-8") as excinfo:
        _run(model, tmp_path / "gbk")

    assert str(model) in str(excinfo.value)


def test_bom_include_file_is_read_as_utf8(tmp_path):
    materials = tmp_path / "materials.in"
    _write(materials, "#material: 4 0 1 0 block\n", bom=True)
    model = tmp_path / "include.in"
    _write(
        model,
        MODEL + "#include_file: materials.in\n#box: 0.02 0.02 inf 0.04 0.04 inf block\n",
        bom=True,
    )

    _run(model, tmp_path / "include")

    assert (tmp_path / "include.h5").is_file()


def test_bom_before_study_is_seen_by_preflight(tmp_path):
    cases = tmp_path / "cases.csv"
    _write(
        cases,
        "case_id,object_id,x_m,y_m,z_m\n"
        "case_1,hertzian_dipole_1,0.04,0.05,0.0\n"
        "case_1,rx_1,0.06,0.05,0.0\n",
        bom=True,
    )
    model = tmp_path / "study.in"
    _write(model, MODEL + "#study: gpr cases.csv\n", bom=True)

    _run(model, tmp_path / "bom_study")

    assert list(tmp_path.glob("bom_study*.h5"))
