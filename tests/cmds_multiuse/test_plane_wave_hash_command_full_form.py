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

"""Regression test for a real bug found while auditing the 2D TE mode PR
(#699) against an orphaned pre-merge snapshot: the token-count checks for
the FULL form (background material_id + start/stop) of `#plane_wave_angles`
and `#plane_wave_vector` in `process_multicmds()`
(gprMax/hash_cmds_multiuse.py) had been swapped.

`#plane_wave_angles` full form is
p1(3) + p2(3) + theta,phi,psi(3) + waveform_id(1) + material_id(1) +
start,stop(2) = 13 tokens - the code checked `elif len(tmp) == 14:`, so a
real 13-token command fell through to the `else` branch and raised
"too many parameters".

`#plane_wave_vector` full form is
p1(3) + p2(3) + m_vec(3) + psi(1) + waveform_id(1) + material_id(1) +
start,stop(2) = 14 tokens - the code checked `elif len(tmp) == 13:`, which
is worse: a 13-token string would have entered that branch and then raised
IndexError accessing tmp[13] (out of range for a 13-element list), while
the correct 14-token form fell through to "too many parameters" instead.

Existing tests (tests/cmds_multiuse/test_plane_wave_2d.py) only exercise
the Python API objects (DiscretePlaneWaveAngles/Vector) directly, never
the `#plane_wave_angles`/`#plane_wave_vector` hash-command string parser,
which is exactly why this went undetected - these tests use a real `.in`
text file, parsed end-to-end via gprMax.run(inputfile=...), to close that
gap.
"""
from pathlib import Path

import gprMax
import gprMax.model as model_mod


def _capture_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def test_plane_wave_angles_full_form_hash_command_parses(monkeypatch, tmp_path: Path):
    infile = tmp_path / "plane_wave_angles_full.in"
    infile.write_text(
        "#title: plane_wave_angles full-form hash command\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.03 0.03 0.03\n"
        "#time_window: 1e-11\n"
        "#waveform: ricker 1 10e9 w\n"
        "#plane_wave_angles: 0.007 0.007 0.007 0.021 0.021 0.021 90 26.565051177 90 w free_space 0 1e-11\n"
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        inputfile=str(infile), n=1, geometry_only=True,
        outputfile=tmp_path / "run", hide_progress_bars=True,
    )
    grid = captured["grid"]
    assert len(grid.discreteplanewaves) == 1
    dpw = grid.discreteplanewaves[0]
    assert dpw.start == 0.0
    assert dpw.stop == 1e-11


def test_plane_wave_vector_full_form_hash_command_parses(monkeypatch, tmp_path: Path):
    infile = tmp_path / "plane_wave_vector_full.in"
    infile.write_text(
        "#title: plane_wave_vector full-form hash command\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.03 0.03 0.03\n"
        "#time_window: 1e-11\n"
        "#waveform: ricker 1 10e9 w\n"
        "#plane_wave_vector: 0.007 0.007 0.007 0.021 0.021 0.021 1 1 1 90 w free_space 0 1e-11\n"
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        inputfile=str(infile), n=1, geometry_only=True,
        outputfile=tmp_path / "run", hide_progress_bars=True,
    )
    grid = captured["grid"]
    assert len(grid.discreteplanewaves) == 1
    dpw = grid.discreteplanewaves[0]
    assert dpw.start == 0.0
    assert dpw.stop == 1e-11
