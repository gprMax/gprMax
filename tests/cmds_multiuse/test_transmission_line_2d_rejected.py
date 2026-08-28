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

"""Regression test: #transmission_line / TransmissionLine must be
rejected in 2D mode (TM or TE), since its internal 1D line model uses a
"magic time step" (TransmissionLineUser.dl = sqrt(3) * c * dt) derived
from the 3D Courant condition - a 2D model's dt comes from the 2-axis
CFL formula instead, breaking that relationship.
"""
import tempfile
from pathlib import Path

import pytest

import gprMax

INF = float("inf")


def _run(scene, tmp_path, label):
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )


def _base_scene(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    return scene


@pytest.mark.parametrize("mode,polarisation", [("TM", "z"), ("TE", "y")])
def test_transmission_line_rejected_in_2d_mode(tmp_path, mode, polarisation):
    scene = _base_scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(
        gprMax.TransmissionLine(
            polarisation=polarisation, p1=(0.01, 0.01, 0.0), resistance=50, waveform_id="w"
        )
    )

    with pytest.raises(ValueError, match="2D mode"):
        _run(scene, tmp_path, f"tl_{mode}")


def test_transmission_line_still_works_in_3d(tmp_path):
    # TransmissionLine's own internal line needs enough iterations to
    # build its line-length array (nl = 0.667 * iterations) past its
    # antenna-connection position (antpos=10) - unrelated to the 2D
    # guard, just needs a real time window, not this file's usual 1ps.
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(
        gprMax.TransmissionLine(
            polarisation="z", p1=(0.01, 0.01, 0.01), resistance=50, waveform_id="w"
        )
    )

    _run(scene, tmp_path, "tl_3d")
