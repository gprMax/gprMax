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

"""Regression test for TimeWindow iterations validation (Codex-reported):
the `time` branch validates > 0, but the `iterations` branch had no
validation at all - iterations=0 produced an apparently-successful run
with empty output, and negative values failed later with a confusing,
unrelated error rather than a clear upfront ValueError.
"""
import pytest

import gprMax


def _run(scene, tmp_path, label):
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )


def _base_scene(time_window):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
    scene.add(time_window)
    return scene


def test_zero_iterations_rejected(tmp_path):
    scene = _base_scene(gprMax.TimeWindow(iterations=0))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "zero_iterations")


def test_negative_iterations_rejected(tmp_path):
    scene = _base_scene(gprMax.TimeWindow(iterations=-5))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "neg_iterations")


def test_positive_iterations_still_works(tmp_path):
    scene = _base_scene(gprMax.TimeWindow(iterations=10))
    _run(scene, tmp_path, "pos_iterations")
