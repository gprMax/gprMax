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

"""Regression test for Scene.process_single_use_objects() (Codex-reported):
duplicate single-use commands (e.g. two `#domain` lines) were never
detected. `set(self.single_use_objects)` relied on UserObject having a
value-based __eq__/__hash__, which it doesn't (default identity-based),
so two distinct Domain(...) instances were never considered "equal" and
the duplicate check never fired - a later duplicate command would
silently build and overwrite the earlier one's state instead of raising
an error.

Fixed by detecting duplicates by command TYPE instead of by set()
membership.
"""
import tempfile
from pathlib import Path

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


def test_duplicate_domain_command_rejected(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.TimeWindow(time=1e-12))

    with pytest.raises(ValueError):
        _run(scene, tmp_path, "dup_domain")


def test_duplicate_discretisation_command_rejected(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Discretisation(p1=(2e-3, 2e-3, 2e-3)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
    scene.add(gprMax.TimeWindow(time=1e-12))

    with pytest.raises(ValueError):
        _run(scene, tmp_path, "dup_dl")


def test_single_use_commands_without_duplicates_still_work(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
    scene.add(gprMax.TimeWindow(time=1e-12))

    _run(scene, tmp_path, "no_dup")
