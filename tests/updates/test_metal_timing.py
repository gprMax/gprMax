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

"""Regression coverage for Metal solver timing."""

import pytest

from gprMax.updates import metal_updates as metal_updates_module
from gprMax.updates.metal_updates import MetalUpdates


def test_time_start_records_the_clock(monkeypatch):
    monkeypatch.setattr(metal_updates_module, "timer", lambda: 100.0)
    updates = MetalUpdates.__new__(MetalUpdates)

    updates.time_start()

    assert updates.timestart == 100.0


def test_calculate_solve_time_returns_elapsed_wall_time(monkeypatch):
    readings = iter([100.0, 137.5])
    monkeypatch.setattr(metal_updates_module, "timer", lambda: next(readings))
    updates = MetalUpdates.__new__(MetalUpdates)

    updates.time_start()

    assert updates.calculate_solve_time() == pytest.approx(37.5)


def test_time_start_can_restart_the_timer(monkeypatch):
    readings = iter([10.0, 50.0, 55.0])
    monkeypatch.setattr(metal_updates_module, "timer", lambda: next(readings))
    updates = MetalUpdates.__new__(MetalUpdates)

    updates.time_start()
    updates.time_start()

    assert updates.calculate_solve_time() == pytest.approx(5.0)
