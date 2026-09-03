# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""OpenCL solver timing regression tests."""

from types import SimpleNamespace

import pytest

from gprMax.updates import opencl_updates
from gprMax.updates.opencl_updates import OpenCLUpdates


class _Marker:
    def __init__(self, *, start=0, end=0):
        self.profile = SimpleNamespace(start=start, end=end)
        self.waited = False

    def wait(self):
        self.waited = True


class _OpenCL:
    def __init__(self, markers):
        self.markers = iter(markers)

    def enqueue_marker(self, queue):
        return next(self.markers)


def _updates(markers):
    updates = object.__new__(OpenCLUpdates)
    updates.cl = _OpenCL(markers)
    updates.queue = object()
    return updates


@pytest.mark.unit
def test_opencl_solve_time_uses_device_profile(monkeypatch):
    first = _Marker(start=100)
    second = _Marker(end=1_500_000_100)
    updates = _updates([first, second])
    monkeypatch.setattr(opencl_updates, "perf_counter", lambda: 10.0)

    updates.time_start()

    assert updates.calculate_solve_time() == pytest.approx(1.5)
    assert first.waited
    assert second.waited


@pytest.mark.unit
def test_opencl_solve_time_falls_back_when_device_profile_is_zero(monkeypatch):
    first = _Marker()
    second = _Marker()
    updates = _updates([first, second])
    wall_times = iter((10.0, 12.5))
    monkeypatch.setattr(opencl_updates, "perf_counter", lambda: next(wall_times))

    updates.time_start()

    assert updates.calculate_solve_time() == pytest.approx(2.5)
    assert first.waited
    assert second.waited
