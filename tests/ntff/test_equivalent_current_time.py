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

"""Cython/NumPy parity checks for equivalent-current transient NTFF."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import c, epsilon_0, mu_0

import gprMax.ntff.equivalent_current_time as equivalent_current_time
from gprMax.ntff.equivalent_current_time import EquivalentCurrentTimeMonitor


def _run_monitor(rng):
    monitor = EquivalentCurrentTimeMonitor(
        "parity",
        lower=(2, 2, 2),
        upper=(5, 5, 5),
        spacing=(0.01, 0.01, 0.01),
        field_shape=(8, 8, 8),
        dt=1e-10,
        iterations=12,
        theta=(20, 70, 130),
        phi=(0, 45, 210),
        origin=(0.035, 0.035, 0.035),
        real_dtype=np.float64,
        wave_speed=c,
        impedance=np.sqrt(mu_0 / epsilon_0),
        nthreads=2,
    )
    fields = rng.standard_normal((12, 6, 8, 8, 8))
    for iteration in range(12):
        monitor.observe_electric(iteration, *fields[iteration, :3])
        monitor.observe_magnetic(iteration, *fields[iteration, 3:])
    monitor.finalise()
    return monitor.result


def test_cython_monitor_matches_numpy_reference(monkeypatch):
    cython_result = _run_monitor(np.random.default_rng(58214))
    monkeypatch.setattr(equivalent_current_time, "_gather_equivalent_current_component", None)
    monkeypatch.setattr(equivalent_current_time, "_deposit_equivalent_current_time", None)
    numpy_result = _run_monitor(np.random.default_rng(58214))

    assert_allclose(cython_result.times, numpy_result.times, rtol=0, atol=0)
    for component in ("Etheta", "Ephi"):
        assert_allclose(
            cython_result.fields[component],
            numpy_result.fields[component],
            rtol=2e-13,
            atol=2e-13,
        )
    assert_allclose(
        cython_result.terminal_field_ratios,
        numpy_result.terminal_field_ratios,
        rtol=2e-13,
        atol=2e-13,
    )
