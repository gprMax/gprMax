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

"""Long-duration stability checks for recursive engineering DFT phases."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.ntff.frequency_domain import (
    DFT_PHASE_REANCHOR_INTERVAL,
    KSIRFrequencyDomainMonitor,
    _ComponentDFTAccumulator,
)
from gprMax.ntff.surfaces import build_component_surface


ITERATIONS = 65_537
DT = 2e-12
CHECK_ITERATIONS = {
    0,
    1,
    DFT_PHASE_REANCHOR_INTERVAL - 2,
    DFT_PHASE_REANCHOR_INTERVAL - 1,
    DFT_PHASE_REANCHOR_INTERVAL,
    DFT_PHASE_REANCHOR_INTERVAL + 1,
    32_767,
    ITERATIONS - 2,
    ITERATIONS - 1,
}


@pytest.mark.parametrize(
    "real_dtype,complex_dtype,tolerance,anchor_tolerance",
    [
        (np.dtype("f4"), np.dtype("c8"), 2e-5, 2e-7),
        (np.dtype("f8"), np.dtype("c16"), 1e-10, 1e-10),
    ],
)
@pytest.mark.parametrize("time_offset_steps", [0.0, 0.5])
def test_component_recursive_phase_remains_bounded_between_reanchors(
    real_dtype,
    complex_dtype,
    tolerance,
    anchor_tolerance,
    time_offset_steps,
):
    frequencies = np.asarray([0.137 / DT, 0.467 / DT], dtype=real_dtype)
    surface = build_component_surface(
        "Hx", (2, 2, 2), (4, 4, 4), (0.01,) * 3, (8, 8, 8)
    )
    accumulator = _ComponentDFTAccumulator(
        surface,
        frequencies,
        DT,
        ITERATIONS,
        time_offset_steps,
        np.ones(ITERATIONS, dtype=real_dtype),
        "rectangular",
        "closed",
        real_dtype,
        complex_dtype,
        1,
    )
    errors = {}

    for iteration in range(ITERATIONS):
        actual = accumulator.sampling_multiplier(iteration) / DT
        if iteration in CHECK_ITERATIONS:
            expected = np.exp(
                -2j
                * np.pi
                * frequencies.astype(np.float64)
                * (iteration + time_offset_steps)
                * DT
            )
            errors[iteration] = float(np.max(np.abs(actual - expected)))

    assert actual.dtype == complex_dtype
    assert max(errors.values()) < tolerance
    assert errors[DFT_PHASE_REANCHOR_INTERVAL] < anchor_tolerance


def _plane_wave(real_dtype):
    return SimpleNamespace(
        m=np.zeros(3, dtype=np.int32),
        origin=np.zeros(3, dtype=np.int32),
        axial=0,
        E_fields=np.ones((3, 1), dtype=real_dtype),
        corners=np.asarray((1, 1, 1, 5, 5, 5), dtype=np.int32),
        waveformID="pulse",
        materialID="free_space",
        actual_angles=np.asarray((90.0, 0.0), dtype=real_dtype),
        psi=0.0,
        start=0.0,
        stop=1.0,
    )


@pytest.mark.parametrize(
    "real_dtype,complex_dtype,tolerance",
    [
        (np.dtype("f4"), np.dtype("c8"), 2e-5),
        (np.dtype("f8"), np.dtype("c16"), 1e-10),
    ],
)
def test_incident_plane_wave_recursive_phase_uses_same_stable_reanchoring(
    real_dtype, complex_dtype, tolerance
):
    frequencies = np.asarray([0.137 / DT, 0.467 / DT], dtype=real_dtype)
    surface = build_component_surface(
        "Hx",
        (2, 2, 2),
        (4, 4, 4),
        (0.01,) * 3,
        (8, 8, 8),
        real_dtype=real_dtype,
    )
    monitor = KSIRFrequencyDomainMonitor(
        "incident_phase",
        {"Hx": surface},
        frequencies,
        [90.0],
        [0.0],
        DT,
        ITERATIONS,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    monitor.associate_plane_wave(_plane_wave(real_dtype), (0.01,) * 3, 0)
    unused = np.empty(0, dtype=real_dtype)
    errors = []

    for iteration in range(ITERATIONS):
        if iteration in CHECK_ITERATIONS:
            expected = np.exp(
                -2j
                * np.pi
                * frequencies.astype(np.float64)
                * iteration
                * DT
            )
            errors.append(float(np.max(np.abs(monitor._incident_phase - expected))))
        monitor.observe_electric(iteration, unused, unused, unused)

    assert monitor._incident_phase.dtype == complex_dtype
    assert max(errors) < tolerance
