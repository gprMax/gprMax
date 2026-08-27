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

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gprMax.ntff.conventions import (
    FORWARD_TRANSFORM_KERNEL,
    OUTGOING_GREEN_RADIAL_FACTOR,
    PHASOR_TIME_DEPENDENCE,
    engineering_dft,
)


def test_engineering_convention_signs_are_explicit_and_fixed():
    assert PHASOR_TIME_DEPENDENCE == "exp(+j*omega*t)"
    assert FORWARD_TRANSFORM_KERNEL == "exp(-j*omega*t)"
    assert OUTGOING_GREEN_RADIAL_FACTOR == "exp(-j*k*R)"


def test_engineering_dft_matches_numpy_fft_positive_frequency_bin():
    nsamples = 128
    dt = 2.5e-12
    bin_index = 9
    frequency = bin_index / (nsamples * dt)
    amplitude = 2.4
    phase = 0.37
    times = dt * np.arange(nsamples)
    samples = amplitude * np.cos(2 * np.pi * frequency * times + phase)

    actual = engineering_dft(samples, [frequency], dt)[0]
    expected = dt * np.fft.fft(samples)[bin_index]

    assert_allclose(actual, expected, rtol=2e-14, atol=2e-24)
    assert_allclose(actual, 0.5 * nsamples * dt * amplitude * np.exp(1j * phase))


def test_engineering_dft_uses_physical_time_offset_and_arbitrary_axis():
    nsamples = 96
    dt = 1e-11
    frequency = 7 / (nsamples * dt)
    time_offset = 0.35 * dt
    phase = -0.61
    times = time_offset + dt * np.arange(nsamples)
    signal = np.cos(2 * np.pi * frequency * times + phase)
    samples = np.stack((signal, 3 * signal), axis=0)

    actual = engineering_dft(
        samples, [frequency], dt, time_offset=time_offset, axis=1
    )
    expected = 0.5 * nsamples * dt * np.exp(1j * phase) * np.array([1, 3])

    assert actual.shape == (1, 2)
    assert_allclose(actual[0], expected, rtol=2e-14, atol=2e-24)


def test_engineering_dft_time_derivative_has_positive_jomega_factor():
    nsamples = 64
    dt = 1e-10
    frequency = 5 / (nsamples * dt)
    omega = 2 * np.pi * frequency
    times = dt * np.arange(nsamples)
    samples = np.exp(1j * omega * times)
    time_derivative = 1j * omega * samples

    field_dft = engineering_dft(samples, [frequency], dt)[0]
    derivative_dft = engineering_dft(time_derivative, [frequency], dt)[0]

    assert_allclose(derivative_dft, 1j * omega * field_dft, rtol=2e-14)


def test_engineering_dft_preserves_configured_single_precision():
    samples = np.linspace(-1, 1, 32, dtype=np.float32)

    result = engineering_dft(samples, np.asarray([2e8], dtype=np.float32), 1e-10)

    assert result.dtype == np.complex64


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"frequencies": [], "dt": 1.0}, "frequencies"),
        ({"frequencies": [-1.0], "dt": 1.0}, "non-negative"),
        ({"frequencies": [1.0], "dt": 0.0}, "dt"),
        ({"frequencies": [1.0], "dt": 1.0, "window": [1.0]}, "window"),
        ({"frequencies": [1.0], "dt": 1.0, "axis": 2}, "axis"),
    ],
)
def test_engineering_dft_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        engineering_dft(np.ones(4), **kwargs)
