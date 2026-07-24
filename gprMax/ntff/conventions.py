# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
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
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Fourier and phasor conventions used by gprMax NTFF calculations.

The convention is the standard electrical-engineering convention:

* physical time dependence: ``Re{X(omega) exp(+j omega t)}``;
* forward transform: ``X(omega) = integral x(t) exp(-j omega t) dt``;
* outgoing Green function: ``exp(-j k R) / (4 pi R)``.

These signs are part of the public data contract, not configurable options.
"""

import operator
from typing import Optional

import numpy as np
import numpy.typing as npt


PHASOR_TIME_DEPENDENCE = "exp(+j*omega*t)"
FORWARD_TRANSFORM_KERNEL = "exp(-j*omega*t)"
OUTGOING_GREEN_RADIAL_FACTOR = "exp(-j*k*R)"


def engineering_dft(
    samples: npt.ArrayLike,
    frequencies: npt.ArrayLike,
    dt: float,
    *,
    time_offset: float = 0.0,
    window: Optional[npt.ArrayLike] = None,
    axis: int = 0,
) -> npt.NDArray[np.complexfloating]:
    """Evaluate the engineering-convention DFT at arbitrary frequencies.

    This is a small NumPy reference implementation for tests and prototypes;
    the production collector will use a recursive phasor. The transformed time
    axis is replaced by a leading frequency axis.

    Args:
        samples: Real or complex time samples.
        frequencies: Non-negative frequencies in Hz.
        dt: Sample interval in seconds.
        time_offset: Physical time of sample zero in seconds.
        window: Optional one-dimensional window with one value per sample.
        axis: Axis of ``samples`` containing time.

    Returns:
        Complex transform values with shape ``(nf, *samples_without_time)``.
    """

    values = np.asarray(samples)
    if values.ndim == 0:
        raise ValueError("samples must have at least one dimension")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and greater than zero")
    if not np.isfinite(time_offset):
        raise ValueError("time_offset must be finite")

    if values.dtype.kind == "c":
        real_dtype = values.real.dtype
        complex_dtype = values.dtype
    elif values.dtype.kind == "f" and values.dtype.itemsize in (4, 8):
        real_dtype = values.dtype
        complex_dtype = np.dtype(f"c{2 * real_dtype.itemsize}")
    else:
        real_dtype = np.dtype(float)
        complex_dtype = np.dtype(complex)
    freqs = np.asarray(frequencies, dtype=real_dtype)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(freqs)) or np.any(freqs < 0):
        raise ValueError("frequencies must contain finite, non-negative values")

    try:
        axis = operator.index(axis)
    except TypeError as exc:
        raise ValueError("axis is not valid for samples") from exc
    if axis < -values.ndim or axis >= values.ndim:
        raise ValueError("axis is not valid for samples")
    values = np.moveaxis(values, axis, 0)

    nsamples = values.shape[0]
    if window is None:
        weights = np.ones(nsamples, dtype=real_dtype)
    else:
        weights = np.asarray(window, dtype=real_dtype)
        if weights.ndim != 1 or weights.size != nsamples:
            raise ValueError("window must have one value per time sample")
        if not np.all(np.isfinite(weights)):
            raise ValueError("window must contain only finite values")

    times = time_offset + dt * np.arange(nsamples, dtype=real_dtype)
    phase = np.exp(
        -2j * np.pi * freqs[:, np.newaxis] * times[np.newaxis, :]
    ).astype(complex_dtype)
    values = values.astype(
        complex_dtype if values.dtype.kind == "c" else real_dtype, copy=False
    )
    weighted_values = values * weights.reshape((nsamples,) + (1,) * (values.ndim - 1))
    return np.asarray(
        dt * np.tensordot(phase, weighted_values, axes=((1,), (0,))),
        dtype=complex_dtype,
    )
