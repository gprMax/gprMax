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

"""OpenMP kernels for CPU near-to-far-field surface collection."""

import numpy as np

cimport numpy as np
from cython.parallel import prange

from gprMax.config cimport float_or_double, float_or_double_complex


cpdef void accumulate_surface_dft(
    int nthreads,
    const np.int64_t[::1] inside_indices,
    const np.int64_t[::1] outside_indices,
    const float_or_double[::1] field,
    const float_or_double_complex[::1] multiplier,
    float_or_double_complex[:, ::1] inside_dft,
    float_or_double_complex[:, ::1] outside_dft,
):
    """Gather a two-sided component surface and accumulate its raw DFT.

    OpenMP distributes surface patches so that even a single-frequency
    monitor can use every configured thread. Each inside/outside field pair
    is gathered once, then applied to every frequency. Successive patches
    advance along contiguous DFT rows, producing one interleaved stream per
    frequency without allocating temporary surface arrays. The multiplier
    already contains the timestep, window, and engineering-convention
    ``exp(-j omega t)`` phase factor.
    """

    cdef Py_ssize_t frequency, patch
    cdef Py_ssize_t nfrequencies = multiplier.shape[0]
    cdef Py_ssize_t npatches = inside_indices.shape[0]
    cdef float_or_double_complex phase
    cdef float_or_double inside_value, outside_value

    for patch in prange(
        npatches,
        nogil=True,
        schedule="static",
        num_threads=nthreads,
    ):
        inside_value = field[inside_indices[patch]]
        outside_value = field[outside_indices[patch]]
        for frequency in range(nfrequencies):
            phase = multiplier[frequency]
            inside_dft[frequency, patch] += phase * inside_value
            outside_dft[frequency, patch] += phase * outside_value
