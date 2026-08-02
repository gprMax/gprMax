# cython: cdivision=True
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
from libc.math cimport cos, sin

from gprMax.config cimport float_or_double, float_or_double_complex


cpdef void evaluate_far_zone_patches(
    int nthreads,
    const float_or_double[:, ::1] patch_positions,
    const float_or_double[:, ::1] patch_normals,
    const float_or_double[::1] area_weights,
    const float_or_double[::1] wavenumbers,
    const float_or_double[:, ::1] directions,
    const float_or_double_complex[:, ::1] surface_field,
    const float_or_double_complex[:, ::1] normal_derivative,
    float_or_double_complex[:, ::1] output,
):
    """Evaluate the frequency-domain far-zone KSIR surface integral.

    OpenMP distributes complete frequency/direction pairs. Each worker
    therefore owns one output element and can accumulate without atomics. The
    combined loop also uses all threads when either the number of frequencies
    or the number of directions is small. Position and normal projections are
    intentionally calculated inside the patch loop so the kernel does not
    allocate a potentially very large ``ndirections x npatches`` temporary
    array.

    Geometry is relative to the requested phase origin and all arrays are
    validated and made contiguous by :mod:`gprMax.ntff.evaluator` before this
    low-level kernel is called.
    """

    cdef Py_ssize_t task, direction, frequency, patch
    cdef Py_ssize_t ndirections = directions.shape[0]
    cdef Py_ssize_t nfrequencies = wavenumbers.shape[0]
    cdef Py_ssize_t npatches = patch_positions.shape[0]
    cdef Py_ssize_t ntasks = ndirections * nfrequencies
    cdef float_or_double k, position_projection, normal_projection, angle
    cdef float_or_double_complex phase, integrand
    cdef float_or_double four_pi = 12.566370614359172953850573533118

    for task in prange(
        ntasks,
        nogil=True,
        schedule="static",
        num_threads=nthreads,
    ):
        frequency = task // ndirections
        direction = task - frequency * ndirections
        k = wavenumbers[frequency]
        output[frequency, direction] = 0
        for patch in range(npatches):
            position_projection = (
                directions[direction, 0] * patch_positions[patch, 0]
                + directions[direction, 1] * patch_positions[patch, 1]
                + directions[direction, 2] * patch_positions[patch, 2]
            )
            normal_projection = (
                directions[direction, 0] * patch_normals[patch, 0]
                + directions[direction, 1] * patch_normals[patch, 1]
                + directions[direction, 2] * patch_normals[patch, 2]
            )
            angle = k * position_projection
            phase = cos(angle) + 1j * sin(angle)
            integrand = -normal_derivative[frequency, patch] + (
                1j
                * k
                * normal_projection
                * surface_field[frequency, patch]
            )
            output[frequency, direction] = (
                output[frequency, direction]
                + integrand * phase * area_weights[patch]
            )
        output[frequency, direction] = output[frequency, direction] / four_pi


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


cpdef void gather_time_domain_surface(
    int nthreads,
    const np.int64_t[::1] inside_indices,
    const np.int64_t[::1] outside_indices,
    const float_or_double[::1] normal_spacing,
    const float_or_double[::1] field,
    float_or_double[::1] surface_value,
    float_or_double[::1] normal_derivative,
):
    """Gather and collocate a two-sided surface for advanced-time KSIR."""

    cdef Py_ssize_t patch
    cdef Py_ssize_t npatches = inside_indices.shape[0]
    cdef float_or_double inside_value, outside_value

    for patch in prange(
        npatches,
        nogil=True,
        schedule="static",
        num_threads=nthreads,
    ):
        inside_value = field[inside_indices[patch]]
        outside_value = field[outside_indices[patch]]
        surface_value[patch] = 0.5 * (inside_value + outside_value)
        normal_derivative[patch] = (
            outside_value - inside_value
        ) / normal_spacing[patch]


cpdef void deposit_time_domain_surface(
    int nthreads,
    Py_ssize_t sample_index,
    const float_or_double[::1] surface_value,
    const float_or_double[::1] normal_derivative,
    const float_or_double[::1] time_derivative,
    const np.int64_t[::1] source_patch_index,
    const float_or_double[:, ::1] normal_derivative_weight,
    const float_or_double[:, ::1] field_weight,
    const float_or_double[:, ::1] time_derivative_weight,
    const np.int64_t[:, ::1] integer_delay,
    const float_or_double[:, ::1] fractional_delay,
    const np.int64_t[::1] time_origin_steps,
    float_or_double[:, ::1] output,
):
    """Deposit one collocated surface time level into per-point traces.

    Each OpenMP worker owns complete output rows, so accumulation requires no
    atomics even when many patches share the same destination time bin.
    """

    cdef Py_ssize_t point, patch, source_patch, destination
    cdef Py_ssize_t npoints = output.shape[0]
    cdef Py_ssize_t npatches = source_patch_index.shape[0]
    cdef float_or_double contribution, fraction

    for point in prange(
        npoints,
        nogil=True,
        schedule="static",
        num_threads=nthreads,
    ):
        for patch in range(npatches):
            source_patch = source_patch_index[patch]
            contribution = (
                normal_derivative_weight[point, patch]
                * normal_derivative[source_patch]
                + field_weight[point, patch] * surface_value[source_patch]
                + time_derivative_weight[point, patch]
                * time_derivative[source_patch]
            )
            fraction = fractional_delay[point, patch]
            destination = (
                sample_index
                + integer_delay[point, patch]
                - time_origin_steps[point]
            )
            output[point, destination] += (1 - fraction) * contribution
            output[point, destination + 1] += fraction * contribution
