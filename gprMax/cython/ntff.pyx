# cython: cdivision=True
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

"""OpenMP kernels for CPU near-to-far-field surface collection."""

import numpy as np

cimport numpy as np
from cython.parallel import prange
from libc.math cimport cos, exp, hypot, sin

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


cpdef void evaluate_equivalent_current_far_zone(
    int nthreads,
    const float_or_double[:, ::1] patch_positions,
    const float_or_double[::1] area_weights,
    const float_or_double[::1] wavenumbers,
    const float_or_double[:, ::1] directions,
    const float_or_double_complex[:, :, ::1] electric_current,
    const float_or_double_complex[:, :, ::1] magnetic_current,
    float_or_double impedance,
    float_or_double_complex[:, :, ::1] output,
):
    """Evaluate range-normalised E from collocated Love currents."""

    cdef Py_ssize_t task, direction, frequency, patch
    cdef Py_ssize_t ndirections = directions.shape[0]
    cdef Py_ssize_t nfrequencies = wavenumbers.shape[0]
    cdef Py_ssize_t npatches = patch_positions.shape[0]
    cdef Py_ssize_t ntasks = ndirections * nfrequencies
    cdef float_or_double k, projection, angle, rx, ry, rz
    cdef float_or_double_complex phase, prefactor
    cdef float_or_double_complex nx, ny, nz, lx, ly, lz
    cdef float_or_double_complex ndot, cx, cy, cz
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
        rx = directions[direction, 0]
        ry = directions[direction, 1]
        rz = directions[direction, 2]
        nx = 0
        ny = 0
        nz = 0
        lx = 0
        ly = 0
        lz = 0
        for patch in range(npatches):
            projection = (
                rx * patch_positions[patch, 0]
                + ry * patch_positions[patch, 1]
                + rz * patch_positions[patch, 2]
            )
            angle = k * projection
            phase = (cos(angle) + 1j * sin(angle)) * area_weights[patch]
            nx = nx + phase * electric_current[frequency, patch, 0]
            ny = ny + phase * electric_current[frequency, patch, 1]
            nz = nz + phase * electric_current[frequency, patch, 2]
            lx = lx + phase * magnetic_current[frequency, patch, 0]
            ly = ly + phase * magnetic_current[frequency, patch, 1]
            lz = lz + phase * magnetic_current[frequency, patch, 2]

        ndot = rx * nx + ry * ny + rz * nz
        cx = ry * lz - rz * ly
        cy = rz * lx - rx * lz
        cz = rx * ly - ry * lx
        prefactor = -1j * k / four_pi
        output[frequency, direction, 0] = prefactor * (
            impedance * (nx - rx * ndot) - cx
        )
        output[frequency, direction, 1] = prefactor * (
            impedance * (ny - ry * ndot) - cy
        )
        output[frequency, direction, 2] = prefactor * (
            impedance * (nz - rz * ndot) - cz
        )


cdef inline float_or_double_complex _exp_plus_i_beta(
    float_or_double_complex beta,
    float_or_double distance,
) noexcept nogil:
    cdef float_or_double angle = beta.real * distance
    cdef float_or_double amplitude = exp(-beta.imag * distance)
    return amplitude * (cos(angle) + 1j * sin(angle))


cdef inline float_or_double_complex _exp_minus_i_beta(
    float_or_double_complex beta,
    float_or_double distance,
) noexcept nogil:
    cdef float_or_double angle = beta.real * distance
    cdef float_or_double amplitude = exp(beta.imag * distance)
    return amplitude * (cos(angle) - 1j * sin(angle))


cpdef void evaluate_layered_equivalent_current_far_zone(
    int nthreads,
    const float_or_double[:, ::1] patch_positions,
    const np.int64_t[::1] patch_layers,
    const float_or_double[::1] layer_top,
    const float_or_double[::1] layer_bottom,
    const float_or_double[::1] area_weights,
    const float_or_double[:, ::1] directions,
    const float_or_double[:, ::1] observation_wavenumber,
    const float_or_double[:, ::1] observation_impedance,
    const float_or_double[:, ::1] electric_factor,
    const float_or_double[:, ::1] magnetic_factor,
    const float_or_double_complex[:, :, ::1] beta,
    const float_or_double_complex[:, :, ::1] epsilon_ratio,
    const float_or_double_complex[:, :, ::1] permeability_ratio,
    const float_or_double_complex[:, :, ::1] eta_e,
    const float_or_double_complex[:, :, ::1] eta_h,
    const float_or_double_complex[:, :, ::1] plus_e,
    const float_or_double_complex[:, :, ::1] minus_e,
    const float_or_double_complex[:, :, ::1] plus_h,
    const float_or_double_complex[:, :, ::1] minus_h,
    const float_or_double_complex[:, :, ::1] electric_current,
    const float_or_double_complex[:, :, ::1] magnetic_current,
    float_or_double_complex[:, :, ::1] electric_output,
    float_or_double_complex[:, :, ::1] magnetic_output,
):
    """Evaluate planar-layered Love-current far fields with OpenMP."""

    cdef Py_ssize_t task, direction, frequency, patch, layer
    cdef Py_ssize_t ndirections = directions.shape[0]
    cdef Py_ssize_t nfrequencies = observation_wavenumber.shape[0]
    cdef Py_ssize_t npatches = patch_positions.shape[0]
    cdef Py_ssize_t ntasks = ndirections * nfrequencies
    cdef float_or_double du, dv, dn, st, cp, sp, sign, angle, area
    cdef float_or_double_complex lateral, phase_plus, phase_minus
    cdef float_or_double_complex vie, vve, vih, vvh
    cdef float_or_double_complex jr, jp, mr, mp
    cdef float_or_double_complex ajt, ajp, fmt, fmp, et, ep
    cdef float_or_double_complex ex, ey, ez
    cdef float_or_double_complex hx, hy, hz

    for task in prange(
        ntasks,
        nogil=True,
        schedule="static",
        num_threads=nthreads,
    ):
        frequency = task // ndirections
        direction = task - frequency * ndirections
        du = directions[direction, 0]
        dv = directions[direction, 1]
        dn = directions[direction, 2]
        st = hypot(du, dv)
        if st > 1e-12:
            cp = du / st
            sp = dv / st
        else:
            cp = 1
            sp = 0
        sign = 1 if dn > 0 else -1
        ajt = 0
        ajp = 0
        fmt = 0
        fmp = 0

        for patch in range(npatches):
            layer = patch_layers[patch]
            phase_plus = _exp_minus_i_beta(
                beta[frequency, direction, layer],
                patch_positions[patch, 2] - layer_bottom[layer],
            )
            phase_minus = _exp_plus_i_beta(
                beta[frequency, direction, layer],
                patch_positions[patch, 2] - layer_top[layer],
            )
            vie = (
                plus_e[frequency, direction, layer] * phase_plus
                + minus_e[frequency, direction, layer] * phase_minus
            )
            vve = (
                -plus_e[frequency, direction, layer] * phase_plus
                + minus_e[frequency, direction, layer] * phase_minus
            ) / eta_e[frequency, direction, layer]
            vih = (
                plus_h[frequency, direction, layer] * phase_plus
                + minus_h[frequency, direction, layer] * phase_minus
            )
            vvh = (
                -plus_h[frequency, direction, layer] * phase_plus
                + minus_h[frequency, direction, layer] * phase_minus
            ) / eta_h[frequency, direction, layer]

            angle = observation_wavenumber[frequency, direction] * (
                du * patch_positions[patch, 0]
                + dv * patch_positions[patch, 1]
            )
            area = area_weights[patch]
            lateral = area * (cos(angle) + 1j * sin(angle))
            jr = (
                cp * electric_current[frequency, patch, 0]
                + sp * electric_current[frequency, patch, 1]
            )
            jp = (
                -sp * electric_current[frequency, patch, 0]
                + cp * electric_current[frequency, patch, 1]
            )
            mr = (
                cp * magnetic_current[frequency, patch, 0]
                + sp * magnetic_current[frequency, patch, 1]
            )
            mp = (
                -sp * magnetic_current[frequency, patch, 0]
                + cp * magnetic_current[frequency, patch, 1]
            )
            ajt = ajt + lateral * sign * (
                vie * jr
                - vve
                * st
                * electric_current[frequency, patch, 2]
                / epsilon_ratio[frequency, direction, layer]
            )
            ajp = ajp + lateral * sign * dn * vih * jp
            fmt = fmt + lateral * sign * dn * (
                vvh * mr
                - vih
                * st
                * magnetic_current[frequency, patch, 2]
                / permeability_ratio[frequency, direction, layer]
            )
            fmp = fmp + lateral * sign * vve * mp

        et = -1j * (
            electric_factor[frequency, direction] * ajt
            + magnetic_factor[frequency, direction] * fmp
        )
        ep = -1j * (
            electric_factor[frequency, direction] * ajp
            - magnetic_factor[frequency, direction] * fmt
        )
        ex = et * dn * cp - ep * sp
        ey = et * dn * sp + ep * cp
        ez = -et * st
        electric_output[frequency, direction, 0] = ex
        electric_output[frequency, direction, 1] = ey
        electric_output[frequency, direction, 2] = ez
        hx = (dv * ez - dn * ey) / observation_impedance[frequency, direction]
        hy = (dn * ex - du * ez) / observation_impedance[frequency, direction]
        hz = (du * ey - dv * ex) / observation_impedance[frequency, direction]
        magnetic_output[frequency, direction, 0] = hx
        magnetic_output[frequency, direction, 1] = hy
        magnetic_output[frequency, direction, 2] = hz


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
            # Explicit assignment avoids invalid MSVC ``complex +=`` code
            # emitted by Cython 3.2 for fused complex memoryviews.
            inside_dft[frequency, patch] = (
                inside_dft[frequency, patch] + phase * inside_value
            )
            outside_dft[frequency, patch] = (
                outside_dft[frequency, patch] + phase * outside_value
            )


cpdef void accumulate_sparse_dft(
    int nthreads,
    const np.int64_t[::1] field_indices,
    const float_or_double[::1] field,
    const float_or_double_complex[::1] multiplier,
    float_or_double_complex[:, ::1] output,
):
    """Gather sparse Yee edges and accumulate arbitrary-frequency DFTs."""

    cdef Py_ssize_t frequency, point
    cdef Py_ssize_t nfrequencies = multiplier.shape[0]
    cdef Py_ssize_t npoints = field_indices.shape[0]
    cdef float_or_double value
    cdef float_or_double_complex phase

    for point in prange(
        npoints,
        nogil=True,
        schedule="static",
        num_threads=nthreads,
    ):
        value = field[field_indices[point]]
        for frequency in range(nfrequencies):
            phase = multiplier[frequency]
            output[frequency, point] = output[frequency, point] + phase * value


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


cpdef void gather_equivalent_current_component(
    int nthreads,
    const np.int64_t[:, ::1] stencil_indices,
    const float_or_double[::1] field,
    float_or_double[::1] output,
):
    """Arithmetic-average one Yee stencil for every common-surface patch."""

    cdef Py_ssize_t patch, sample
    cdef Py_ssize_t npatches = stencil_indices.shape[1]
    cdef Py_ssize_t nsamples = stencil_indices.shape[0]
    cdef float_or_double value

    for patch in prange(
        npatches,
        nogil=True,
        schedule="static",
        num_threads=nthreads,
    ):
        value = 0
        for sample in range(nsamples):
            value = value + field[stencil_indices[sample, patch]]
        output[patch] = value / nsamples


cpdef void deposit_equivalent_current_time(
    int nthreads,
    Py_ssize_t sample_index,
    const float_or_double[:, ::1] current,
    const float_or_double[:, ::1] theta_basis,
    const float_or_double[:, ::1] phi_basis,
    const np.int64_t[:, ::1] integer_delay,
    const float_or_double[:, ::1] fractional_delay,
    const float_or_double[::1] area_weights,
    Py_ssize_t time_origin_step,
    float_or_double[:, ::1] output_theta,
    float_or_double[:, ::1] output_phi,
):
    """Deposit one differentiated Love-current level with linear delay."""

    cdef Py_ssize_t direction, patch, destination
    cdef Py_ssize_t ndirections = output_theta.shape[0]
    cdef Py_ssize_t npatches = current.shape[0]
    cdef float_or_double theta_value, phi_value, fraction, area

    for direction in prange(
        ndirections,
        nogil=True,
        schedule="static",
        num_threads=nthreads,
    ):
        for patch in range(npatches):
            theta_value = (
                current[patch, 0] * theta_basis[direction, 0]
                + current[patch, 1] * theta_basis[direction, 1]
                + current[patch, 2] * theta_basis[direction, 2]
            )
            phi_value = (
                current[patch, 0] * phi_basis[direction, 0]
                + current[patch, 1] * phi_basis[direction, 1]
                + current[patch, 2] * phi_basis[direction, 2]
            )
            fraction = fractional_delay[direction, patch]
            area = area_weights[patch]
            destination = (
                sample_index
                + integer_delay[direction, patch]
                - time_origin_step
            )
            output_theta[direction, destination] += (
                (1 - fraction) * area * theta_value
            )
            output_theta[direction, destination + 1] += (
                fraction * area * theta_value
            )
            output_phi[direction, destination] += (
                (1 - fraction) * area * phi_value
            )
            output_phi[direction, destination + 1] += (
                fraction * area * phi_value
            )


cpdef void deposit_layered_impulse_time(
    int nthreads,
    Py_ssize_t sample_index,
    int yee_half_step,
    const float_or_double[:, ::1] values,
    const np.int64_t[::1] row_template,
    const np.int64_t[::1] row_integer_shift,
    const float_or_double[::1] row_fractional_shift,
    const np.int64_t[::1] response_offsets,
    const np.int64_t[::1] integer_delay,
    const float_or_double[::1] fractional_delay,
    const float_or_double[::1] impulse_amplitude,
    Py_ssize_t time_origin_step,
    float_or_double[:, ::1] output,
):
    """Deposit one scalar layered response for every direction and patch.

    ``response_offsets`` is a CSR row pointer over unique axial response
    templates. Each flattened ``direction * npatches + patch`` row selects a
    template and adds its lateral integer/fractional delay. Complete output
    directions are owned by OpenMP workers, so the ragged impulse accumulation
    needs no atomics.
    """

    cdef Py_ssize_t direction, patch, row, template, impulse, destination
    cdef Py_ssize_t ndirections = values.shape[0]
    cdef Py_ssize_t npatches = values.shape[1]
    cdef float_or_double fraction, contribution

    for direction in prange(
        ndirections,
        nogil=True,
        schedule="static",
        num_threads=nthreads,
    ):
        for patch in range(npatches):
            row = direction * npatches + patch
            template = row_template[row]
            contribution = values[direction, patch]
            if contribution == 0:
                continue
            for impulse in range(
                response_offsets[template], response_offsets[template + 1]
            ):
                fraction = (
                    fractional_delay[impulse]
                    + row_fractional_shift[row]
                    + 0.5 * yee_half_step
                )
                destination = (
                    sample_index
                    + integer_delay[impulse]
                    + row_integer_shift[row]
                    - time_origin_step
                )
                if fraction >= 2:
                    fraction = fraction - 2
                    destination = destination + 2
                elif fraction >= 1:
                    fraction = fraction - 1
                    destination = destination + 1
                output[direction, destination] += (
                    (1 - fraction) * contribution * impulse_amplitude[impulse]
                )
                output[direction, destination + 1] += (
                    fraction * contribution * impulse_amplitude[impulse]
                )


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
