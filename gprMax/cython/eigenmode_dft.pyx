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

#
# cython: cdivision=True
import cython
from cython.parallel import prange

from gprMax.config cimport float_or_double, float_or_double_complex


cdef inline int _imax(int a, int b) noexcept nogil:
    return a if a > b else b


cdef inline int _imin(int a, int b) noexcept nogil:
    return a if a < b else b


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void accumulate_eigenmode_dft(
    int nthreads,
    int normal_axis,
    int direction_sign,
    int magnetic_side,
    int u0,
    int v0,
    int u1,
    int v1,
    int plane_index,
    int[:] owned_lower,
    int[:] owned_upper,
    float_or_double dt,
    float_or_double measure,
    int handedness,
    float_or_double_complex[::1] electric_phase,
    float_or_double_complex[::1] magnetic_phase,
    const float_or_double_complex[::1] phase_step,
    const float_or_double_complex[:, :, :, ::1] conj_eu,
    const float_or_double_complex[:, :, :, ::1] conj_ev,
    const float_or_double_complex[:, :, :, ::1] conj_hu,
    const float_or_double_complex[:, :, :, ::1] conj_hv,
    float_or_double_complex[:, ::1] electric_dft,
    float_or_double_complex[:, ::1] magnetic_dft,
    const float_or_double[:, :, ::1] Ex,
    const float_or_double[:, :, ::1] Ey,
    const float_or_double[:, :, ::1] Ez,
    const float_or_double[:, :, ::1] Hx,
    const float_or_double[:, :, ::1] Hy,
    const float_or_double[:, :, ::1] Hz,
):
    """Project one Yee plane and advance every requested DFT bin once."""
    cdef Py_ssize_t frequency, mode, u, v
    cdef Py_ssize_t nf = electric_phase.shape[0]
    cdef Py_ssize_t nm = electric_dft.shape[1]
    cdef int hplane = plane_index if direction_sign * magnetic_side > 0 else plane_index - 1
    cdef float_or_double eu_value, ev_value, hu_value, hv_value
    cdef float_or_double factor = 0.5 * handedness * measure * dt
    cdef float_or_double_complex electric_sum, magnetic_sum
    cdef int sample_u0, sample_v0, sample_u1, sample_v1
    cdef bint plane_owned

    if normal_axis == 0:
        plane_owned = owned_lower[0] <= plane_index < owned_upper[0]
        sample_u0 = _imax(u0, owned_lower[1])
        sample_u1 = _imin(u1, owned_upper[1])
        sample_v0 = _imax(v0, owned_lower[2])
        sample_v1 = _imin(v1, owned_upper[2])
    elif normal_axis == 1:
        plane_owned = owned_lower[1] <= plane_index < owned_upper[1]
        sample_u0 = _imax(u0, owned_lower[0])
        sample_u1 = _imin(u1, owned_upper[0])
        sample_v0 = _imax(v0, owned_lower[2])
        sample_v1 = _imin(v1, owned_upper[2])
    else:
        plane_owned = owned_lower[2] <= plane_index < owned_upper[2]
        sample_u0 = _imax(u0, owned_lower[0])
        sample_u1 = _imin(u1, owned_upper[0])
        sample_v0 = _imax(v0, owned_lower[1])
        sample_v1 = _imin(v1, owned_upper[1])

    if not plane_owned:
        sample_u1 = sample_u0
        sample_v1 = sample_v0
    if sample_u1 < sample_u0:
        sample_u1 = sample_u0
    if sample_v1 < sample_v0:
        sample_v1 = sample_v0

    for frequency in prange(
        nf, nogil=True, schedule="static", num_threads=nthreads
    ):
        for mode in range(nm):
            electric_sum = 0
            magnetic_sum = 0
            for u in range(sample_u0, sample_u1):
                for v in range(sample_v0, sample_v1):
                    if normal_axis == 0:
                        eu_value = 0.5 * (Ey[plane_index, u, v] + Ey[plane_index, u, v + 1])
                        ev_value = 0.5 * (Ez[plane_index, u, v] + Ez[plane_index, u + 1, v])
                        hu_value = 0.5 * (Hy[hplane, u, v] + Hy[hplane, u + 1, v])
                        hv_value = 0.5 * (Hz[hplane, u, v] + Hz[hplane, u, v + 1])
                    elif normal_axis == 1:
                        eu_value = 0.5 * (Ex[u, plane_index, v] + Ex[u, plane_index, v + 1])
                        ev_value = 0.5 * (Ez[u, plane_index, v] + Ez[u + 1, plane_index, v])
                        hu_value = 0.5 * (Hx[u, hplane, v] + Hx[u + 1, hplane, v])
                        hv_value = 0.5 * (Hz[u, hplane, v] + Hz[u, hplane, v + 1])
                    else:
                        eu_value = 0.5 * (Ex[u, v, plane_index] + Ex[u, v + 1, plane_index])
                        ev_value = 0.5 * (Ey[u, v, plane_index] + Ey[u + 1, v, plane_index])
                        hu_value = 0.5 * (Hx[u, v, hplane] + Hx[u + 1, v, hplane])
                        hv_value = 0.5 * (Hy[u, v, hplane] + Hy[u, v + 1, hplane])

                    electric_sum = electric_sum + (
                        eu_value * conj_hv[frequency, mode, u - u0, v - v0]
                        - ev_value * conj_hu[frequency, mode, u - u0, v - v0]
                    )
                    magnetic_sum = magnetic_sum + direction_sign * (
                        conj_eu[frequency, mode, u - u0, v - v0] * hv_value
                        - conj_ev[frequency, mode, u - u0, v - v0] * hu_value
                    )

            electric_dft[frequency, mode] = (
                electric_dft[frequency, mode]
                + factor * electric_phase[frequency] * electric_sum
            )
            magnetic_dft[frequency, mode] = (
                magnetic_dft[frequency, mode]
                + factor * magnetic_phase[frequency] * magnetic_sum
            )

        electric_phase[frequency] = electric_phase[frequency] * phase_step[frequency]
        magnetic_phase[frequency] = magnetic_phase[frequency] * phase_step[frequency]
