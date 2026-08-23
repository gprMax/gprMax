# cython: cdivision=True
# Copyright (C) 2026: The University of Edinburgh, United Kingdom

"""Sparse locally implicit surface-impedance update."""

import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.config cimport float_or_double


cdef inline double _electric_get(
    int component,
    int i,
    int j,
    int k,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
) noexcept nogil:
    if component == 0:
        return Ex[i, j, k]
    if component == 1:
        return Ey[i, j, k]
    return Ez[i, j, k]


cdef inline void _electric_set(
    int component,
    int i,
    int j,
    int k,
    double value,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
) noexcept nogil:
    if component == 0:
        Ex[i, j, k] = value
    elif component == 1:
        Ey[i, j, k] = value
    else:
        Ez[i, j, k] = value


cdef inline double _magnetic_get(
    int component,
    int i,
    int j,
    int k,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
) noexcept nogil:
    if component == 0:
        return Hx[i, j, k]
    if component == 1:
        return Hy[i, j, k]
    return Hz[i, j, k]


cpdef void update_impedance_surfaces(
    int nthreads,
    np.int32_t[:, ::1] edge_info,
    float_or_double[:, ::1] edge_params,
    np.int32_t[:, ::1] h_info,
    float_or_double[::1] h_weight,
    np.int32_t[:, ::1] port_info,
    float_or_double[::1] port_g,
    np.int32_t[:, ::1] model_info,
    float_or_double[::1] model_F,
    float_or_double[::1] model_G,
    float_or_double[::1] model_L,
    float_or_double[::1] model_Z0,
    float_or_double[::1] state_old,
    float_or_double[::1] state_new,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
):
    """Advance every independent boundary E edge and its attached port states."""

    cdef Py_ssize_t edge_index, h_index, port_index, row, column
    cdef int component, i, j, k, h_start, h_count, port_start, port_count
    cdef int h_component, hi, hj, hk, model_index, state_start, state_count
    cdef int matrix_start, vector_start
    cdef double e_old, e_new, r_h, denominator, rhs, history, current, value
    cdef double g, z0

    for edge_index in prange(edge_info.shape[0], nogil=True, num_threads=nthreads):
        component = edge_info[edge_index, 0]
        i = edge_info[edge_index, 1]
        j = edge_info[edge_index, 2]
        k = edge_info[edge_index, 3]
        h_start = edge_info[edge_index, 4]
        h_count = edge_info[edge_index, 5]
        port_start = edge_info[edge_index, 6]
        port_count = edge_info[edge_index, 7]
        e_old = _electric_get(component, i, j, k, Ex, Ey, Ez)
        r_h = 0.0
        for h_index in range(h_start, h_start + h_count):
            h_component = h_info[h_index, 0]
            hi = h_info[h_index, 1]
            hj = h_info[h_index, 2]
            hk = h_info[h_index, 3]
            r_h = r_h + h_weight[h_index] * _magnetic_get(
                h_component, hi, hj, hk, Hx, Hy, Hz
            )

        denominator = edge_params[edge_index, 0]
        rhs = edge_params[edge_index, 1] * e_old + r_h
        for port_index in range(port_start, port_start + port_count):
            model_index = port_info[port_index, 0]
            state_start = port_info[port_index, 1]
            state_count = model_info[model_index, 0]
            vector_start = model_info[model_index, 2]
            history = 0.0
            for row in range(state_count):
                history = history + model_L[vector_start + row] * state_old[state_start + row]
            g = port_g[port_index]
            z0 = model_Z0[model_index]
            denominator = denominator - g / (2.0 * z0)
            rhs = rhs + g * e_old / (2.0 * z0) - g * history / z0

        e_new = rhs / denominator
        _electric_set(component, i, j, k, e_new, Ex, Ey, Ez)

        for port_index in range(port_start, port_start + port_count):
            model_index = port_info[port_index, 0]
            state_start = port_info[port_index, 1]
            state_count = model_info[model_index, 0]
            matrix_start = model_info[model_index, 1]
            vector_start = model_info[model_index, 2]
            history = 0.0
            for row in range(state_count):
                history = history + model_L[vector_start + row] * state_old[state_start + row]
            current = (0.5 * (e_new + e_old) - history) / model_Z0[model_index]
            for row in range(state_count):
                value = model_G[vector_start + row] * current
                for column in range(state_count):
                    value = value + model_F[
                        matrix_start + row * state_count + column
                    ] * state_old[state_start + column]
                state_new[state_start + row] = value
