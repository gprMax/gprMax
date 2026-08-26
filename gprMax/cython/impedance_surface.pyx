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


cdef inline double _port_history(
    Py_ssize_t port_index,
    np.int32_t[:, ::1] port_info,
    np.int32_t[:, ::1] model_info,
    float_or_double[::1] state_y,
) noexcept nogil:
    """Return the local Foster history ``sum(y_m)`` for one surface port."""

    cdef Py_ssize_t row
    cdef int model_index = port_info[port_index, 0]
    cdef int state_start = port_info[port_index, 1]
    cdef int state_count = model_info[model_index, 0]
    cdef double history = 0.0

    for row in range(state_count):
        history = history + state_y[state_start + row]
    return history


cdef inline void _advance_port(
    Py_ssize_t port_index,
    double midpoint_e,
    double history,
    np.int32_t[:, ::1] port_info,
    float_or_double[::1] port_inv_Z0,
    np.int32_t[:, ::1] model_info,
    float_or_double[::1] model_f,
    float_or_double[::1] model_q,
    float_or_double[::1] state_y,
) noexcept nogil:
    """Advance independent scaled Foster states in place for one port."""

    cdef Py_ssize_t row
    cdef int model_index = port_info[port_index, 0]
    cdef int state_start = port_info[port_index, 1]
    cdef int state_count = model_info[model_index, 0]
    cdef int coefficient_start = model_info[model_index, 1]
    cdef double current

    if state_count == 0:
        return
    current = (midpoint_e - history) * port_inv_Z0[port_index]
    for row in range(state_count):
        state_y[state_start + row] = (
            model_f[coefficient_start + row] * state_y[state_start + row]
            + model_q[coefficient_start + row] * current
        )


cpdef void update_impedance_surfaces(
    int nthreads,
    np.int32_t[:, ::1] edge_info,
    float_or_double[:, ::1] edge_runtime,
    np.int32_t[:, ::1] h_info,
    float_or_double[::1] h_weight,
    np.int32_t[:, ::1] port_info,
    float_or_double[::1] port_g_over_Z0,
    float_or_double[::1] port_inv_Z0,
    np.int32_t[:, ::1] model_info,
    float_or_double[::1] model_f,
    float_or_double[::1] model_q,
    float_or_double[::1] state_y,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
):
    """Advance every boundary E edge and its local scalar Foster states."""

    cdef Py_ssize_t edge_index, h_index
    cdef int component, i, j, k, h_start, h_count, port_start, port_count
    cdef int h_component, hi, hj, hk
    cdef double e_old, e_new, midpoint_e, r_h, rhs, history0, history1

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

        history0 = _port_history(port_start, port_info, model_info, state_y)
        history1 = 0.0
        rhs = (
            edge_runtime[edge_index, 0] * e_old
            + r_h
            - port_g_over_Z0[port_start] * history0
        )
        if port_count == 2:
            history1 = _port_history(port_start + 1, port_info, model_info, state_y)
            rhs = rhs - port_g_over_Z0[port_start + 1] * history1

        e_new = rhs * edge_runtime[edge_index, 1]
        _electric_set(component, i, j, k, e_new, Ex, Ey, Ez)

        midpoint_e = 0.5 * (e_new + e_old)
        _advance_port(
            port_start,
            midpoint_e,
            history0,
            port_info,
            port_inv_Z0,
            model_info,
            model_f,
            model_q,
            state_y,
        )
        if port_count == 2:
            _advance_port(
                port_start + 1,
                midpoint_e,
                history1,
                port_info,
                port_inv_Z0,
                model_info,
                model_f,
                model_q,
                state_y,
            )
