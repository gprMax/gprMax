# cython: cdivision=True
import numpy as np
cimport numpy as np

cimport cython
from cython.parallel import prange
from gprMax.config cimport float_or_double


cdef inline int _imax(int a, int b) noexcept nogil:
    return a if a > b else b


cdef inline int _imin(int a, int b) noexcept nogil:
    return a if a < b else b


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void update_eigenmode_magnetic(
    int nthreads,
    int normal_axis,
    int direction_sign,
    int u0,
    int v0,
    int u1,
    int v1,
    int plane_index,
    int[:] owned_lower,
    int[:] owned_upper,
    float_or_double envelope,
    float_or_double[:, ::1] modal_Ex,
    float_or_double[:, ::1] modal_Ey,
    float_or_double[:, ::1] modal_Ez,
    float_or_double[:, ::1] updatecoeffsH,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
):
    cdef Py_ssize_t i, j, k
    cdef int target
    cdef float_or_double coeff
    cdef int x0 = owned_lower[0]
    cdef int y0 = owned_lower[1]
    cdef int z0 = owned_lower[2]
    cdef int x1 = owned_upper[0]
    cdef int y1 = owned_upper[1]
    cdef int z1 = owned_upper[2]

    if normal_axis == 0:
        i = plane_index
        if direction_sign > 0:
            target = i - 1
            if not x0 <= target < x1:
                return
            for j in prange(_imax(u0, y0), _imin(u1 + 1, y1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1, z1)):
                    coeff = updatecoeffsH[ID[4, target, j, k], 1]
                    Hy[target, j, k] -= coeff * envelope * modal_Ez[j - u0, k - v0]
            for j in prange(_imax(u0, y0), _imin(u1, y1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1 + 1, z1)):
                    coeff = updatecoeffsH[ID[5, target, j, k], 1]
                    Hz[target, j, k] += coeff * envelope * modal_Ey[j - u0, k - v0]
        else:
            target = i
            if not x0 <= target < x1:
                return
            for j in prange(_imax(u0, y0), _imin(u1 + 1, y1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1, z1)):
                    coeff = updatecoeffsH[ID[4, target, j, k], 1]
                    Hy[target, j, k] += coeff * envelope * modal_Ez[j - u0, k - v0]
            for j in prange(_imax(u0, y0), _imin(u1, y1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1 + 1, z1)):
                    coeff = updatecoeffsH[ID[5, target, j, k], 1]
                    Hz[target, j, k] -= coeff * envelope * modal_Ey[j - u0, k - v0]

    elif normal_axis == 1:
        j = plane_index
        if direction_sign > 0:
            target = j - 1
            if not y0 <= target < y1:
                return
            for i in prange(_imax(u0, x0), _imin(u1 + 1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1, z1)):
                    coeff = updatecoeffsH[ID[3, i, target, k], 2]
                    Hx[i, target, k] += coeff * envelope * modal_Ez[i - u0, k - v0]
            for i in prange(_imax(u0, x0), _imin(u1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1 + 1, z1)):
                    coeff = updatecoeffsH[ID[5, i, target, k], 2]
                    Hz[i, target, k] -= coeff * envelope * modal_Ex[i - u0, k - v0]
        else:
            target = j
            if not y0 <= target < y1:
                return
            for i in prange(_imax(u0, x0), _imin(u1 + 1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1, z1)):
                    coeff = updatecoeffsH[ID[3, i, target, k], 2]
                    Hx[i, target, k] -= coeff * envelope * modal_Ez[i - u0, k - v0]
            for i in prange(_imax(u0, x0), _imin(u1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1 + 1, z1)):
                    coeff = updatecoeffsH[ID[5, i, target, k], 2]
                    Hz[i, target, k] += coeff * envelope * modal_Ex[i - u0, k - v0]

    else:
        k = plane_index
        if direction_sign > 0:
            target = k - 1
            if not z0 <= target < z1:
                return
            for i in prange(_imax(u0, x0), _imin(u1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for j in range(_imax(v0, y0), _imin(v1 + 1, y1)):
                    coeff = updatecoeffsH[ID[4, i, j, target], 3]
                    Hy[i, j, target] += coeff * envelope * modal_Ex[i - u0, j - v0]
            for i in prange(_imax(u0, x0), _imin(u1 + 1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for j in range(_imax(v0, y0), _imin(v1, y1)):
                    coeff = updatecoeffsH[ID[3, i, j, target], 3]
                    Hx[i, j, target] -= coeff * envelope * modal_Ey[i - u0, j - v0]
        else:
            target = k
            if not z0 <= target < z1:
                return
            for i in prange(_imax(u0, x0), _imin(u1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for j in range(_imax(v0, y0), _imin(v1 + 1, y1)):
                    coeff = updatecoeffsH[ID[4, i, j, target], 3]
                    Hy[i, j, target] -= coeff * envelope * modal_Ex[i - u0, j - v0]
            for i in prange(_imax(u0, x0), _imin(u1 + 1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for j in range(_imax(v0, y0), _imin(v1, y1)):
                    coeff = updatecoeffsH[ID[3, i, j, target], 3]
                    Hx[i, j, target] += coeff * envelope * modal_Ey[i - u0, j - v0]


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void update_eigenmode_electric(
    int nthreads,
    int normal_axis,
    int direction_sign,
    int u0,
    int v0,
    int u1,
    int v1,
    int plane_index,
    int[:] owned_lower,
    int[:] owned_upper,
    float_or_double envelope,
    float_or_double[:, ::1] modal_Hx,
    float_or_double[:, ::1] modal_Hy,
    float_or_double[:, ::1] modal_Hz,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
):
    cdef Py_ssize_t i, j, k
    cdef float_or_double hsign = 1.0 if direction_sign > 0 else -1.0
    cdef float_or_double coeff
    cdef int x0 = owned_lower[0]
    cdef int y0 = owned_lower[1]
    cdef int z0 = owned_lower[2]
    cdef int x1 = owned_upper[0]
    cdef int y1 = owned_upper[1]
    cdef int z1 = owned_upper[2]

    if normal_axis == 0:
        i = plane_index
        if not x0 <= i < x1:
            return
        if direction_sign > 0:
            for j in prange(_imax(u0, y0), _imin(u1 + 1, y1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1, z1)):
                    coeff = updatecoeffsE[ID[2, i, j, k], 1]
                    Ez[i, j, k] -= coeff * envelope * hsign * modal_Hy[j - u0, k - v0]
            for j in prange(_imax(u0, y0), _imin(u1, y1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1 + 1, z1)):
                    coeff = updatecoeffsE[ID[1, i, j, k], 1]
                    Ey[i, j, k] += coeff * envelope * hsign * modal_Hz[j - u0, k - v0]
        else:
            for j in prange(_imax(u0, y0), _imin(u1 + 1, y1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1, z1)):
                    coeff = updatecoeffsE[ID[2, i, j, k], 1]
                    Ez[i, j, k] += coeff * envelope * hsign * modal_Hy[j - u0, k - v0]
            for j in prange(_imax(u0, y0), _imin(u1, y1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1 + 1, z1)):
                    coeff = updatecoeffsE[ID[1, i, j, k], 1]
                    Ey[i, j, k] -= coeff * envelope * hsign * modal_Hz[j - u0, k - v0]

    elif normal_axis == 1:
        j = plane_index
        if not y0 <= j < y1:
            return
        if direction_sign > 0:
            for i in prange(_imax(u0, x0), _imin(u1 + 1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1, z1)):
                    coeff = updatecoeffsE[ID[2, i, j, k], 2]
                    Ez[i, j, k] += coeff * envelope * hsign * modal_Hx[i - u0, k - v0]
            for i in prange(_imax(u0, x0), _imin(u1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1 + 1, z1)):
                    coeff = updatecoeffsE[ID[0, i, j, k], 2]
                    Ex[i, j, k] -= coeff * envelope * hsign * modal_Hz[i - u0, k - v0]
        else:
            for i in prange(_imax(u0, x0), _imin(u1 + 1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1, z1)):
                    coeff = updatecoeffsE[ID[2, i, j, k], 2]
                    Ez[i, j, k] -= coeff * envelope * hsign * modal_Hx[i - u0, k - v0]
            for i in prange(_imax(u0, x0), _imin(u1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for k in range(_imax(v0, z0), _imin(v1 + 1, z1)):
                    coeff = updatecoeffsE[ID[0, i, j, k], 2]
                    Ex[i, j, k] += coeff * envelope * hsign * modal_Hz[i - u0, k - v0]

    else:
        k = plane_index
        if not z0 <= k < z1:
            return
        if direction_sign > 0:
            for i in prange(_imax(u0, x0), _imin(u1 + 1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for j in range(_imax(v0, y0), _imin(v1, y1)):
                    coeff = updatecoeffsE[ID[1, i, j, k], 3]
                    Ey[i, j, k] -= coeff * envelope * hsign * modal_Hx[i - u0, j - v0]
            for i in prange(_imax(u0, x0), _imin(u1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for j in range(_imax(v0, y0), _imin(v1 + 1, y1)):
                    coeff = updatecoeffsE[ID[0, i, j, k], 3]
                    Ex[i, j, k] += coeff * envelope * hsign * modal_Hy[i - u0, j - v0]
        else:
            for i in prange(_imax(u0, x0), _imin(u1 + 1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for j in range(_imax(v0, y0), _imin(v1, y1)):
                    coeff = updatecoeffsE[ID[1, i, j, k], 3]
                    Ey[i, j, k] += coeff * envelope * hsign * modal_Hx[i - u0, j - v0]
            for i in prange(_imax(u0, x0), _imin(u1, x1), nogil=True, schedule="static", num_threads=nthreads):
                for j in range(_imax(v0, y0), _imin(v1 + 1, y1)):
                    coeff = updatecoeffsE[ID[0, i, j, k], 3]
                    Ex[i, j, k] -= coeff * envelope * hsign * modal_Hy[i - u0, j - v0]
