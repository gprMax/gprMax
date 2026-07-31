import numpy as np
cimport numpy as np

cimport cython
from cython.parallel import prange


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
    float envelope,
    float[:, ::1] modal_Ex,
    float[:, ::1] modal_Ey,
    float[:, ::1] modal_Ez,
    float[:, ::1] updatecoeffsH,
    np.uint32_t[:, :, :, ::1] ID,
    float[:, :, ::1] Hx,
    float[:, :, ::1] Hy,
    float[:, :, ::1] Hz,
):
    cdef Py_ssize_t i, j, k
    cdef int target
    cdef float coeff

    if normal_axis == 0:
        i = plane_index
        if direction_sign > 0:
            target = i - 1
            for j in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1):
                    coeff = updatecoeffsH[ID[4, target, j, k], 1]
                    Hy[target, j, k] -= coeff * envelope * modal_Ez[j - u0, k - v0]
            for j in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1 + 1):
                    coeff = updatecoeffsH[ID[5, target, j, k], 1]
                    Hz[target, j, k] += coeff * envelope * modal_Ey[j - u0, k - v0]
        else:
            target = i
            for j in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1):
                    coeff = updatecoeffsH[ID[4, target, j, k], 1]
                    Hy[target, j, k] += coeff * envelope * modal_Ez[j - u0, k - v0]
            for j in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1 + 1):
                    coeff = updatecoeffsH[ID[5, target, j, k], 1]
                    Hz[target, j, k] -= coeff * envelope * modal_Ey[j - u0, k - v0]

    elif normal_axis == 1:
        j = plane_index
        if direction_sign > 0:
            target = j - 1
            for i in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1):
                    coeff = updatecoeffsH[ID[3, i, target, k], 2]
                    Hx[i, target, k] += coeff * envelope * modal_Ez[i - u0, k - v0]
            for i in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1 + 1):
                    coeff = updatecoeffsH[ID[5, i, target, k], 2]
                    Hz[i, target, k] -= coeff * envelope * modal_Ex[i - u0, k - v0]
        else:
            target = j
            for i in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1):
                    coeff = updatecoeffsH[ID[3, i, target, k], 2]
                    Hx[i, target, k] -= coeff * envelope * modal_Ez[i - u0, k - v0]
            for i in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1 + 1):
                    coeff = updatecoeffsH[ID[5, i, target, k], 2]
                    Hz[i, target, k] += coeff * envelope * modal_Ex[i - u0, k - v0]

    else:
        k = plane_index
        if direction_sign > 0:
            target = k - 1
            for i in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for j in range(v0, v1 + 1):
                    coeff = updatecoeffsH[ID[4, i, j, target], 3]
                    Hy[i, j, target] += coeff * envelope * modal_Ex[i - u0, j - v0]
            for i in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for j in range(v0, v1):
                    coeff = updatecoeffsH[ID[3, i, j, target], 3]
                    Hx[i, j, target] -= coeff * envelope * modal_Ey[i - u0, j - v0]
        else:
            target = k
            for i in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for j in range(v0, v1 + 1):
                    coeff = updatecoeffsH[ID[4, i, j, target], 3]
                    Hy[i, j, target] -= coeff * envelope * modal_Ex[i - u0, j - v0]
            for i in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for j in range(v0, v1):
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
    float envelope,
    float[:, ::1] modal_Hx,
    float[:, ::1] modal_Hy,
    float[:, ::1] modal_Hz,
    float[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float[:, :, ::1] Ex,
    float[:, :, ::1] Ey,
    float[:, :, ::1] Ez,
):
    cdef Py_ssize_t i, j, k
    cdef float hsign = 1.0 if direction_sign > 0 else -1.0
    cdef float coeff

    if normal_axis == 0:
        i = plane_index
        if direction_sign > 0:
            for j in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1):
                    coeff = updatecoeffsE[ID[2, i, j, k], 1]
                    Ez[i, j, k] -= coeff * envelope * hsign * modal_Hy[j - u0, k - v0]
            for j in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1 + 1):
                    coeff = updatecoeffsE[ID[1, i, j, k], 1]
                    Ey[i, j, k] += coeff * envelope * hsign * modal_Hz[j - u0, k - v0]
        else:
            for j in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1):
                    coeff = updatecoeffsE[ID[2, i, j, k], 1]
                    Ez[i, j, k] += coeff * envelope * hsign * modal_Hy[j - u0, k - v0]
            for j in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1 + 1):
                    coeff = updatecoeffsE[ID[1, i, j, k], 1]
                    Ey[i, j, k] -= coeff * envelope * hsign * modal_Hz[j - u0, k - v0]

    elif normal_axis == 1:
        j = plane_index
        if direction_sign > 0:
            for i in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1):
                    coeff = updatecoeffsE[ID[2, i, j, k], 2]
                    Ez[i, j, k] += coeff * envelope * hsign * modal_Hx[i - u0, k - v0]
            for i in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1 + 1):
                    coeff = updatecoeffsE[ID[0, i, j, k], 2]
                    Ex[i, j, k] -= coeff * envelope * hsign * modal_Hz[i - u0, k - v0]
        else:
            for i in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1):
                    coeff = updatecoeffsE[ID[2, i, j, k], 2]
                    Ez[i, j, k] -= coeff * envelope * hsign * modal_Hx[i - u0, k - v0]
            for i in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for k in range(v0, v1 + 1):
                    coeff = updatecoeffsE[ID[0, i, j, k], 2]
                    Ex[i, j, k] += coeff * envelope * hsign * modal_Hz[i - u0, k - v0]

    else:
        k = plane_index
        if direction_sign > 0:
            for i in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for j in range(v0, v1):
                    coeff = updatecoeffsE[ID[1, i, j, k], 3]
                    Ey[i, j, k] -= coeff * envelope * hsign * modal_Hx[i - u0, j - v0]
            for i in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for j in range(v0, v1 + 1):
                    coeff = updatecoeffsE[ID[0, i, j, k], 3]
                    Ex[i, j, k] += coeff * envelope * hsign * modal_Hy[i - u0, j - v0]
        else:
            for i in prange(u0, u1 + 1, nogil=True, schedule="static", num_threads=nthreads):
                for j in range(v0, v1):
                    coeff = updatecoeffsE[ID[1, i, j, k], 3]
                    Ey[i, j, k] += coeff * envelope * hsign * modal_Hx[i - u0, j - v0]
            for i in prange(u0, u1, nogil=True, schedule="static", num_threads=nthreads):
                for j in range(v0, v1 + 1):
                    coeff = updatecoeffsE[ID[0, i, j, k], 3]
                    Ex[i, j, k] -= coeff * envelope * hsign * modal_Hy[i - u0, j - v0]
