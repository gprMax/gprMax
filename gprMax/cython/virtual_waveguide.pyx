# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
# Authors: Craig Warren, Antonis Giannopoulos, John Hartley, and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

"""Yee-face coupling between the main grid and a virtual waveguide."""

# cython: cdivision=True

import numpy as np
cimport numpy as np

cimport cython
from cython.parallel import prange
from gprMax.config cimport float_or_double


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void couple_virtual_waveguide_magnetic(
    int nthreads,
    int normal_axis,
    int direction_sign,
    int u0,
    int v0,
    int u1,
    int v1,
    int plane_index,
    float_or_double[:, :, ::1] main_Hx,
    float_or_double[:, :, ::1] main_Hy,
    float_or_double[:, :, ::1] main_Hz,
    float_or_double[:, :, ::1] aux_Hx,
    float_or_double[:, :, ::1] aux_Hy,
    float_or_double[:, :, ::1] aux_Hz,
):
    """Share the aperture-normal H field and clear the disconnected rear."""

    cdef Py_ssize_t i, j, k, u, v
    cdef int nu = u1 - u0
    cdef int nv = v1 - v0
    cdef int aperture
    cdef int normal_cells

    if normal_axis == 0:
        normal_cells = aux_Hx.shape[0] - 1
        aperture = 0 if direction_sign < 0 else normal_cells
        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                aux_Hx[aperture, u, v] = main_Hx[plane_index, u0 + u, v0 + v]

        if direction_sign < 0:
            for i in prange(plane_index, main_Hx.shape[0], nogil=True, schedule="static", num_threads=nthreads):
                if i > plane_index:
                    for u in range(nu):
                        for v in range(nv):
                            main_Hx[i, u0 + u, v0 + v] = 0
                if i < main_Hx.shape[0] - 1:
                    for u in range(nu + 1):
                        for v in range(nv):
                            main_Hy[i, u0 + u, v0 + v] = 0
                    for u in range(nu):
                        for v in range(nv + 1):
                            main_Hz[i, u0 + u, v0 + v] = 0
        else:
            for i in prange(0, plane_index, nogil=True, schedule="static", num_threads=nthreads):
                for u in range(nu):
                    for v in range(nv):
                        main_Hx[i, u0 + u, v0 + v] = 0
                for u in range(nu + 1):
                    for v in range(nv):
                        main_Hy[i, u0 + u, v0 + v] = 0
                for u in range(nu):
                    for v in range(nv + 1):
                        main_Hz[i, u0 + u, v0 + v] = 0

    elif normal_axis == 1:
        normal_cells = aux_Hy.shape[1] - 1
        aperture = 0 if direction_sign < 0 else normal_cells
        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                aux_Hy[u, aperture, v] = main_Hy[u0 + u, plane_index, v0 + v]

        if direction_sign < 0:
            for j in prange(plane_index, main_Hy.shape[1], nogil=True, schedule="static", num_threads=nthreads):
                if j > plane_index:
                    for u in range(nu):
                        for v in range(nv):
                            main_Hy[u0 + u, j, v0 + v] = 0
                if j < main_Hy.shape[1] - 1:
                    for u in range(nu + 1):
                        for v in range(nv):
                            main_Hx[u0 + u, j, v0 + v] = 0
                    for u in range(nu):
                        for v in range(nv + 1):
                            main_Hz[u0 + u, j, v0 + v] = 0
        else:
            for j in prange(0, plane_index, nogil=True, schedule="static", num_threads=nthreads):
                for u in range(nu):
                    for v in range(nv):
                        main_Hy[u0 + u, j, v0 + v] = 0
                for u in range(nu + 1):
                    for v in range(nv):
                        main_Hx[u0 + u, j, v0 + v] = 0
                for u in range(nu):
                    for v in range(nv + 1):
                        main_Hz[u0 + u, j, v0 + v] = 0

    else:
        normal_cells = aux_Hz.shape[2] - 1
        aperture = 0 if direction_sign < 0 else normal_cells
        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                aux_Hz[u, v, aperture] = main_Hz[u0 + u, v0 + v, plane_index]

        if direction_sign < 0:
            for k in prange(plane_index, main_Hz.shape[2], nogil=True, schedule="static", num_threads=nthreads):
                if k > plane_index:
                    for u in range(nu):
                        for v in range(nv):
                            main_Hz[u0 + u, v0 + v, k] = 0
                if k < main_Hz.shape[2] - 1:
                    for u in range(nu + 1):
                        for v in range(nv):
                            main_Hx[u0 + u, v0 + v, k] = 0
                    for u in range(nu):
                        for v in range(nv + 1):
                            main_Hy[u0 + u, v0 + v, k] = 0
        else:
            for k in prange(0, plane_index, nogil=True, schedule="static", num_threads=nthreads):
                for u in range(nu):
                    for v in range(nv):
                        main_Hz[u0 + u, v0 + v, k] = 0
                for u in range(nu + 1):
                    for v in range(nv):
                        main_Hx[u0 + u, v0 + v, k] = 0
                for u in range(nu):
                    for v in range(nv + 1):
                        main_Hy[u0 + u, v0 + v, k] = 0


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void couple_virtual_waveguide_electric(
    int nthreads,
    int normal_axis,
    int direction_sign,
    int u0,
    int v0,
    int u1,
    int v1,
    int plane_index,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] aux_ID,
    float_or_double[:, :, ::1] main_Ex,
    float_or_double[:, :, ::1] main_Ey,
    float_or_double[:, :, ::1] main_Ez,
    float_or_double[:, :, ::1] main_Hx,
    float_or_double[:, :, ::1] main_Hy,
    float_or_double[:, :, ::1] main_Hz,
    float_or_double[:, :, ::1] aux_Ex,
    float_or_double[:, :, ::1] aux_Ey,
    float_or_double[:, :, ::1] aux_Ez,
    float_or_double[:, :, ::1] aux_Hx,
    float_or_double[:, :, ::1] aux_Hy,
    float_or_double[:, :, ::1] aux_Hz,
):
    """Close the Yee curl across the split, share E, and clear the rear."""

    cdef Py_ssize_t i, j, k, u, v
    cdef int nu = u1 - u0
    cdef int nv = v1 - v0
    cdef int aperture, inside, material
    cdef float_or_double cross_field

    if normal_axis == 0:
        aperture = 0 if direction_sign < 0 else aux_Ex.shape[0] - 1
        inside = 0 if direction_sign < 0 else aperture - 1

        # Ey: z derivative is local; x derivative crosses the aperture.
        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(1, nv):
                material = aux_ID[1, aperture, u, v]
                if direction_sign < 0:
                    cross_field = aux_Hz[0, u, v] - main_Hz[plane_index - 1, u0 + u, v0 + v]
                else:
                    cross_field = main_Hz[plane_index, u0 + u, v0 + v] - aux_Hz[inside, u, v]
                aux_Ey[aperture, u, v] = (
                    updatecoeffsE[material, 0] * aux_Ey[aperture, u, v]
                    + updatecoeffsE[material, 3] * (aux_Hx[aperture, u, v] - aux_Hx[aperture, u, v - 1])
                    - updatecoeffsE[material, 1] * cross_field
                )

        # Ez: y derivative is local; x derivative crosses the aperture.
        for u in prange(1, nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                material = aux_ID[2, aperture, u, v]
                if direction_sign < 0:
                    cross_field = aux_Hy[0, u, v] - main_Hy[plane_index - 1, u0 + u, v0 + v]
                else:
                    cross_field = main_Hy[plane_index, u0 + u, v0 + v] - aux_Hy[inside, u, v]
                aux_Ez[aperture, u, v] = (
                    updatecoeffsE[material, 0] * aux_Ez[aperture, u, v]
                    + updatecoeffsE[material, 1] * cross_field
                    - updatecoeffsE[material, 2] * (aux_Hx[aperture, u, v] - aux_Hx[aperture, u - 1, v])
                )

        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv + 1):
                main_Ey[plane_index, u0 + u, v0 + v] = aux_Ey[aperture, u, v]
        for u in prange(nu + 1, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                main_Ez[plane_index, u0 + u, v0 + v] = aux_Ez[aperture, u, v]
        if direction_sign < 0:
            for u in prange(nu + 1, nogil=True, schedule="static", num_threads=nthreads):
                for v in range(nv + 1):
                    main_Ex[plane_index, u0 + u, v0 + v] = aux_Ex[0, u, v]
            for i in prange(plane_index + 1, main_Ex.shape[0], nogil=True, schedule="static", num_threads=nthreads):
                for u in range(nu + 1):
                    for v in range(nv + 1):
                        main_Ex[i, u0 + u, v0 + v] = 0
                for u in range(nu):
                    for v in range(nv + 1):
                        main_Ey[i, u0 + u, v0 + v] = 0
                for u in range(nu + 1):
                    for v in range(nv):
                        main_Ez[i, u0 + u, v0 + v] = 0
        else:
            for u in prange(nu + 1, nogil=True, schedule="static", num_threads=nthreads):
                for v in range(nv + 1):
                    main_Ex[plane_index - 1, u0 + u, v0 + v] = aux_Ex[inside, u, v]
            for i in prange(0, plane_index, nogil=True, schedule="static", num_threads=nthreads):
                for u in range(nu + 1):
                    for v in range(nv + 1):
                        if i < plane_index - 1:
                            main_Ex[i, u0 + u, v0 + v] = 0
                for u in range(nu):
                    for v in range(nv + 1):
                        main_Ey[i, u0 + u, v0 + v] = 0
                for u in range(nu + 1):
                    for v in range(nv):
                        main_Ez[i, u0 + u, v0 + v] = 0

    elif normal_axis == 1:
        aperture = 0 if direction_sign < 0 else aux_Ey.shape[1] - 1
        inside = 0 if direction_sign < 0 else aperture - 1

        # Ex: z derivative is local; y derivative crosses the aperture.
        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(1, nv):
                material = aux_ID[0, u, aperture, v]
                if direction_sign < 0:
                    cross_field = aux_Hz[u, 0, v] - main_Hz[u0 + u, plane_index - 1, v0 + v]
                else:
                    cross_field = main_Hz[u0 + u, plane_index, v0 + v] - aux_Hz[u, inside, v]
                aux_Ex[u, aperture, v] = (
                    updatecoeffsE[material, 0] * aux_Ex[u, aperture, v]
                    + updatecoeffsE[material, 2] * cross_field
                    - updatecoeffsE[material, 3] * (aux_Hy[u, aperture, v] - aux_Hy[u, aperture, v - 1])
                )

        # Ez: x derivative is local; y derivative crosses the aperture.
        for u in prange(1, nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                material = aux_ID[2, u, aperture, v]
                if direction_sign < 0:
                    cross_field = aux_Hx[u, 0, v] - main_Hx[u0 + u, plane_index - 1, v0 + v]
                else:
                    cross_field = main_Hx[u0 + u, plane_index, v0 + v] - aux_Hx[u, inside, v]
                aux_Ez[u, aperture, v] = (
                    updatecoeffsE[material, 0] * aux_Ez[u, aperture, v]
                    + updatecoeffsE[material, 1] * (aux_Hy[u, aperture, v] - aux_Hy[u - 1, aperture, v])
                    - updatecoeffsE[material, 2] * cross_field
                )

        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv + 1):
                main_Ex[u0 + u, plane_index, v0 + v] = aux_Ex[u, aperture, v]
        for u in prange(nu + 1, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                main_Ez[u0 + u, plane_index, v0 + v] = aux_Ez[u, aperture, v]
        if direction_sign < 0:
            for u in prange(nu + 1, nogil=True, schedule="static", num_threads=nthreads):
                for v in range(nv + 1):
                    main_Ey[u0 + u, plane_index, v0 + v] = aux_Ey[u, 0, v]
            for j in prange(plane_index + 1, main_Ey.shape[1], nogil=True, schedule="static", num_threads=nthreads):
                for u in range(nu + 1):
                    for v in range(nv + 1):
                        main_Ey[u0 + u, j, v0 + v] = 0
                for u in range(nu):
                    for v in range(nv + 1):
                        main_Ex[u0 + u, j, v0 + v] = 0
                for u in range(nu + 1):
                    for v in range(nv):
                        main_Ez[u0 + u, j, v0 + v] = 0
        else:
            for u in prange(nu + 1, nogil=True, schedule="static", num_threads=nthreads):
                for v in range(nv + 1):
                    main_Ey[u0 + u, plane_index - 1, v0 + v] = aux_Ey[u, inside, v]
            for j in prange(0, plane_index, nogil=True, schedule="static", num_threads=nthreads):
                for u in range(nu + 1):
                    for v in range(nv + 1):
                        if j < plane_index - 1:
                            main_Ey[u0 + u, j, v0 + v] = 0
                for u in range(nu):
                    for v in range(nv + 1):
                        main_Ex[u0 + u, j, v0 + v] = 0
                for u in range(nu + 1):
                    for v in range(nv):
                        main_Ez[u0 + u, j, v0 + v] = 0

    else:
        aperture = 0 if direction_sign < 0 else aux_Ez.shape[2] - 1
        inside = 0 if direction_sign < 0 else aperture - 1

        # Ex: y derivative is local; z derivative crosses the aperture.
        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(1, nv):
                material = aux_ID[0, u, v, aperture]
                if direction_sign < 0:
                    cross_field = aux_Hy[u, v, 0] - main_Hy[u0 + u, v0 + v, plane_index - 1]
                else:
                    cross_field = main_Hy[u0 + u, v0 + v, plane_index] - aux_Hy[u, v, inside]
                aux_Ex[u, v, aperture] = (
                    updatecoeffsE[material, 0] * aux_Ex[u, v, aperture]
                    + updatecoeffsE[material, 2] * (aux_Hz[u, v, aperture] - aux_Hz[u, v - 1, aperture])
                    - updatecoeffsE[material, 3] * cross_field
                )

        # Ey: x derivative is local; z derivative crosses the aperture.
        for u in prange(1, nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                material = aux_ID[1, u, v, aperture]
                if direction_sign < 0:
                    cross_field = aux_Hx[u, v, 0] - main_Hx[u0 + u, v0 + v, plane_index - 1]
                else:
                    cross_field = main_Hx[u0 + u, v0 + v, plane_index] - aux_Hx[u, v, inside]
                aux_Ey[u, v, aperture] = (
                    updatecoeffsE[material, 0] * aux_Ey[u, v, aperture]
                    + updatecoeffsE[material, 3] * cross_field
                    - updatecoeffsE[material, 1] * (aux_Hz[u, v, aperture] - aux_Hz[u - 1, v, aperture])
                )

        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv + 1):
                main_Ex[u0 + u, v0 + v, plane_index] = aux_Ex[u, v, aperture]
        for u in prange(nu + 1, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                main_Ey[u0 + u, v0 + v, plane_index] = aux_Ey[u, v, aperture]
        if direction_sign < 0:
            for u in prange(nu + 1, nogil=True, schedule="static", num_threads=nthreads):
                for v in range(nv + 1):
                    main_Ez[u0 + u, v0 + v, plane_index] = aux_Ez[u, v, 0]
            for k in prange(plane_index + 1, main_Ez.shape[2], nogil=True, schedule="static", num_threads=nthreads):
                for u in range(nu + 1):
                    for v in range(nv + 1):
                        main_Ez[u0 + u, v0 + v, k] = 0
                for u in range(nu):
                    for v in range(nv + 1):
                        main_Ex[u0 + u, v0 + v, k] = 0
                for u in range(nu + 1):
                    for v in range(nv):
                        main_Ey[u0 + u, v0 + v, k] = 0
        else:
            for u in prange(nu + 1, nogil=True, schedule="static", num_threads=nthreads):
                for v in range(nv + 1):
                    main_Ez[u0 + u, v0 + v, plane_index - 1] = aux_Ez[u, v, inside]
            for k in prange(0, plane_index, nogil=True, schedule="static", num_threads=nthreads):
                for u in range(nu + 1):
                    for v in range(nv + 1):
                        if k < plane_index - 1:
                            main_Ez[u0 + u, v0 + v, k] = 0
                for u in range(nu):
                    for v in range(nv + 1):
                        main_Ex[u0 + u, v0 + v, k] = 0
                for u in range(nu + 1):
                    for v in range(nv):
                        main_Ey[u0 + u, v0 + v, k] = 0


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void couple_virtual_waveguide_electric_aperture(
    int nthreads,
    int normal_axis,
    int direction_sign,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] aux_ID,
    float_or_double[:, ::1] main_Hu,
    float_or_double[:, ::1] main_Hv,
    float_or_double[:, :, ::1] aux_Ex,
    float_or_double[:, :, ::1] aux_Ey,
    float_or_double[:, :, ::1] aux_Ez,
    float_or_double[:, :, ::1] aux_Hx,
    float_or_double[:, :, ::1] aux_Hy,
    float_or_double[:, :, ::1] aux_Hz,
):
    """Close an MPI virtual-guide aperture from compact global H sheets.

    ``main_Hu`` and ``main_Hv`` are the magnetic components parallel to
    the first and second transverse axes, respectively. They are assembled
    collectively after the main-grid H halo exchange.
    """

    cdef Py_ssize_t u, v
    cdef int aperture, inside, material
    cdef int nu = main_Hv.shape[0]
    cdef int nv = main_Hu.shape[1]
    cdef float_or_double cross_field

    if normal_axis == 0:
        aperture = 0 if direction_sign < 0 else aux_Ex.shape[0] - 1
        inside = 0 if direction_sign < 0 else aperture - 1
        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(1, nv):
                material = aux_ID[1, aperture, u, v]
                if direction_sign < 0:
                    cross_field = aux_Hz[0, u, v] - main_Hv[u, v]
                else:
                    cross_field = main_Hv[u, v] - aux_Hz[inside, u, v]
                aux_Ey[aperture, u, v] = (
                    updatecoeffsE[material, 0] * aux_Ey[aperture, u, v]
                    + updatecoeffsE[material, 3]
                    * (aux_Hx[aperture, u, v] - aux_Hx[aperture, u, v - 1])
                    - updatecoeffsE[material, 1] * cross_field
                )
        for u in prange(1, nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                material = aux_ID[2, aperture, u, v]
                if direction_sign < 0:
                    cross_field = aux_Hy[0, u, v] - main_Hu[u, v]
                else:
                    cross_field = main_Hu[u, v] - aux_Hy[inside, u, v]
                aux_Ez[aperture, u, v] = (
                    updatecoeffsE[material, 0] * aux_Ez[aperture, u, v]
                    + updatecoeffsE[material, 1] * cross_field
                    - updatecoeffsE[material, 2]
                    * (aux_Hx[aperture, u, v] - aux_Hx[aperture, u - 1, v])
                )

    elif normal_axis == 1:
        aperture = 0 if direction_sign < 0 else aux_Ey.shape[1] - 1
        inside = 0 if direction_sign < 0 else aperture - 1
        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(1, nv):
                material = aux_ID[0, u, aperture, v]
                if direction_sign < 0:
                    cross_field = aux_Hz[u, 0, v] - main_Hv[u, v]
                else:
                    cross_field = main_Hv[u, v] - aux_Hz[u, inside, v]
                aux_Ex[u, aperture, v] = (
                    updatecoeffsE[material, 0] * aux_Ex[u, aperture, v]
                    + updatecoeffsE[material, 2] * cross_field
                    - updatecoeffsE[material, 3]
                    * (aux_Hy[u, aperture, v] - aux_Hy[u, aperture, v - 1])
                )
        for u in prange(1, nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                material = aux_ID[2, u, aperture, v]
                if direction_sign < 0:
                    cross_field = aux_Hx[u, 0, v] - main_Hu[u, v]
                else:
                    cross_field = main_Hu[u, v] - aux_Hx[u, inside, v]
                aux_Ez[u, aperture, v] = (
                    updatecoeffsE[material, 0] * aux_Ez[u, aperture, v]
                    + updatecoeffsE[material, 1]
                    * (aux_Hy[u, aperture, v] - aux_Hy[u - 1, aperture, v])
                    - updatecoeffsE[material, 2] * cross_field
                )

    else:
        aperture = 0 if direction_sign < 0 else aux_Ez.shape[2] - 1
        inside = 0 if direction_sign < 0 else aperture - 1
        for u in prange(nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(1, nv):
                material = aux_ID[0, u, v, aperture]
                if direction_sign < 0:
                    cross_field = aux_Hy[u, v, 0] - main_Hv[u, v]
                else:
                    cross_field = main_Hv[u, v] - aux_Hy[u, v, inside]
                aux_Ex[u, v, aperture] = (
                    updatecoeffsE[material, 0] * aux_Ex[u, v, aperture]
                    + updatecoeffsE[material, 2]
                    * (aux_Hz[u, v, aperture] - aux_Hz[u, v - 1, aperture])
                    - updatecoeffsE[material, 3] * cross_field
                )
        for u in prange(1, nu, nogil=True, schedule="static", num_threads=nthreads):
            for v in range(nv):
                material = aux_ID[1, u, v, aperture]
                if direction_sign < 0:
                    cross_field = aux_Hx[u, v, 0] - main_Hu[u, v]
                else:
                    cross_field = main_Hu[u, v] - aux_Hx[u, v, inside]
                aux_Ey[u, v, aperture] = (
                    updatecoeffsE[material, 0] * aux_Ey[u, v, aperture]
                    + updatecoeffsE[material, 3] * cross_field
                    - updatecoeffsE[material, 1]
                    * (aux_Hz[u, v, aperture] - aux_Hz[u - 1, v, aperture])
                )
