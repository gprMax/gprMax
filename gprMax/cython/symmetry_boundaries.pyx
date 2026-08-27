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

import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.config cimport float_or_double


# Per-iteration PMC ghost-node E update for the face-interior (non-edge)
# region of a PMC symmetry boundary, standard (non-dispersive) materials
# only. One function per domain face, mirroring the per-face convention
# already used for PML (pml_updates_electric_HORIPML.pyx's xminus/xplus/etc.)
#
# Ghost-node derivation: tangential H is odd under the PMC mirror, so the
# ghost H node just outside the domain equals minus the real interior H node
# it mirrors. Substituting into the standard curl term collapses the missing
# outside-neighbour difference into double one real H value. A "0" face
# (x0/y0/z0) uses the wall's own H index with the bulk kernel's own sign; a
# "max" face (xmax/ymax/zmax) uses the interior-adjacent H index (one cell
# in, not the wall index) with the opposite sign.


cpdef void update_symmetry_boundary_electric_x0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates the Ey and Ez face-interior components on the x0 PMC face.

    Args:
        nx, ny, nz: ints for grid size in cells.
        nthreads: int for number of threads to use.
        updatecoeffsE, ID, E, H: memoryviews to access update coefficients,
                                  ID and field component arrays.
    """
    cdef Py_ssize_t j, k
    cdef int materialEy, materialEz

    # Ey[0, 0:ny, 1:nz] - standard term Hx (dep k), ghost term Hz (dep i, doubled)
    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEy = ID[1, 0, j, k]
            Ey[0, j, k] = (updatecoeffsE[materialEy, 0] * Ey[0, j, k] +
                           updatecoeffsE[materialEy, 3] * (Hx[0, j, k] - Hx[0, j, k - 1]) -
                           updatecoeffsE[materialEy, 1] * (2 * Hz[0, j, k]))

    # Ez[0, 1:ny, 0:nz] - standard term Hx (dep j), ghost term Hy (dep i, doubled)
    for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, 0, j, k]
            Ez[0, j, k] = (updatecoeffsE[materialEz, 0] * Ez[0, j, k] -
                           updatecoeffsE[materialEz, 2] * (Hx[0, j, k] - Hx[0, j - 1, k]) +
                           updatecoeffsE[materialEz, 1] * (2 * Hy[0, j, k]))


cpdef void update_symmetry_boundary_electric_xmax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates the Ey and Ez face-interior components on the xmax PMC face."""
    cdef Py_ssize_t j, k
    cdef int materialEy, materialEz

    # Ey[nx, 0:ny, 1:nz] - ghost uses interior-adjacent Hz index, flipped sign
    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEy = ID[1, nx, j, k]
            Ey[nx, j, k] = (updatecoeffsE[materialEy, 0] * Ey[nx, j, k] +
                             updatecoeffsE[materialEy, 3] * (Hx[nx, j, k] - Hx[nx, j, k - 1]) +
                             updatecoeffsE[materialEy, 1] * (2 * Hz[nx - 1, j, k]))

    # Ez[nx, 1:ny, 0:nz] - ghost uses interior-adjacent Hy index, flipped sign
    for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, nx, j, k]
            Ez[nx, j, k] = (updatecoeffsE[materialEz, 0] * Ez[nx, j, k] -
                             updatecoeffsE[materialEz, 2] * (Hx[nx, j, k] - Hx[nx, j - 1, k]) -
                             updatecoeffsE[materialEz, 1] * (2 * Hy[nx - 1, j, k]))


cpdef void update_symmetry_boundary_electric_y0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates the Ex and Ez face-interior components on the y0 PMC face."""
    cdef Py_ssize_t i, k
    cdef int materialEx, materialEz

    # Ex[0:nx, 0, 1:nz] - standard term Hy (dep k), ghost term Hz (dep j, doubled)
    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEx = ID[0, i, 0, k]
            Ex[i, 0, k] = (updatecoeffsE[materialEx, 0] * Ex[i, 0, k] -
                           updatecoeffsE[materialEx, 3] * (Hy[i, 0, k] - Hy[i, 0, k - 1]) +
                           updatecoeffsE[materialEx, 2] * (2 * Hz[i, 0, k]))

    # Ez[1:nx, 0, 0:nz] - standard term Hy (dep i), ghost term Hx (dep j, doubled)
    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, i, 0, k]
            Ez[i, 0, k] = (updatecoeffsE[materialEz, 0] * Ez[i, 0, k] +
                           updatecoeffsE[materialEz, 1] * (Hy[i, 0, k] - Hy[i - 1, 0, k]) -
                           updatecoeffsE[materialEz, 2] * (2 * Hx[i, 0, k]))


cpdef void update_symmetry_boundary_electric_ymax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates the Ex and Ez face-interior components on the ymax PMC face."""
    cdef Py_ssize_t i, k
    cdef int materialEx, materialEz

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEx = ID[0, i, ny, k]
            Ex[i, ny, k] = (updatecoeffsE[materialEx, 0] * Ex[i, ny, k] -
                             updatecoeffsE[materialEx, 3] * (Hy[i, ny, k] - Hy[i, ny, k - 1]) -
                             updatecoeffsE[materialEx, 2] * (2 * Hz[i, ny - 1, k]))

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, i, ny, k]
            Ez[i, ny, k] = (updatecoeffsE[materialEz, 0] * Ez[i, ny, k] +
                             updatecoeffsE[materialEz, 1] * (Hy[i, ny, k] - Hy[i - 1, ny, k]) +
                             updatecoeffsE[materialEz, 2] * (2 * Hx[i, ny - 1, k]))


cpdef void update_symmetry_boundary_electric_z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates the Ex and Ey face-interior components on the z0 PMC face."""
    cdef Py_ssize_t i, j
    cdef int materialEx, materialEy

    # Ex[0:nx, 1:ny, 0] - standard term Hz (dep j), ghost term Hy (dep k, doubled)
    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(1, ny):
            materialEx = ID[0, i, j, 0]
            Ex[i, j, 0] = (updatecoeffsE[materialEx, 0] * Ex[i, j, 0] +
                           updatecoeffsE[materialEx, 2] * (Hz[i, j, 0] - Hz[i, j - 1, 0]) -
                           updatecoeffsE[materialEx, 3] * (2 * Hy[i, j, 0]))

    # Ey[1:nx, 0:ny, 0] - standard term Hz (dep i), ghost term Hx (dep k, doubled)
    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(0, ny):
            materialEy = ID[1, i, j, 0]
            Ey[i, j, 0] = (updatecoeffsE[materialEy, 0] * Ey[i, j, 0] -
                           updatecoeffsE[materialEy, 1] * (Hz[i, j, 0] - Hz[i - 1, j, 0]) +
                           updatecoeffsE[materialEy, 3] * (2 * Hx[i, j, 0]))


cpdef void update_symmetry_boundary_electric_zmax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates the Ex and Ey face-interior components on the zmax PMC face."""
    cdef Py_ssize_t i, j
    cdef int materialEx, materialEy

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(1, ny):
            materialEx = ID[0, i, j, nz]
            Ex[i, j, nz] = (updatecoeffsE[materialEx, 0] * Ex[i, j, nz] +
                             updatecoeffsE[materialEx, 2] * (Hz[i, j, nz] - Hz[i, j - 1, nz]) +
                             updatecoeffsE[materialEx, 3] * (2 * Hy[i, j, nz - 1]))

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(0, ny):
            materialEy = ID[1, i, j, nz]
            Ey[i, j, nz] = (updatecoeffsE[materialEy, 0] * Ey[i, j, nz] -
                             updatecoeffsE[materialEy, 1] * (Hz[i, j, nz] - Hz[i - 1, j, nz]) -
                             updatecoeffsE[materialEy, 3] * (2 * Hx[i, j, nz - 1]))


# Per-iteration PMC ghost-node E update for the 12 domain edges (where two
# faces meet). Each edge carries exactly one tangential E component (never a
# 3-way corner case). Named <component>_<faceA>_<faceB>, faceA/faceB in
# canonical order (x0, y0, z0, xmax, ymax, zmax).
#
# Simplified per-edge scheme (no owner/increment split or execution-order
# requirement): the self term
# (Ca*E) is applied once if EITHER bordering face is PMC; each face then
# separately, additively contributes its own doubled ghost term only if
# THAT SPECIFIC face is PMC. A single-PMC-neighbour edge reduces correctly
# to zero on its own, because the edge's ID has already been forced to pec
# elsewhere whenever the other bordering face isn't PMC (Ca=Cb=0 there) -
# no conditional on "is the other face PMC" is needed inside the ghost-term
# additions themselves, only on whether THIS face is PMC.


cpdef void update_symmetry_boundary_electric_Ez_X0_Y0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint x0_pmc,
    bint y0_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy
):
    """Updates Ez along the x0-y0 edge (i=0, j=0, k free)."""
    cdef Py_ssize_t k
    cdef int mat

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, 0, 0, k]
        if x0_pmc or y0_pmc:
            Ez[0, 0, k] = updatecoeffsE[mat, 0] * Ez[0, 0, k]
        if x0_pmc:
            Ez[0, 0, k] = Ez[0, 0, k] + updatecoeffsE[mat, 1] * (2 * Hy[0, 0, k])
        if y0_pmc:
            Ez[0, 0, k] = Ez[0, 0, k] - updatecoeffsE[mat, 2] * (2 * Hx[0, 0, k])


cpdef void update_symmetry_boundary_electric_Ez_X0_YMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint x0_pmc,
    bint ymax_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy
):
    """Updates Ez along the x0-ymax edge (i=0, j=ny, k free)."""
    cdef Py_ssize_t k
    cdef int mat

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, 0, ny, k]
        if x0_pmc or ymax_pmc:
            Ez[0, ny, k] = updatecoeffsE[mat, 0] * Ez[0, ny, k]
        if x0_pmc:
            Ez[0, ny, k] = Ez[0, ny, k] + updatecoeffsE[mat, 1] * (2 * Hy[0, ny, k])
        if ymax_pmc:
            Ez[0, ny, k] = Ez[0, ny, k] + updatecoeffsE[mat, 2] * (2 * Hx[0, ny - 1, k])


cpdef void update_symmetry_boundary_electric_Ez_XMax_Y0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint xmax_pmc,
    bint y0_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy
):
    """Updates Ez along the xmax-y0 edge (i=nx, j=0, k free)."""
    cdef Py_ssize_t k
    cdef int mat

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, nx, 0, k]
        if xmax_pmc or y0_pmc:
            Ez[nx, 0, k] = updatecoeffsE[mat, 0] * Ez[nx, 0, k]
        if xmax_pmc:
            Ez[nx, 0, k] = Ez[nx, 0, k] - updatecoeffsE[mat, 1] * (2 * Hy[nx - 1, 0, k])
        if y0_pmc:
            Ez[nx, 0, k] = Ez[nx, 0, k] - updatecoeffsE[mat, 2] * (2 * Hx[nx, 0, k])


cpdef void update_symmetry_boundary_electric_Ez_XMax_YMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint xmax_pmc,
    bint ymax_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy
):
    """Updates Ez along the xmax-ymax edge (i=nx, j=ny, k free)."""
    cdef Py_ssize_t k
    cdef int mat

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, nx, ny, k]
        if xmax_pmc or ymax_pmc:
            Ez[nx, ny, k] = updatecoeffsE[mat, 0] * Ez[nx, ny, k]
        if xmax_pmc:
            Ez[nx, ny, k] = Ez[nx, ny, k] - updatecoeffsE[mat, 1] * (2 * Hy[nx - 1, ny, k])
        if ymax_pmc:
            Ez[nx, ny, k] = Ez[nx, ny, k] + updatecoeffsE[mat, 2] * (2 * Hx[nx, ny - 1, k])


cpdef void update_symmetry_boundary_electric_Ey_X0_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint x0_pmc,
    bint z0_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hz
):
    """Updates Ey along the x0-z0 edge (i=0, k=0, j free)."""
    cdef Py_ssize_t j
    cdef int mat

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, 0, j, 0]
        if x0_pmc or z0_pmc:
            Ey[0, j, 0] = updatecoeffsE[mat, 0] * Ey[0, j, 0]
        if x0_pmc:
            Ey[0, j, 0] = Ey[0, j, 0] - updatecoeffsE[mat, 1] * (2 * Hz[0, j, 0])
        if z0_pmc:
            Ey[0, j, 0] = Ey[0, j, 0] + updatecoeffsE[mat, 3] * (2 * Hx[0, j, 0])


cpdef void update_symmetry_boundary_electric_Ey_X0_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint x0_pmc,
    bint zmax_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hz
):
    """Updates Ey along the x0-zmax edge (i=0, k=nz, j free)."""
    cdef Py_ssize_t j
    cdef int mat

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, 0, j, nz]
        if x0_pmc or zmax_pmc:
            Ey[0, j, nz] = updatecoeffsE[mat, 0] * Ey[0, j, nz]
        if x0_pmc:
            Ey[0, j, nz] = Ey[0, j, nz] - updatecoeffsE[mat, 1] * (2 * Hz[0, j, nz])
        if zmax_pmc:
            Ey[0, j, nz] = Ey[0, j, nz] - updatecoeffsE[mat, 3] * (2 * Hx[0, j, nz - 1])


cpdef void update_symmetry_boundary_electric_Ey_XMax_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint xmax_pmc,
    bint z0_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hz
):
    """Updates Ey along the xmax-z0 edge (i=nx, k=0, j free)."""
    cdef Py_ssize_t j
    cdef int mat

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, nx, j, 0]
        if xmax_pmc or z0_pmc:
            Ey[nx, j, 0] = updatecoeffsE[mat, 0] * Ey[nx, j, 0]
        if xmax_pmc:
            Ey[nx, j, 0] = Ey[nx, j, 0] + updatecoeffsE[mat, 1] * (2 * Hz[nx - 1, j, 0])
        if z0_pmc:
            Ey[nx, j, 0] = Ey[nx, j, 0] + updatecoeffsE[mat, 3] * (2 * Hx[nx, j, 0])


cpdef void update_symmetry_boundary_electric_Ey_XMax_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint xmax_pmc,
    bint zmax_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hz
):
    """Updates Ey along the xmax-zmax edge (i=nx, k=nz, j free)."""
    cdef Py_ssize_t j
    cdef int mat

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, nx, j, nz]
        if xmax_pmc or zmax_pmc:
            Ey[nx, j, nz] = updatecoeffsE[mat, 0] * Ey[nx, j, nz]
        if xmax_pmc:
            Ey[nx, j, nz] = Ey[nx, j, nz] + updatecoeffsE[mat, 1] * (2 * Hz[nx - 1, j, nz])
        if zmax_pmc:
            Ey[nx, j, nz] = Ey[nx, j, nz] - updatecoeffsE[mat, 3] * (2 * Hx[nx, j, nz - 1])


cpdef void update_symmetry_boundary_electric_Ex_Y0_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint y0_pmc,
    bint z0_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates Ex along the y0-z0 edge (j=0, k=0, i free)."""
    cdef Py_ssize_t i
    cdef int mat

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, 0, 0]
        if y0_pmc or z0_pmc:
            Ex[i, 0, 0] = updatecoeffsE[mat, 0] * Ex[i, 0, 0]
        if y0_pmc:
            Ex[i, 0, 0] = Ex[i, 0, 0] + updatecoeffsE[mat, 2] * (2 * Hz[i, 0, 0])
        if z0_pmc:
            Ex[i, 0, 0] = Ex[i, 0, 0] - updatecoeffsE[mat, 3] * (2 * Hy[i, 0, 0])


cpdef void update_symmetry_boundary_electric_Ex_Y0_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint y0_pmc,
    bint zmax_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates Ex along the y0-zmax edge (j=0, k=nz, i free)."""
    cdef Py_ssize_t i
    cdef int mat

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, 0, nz]
        if y0_pmc or zmax_pmc:
            Ex[i, 0, nz] = updatecoeffsE[mat, 0] * Ex[i, 0, nz]
        if y0_pmc:
            Ex[i, 0, nz] = Ex[i, 0, nz] + updatecoeffsE[mat, 2] * (2 * Hz[i, 0, nz])
        if zmax_pmc:
            Ex[i, 0, nz] = Ex[i, 0, nz] + updatecoeffsE[mat, 3] * (2 * Hy[i, 0, nz - 1])


cpdef void update_symmetry_boundary_electric_Ex_YMax_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint ymax_pmc,
    bint z0_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates Ex along the ymax-z0 edge (j=ny, k=0, i free)."""
    cdef Py_ssize_t i
    cdef int mat

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, ny, 0]
        if ymax_pmc or z0_pmc:
            Ex[i, ny, 0] = updatecoeffsE[mat, 0] * Ex[i, ny, 0]
        if ymax_pmc:
            Ex[i, ny, 0] = Ex[i, ny, 0] - updatecoeffsE[mat, 2] * (2 * Hz[i, ny - 1, 0])
        if z0_pmc:
            Ex[i, ny, 0] = Ex[i, ny, 0] - updatecoeffsE[mat, 3] * (2 * Hy[i, ny, 0])


cpdef void update_symmetry_boundary_electric_Ex_YMax_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint ymax_pmc,
    bint zmax_pmc,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates Ex along the ymax-zmax edge (j=ny, k=nz, i free)."""
    cdef Py_ssize_t i
    cdef int mat

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, ny, nz]
        if ymax_pmc or zmax_pmc:
            Ex[i, ny, nz] = updatecoeffsE[mat, 0] * Ex[i, ny, nz]
        if ymax_pmc:
            Ex[i, ny, nz] = Ex[i, ny, nz] - updatecoeffsE[mat, 2] * (2 * Hz[i, ny - 1, nz])
        if zmax_pmc:
            Ex[i, ny, nz] = Ex[i, ny, nz] + updatecoeffsE[mat, 3] * (2 * Hy[i, ny, nz - 1])
