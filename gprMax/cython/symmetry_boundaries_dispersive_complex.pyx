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
from gprMax.config cimport float_or_double_complex

# Complex-pole dispersive (Lorentz/Drude) counterpart of
# gprMax/cython/symmetry_boundaries.pyx's PMC ghost-node E update and of
# symmetry_boundaries_dispersive.pyx's real-pole (Debye) counterpart - see
# those files for the ghost-node derivation and the ADE phi/T-array
# bookkeeping this file adds, both unchanged here. The only difference from
# the real-pole file is the phi accumulation: Tx/Ty/Tz and
# updatecoeffsdispersive are complex (float_or_double_complex) here, so phi
# (which must stay real - it feeds directly into a real E-field update)
# accumulates Real(updatecoeffsdispersive[...] * T[...]) per pole, exactly
# matching fields_updates_dispersive_template.jinja's complex branch. The
# imaginary cross term is essential for Lorentz poles: taking the real part
# of each factor separately changes the implemented susceptibility and gives
# the wrong phase and loss around a resonance.
# The T-array recursion itself (Tx[pole,i,j,k] = beta*Tx_old + gamma*E_old)
# and Phase B's correction are unchanged in form from the real-pole file -
# ordinary complex arithmetic (complex*complex, complex*real, complex-complex)
# handles them directly, with no real-part extraction needed there.
#
# Two phases, mirroring the bulk kernel's own A/B split (see
# fields_updates_dispersive_template.jinja and gprMax/updates/cpu_updates.py):
#   Phase A ("_dispersive_"): called at the same point as the existing
#     non-dispersive boundary update (right after update_electric_a(), before
#     PML/sources). Accumulates phi from the OLD T, updates T from the OLD E,
#     then assembles the new E from the ghost-doubled curl term minus
#     updatecoeffsE[mat, 4]*phi - the same per-cell logic as the bulk kernel's
#     Phase A, applied at wall/edge positions the bulk kernel's own loop
#     bounds structurally exclude.
#   Phase B ("_dispersive_b_"): called at the same point as the bulk kernel's
#     own Phase B (right before update_electric_b(), i.e. after PML/sources
#     have possibly further modified E - a source sitting at/adjacent to the
#     wall, as in a center-fed dipole straddling a PMC plane, is exactly this
#     case). Corrects T using the now-final E. No H arrays needed, matching
#     the bulk Phase B's own signature.
#
# maxpoles is always looped unconditionally (no 1-pole specialisation, unlike
# the bulk kernel's separate "1pole" functions) - boundary cells are a
# vanishing fraction of total domain cells, so the loop-unrolling
# optimisation that matters at bulk (O(n^3)) scale is immaterial here.
#
# Edges: no explicit PEC-transparency branch is needed for the phi/T block,
# unlike you might expect - it is computed unconditionally (not gated by
# a_pmc/b_pmc) for the same reason the non-dispersive edge kernels don't
# need one for their self term: when only one bordering face is PMC, the
# other face has already forced this edge's material to "pec" at build time
# (FDTDGrid._terminate_pmls_with_pec/_force_pec_tangential_e), and a pec
# material's updatecoeffsdispersive row is never written (stays zero,
# since it's a plain Material, not a DispersiveMaterial) - phi and the T
# recursion both correctly evaluate to zero for free. When both bordering
# faces are PMC, the phi/T machinery runs for real, correctly, on whatever
# genuinely dispersive material occupies the corner.


###############################################
# Phase A - face-interior, one function/face #
###############################################


cpdef void update_symmetry_boundary_electric_dispersive_x0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_x0()."""
    cdef Py_ssize_t j, k, pole
    cdef int materialEy, materialEz
    cdef float_or_double phi

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEy = ID[1, 0, j, k]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEy, pole * 3] * Ty[pole, 0, j, k]).real
                Ty[pole, 0, j, k] = (updatecoeffsdispersive[materialEy, 1 + (pole * 3)] * Ty[pole, 0, j, k]
                                     + updatecoeffsdispersive[materialEy, 2 + (pole * 3)] * Ey[0, j, k])
            Ey[0, j, k] = (updatecoeffsE[materialEy, 0] * Ey[0, j, k] +
                           updatecoeffsE[materialEy, 3] * (Hx[0, j, k] - Hx[0, j, k - 1]) -
                           updatecoeffsE[materialEy, 1] * (2 * Hz[0, j, k]) -
                           updatecoeffsE[materialEy, 4] * phi)

    for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, 0, j, k]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEz, pole * 3] * Tz[pole, 0, j, k]).real
                Tz[pole, 0, j, k] = (updatecoeffsdispersive[materialEz, 1 + (pole * 3)] * Tz[pole, 0, j, k]
                                     + updatecoeffsdispersive[materialEz, 2 + (pole * 3)] * Ez[0, j, k])
            Ez[0, j, k] = (updatecoeffsE[materialEz, 0] * Ez[0, j, k] -
                           updatecoeffsE[materialEz, 2] * (Hx[0, j, k] - Hx[0, j - 1, k]) +
                           updatecoeffsE[materialEz, 1] * (2 * Hy[0, j, k]) -
                           updatecoeffsE[materialEz, 4] * phi)


cpdef void update_symmetry_boundary_electric_dispersive_xmax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_xmax()."""
    cdef Py_ssize_t j, k, pole
    cdef int materialEy, materialEz
    cdef float_or_double phi

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEy = ID[1, nx, j, k]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEy, pole * 3] * Ty[pole, nx, j, k]).real
                Ty[pole, nx, j, k] = (updatecoeffsdispersive[materialEy, 1 + (pole * 3)] * Ty[pole, nx, j, k]
                                      + updatecoeffsdispersive[materialEy, 2 + (pole * 3)] * Ey[nx, j, k])
            Ey[nx, j, k] = (updatecoeffsE[materialEy, 0] * Ey[nx, j, k] +
                             updatecoeffsE[materialEy, 3] * (Hx[nx, j, k] - Hx[nx, j, k - 1]) +
                             updatecoeffsE[materialEy, 1] * (2 * Hz[nx - 1, j, k]) -
                             updatecoeffsE[materialEy, 4] * phi)

    for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, nx, j, k]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEz, pole * 3] * Tz[pole, nx, j, k]).real
                Tz[pole, nx, j, k] = (updatecoeffsdispersive[materialEz, 1 + (pole * 3)] * Tz[pole, nx, j, k]
                                      + updatecoeffsdispersive[materialEz, 2 + (pole * 3)] * Ez[nx, j, k])
            Ez[nx, j, k] = (updatecoeffsE[materialEz, 0] * Ez[nx, j, k] -
                             updatecoeffsE[materialEz, 2] * (Hx[nx, j, k] - Hx[nx, j - 1, k]) -
                             updatecoeffsE[materialEz, 1] * (2 * Hy[nx - 1, j, k]) -
                             updatecoeffsE[materialEz, 4] * phi)


cpdef void update_symmetry_boundary_electric_dispersive_y0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_y0()."""
    cdef Py_ssize_t i, k, pole
    cdef int materialEx, materialEz
    cdef float_or_double phi

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEx = ID[0, i, 0, k]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEx, pole * 3] * Tx[pole, i, 0, k]).real
                Tx[pole, i, 0, k] = (updatecoeffsdispersive[materialEx, 1 + (pole * 3)] * Tx[pole, i, 0, k]
                                     + updatecoeffsdispersive[materialEx, 2 + (pole * 3)] * Ex[i, 0, k])
            Ex[i, 0, k] = (updatecoeffsE[materialEx, 0] * Ex[i, 0, k] -
                           updatecoeffsE[materialEx, 3] * (Hy[i, 0, k] - Hy[i, 0, k - 1]) +
                           updatecoeffsE[materialEx, 2] * (2 * Hz[i, 0, k]) -
                           updatecoeffsE[materialEx, 4] * phi)

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, i, 0, k]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEz, pole * 3] * Tz[pole, i, 0, k]).real
                Tz[pole, i, 0, k] = (updatecoeffsdispersive[materialEz, 1 + (pole * 3)] * Tz[pole, i, 0, k]
                                     + updatecoeffsdispersive[materialEz, 2 + (pole * 3)] * Ez[i, 0, k])
            Ez[i, 0, k] = (updatecoeffsE[materialEz, 0] * Ez[i, 0, k] +
                           updatecoeffsE[materialEz, 1] * (Hy[i, 0, k] - Hy[i - 1, 0, k]) -
                           updatecoeffsE[materialEz, 2] * (2 * Hx[i, 0, k]) -
                           updatecoeffsE[materialEz, 4] * phi)


cpdef void update_symmetry_boundary_electric_dispersive_ymax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_ymax()."""
    cdef Py_ssize_t i, k, pole
    cdef int materialEx, materialEz
    cdef float_or_double phi

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEx = ID[0, i, ny, k]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEx, pole * 3] * Tx[pole, i, ny, k]).real
                Tx[pole, i, ny, k] = (updatecoeffsdispersive[materialEx, 1 + (pole * 3)] * Tx[pole, i, ny, k]
                                      + updatecoeffsdispersive[materialEx, 2 + (pole * 3)] * Ex[i, ny, k])
            Ex[i, ny, k] = (updatecoeffsE[materialEx, 0] * Ex[i, ny, k] -
                             updatecoeffsE[materialEx, 3] * (Hy[i, ny, k] - Hy[i, ny, k - 1]) -
                             updatecoeffsE[materialEx, 2] * (2 * Hz[i, ny - 1, k]) -
                             updatecoeffsE[materialEx, 4] * phi)

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, i, ny, k]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEz, pole * 3] * Tz[pole, i, ny, k]).real
                Tz[pole, i, ny, k] = (updatecoeffsdispersive[materialEz, 1 + (pole * 3)] * Tz[pole, i, ny, k]
                                      + updatecoeffsdispersive[materialEz, 2 + (pole * 3)] * Ez[i, ny, k])
            Ez[i, ny, k] = (updatecoeffsE[materialEz, 0] * Ez[i, ny, k] +
                             updatecoeffsE[materialEz, 1] * (Hy[i, ny, k] - Hy[i - 1, ny, k]) +
                             updatecoeffsE[materialEz, 2] * (2 * Hx[i, ny - 1, k]) -
                             updatecoeffsE[materialEz, 4] * phi)


cpdef void update_symmetry_boundary_electric_dispersive_z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_z0()."""
    cdef Py_ssize_t i, j, pole
    cdef int materialEx, materialEy
    cdef float_or_double phi

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(1, ny):
            materialEx = ID[0, i, j, 0]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEx, pole * 3] * Tx[pole, i, j, 0]).real
                Tx[pole, i, j, 0] = (updatecoeffsdispersive[materialEx, 1 + (pole * 3)] * Tx[pole, i, j, 0]
                                     + updatecoeffsdispersive[materialEx, 2 + (pole * 3)] * Ex[i, j, 0])
            Ex[i, j, 0] = (updatecoeffsE[materialEx, 0] * Ex[i, j, 0] +
                           updatecoeffsE[materialEx, 2] * (Hz[i, j, 0] - Hz[i, j - 1, 0]) -
                           updatecoeffsE[materialEx, 3] * (2 * Hy[i, j, 0]) -
                           updatecoeffsE[materialEx, 4] * phi)

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(0, ny):
            materialEy = ID[1, i, j, 0]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEy, pole * 3] * Ty[pole, i, j, 0]).real
                Ty[pole, i, j, 0] = (updatecoeffsdispersive[materialEy, 1 + (pole * 3)] * Ty[pole, i, j, 0]
                                     + updatecoeffsdispersive[materialEy, 2 + (pole * 3)] * Ey[i, j, 0])
            Ey[i, j, 0] = (updatecoeffsE[materialEy, 0] * Ey[i, j, 0] -
                           updatecoeffsE[materialEy, 1] * (Hz[i, j, 0] - Hz[i - 1, j, 0]) +
                           updatecoeffsE[materialEy, 3] * (2 * Hx[i, j, 0]) -
                           updatecoeffsE[materialEy, 4] * phi)


cpdef void update_symmetry_boundary_electric_dispersive_zmax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_zmax()."""
    cdef Py_ssize_t i, j, pole
    cdef int materialEx, materialEy
    cdef float_or_double phi

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(1, ny):
            materialEx = ID[0, i, j, nz]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEx, pole * 3] * Tx[pole, i, j, nz]).real
                Tx[pole, i, j, nz] = (updatecoeffsdispersive[materialEx, 1 + (pole * 3)] * Tx[pole, i, j, nz]
                                      + updatecoeffsdispersive[materialEx, 2 + (pole * 3)] * Ex[i, j, nz])
            Ex[i, j, nz] = (updatecoeffsE[materialEx, 0] * Ex[i, j, nz] +
                             updatecoeffsE[materialEx, 2] * (Hz[i, j, nz] - Hz[i, j - 1, nz]) +
                             updatecoeffsE[materialEx, 3] * (2 * Hy[i, j, nz - 1]) -
                             updatecoeffsE[materialEx, 4] * phi)

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(0, ny):
            materialEy = ID[1, i, j, nz]
            phi = 0
            for pole in range(maxpoles):
                phi = phi + (updatecoeffsdispersive[materialEy, pole * 3] * Ty[pole, i, j, nz]).real
                Ty[pole, i, j, nz] = (updatecoeffsdispersive[materialEy, 1 + (pole * 3)] * Ty[pole, i, j, nz]
                                      + updatecoeffsdispersive[materialEy, 2 + (pole * 3)] * Ey[i, j, nz])
            Ey[i, j, nz] = (updatecoeffsE[materialEy, 0] * Ey[i, j, nz] -
                             updatecoeffsE[materialEy, 1] * (Hz[i, j, nz] - Hz[i - 1, j, nz]) -
                             updatecoeffsE[materialEy, 3] * (2 * Hx[i, j, nz - 1]) -
                             updatecoeffsE[materialEy, 4] * phi)


######################################################
# Phase A - domain edges, one function/edge (12 total) #
######################################################


cpdef void update_symmetry_boundary_electric_dispersive_Ez_X0_Y0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint x0_pmc,
    bint y0_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ez_X0_Y0()."""
    cdef Py_ssize_t k, pole
    cdef int mat
    cdef float_or_double phi

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, 0, 0, k]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Tz[pole, 0, 0, k]).real
            Tz[pole, 0, 0, k] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Tz[pole, 0, 0, k]
                                 + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ez[0, 0, k])
        if x0_pmc or y0_pmc:
            Ez[0, 0, k] = updatecoeffsE[mat, 0] * Ez[0, 0, k] - updatecoeffsE[mat, 4] * phi
        if x0_pmc:
            Ez[0, 0, k] = Ez[0, 0, k] + updatecoeffsE[mat, 1] * (2 * Hy[0, 0, k])
        if y0_pmc:
            Ez[0, 0, k] = Ez[0, 0, k] - updatecoeffsE[mat, 2] * (2 * Hx[0, 0, k])


cpdef void update_symmetry_boundary_electric_dispersive_Ez_X0_YMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint x0_pmc,
    bint ymax_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ez_X0_YMax()."""
    cdef Py_ssize_t k, pole
    cdef int mat
    cdef float_or_double phi

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, 0, ny, k]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Tz[pole, 0, ny, k]).real
            Tz[pole, 0, ny, k] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Tz[pole, 0, ny, k]
                                  + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ez[0, ny, k])
        if x0_pmc or ymax_pmc:
            Ez[0, ny, k] = updatecoeffsE[mat, 0] * Ez[0, ny, k] - updatecoeffsE[mat, 4] * phi
        if x0_pmc:
            Ez[0, ny, k] = Ez[0, ny, k] + updatecoeffsE[mat, 1] * (2 * Hy[0, ny, k])
        if ymax_pmc:
            Ez[0, ny, k] = Ez[0, ny, k] + updatecoeffsE[mat, 2] * (2 * Hx[0, ny - 1, k])


cpdef void update_symmetry_boundary_electric_dispersive_Ez_XMax_Y0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint xmax_pmc,
    bint y0_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ez_XMax_Y0()."""
    cdef Py_ssize_t k, pole
    cdef int mat
    cdef float_or_double phi

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, nx, 0, k]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Tz[pole, nx, 0, k]).real
            Tz[pole, nx, 0, k] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Tz[pole, nx, 0, k]
                                  + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ez[nx, 0, k])
        if xmax_pmc or y0_pmc:
            Ez[nx, 0, k] = updatecoeffsE[mat, 0] * Ez[nx, 0, k] - updatecoeffsE[mat, 4] * phi
        if xmax_pmc:
            Ez[nx, 0, k] = Ez[nx, 0, k] - updatecoeffsE[mat, 1] * (2 * Hy[nx - 1, 0, k])
        if y0_pmc:
            Ez[nx, 0, k] = Ez[nx, 0, k] - updatecoeffsE[mat, 2] * (2 * Hx[nx, 0, k])


cpdef void update_symmetry_boundary_electric_dispersive_Ez_XMax_YMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint xmax_pmc,
    bint ymax_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ez_XMax_YMax()."""
    cdef Py_ssize_t k, pole
    cdef int mat
    cdef float_or_double phi

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, nx, ny, k]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Tz[pole, nx, ny, k]).real
            Tz[pole, nx, ny, k] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Tz[pole, nx, ny, k]
                                   + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ez[nx, ny, k])
        if xmax_pmc or ymax_pmc:
            Ez[nx, ny, k] = updatecoeffsE[mat, 0] * Ez[nx, ny, k] - updatecoeffsE[mat, 4] * phi
        if xmax_pmc:
            Ez[nx, ny, k] = Ez[nx, ny, k] - updatecoeffsE[mat, 1] * (2 * Hy[nx - 1, ny, k])
        if ymax_pmc:
            Ez[nx, ny, k] = Ez[nx, ny, k] + updatecoeffsE[mat, 2] * (2 * Hx[nx, ny - 1, k])


cpdef void update_symmetry_boundary_electric_dispersive_Ey_X0_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint x0_pmc,
    bint z0_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ey_X0_Z0()."""
    cdef Py_ssize_t j, pole
    cdef int mat
    cdef float_or_double phi

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, 0, j, 0]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Ty[pole, 0, j, 0]).real
            Ty[pole, 0, j, 0] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Ty[pole, 0, j, 0]
                                 + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ey[0, j, 0])
        if x0_pmc or z0_pmc:
            Ey[0, j, 0] = updatecoeffsE[mat, 0] * Ey[0, j, 0] - updatecoeffsE[mat, 4] * phi
        if x0_pmc:
            Ey[0, j, 0] = Ey[0, j, 0] - updatecoeffsE[mat, 1] * (2 * Hz[0, j, 0])
        if z0_pmc:
            Ey[0, j, 0] = Ey[0, j, 0] + updatecoeffsE[mat, 3] * (2 * Hx[0, j, 0])


cpdef void update_symmetry_boundary_electric_dispersive_Ey_X0_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint x0_pmc,
    bint zmax_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ey_X0_ZMax()."""
    cdef Py_ssize_t j, pole
    cdef int mat
    cdef float_or_double phi

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, 0, j, nz]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Ty[pole, 0, j, nz]).real
            Ty[pole, 0, j, nz] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Ty[pole, 0, j, nz]
                                  + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ey[0, j, nz])
        if x0_pmc or zmax_pmc:
            Ey[0, j, nz] = updatecoeffsE[mat, 0] * Ey[0, j, nz] - updatecoeffsE[mat, 4] * phi
        if x0_pmc:
            Ey[0, j, nz] = Ey[0, j, nz] - updatecoeffsE[mat, 1] * (2 * Hz[0, j, nz])
        if zmax_pmc:
            Ey[0, j, nz] = Ey[0, j, nz] - updatecoeffsE[mat, 3] * (2 * Hx[0, j, nz - 1])


cpdef void update_symmetry_boundary_electric_dispersive_Ey_XMax_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint xmax_pmc,
    bint z0_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ey_XMax_Z0()."""
    cdef Py_ssize_t j, pole
    cdef int mat
    cdef float_or_double phi

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, nx, j, 0]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Ty[pole, nx, j, 0]).real
            Ty[pole, nx, j, 0] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Ty[pole, nx, j, 0]
                                  + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ey[nx, j, 0])
        if xmax_pmc or z0_pmc:
            Ey[nx, j, 0] = updatecoeffsE[mat, 0] * Ey[nx, j, 0] - updatecoeffsE[mat, 4] * phi
        if xmax_pmc:
            Ey[nx, j, 0] = Ey[nx, j, 0] + updatecoeffsE[mat, 1] * (2 * Hz[nx - 1, j, 0])
        if z0_pmc:
            Ey[nx, j, 0] = Ey[nx, j, 0] + updatecoeffsE[mat, 3] * (2 * Hx[nx, j, 0])


cpdef void update_symmetry_boundary_electric_dispersive_Ey_XMax_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint xmax_pmc,
    bint zmax_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ey_XMax_ZMax()."""
    cdef Py_ssize_t j, pole
    cdef int mat
    cdef float_or_double phi

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, nx, j, nz]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Ty[pole, nx, j, nz]).real
            Ty[pole, nx, j, nz] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Ty[pole, nx, j, nz]
                                   + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ey[nx, j, nz])
        if xmax_pmc or zmax_pmc:
            Ey[nx, j, nz] = updatecoeffsE[mat, 0] * Ey[nx, j, nz] - updatecoeffsE[mat, 4] * phi
        if xmax_pmc:
            Ey[nx, j, nz] = Ey[nx, j, nz] + updatecoeffsE[mat, 1] * (2 * Hz[nx - 1, j, nz])
        if zmax_pmc:
            Ey[nx, j, nz] = Ey[nx, j, nz] - updatecoeffsE[mat, 3] * (2 * Hx[nx, j, nz - 1])


cpdef void update_symmetry_boundary_electric_dispersive_Ex_Y0_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint y0_pmc,
    bint z0_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ex_Y0_Z0()."""
    cdef Py_ssize_t i, pole
    cdef int mat
    cdef float_or_double phi

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, 0, 0]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Tx[pole, i, 0, 0]).real
            Tx[pole, i, 0, 0] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Tx[pole, i, 0, 0]
                                 + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ex[i, 0, 0])
        if y0_pmc or z0_pmc:
            Ex[i, 0, 0] = updatecoeffsE[mat, 0] * Ex[i, 0, 0] - updatecoeffsE[mat, 4] * phi
        if y0_pmc:
            Ex[i, 0, 0] = Ex[i, 0, 0] + updatecoeffsE[mat, 2] * (2 * Hz[i, 0, 0])
        if z0_pmc:
            Ex[i, 0, 0] = Ex[i, 0, 0] - updatecoeffsE[mat, 3] * (2 * Hy[i, 0, 0])


cpdef void update_symmetry_boundary_electric_dispersive_Ex_Y0_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint y0_pmc,
    bint zmax_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ex_Y0_ZMax()."""
    cdef Py_ssize_t i, pole
    cdef int mat
    cdef float_or_double phi

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, 0, nz]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Tx[pole, i, 0, nz]).real
            Tx[pole, i, 0, nz] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Tx[pole, i, 0, nz]
                                  + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ex[i, 0, nz])
        if y0_pmc or zmax_pmc:
            Ex[i, 0, nz] = updatecoeffsE[mat, 0] * Ex[i, 0, nz] - updatecoeffsE[mat, 4] * phi
        if y0_pmc:
            Ex[i, 0, nz] = Ex[i, 0, nz] + updatecoeffsE[mat, 2] * (2 * Hz[i, 0, nz])
        if zmax_pmc:
            Ex[i, 0, nz] = Ex[i, 0, nz] + updatecoeffsE[mat, 3] * (2 * Hy[i, 0, nz - 1])


cpdef void update_symmetry_boundary_electric_dispersive_Ex_YMax_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint ymax_pmc,
    bint z0_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ex_YMax_Z0()."""
    cdef Py_ssize_t i, pole
    cdef int mat
    cdef float_or_double phi

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, ny, 0]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Tx[pole, i, ny, 0]).real
            Tx[pole, i, ny, 0] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Tx[pole, i, ny, 0]
                                  + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ex[i, ny, 0])
        if ymax_pmc or z0_pmc:
            Ex[i, ny, 0] = updatecoeffsE[mat, 0] * Ex[i, ny, 0] - updatecoeffsE[mat, 4] * phi
        if ymax_pmc:
            Ex[i, ny, 0] = Ex[i, ny, 0] - updatecoeffsE[mat, 2] * (2 * Hz[i, ny - 1, 0])
        if z0_pmc:
            Ex[i, ny, 0] = Ex[i, ny, 0] - updatecoeffsE[mat, 3] * (2 * Hy[i, ny, 0])


cpdef void update_symmetry_boundary_electric_dispersive_Ex_YMax_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint ymax_pmc,
    bint zmax_pmc,
    int maxpoles,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Dispersive counterpart of update_symmetry_boundary_electric_Ex_YMax_ZMax()."""
    cdef Py_ssize_t i, pole
    cdef int mat
    cdef float_or_double phi

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, ny, nz]
        phi = 0
        for pole in range(maxpoles):
            phi = phi + (updatecoeffsdispersive[mat, pole * 3] * Tx[pole, i, ny, nz]).real
            Tx[pole, i, ny, nz] = (updatecoeffsdispersive[mat, 1 + (pole * 3)] * Tx[pole, i, ny, nz]
                                   + updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ex[i, ny, nz])
        if ymax_pmc or zmax_pmc:
            Ex[i, ny, nz] = updatecoeffsE[mat, 0] * Ex[i, ny, nz] - updatecoeffsE[mat, 4] * phi
        if ymax_pmc:
            Ex[i, ny, nz] = Ex[i, ny, nz] - updatecoeffsE[mat, 2] * (2 * Hz[i, ny - 1, nz])
        if zmax_pmc:
            Ex[i, ny, nz] = Ex[i, ny, nz] + updatecoeffsE[mat, 3] * (2 * Hy[i, ny, nz - 1])


###########################################################
# Phase B - face-interior, T-array correction only (6 total) #
###########################################################


cpdef void update_symmetry_boundary_electric_dispersive_b_x0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B (post-PML/source T-array correction) counterpart of
    update_symmetry_boundary_electric_dispersive_x0()."""
    cdef Py_ssize_t j, k, pole
    cdef int materialEy, materialEz

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEy = ID[1, 0, j, k]
            for pole in range(maxpoles):
                Ty[pole, 0, j, k] = (Ty[pole, 0, j, k]
                                     - updatecoeffsdispersive[materialEy, 2 + (pole * 3)] * Ey[0, j, k])

    for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, 0, j, k]
            for pole in range(maxpoles):
                Tz[pole, 0, j, k] = (Tz[pole, 0, j, k]
                                     - updatecoeffsdispersive[materialEz, 2 + (pole * 3)] * Ez[0, j, k])


cpdef void update_symmetry_boundary_electric_dispersive_b_xmax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_xmax()."""
    cdef Py_ssize_t j, k, pole
    cdef int materialEy, materialEz

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEy = ID[1, nx, j, k]
            for pole in range(maxpoles):
                Ty[pole, nx, j, k] = (Ty[pole, nx, j, k]
                                      - updatecoeffsdispersive[materialEy, 2 + (pole * 3)] * Ey[nx, j, k])

    for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, nx, j, k]
            for pole in range(maxpoles):
                Tz[pole, nx, j, k] = (Tz[pole, nx, j, k]
                                      - updatecoeffsdispersive[materialEz, 2 + (pole * 3)] * Ez[nx, j, k])


cpdef void update_symmetry_boundary_electric_dispersive_b_y0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_y0()."""
    cdef Py_ssize_t i, k, pole
    cdef int materialEx, materialEz

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEx = ID[0, i, 0, k]
            for pole in range(maxpoles):
                Tx[pole, i, 0, k] = (Tx[pole, i, 0, k]
                                     - updatecoeffsdispersive[materialEx, 2 + (pole * 3)] * Ex[i, 0, k])

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, i, 0, k]
            for pole in range(maxpoles):
                Tz[pole, i, 0, k] = (Tz[pole, i, 0, k]
                                     - updatecoeffsdispersive[materialEz, 2 + (pole * 3)] * Ez[i, 0, k])


cpdef void update_symmetry_boundary_electric_dispersive_b_ymax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_ymax()."""
    cdef Py_ssize_t i, k, pole
    cdef int materialEx, materialEz

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(1, nz):
            materialEx = ID[0, i, ny, k]
            for pole in range(maxpoles):
                Tx[pole, i, ny, k] = (Tx[pole, i, ny, k]
                                      - updatecoeffsdispersive[materialEx, 2 + (pole * 3)] * Ex[i, ny, k])

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            materialEz = ID[2, i, ny, k]
            for pole in range(maxpoles):
                Tz[pole, i, ny, k] = (Tz[pole, i, ny, k]
                                      - updatecoeffsdispersive[materialEz, 2 + (pole * 3)] * Ez[i, ny, k])


cpdef void update_symmetry_boundary_electric_dispersive_b_z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_z0()."""
    cdef Py_ssize_t i, j, pole
    cdef int materialEx, materialEy

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(1, ny):
            materialEx = ID[0, i, j, 0]
            for pole in range(maxpoles):
                Tx[pole, i, j, 0] = (Tx[pole, i, j, 0]
                                     - updatecoeffsdispersive[materialEx, 2 + (pole * 3)] * Ex[i, j, 0])

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(0, ny):
            materialEy = ID[1, i, j, 0]
            for pole in range(maxpoles):
                Ty[pole, i, j, 0] = (Ty[pole, i, j, 0]
                                     - updatecoeffsdispersive[materialEy, 2 + (pole * 3)] * Ey[i, j, 0])


cpdef void update_symmetry_boundary_electric_dispersive_b_zmax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_zmax()."""
    cdef Py_ssize_t i, j, pole
    cdef int materialEx, materialEy

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(1, ny):
            materialEx = ID[0, i, j, nz]
            for pole in range(maxpoles):
                Tx[pole, i, j, nz] = (Tx[pole, i, j, nz]
                                      - updatecoeffsdispersive[materialEx, 2 + (pole * 3)] * Ex[i, j, nz])

    for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(0, ny):
            materialEy = ID[1, i, j, nz]
            for pole in range(maxpoles):
                Ty[pole, i, j, nz] = (Ty[pole, i, j, nz]
                                      - updatecoeffsdispersive[materialEy, 2 + (pole * 3)] * Ey[i, j, nz])


##############################################################
# Phase B - domain edges, T-array correction only (12 total) #
##############################################################


cpdef void update_symmetry_boundary_electric_dispersive_b_Ez_X0_Y0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ez_X0_Y0()."""
    cdef Py_ssize_t k, pole
    cdef int mat

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, 0, 0, k]
        for pole in range(maxpoles):
            Tz[pole, 0, 0, k] = Tz[pole, 0, 0, k] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ez[0, 0, k]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ez_X0_YMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ez_X0_YMax()."""
    cdef Py_ssize_t k, pole
    cdef int mat

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, 0, ny, k]
        for pole in range(maxpoles):
            Tz[pole, 0, ny, k] = Tz[pole, 0, ny, k] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ez[0, ny, k]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ez_XMax_Y0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ez_XMax_Y0()."""
    cdef Py_ssize_t k, pole
    cdef int mat

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, nx, 0, k]
        for pole in range(maxpoles):
            Tz[pole, nx, 0, k] = Tz[pole, nx, 0, k] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ez[nx, 0, k]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ez_XMax_YMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tz,
    float_or_double[:, :, ::1] Ez
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ez_XMax_YMax()."""
    cdef Py_ssize_t k, pole
    cdef int mat

    for k in prange(0, nz, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[2, nx, ny, k]
        for pole in range(maxpoles):
            Tz[pole, nx, ny, k] = Tz[pole, nx, ny, k] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ez[nx, ny, k]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ey_X0_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double[:, :, ::1] Ey
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ey_X0_Z0()."""
    cdef Py_ssize_t j, pole
    cdef int mat

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, 0, j, 0]
        for pole in range(maxpoles):
            Ty[pole, 0, j, 0] = Ty[pole, 0, j, 0] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ey[0, j, 0]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ey_X0_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double[:, :, ::1] Ey
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ey_X0_ZMax()."""
    cdef Py_ssize_t j, pole
    cdef int mat

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, 0, j, nz]
        for pole in range(maxpoles):
            Ty[pole, 0, j, nz] = Ty[pole, 0, j, nz] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ey[0, j, nz]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ey_XMax_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double[:, :, ::1] Ey
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ey_XMax_Z0()."""
    cdef Py_ssize_t j, pole
    cdef int mat

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, nx, j, 0]
        for pole in range(maxpoles):
            Ty[pole, nx, j, 0] = Ty[pole, nx, j, 0] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ey[nx, j, 0]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ey_XMax_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Ty,
    float_or_double[:, :, ::1] Ey
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ey_XMax_ZMax()."""
    cdef Py_ssize_t j, pole
    cdef int mat

    for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[1, nx, j, nz]
        for pole in range(maxpoles):
            Ty[pole, nx, j, nz] = Ty[pole, nx, j, nz] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ey[nx, j, nz]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ex_Y0_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double[:, :, ::1] Ex
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ex_Y0_Z0()."""
    cdef Py_ssize_t i, pole
    cdef int mat

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, 0, 0]
        for pole in range(maxpoles):
            Tx[pole, i, 0, 0] = Tx[pole, i, 0, 0] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ex[i, 0, 0]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ex_Y0_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double[:, :, ::1] Ex
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ex_Y0_ZMax()."""
    cdef Py_ssize_t i, pole
    cdef int mat

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, 0, nz]
        for pole in range(maxpoles):
            Tx[pole, i, 0, nz] = Tx[pole, i, 0, nz] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ex[i, 0, nz]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ex_YMax_Z0(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double[:, :, ::1] Ex
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ex_YMax_Z0()."""
    cdef Py_ssize_t i, pole
    cdef int mat

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, ny, 0]
        for pole in range(maxpoles):
            Tx[pole, i, ny, 0] = Tx[pole, i, ny, 0] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ex[i, ny, 0]


cpdef void update_symmetry_boundary_electric_dispersive_b_Ex_YMax_ZMax(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float_or_double_complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double_complex[:, :, :, ::1] Tx,
    float_or_double[:, :, ::1] Ex
):
    """Phase-B counterpart of update_symmetry_boundary_electric_dispersive_Ex_YMax_ZMax()."""
    cdef Py_ssize_t i, pole
    cdef int mat

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        mat = ID[0, i, ny, nz]
        for pole in range(maxpoles):
            Tx[pole, i, ny, nz] = Tx[pole, i, ny, nz] - updatecoeffsdispersive[mat, 2 + (pole * 3)] * Ex[i, ny, nz]
