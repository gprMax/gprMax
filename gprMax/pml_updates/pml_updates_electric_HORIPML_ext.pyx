# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU GenRAl Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU GenRAl Public License for more details.
#
# You should have received a copy of the GNU GenRAl Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.constants cimport floattype_t


cpdef void order1_xminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef floattype_t dx, dHy, dHz, RA01, RB0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA01 = RA[0, i] - 1
        RB0 = RB[0, i]
        RE0 = RE[0, i]
        RF0 = RF[0, i]
        ii = xf - i
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA01 * dHz + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA01 * dHy + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHy

cpdef void order2_xminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef floattype_t dx, dHy, dHz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA0 = RA[0, i]
        RB0 = RB[0, i]
        RE0 = RE[0, i]
        RF0 = RF[0, i]
        RA1 = RA[1, i]
        RB1 = RB[1, i]
        RE1 = RE[1, i]
        RF1 = RF[1, i]
        RA01 = RA[0, i] * RA[1, i] - 1
        ii = xf - i
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA01 * dHz + RA1 * RB0 * Phi1[0, i, j, k] + RB1 * Phi1[1, i, j, k])
                Phi1[1, i, j, k] = RE1 * Phi1[1, i, j, k] - RF1 * (RA0 * dHz + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA01 * dHy + RA1 * RB0 * Phi2[0, i, j, k] + RB1 * Phi2[1, i, j, k])
                Phi2[1, i, j, k] = RE1 * Phi2[1, i, j, k] - RF1 * (RA0 * dHy + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHy


cpdef void order1_xplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef floattype_t dx, dHy, dHz, RA01, RB0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA01 = RA[0, i] - 1
        RB0 = RB[0, i]
        RE0 = RE[0, i]
        RF0 = RF[0, i]
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA01 * dHz + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA01 * dHy + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHy

cpdef void order2_xplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef floattype_t dx, dHy, dHz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA0 = RA[0, i]
        RB0 = RB[0, i]
        RE0 = RE[0, i]
        RF0 = RF[0, i]
        RA1 = RA[1, i]
        RB1 = RB[1, i]
        RE1 = RE[1, i]
        RF1 = RF[1, i]
        RA01 = RA[0, i] * RA[1, i] - 1
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA01 * dHz + RA1 * RB0 * Phi1[0, i, j, k] + RB1 * Phi1[1, i, j, k])
                Phi1[1, i, j, k] = RE1 * Phi1[1, i, j, k] - RF1 * (RA0 * dHz + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA01 * dHy + RA1 * RB0 * Phi2[0, i, j, k] + RB1 * Phi2[1, i, j, k])
                Phi2[1, i, j, k] = RE1 * Phi2[1, i, j, k] - RF1 * (RA0 * dHy + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHy


cpdef void order1_yminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef floattype_t dy, dHx, dHz, RA01, RB0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - j
            RA01 = RA[0, j] - 1
            RB0 = RB[0, j]
            RE0 = RE[0, j]
            RF0 = RF[0, j]
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA01 * dHz + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA01 * dHx + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHx

cpdef void order2_yminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef floattype_t dy, dHx, dHz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - j
            RA0 = RA[0, j]
            RB0 = RB[0, j]
            RE0 = RE[0, j]
            RF0 = RF[0, j]
            RA1 = RA[1, j]
            RB1 = RB[1, j]
            RE1 = RE[1, j]
            RF1 = RF[1, j]
            RA01 = RA[0, j] * RA[1, j] - 1
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA01 * dHz + RA1 * RB0 * Phi1[0, i, j, k] + RB1 * Phi1[1, i, j, k])
                Phi1[1, i, j, k] = RE1 * Phi1[1, i, j, k] - RF1 * (RA0 * dHz + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA01 * dHx + RA1 * RB0 * Phi2[0, i, j, k] + RB1 * Phi2[1, i, j, k])
                Phi2[1, i, j, k] = RE1 * Phi2[1, i, j, k] - RF1 * (RA0 * dHx + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHx


cpdef void order1_yplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef floattype_t dy, dHx, dHz, RA01, RB0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA01 = RA[0, j] - 1
            RB0 = RB[0, j]
            RE0 = RE[0, j]
            RF0 = RF[0, j]
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA01 * dHz + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA01 * dHx + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHx

cpdef void order2_yplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef floattype_t dy, dHx, dHz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA0 = RA[0, j]
            RB0 = RB[0, j]
            RE0 = RE[0, j]
            RF0 = RF[0, j]
            RA1 = RA[1, j]
            RB1 = RB[1, j]
            RE1 = RE[1, j]
            RF1 = RF[1, j]
            RA01 = RA[0, j] * RA[1, j] - 1
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA01 * dHz + RA1 * RB0 * Phi1[0, i, j, k] + RB1 * Phi1[1, i, j, k])
                Phi1[1, i, j, k] = RE1 * Phi1[1, i, j, k] - RF1 * (RA0 * dHz + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA01 * dHx + RA1 * RB0 * Phi2[0, i, j, k] + RB1 * Phi2[1, i, j, k])
                Phi2[1, i, j, k] = RE1 * Phi2[1, i, j, k] - RF1 * (RA0 * dHx + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHx


cpdef void order1_zminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
        """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef floattype_t dz, dHx, dHy, RA01, RB0, RE0, RF0
    dz = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = zf - k
                RA01 = RA[0, k] - 1
                RB0 = RB[0, k]
                RE0 = RE[0, k]
                RF0 = RF[0, k]
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA01 * dHy + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHy
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA01 * dHx + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHx

cpdef void order2_zminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
        """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef floattype_t dz, dHx, dHy, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dz = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = zf - k
                RA0 = RA[0, k]
                RB0 = RB[0, k]
                RE0 = RE[0, k]
                RF0 = RF[0, k]
                RA1 = RA[1, k]
                RB1 = RB[1, k]
                RE1 = RE[1, k]
                RF1 = RF[1, k]
                RA01 = RA[0, k] * RA[1, k] - 1
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA01 * dHy + RA1 * RB0 * Phi1[0, i, j, k] + RB1 * Phi1[1, i, j, k])
                Phi1[1, i, j, k] = RE1 * Phi1[1, i, j, k] - RF1 * (RA0 * dHy + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHy
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA01 * dHx + RA1 * RB0 * Phi2[0, i, j, k] + RB1 * Phi2[1, i, j, k])
                Phi2[1, i, j, k] = RE1 * Phi2[1, i, j, k] - RF1 * (RA0 * dHx + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHx


cpdef void order1_zplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef floattype_t dz, dHx, dHy, RA01, RB0, RE0, RF0
    dz = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                RA01 = RA[0, k] - 1
                RB0 = RB[0, k]
                RE0 = RE[0, k]
                RF0 = RF[0, k]
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA01 * dHy + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHy
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA01 * dHx + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHx

cpdef void order2_zplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsE,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] Phi1,
                        floattype_t[:, :, :, ::1] Phi2,
                        floattype_t[:, ::1] RA,
                        floattype_t[:, ::1] RB,
                        floattype_t[:, ::1] RE,
                        floattype_t[:, ::1] RF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coefficients, ID and field component arrays
        EPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef floattype_t dz, dHx, dHy, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dz = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                RA0 = RA[0, k]
                RB0 = RB[0, k]
                RE0 = RE[0, k]
                RF0 = RF[0, k]
                RA1 = RA[1, k]
                RB1 = RB[1, k]
                RE1 = RE[1, k]
                RF1 = RF[1, k]
                RA01 = RA[0, k] * RA[1, k] - 1
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA01 * dHy + RA1 * RB0 * Phi1[0, i, j, k] + RB1 * Phi1[1, i, j, k])
                Phi1[1, i, j, k] = RE1 * Phi1[1, i, j, k] - RF1 * (RA0 * dHy + RB0 * Phi1[0, i, j, k])
                Phi1[0, i, j, k] = RE0 * Phi1[0, i, j, k] - RF0 * dHy
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA01 * dHx + RA1 * RB0 * Phi2[0, i, j, k] + RB1 * Phi2[1, i, j, k])
                Phi2[1, i, j, k] = RE1 * Phi2[1, i, j, k] - RF1 * (RA0 * dHx + RB0 * Phi2[0, i, j, k])
                Phi2[0, i, j, k] = RE0 * Phi2[0, i, j, k] - RF0 * dHx
