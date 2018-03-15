# Copyright (C) 2015-2018: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.constants cimport floattype_t, complextype_t


########################################################
# Electric field PML updates - 1st order - xminus slab #
########################################################
cpdef void update_pml_1order_electric_HORIPML_xminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef float dx, dHy, dHz, RA0, RB0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA0 = (ERA[0, i] - 1)
        RB0 = ERB[0, i]
        RE0 = ERE[0, i]
        RF0 = ERF[0, i]
        ii = xf - i
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA0 * dHz + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA0 * dHy + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHy


cpdef void update_pml_1order_electric_MRIPML_xminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef float dx, dHy, dHz, RA0, RA1, RB0, RC0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA1 = ((1 / ERA[0, i]) - 1)
        RA0 = (1 / ERA[0, i])
        RB0 = ERB[0, i]
        RE0 = ERE[0, i]
        RF0 = ERF[0, i]
        RC0 = RF0*RB0*RA0
        ii = xf - i
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA1 * dHz - RA0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHz - RC0 * EPhi1[0, i, j, k]
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA1 * dHy - RA0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHy - RC0 * EPhi2[0, i, j, k]


#######################################################
# Electric field PML updates - 1st order - xplus slab #
#######################################################
cpdef void update_pml_1order_electric_HORIPML_xplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef float dx, dHy, dHz, RA0, RB0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA0 = (ERA[0, i] - 1)
        RB0 = ERB[0, i]
        RE0 = ERE[0, i]
        RF0 = ERF[0, i]
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA0 * dHz + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA0 * dHy + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHy

cpdef void update_pml_1order_electric_MRIPML_xplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef float dx, dHy, dHz, RA0, RA1, RB0, RC0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA1 = (1 / ERA[0, i])  - 1
        RA0 = 1 / ERA[0, i]
        RB0 = ERB[0, i]
        RE0 = ERE[0, i]
        RF0 = ERF[0, i]
        RC0 = RF0*RB0*RA0
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA1 * dHz - RA0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHz  - RC0 * EPhi1[0, i, j, k]
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA1 * dHy - RA0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHy - RC0 * EPhi2[0, i, j, k]


########################################################
# Electric field PML updates - 1st order - yminus slab #
########################################################
cpdef void update_pml_1order_electric_HORIPML_yminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef float dy, dHx, dHz, RA0, RB0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - j
            RA0 = (ERA[0, j] - 1)
            RB0 = ERB[0, j]
            RE0 = ERE[0, j]
            RF0 = ERF[0, j]
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA0 * dHz + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA0 * dHx + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHx

cpdef void update_pml_1order_electric_MRIPML_yminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef float dy, dHx, dHz, RA0, RA1, RB0, RC0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - j
            RA1 = (1 / ERA[0, j]) - 1
            RA0 = 1 / ERA[0, j]
            RB0 = ERB[0, j]
            RE0 = ERE[0, j]
            RF0 = ERF[0, j]
            RC0 = RF0*RB0*RA0
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA1 * dHz - RA0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHz - RC0 * EPhi1[0, i, j, k]
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA1 * dHx - RA0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHx - RC0 * EPhi2[0, i, j, k]


#######################################################
# Electric field PML updates - 1st order - yplus slab #
#######################################################
cpdef void update_pml_1order_electric_HORIPML_yplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef float dy, dHx, dHz, RA0, RB0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA0 = (ERA[0, j] - 1)
            RB0 = ERB[0, j]
            RE0 = ERE[0, j]
            RF0 = ERF[0, j]
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA0 * dHz + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA0 * dHx + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHx

cpdef void update_pml_1order_electric_MRIPML_yplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef float dy, dHx, dHz, RA0, RA1, RB0, RC0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA1 = (1 / ERA[0, j]) - 1
            RA0 = 1 / ERA[0, j]
            RB0 = ERB[0, j]
            RE0 = ERE[0, j]
            RF0 = ERF[0, j]
            RC0 = RF0*RB0*RA0
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA1 * dHz - RA0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHz - RC0 * EPhi1[0, i, j, k]
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA1 * dHx - RA0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHx - RC0 * EPhi2[0, i, j, k]



########################################################
# Electric field PML updates - 1st order - zminus slab #
########################################################
cpdef void update_pml_1order_electric_HORIPML_zminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
        """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef float dz, dHx, dHy, RA0, RB0, RE0, RF0
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
                RA0 = (ERA[0, k] - 1)
                RB0 = ERB[0, k]
                RE0 = ERE[0, k]
                RF0 = ERF[0, k]
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA0 * dHy + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHy
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA0 * dHx + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHx

cpdef void update_pml_1order_electric_MRIPML_zminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
        """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef float dz, dHx, dHy, RA0, RA1, RB0, RC0, RE0, RF0
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
                RA1 = (1 / ERA[0, k]) - 1
                RA0 = 1 / ERA[0, k]
                RB0 = ERB[0, k]
                RE0 = ERE[0, k]
                RF0 = ERF[0, k]
                RC0 = RF0*RB0*RA0
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA1 * dHy - RA0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHy - RC0 * EPhi1[0, i, j, k]
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA1 * dHx - RA0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHx - RC0 * EPhi2[0, i, j, k]


#######################################################
# Electric field PML updates - 1st order - zplus slab #
#######################################################
cpdef void update_pml_1order_electric_HORIPML_zplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef float dz, dHx, dHy, RA0, RB0, RE0, RF0
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
                RA0 = (ERA[0, k] - 1)
                RB0 = ERB[0, k]
                RE0 = ERE[0, k]
                RF0 = ERF[0, k]
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA0 * dHy + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHy
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA0 * dHx + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHx

cpdef void update_pml_1order_electric_MRIPML_zplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef float dz, dHx, dHy, RA0, RA1, RB0, RC0, RE0, RF0
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
                RA1 = (1 / ERA[0, k]) - 1
                RA0 = 1 / ERA[0, k]
                RB0 = ERB[0, k]
                RE0 = ERE[0, k]
                RF0 = ERF[0, k]
                RC0 = RF0*RB0*RA0
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA1 * dHy - RA0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHy - RC0 * EPhi1[0, i, j, k]
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA1 * dHx - RA0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHx - RC0 * EPhi2[0, i, j, k]


########################################################
# Magnetic field PML updates - 1st order - xminus slab #
########################################################
cpdef void update_pml_1order_magnetic_HORIPML_xminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hy and Hz field components for the xminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, ERE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHy, materialHz
    cdef float dx, dEy, dEz, RA0, RB0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = xf - (i + 1)
        RA0 = (HRA[0, i] - 1)
        RB0 = HRB[0, i]
        RE0 = HRE[0, i]
        RF0 = HRF[0, i]
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEz = (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk]) / dx
                Hy[ii, jj, kk] = Hy[ii, jj, kk] + updatecoeffsH[materialHy, 4] * (RA0 * dEz + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEz
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEy = (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk]) / dx
                Hz[ii, jj, kk] = Hz[ii, jj, kk] - updatecoeffsH[materialHz, 4] * (RA0 * dEy + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEy

cpdef void update_pml_1order_magnetic_MRIPML_xminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hy and Hz field components for the xminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, ERE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHy, materialHz
    cdef float dx, dEy, dEz, RA0, RA1, RB0, RC0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = xf - (i + 1)
        RA1 = (1 / HRA[0, i]) - 1
        RA0 = 1 / HRA[0, i]
        RB0 = HRB[0, i]
        RE0 = HRE[0, i]
        RF0 = HRF[0, i]
        RC0 = RF0*RB0*RA0
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEz = (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk]) / dx
                Hy[ii, jj, kk] = Hy[ii, jj, kk] + updatecoeffsH[materialHy, 4] * (RA1 * dEz - RA0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEz - RC0 * HPhi1[0, i, j, k]
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEy = (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk]) / dx
                Hz[ii, jj, kk] = Hz[ii, jj, kk] - updatecoeffsH[materialHz, 4] * (RA1 * dEy - RA0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEy - RC0 * HPhi2[0, i, j, k]


#######################################################
# Magnetic field PML updates - 1st order - xplus slab #
#######################################################
cpdef void update_pml_1order_magnetic_HORIPML_xplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hy and Hz field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHy, materialHz
    cdef float dx, dEy, dEz, RA0, RB0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        RA0 = (HRA[0, i] - 1)
        RB0 = HRB[0, i]
        RE0 = HRE[0, i]
        RF0 = HRF[0, i]
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEz = (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk]) / dx
                Hy[ii, jj, kk] = Hy[ii, jj, kk] + updatecoeffsH[materialHy, 4] * (RA0 * dEz + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEz
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEy = (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk]) / dx
                Hz[ii, jj, kk] = Hz[ii, jj, kk] - updatecoeffsH[materialHz, 4] * (RA0 * dEy + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEy

cpdef void update_pml_1order_magnetic_MRIPML_xplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hy and Hz field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHy, materialHz
    cdef float dx, dEy, dEz, RA0, RA1, RB0, RC0, RE0, RF0
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        RA1 = (1 / HRA[0, i]) - 1
        RA0 = 1 / HRA[0, i]
        RB0 = HRB[0, i]
        RE0 = HRE[0, i]
        RF0 = HRF[0, i]
        RC0 = RF0*RB0*RA0
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEz = (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk]) / dx
                Hy[ii, jj, kk] = Hy[ii, jj, kk] + updatecoeffsH[materialHy, 4] * (RA1 * dEz - RA0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEz - RC0 * HPhi1[0, i, j, k]
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEy = (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk]) / dx
                Hz[ii, jj, kk] = Hz[ii, jj, kk] - updatecoeffsH[materialHz, 4] * (RA1 * dEy - RA0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEy - RC0 * HPhi2[0, i, j, k]


########################################################
# Magnetic field PML updates - 1st order - yminus slab #
########################################################
cpdef void update_pml_1order_magnetic_HORIPML_yminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hz field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHz
    cdef float dy, dEx, dEz, RA0, RB0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - (j + 1)
            RA0 = (HRA[0, j] - 1)
            RB0 = HRB[0, j]
            RE0 = HRE[0, j]
            RF0 = HRF[0, j]
            for k in range(0, nz):
                kk = k + zs
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEz = (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk]) / dy
                Hx[ii, jj, kk] = Hx[ii, jj, kk] - updatecoeffsH[materialHx, 4] * (RA0 * dEz + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEz
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEx = (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk]) / dy
                Hz[ii, jj, kk] = Hz[ii, jj, kk] + updatecoeffsH[materialHz, 4] * (RA0 * dEx + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEx

cpdef void update_pml_1order_magnetic_MRIPML_yminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hz field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHz
    cdef float dy, dEx, dEz, RA0, RA1, RB0, RC0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - (j + 1)
            RA1 = (1 / HRA[0, j]) - 1
            RA0 = 1 / HRA[0, j]
            RB0 = HRB[0, j]
            RE0 = HRE[0, j]
            RF0 = HRF[0, j]
            RC0 = RF0*RB0*RA0
            for k in range(0, nz):
                kk = k + zs
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEz = (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk]) / dy
                Hx[ii, jj, kk] = Hx[ii, jj, kk] - updatecoeffsH[materialHx, 4] * (RA1 * dEz - RA0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEz - RC0 * HPhi1[0, i, j, k]
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEx = (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk]) / dy
                Hz[ii, jj, kk] = Hz[ii, jj, kk] + updatecoeffsH[materialHz, 4] * (RA1 * dEx - RA0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEx - RC0 * HPhi2[0, i, j, k]


#######################################################
# Magnetic field PML updates - 1st order - yplus slab #
#######################################################
cpdef void update_pml_1order_magnetic_HORIPML_yplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hz field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHz
    cdef float dy, dEx, dEz, RA0, RB0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA0 = (HRA[0, j] - 1)
            RB0 = HRB[0, j]
            RE0 = HRE[0, j]
            RF0 = HRF[0, j]
            for k in range(0, nz):
                kk = k + zs
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEz = (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk]) / dy
                Hx[ii, jj, kk] = Hx[ii, jj, kk] - updatecoeffsH[materialHx, 4] * (RA0 * dEz + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEz
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEx = (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk]) / dy
                Hz[ii, jj, kk] = Hz[ii, jj, kk] + updatecoeffsH[materialHz, 4] * (RA0 * dEx + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEx

cpdef void update_pml_1order_magnetic_MRIPML_yplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hz field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHz
    cdef float dy, dEx, dEz, RA0, RA1, RB0, RC0, RE0, RF0
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA1 = (1 / HRA[0, j]) - 1
            RA0 = 1 / HRA[0, j]
            RB0 = HRB[0, j]
            RE0 = HRE[0, j]
            RF0 = HRF[0, j]
            RC0 = RF0*RB0*RA0
            for k in range(0, nz):
                kk = k + zs
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEz = (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk]) / dy
                Hx[ii, jj, kk] = Hx[ii, jj, kk] - updatecoeffsH[materialHx, 4] * (RA1 * dEz - RA0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEz - RC0 * HPhi1[0, i, j, k]
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEx = (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk]) / dy
                Hz[ii, jj, kk] = Hz[ii, jj, kk] + updatecoeffsH[materialHz, 4] * (RA1 * dEx - RA0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEx - RC0 * HPhi2[0, i, j, k]


########################################################
# Magnetic field PML updates - 1st order - zminus slab #
########################################################
cpdef void update_pml_1order_magnetic_HORIPML_zminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hy field components for the zminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHy
    cdef float dz, dEx, dEy, RA0, RB0, RE0, RF0
    dz = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = zf - (k + 1)
                RA0 = (HRA[0, k] - 1)
                RB0 = HRB[0, k]
                RE0 = HRE[0, k]
                RF0 = HRF[0, k]
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEy = (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk]) / dz
                Hx[ii, jj, kk] = Hx[ii, jj, kk] + updatecoeffsH[materialHx, 4] * (RA0 * dEy + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEy
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEx = (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk]) / dz
                Hy[ii, jj, kk] = Hy[ii, jj, kk] - updatecoeffsH[materialHy, 4] * (RA0 * dEx + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEx

cpdef void update_pml_1order_magnetic_MRIPML_zminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hy field components for the zminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHy
    cdef float dz, dEx, dEy, RA0, RA1, RB0, RC0, RE0, RF0
    dz = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = zf - (k + 1)
                RA1 = (1 / HRA[0, k]) - 1
                RA0 = 1 / HRA[0, k]
                RB0 = HRB[0, k]
                RE0 = HRE[0, k]
                RF0 = HRF[0, k]
                RC0 = RF0*RB0*RA0
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEy = (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk]) / dz
                Hx[ii, jj, kk] = Hx[ii, jj, kk] + updatecoeffsH[materialHx, 4] * (RA1 * dEy - RA0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEy - RC0 * HPhi1[0, i, j, k]
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEx = (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk]) / dz
                Hy[ii, jj, kk] = Hy[ii, jj, kk] - updatecoeffsH[materialHy, 4] * (RA1 * dEx - RA0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEx - RC0 * HPhi2[0, i, j, k]



#######################################################
# Magnetic field PML updates - 1st order - zplus slab #
#######################################################
cpdef void update_pml_1order_magnetic_HORIPML_zplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hy field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHy
    cdef float dz, dEx, dEy, RA0, RB0, RE0, RF0
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
                RA0 = (HRA[0, k] - 1)
                RB0 = HRB[0, k]
                RE0 = HRE[0, k]
                RF0 = HRF[0, k]
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEy = (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk]) / dz
                Hx[ii, jj, kk] = Hx[ii, jj, kk] + updatecoeffsH[materialHx, 4] * (RA0 * dEy + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEy
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEx = (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk]) / dz
                Hy[ii, jj, kk] = Hy[ii, jj, kk] - updatecoeffsH[materialHy, 4] * (RA0 * dEx + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEx

cpdef void update_pml_1order_magnetic_MRIPML_zplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hy field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHy
    cdef float dz, dEx, dEy, RA0, RA1, RB0, RC0, RE0, RF0
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
                RA1 = (1 / HRA[0, k]) - 1
                RA0 = 1 / HRA[0, k]
                RB0 = HRB[0, k]
                RE0 = HRE[0, k]
                RF0 = HRF[0, k]
                RC0 = RF0*RB0*RA0
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEy = (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk]) / dz
                Hx[ii, jj, kk] = Hx[ii, jj, kk] + updatecoeffsH[materialHx, 4] * (RA1 * dEy - RA0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEy - RC0 * HPhi1[0, i, j, k]
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEx = (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk]) / dz
                Hy[ii, jj, kk] = Hy[ii, jj, kk] - updatecoeffsH[materialHy, 4] * (RA1 * dEx - RA0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEx - RC0 * HPhi2[0, i, j, k]


########################################################
# Electric field PML updates - 2nd order - xminus slab #
########################################################
cpdef void update_pml_2order_electric_HORIPML_xminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef float dx, dHy, dHz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA0 = ERA[0, i]
        RB0 = ERB[0, i]
        RE0 = ERE[0, i]
        RF0 = ERF[0, i]
        RA1 = ERA[1, i]
        RB1 = ERB[1, i]
        RE1 = ERE[1, i]
        RF1 = ERF[1, i]
        RA01 = ERA[0, i] * ERA[1, i] - 1
        ii = xf - i
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA01 * dHz + RA1 * RB0 * EPhi1[0, i, j, k] + RB1 * EPhi1[1, i, j, k])
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] - RF1 * (RA0 * dHz + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA01 * dHy + RA1 * RB0 * EPhi2[0, i, j, k] + RB1 * EPhi2[1, i, j, k])
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] - RF1 * (RA0 * dHy + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHy

cpdef void update_pml_2order_electric_MRIPML_xminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef float dx, dHy, dHz, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RE1, RC1, RF1, EPsi1, EPsi2
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA = ERA[0, i] + ERA[1, i]
        IRA1 = (1 / RA) - 1
        IRA = 1 / RA
        RB0 = ERB[0, i]
        RE0 = ERE[0, i]
        RF0 = ERF[0, i]
        RC0 = RF0*RB0*IRA
        RB1 = ERB[1, i]
        RE1 = ERE[1, i]
        RF1 = ERF[1, i]
        RC1 = RF1*RB1*IRA
        ii = xf - i
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                EPsi1 = EPhi1[0, i, j, k] + EPhi1[1, i, j, k]
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (IRA1 * dHz - IRA * EPsi1)
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] + RC1 * dHz - RC1 * EPsi1
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHz - RC0 * EPsi1
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                EPsi2 = EPhi2[0, i, j, k] + EPhi2[1, i, j, k]
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (IRA1 * dHy - IRA * EPsi2)
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] + RC1 * dHy - RC1 * EPsi2
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHy - RC0 * EPsi2



#######################################################
# Electric field PML updates - 2nd order - xplus slab #
#######################################################
cpdef void update_pml_2order_electric_HORIPML_xplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef float dx, dHy, dHz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA0 = ERA[0, i]
        RB0 = ERB[0, i]
        RE0 = ERE[0, i]
        RF0 = ERF[0, i]
        RA1 = ERA[1, i]
        RB1 = ERB[1, i]
        RE1 = ERE[1, i]
        RF1 = ERF[1, i]
        RA01 = ERA[0, i] * ERA[1, i] - 1
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (RA01 * dHz + RA1 * RB0 * EPhi1[0, i, j, k] + RB1 * EPhi1[1, i, j, k])
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] - RF1 * (RA0 * dHz + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (RA01 * dHy + RA1 * RB0 * EPhi2[0, i, j, k] + RB1 * EPhi2[1, i, j, k])
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] - RF1 * (RA0 * dHy + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHy

cpdef void update_pml_2order_electric_MRIPML_xplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ey and Ez field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEy, materialEz
    cdef float dx, dHy, dHz, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, EPsi1, EPsi2
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        RA = ERA[0, i] + ERA[1, i]
        IRA1 = (1 / RA) - 1
        IRA = 1 / RA
        RB0 = ERB[0, i]
        RE0 = ERE[0, i]
        RF0 = ERF[0, i]
        RC0 = RF0*RB0*IRA
        RB1 = ERB[1, i]
        RE1 = ERE[1, i]
        RF1 = ERF[1, i]
        RC1 = RF1*RB1*IRA
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]) / dx
                EPsi1 = EPhi1[0, i, j, k] + EPhi1[1, i, j, k]
                Ey[ii, jj, kk] = Ey[ii, jj, kk] - updatecoeffsE[materialEy, 4] * (IRA1 * dHz - IRA * EPsi1)
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] + RC1 * dHz - RC1 * EPsi1
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHz - RC0 * EPsi1
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) / dx
                EPsi2 = EPhi2[0, i, j, k] + EPhi2[1, i, j, k]
                Ez[ii, jj, kk] = Ez[ii, jj, kk] + updatecoeffsE[materialEz, 4] * (IRA1 * dHy - IRA * EPsi2)
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] + RC1 * dHy - RC1 * EPsi2
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHy - RC0 * EPsi2


########################################################
# Electric field PML updates - 2nd order - yminus slab #
########################################################
cpdef void update_pml_2order_electric_HORIPML_yminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef float dy, dHx, dHz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - j
            RA0 = ERA[0, j]
            RB0 = ERB[0, j]
            RE0 = ERE[0, j]
            RF0 = ERF[0, j]
            RA1 = ERA[1, j]
            RB1 = ERB[1, j]
            RE1 = ERE[1, j]
            RF1 = ERF[1, j]
            RA01 = ERA[0, j] * ERA[1, j] - 1
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA01 * dHz + RA1 * RB0 * EPhi1[0, i, j, k] + RB1 * EPhi1[1, i, j, k])
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] - RF1 * (RA0 * dHz + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA01 * dHx + RA1 * RB0 * EPhi2[0, i, j, k] + RB1 * EPhi2[1, i, j, k])
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] - RF1 * (RA0 * dHx + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHx

cpdef void update_pml_2order_electric_MRIPML_yminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef float dy, dHx, dHz, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, EPsi1, EPsi2
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - j
            RA = ERA[0, j] + ERA[1, j]
            IRA1 = (1 / RA) - 1
            IRA = 1 / RA
            RB0 = ERB[0, j]
            RE0 = ERE[0, j]
            RF0 = ERF[0, j]
            RC0 = RF0*RB0*IRA
            RB1 = ERB[1, j]
            RE1 = ERE[1, j]
            RF1 = ERF[1, j]
            RC1 = RF1*RB1*IRA
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                EPsi1 = EPhi1[0, i, j, k] + EPhi1[1, i, j, k]
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (IRA1 * dHz - IRA * EPsi1)
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] + RC1 * dHz - RC1 * EPsi1
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHz - RC0 * EPsi1
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                EPsi2 = EPhi2[0, i, j, k] + EPhi2[1, i, j, k]
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (IRA1 * dHx - IRA * EPsi2)
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] + RC1 * dHx - RC1 * EPsi2
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHx - RC0 * EPsi2


#######################################################
# Electric field PML updates - 2nd order - yplus slab #
#######################################################
cpdef void update_pml_2order_electric_HORIPML_yplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef float dy, dHx, dHz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA0 = ERA[0, j]
            RB0 = ERB[0, j]
            RE0 = ERE[0, j]
            RF0 = ERF[0, j]
            RA1 = ERA[1, j]
            RB1 = ERB[1, j]
            RE1 = ERE[1, j]
            RF1 = ERF[1, j]
            RA01 = ERA[0, j] * ERA[1, j] - 1
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (RA01 * dHz + RA1 * RB0 * EPhi1[0, i, j, k] + RB1 * EPhi1[1, i, j, k])
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] - RF1 * (RA0 * dHz + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHz
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (RA01 * dHx + RA1 * RB0 * EPhi2[0, i, j, k] + RB1 * EPhi2[1, i, j, k])
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] - RF1 * (RA0 * dHx + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHx

cpdef void update_pml_2order_electric_MRIPML_yplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ez field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEz
    cdef float dy, dHx, dHz, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, EPsi1, EPsi2
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA = ERA[0, j] + ERA[1, j]
            IRA1 = (1 / RA) - 1
            IRA = 1 / RA
            RB0 = ERB[0, j]
            RE0 = ERE[0, j]
            RF0 = ERF[0, j]
            RC0 = RF0*RB0*IRA
            RB1 = ERB[1, j]
            RE1 = ERE[1, j]
            RF1 = ERF[1, j]
            RC1 = RF1*RB1*IRA
            for k in range(0, nz):
                kk = k + zs
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHz = (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) / dy
                EPsi1 = EPhi1[0, i, j, k] + EPhi1[1, i, j, k]
                Ex[ii, jj, kk] = Ex[ii, jj, kk] + updatecoeffsE[materialEx, 4] * (IRA1 * dHz - IRA * EPsi1)
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] + RC1 * dHz - RC1 * EPsi1
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHz - RC0 * EPsi1
                # Ez
                materialEz = ID[2, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]) / dy
                EPsi2 = EPhi2[0, i, j, k] + EPhi2[1, i, j, k]
                Ez[ii, jj, kk] = Ez[ii, jj, kk] - updatecoeffsE[materialEz, 4] * (IRA1 * dHx - IRA * EPsi2)
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] + RC1 * dHx - RC1 * EPsi2
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHx - RC0 * EPsi2


########################################################
# Electric field PML updates - 2nd order - zminus slab #
########################################################
cpdef void update_pml_2order_electric_HORIPML_zminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
        """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef float dz, dHx, dHy, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
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
                RA0 = ERA[0, k]
                RB0 = ERB[0, k]
                RE0 = ERE[0, k]
                RF0 = ERF[0, k]
                RA1 = ERA[1, k]
                RB1 = ERB[1, k]
                RE1 = ERE[1, k]
                RF1 = ERF[1, k]
                RA01 = ERA[0, k] * ERA[1, k] - 1
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA01 * dHy + RA1 * RB0 * EPhi1[0, i, j, k] + RB1 * EPhi1[1, i, j, k])
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] - RF1 * (RA0 * dHy + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHy
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA01 * dHx + RA1 * RB0 * EPhi2[0, i, j, k] + RB1 * EPhi2[1, i, j, k])
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] - RF1 * (RA0 * dHx + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHx

cpdef void update_pml_2order_electric_MRIPML_zminus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zminus slab.

        Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
        """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef float dz, dHx, dHy, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, EPsi1, EPsi2
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
                RA = ERA[0, k] + ERA[1, k]
                IRA1 = (1 /RA) - 1
                IRA = 1 / RA
                RB0 = ERB[0, k]
                RE0 = ERE[0, k]
                RF0 = ERF[0, k]
                RC0 = RF0*RB0*IRA
                RB1 = ERB[1, k]
                RE1 = ERE[1, k]
                RF1 = ERF[1, k]
                RC1 = RF1*RB1*IRA
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                EPsi1 = EPhi1[0, i, j, k] + EPhi1[1, i, j, k]
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (IRA1 * dHy - IRA * EPsi1)
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] + RC1 * dHy - RC1 * EPsi1
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHy - RC0 * EPsi1
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                EPsi2 = EPhi2[0, i, j, k] + EPhi2[1, i, j, k]
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (IRA1 * dHx - IRA * EPsi2)
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] + RC1 * dHx - RC1 * EPsi2
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHx - RC0 * EPsi2


#######################################################
# Electric field PML updates - 2nd order - zplus slab #
#######################################################
cpdef void update_pml_2order_electric_HORIPML_zplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef float dz, dHx, dHy, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
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
                RA0 = ERA[0, k]
                RB0 = ERB[0, k]
                RE0 = ERE[0, k]
                RF0 = ERF[0, k]
                RA1 = ERA[1, k]
                RB1 = ERB[1, k]
                RE1 = ERE[1, k]
                RF1 = ERF[1, k]
                RA01 = ERA[0, k] * ERA[1, k] - 1
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (RA01 * dHy + RA1 * RB0 * EPhi1[0, i, j, k] + RB1 * EPhi1[1, i, j, k])
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] - RF1 * (RA0 * dHy + RB0 * EPhi1[0, i, j, k])
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] - RF0 * dHy
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (RA01 * dHx + RA1 * RB0 * EPhi2[0, i, j, k] + RB1 * EPhi2[1, i, j, k])
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] - RF1 * (RA0 * dHx + RB0 * EPhi2[0, i, j, k])
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] - RF0 * dHx

cpdef void update_pml_2order_electric_MRIPML_zplus(
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
                        floattype_t[:, :, :, ::1] EPhi1,
                        floattype_t[:, :, :, ::1] EPhi2,
                        floattype_t[:, ::1] ERA,
                        floattype_t[:, ::1] ERB,
                        floattype_t[:, ::1] ERE,
                        floattype_t[:, ::1] ERF,
                        float d
                ):
    """This function updates the Ex and Ey field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, ERA, ERB, ERE, ERF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialEx, materialEy
    cdef float dz, dHx, dHy, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, EPsi1, EPsi2
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
                RA = ERA[0, k] + ERA[1, k]
                IRA1 = (1 / RA) - 1
                IRA = 1 / RA
                RB0 = ERB[0, k]
                RE0 = ERE[0, k]
                RF0 = ERF[0, k]
                RC0 = RF0*RB0*IRA
                RB1 = ERB[1, k]
                RE1 = ERE[1, k]
                RF1 = ERF[1, k]
                RC1 = RF1*RB1*IRA
                # Ex
                materialEx = ID[0, ii, jj, kk]
                dHy = (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]) / dz
                EPsi1 = EPhi1[0, i, j, k] + EPhi1[1, i, j, k]
                Ex[ii, jj, kk] = Ex[ii, jj, kk] - updatecoeffsE[materialEx, 4] * (IRA1 * dHy - IRA * EPsi1)
                EPhi1[1, i, j, k] = RE1 * EPhi1[1, i, j, k] + RC1 * dHy - RC1 * EPsi1
                EPhi1[0, i, j, k] = RE0 * EPhi1[0, i, j, k] + RC0 * dHy - RC0 * EPsi1
                # Ey
                materialEy = ID[1, ii, jj, kk]
                dHx = (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) / dz
                EPsi2 = EPhi2[0, i, j, k] + EPhi2[1, i, j, k]
                Ey[ii, jj, kk] = Ey[ii, jj, kk] + updatecoeffsE[materialEy, 4] * (IRA1 * dHx - IRA * EPsi2)
                EPhi2[1, i, j, k] = RE1 * EPhi2[1, i, j, k] + RC1 * dHx - RC1 * EPsi2
                EPhi2[0, i, j, k] = RE0 * EPhi2[0, i, j, k] + RC0 * dHx - RC0 * EPsi2


########################################################
# Magnetic field PML updates - 2nd order - xminus slab #
########################################################
cpdef void update_pml_2order_magnetic_HORIPML_xminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hy and Hz field components for the xminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, ERE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHy, materialHz
    cdef float dx, dEy, dEz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = xf - (i + 1)
        RA0 = HRA[0, i]
        RB0 = HRB[0, i]
        RE0 = HRE[0, i]
        RF0 = HRF[0, i]
        RA1 = HRA[1, i]
        RB1 = HRB[1, i]
        RE1 = HRE[1, i]
        RF1 = HRF[1, i]
        RA01 = HRA[0, i] * HRA[1, i] - 1
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEz = (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk]) / dx
                Hy[ii, jj, kk] = Hy[ii, jj, kk] + updatecoeffsH[materialHy, 4] * (RA01 * dEz + RA1 * RB0 * HPhi1[0, i, j, k] + RB1 * HPhi1[1, i, j, k])
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] - RF1 * (RA0 * dEz + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEz
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEy = (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk]) / dx
                Hz[ii, jj, kk] = Hz[ii, jj, kk] - updatecoeffsH[materialHz, 4] * (RA01 * dEy + RA1 * RB0 * HPhi2[0, i, j, k] + RB1 * HPhi2[1, i, j, k])
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] - RF1 * (RA0 * dEy + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEy

cpdef void update_pml_2order_magnetic_MRIPML_xminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hy and Hz field components for the xminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, ERE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHy, materialHz
    cdef float dx, dEy, dEz, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, HPsi1, HPsi2
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = xf - (i + 1)
        RA = HRA[0, i] + HRA[1, i]
        IRA1 = (1 / RA) - 1
        IRA = 1 / RA
        RB0 = HRB[0, i]
        RE0 = HRE[0, i]
        RF0 = HRF[0, i]
        RC0 = RF0*RB0*IRA
        RB1 = HRB[1, i]
        RE1 = HRE[1, i]
        RF1 = HRF[1, i]
        RC1 = RF1*RB1*IRA
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEz = (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk]) / dx
                HPsi1 = HPhi1[0, i, j, k] + HPhi1[1, i, j, k]
                Hy[ii, jj, kk] = Hy[ii, jj, kk] + updatecoeffsH[materialHy, 4] * (IRA1 * dEz - IRA * HPsi1)
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] + RC1 * dEz - RC1 * HPsi1
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEz - RC0 * HPsi1
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEy = (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk]) / dx
                HPsi2 = HPhi2[0, i, j, k] + HPhi2[1, i, j, k]
                Hz[ii, jj, kk] = Hz[ii, jj, kk] - updatecoeffsH[materialHz, 4] * (IRA1 * dEy - IRA * HPsi2)
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] + RC1 * dEy - RC1 * HPsi2
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEy - RC0 * HPsi2


#######################################################
# Magnetic field PML updates - 2nd order - xplus slab #
#######################################################
cpdef void update_pml_2order_magnetic_HORIPML_xplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hy and Hz field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHy, materialHz
    cdef float dx, dEy, dEz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        RA0 = HRA[0, i]
        RB0 = HRB[0, i]
        RE0 = HRE[0, i]
        RF0 = HRF[0, i]
        RA1 = HRA[1, i]
        RB1 = HRB[1, i]
        RE1 = HRE[1, i]
        RF1 = HRF[1, i]
        RA01 = HRA[0, i] * HRA[1, i] - 1
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEz = (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk]) / dx
                Hy[ii, jj, kk] = Hy[ii, jj, kk] + updatecoeffsH[materialHy, 4] * (RA01 * dEz + RA1 * RB0 * HPhi1[0, i, j, k] + RB1 * HPhi1[1, i, j, k])
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] - RF1 * (RA0 * dEz + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEz
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEy = (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk]) / dx
                Hz[ii, jj, kk] = Hz[ii, jj, kk] - updatecoeffsH[materialHz, 4] * (RA01 * dEy + RA1 * RB0 * HPhi2[0, i, j, k] + RB1 * HPhi2[1, i, j, k])
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] - RF1 * (RA0 * dEy + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEy

cpdef void update_pml_2order_magnetic_MRIPML_xplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hy and Hz field components for the xplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHy, materialHz
    cdef float dx, dEy, dEz, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, HPsi1, HPsi2
    dx = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        RA = HRA[0, i] + HRA[1, i]
        IRA1 = (1 / RA) - 1
        IRA = 1 / RA
        RB0 = HRB[0, i]
        RE0 = HRE[0, i]
        RF0 = HRF[0, i]
        RC0 = RF0*RB0*IRA
        RB1 = HRB[1, i]
        RE1 = HRE[1, i]
        RF1 = HRF[1, i]
        RC1 = RF1*RB1*IRA
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = k + zs
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEz = (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk]) / dx
                HPsi1 = HPhi1[0, i, j, k] + HPhi1[1, i, j, k]
                Hy[ii, jj, kk] = Hy[ii, jj, kk] + updatecoeffsH[materialHy, 4] * (IRA1 * dEz - IRA * HPsi1)
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] + RC1 * dEz - RC1 * HPsi1
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEz - RC0 * HPsi1
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEy = (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk]) / dx
                HPsi2 = HPhi2[0, i, j, k] + HPhi2[1, i, j, k]
                Hz[ii, jj, kk] = Hz[ii, jj, kk] - updatecoeffsH[materialHz, 4] * (IRA1 * dEy - IRA * HPsi2)
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] + RC1 * dEy - RC1 * HPsi2
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEy - RC0 * HPsi2


########################################################
# Magnetic field PML updates - 2nd order - yminus slab #
########################################################
cpdef void update_pml_2order_magnetic_HORIPML_yminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hz field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHz
    cdef float dy, dEx, dEz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - (j + 1)
            RA0 = HRA[0, j]
            RB0 = HRB[0, j]
            RE0 = HRE[0, j]
            RF0 = HRF[0, j]
            RA1 = HRA[1, j]
            RB1 = HRB[1, j]
            RE1 = HRE[1, j]
            RF1 = HRF[1, j]
            RA01 = HRA[0, j] * HRA[1, j] - 1
            for k in range(0, nz):
                kk = k + zs
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEz = (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk]) / dy
                Hx[ii, jj, kk] = Hx[ii, jj, kk] - updatecoeffsH[materialHx, 4] * (RA01 * dEz + RA1 * RB0 * HPhi1[0, i, j, k] + RB1 * HPhi1[1, i, j, k])
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] - RF1 * (RA0 * dEz + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEz
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEx = (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk]) / dy
                Hz[ii, jj, kk] = Hz[ii, jj, kk] + updatecoeffsH[materialHz, 4] * (RA01 * dEx + RA1 * RB0 * HPhi2[0, i, j, k] + RB1 * HPhi2[1, i, j, k])
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] - RF1 * (RA0 * dEx + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEx

cpdef void update_pml_2order_magnetic_MRIPML_yminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hz field components for the yminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHz
    cdef float dy, dEx, dEz, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, HPsi1, HPsi2
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = yf - (j + 1)
            RA = HRA[0, j] + HRA[1, j]
            IRA1 = (1 / RA) - 1
            IRA = 1 / RA
            RB0 = HRB[0, j]
            RE0 = HRE[0, j]
            RF0 = HRF[0, j]
            RC0 = RF0*RB0*IRA
            RB1 = HRB[1, j]
            RE1 = HRE[1, j]
            RF1 = HRF[1, j]
            RC1 = RF1*RB1*IRA
            for k in range(0, nz):
                kk = k + zs
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEz = (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk]) / dy
                HPsi1 = HPhi1[0, i, j, k] + HPhi1[1, i, j, k]
                Hx[ii, jj, kk] = Hx[ii, jj, kk] - updatecoeffsH[materialHx, 4] * (IRA1 * dEz - IRA * HPsi1)
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] + RC1 * dEz - RC1 * HPsi1
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEz - RC0 * HPsi1
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEx = (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk]) / dy
                HPsi2 = HPhi2[0, i, j, k] + HPhi2[1, i, j, k]
                Hz[ii, jj, kk] = Hz[ii, jj, kk] + updatecoeffsH[materialHz, 4] * (IRA1 * dEx - IRA * HPsi2)
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] + RC1 * dEx - RC1 * HPsi2
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEx - RC0 * HPsi2


#######################################################
# Magnetic field PML updates - 2nd order - yplus slab #
#######################################################
cpdef void update_pml_2order_magnetic_HORIPML_yplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hz field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHz
    cdef float dy, dEx, dEz, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA0 = HRA[0, j]
            RB0 = HRB[0, j]
            RE0 = HRE[0, j]
            RF0 = HRF[0, j]
            RA1 = HRA[1, j]
            RB1 = HRB[1, j]
            RE1 = HRE[1, j]
            RF1 = HRF[1, j]
            RA01 = HRA[0, j] * HRA[1, j] - 1
            for k in range(0, nz):
                kk = k + zs
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEz = (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk]) / dy
                Hx[ii, jj, kk] = Hx[ii, jj, kk] - updatecoeffsH[materialHx, 4] * (RA01 * dEz + RA1 * RB0 * HPhi1[0, i, j, k] + RB1 * HPhi1[1, i, j, k])
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] - RF1 * (RA0 * dEz + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEz
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEx = (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk]) / dy
                Hz[ii, jj, kk] = Hz[ii, jj, kk] + updatecoeffsH[materialHz, 4] * (RA01 * dEx + RA1 * RB0 * HPhi2[0, i, j, k] + RB1 * HPhi2[1, i, j, k])
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] - RF1 * (RA0 * dEx + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEx

cpdef void update_pml_2order_magnetic_MRIPML_yplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hz field components for the yplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHz
    cdef float dy, dEx, dEz, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, HPsi1, HPsi2
    dy = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            RA = HRA[0, j] + HRA[1, j]
            IRA1 = (1 / RA) - 1
            IRA = 1 / RA
            RB0 = HRB[0, j]
            RE0 = HRE[0, j]
            RF0 = HRF[0, j]
            RC0 = RF0*RB0*IRA
            RB1 = HRB[1, j]
            RE1 = HRE[1, j]
            RF1 = HRF[1, j]
            RC1 = RF1*RB1*IRA
            for k in range(0, nz):
                kk = k + zs
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEz = (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk]) / dy
                HPsi1 = HPhi1[0, i, j, k] + HPhi1[1, i, j, k]
                Hx[ii, jj, kk] = Hx[ii, jj, kk] - updatecoeffsH[materialHx, 4] * (IRA1 * dEz - IRA * HPsi1)
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] + RC1 * dEz - RC1 * HPsi1
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEz - RC0 * HPsi1
                # Hz
                materialHz = ID[5, ii, jj, kk]
                dEx = (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk]) / dy
                HPsi2 = HPhi2[0, i, j, k] + HPhi2[1, i, j, k]
                Hz[ii, jj, kk] = Hz[ii, jj, kk] + updatecoeffsH[materialHz, 4] * (IRA1 * dEx - IRA * HPsi2)
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] + RC1 * dEx - RC1 * HPsi2
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEx - RC0 * HPsi2


########################################################
# Magnetic field PML updates - 2nd order - zminus slab #
########################################################
cpdef void update_pml_2order_magnetic_HORIPML_zminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hy field components for the zminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHy
    cdef float dz, dEx, dEy, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
    dz = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = zf - (k + 1)
                RA0 = HRA[0, k]
                RB0 = HRB[0, k]
                RE0 = HRE[0, k]
                RF0 = HRF[0, k]
                RA1 = HRA[1, k]
                RB1 = HRB[1, k]
                RE1 = HRE[1, k]
                RF1 = HRF[1, k]
                RA01 = HRA[0, k] * HRA[1, k] - 1
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEy = (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk]) / dz
                Hx[ii, jj, kk] = Hx[ii, jj, kk] + updatecoeffsH[materialHx, 4] * (RA01 * dEy + RA1 * RB0 * HPhi1[0, i, j, k] + RB1 * HPhi1[1, i, j, k])
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] - RF1 * (RA0 * dEy + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEy
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEx = (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk]) / dz
                Hy[ii, jj, kk] = Hy[ii, jj, kk] - updatecoeffsH[materialHy, 4] * (RA01 * dEx + RA1 * RB0 * HPhi2[0, i, j, k] + RB1 * HPhi2[1, i, j, k])
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] - RF1 * (RA0 * dEx + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEx

cpdef void update_pml_2order_magnetic_MRIPML_zminus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hy field components for the zminus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHy
    cdef float dz, dEx, dEy, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, HPsi1, HPsi2
    dz = d
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + xs
        for j in range(0, ny):
            jj = j + ys
            for k in range(0, nz):
                kk = zf - (k + 1)
                RA = HRA[0, k] + HRA[1, k]
                IRA1 = (1 / RA) - 1
                IRA = 1 / RA
                RB0 = HRB[0, k]
                RE0 = HRE[0, k]
                RF0 = HRF[0, k]
                RC0 = RF0*RB0*IRA
                RB1 = HRB[1, k]
                RE1 = HRE[1, k]
                RF1 = HRF[1, k]
                RC1 = RF1*RB1*IRA
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEy = (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk]) / dz
                HPsi1 = HPhi1[0, i, j, k] + HPhi1[1, i, j, k]
                Hx[ii, jj, kk] = Hx[ii, jj, kk] + updatecoeffsH[materialHx, 4] * (IRA1 * dEy - IRA * HPsi1)
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] + RC1 * dEy - RC1 * HPsi1
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEy - RC0 * HPsi1
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEx = (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk]) / dz
                HPsi2 = HPhi2[0, i, j, k] + HPhi2[1, i, j, k]
                Hy[ii, jj, kk] = Hy[ii, jj, kk] - updatecoeffsH[materialHy, 4] * (IRA1 * dEx - IRA * HPsi2)
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] + RC1 * dEx - RC1 * HPsi2
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEx - RC0 * HPsi2


#######################################################
# Magnetic field PML updates - 2nd order - zplus slab #
#######################################################
cpdef void update_pml_2order_magnetic_HORIPML_zplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hy field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHy
    cdef float dz, dEx, dEy, RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01
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
                RA0 = HRA[0, k]
                RB0 = HRB[0, k]
                RE0 = HRE[0, k]
                RF0 = HRF[0, k]
                RA1 = HRA[1, k]
                RB1 = HRB[1, k]
                RE1 = HRE[1, k]
                RF1 = HRF[1, k]
                RA01 = HRA[0, k] * HRA[1, k] - 1
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEy = (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk]) / dz
                Hx[ii, jj, kk] = Hx[ii, jj, kk] + updatecoeffsH[materialHx, 4] * (RA01 * dEy + RA1 * RB0 * HPhi1[0, i, j, k] + RB1 * HPhi1[1, i, j, k])
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] - RF1 * (RA0 * dEy + RB0 * HPhi1[0, i, j, k])
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] - RF0 * dEy
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEx = (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk]) / dz
                Hy[ii, jj, kk] = Hy[ii, jj, kk] - updatecoeffsH[materialHy, 4] * (RA01 * dEx + RA1 * RB0 * HPhi2[0, i, j, k] + RB1 * HPhi2[1, i, j, k])
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] - RF1 * (RA0 * dEx + RB0 * HPhi2[0, i, j, k])
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] - RF0 * dEx

cpdef void update_pml_2order_magnetic_MRIPML_zplus(
                        int xs,
                        int xf,
                        int ys,
                        int yf,
                        int zs,
                        int zf,
                        int nthreads,
                        floattype_t[:, ::1] updatecoeffsH,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Ex,
                        floattype_t[:, :, ::1] Ey,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] Hx,
                        floattype_t[:, :, ::1] Hy,
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, :, ::1] HPhi1,
                        floattype_t[:, :, :, ::1] HPhi2,
                        floattype_t[:, ::1] HRA,
                        floattype_t[:, ::1] HRB,
                        floattype_t[:, ::1] HRE,
                        floattype_t[:, ::1] HRF,
                        float d
                ):
    """This function updates the Hx and Hy field components for the zplus slab.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, HRA, HRB, HRE, HRF (memoryviews): Access to PML coefficient arrays
        d (float): Spatial discretisation, e.g. dx, dy or dz
    """

    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef int nx, ny, nz, materialHx, materialHy
    cdef float dz, dEx, dEy, RA, IRA1, IRA, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, HPsi1, HPsi2
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
                RA = HRA[0, k] + HRA[1, k]
                IRA1 = (1 / RA) - 1
                IRA = 1 / RA
                RB0 = HRB[0, k]
                RE0 = HRE[0, k]
                RF0 = HRF[0, k]
                RC0 = RF0*RB0*IRA
                RB1 = HRB[1, k]
                RE1 = HRE[1, k]
                RF1 = HRF[1, k]
                RC1 = RF1*RB1*IRA
                # Hx
                materialHx = ID[3, ii, jj, kk]
                dEy = (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk]) / dz
                HPsi1 = HPhi1[0, i, j, k] + HPhi1[1, i, j, k]
                Hx[ii, jj, kk] = Hx[ii, jj, kk] + updatecoeffsH[materialHx, 4] * (IRA1 * dEy - IRA * HPsi1)
                HPhi1[1, i, j, k] = RE1 * HPhi1[1, i, j, k] + RC1 * dEy - RC1 * HPsi1
                HPhi1[0, i, j, k] = RE0 * HPhi1[0, i, j, k] + RC0 * dEy - RC0 * HPsi1
                # Hy
                materialHy = ID[4, ii, jj, kk]
                dEx = (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk]) / dz
                HPsi2 = HPhi2[0, i, j, k] + HPhi2[1, i, j, k]
                Hy[ii, jj, kk] = Hy[ii, jj, kk] - updatecoeffsH[materialHy, 4] * (IRA1 * dEx - IRA * HPsi2)
                HPhi2[1, i, j, k] = RE1 * HPhi2[1, i, j, k] + RC1 * dEx - RC1 * HPsi2
                HPhi2[0, i, j, k] = RE0 * HPhi2[0, i, j, k] + RC0 * dEx - RC0 * HPsi2
