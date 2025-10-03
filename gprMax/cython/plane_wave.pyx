# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          Adittya Pal, and Nathan Mannall
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

cimport cython
from libc.math cimport M_PI, abs, atan2, ceil, cos, exp, floor, pow, round, sin, sqrt, tan
from libc.stdio cimport FILE, fclose, fopen, fwrite
from libc.string cimport strcmp

from cython.parallel import prange

from gprMax.config cimport float_or_double
from gprMax.config cimport float_or_double_complex


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void applyTFSFMagnetic(
    int nthreads,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
    float_or_double[:, ::1] E_fields,
    float_or_double[:] updatecoeffsH,
    int[:] m,
    int[:] origin,
    int[:] corners 
):
    """Implements total field-scattered field formulation for magnetic field on
        the edge of the TF/SF region of the TFSF Box.

    Args:
        nthreads: int of number of threads to parallelize for loops.
        Hx, Hy, Hz: double array to store magnetic fields for grid cells over
                    the TFSF box at particular indices.
        E_fields: double array to store electric fields of 1D representation of
                    plane wave in a direction along which the wave propagates.
        updatecoeffsH: float of coefficients of fields in TFSF assignment
                        equation for the magnetic field.
        m: int array of integer mappings, m_x, m_y, m_z which determine rational
            angles for assignment of correct element to 3D FDTD grid from 1D
            representation, last element stores max(m_x, m_y, m_z).
        origin: int array of coordinates of origin of the TF/SF box.
        corners: int array of coordinates of corners of TF/SF field boundaries.        
    """

    cdef Py_ssize_t i, j, k = 0

    # Precompute index values
    cdef int index = 0
    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef int Ox = origin[0]
    cdef int Oy = origin[1]
    cdef int Oz = origin[2] 
    
    cdef int x_start = corners[0]
    cdef int y_start = corners[1]
    cdef int z_start = corners[2]
    cdef int x_stop = corners[3]
    cdef int y_stop = corners[4]
    cdef int z_stop = corners[5]

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]

    cdef float_or_double coef_H_xy = updatecoeffsH[2]
    cdef float_or_double coef_H_xz = updatecoeffsH[3]
    cdef float_or_double coef_H_yz = updatecoeffsH[3]
    cdef float_or_double coef_H_yx = updatecoeffsH[1]
    cdef float_or_double coef_H_zx = updatecoeffsH[1]
    cdef float_or_double coef_H_zy = updatecoeffsH[2]


    #**** constant x faces -- scattered-field nodes ****
    i = x_start
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hy at firstX-1/2 by subtracting Ez_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hy[i-1, j, k] -= coef_H_yx * E_z[index]

    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at firstX-1/2 by adding Ey_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hz[i-1, j, k] += coef_H_zx * E_y[index]

    i = x_stop
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hy at lastX+1/2 by adding Ez_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hy[i, j, k] += coef_H_yx * E_z[index]

    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at lastX+1/2 by subtractinging Ey_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hz[i, j, k] -= coef_H_zx * E_y[index]

    #**** constant y faces -- scattered-field nodes ****
    j = y_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hx at firstY-1/2 by adding Ez_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hx[i, j-1, k] += coef_H_xy * E_z[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at firstY-1/2 by subtracting Ex_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hz[i, j-1, k] -= coef_H_zy * E_x[index]

    j = y_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hx at lastY+1/2 by subtracting Ez_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hx[i, j, k] -= coef_H_xy * E_z[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at lastY-1/2 by adding Ex_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hz[i, j, k] += coef_H_zy * E_x[index]

    #**** constant z faces -- scattered-field nodes ****
    k = z_start
    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Hy at firstZ-1/2 by adding Ex_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hy[i, j, k-1] += coef_H_yz * E_x[index]

    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Hx at firstZ-1/2 by subtracting Ey_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hx[i, j, k-1] -= coef_H_xz * E_y[index]

    k = z_stop
    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Hy at firstZ-1/2 by subtracting Ex_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hy[i, j, k] -= coef_H_yz * E_x[index]

    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Hx at lastZ+1/2 by adding Ey_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hx[i, j, k] += coef_H_xz * E_y[index]


cdef void applyTFSFMagnetic_axial(
    int nthreads,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] updatecoeffsH,
    np.uint32_t[:, :, :, ::1] GID,
    int[:] m,
    int[:] origin,
    int[:] corners 
):
    """Implements total field-scattered field formulation for magnetic field on
        the edge of the TF/SF region of the TFSF Box.

    Args:
        nthreads: int of number of threads to parallelize for loops.
        Hx, Hy, Hz: double array to store magnetic fields for grid cells over
                    the TFSF box at particular indices.
        E_fields: double array to store electric fields of 1D representation of
                    plane wave in a direction along which the wave propagates.
        updatecoeffsH: float of coefficients of fields in TFSF assignment
                        equation for the magnetic field.
        m: int array of integer mappings, m_x, m_y, m_z which determine rational
            angles for assignment of correct element to 3D FDTD grid from 1D
            representation, last element stores max(m_x, m_y, m_z).
        origin: int array of coordinates of origin of the TF/SF box.
        corners: int array of coordinates of corners of TF/SF field boundaries.        
    """

    cdef Py_ssize_t i, j, k = 0

    # Precompute index values
    cdef int index = 0
    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef int Ox = origin[0]
    cdef int Oy = origin[1]
    cdef int Oz = origin[2] 
    
    cdef int x_start = corners[0]
    cdef int y_start = corners[1]
    cdef int z_start = corners[2]
    cdef int x_stop = corners[3]
    cdef int y_stop = corners[4]
    cdef int z_stop = corners[5]

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]


    #**** constant x faces -- scattered-field nodes ****
    i = x_start
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hy at firstX-1/2 by subtracting Ez_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hy[i-1, j, k] -= updatecoeffsH[GID[4,i-1,j,k],1] * E_z[index]

    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at firstX-1/2 by adding Ey_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hz[i-1, j, k] += updatecoeffsH[GID[5,i-1,j,k],1] * E_y[index]

    i = x_stop
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hy at lastX+1/2 by adding Ez_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hy[i, j, k] += updatecoeffsH[GID[4,i,j,k],1] * E_z[index]

    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at lastX+1/2 by subtractinging Ey_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hz[i, j, k] -= updatecoeffsH[GID[5,i,j,k],1] * E_y[index]

    #**** constant y faces -- scattered-field nodes ****
    j = y_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hx at firstY-1/2 by adding Ez_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hx[i, j-1, k] += updatecoeffsH[GID[3,i,j-1,k],2] * E_z[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at firstY-1/2 by subtracting Ex_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hz[i, j-1, k] -= updatecoeffsH[GID[5,i,j-1,k],2] * E_x[index]

    j = y_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hx at lastY+1/2 by subtracting Ez_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hx[i, j, k] -= updatecoeffsH[GID[3,i,j,k],2] * E_z[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at lastY-1/2 by adding Ex_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hz[i, j, k] += updatecoeffsH[GID[5,i,j,k],2] * E_x[index]

    #**** constant z faces -- scattered-field nodes ****
    k = z_start
    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Hy at firstZ-1/2 by adding Ex_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hy[i, j, k-1] += updatecoeffsH[GID[4,i,j,k-1],3] * E_x[index]

    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Hx at firstZ-1/2 by subtracting Ey_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hx[i, j, k-1] -= updatecoeffsH[GID[3,i,j,k-1],3] * E_y[index]

    k = z_stop
    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Hy at firstZ-1/2 by subtracting Ex_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hy[i, j, k] -= updatecoeffsH[GID[4,i,j,k],3] * E_x[index]

    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Hx at lastZ+1/2 by adding Ey_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Hx[i, j, k] += updatecoeffsH[GID[3,i,j,k],3] * E_y[index]



cdef void applyTFSFElectric(
    int nthreads,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, ::1] H_fields,
    float_or_double[:] updatecoeffsE,
    int[:] m,
    int[:] origin,
    int[:] corners
 ):

    """Implements total field-scattered field formulation for electric field on
        edge of the TF/SF region of the TFSF Box.

    Args:
        nthreads: int for number of threads to parallelize the for loops.
        Ex, Ey, Ez: double array for magnetic fields for grid cells over TFSF
                    box at particular indices.
        H_fields: double array to store electric fields of 1D representation of
                    plane wave in direction along which wave propagates.
        updatecoeffsE: float of coefficients of fields in TFSF assignment
                        equation for magnetic field.
        m: int array of integer mappings, m_x, m_y, m_z which determine rational
            angles for assignment of correct element to 3D FDTD grid from 1D
            representation, last element stores max(m_x, m_y, m_z).
        origin: int array of coordinates of origin of the TF/SF box.
        corners: int array for coordinates of corners of TF/SF field boundaries.      
    """

    cdef Py_ssize_t i, j, k = 0

    # Precompute index values
    cdef int index = 0
    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef int Ox = origin[0]
    cdef int Oy = origin[1]
    cdef int Oz = origin[2] 
    
    cdef int x_start = corners[0]
    cdef int y_start = corners[1]
    cdef int z_start = corners[2]
    cdef int x_stop = corners[3]
    cdef int y_stop = corners[4]
    cdef int z_stop = corners[5]

    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]

    cdef float_or_double coef_E_xy = updatecoeffsE[2]
    cdef float_or_double coef_E_xz = updatecoeffsE[3]
    cdef float_or_double coef_E_yz = updatecoeffsE[3]
    cdef float_or_double coef_E_yx = updatecoeffsE[1]
    cdef float_or_double coef_E_zx = updatecoeffsE[1]
    cdef float_or_double coef_E_zy = updatecoeffsE[2]


    #**** constant x faces -- total-field nodes ****/
    i = x_start
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at firstX face by subtracting Hy_inc
            index = m_x * (i-1-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ez[i, j, k] -= coef_E_zx * H_y[index]

    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ey at firstX face by adding Hz_inc
            index = m_x * (i-1-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ey[i, j, k] += coef_E_yx * H_z[index]

    i = x_stop
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at lastX face by adding Hy_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ez[i, j, k] += coef_E_zx * H_y[index]

    i = x_stop
    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ey at lastX face by subtracting Hz_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ey[i, j, k] -= coef_E_yx * H_z[index]

    #**** constant y faces -- total-field nodes ****/
    j = y_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at firstY face by adding Hx_inc
            index = m_x * (i-Ox) + m_y * (j-1-Oy) + m_z * (k-Oz)
            Ez[i, j, k] += coef_E_zy * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ex at firstY face by subtracting Hz_inc
            index = m_x * (i-Ox) + m_y * (j-1-Oy) + m_z * (k-Oz)
            Ex[i, j, k] -= coef_E_xy * H_z[index]

    j = y_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at lastY face by subtracting Hx_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ez[i, j, k] -= coef_E_zy * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ex at lastY face by adding Hz_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ex[i, j, k] += coef_E_xy * H_z[index]

    #**** constant z faces -- total-field nodes ****/
    k = z_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Ey at firstZ face by subtracting Hx_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-1-Oz)
            Ey[i, j, k] -= coef_E_yz * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Ex at firstZ face by adding Hy_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-1-Oz)
            Ex[i, j, k] += coef_E_xz * H_y[index]

    k = z_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Ey at lastZ face by adding Hx_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ey[i, j, k] += coef_E_yz * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Ex at lastZ face by subtracting Hy_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ex[i, j, k] -= coef_E_xz * H_y[index]



cdef void applyTFSFElectric_axial(
    int nthreads,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t [:, :, :, ::1] GID,
    int[:] m,
    int[:] origin,
    int[:] corners
 ):

    """Implements total field-scattered field formulation for electric field on
        edge of the TF/SF region of the TFSF Box.

    Args:
        nthreads: int for number of threads to parallelize the for loops.
        Ex, Ey, Ez: double array for magnetic fields for grid cells over TFSF
                    box at particular indices.
        H_fields: double array to store electric fields of 1D representation of
                    plane wave in direction along which wave propagates.
        updatecoeffsE: float of coefficients of fields in TFSF assignment
                        equation for magnetic field.
        m: int array of integer mappings, m_x, m_y, m_z which determine rational
            angles for assignment of correct element to 3D FDTD grid from 1D
            representation, last element stores max(m_x, m_y, m_z).
        origin: int array of coordinates of origin of the TF/SF box.
        corners: int array for coordinates of corners of TF/SF field boundaries.      
    """

    cdef Py_ssize_t i, j, k = 0

    # Precompute index values
    cdef int index = 0
    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef int Ox = origin[0]
    cdef int Oy = origin[1]
    cdef int Oz = origin[2] 
    
    cdef int x_start = corners[0]
    cdef int y_start = corners[1]
    cdef int z_start = corners[2]
    cdef int x_stop = corners[3]
    cdef int y_stop = corners[4]
    cdef int z_stop = corners[5]

    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]

    #**** constant x faces -- total-field nodes ****/
    i = x_start
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at firstX face by subtracting Hy_inc
            index = m_x * (i-1-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ez[i, j, k] -= updatecoeffsE[GID[2,i,j,k],1] * H_y[index]

    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ey at firstX face by adding Hz_inc
            index = m_x * (i-1-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ey[i, j, k] += updatecoeffsE[GID[1,i,j,k],1] * H_z[index]

    i = x_stop
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at lastX face by adding Hy_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ez[i, j, k] += updatecoeffsE[GID[2,i,j,k],1] * H_y[index]

    i = x_stop
    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ey at lastX face by subtracting Hz_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ey[i, j, k] -= updatecoeffsE[GID[1,i,j,k],1] * H_z[index]

    #**** constant y faces -- total-field nodes ****/
    j = y_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at firstY face by adding Hx_inc
            index = m_x * (i-Ox) + m_y * (j-1-Oy) + m_z * (k-Oz)
            Ez[i, j, k] += updatecoeffsE[GID[2,i,j,k],2] * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ex at firstY face by subtracting Hz_inc
            index = m_x * (i-Ox) + m_y * (j-1-Oy) + m_z * (k-Oz)
            Ex[i, j, k] -= updatecoeffsE[GID[1,i,j,k],2] * H_z[index]

    j = y_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at lastY face by subtracting Hx_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ez[i, j, k] -= updatecoeffsE[GID[2,i,j,k],2] * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ex at lastY face by adding Hz_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ex[i, j, k] += updatecoeffsE[GID[1,i,j,k],2] * H_z[index]

    #**** constant z faces -- total-field nodes ****/
    k = z_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Ey at firstZ face by subtracting Hx_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-1-Oz)
            Ey[i, j, k] -= updatecoeffsE[GID[1,i,j,k],3] * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Ex at firstZ face by adding Hy_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-1-Oz)
            Ex[i, j, k] += updatecoeffsE[GID[0,i,j,k],3]* H_y[index]

    k = z_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Ey at lastZ face by adding Hx_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ey[i, j, k] += updatecoeffsE[GID[1,i,j,k],3] * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Ex at lastZ face by subtracting Hy_inc
            index = m_x * (i-Ox) + m_y * (j-Oy) + m_z * (k-Oz)
            Ex[i, j, k] -= updatecoeffsE[GID[0,i,j,k],3] * H_y[index]


cdef void initializeMagneticFields(
    int[:] m,
    float_or_double[:, ::1] H_fields,
    double[:] projections,
    float_or_double[:, :, ::1] waveformvalues_wholedt,
    bint precompute,
    int iteration,
    double dt,
    double ds,
    double c,
    double start,
    double stop,
    double freq,
    char* wavetype
):
    """Initialises first few grid points of source waveform.

    Args:
        m: int array of integer mappings, m_x, m_y, m_z which determine rational
            angles for assignment of correct element to 3D FDTD grid from 1D
            representation, last element stores max(m_x, m_y, m_z).
        H_fields: double array for magnetic fields of 1D representation of plane
                    wave in a direction along which the wave propagates.
        projections: float array for projections of magnetic fields along
                        different cartesian axes.
        waveformvalues_wholedt: double array stores precomputed waveforms at
                                each timestep to initialize magnetic fields.
        precompute: boolean to store whether fields values have been precomputed
                    or should be computed on the fly.
        iterations: int stores number of iterations in the simulation.
        dt: float of timestep for the simulation.
        ds: float of projection vector for sourcing the plane wave.
        c: float of speed of light in the medium.
        start: float of start time at which source is placed in the TFSF grid.
        stop: float of stop time at which source is removed from TFSF grid.
        freq: float of frequency of introduced wave which determines grid points
                per wavelength for wave source.
        wavetype: string of type of waveform whose magnitude should be returned.
    """

    cdef Py_ssize_t r = 0
    cdef double time_x, time_y, time_z = 0.0
    if (precompute == True):
        for r in range(m[3]):      #loop to assign the source values of magnetic field to the first few gridpoints
            H_fields[0, r] = projections[3] * waveformvalues_wholedt[iteration, 0, r]
            H_fields[1, r] = projections[4] * waveformvalues_wholedt[iteration, 1, r]
            H_fields[2, r] = projections[5] * waveformvalues_wholedt[iteration, 2, r]
    else:
        for r in range(m[3]):      #loop to assign the source values of magnetic field to the first few gridpoints
            time_x = dt * iteration - (r+ (abs(m[1])+abs(m[2]))*0.5) * ds/c
            time_y = dt * iteration - (r+ (abs(m[2])+abs(m[0]))*0.5) * ds/c
            time_z = dt * iteration - (r+ (abs(m[0])+abs(m[1]))*0.5) * ds/c
            if (dt * iteration >= start and dt * iteration <= stop):
            # Set the time of the waveform evaluation to account for any delay in the start
                H_fields[0, r] = projections[3] * getSource(time_x-start, freq, wavetype, dt)
                H_fields[1, r] = projections[4] * getSource(time_y-start, freq, wavetype, dt)
                H_fields[2, r] = projections[5] * getSource(time_z-start, freq, wavetype, dt)


cdef void initializeElectricFields(
    int[:] m,
    float_or_double[:, ::1] E_fields,
    double[:] projections,
    float_or_double[:, :, ::1] waveformvalues_halfdt,
    bint precompute,
    int iteration,
    double dt,
    double ds,
    double c,
    double start,
    double stop,
    double freq,
    char* wavetype
):
    """Initialises first few grid points of source waveform.

    Args:
        m: int array of integer mappings, m_x, m_y, m_z which determine rational
            angles for assignment of correct element to 3D FDTD grid from 1D
            representation, last element stores max(m_x, m_y, m_z).
        E_fields: double array for electric fields of 1D representation of plane
                    wave in a direction along which the wave propagates.
        projections: float array for projections of magnetic fields along
                        different cartesian axes.
        waveformvalues_wholedt: double array stores precomputed waveforms at
                                each timestep to initialize magnetic fields.
        precompute: boolean to store whether fields values have been precomputed
                    or should be computed on the fly.
        iterations: int stores number of iterations in the simulation.
        dt: float of timestep for the simulation.
        ds: float of projection vector for sourcing the plane wave.
        c: float of speed of light in the medium.
        start: float of start time at which source is placed in the TFSF grid.
        stop: float of stop time at which source is removed from TFSF grid.
        freq: float of frequency of introduced wave which determines grid points
                per wavelength for wave source.
        wavetype: string of type of waveform whose magnitude should be returned.
    """

    cdef Py_ssize_t r = 0
    cdef double time_x, time_y, time_z = 0.0
    if (precompute == True):
        for r in range(m[3]):      #loop to assign the source values of magnetic field to the first few gridpoints
            E_fields[0, r] = projections[0] * waveformvalues_halfdt[iteration, 0, r]
            E_fields[1, r] = projections[1] * waveformvalues_halfdt[iteration, 1, r]
            E_fields[2, r] = projections[2] * waveformvalues_halfdt[iteration, 2, r]
    else:
        for r in range(m[3]):      #loop to assign the source values of magnetic field to the first few gridpoints
            time_x = dt * (iteration + 0.5) - (r+ abs(m[0])*0.5) * ds/c
            time_y = dt * (iteration + 0.5) - (r+ abs(m[1])*0.5) * ds/c
            time_z = dt * (iteration + 0.5) - (r+ abs(m[2])*0.5) * ds/c
            if (dt * (iteration + 0.5) >= start and dt * (iteration + 0.5) <= stop):
            # Set the time of the waveform evaluation to account for any delay in the start
                E_fields[0, r] = projections[0] * getSource(time_x-start, freq, wavetype, dt)
                E_fields[1, r] = projections[1] * getSource(time_y-start, freq, wavetype, dt)
                E_fields[2, r] = projections[2] * getSource(time_z-start, freq, wavetype, dt)


@cython.cdivision(True)
cdef void updateMagneticFields(
    int n,
    int p,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:, ::1] rcHx,
    float_or_double[:, ::1] rcHy,
    float_or_double[:, ::1] rcHz,
    float_or_double dx,
    float_or_double dy,
    float_or_double dz,
    float_or_double[:] updatecoeffsH,
    int[:] m
):
    """Updates magnetic fields for next time step using Equation 8 of
        DOI: 10.1109/LAWP.2009.2016851.

        Perform PML absorbing boundary condition updates as well according to the First order RIPML scheme.
    Args:
        n: int for spatial length of DPW array to update each length grid
            cell.
        p: int for PML thickness in number of grid cells.
        H_fields: double array of magnetic fields of DPW until temporal
                    index time.
        E_fields: double array of electric fields of DPW until temporal
                    index time.
        Ix, Iy, Iz: double arrays to store integral terms for PML regions
                    along different coordinate axes.
        rcHx, rcHy, rcHz: double arrays to store precomputed coefficients
                            for PML regions along different coordinate axes.
        dx, dy, dz: float of spatial step sizes along different coordinate axes.    
        updatecoeffsH: double array of coefficients of fields in update
                        equation for magnetic field.
        m: int array of integer mappings, m_x, m_y, m_z which determine
            rational angles for assignment of correct element to 3D FDTD
            grid from 1D representation, last element stores
            max(m_x, m_y, m_z).
    """

    cdef Py_ssize_t j, i = 0

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]
    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]

    cdef float_or_double coef_H_xt = updatecoeffsH[0]
    cdef float_or_double coef_H_xy = updatecoeffsH[2]
    cdef float_or_double coef_H_xz = updatecoeffsH[3]

    cdef float_or_double coef_H_yt = updatecoeffsH[0]
    cdef float_or_double coef_H_yz = updatecoeffsH[3]
    cdef float_or_double coef_H_yx = updatecoeffsH[1]

    cdef float_or_double coef_H_zt = updatecoeffsH[0]
    cdef float_or_double coef_H_zx = updatecoeffsH[1]
    cdef float_or_double coef_H_zy = updatecoeffsH[2]

    cdef float_or_double coef_H_D = updatecoeffsH[4]

    cdef float_or_double[:] Ixmyz = Ix[2, :]
    cdef float_or_double[:] Ixmzy = Ix[3, :]
    cdef float_or_double[:] Iymxz = Iy[2, :]
    cdef float_or_double[:] Iymzx = Iy[3, :]
    cdef float_or_double[:] Izmxy = Iz[2, :]
    cdef float_or_double[:] Izmyx = Iz[3, :]   

    cdef float_or_double[:] RAHx = rcHx[0, :]   
    cdef float_or_double[:] RBHx = rcHx[1, :]
    cdef float_or_double[:] RCHx = rcHx[2, :]
    cdef float_or_double[:] RDHx = rcHx[3, :]   

    cdef float_or_double[:] RAHy = rcHy[0, :]   
    cdef float_or_double[:] RBHy = rcHy[1, :]
    cdef float_or_double[:] RCHy = rcHy[2, :]
    cdef float_or_double[:] RDHy = rcHy[3, :]   

    cdef float_or_double[:] RAHz = rcHz[0, :]   
    cdef float_or_double[:] RBHz = rcHz[1, :]
    cdef float_or_double[:] RCHz = rcHz[2, :]
    cdef float_or_double[:] RDHz = rcHz[3, :]

    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef float_or_double dEzy, dEyz, dEzx, dExz, dEyx, dExy = 0.0
    cdef float_or_double mxy, mxz, myx, myz, mzx, mzy = 0.0

    cdef int idx = 0    
   
    
    for j in range(m[3], n-m[3]):  #loop to update the magnetic field at each spatial index
        H_x[j] = coef_H_xt * H_x[j] + coef_H_xz * ( E_y[j+m_z] - E_y[j] ) - coef_H_xy * ( E_z[j+m_y] - E_z[j] )     #equation 8 of Tan, Potter paper
        H_y[j] = coef_H_yt * H_y[j] + coef_H_yx * ( E_z[j+m_x] - E_z[j] ) - coef_H_yz * ( E_x[j+m_z] - E_x[j] )     #equation 8 of Tan, Potter paper
        H_z[j] = coef_H_zt * H_z[j] + coef_H_zy * ( E_x[j+m_y] - E_x[j] ) - coef_H_zx * ( E_y[j+m_x] - E_y[j] )     #equation 8 of Tan, Potter paper


    # PML regions

    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)
    
        dEzy = (E_z[idx + m_y] - E_z[idx]) / dy
        dEyz = (E_y[idx + m_z] - E_y[idx]) / dz

        mxy = RAHx[i] * dEzy + RBHx[i] * Ixmzy[i]
        mxz = RAHx[i] * dEyz + RBHx[i] * Ixmyz[i]

        H_x[idx] = H_x[idx] - coef_H_D * mxy
        H_x[idx] = H_x[idx] + coef_H_D * mxz

        Ixmzy[i] = Ixmzy[i] - RCHx[i] * mxy + RDHx[i] * dEzy
        Ixmyz[i] = Ixmyz[i] - RCHx[i] * mxz + RDHx[i] * dEyz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dEzx = (E_z[idx + m_x] - E_z[idx]) / dx
        dExz = (E_x[idx + m_z] - E_x[idx]) / dz

        myx = RAHy[i] * dEzx + RBHy[i] * Iymzx[i]
        myz = RAHy[i] * dExz + RBHy[i] * Iymxz[i]

        H_y[idx] = H_y[idx] + coef_H_D * myx
        H_y[idx] = H_y[idx] - coef_H_D * myz

        Iymzx[i] = Iymzx[i] - RCHy[i] * myx + RDHy[i] * dEzx
        Iymxz[i] = Iymxz[i] - RCHy[i] * myz + RDHy[i] * dExz



    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        # --- Hz PML update (from your MATLAB code) ---
        dEyx = (E_y[idx + m_x] - E_y[idx]) / dx
        dExy = (E_x[idx + m_y] - E_x[idx]) / dy

        mzx = RAHz[i] * dEyx + RBHz[i] * Izmyx[i]
        mzy = RAHz[i] * dExy + RBHz[i] * Izmxy[i]

        H_z[idx] = H_z[idx] - coef_H_D * mzx
        H_z[idx] = H_z[idx] + coef_H_D * mzy

        Izmyx[i] = Izmyx[i] - RCHz[i] * mzx + RDHz[i] * dEyx
        Izmxy[i] = Izmxy[i] - RCHz[i] * mzy + RDHz[i] * dExy



@cython.cdivision(True)
cdef void updateMagneticFields_axial(
    int n,
    int p,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:, ::1] rcHx,
    float_or_double[:, ::1] rcHy,
    float_or_double[:, ::1] rcHz,
    float_or_double dx,
    float_or_double dy,
    float_or_double dz,
    float_or_double[:, ::1] updatecoeffsH,
    np.uint32_t[:, ::1] GID,
    int[:] m
):
    """Updates magnetic fields for next time step using Equation 8 of
        DOI: 10.1109/LAWP.2009.2016851.

        Perform PML absorbing boundary condition updates as well according to the First order RIPML scheme.
    Args:
        n: int for spatial length of DPW array to update each length grid
            cell.
        p: int for PML thickness in number of grid cells.
        H_fields: double array of magnetic fields of DPW until temporal
                    index time.
        E_fields: double array of electric fields of DPW until temporal
                    index time.
        Ix, Iy, Iz: double arrays to store integral terms for PML regions
                    along different coordinate axes.
        rcHx, rcHy, rcHz: double arrays to store precomputed coefficients
                            for PML regions along different coordinate axes.
        dx, dy, dz: float of spatial step sizes along different coordinate axes.    
        updatecoeffsH: double array of coefficients of fields in update
                        equation for magnetic field.
        m: int array of integer mappings, m_x, m_y, m_z which determine
            rational angles for assignment of correct element to 3D FDTD
            grid from 1D representation, last element stores
            max(m_x, m_y, m_z).
    """

    cdef Py_ssize_t j, i = 0

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]
    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]

    cdef float_or_double[:] Ixmyz = Ix[2, :]
    cdef float_or_double[:] Ixmzy = Ix[3, :]
    cdef float_or_double[:] Iymxz = Iy[2, :]
    cdef float_or_double[:] Iymzx = Iy[3, :]
    cdef float_or_double[:] Izmxy = Iz[2, :]
    cdef float_or_double[:] Izmyx = Iz[3, :]   

    cdef float_or_double[:] RAHx = rcHx[0, :]   
    cdef float_or_double[:] RBHx = rcHx[1, :]
    cdef float_or_double[:] RCHx = rcHx[2, :]
    cdef float_or_double[:] RDHx = rcHx[3, :]   

    cdef float_or_double[:] RAHy = rcHy[0, :]   
    cdef float_or_double[:] RBHy = rcHy[1, :]
    cdef float_or_double[:] RCHy = rcHy[2, :]
    cdef float_or_double[:] RDHy = rcHy[3, :]   

    cdef float_or_double[:] RAHz = rcHz[0, :]   
    cdef float_or_double[:] RBHz = rcHz[1, :]
    cdef float_or_double[:] RCHz = rcHz[2, :]
    cdef float_or_double[:] RDHz = rcHz[3, :]

    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef float_or_double dEzy, dEyz, dEzx, dExz, dEyx, dExy = 0.0
    cdef float_or_double mxy, mxz, myx, myz, mzx, mzy = 0.0

    cdef int idx = 0    
  
    for j in range(m[3], n-m[3]):  #loop to update the magnetic field at each spatial index
        H_x[j] = updatecoeffsH[GID[3,j],0] * H_x[j] + updatecoeffsH[GID[3,j],3] * ( E_y[j+m_z] - E_y[j] ) - updatecoeffsH[GID[3,j],2] * ( E_z[j+m_y] - E_z[j] ) #equation 8 of Tan, Potter paper
        H_y[j] = updatecoeffsH[GID[4,j],0] * H_y[j] + updatecoeffsH[GID[4,j],1] * ( E_z[j+m_x] - E_z[j] ) - updatecoeffsH[GID[4,j],3] * ( E_x[j+m_z] - E_x[j] ) #equation 8 of Tan, Potter paper
        H_z[j] = updatecoeffsH[GID[5,j],0] * H_z[j] + updatecoeffsH[GID[5,j],2] * ( E_x[j+m_y] - E_x[j] ) - updatecoeffsH[GID[5,j],1] * ( E_y[j+m_x] - E_y[j] ) #equation 8 of Tan, Potter paper


    # PML regions

    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)
    
        dEzy = (E_z[idx + m_y] - E_z[idx]) / dy
        dEyz = (E_y[idx + m_z] - E_y[idx]) / dz

        mxy = RAHx[i] * dEzy + RBHx[i] * Ixmzy[i]
        mxz = RAHx[i] * dEyz + RBHx[i] * Ixmyz[i]

        H_x[idx] = H_x[idx] - updatecoeffsH[GID[3,idx],4] * mxy
        H_x[idx] = H_x[idx] + updatecoeffsH[GID[3,idx],4] * mxz

        Ixmzy[i] = Ixmzy[i] - RCHx[i] * mxy + RDHx[i] * dEzy
        Ixmyz[i] = Ixmyz[i] - RCHx[i] * mxz + RDHx[i] * dEyz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dEzx = (E_z[idx + m_x] - E_z[idx]) / dx
        dExz = (E_x[idx + m_z] - E_x[idx]) / dz

        myx = RAHy[i] * dEzx + RBHy[i] * Iymzx[i]
        myz = RAHy[i] * dExz + RBHy[i] * Iymxz[i]

        H_y[idx] = H_y[idx] + updatecoeffsH[GID[4,idx],4] * myx
        H_y[idx] = H_y[idx] - updatecoeffsH[GID[4,idx],4] * myz

        Iymzx[i] = Iymzx[i] - RCHy[i] * myx + RDHy[i] * dEzx
        Iymxz[i] = Iymxz[i] - RCHy[i] * myz + RDHy[i] * dExz



    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        # --- Hz PML update (from your MATLAB code) ---
        dEyx = (E_y[idx + m_x] - E_y[idx]) / dx
        dExy = (E_x[idx + m_y] - E_x[idx]) / dy

        mzx = RAHz[i] * dEyx + RBHz[i] * Izmyx[i]
        mzy = RAHz[i] * dExy + RBHz[i] * Izmxy[i]

        H_z[idx] = H_z[idx] - updatecoeffsH[GID[5,idx],4] * dx * mzx
        H_z[idx] = H_z[idx] + updatecoeffsH[GID[5,idx],4] * mzy

        Izmyx[i] = Izmyx[i] - RCHz[i] * mzx + RDHz[i] * dEyx
        Izmxy[i] = Izmxy[i] - RCHz[i] * mzy + RDHz[i] * dExy


@cython.cdivision(True)
cdef void updateElectricFields(
    int n,
    int p,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double dx,
    float_or_double dy,
    float_or_double dz,
    float_or_double[:] updatecoeffsE,
    int[:] m
):
    """Updates electric fields for next time step using Equation 9 of
        DOI: 10.1109/LAWP.2009.2016851.

        Perform PML absorbing boundary condition updates as well according to the First order RIPML scheme.

    Args:
        n: int for spatial length of DPW array to update each length grid
            cell.
        p: int for PML thickness in number of grid cells.
        H_fields: double array of magnetic fields of DPW until temporal
                    index time.
        E_fields: double array of electric fields of DPW until temporal
                    index time.
        Ix, Iy, Iz: double arrays to store integral terms for PML regions
                    along different coordinate axes.
        rcHx, rcHy, rcHz: double arrays to store precomputed coefficients
                            for PML regions along different coordinate axes.
        dx, dy, dz: float of spatial step sizes along different coordinate axes.
        updatecoeffsE: double array of coefficients of fields in update
                        equation for electric field.
        m: int array of integer mappings, m_x, m_y, m_z which determine
            rational angles for assignment of correct element to 3D FDTD
            grid from 1D representation, last element stores
            max(m_x, m_y, m_z).
    """

    cdef Py_ssize_t j, i = 0

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]
    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]

    cdef float_or_double coef_E_xt = updatecoeffsE[0]
    cdef float_or_double coef_E_xy = updatecoeffsE[2]
    cdef float_or_double coef_E_xz = updatecoeffsE[3]

    cdef float_or_double coef_E_yt = updatecoeffsE[0]
    cdef float_or_double coef_E_yz = updatecoeffsE[3]
    cdef float_or_double coef_E_yx = updatecoeffsE[1]

    cdef float_or_double coef_E_zt = updatecoeffsE[0]
    cdef float_or_double coef_E_zx = updatecoeffsE[1]
    cdef float_or_double coef_E_zy = updatecoeffsE[2]

    cdef float_or_double coef_E_D = updatecoeffsE[4]

    cdef float_or_double[:] Ixjyz = Ix[0, :]
    cdef float_or_double[:] Ixjzy = Ix[1, :]
    cdef float_or_double[:] Iyjxz = Iy[0, :]
    cdef float_or_double[:] Iyjzx = Iy[1, :]
    cdef float_or_double[:] Izjxy = Iz[0, :]
    cdef float_or_double[:] Izjyx = Iz[1, :]

    cdef float_or_double[:] RAEx = rcEx[0, :]   
    cdef float_or_double[:] RBEx = rcEx[1, :]
    cdef float_or_double[:] RCEx = rcEx[2, :]
    cdef float_or_double[:] RDEx = rcEx[3, :]   

    cdef float_or_double[:] RAEy = rcEy[0, :]   
    cdef float_or_double[:] RBEy = rcEy[1, :]
    cdef float_or_double[:] RCEy = rcEy[2, :]
    cdef float_or_double[:] RDEy = rcEy[3, :]   

    cdef float_or_double[:] RAEz = rcEz[0, :]   
    cdef float_or_double[:] RBEz = rcEz[1, :]
    cdef float_or_double[:] RCEz = rcEz[2, :]
    cdef float_or_double[:] RDEz = rcEz[3, :]

    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef float_or_double dHzy, dHyz, dHzx, dHxz, dHyx, dHxy = 0.0
    cdef float_or_double jxy, jxz, jyx, jyz, jzx, jzy = 0.0

    cdef int idx = 0    
    

    for j in range(m[3], n-m[3]):   #loop to update the electric field at each spatial index
        E_x[j] = coef_E_xt * E_x[j] + coef_E_xy * ( H_z[j] - H_z[j-m_y] ) - coef_E_xz * ( H_y[j] - H_y[j-m_z] )  #equation 9 of Tan, Potter paper
        E_y[j] = coef_E_yt * E_y[j] + coef_E_yx * ( H_x[j] - H_x[j-m_z] ) - coef_E_yx * ( H_z[j] - H_z[j-m_x] )  #equation 9 of Tan, Potter paper
        E_z[j] = coef_E_zt * E_z[j] + coef_E_zx * ( H_y[j] - H_y[j-m_x] ) - coef_E_zy * ( H_x[j] - H_x[j-m_y] )  #equation 9 of Tan, Potter paper


    # PML regions
    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHzy = (H_z[idx] - H_z[idx - m_y]) / dy
        dHyz = (H_y[idx] - H_y[idx - m_z]) / dz

        jxy = RAEx[i] * dHzy + RBEx[i] * Ixjzy[i]
        jxz = RAEx[i] * dHyz + RBEx[i] * Ixjyz[i]

        E_x[idx] = E_x[idx] + coef_E_D * jxy
        E_x[idx] = E_x[idx] - coef_E_D  * jxz

        Ixjzy[i] = Ixjzy[i] - RCEx[i] * jxy + RDEx[i] * dHzy
        Ixjyz[i] = Ixjyz[i] - RCEx[i] * jxz + RDEx[i] * dHyz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHzx = (H_z[idx] - H_z[idx - m_x]) / dx
        dHxz = (H_x[idx] - H_x[idx - m_z]) / dz

        jyx = RAEy[i] * dHzx + RBEy[i] * Iyjzx[i]
        jyz = RAEy[i] * dHxz + RBEy[i] * Iyjxz[i]

        E_y[idx] = E_y[idx] - coef_E_D  * jyx
        E_y[idx] = E_y[idx] + coef_E_D  * jyz

        Iyjzx[i] = Iyjzx[i] - RCEy[i] * jyx + RDEy[i] * dHzx
        Iyjxz[i] = Iyjxz[i] - RCEy[i] * jyz + RDEy[i] * dHxz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHyx = (H_y[idx] - H_y[idx - m_x]) / dx
        dHxy = (H_x[idx] - H_x[idx - m_y]) / dy

        jzx = RAEz[i] * dHyx + RBEz[i] * Izjyx[i]
        jzy = RAEz[i] * dHxy + RBEz[i] * Izjxy[i]

        E_z[idx] = E_z[idx] + coef_E_D  * jzx
        E_z[idx] = E_z[idx] - coef_E_D  * jzy

        Izjyx[i] = Izjyx[i] - RCEz[i] * jzx + RDEz[i] * dHyx
        Izjxy[i] = Izjxy[i] - RCEz[i] * jzy + RDEz[i] * dHxy


@cython.cdivision(True)
cdef void updateElectricFields_axial(
    int n,
    int p,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double dx,
    float_or_double dy,
    float_or_double dz,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, ::1] GID,
    int[:] m
):
    """Updates electric fields for next time step using Equation 9 of
        DOI: 10.1109/LAWP.2009.2016851.

        Perform PML absorbing boundary condition updates as well according to the First order RIPML scheme.

    Args:
        n: int for spatial length of DPW array to update each length grid
            cell.
        p: int for PML thickness in number of grid cells.
        H_fields: double array of magnetic fields of DPW until temporal
                    index time.
        E_fields: double array of electric fields of DPW until temporal
                    index time.
        Ix, Iy, Iz: double arrays to store integral terms for PML regions
                    along different coordinate axes.
        rcHx, rcHy, rcHz: double arrays to store precomputed coefficients
                            for PML regions along different coordinate axes.
        dx, dy, dz: float of spatial step sizes along different coordinate axes.
        updatecoeffsE: double array of coefficients of fields in update
                        equation for electric field.
        m: int array of integer mappings, m_x, m_y, m_z which determine
            rational angles for assignment of correct element to 3D FDTD
            grid from 1D representation, last element stores
            max(m_x, m_y, m_z).
    """

    cdef Py_ssize_t j, i = 0

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]
    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]

    cdef float_or_double[:] Ixjyz = Ix[0, :]
    cdef float_or_double[:] Ixjzy = Ix[1, :]
    cdef float_or_double[:] Iyjxz = Iy[0, :]
    cdef float_or_double[:] Iyjzx = Iy[1, :]
    cdef float_or_double[:] Izjxy = Iz[0, :]
    cdef float_or_double[:] Izjyx = Iz[1, :]

    cdef float_or_double[:] RAEx = rcEx[0, :]   
    cdef float_or_double[:] RBEx = rcEx[1, :]
    cdef float_or_double[:] RCEx = rcEx[2, :]
    cdef float_or_double[:] RDEx = rcEx[3, :]   

    cdef float_or_double[:] RAEy = rcEy[0, :]   
    cdef float_or_double[:] RBEy = rcEy[1, :]
    cdef float_or_double[:] RCEy = rcEy[2, :]
    cdef float_or_double[:] RDEy = rcEy[3, :]   

    cdef float_or_double[:] RAEz = rcEz[0, :]   
    cdef float_or_double[:] RBEz = rcEz[1, :]
    cdef float_or_double[:] RCEz = rcEz[2, :]
    cdef float_or_double[:] RDEz = rcEz[3, :]

    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef float_or_double dHzy, dHyz, dHzx, dHxz, dHyx, dHxy = 0.0
    cdef float_or_double jxy, jxz, jyx, jyz, jzx, jzy = 0.0

    cdef int idx = 0    
   

    for j in range(m[3], n-m[3]):   #loop to update the electric field at each spatial index
        E_x[j] = updatecoeffsE[GID[0,j],0] * E_x[j] + updatecoeffsE[GID[0,j],2] * ( H_z[j] - H_z[j-m_y] ) - updatecoeffsE[GID[0,j],3] * ( H_y[j] - H_y[j-m_z] )  #equation 9 of Tan, Potter paper
        E_y[j] = updatecoeffsE[GID[1,j],0] * E_y[j] + updatecoeffsE[GID[1,j],3] * ( H_x[j] - H_x[j-m_z] ) - updatecoeffsE[GID[1,j],1] * ( H_z[j] - H_z[j-m_x] )  #equation 9 of Tan, Potter paper
        E_z[j] = updatecoeffsE[GID[2,j],0] * E_z[j] + updatecoeffsE[GID[2,j],1] * ( H_y[j] - H_y[j-m_x] ) - updatecoeffsE[GID[2,j],2] * ( H_x[j] - H_x[j-m_y] )  #equation 9 of Tan, Potter paper


    # PML regions
    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHzy = (H_z[idx] - H_z[idx - m_y]) / dy
        dHyz = (H_y[idx] - H_y[idx - m_z]) / dz

        jxy = RAEx[i] * dHzy + RBEx[i] * Ixjzy[i]
        jxz = RAEx[i] * dHyz + RBEx[i] * Ixjyz[i]

        E_x[idx] = E_x[idx] + updatecoeffsE[GID[0,idx],4] * jxy
        E_x[idx] = E_x[idx] - updatecoeffsE[GID[0,idx],4] * jxz

        Ixjzy[i] = Ixjzy[i] - RCEx[i] * jxy + RDEx[i] * dHzy
        Ixjyz[i] = Ixjyz[i] - RCEx[i] * jxz + RDEx[i] * dHyz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHzx = (H_z[idx] - H_z[idx - m_x]) / dx
        dHxz = (H_x[idx] - H_x[idx - m_z]) / dz

        jyx = RAEy[i] * dHzx + RBEy[i] * Iyjzx[i]
        jyz = RAEy[i] * dHxz + RBEy[i] * Iyjxz[i]

        E_y[idx] = E_y[idx] - updatecoeffsE[GID[1,idx],4] * jyx
        E_y[idx] = E_y[idx] + updatecoeffsE[GID[1,idx],4] * jyz

        Iyjzx[i] = Iyjzx[i] - RCEy[i] * jyx + RDEy[i] * dHzx
        Iyjxz[i] = Iyjxz[i] - RCEy[i] * jyz + RDEy[i] * dHxz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHyx = (H_y[idx] - H_y[idx - m_x]) / dx
        dHxy = (H_x[idx] - H_x[idx - m_y]) / dy

        jzx = RAEz[i] * dHyx + RBEz[i] * Izjyx[i]
        jzy = RAEz[i] * dHxy + RBEz[i] * Izjxy[i]

        E_z[idx] = E_z[idx] + updatecoeffsE[GID[2,idx],4] * jzx
        E_z[idx] = E_z[idx] - updatecoeffsE[GID[2,idx],4] * jzy

        Izjyx[i] = Izjyx[i] - RCEz[i] * jzx + RDEz[i] * dHyx
        Izjxy[i] = Izjxy[i] - RCEz[i] * jzy + RDEz[i] * dHxy



@cython.cdivision(True)
cdef void updateElectricFields_dispersive(
    int n,
    int p,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Px,
    float_or_double[:, ::1] Py,
    float_or_double[:, ::1] Pz,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double dx,
    float_or_double dy,
    float_or_double dz,
    float_or_double[:] updatecoeffsE,
    float_or_double[:] updatecoeffsdispersive,
    int num_poles,
    int[:] m
):
    """Updates electric fields for next time step using Equation 9 of
        DOI: 10.1109/LAWP.2009.2016851.

        Perform PML absorbing boundary condition updates as well according to the First order RIPML scheme.

    Args:
        n: int for spatial length of DPW array to update each length grid
            cell.
        p: int for PML thickness in number of grid cells.
        H_fields: double array of magnetic fields of DPW until temporal
                    index time.
        E_fields: double array of electric fields of DPW until temporal
                    index time.
        Ix, Iy, Iz: double arrays to store integral terms for PML regions
                    along different coordinate axes.
        rcHx, rcHy, rcHz: double arrays to store precomputed coefficients
                            for PML regions along different coordinate axes.
        dx, dy, dz: float of spatial step sizes along different coordinate axes.
        updatecoeffsE: double array of coefficients of fields in update
                        equation for electric field.
        updatecoeffsdispersive: double array of coefficients for dispersive
                                material update equations.
        num_poles: int number of poles in the dispersive material model.
        m: int array of integer mappings, m_x, m_y, m_z which determine
            rational angles for assignment of correct element to 3D FDTD
            grid from 1D representation, last element stores
            max(m_x, m_y, m_z).
    """

    cdef Py_ssize_t j, i = 0

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]
    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]
    cdef float_or_double[:, ::1] T_x = Px
    cdef float_or_double[:, ::1] T_y = Py
    cdef float_or_double[:, ::1] T_z = Pz


    cdef float_or_double coef_E_xt = updatecoeffsE[0]
    cdef float_or_double coef_E_xy = updatecoeffsE[2]
    cdef float_or_double coef_E_xz = updatecoeffsE[3]

    cdef float_or_double coef_E_yt = updatecoeffsE[0]
    cdef float_or_double coef_E_yz = updatecoeffsE[3]
    cdef float_or_double coef_E_yx = updatecoeffsE[1]

    cdef float_or_double coef_E_zt = updatecoeffsE[0]
    cdef float_or_double coef_E_zx = updatecoeffsE[1]
    cdef float_or_double coef_E_zy = updatecoeffsE[2]

    cdef float_or_double coef_E_D = updatecoeffsE[4]  
    
    cdef float_or_double[:] coef_D = updatecoeffsdispersive

    cdef float_or_double[:] Ixjyz = Ix[0, :]
    cdef float_or_double[:] Ixjzy = Ix[1, :]
    cdef float_or_double[:] Iyjxz = Iy[0, :]
    cdef float_or_double[:] Iyjzx = Iy[1, :]
    cdef float_or_double[:] Izjxy = Iz[0, :]
    cdef float_or_double[:] Izjyx = Iz[1, :]

    cdef float_or_double[:] RAEx = rcEx[0, :]   
    cdef float_or_double[:] RBEx = rcEx[1, :]
    cdef float_or_double[:] RCEx = rcEx[2, :]
    cdef float_or_double[:] RDEx = rcEx[3, :]   

    cdef float_or_double[:] RAEy = rcEy[0, :]   
    cdef float_or_double[:] RBEy = rcEy[1, :]
    cdef float_or_double[:] RCEy = rcEy[2, :]
    cdef float_or_double[:] RDEy = rcEy[3, :]   

    cdef float_or_double[:] RAEz = rcEz[0, :]   
    cdef float_or_double[:] RBEz = rcEz[1, :]
    cdef float_or_double[:] RCEz = rcEz[2, :]
    cdef float_or_double[:] RDEz = rcEz[3, :]

    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef float_or_double dHzy, dHyz, dHzx, dHxz, dHyx, dHxy = 0.0
    cdef float_or_double jxy, jxz, jyx, jyz, jzx, jzy = 0.0

    cdef float_or_double phi = 0.0

    cdef int idx = 0    
   

    for j in range(m[3], n-m[3]):   #loop to update the electric field at each spatial index

        phi = 0
        for pole in range(num_poles):
            phi = phi + coef_D[pole * 3] * T_x[pole, j]

            T_x[pole, j] = coef_D[1 + (pole * 3)] * T_x[pole, j] + coef_D[2 + (pole * 3)] * E_x[j]
            
        # equation 9 of Tan, Potter paper modified for dispersive materials
        E_x[j] = coef_E_xt * E_x[j] + coef_E_xy * ( H_z[j] - H_z[j-m_y] ) - coef_E_xz * ( H_y[j] - H_y[j-m_z] ) - coef_E_D * phi 

        #T_x[pole, j] = T_x[pole, j] - coef_D[2 + (pole * 3)] * E_x[j]        
        
        phi = 0
        for pole in range(num_poles):
            phi = phi + coef_D[pole * 3] * T_y[pole, j]

            T_y[pole, j] = coef_D[1 + (pole * 3)] * T_y[pole, j] + coef_D[2 + (pole * 3)]* E_y[j]
            
        # equation 9 of Tan, Potter paper modified for dispersive materials
        E_y[j] = coef_E_yt * E_y[j] + coef_E_yz * ( H_x[j] - H_x[j-m_z] ) - coef_E_yx * ( H_z[j] - H_z[j-m_x] ) - coef_E_D * phi  

        #T_y[pole, j] = T_y[pole, j] - coef_D[2 + (pole * 3)] * E_y[j]
        
        
        phi = 0
        for pole in range(num_poles):
            phi = phi + coef_D[pole * 3] * T_z[pole, j]

            T_z[pole,j] = coef_D[1 + (pole * 3)] * T_z[pole, j] + coef_D[2 + (pole * 3)]* E_z[j]
            
        # equation 9 of Tan, Potter paper modified for dispersive materials
        E_z[j] = coef_E_zt * E_z[j] + coef_E_zx * ( H_y[j] - H_y[j-m_x] ) - coef_E_zy * ( H_x[j] - H_x[j-m_y] ) - coef_E_D * phi

        #T_z[pole, j] = T_z[pole, j] - coef_D[2 + (pole * 3)] * E_z[j]



    # PML regions
    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHzy = (H_z[idx] - H_z[idx - m_y]) / dy
        dHyz = (H_y[idx] - H_y[idx - m_z]) / dz

        jxy = RAEx[i] * dHzy + RBEx[i] * Ixjzy[i]
        jxz = RAEx[i] * dHyz + RBEx[i] * Ixjyz[i]

        E_x[idx] = E_x[idx] + coef_E_D * jxy
        E_x[idx] = E_x[idx] - coef_E_D * jxz

        Ixjzy[i] = Ixjzy[i] - RCEx[i] * jxy + RDEx[i] * dHzy
        Ixjyz[i] = Ixjyz[i] - RCEx[i] * jxz + RDEx[i] * dHyz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHzx = (H_z[idx] - H_z[idx - m_x]) / dx
        dHxz = (H_x[idx] - H_x[idx - m_z]) / dz

        jyx = RAEy[i] * dHzx + RBEy[i] * Iyjzx[i]
        jyz = RAEy[i] * dHxz + RBEy[i] * Iyjxz[i]

        E_y[idx] = E_y[idx] - coef_E_D * jyx
        E_y[idx] = E_y[idx] + coef_E_D * jyz

        Iyjzx[i] = Iyjzx[i] - RCEy[i] * jyx + RDEy[i] * dHzx
        Iyjxz[i] = Iyjxz[i] - RCEy[i] * jyz + RDEy[i] * dHxz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHyx = (H_y[idx] - H_y[idx - m_x]) / dx
        dHxy = (H_x[idx] - H_x[idx - m_y]) / dy

        jzx = RAEz[i] * dHyx + RBEz[i] * Izjyx[i]
        jzy = RAEz[i] * dHxy + RBEz[i] * Izjxy[i]

        E_z[idx] = E_z[idx] + coef_E_D * jzx
        E_z[idx] = E_z[idx] - coef_E_D * jzy

        Izjyx[i] = Izjyx[i] - RCEz[i] * jzx + RDEz[i] * dHyx
        Izjxy[i] = Izjxy[i] - RCEz[i] * jzy + RDEz[i] * dHxy


    for j in range(m[3], n-m[3]):   #loop to update the electric field at each spatial index
        for pole in range(num_poles):
            T_x[pole, j] = T_x[pole, j] - coef_D[2 + (pole * 3)] * E_x[j]  
            T_y[pole, j] = T_y[pole, j] - coef_D[2 + (pole * 3)] * E_y[j]  
            T_z[pole, j] = T_z[pole, j] - coef_D[2 + (pole * 3)] * E_z[j]


@cython.cdivision(True)
cdef void updateElectricFields_dispersive_axial(
    int n,
    int p,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Px,
    float_or_double[:, ::1] Py,
    float_or_double[:, ::1] Pz,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double dx,
    float_or_double dy,
    float_or_double dz,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, ::1] GID,
    int num_poles,
    int[:] m
):
    """Updates electric fields for next time step using Equation 9 of
        DOI: 10.1109/LAWP.2009.2016851.

        Perform PML absorbing boundary condition updates as well according to the First order RIPML scheme.

    Args:
        n: int for spatial length of DPW array to update each length grid
            cell.
        p: int for PML thickness in number of grid cells.
        H_fields: double array of magnetic fields of DPW until temporal
                    index time.
        E_fields: double array of electric fields of DPW until temporal
                    index time.
        Ix, Iy, Iz: double arrays to store integral terms for PML regions
                    along different coordinate axes.
        rcHx, rcHy, rcHz: double arrays to store precomputed coefficients
                            for PML regions along different coordinate axes.
        dx, dy, dz: float of spatial step sizes along different coordinate axes.
        updatecoeffsE: double array of coefficients of fields in update
                        equation for electric field.
        updatecoeffsdispersive: double array of coefficients for dispersive
                                material update equations.
        num_poles: int number of poles in the dispersive material model.
        m: int array of integer mappings, m_x, m_y, m_z which determine
            rational angles for assignment of correct element to 3D FDTD
            grid from 1D representation, last element stores
            max(m_x, m_y, m_z).
    """

    cdef Py_ssize_t j, i = 0

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]
    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]
    cdef float_or_double[:, ::1] T_x = Px
    cdef float_or_double[:, ::1] T_y = Py
    cdef float_or_double[:, ::1] T_z = Pz

    cdef float_or_double[:] Ixjyz = Ix[0, :]
    cdef float_or_double[:] Ixjzy = Ix[1, :]
    cdef float_or_double[:] Iyjxz = Iy[0, :]
    cdef float_or_double[:] Iyjzx = Iy[1, :]
    cdef float_or_double[:] Izjxy = Iz[0, :]
    cdef float_or_double[:] Izjyx = Iz[1, :]

    cdef float_or_double[:] RAEx = rcEx[0, :]   
    cdef float_or_double[:] RBEx = rcEx[1, :]
    cdef float_or_double[:] RCEx = rcEx[2, :]
    cdef float_or_double[:] RDEx = rcEx[3, :]   

    cdef float_or_double[:] RAEy = rcEy[0, :]   
    cdef float_or_double[:] RBEy = rcEy[1, :]
    cdef float_or_double[:] RCEy = rcEy[2, :]
    cdef float_or_double[:] RDEy = rcEy[3, :]   

    cdef float_or_double[:] RAEz = rcEz[0, :]   
    cdef float_or_double[:] RBEz = rcEz[1, :]
    cdef float_or_double[:] RCEz = rcEz[2, :]
    cdef float_or_double[:] RDEz = rcEz[3, :]

    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef float_or_double dHzy, dHyz, dHzx, dHxz, dHyx, dHxy = 0.0
    cdef float_or_double jxy, jxz, jyx, jyz, jzx, jzy = 0.0

    cdef float_or_double phi = 0.0

    cdef int idx = 0    
    cdef int mat = 0
   

    for j in range(m[3], n-m[3]):   #loop to update the electric field at each spatial index
        
        mat = GID[0,j]
        phi = 0
        for pole in range(num_poles):
            phi = phi + updatecoeffsdispersive[mat,pole * 3] * T_x[pole, j]
            T_x[pole, j] = updatecoeffsdispersive[mat,1 + (pole * 3)] * T_x[pole, j] + updatecoeffsdispersive[mat,2 + (pole * 3)] * E_x[j]
            
        # equation 9 of Tan, Potter paper modified for dispersive materials
        E_x[j] = updatecoeffsE[mat,0] * E_x[j] + updatecoeffsE[mat,2] * ( H_z[j] - H_z[j-m_y] ) - updatecoeffsE[mat,3] * ( H_y[j] - H_y[j-m_z] ) - updatecoeffsE[mat,4] * phi 
 
        mat=GID[1,j]
        phi = 0
        for pole in range(num_poles):
            phi = phi + updatecoeffsdispersive[mat,pole * 3] * T_y[pole, j]
            T_y[pole, j] = updatecoeffsdispersive[mat,1 + (pole * 3)] * T_y[pole, j] + updatecoeffsdispersive[mat,2 + (pole * 3)]* E_y[j]
            
        # equation 9 of Tan, Potter paper modified for dispersive materials
        E_y[j] = updatecoeffsE[mat,0] * E_y[j] + updatecoeffsE[mat,3] * ( H_x[j] - H_x[j-m_z] ) - updatecoeffsE[mat,1] * ( H_z[j] - H_z[j-m_x] ) - updatecoeffsE[mat,4] * phi  

        mat=GID[2,j]       
        phi = 0
        for pole in range(num_poles):
            phi = phi + updatecoeffsdispersive[mat,pole * 3] * T_z[pole, j]
            T_z[pole,j] = updatecoeffsdispersive[mat, 1 + (pole * 3)] * T_z[pole, j] + updatecoeffsdispersive[mat,2 + (pole * 3)]* E_z[j]
            
        # equation 9 of Tan, Potter paper modified for dispersive materials
        E_z[j] = updatecoeffsE[mat,0] * E_z[j] + updatecoeffsE[mat,1] * ( H_y[j] - H_y[j-m_x] ) - updatecoeffsE[mat,2] * ( H_x[j] - H_x[j-m_y] ) - updatecoeffsE[mat,4] * phi

     
    # PML regions
    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHzy = (H_z[idx] - H_z[idx - m_y]) / dy
        dHyz = (H_y[idx] - H_y[idx - m_z]) / dz

        jxy = RAEx[i] * dHzy + RBEx[i] * Ixjzy[i]
        jxz = RAEx[i] * dHyz + RBEx[i] * Ixjyz[i]

        E_x[idx] = E_x[idx] + updatecoeffsE[GID[0,idx],4] * jxy
        E_x[idx] = E_x[idx] - updatecoeffsE[GID[0,idx],4] * jxz

        Ixjzy[i] = Ixjzy[i] - RCEx[i] * jxy + RDEx[i] * dHzy
        Ixjyz[i] = Ixjyz[i] - RCEx[i] * jxz + RDEx[i] * dHyz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHzx = (H_z[idx] - H_z[idx - m_x]) / dx
        dHxz = (H_x[idx] - H_x[idx - m_z]) / dz

        jyx = RAEy[i] * dHzx + RBEy[i] * Iyjzx[i]
        jyz = RAEy[i] * dHxz + RBEy[i] * Iyjxz[i]

        E_y[idx] = E_y[idx] - updatecoeffsE[GID[1,idx],4] * jyx
        E_y[idx] = E_y[idx] + updatecoeffsE[GID[1,idx],4] * jyz

        Iyjzx[i] = Iyjzx[i] - RCEy[i] * jyx + RDEy[i] * dHzx
        Iyjxz[i] = Iyjxz[i] - RCEy[i] * jyz + RDEy[i] * dHxz


    for i in range(p-1, -1, -1):  # MATLAB's PMLSize:-1:1
        idx = (n-m[3]) - (p - i)

        dHyx = (H_y[idx] - H_y[idx - m_x]) / dx
        dHxy = (H_x[idx] - H_x[idx - m_y]) / dy

        jzx = RAEz[i] * dHyx + RBEz[i] * Izjyx[i]
        jzy = RAEz[i] * dHxy + RBEz[i] * Izjxy[i]

        E_z[idx] = E_z[idx] + updatecoeffsE[GID[2,idx],4] * jzx
        E_z[idx] = E_z[idx] - updatecoeffsE[GID[2,idx],4] * jzy

        Izjyx[i] = Izjyx[i] - RCEz[i] * jzx + RDEz[i] * dHyx
        Izjxy[i] = Izjxy[i] - RCEz[i] * jzy + RDEz[i] * dHxy


    for j in range(m[3], n-m[3]):   #loop to update the electric field at each spatial index
        for pole in range(num_poles):
            T_x[pole, j] = T_x[pole, j] - updatecoeffsdispersive[GID[0,j],2 + (pole * 3)] * E_x[j]  
            T_y[pole, j] = T_y[pole, j] - updatecoeffsdispersive[GID[1,j],2 + (pole * 3)] * E_y[j]  
            T_z[pole, j] = T_z[pole, j] - updatecoeffsdispersive[GID[2,j],2 + (pole * 3)] * E_z[j]




@cython.cdivision(True)
cpdef double getSource(
    double time,
    double freq,
    char* wavetype,
    double dt    
):
    """Gets magnitude of source field in direction perpendicular to propagation
        of plane wave.

    Args:
        time: float of time at which magnitude of source is calculated.
        freq: float of frequency of introduced wave which determines grid points
                per wavelength for wave source.
        wavetype: string of type of waveform whose magnitude should be returned.
        dt: double of time upto which wave should exist in a impulse delta pulse.

    Returns:
        sourceMagnitude: double of magnitude of source for requested indices at
                            current time.
    """

    cdef double chi = 0.0
    cdef double zeta = 0.0
    cdef double delay = 0.0 
    cdef double normalise = 1.0    

    # Waveforms
    if (strcmp(wavetype, "gaussian") == 0):
        #return exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))
        chi = 1 / freq
        zeta = 2 * pow(M_PI,2) * pow(freq,2)
        delay = time - chi
        return exp(-zeta * pow(delay,2))

    elif (strcmp(wavetype, "gaussiandot") == 0 or strcmp(wavetype, "gaussianprime") == 0):
        #return -4.0 * M_PI * M_PI * freq * (time * freq - 1.0
        #        ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))   
        chi = 1 / freq
        zeta = 2 * pow(M_PI,2) * pow(freq,2)
        delay = time - chi
        return -2.0 * zeta * delay * exp(-zeta * pow(delay,2))      

    elif (strcmp(wavetype, "gaussiandotnorm") == 0):
        #return -2.0 * M_PI * (time * freq - 1.0
        #        ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0))) * exp(0.5)
        chi = 1 / freq
        zeta = 2 * pow(M_PI,2) * pow(freq,2)
        delay = time - chi
        normalise = sqrt(exp(1) / (2.0 * zeta))
        return -(2.0 * zeta * delay * exp(-zeta * pow(delay,2)) * normalise)

    elif (strcmp(wavetype, "gaussiandotdot") == 0 or strcmp(wavetype, "gaussiandoubleprime") == 0):
       # return (2.0 * M_PI * freq) * (2.0 * M_PI * freq) * (2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)) - 1.0
       #         ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))
        chi = sqrt(2) / freq
        zeta = pow(M_PI,2) * pow(freq,2)
        delay = time - chi
        return (2 * zeta * (2 * zeta * pow(delay,2) - 1) * exp(-zeta * pow(delay,2)))

    elif (strcmp(wavetype, "gaussiandotdotnorm") == 0):
       # return (2.0 * (M_PI *(time * freq - 1.0)) * (M_PI * (time * freq - 1.0)) - 1.0
       #         ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))
        chi = sqrt(2) / freq
        zeta = pow(M_PI,2) * pow(freq,2)
        delay = time - chi
        normalise = 1.0 / (2.0 * zeta)
        return (2.0 * zeta * (2.0 * zeta * pow(delay,2) - 1) * exp(-zeta * pow(delay,2)) * normalise)

    elif (strcmp(wavetype, "ricker") == 0):
       # return (1.0 - 2.0 * (M_PI *(time * freq - 1.0)) * (M_PI * (time * freq - 1.0))
       #         ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))  # define a Ricker wave source
        chi = sqrt(2) / freq
        zeta = pow(M_PI,2) * pow(freq,2)
        delay = time - chi
        normalise = 1.0 / (2.0 * zeta)
        return -(2 * zeta * (2 * zeta * pow(delay,2) - 1) * exp(-zeta * pow(delay,2)) * normalise)

    elif (strcmp(wavetype, "sine") == 0):
        if (time * freq <= 1):
            return sin(2.0 * M_PI * freq * time)
        else:
            return 0.0

    elif (strcmp(wavetype, "contsine") == 0):
        return min(0.25 * time* freq, 1) * sin(2 * M_PI * time* freq)

    elif (strcmp(wavetype, "impulse") == 0):
        if (time < dt):                         # time < dt condition required to do impulsive magnetic dipole
            return 1.0
        else:
            return 0.0


@cython.cdivision(True)
cpdef void calculate1DWaveformValues(
    float_or_double[:, :, ::1] waveformvalues_wholedt,
    float_or_double[:, :, ::1] waveformvalues_halfdt,
    int iterations,
    int[:] m,
    double dt,
    double ds,
    double c,
    double start,
    double stop,
    double freq,
    char* wavetype
):
    """Precomputes source waveform values so that initialization is faster,
        if requested.

    Args:
        waveformvalues_wholedt: double array of precomputed waveforms at each
                                    timestep to initialize magnetic fields.
        iterations: int of number of iterations in simulation.
        m: int array of integer mappings, m_x, m_y, m_z which determine rational
            angles for assignment of correct element to 3D FDTD grid from 1D
            representation, last element stores max(m_x, m_y, m_z).
        dt: float of timestep for the simulation.
        ds: float of projection vector for sourcing the plane wave.
        c: float of speed of light in the medium.
        start: float of start time at which source is placed in the TFSF grid.
        stop: float of stop time at which source is removed from TFSF grid.
        freq: float of frequency of introduced wave which determines grid points
                per wavelength for wave source.
        wavetype: string of type of waveform whose magnitude should be returned.
    """

    cdef double time1_x, time1_y, time1_z = 0.0
    cdef double time2_x, time2_y, time2_z = 0.0
    cdef Py_ssize_t iteration, r = 0

    for iteration in range(iterations):
        for r in range(m[3]):
            time1_x = dt * iteration - (r + (abs(m[1])+abs(m[2]))*0.5) * ds/c
            time1_y = dt * iteration - (r + (abs(m[2])+abs(m[0]))*0.5) * ds/c
            time1_z = dt * iteration - (r + (abs(m[0])+abs(m[1]))*0.5) * ds/c
            if (dt * iteration >= start and dt * iteration <= stop):
            # Set the time of the waveform evaluation to account for any delay in the start
                waveformvalues_wholedt[iteration, 0, r] = getSource(time1_x-start, freq, wavetype, dt)
                waveformvalues_wholedt[iteration, 1, r] = getSource(time1_y-start, freq, wavetype, dt)
                waveformvalues_wholedt[iteration, 2, r] = getSource(time1_z-start, freq, wavetype, dt)

        for r in range(m[3]):
            time2_x = dt * (iteration + 0.5) - (r + abs(m[0])*0.5) * ds/c
            time2_y = dt * (iteration + 0.5) - (r + abs(m[1])*0.5) * ds/c
            time2_z = dt * (iteration + 0.5) - (r + abs(m[2])*0.5) * ds/c
            if (dt * (iteration + 0.5) >= start and dt * (iteration + 0.5) <= stop):
            # Set the time of the waveform evaluation to account for any delay in the start
                waveformvalues_halfdt[iteration, 0, r] = getSource(time2_x-start, freq, wavetype, dt)
                waveformvalues_halfdt[iteration, 1, r] = getSource(time2_y-start, freq, wavetype, dt)
                waveformvalues_halfdt[iteration, 2, r] = getSource(time2_z-start, freq, wavetype, dt)


cpdef void updatePlaneWave_magnetic(
    int n,
    int p,
    int nthreads,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:] updatecoeffsE,
    float_or_double[:] updatecoeffsH,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double[:, ::1] rcHx,
    float_or_double[:, ::1] rcHy,
    float_or_double[:, ::1] rcHz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
    double[:] projections,
    float_or_double[:, :, ::1] waveformvalues_wholedt,
    float_or_double[:, :, ::1] waveformvalues_halfdt,
    int[:] m,
    int[:] origin,
    int[:] corners,
    bint precompute,
    int iteration,
    double dt,
    double dx,
    double dy,
    double dz,
    double ds,
    double c,
    double start,
    double stop,
    double freq,
    char* wavetype
):
    initializeMagneticFields(m, H_fields, projections, waveformvalues_wholedt, precompute, iteration, dt, ds, c, start, stop, freq, wavetype)
    updateMagneticFields(n, p, H_fields, E_fields, Ix, Iy, Iz, rcHx, rcHy, rcHz, dx, dy, dz, updatecoeffsH, m)
    applyTFSFMagnetic(nthreads, Hx, Hy, Hz, E_fields, updatecoeffsH, m, origin, corners)
    

cpdef void updatePlaneWave_magnetic_axial(
    int n,
    int p,
    int nthreads,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double[:, ::1] updatecoeffsH,
    np.uint32_t[:, ::1] ID,
    np.uint32_t[:, :, :, ::1] GID,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double[:, ::1] rcHx,
    float_or_double[:, ::1] rcHy,
    float_or_double[:, ::1] rcHz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
    double[:] projections,
    float_or_double[:, :, ::1] waveformvalues_wholedt,
    float_or_double[:, :, ::1] waveformvalues_halfdt,
    int[:] m,
    int[:] origin,
    int[:] corners,
    bint precompute,
    int iteration,
    double dt,
    double dx,
    double dy,
    double dz,
    double ds,
    double c,
    double start,
    double stop,
    double freq,
    char* wavetype
):
    
        initializeMagneticFields(m, H_fields, projections, waveformvalues_wholedt, precompute, iteration, dt, ds, c, start, stop, freq, wavetype)
        updateMagneticFields_axial(n, p, H_fields, E_fields, Ix, Iy, Iz, rcHx, rcHy, rcHz, dx, dy, dz, updatecoeffsH, ID, m)
        applyTFSFMagnetic_axial(nthreads, Hx, Hy, Hz, E_fields, updatecoeffsH, GID, m, origin, corners)
    
  

cpdef void updatePlaneWave_electric(
    int n,
    int p,
    int nthreads,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:] updatecoeffsE,
    float_or_double[:] updatecoeffsH,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double[:, ::1] rcHx,
    float_or_double[:, ::1] rcHy,
    float_or_double[:, ::1] rcHz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
    double[:] projections,
    float_or_double[:, :, ::1] waveformvalues_wholedt,
    float_or_double[:, :, ::1] waveformvalues_halfdt,
    int[:] m,
    int[:] origin,
    int[:] corners,
    bint precompute,
    int iteration,
    double dt,
    double dx,
    double dy,
    double dz,
    double ds,
    double c,
    double start,
    double stop,
    double freq,
    char* wavetype
):
    initializeElectricFields(m, E_fields, projections, waveformvalues_halfdt, precompute, iteration, dt, ds, c, start, stop, freq, wavetype)
    updateElectricFields(n, p, H_fields, E_fields, Ix, Iy, Iz, rcEx, rcEy, rcEz, dx, dy, dz, updatecoeffsE, m)
    applyTFSFElectric(nthreads, Ex, Ey, Ez, H_fields, updatecoeffsE, m, origin, corners)


cpdef void updatePlaneWave_electric_axial(
    int n,
    int p,
    int nthreads,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double[:, ::1] updatecoeffsH,
    np.uint32_t[:, ::1] ID,
    np.uint32_t[:, :, :, ::1] GID,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double[:, ::1] rcHx,
    float_or_double[:, ::1] rcHy,
    float_or_double[:, ::1] rcHz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
    double[:] projections,
    float_or_double[:, :, ::1] waveformvalues_wholedt,
    float_or_double[:, :, ::1] waveformvalues_halfdt,
    int[:] m,
    int[:] origin,
    int[:] corners,
    bint precompute,
    int iteration,
    double dt,
    double dx,
    double dy,
    double dz,
    double ds,
    double c,
    double start,
    double stop,
    double freq,
    char* wavetype
):

        initializeElectricFields(m, E_fields, projections, waveformvalues_halfdt, precompute, iteration, dt, ds, c, start, stop, freq, wavetype)
        updateElectricFields_axial(n, p, H_fields, E_fields, Ix, Iy, Iz, rcEx, rcEy, rcEz, dx, dy, dz, updatecoeffsE, ID, m)
        applyTFSFElectric_axial(nthreads, Ex, Ey, Ez, H_fields, updatecoeffsE, GID, m, origin, corners)




cpdef void updatePlaneWave_electric_dispersive(
    int n,
    int p,
    int nthreads,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Px,
    float_or_double[:, ::1] Py,
    float_or_double[:, ::1] Pz,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:] updatecoeffsE,
    float_or_double[:] updatecoeffsH,
    float_or_double[:] updatecoeffsdispersive,
    int num_poles,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double[:, ::1] rcHx,
    float_or_double[:, ::1] rcHy,
    float_or_double[:, ::1] rcHz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
    double[:] projections,
    float_or_double[:, :, ::1] waveformvalues_wholedt,
    float_or_double[:, :, ::1] waveformvalues_halfdt,
    int[:] m,
    int[:] origin,
    int[:] corners,
    bint precompute,
    int iteration,
    double dt,
    double dx,
    double dy,
    double dz,
    double ds,
    double c,
    double start,
    double stop,
    double freq,
    char* wavetype
):  
    initializeElectricFields(m, E_fields, projections, waveformvalues_halfdt, precompute, iteration, dt, ds, c, start, stop, freq, wavetype)
    updateElectricFields_dispersive(n, p, H_fields, E_fields, Px, Py, Pz, Ix, Iy, Iz, rcEx, rcEy, rcEz, dx, dy, dz, updatecoeffsE, updatecoeffsdispersive, num_poles, m)
    applyTFSFElectric(nthreads, Ex, Ey, Ez, H_fields, updatecoeffsE, m, origin, corners)


cpdef void updatePlaneWave_electric_dispersive_axial(
    int n,
    int p,
    int nthreads,
    float_or_double[:, ::1] H_fields,
    float_or_double[:, ::1] E_fields,
    float_or_double[:, ::1] Px,
    float_or_double[:, ::1] Py,
    float_or_double[:, ::1] Pz,
    float_or_double[:, ::1] Ix,
    float_or_double[:, ::1] Iy,
    float_or_double[:, ::1] Iz,
    float_or_double[:, ::1] updatecoeffsE,
    float_or_double[:, ::1] updatecoeffsH,
    float_or_double[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, ::1] ID,
    np.uint32_t[:, :, :, ::1] GID,
    int num_poles,
    float_or_double[:, ::1] rcEx,
    float_or_double[:, ::1] rcEy,
    float_or_double[:, ::1] rcEz,
    float_or_double[:, ::1] rcHx,
    float_or_double[:, ::1] rcHy,
    float_or_double[:, ::1] rcHz,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz,
    double[:] projections,
    float_or_double[:, :, ::1] waveformvalues_wholedt,
    float_or_double[:, :, ::1] waveformvalues_halfdt,
    int[:] m,
    int[:] origin,
    int[:] corners,
    bint precompute,
    int iteration,
    double dt,
    double dx,
    double dy,
    double dz,
    double ds,
    double c,
    double start,
    double stop,
    double freq,
    char* wavetype
):  
    initializeElectricFields(m, E_fields, projections, waveformvalues_halfdt, precompute, iteration, dt, ds, c, start, stop, freq, wavetype)
    updateElectricFields_dispersive_axial(n, p, H_fields, E_fields, Px, Py, Pz, Ix, Iy, Iz, rcEx, rcEy, rcEz, dx, dy, dz, updatecoeffsE, updatecoeffsdispersive, ID, num_poles, m)
    applyTFSFElectric_axial(nthreads, Ex, Ey, Ez, H_fields, updatecoeffsE, GID, m, origin, corners)




@cython.cdivision(True)
cdef void takeSnapshot3D(double[:, :, ::1] field, char* filename):
    """Writes fields of plane wave simulation at a particular time step.
    ONLY USED FOR QUICKLY TESTING THE DPW IMPLEMENTATION

    Args:
        fields: double array of fields for grid cells over TFSF box at
                particular indices of TFSF box at particular time step.
        filename: string of file location where fields are to be written.

    """

    cdef FILE *fptr = fopen(filename, "wb")
    fwrite(&field[0, 0, 0], sizeof(double), field.size, fptr)
    fclose(fptr)
