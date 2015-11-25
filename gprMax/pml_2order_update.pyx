# Copyright (C) 2015: The University of Edinburgh
#            Authors: Craig Warren and Antonis Giannopoulos
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


#############################################
# Electric field PML updates - Ex component #
#############################################
cpdef update_pml_2order_ex_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ex, floattype_t[:, :, :] Hz, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dy):
    """This function updates the Ex field components in the y stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHz
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[0, i + xs, j + ys, k + zs]
                dHz = (Hz[i + xs, j + ys, k + zs] - Hz[i + xs, j - 1 + ys, k + zs]) / dy
                Ex[i + xs, j + ys, k + zs] = Ex[i + xs, j + ys, k + zs] + updatecoeffsE[listIndex, 4] * ((RA[0, j] * RA[1, j] - 1) * dHz + RA[1, j] * RB[0, j] * EPhi[0, i, j, k] + RB[1, j] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, j] * EPhi[1, i, j, k] - RF[1, j] * (RA[0, j] * dHz + RB[0, j] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, j] * EPhi[0, i, j, k] - RF[0, j] * dHz


cpdef update_pml_2order_ex_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ex, floattype_t[:, :, :] Hz, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dy):
    """This function updates the Ex field components in the y stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHz
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[0, i + xs, yf - j, k + zs]
                dHz = (Hz[i + xs, yf - j, k + zs] - Hz[i + xs, yf - j - 1, k + zs]) / dy
                Ex[i + xs, yf - j, k + zs] = Ex[i + xs, yf - j, k + zs] + updatecoeffsE[listIndex, 4] * ((RA[0, j] * RA[1, j] - 1) * dHz + RA[1, j] * RB[0, j] * EPhi[0, i, j, k] + RB[1, j] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, j] * EPhi[1, i, j, k] - RF[1, j] * (RA[0, j] * dHz + RB[0, j] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, j] * EPhi[0, i, j, k] - RF[0, j] * dHz


cpdef update_pml_2order_ex_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ex, floattype_t[:, :, :] Hy, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dz):
    """This function updates the Ex field components in the z stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHy
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[0, i + xs, j + ys, k + zs]
                dHy = (Hy[i + xs, j + ys, k + zs] - Hy[i + xs, j + ys, k - 1 + zs]) / dz
                Ex[i + xs, j + ys, k + zs] = Ex[i + xs, j + ys, k + zs] - updatecoeffsE[listIndex, 4] * ((RA[0, k] * RA[1, k] - 1) * dHy + RA[1, k] * RB[0, k] * EPhi[0, i, j, k] + RB[1, k] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, k] * EPhi[1, i, j, k] - RF[1, k] * (RA[0, k] * dHy + RB[0, k] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, k] * EPhi[0, i, j, k] - RF[0, k] * dHy


cpdef update_pml_2order_ex_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ex, floattype_t[:, :, :] Hy, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dz):
    """This function updates the Ex field components in the z stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHy
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[0, i + xs, j + ys, zf - k]
                dHy = (Hy[i + xs, j + ys, zf - k] - Hy[i + xs, j + ys, zf - k - 1]) / dz
                Ex[i + xs, j + ys, zf - k] = Ex[i + xs, j + ys, zf - k] - updatecoeffsE[listIndex, 4] * ((RA[0, k] * RA[1, k] - 1) * dHy + RA[1, k] * RB[0, k] * EPhi[0, i, j, k] + RB[1, k] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, k] * EPhi[1, i, j, k] - RF[1, k] * (RA[0, k] * dHy + RB[0, k] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, k] * EPhi[0, i, j, k] - RF[0, k] * dHy


#############################################
# Electric field PML updates - Ey component #
#############################################
cpdef update_pml_2order_ey_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ey, floattype_t[:, :, :] Hz, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dx):
    """This function updates the Ey field components in the x stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHz
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[1, i + xs, j + ys, k + zs]
                dHz = (Hz[i + xs, j + ys, k + zs] - Hz[i - 1 + xs, j + ys, k + zs]) / dx
                Ey[i + xs, j + ys, k + zs] = Ey[i + xs, j + ys, k + zs] - updatecoeffsE[listIndex, 4] * ((RA[0, i] * RA[1, i] - 1) * dHz + RA[1, i] * RB[0, i] * EPhi[0, i, j, k] + RB[1, i] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, i] * EPhi[1, i, j, k] - RF[1, i] * (RA[0, i] * dHz + RB[0, i] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, i] * EPhi[0, i, j, k] - RF[0, i] * dHz


cpdef update_pml_2order_ey_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ey, floattype_t[:, :, :] Hz, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dx):
    """This function updates the Ey field components in the x stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHz
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[1, xf - i, j + ys, k + zs]
                dHz = (Hz[xf - i, j + ys, k + zs] - Hz[xf - i - 1, j + ys, k + zs]) / dx
                Ey[xf - i, j + ys, k + zs] = Ey[xf - i, j + ys, k + zs] - updatecoeffsE[listIndex, 4] * ((RA[0, i] * RA[1, i] - 1) * dHz + RA[1, i] * RB[0, i] * EPhi[0, i, j, k] + RB[1, i] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, i] * EPhi[1, i, j, k] - RF[1, i] * (RA[0, i] * dHz + RB[0, i] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, i] * EPhi[0, i, j, k] - RF[0, i] * dHz


cpdef update_pml_2order_ey_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ey, floattype_t[:, :, :] Hx, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dz):
    """This function updates the Ey field components in the z stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHx
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[1, i + xs, j + ys, k + zs]
                dHx = (Hx[i + xs, j + ys, k + zs] - Hx[i + xs, j + ys, k - 1 + zs]) / dz
                Ey[i + xs, j + ys, k + zs] = Ey[i + xs, j + ys, k + zs] + updatecoeffsE[listIndex, 4] * ((RA[0, k] * RA[1, k] - 1) * dHx + RA[1, k] * RB[0, k] * EPhi[0, i, j, k] + RB[1, k] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, k] * EPhi[1, i, j, k] - RF[1, k] * (RA[0, k] * dHx + RB[0, k] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, k] * EPhi[0, i, j, k] - RF[0, k] * dHx


cpdef update_pml_2order_ey_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ey, floattype_t[:, :, :] Hx, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dz):
    """This function updates the Ey field components in the z stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHx
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[1, i + xs, j + ys, zf - k]
                dHx = (Hx[i + xs, j + ys, zf - k] - Hx[i + xs, j + ys, zf - k - 1]) / dz
                Ey[i + xs, j + ys, zf - k] = Ey[i + xs, j + ys, zf - k] + updatecoeffsE[listIndex, 4] * ((RA[0, k] * RA[1, k] - 1) * dHx + RA[1, k] * RB[0, k] * EPhi[0, i, j, k] + RB[1, k] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, k] * EPhi[1, i, j, k] - RF[1, k] * (RA[0, k] * dHx + RB[0, k] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, k] * EPhi[0, i, j, k] - RF[0, k] * dHx


#############################################
# Electric field PML updates - Ez component #
#############################################
cpdef update_pml_2order_ez_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ez, floattype_t[:, :, :] Hy, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dx):
    """This function updates the Ey field components in the z stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHy
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[2, i + xs, j + ys, k + zs]
                dHy = (Hy[i + xs, j + ys, k + zs] - Hy[i - 1 + xs, j + ys, k + zs]) / dx
                Ez[i + xs, j + ys, k + zs] = Ez[i + xs, j + ys, k + zs] + updatecoeffsE[listIndex, 4] * ((RA[0, i] * RA[1, i] - 1) * dHy + RA[1, i] * RB[0, i] * EPhi[0, i, j, k] + RB[1, i] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, i] * EPhi[1, i, j, k] - RF[1, i] * (RA[0, i] * dHy + RB[0, i] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, i] * EPhi[0, i, j, k] - RF[0, i] * dHy


cpdef update_pml_2order_ez_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ez, floattype_t[:, :, :] Hy, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dx):
    """This function updates the Ez field components in the x stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHy
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[2, xf - i, j + ys, k + zs]
                dHy = (Hy[xf - i, j + ys, k + zs] - Hy[xf - i - 1, j + ys, k + zs]) / dx
                Ez[xf - i, j + ys, k + zs] = Ez[xf - i, j + ys, k + zs] + updatecoeffsE[listIndex, 4] * ((RA[0, i] * RA[1, i] - 1) * dHy + RA[1, i] * RB[0, i] * EPhi[0, i, j, k] + RB[1, i] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, i] * EPhi[1, i, j, k] - RF[1, i] * (RA[0, i] * dHy + RB[0, i] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, i] * EPhi[0, i, j, k] - RF[0, i] * dHy


cpdef update_pml_2order_ez_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ez, floattype_t[:, :, :] Hx, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dy):
    """This function updates the Ez field components in the y stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHx
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[2, i + xs, j + ys, k + zs]
                dHx = (Hx[i + xs, j + ys, k + zs] - Hx[i + xs, j - 1 + ys, k + zs]) / dy
                Ez[i + xs, j + ys, k + zs] = Ez[i + xs, j + ys, k + zs] - updatecoeffsE[listIndex, 4] * ((RA[0, j] * RA[1, j] - 1) * dHx + RA[1, j] * RB[0, j] * EPhi[0, i, j, k] + RB[1, j] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, j] * EPhi[1, i, j, k] - RF[1, j] * (RA[0, j] * dHx + RB[0, j] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, j] * EPhi[0, i, j, k] - RF[0, j] * dHx


cpdef update_pml_2order_ez_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ez, floattype_t[:, :, :] Hx, floattype_t[:, :, :, :] EPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dy):
    """This function updates the Ez field components in the y stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dHx
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[2, i + xs, yf - j, k + zs]
                dHx = (Hx[i + xs, yf - j, k + zs] - Hx[i + xs, yf - j - 1, k + zs]) / dy
                Ez[i + xs, yf - j, k + zs] = Ez[i + xs, yf - j, k + zs] - updatecoeffsE[listIndex, 4] * ((RA[0, j] * RA[1, j] - 1) * dHx + RA[1, j] * RB[0, j] * EPhi[0, i, j, k] + RB[1, j] * EPhi[1, i, j, k])
                EPhi[1, i, j, k] = RE[1, j] * EPhi[1, i, j, k] - RF[1, j] * (RA[0, j] * dHx + RB[0, j] * EPhi[0, i, j, k])
                EPhi[0, i, j, k] = RE[0, j] * EPhi[0, i, j, k] - RF[0, j] * dHx


#############################################
# Magnetic field PML updates - Hx component #
#############################################
cpdef update_pml_2order_hx_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Ez, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dy):
    """This function updates the Hx field components in the y stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEz
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[3, i + xs, j + ys, k + zs]
                dEz = (Ez[i + xs, j + 1 + ys, k + zs] - Ez[i + xs, j + ys, k + zs]) / dy
                Hx[i + xs, j + ys, k + zs] = Hx[i + xs, j + ys, k + zs] - updatecoeffsH[listIndex, 4] * ((RA[0, j] * RA[1, j] - 1) * dEz + RA[1, j] * RB[0, j] * HPhi[0, i, j, k] + RB[1, j] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, j] * HPhi[1, i, j, k] - RF[1, j] * (RA[0, j] * dEz + RB[0, j] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, j] * HPhi[0, i, j, k] - RF[0, j] * dEz


cpdef update_pml_2order_hx_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Ez, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dy):
    """This function updates the Hx field components in the y stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEz
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[3, i + xs, yf - (j + 1), k + zs]
                dEz = (Ez[i + xs, yf - j, k + zs] - Ez[i + xs, yf - (j + 1), k + zs]) / dy
                Hx[i + xs, yf - (j + 1), k + zs] = Hx[i + xs, yf - (j + 1), k + zs] - updatecoeffsH[listIndex, 4] * ((RA[0, j] * RA[1, j] - 1) * dEz + RA[1, j] * RB[0, j] * HPhi[0, i, j, k] + RB[1, j] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, j] * HPhi[1, i, j, k] - RF[1, j] * (RA[0, j] * dEz + RB[0, j] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, j] * HPhi[0, i, j, k] - RF[0, j] * dEz


cpdef update_pml_2order_hx_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Ey, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dz):
    """This function updates the Hx field components in the z stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEy
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[3, i + xs, j + ys, k + zs]
                dEy = (Ey[i + xs, j + ys, k + 1 + zs] - Ey[i + xs, j + ys, k + zs]) / dz
                Hx[i + xs, j + ys, k + zs] = Hx[i + xs, j + ys, k + zs] + updatecoeffsH[listIndex, 4] * ((RA[0, k] * RA[1, k] - 1) * dEy + RA[1, k] * RB[0, k] * HPhi[0, i, j, k] + RB[1, k] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, k] * HPhi[1, i, j, k] - RF[1, k] * (RA[0, k] * dEy + RB[0, k] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, k] * HPhi[0, i, j, k] - RF[0, k] * dEy


cpdef update_pml_2order_hx_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Ey, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dz):
    """This function updates the Hx field components in the z stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEy
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[3, i + xs, j + ys, zf - (k + 1)]
                dEy = (Ey[i + xs, j + ys, zf - k] - Ey[i + xs, j + ys, zf - (k + 1)]) / dz
                Hx[i + xs, j + ys, zf - (k + 1)] = Hx[i + xs, j + ys, zf - (k + 1)] + updatecoeffsH[listIndex, 4] * ((RA[0, k] * RA[1, k] - 1) * dEy + RA[1, k] * RB[0, k] * HPhi[0, i, j, k] + RB[1, k] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, k] * HPhi[1, i, j, k] - RF[1, k] * (RA[0, k] * dEy + RB[0, k] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, k] * HPhi[0, i, j, k] - RF[0, k] * dEy


#############################################
# Magnetic field PML updates - Hy component #
#############################################
cpdef update_pml_2order_hy_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hy, floattype_t[:, :, :] Ez, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dx):
    """This function updates the Hy field components in the x stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEz
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[4, i + xs, j + ys, k + zs]
                dEz = (Ez[i + 1 + xs, j + ys, k + zs] - Ez[i + xs, j + ys, k + zs]) / dx
                Hy[i + xs, j + ys, k + zs] = Hy[i + xs, j + ys, k + zs] + updatecoeffsH[listIndex, 4] * ((RA[0, i] * RA[1, i] - 1) * dEz + RA[1, i] * RB[0, i] * HPhi[0, i, j, k] + RB[1, i] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, i] * HPhi[1, i, j, k] - RF[1, i] * (RA[0, i] * dEz + RB[0, i] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, i] * HPhi[0, i, j, k] - RF[0, i] * dEz


cpdef update_pml_2order_hy_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hy, floattype_t[:, :, :] Ez, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dx):
    """This function updates the Hy field components in the x stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEz
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[4, xf - (i + 1), j + ys, k + zs]
                dEz = (Ez[xf - i, j + ys, k + zs] - Ez[xf - (i + 1), j + ys, k + zs]) / dx
                Hy[xf - (i + 1), j + ys, k + zs] = Hy[xf - (i + 1), j + ys, k + zs] + updatecoeffsH[listIndex, 4] * ((RA[0, i] * RA[1, i] - 1) * dEz + RA[1, i] * RB[0, i] * HPhi[0, i, j, k] + RB[1, i] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, i] * HPhi[1, i, j, k] - RF[1, i] * (RA[0, i] * dEz + RB[0, i] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, i] * HPhi[0, i, j, k] - RF[0, i] * dEz


cpdef update_pml_2order_hy_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hy, floattype_t[:, :, :] Ex, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dz):
    """This function updates the Hy field components in the z stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEx
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[4, i + xs, j + ys, k + zs]
                dEx = (Ex[i + xs, j + ys, k + 1 + zs] - Ex[i + xs, j + ys, k + zs]) / dz
                Hy[i + xs, j + ys, k + zs] = Hy[i + xs, j + ys, k + zs] - updatecoeffsH[listIndex, 4] * ((RA[0, k] * RA[1, k] - 1) * dEx + RA[1, k] * RB[0, k] * HPhi[0, i, j, k] + RB[1, k] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, k] * HPhi[1, i, j, k] - RF[1, k] * (RA[0, k] * dEx + RB[0, k] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, k] * HPhi[0, i, j, k] - RF[0, k] * dEx


cpdef update_pml_2order_hy_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hy, floattype_t[:, :, :] Ex, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dz):
    """This function updates the Hy field components in the z stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEx
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs
    
    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[4, i + xs, j + ys, zf - (k + 1)]
                dEx = (Ex[i + xs, j + ys, zf - k] - Ex[i + xs, j + ys, zf - (k + 1)]) / dz
                Hy[i + xs, j + ys, zf - (k + 1)] = Hy[i + xs, j + ys, zf - (k + 1)] - updatecoeffsH[listIndex, 4] * ((RA[0, k] * RA[1, k] - 1) * dEx + RA[1, k] * RB[0, k] * HPhi[0, i, j, k] + RB[1, k] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, k] * HPhi[1, i, j, k] - RF[1, k] * (RA[0, k] * dEx + RB[0, k] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, k] * HPhi[0, i, j, k] - RF[0, k] * dEx


#############################################
# Magnetic field PML updates - Hz component #
#############################################
cpdef update_pml_2order_hz_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hz, floattype_t[:, :, :] Ey, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dx):
    """This function updates the Hz field components in the x stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEy
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[5, i + xs, j + ys, k + zs]
                dEy = (Ey[i + 1 + xs, j + ys, k + zs] - Ey[i + xs, j + ys, k + zs]) / dx
                Hz[i + xs, j + ys, k + zs] = Hz[i + xs, j + ys, k + zs] - updatecoeffsH[listIndex, 4] * ((RA[0, i] * RA[1, i] - 1) * dEy + RA[1, i] * RB[0, i] * HPhi[0, i, j, k] + RB[1, i] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, i] * HPhi[1, i, j, k] - RF[1, i] * (RA[0, i] * dEy + RB[0, i] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, i] * HPhi[0, i, j, k] - RF[0, i] * dEy


cpdef update_pml_2order_hz_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hz, floattype_t[:, :, :] Ey, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dx):
    """This function updates the Hz field components in the x stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEy
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[5, xf - (i + 1), j + ys, k + zs]
                dEy = (Ey[xf - i, j + ys, k + zs] - Ey[xf - (i + 1), j + ys, k + zs]) / dx
                Hz[xf - (i + 1), j + ys, k + zs] = Hz[xf - (i + 1), j + ys, k + zs] - updatecoeffsH[listIndex, 4] * ((RA[0, i] * RA[1, i] - 1) * dEy + RA[1, i] * RB[0, i] * HPhi[0, i, j, k] + RB[1, i] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, i] * HPhi[1, i, j, k] - RF[1, i] * (RA[0, i] * dEy + RB[0, i] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, i] * HPhi[0, i, j, k] - RF[0, i] * dEy


cpdef update_pml_2order_hz_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hz, floattype_t[:, :, :] Ex, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dy):
    """This function updates the Hz field components in the y stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEx
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[5, i + xs, j + ys, k + zs]
                dEx = (Ex[i + xs, j + 1 + ys, k + zs] - Ex[i + xs, j + ys, k + zs]) / dy
                Hz[i + xs, j + ys, k + zs] = Hz[i + xs, j + ys, k + zs] + updatecoeffsH[listIndex, 4] * ((RA[0, j] * RA[1, j] - 1) * dEx + RA[1, j] * RB[0, j] * HPhi[0, i, j, k] + RB[1, j] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, j] * HPhi[1, i, j, k] - RF[1, j] * (RA[0, j] * dEx + RB[0, j] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, j] * HPhi[0, i, j, k] - RF[0, j] * dEx


cpdef update_pml_2order_hz_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hz, floattype_t[:, :, :] Ex, floattype_t[:, :, :, :] HPhi, floattype_t[:, :] RA, floattype_t[:, :] RB, floattype_t[:, :] RE, floattype_t[:, :] RF, float dy):
    """This function updates the Hz field components in the y stretching direction.
        
    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
        EPhi, HPhi, RA, RB, RE, RF (memoryviews): Access to PML coefficient arrays
        dx, dy, dz (float): Spatial discretisation
    """
    
    cdef int i, j, k, nx, ny, nz, listIndex
    cdef float dEx
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[5, i + xs, yf - (j + 1), k + zs]
                dEx = (Ex[i + xs, yf - j, k + zs] - Ex[i + xs, yf - (j + 1), k + zs]) / dy
                Hz[i + xs, yf - (j + 1), k + zs] = Hz[i + xs, yf - (j + 1), k + zs] + updatecoeffsH[listIndex, 4] * ((RA[0, j] * RA[1, j] - 1) * dEx + RA[1, j] * RB[0, j] * HPhi[0, i, j, k] + RB[1, j] * HPhi[1, i, j, k])
                HPhi[1, i, j, k] = RE[1, j] * HPhi[1, i, j, k] - RF[1, j] * (RA[0, j] * dEx + RB[0, j] * HPhi[0, i, j, k])
                HPhi[0, i, j, k] = RE[0, j] * HPhi[0, i, j, k] - RF[0, j] * dEx

