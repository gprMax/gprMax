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
from .constants cimport floattype_t, complextype_t


#########################################
# Electric field updates - Ex component #
#########################################
cpdef update_ex(int nx, int ny, int nz, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ex, floattype_t[:, :, :] Hy, floattype_t[:, :, :] Hz):
    """This function updates the Ex field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(1, nz):
                listIndex = ID[0, i, j, k]
                Ex[i, j, k] = updatecoeffsE[listIndex, 0] * Ex[i, j, k] + updatecoeffsE[listIndex, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[listIndex, 3] * (Hy[i, j, k] - Hy[i, j, k - 1])


cpdef update_ex_dispersive_multipole_A(int nx, int ny, int nz, int nthreads, int maxpoles, floattype_t[:, :] updatecoeffsE, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Tx, floattype_t[:, :, :] Ex, floattype_t[:, :, :] Hy, floattype_t[:, :, :] Hz):
    """This function updates the Ex field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex, p
    cdef float phi = 0.0

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(1, nz):
                listIndex = ID[0, i, j, k]
                phi = 0.0
                for p in range(0, maxpoles):
                    phi = phi + updatecoeffsdispersive[listIndex, p * 3].real * Tx[p, i, j, k].real
                    Tx[p, i, j, k] = updatecoeffsdispersive[listIndex, 1 + (p * 3)] * Tx[p, i, j, k] + updatecoeffsdispersive[listIndex, 2 + (p * 3)] * Ex[i, j, k]
                Ex[i, j, k] = updatecoeffsE[listIndex, 0] * Ex[i, j, k] + updatecoeffsE[listIndex, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[listIndex, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[listIndex, 4] * phi

cpdef update_ex_dispersive_multipole_B(int nx, int ny, int nz, int nthreads, int maxpoles, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Tx, floattype_t[:, :, :] Ex):
    """This function updates the Ex field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex, p
    
    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(1, nz):
                listIndex = ID[0, i, j, k]
                for p in range(0, maxpoles):
                    Tx[p, i, j, k] = Tx[p, i, j, k] - updatecoeffsdispersive[listIndex, 2 + (p * 3)] * Ex[i, j, k]


cpdef update_ex_dispersive_1pole_A(int nx, int ny, int nz, int nthreads, floattype_t[:, :] updatecoeffsE, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Tx, floattype_t[:, :, :] Ex, floattype_t[:, :, :] Hy, floattype_t[:, :, :] Hz):
    """This function updates the Ex field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex
    cdef float phi = 0.0

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(1, nz):
                listIndex = ID[0, i, j, k]
                phi = updatecoeffsdispersive[listIndex, 0].real * Tx[0, i, j, k].real
                Tx[0, i, j, k] = updatecoeffsdispersive[listIndex, 1] * Tx[0, i, j, k] + updatecoeffsdispersive[listIndex, 2] * Ex[i, j, k]
                Ex[i, j, k] = updatecoeffsE[listIndex, 0] * Ex[i, j, k] + updatecoeffsE[listIndex, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[listIndex, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[listIndex, 4] * phi


cpdef update_ex_dispersive_1pole_B(int nx, int ny, int nz, int nthreads, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Tx, floattype_t[:, :, :] Ex):
    """This function updates the Ex field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex
    
    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(1, nz):
                listIndex = ID[0, i, j, k]
                Tx[0, i, j, k] = Tx[0, i, j, k] - updatecoeffsdispersive[listIndex, 2] * Ex[i, j, k]


#########################################
# Electric field updates - Ey component #
#########################################
cpdef update_ey(int nx, int ny, int nz, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ey, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Hz):
    """This function updates the Ey field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex

    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(1, nz):
                listIndex = ID[1, i, j, k]
                Ey[i, j, k] = updatecoeffsE[listIndex, 0] * Ey[i, j, k] + updatecoeffsE[listIndex, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[listIndex, 1] * (Hz[i, j, k] - Hz[i - 1, j, k])


cpdef update_ey_dispersive_multipole_A(int nx, int ny, int nz, int nthreads, int maxpoles, floattype_t[:, :] updatecoeffsE, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Ty, floattype_t[:, :, :] Ey, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Hz):
    """This function updates the Ey field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex, p
    cdef float phi = 0.0

    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(1, nz):
                listIndex = ID[1, i, j, k]
                phi = 0.0
                for p in range(0, maxpoles):
                    phi = phi + updatecoeffsdispersive[listIndex, p * 3].real * Ty[p, i, j, k].real
                    Ty[p, i, j, k] = updatecoeffsdispersive[listIndex, 1 + (p * 3)] * Ty[p, i, j, k] + updatecoeffsdispersive[listIndex, 2 + (p * 3)] * Ey[i, j, k]
                Ey[i, j, k] = updatecoeffsE[listIndex, 0] * Ey[i, j, k] + updatecoeffsE[listIndex, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[listIndex, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[listIndex, 4] * phi


cpdef update_ey_dispersive_multipole_B(int nx, int ny, int nz, int nthreads, int maxpoles, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Ty, floattype_t[:, :, :] Ey):
    """This function updates the Ey field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex, p
    
    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(1, nz):
                listIndex = ID[1, i, j, k]
                for p in range(0, maxpoles):
                    Ty[p, i, j, k] = Ty[p, i, j, k] - updatecoeffsdispersive[listIndex, 2 + (p * 3)] * Ey[i, j, k]


cpdef update_ey_dispersive_1pole_A(int nx, int ny, int nz, int nthreads, floattype_t[:, :] updatecoeffsE, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Ty, floattype_t[:, :, :] Ey, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Hz):
    """This function updates the Ey field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex
    cdef float phi = 0.0

    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(1, nz):
                listIndex = ID[1, i, j, k]
                phi = updatecoeffsdispersive[listIndex, 0].real * Ty[0, i, j, k].real
                Ty[0, i, j, k] = updatecoeffsdispersive[listIndex, 1] * Ty[0, i, j, k] + updatecoeffsdispersive[listIndex, 2] * Ey[i, j, k]
                Ey[i, j, k] = updatecoeffsE[listIndex, 0] * Ey[i, j, k] + updatecoeffsE[listIndex, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[listIndex, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[listIndex, 4] * phi


cpdef update_ey_dispersive_1pole_B(int nx, int ny, int nz, int nthreads, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Ty, floattype_t[:, :, :] Ey):
    """This function updates the Ey field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex
    
    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(1, nz):
                listIndex = ID[1, i, j, k]
                Ty[0, i, j, k] = Ty[0, i, j, k] - updatecoeffsdispersive[listIndex, 2] * Ey[i, j, k]


#########################################
# Electric field updates - Ez component #
#########################################
cpdef update_ez(int nx, int ny, int nz, int nthreads, floattype_t[:, :] updatecoeffsE, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Ez, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Hy):
    """This function updates the Ez field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
        
    cdef int i, j, k, listIndex

    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(0, nz):
                listIndex = ID[2, i, j, k]
                Ez[i, j, k] = updatecoeffsE[listIndex, 0] * Ez[i, j, k] + updatecoeffsE[listIndex, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[listIndex, 2] * (Hx[i, j, k] - Hx[i, j - 1, k])


cpdef update_ez_dispersive_multipole_A(int nx, int ny, int nz, int nthreads, int maxpoles, floattype_t[:, :] updatecoeffsE, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Tz, floattype_t[:, :, :] Ez, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Hy):
    """This function updates the Ez field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
        
    cdef int i, j, k, listIndex, p
    cdef float phi = 0.0
    
    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(0, nz):
                listIndex = ID[2, i, j, k]
                phi = 0.0
                for p in range(0, maxpoles):
                    phi = phi + updatecoeffsdispersive[listIndex, p * 3].real * Tz[p, i, j, k].real
                    Tz[p, i, j, k] = updatecoeffsdispersive[listIndex, 1 + (p * 3)] * Tz[p, i, j, k] + updatecoeffsdispersive[listIndex, 2 + (p * 3)] * Ez[i, j, k]
                Ez[i, j, k] = updatecoeffsE[listIndex, 0] * Ez[i, j, k] + updatecoeffsE[listIndex, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[listIndex, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[listIndex, 4] * phi


cpdef update_ez_dispersive_multipole_B(int nx, int ny, int nz, int nthreads, int maxpoles, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Tz, floattype_t[:, :, :] Ez):
    """This function updates the Ez field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex, p
    
    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(0, nz):
                listIndex = ID[2, i, j, k]
                for p in range(0, maxpoles):
                    Tz[p, i, j, k] = Tz[p, i, j, k] - updatecoeffsdispersive[listIndex, 2 + (p * 3)] * Ez[i, j, k]


cpdef update_ez_dispersive_1pole_A(int nx, int ny, int nz, int nthreads, floattype_t[:, :] updatecoeffsE, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Tz, floattype_t[:, :, :] Ez, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Hy):
    """This function updates the Ez field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
        
    cdef int i, j, k, listIndex
    cdef float phi = 0.0
    
    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(0, nz):
                listIndex = ID[2, i, j, k]
                phi = updatecoeffsdispersive[listIndex, 0].real * Tz[0, i, j, k].real
                Tz[0, i, j, k] = updatecoeffsdispersive[listIndex, 1] * Tz[0, i, j, k] + updatecoeffsdispersive[listIndex, 2] * Ez[i, j, k]
                Ez[i, j, k] = updatecoeffsE[listIndex, 0] * Ez[i, j, k] + updatecoeffsE[listIndex, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[listIndex, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[listIndex, 4] * phi


cpdef update_ez_dispersive_1pole_B(int nx, int ny, int nz, int nthreads, complextype_t[:, :] updatecoeffsdispersive, np.uint32_t[:, :, :, :] ID, complextype_t[:, :, :, :] Tz, floattype_t[:, :, :] Ez):
    """This function updates the Ez field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex
    
    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(0, nz):
                listIndex = ID[2, i, j, k]
                Tz[0, i, j, k] = Tz[0, i, j, k] - updatecoeffsdispersive[listIndex, 2] * Ez[i, j, k]


#########################################
# Magnetic field updates - Hx component #
#########################################
cpdef update_hx(int nx, int ny, int nz, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hx, floattype_t[:, :, :] Ey, floattype_t[:, :, :] Ez):
    """This function updates the Hx field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex

    for i in prange(1, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(0, nz):
                listIndex = ID[3, i, j, k]
                Hx[i, j, k] = updatecoeffsH[listIndex, 0] * Hx[i, j, k] - updatecoeffsH[listIndex, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) + updatecoeffsH[listIndex, 3] * (Ey[i, j, k + 1] - Ey[i, j, k])


#########################################
# Magnetic field updates - Hy component #
#########################################
cpdef update_hy(int nx, int ny, int nz, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hy, floattype_t[:, :, :] Ex, floattype_t[:, :, :] Ez):
    """This function updates the Hy field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(1, ny):
            for k in range(0, nz):
                listIndex = ID[4, i, j, k]
                Hy[i, j, k] = updatecoeffsH[listIndex, 0] * Hy[i, j, k] - updatecoeffsH[listIndex, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) + updatecoeffsH[listIndex, 1] * (Ez[i + 1, j, k] - Ez[i, j, k])


#########################################
# Magnetic field updates - Hz component #
#########################################
cpdef update_hz(int nx, int ny, int nz, int nthreads, floattype_t[:, :] updatecoeffsH, np.uint32_t[:, :, :, :] ID, floattype_t[:, :, :] Hz, floattype_t[:, :, :] Ex, floattype_t[:, :, :] Ey):
    """This function updates the Hz field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef int i, j, k, listIndex

    for i in prange(0, nx, nogil=True, schedule='static', chunksize=1, num_threads=nthreads):
        for j in range(0, ny):
            for k in range(1, nz):
                listIndex = ID[5, i, j, k]
                Hz[i, j, k] = updatecoeffsH[listIndex, 0] * Hz[i, j, k] - updatecoeffsH[listIndex, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) + updatecoeffsH[listIndex, 2] * (Ex[i, j + 1, k] - Ex[i, j, k])

