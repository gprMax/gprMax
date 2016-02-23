# Copyright (C) 2015-2016: The University of Edinburgh
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


#########################################
# Electric field updates - Ex component #
#########################################
cpdef void update_ex(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsE, np.uint32_t[:, :, :, ::1] ID, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Hy, floattype_t[:, :, ::1] Hz):
    """This function updates the Ex field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material

    if ny == 1 or nz == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    Ex[i, j, k] = updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] * (Hy[i, j, k] - Hy[i, j, k - 1])


cpdef void update_ex_dispersive_multipole_A(int nx, int ny, int nz, int nthreads, int maxpoles, floattype_t[:, ::1] updatecoeffsE, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tx, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Hy, floattype_t[:, :, ::1] Hz):
    """This function updates the Ex field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k, p
    cdef int material
    cdef float phi = 0.0

    if ny == 1 or nz == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    phi = 0.0
                    for p in range(0, maxpoles):
                        phi = phi + updatecoeffsdispersive[material, p * 3].real * Tx[p, i, j, k].real
                        Tx[p, i, j, k] = updatecoeffsdispersive[material, 1 + (p * 3)] * Tx[p, i, j, k] + updatecoeffsdispersive[material, 2 + (p * 3)] * Ex[i, j, k]
                    Ex[i, j, k] = updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi

cpdef void update_ex_dispersive_multipole_B(int nx, int ny, int nz, int nthreads, int maxpoles, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tx, floattype_t[:, :, ::1] Ex):
    """This function updates the Ex field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k, p
    cdef int material

    if ny == 1 or nz == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    for p in range(0, maxpoles):
                        Tx[p, i, j, k] = Tx[p, i, j, k] - updatecoeffsdispersive[material, 2 + (p * 3)] * Ex[i, j, k]


cpdef void update_ex_dispersive_1pole_A(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsE, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tx, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Hy, floattype_t[:, :, ::1] Hz):
    """This function updates the Ex field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material
    cdef float phi = 0.0

    if ny == 1 or nz == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    phi = updatecoeffsdispersive[material, 0].real * Tx[0, i, j, k].real
                    Tx[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tx[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ex[i, j, k]
                    Ex[i, j, k] = updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi


cpdef void update_ex_dispersive_1pole_B(int nx, int ny, int nz, int nthreads, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tx, floattype_t[:, :, ::1] Ex):
    """This function updates the Ex field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material

    if ny == 1 or nz == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    Tx[0, i, j, k] = Tx[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ex[i, j, k]


#########################################
# Electric field updates - Ey component #
#########################################
cpdef void update_ey(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsE, np.uint32_t[:, :, :, ::1] ID, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hz):
    """This function updates the Ey field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material

    if nx == 1 or nz == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    Ey[i, j, k] = updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] * (Hz[i, j, k] - Hz[i - 1, j, k])


cpdef void update_ey_dispersive_multipole_A(int nx, int ny, int nz, int nthreads, int maxpoles, floattype_t[:, ::1] updatecoeffsE, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Ty, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hz):
    """This function updates the Ey field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k, p
    cdef int material
    cdef float phi = 0.0

    if nx == 1 or nz == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = 0.0
                    for p in range(0, maxpoles):
                        phi = phi + updatecoeffsdispersive[material, p * 3].real * Ty[p, i, j, k].real
                        Ty[p, i, j, k] = updatecoeffsdispersive[material, 1 + (p * 3)] * Ty[p, i, j, k] + updatecoeffsdispersive[material, 2 + (p * 3)] * Ey[i, j, k]
                    Ey[i, j, k] = updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi


cpdef void update_ey_dispersive_multipole_B(int nx, int ny, int nz, int nthreads, int maxpoles, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Ty, floattype_t[:, :, ::1] Ey):
    """This function updates the Ey field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k, p
    cdef int material

    if nx == 1 or nz == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    for p in range(0, maxpoles):
                        Ty[p, i, j, k] = Ty[p, i, j, k] - updatecoeffsdispersive[material, 2 + (p * 3)] * Ey[i, j, k]


cpdef void update_ey_dispersive_1pole_A(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsE, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Ty, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hz):
    """This function updates the Ey field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material
    cdef float phi = 0.0

    if nx == 1 or nz == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = updatecoeffsdispersive[material, 0].real * Ty[0, i, j, k].real
                    Ty[0, i, j, k] = updatecoeffsdispersive[material, 1] * Ty[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ey[i, j, k]
                    Ey[i, j, k] = updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi


cpdef void update_ey_dispersive_1pole_B(int nx, int ny, int nz, int nthreads, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Ty, floattype_t[:, :, ::1] Ey):
    """This function updates the Ey field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material

    if nx == 1 or nz == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    Ty[0, i, j, k] = Ty[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ey[i, j, k]


#########################################
# Electric field updates - Ez component #
#########################################
cpdef void update_ez(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsE, np.uint32_t[:, :, :, ::1] ID, floattype_t[:, :, ::1] Ez, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hy):
    """This function updates the Ez field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
        
    cdef Py_ssize_t i, j, k
    cdef int material

    if nx == 1 or ny == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k])


cpdef void update_ez_dispersive_multipole_A(int nx, int ny, int nz, int nthreads, int maxpoles, floattype_t[:, ::1] updatecoeffsE, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tz, floattype_t[:, :, ::1] Ez, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hy):
    """This function updates the Ez field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
        
    cdef Py_ssize_t i, j, k, p
    cdef int material
    cdef float phi = 0.0

    if nx == 1 or ny == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = 0.0
                    for p in range(0, maxpoles):
                        phi = phi + updatecoeffsdispersive[material, p * 3].real * Tz[p, i, j, k].real
                        Tz[p, i, j, k] = updatecoeffsdispersive[material, 1 + (p * 3)] * Tz[p, i, j, k] + updatecoeffsdispersive[material, 2 + (p * 3)] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi


cpdef void update_ez_dispersive_multipole_B(int nx, int ny, int nz, int nthreads, int maxpoles, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tz, floattype_t[:, :, ::1] Ez):
    """This function updates the Ez field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k, p
    cdef int material

    if nx == 1 or ny == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    for p in range(0, maxpoles):
                        Tz[p, i, j, k] = Tz[p, i, j, k] - updatecoeffsdispersive[material, 2 + (p * 3)] * Ez[i, j, k]


cpdef void update_ez_dispersive_1pole_A(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsE, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tz, floattype_t[:, :, ::1] Ez, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hy):
    """This function updates the Ez field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
        
    cdef Py_ssize_t i, j, k
    cdef int material
    cdef float phi = 0.0

    if nx == 1 or ny == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = updatecoeffsdispersive[material, 0].real * Tz[0, i, j, k].real
                    Tz[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tz[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi


cpdef void update_ez_dispersive_1pole_B(int nx, int ny, int nz, int nthreads, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tz, floattype_t[:, :, ::1] Ez):
    """This function updates the Ez field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material

    if nx == 1 or ny == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    Tz[0, i, j, k] = Tz[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ez[i, j, k]


#########################################
# Magnetic field updates - Hx component #
#########################################
cpdef void update_hx(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsH, np.uint32_t[:, :, :, ::1] ID, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Ez):
    """This function updates the Hx field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material

    if nx == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(0, nz):
                    material = ID[3, i, j, k]
                    Hx[i, j, k] = updatecoeffsH[material, 0] * Hx[i, j, k] - updatecoeffsH[material, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) + updatecoeffsH[material, 3] * (Ey[i, j, k + 1] - Ey[i, j, k])


#########################################
# Magnetic field updates - Hy component #
#########################################
cpdef void update_hy(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsH, np.uint32_t[:, :, :, ::1] ID, floattype_t[:, :, ::1] Hy, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Ez):
    """This function updates the Hy field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material

    if ny == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[4, i, j, k]
                    Hy[i, j, k] = updatecoeffsH[material, 0] * Hy[i, j, k] - updatecoeffsH[material, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) + updatecoeffsH[material, 1] * (Ez[i + 1, j, k] - Ez[i, j, k])


#########################################
# Magnetic field updates - Hz component #
#########################################
cpdef void update_hz(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsH, np.uint32_t[:, :, :, ::1] ID, floattype_t[:, :, ::1] Hz, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Ey):
    """This function updates the Hz field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material

    if nz == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[5, i, j, k]
                    Hz[i, j, k] = updatecoeffsH[material, 0] * Hz[i, j, k] - updatecoeffsH[material, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) + updatecoeffsH[material, 2] * (Ex[i, j + 1, k] - Ex[i, j, k])

