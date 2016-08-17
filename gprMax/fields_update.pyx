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


#####################################
# Electric field updates - standard #
#####################################
cpdef void update_electric(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsE, np.uint32_t[:, :, :, ::1] ID, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Ez, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hy, floattype_t[:, :, ::1] Hz):
    """This function updates the electric field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int materialEx, materialEy, materialEz

    # 2D
    if nx == 1 or ny == 1 or nz == 1:
        # Ex component
        if nx == 1:
            for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
                for j in range(1, ny):
                    for k in range(1, nz):
                        materialEx = ID[0, i, j, k]
                        Ex[i, j, k] = updatecoeffsE[materialEx, 0] * Ex[i, j, k] + updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1])

        # Ey component
        if ny == 1:
            for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
                for j in range(0, ny):
                    for k in range(1, nz):
                        materialEy = ID[1, i, j, k]
                        Ey[i, j, k] = updatecoeffsE[materialEy, 0] * Ey[i, j, k] + updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k])

        # Ez component
        if nz == 1:
            for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
                for j in range(1, ny):
                    for k in range(0, nz):
                        materialEz = ID[2, i, j, k]
                        Ez[i, j, k] = updatecoeffsE[materialEz, 0] * Ez[i, j, k] + updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k])

    # 3D
    else:
        for i in prange(0, nx - 1, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny - 1):
                for k in range(0, nz - 1):
                    materialEx = ID[0, i, j + 1, k + 1]
                    materialEy = ID[1, i + 1, j, k + 1]
                    materialEz = ID[2, i + 1, j + 1, k]
                    Ex[i, j + 1, k + 1] = updatecoeffsE[materialEx, 0] * Ex[i, j + 1, k + 1] + updatecoeffsE[materialEx, 2] * (Hz[i, j + 1, k + 1] - Hz[i, j, k + 1]) - updatecoeffsE[materialEx, 3] * (Hy[i, j + 1, k + 1] - Hy[i, j + 1, k])
                    Ey[i + 1, j, k + 1] = updatecoeffsE[materialEy, 0] * Ey[i + 1, j, k + 1] + updatecoeffsE[materialEy, 3] * (Hx[i + 1, j, k + 1] - Hx[i + 1, j, k]) - updatecoeffsE[materialEy, 1] * (Hz[i + 1, j, k + 1] - Hz[i, j, k + 1])
                    Ez[i + 1, j + 1, k] = updatecoeffsE[materialEz, 0] * Ez[i + 1, j + 1, k] + updatecoeffsE[materialEz, 1] * (Hy[i + 1, j + 1, k] - Hy[i, j + 1, k]) - updatecoeffsE[materialEz, 2] * (Hx[i + 1, j + 1, k] - Hx[i + 1, j, k])

        # Ex components at nx - 1
        for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(1, nz):
                materialEx = ID[0, nx - 1, j, k]
                Ex[nx - 1, j, k] = updatecoeffsE[materialEx, 0] * Ex[nx - 1, j, k] + updatecoeffsE[materialEx, 2] * (Hz[nx - 1, j, k] - Hz[nx - 1, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[nx - 1, j, k] - Hy[nx - 1, j, k - 1])

        # Ey components at ny - 1
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(1, nz):
                materialEy = ID[1, i, ny - 1, k]
                Ey[i, ny - 1, k] = updatecoeffsE[materialEy, 0] * Ey[i, ny - 1, k] + updatecoeffsE[materialEy, 3] * (Hx[i, ny - 1, k] - Hx[i, ny - 1, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, ny - 1, k] - Hz[i - 1, ny - 1, k])

        # Ez components at nz - 1
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                materialEz = ID[2, i, j, nz - 1]
                Ez[i, j, nz - 1] = updatecoeffsE[materialEz, 0] * Ez[i, j, nz - 1] + updatecoeffsE[materialEz, 1] * (Hy[i, j, nz - 1] - Hy[i - 1, j, nz - 1]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, nz - 1] - Hx[i, j - 1, nz - 1])


#################################################
# Electric field updates - dispersive materials #
#################################################
cpdef void update_electric_dispersive_multipole_A(int nx, int ny, int nz, int nthreads, int maxpoles, floattype_t[:, ::1] updatecoeffsE, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tx, complextype_t[:, :, :, ::1] Ty, complextype_t[:, :, :, ::1] Tz, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Ez, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hy, floattype_t[:, :, ::1] Hz):
    """This function updates the electric field components when dispersive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k, pole
    cdef int material
    cdef float phi = 0

    # Ex component
    if ny == 1 or nz == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        phi = phi + updatecoeffsdispersive[material, pole * 3].real * Tx[pole, i, j, k].real
                        Tx[pole, i, j, k] = updatecoeffsdispersive[material, 1 + (pole * 3)] * Tx[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] * Ex[i, j, k]
                    Ex[i, j, k] = updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi

    # Ey component
    if nx == 1 or nz == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        phi = phi + updatecoeffsdispersive[material, pole * 3].real * Ty[pole, i, j, k].real
                        Ty[pole, i, j, k] = updatecoeffsdispersive[material, 1 + (pole * 3)] * Ty[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] * Ey[i, j, k]
                    Ey[i, j, k] = updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi

    # Ez component
    if nx == 1 or ny == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        phi = phi + updatecoeffsdispersive[material, pole * 3].real * Tz[pole, i, j, k].real
                        Tz[pole, i, j, k] = updatecoeffsdispersive[material, 1 + (pole * 3)] * Tz[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi



cpdef void update_electric_dispersive_multipole_B(int nx, int ny, int nz, int nthreads, int maxpoles, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tx, complextype_t[:, :, :, ::1] Ty, complextype_t[:, :, :, ::1] Tz, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Ez):
    """This function updates a temporary dispersive material array when disperisive materials (with multiple poles) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k, pole
    cdef int material

    # Ex component
    if ny == 1 or nz == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    for pole in range(maxpoles):
                        Tx[pole, i, j, k] = Tx[pole, i, j, k] - updatecoeffsdispersive[material, 2 + (pole * 3)] * Ex[i, j, k]

    # Ey component
    if nx == 1 or nz == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    for pole in range(maxpoles):
                        Ty[pole, i, j, k] = Ty[pole, i, j, k] - updatecoeffsdispersive[material, 2 + (pole * 3)] * Ey[i, j, k]

    # Ez component
    if nx == 1 or ny == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    for pole in range(maxpoles):
                        Tz[pole, i, j, k] = Tz[pole, i, j, k] - updatecoeffsdispersive[material, 2 + (pole * 3)] * Ez[i, j, k]


cpdef void update_electric_dispersive_1pole_A(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsE, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tx, complextype_t[:, :, :, ::1] Ty, complextype_t[:, :, :, ::1] Tz, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Ez, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hy, floattype_t[:, :, ::1] Hz):
    """This function updates the electric field components when dispersive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material
    cdef float phi = 0

    # Ex component
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

    # Ey component
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

    # Ez component
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


cpdef void update_electric_dispersive_1pole_B(int nx, int ny, int nz, int nthreads, complextype_t[:, ::1] updatecoeffsdispersive, np.uint32_t[:, :, :, ::1] ID, complextype_t[:, :, :, ::1] Tx, complextype_t[:, :, :, ::1] Ty, complextype_t[:, :, :, ::1] Tz, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Ez):
    """This function updates a temporary dispersive material array when disperisive materials (with 1 pole) are present.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int material

    # Ex component
    if ny == 1 or nz == 1:
        pass
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    Tx[0, i, j, k] = Tx[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ex[i, j, k]

    # Ey component
    if nx == 1 or nz == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    Ty[0, i, j, k] = Ty[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ey[i, j, k]

    # Ez component
    if nx == 1 or ny == 1:
        pass
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    Tz[0, i, j, k] = Tz[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ez[i, j, k]


##########################
# Magnetic field updates #
##########################
cpdef void update_magnetic(int nx, int ny, int nz, int nthreads, floattype_t[:, ::1] updatecoeffsH, np.uint32_t[:, :, :, ::1] ID, floattype_t[:, :, ::1] Ex, floattype_t[:, :, ::1] Ey, floattype_t[:, :, ::1] Ez, floattype_t[:, :, ::1] Hx, floattype_t[:, :, ::1] Hy, floattype_t[:, :, ::1] Hz):
    """This function updates the magnetic field components.
        
    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """
    
    cdef Py_ssize_t i, j, k
    cdef int materialHx, materialHy, materialHz

    # 2D
    if nx == 1 or ny == 1 or nz == 1:
    # Hx component
        if ny == 1 or nz == 1:
            for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
                for j in range(0, ny):
                    for k in range(0, nz):
                        materialHx = ID[3, i, j, k]
                        Hx[i, j, k] = updatecoeffsH[materialHx, 0] * Hx[i, j, k] - updatecoeffsH[materialHx, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) + updatecoeffsH[materialHx, 3] * (Ey[i, j, k + 1] - Ey[i, j, k])

        # Hy component
        if nx == 1 or nz == 1:
            for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
                for j in range(1, ny):
                    for k in range(0, nz):
                        materialHy = ID[4, i, j, k]
                        Hy[i, j, k] = updatecoeffsH[materialHy, 0] * Hy[i, j, k] - updatecoeffsH[materialHy, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) + updatecoeffsH[materialHy, 1] * (Ez[i + 1, j, k] - Ez[i, j, k])

        # Hz component
        if nx == 1 or ny == 1:
            for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
                for j in range(0, ny):
                    for k in range(1, nz):
                        materialHz = ID[5, i, j, k]
                        Hz[i, j, k] = updatecoeffsH[materialHz, 0] * Hz[i, j, k] - updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) + updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k] - Ex[i, j, k])
    # 3D
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(0, nz):
                    materialHx = ID[3, i + 1, j, k]
                    materialHy = ID[4, i, j + 1, k]
                    materialHz = ID[5, i, j, k + 1]
                    Hx[i + 1, j, k] = updatecoeffsH[materialHx, 0] * Hx[i + 1, j, k] - updatecoeffsH[materialHx, 2] * (Ez[i + 1, j + 1, k] - Ez[i + 1, j, k]) + updatecoeffsH[materialHx, 3] * (Ey[i + 1, j, k + 1] - Ey[i + 1, j, k])
                    Hy[i, j + 1, k] = updatecoeffsH[materialHy, 0] * Hy[i, j + 1, k] - updatecoeffsH[materialHy, 3] * (Ex[i, j + 1, k + 1] - Ex[i, j + 1, k]) + updatecoeffsH[materialHy, 1] * (Ez[i + 1, j + 1, k] - Ez[i, j + 1, k])
                    Hz[i, j, k + 1] = updatecoeffsH[materialHz, 0] * Hz[i, j, k + 1] - updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k + 1] - Ey[i, j, k + 1]) + updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k + 1] - Ex[i, j, k + 1])
