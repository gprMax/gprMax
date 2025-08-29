# Copyright (C) 2015-2023: The University of Edinburgh
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

from gprMax.constants cimport floattype_t
from gprMax.constants cimport complextype_t


###############################################
# Electric field updates - standard materials #
###############################################
cpdef void update_electric(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    floattype_t[:, ::1] updatecoeffsE,
                    np.uint32_t[:, :, :, ::1] ID,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz
            ):
    """This function updates the electric field components.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """

    cdef Py_ssize_t i, j, k
    cdef int materialEx, materialEy, materialEz

    # 2D - Ex component
    if nx == 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    materialEx = ID[0, i, j, k]
                    Ex[i, j, k] = updatecoeffsE[materialEx, 0] * Ex[i, j, k] + updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1])

    # 2D - Ey component
    elif ny == 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    materialEy = ID[1, i, j, k]
                    Ey[i, j, k] = updatecoeffsE[materialEy, 0] * Ey[i, j, k] + updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k])

    # 2D - Ez component
    elif nz == 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    materialEz = ID[2, i, j, k]
                    Ez[i, j, k] = updatecoeffsE[materialEz, 0] * Ez[i, j, k] + updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k])

    # 3D
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    materialEx = ID[0, i, j, k]
                    materialEy = ID[1, i, j, k]
                    materialEz = ID[2, i, j, k]
                    Ex[i, j, k] = updatecoeffsE[materialEx, 0] * Ex[i, j, k] + updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1])
                    Ey[i, j, k] = updatecoeffsE[materialEy, 0] * Ey[i, j, k] + updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k])
                    Ez[i, j, k] = updatecoeffsE[materialEz, 0] * Ez[i, j, k] + updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k])

        # Ex components at i = 0
        for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(1, nz):
                materialEx = ID[0, 0, j, k]
                Ex[0, j, k] = updatecoeffsE[materialEx, 0] * Ex[0, j, k] + updatecoeffsE[materialEx, 2] * (Hz[0, j, k] - Hz[0, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[0, j, k] - Hy[0, j, k - 1])

        # Ey components at j = 0
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(1, nz):
                materialEy = ID[1, i, 0, k]
                Ey[i, 0, k] = updatecoeffsE[materialEy, 0] * Ey[i, 0, k] + updatecoeffsE[materialEy, 3] * (Hx[i, 0, k] - Hx[i, 0, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, 0, k] - Hz[i - 1, 0, k])

        # Ez components at k = 0
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                materialEz = ID[2, i, j, 0]
                Ez[i, j, 0] = updatecoeffsE[materialEz, 0] * Ez[i, j, 0] + updatecoeffsE[materialEz, 1] * (Hy[i, j, 0] - Hy[i - 1, j, 0]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, 0] - Hx[i, j - 1, 0])


#################################################
# Electric field updates - dispersive materials #
#################################################
cpdef void update_electric_dispersive_multipole_A(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    floattype_t[:, ::1] updatecoeffsE,
                    complextype_t[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    complextype_t[:, :, :, ::1] Tx,
                    complextype_t[:, :, :, ::1] Ty,
                    complextype_t[:, :, :, ::1] Tz,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz
            ):
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
    if ny != 1 or nz != 1:
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
    if nx != 1 or nz != 1:
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
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        phi = phi + updatecoeffsdispersive[material, pole * 3].real * Tz[pole, i, j, k].real
                        Tz[pole, i, j, k] = updatecoeffsdispersive[material, 1 + (pole * 3)] * Tz[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi



cpdef void update_electric_dispersive_multipole_B(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    complextype_t[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    complextype_t[:, :, :, ::1] Tx,
                    complextype_t[:, :, :, ::1] Ty,
                    complextype_t[:, :, :, ::1] Tz,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez
            ):
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
    if ny != 1 or nz != 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    for pole in range(maxpoles):
                        Tx[pole, i, j, k] = Tx[pole, i, j, k] - updatecoeffsdispersive[material, 2 + (pole * 3)] * Ex[i, j, k]

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    for pole in range(maxpoles):
                        Ty[pole, i, j, k] = Ty[pole, i, j, k] - updatecoeffsdispersive[material, 2 + (pole * 3)] * Ey[i, j, k]

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    for pole in range(maxpoles):
                        Tz[pole, i, j, k] = Tz[pole, i, j, k] - updatecoeffsdispersive[material, 2 + (pole * 3)] * Ez[i, j, k]


cpdef void update_electric_dispersive_1pole_A(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    floattype_t[:, ::1] updatecoeffsE,
                    complextype_t[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    complextype_t[:, :, :, ::1] Tx,
                    complextype_t[:, :, :, ::1] Ty,
                    complextype_t[:, :, :, ::1] Tz,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz
            ):
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
    if ny != 1 or nz != 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    phi = updatecoeffsdispersive[material, 0].real * Tx[0, i, j, k].real
                    Tx[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tx[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ex[i, j, k]
                    Ex[i, j, k] = updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = updatecoeffsdispersive[material, 0].real * Ty[0, i, j, k].real
                    Ty[0, i, j, k] = updatecoeffsdispersive[material, 1] * Ty[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ey[i, j, k]
                    Ey[i, j, k] = updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = updatecoeffsdispersive[material, 0].real * Tz[0, i, j, k].real
                    Tz[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tz[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi


cpdef void update_electric_dispersive_1pole_B(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    complextype_t[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    complextype_t[:, :, :, ::1] Tx,
                    complextype_t[:, :, :, ::1] Ty,
                    complextype_t[:, :, :, ::1] Tz,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez
            ):
    """This function updates a temporary dispersive material array when disperisive materials (with 1 pole) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """

    cdef Py_ssize_t i, j, k
    cdef int material

    # Ex component
    if ny != 1 or nz != 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    Tx[0, i, j, k] = Tx[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ex[i, j, k]

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    Ty[0, i, j, k] = Ty[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ey[i, j, k]

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    Tz[0, i, j, k] = Tz[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ez[i, j, k]


##########################
# Magnetic field updates #
##########################
cpdef void update_magnetic(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    floattype_t[:, ::1] updatecoeffsH,
                    np.uint32_t[:, :, :, ::1] ID,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz
            ):
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
