# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

from gprMax.config cimport float_or_double


###############################################
# Electric field updates - standard materials #
###############################################
cpdef void update_electric(
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


##########################
# Magnetic field updates #
##########################
cpdef void update_magnetic(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    float_or_double[:, ::1] updatecoeffsH,
                    np.uint32_t[:, :, :, ::1] ID,
                    float_or_double[:, :, ::1] Ex,
                    float_or_double[:, :, ::1] Ey,
                    float_or_double[:, :, ::1] Ez,
                    float_or_double[:, :, ::1] Hx,
                    float_or_double[:, :, ::1] Hy,
                    float_or_double[:, :, ::1] Hz
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
