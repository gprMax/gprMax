# Copyright (C) 2015-2019: The University of Edinburgh
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

from gprMax.config cimport float_or_double
from gprMax.config cimport real_or_complex

cdef extern from "complex.h" nogil:
    double creal(double complex z)

#########################################################
# Electric field updates - dispersive materials - Debye #
#########################################################
cpdef void update_electric_dispersive_debye_multipole_A(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    float_or_double[:, ::1] updatecoeffsE,
                    float_or_double[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    float_or_double[:, :, :, ::1] Tx,
                    float_or_double[:, :, :, ::1] Ty,
                    float_or_double[:, :, :, ::1] Tz,
                    float_or_double[:, :, ::1] Ex,
                    float_or_double[:, :, ::1] Ey,
                    float_or_double[:, :, ::1] Ez,
                    float_or_double[:, :, ::1] Hx,
                    float_or_double[:, :, ::1] Hy,
                    float_or_double[:, :, ::1] Hz
            ):
    """This function updates the electric field components when dispersive materials (with multiple poles) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        maxpoles (int): Maximum number of poles
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
                        phi = phi + updatecoeffsdispersive[material, pole * 3] * Tx[pole, i, j, k]
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
                        phi = phi + updatecoeffsdispersive[material, pole * 3] * Ty[pole, i, j, k]
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
                        phi = phi + updatecoeffsdispersive[material, pole * 3] * Tz[pole, i, j, k]
                        Tz[pole, i, j, k] = updatecoeffsdispersive[material, 1 + (pole * 3)] * Tz[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi



cpdef void update_electric_dispersive_debye_multipole_B(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    float_or_double[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    float_or_double[:, :, :, ::1] Tx,
                    float_or_double[:, :, :, ::1] Ty,
                    float_or_double[:, :, :, ::1] Tz,
                    float_or_double[:, :, ::1] Ex,
                    float_or_double[:, :, ::1] Ey,
                    float_or_double[:, :, ::1] Ez
            ):
    """This function updates a temporary dispersive material array when disperisive materials (with multiple poles) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        maxpoles (int): Maximum number of poles
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


cpdef void update_electric_dispersive_debye_1pole_A(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    float_or_double[:, ::1] updatecoeffsE,
                    float_or_double[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    float_or_double[:, :, :, ::1] Tx,
                    float_or_double[:, :, :, ::1] Ty,
                    float_or_double[:, :, :, ::1] Tz,
                    float_or_double[:, :, ::1] Ex,
                    float_or_double[:, :, ::1] Ey,
                    float_or_double[:, :, ::1] Ez,
                    float_or_double[:, :, ::1] Hx,
                    float_or_double[:, :, ::1] Hy,
                    float_or_double[:, :, ::1] Hz
            ):
    """This function updates the electric field components when dispersive materials (with 1 pole) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        maxpoles (int): Maximum number of poles
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
                    phi = updatecoeffsdispersive[material, 0] * Tx[0, i, j, k]
                    Tx[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tx[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ex[i, j, k]
                    Ex[i, j, k] = updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = updatecoeffsdispersive[material, 0] * Ty[0, i, j, k]
                    Ty[0, i, j, k] = updatecoeffsdispersive[material, 1] * Ty[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ey[i, j, k]
                    Ey[i, j, k] = updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = updatecoeffsdispersive[material, 0] * Tz[0, i, j, k]
                    Tz[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tz[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi


cpdef void update_electric_dispersive_debye_1pole_B(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    float_or_double[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    float_or_double[:, :, :, ::1] Tx,
                    float_or_double[:, :, :, ::1] Ty,
                    float_or_double[:, :, :, ::1] Tz,
                    float_or_double[:, :, ::1] Ex,
                    float_or_double[:, :, ::1] Ey,
                    float_or_double[:, :, ::1] Ez
            ):
    """This function updates a temporary dispersive material array when disperisive materials (with 1 pole) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        maxpoles (int): Maximum number of poles
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


#################################################################
# Electric field updates - dispersive materials - Drude, Lorenz #
#################################################################
cpdef void update_electric_dispersive_multipole_A(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    float_or_double[:, ::1] updatecoeffsE,
                    real_or_complex[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    real_or_complex[:, :, :, ::1] Tx,
                    real_or_complex[:, :, :, ::1] Ty,
                    real_or_complex[:, :, :, ::1] Tz,
                    float_or_double[:, :, ::1] Ex,
                    float_or_double[:, :, ::1] Ey,
                    float_or_double[:, :, ::1] Ez,
                    float_or_double[:, :, ::1] Hx,
                    float_or_double[:, :, ::1] Hy,
                    float_or_double[:, :, ::1] Hz
            ):
    """This function updates the electric field components when dispersive materials (with multiple poles) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        maxpoles (int): Maximum number of poles
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """

    cdef Py_ssize_t i, j, k, pole
    cdef int material
    cdef float phi

    # Ex component
    if ny != 1 or nz != 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        phi = phi + creal(updatecoeffsdispersive[material, pole * 3]) * creal(Tx[pole, i, j, k])
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
                        phi = phi + creal(updatecoeffsdispersive[material, pole * 3]) * creal(Ty[pole, i, j, k])
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
                        phi = phi + creal(updatecoeffsdispersive[material, pole * 3]) * creal(Tz[pole, i, j, k])
                        Tz[pole, i, j, k] = updatecoeffsdispersive[material, 1 + (pole * 3)] * Tz[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi



cpdef void update_electric_dispersive_multipole_B(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    real_or_complex[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    real_or_complex[:, :, :, ::1] Tx,
                    real_or_complex[:, :, :, ::1] Ty,
                    real_or_complex[:, :, :, ::1] Tz,
                    float_or_double[:, :, ::1] Ex,
                    float_or_double[:, :, ::1] Ey,
                    float_or_double[:, :, ::1] Ez
            ):
    """This function updates a temporary dispersive material array when disperisive materials (with multiple poles) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        maxpoles (int): Maximum number of poles
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
                    int maxpoles,
                    float_or_double[:, ::1] updatecoeffsE,
                    real_or_complex[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    real_or_complex[:, :, :, ::1] Tx,
                    real_or_complex[:, :, :, ::1] Ty,
                    real_or_complex[:, :, :, ::1] Tz,
                    float_or_double[:, :, ::1] Ex,
                    float_or_double[:, :, ::1] Ey,
                    float_or_double[:, :, ::1] Ez,
                    float_or_double[:, :, ::1] Hx,
                    float_or_double[:, :, ::1] Hy,
                    float_or_double[:, :, ::1] Hz
            ):
    """This function updates the electric field components when dispersive materials (with 1 pole) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        maxpoles (int): Maximum number of poles
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
                    phi = creal(updatecoeffsdispersive[material, 0]) * creal(Tx[0, i, j, k])
                    Tx[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tx[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ex[i, j, k]
                    Ex[i, j, k] = updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = creal(updatecoeffsdispersive[material, 0]) * creal(Ty[0, i, j, k])
                    Ty[0, i, j, k] = updatecoeffsdispersive[material, 1] * Ty[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ey[i, j, k]
                    Ey[i, j, k] = updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = creal(updatecoeffsdispersive[material, 0]) * creal(Tz[0, i, j, k])
                    Tz[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tz[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi


cpdef void update_electric_dispersive_1pole_B(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    real_or_complex[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    real_or_complex[:, :, :, ::1] Tx,
                    real_or_complex[:, :, :, ::1] Ty,
                    real_or_complex[:, :, :, ::1] Tz,
                    float_or_double[:, :, ::1] Ex,
                    float_or_double[:, :, ::1] Ey,
                    float_or_double[:, :, ::1] Ez
            ):
    """This function updates a temporary dispersive material array when disperisive materials (with 1 pole) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        maxpoles (int): Maximum number of poles
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
