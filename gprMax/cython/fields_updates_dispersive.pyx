# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
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


# Use C-functions from 'complex.h' but doesn't work for Windows as it doesn't
#  support 'double complex' but instead defines its own type '_Dcomplex'.
#  https://docs.microsoft.com/en-us/cpp/c-runtime-library/complex-math-support?view=vs-2019
#  https://stackoverflow.com/questions/57837255/defining-dcomplex-externally-in-cython?rq=1

cdef extern from "complex.h" nogil:
    double creal(double complex z)
    float crealf(float complex z)



###############################################################
# Electric field updates - dispersive materials - multipole A #
###############################################################


cpdef void update_electric_dispersive_multipole_A_double_real(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    double[:, ::1] updatecoeffsE,
    double[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    double[:, :, :, ::1] Tx,
    double[:, :, :, ::1] Ty,
    double[:, :, :, ::1] Tz,
    double[:, :, ::1] Ex,
    double[:, :, ::1] Ey,
    double[:, :, ::1] Ez,
    double[:, :, ::1] Hx,
    double[:, :, ::1] Hy,
    double[:, :, ::1] Hz
):
    """Updates the electric field components when dispersive materials 
        (with multiple poles) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                        
                        Tx[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Tx[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ex[i, j, k])
                    Ex[i, j, k] = (updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] 
                                   * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] 
                                   * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi)

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        
                        phi = phi + updatecoeffsdispersive[material, pole * 3] * Ty[pole, i, j, k]
                        
                        Ty[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Ty[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ey[i, j, k])
                    Ey[i, j, k] = (updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] 
                                   * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] 
                                   * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi)

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        
                        phi = phi + updatecoeffsdispersive[material, pole * 3] * Tz[pole, i, j, k]
                        
                        Tz[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Tz[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ez[i, j, k])
                    Ez[i, j, k] = (updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] 
                                   * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] 
                                   * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi)

cpdef void update_electric_dispersive_multipole_A_float_real(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float[:, ::1] updatecoeffsE,
    float[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float[:, :, :, ::1] Tx,
    float[:, :, :, ::1] Ty,
    float[:, :, :, ::1] Tz,
    float[:, :, ::1] Ex,
    float[:, :, ::1] Ey,
    float[:, :, ::1] Ez,
    float[:, :, ::1] Hx,
    float[:, :, ::1] Hy,
    float[:, :, ::1] Hz
):
    """Updates the electric field components when dispersive materials 
        (with multiple poles) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                        
                        Tx[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Tx[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ex[i, j, k])
                    Ex[i, j, k] = (updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] 
                                   * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] 
                                   * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi)

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        
                        phi = phi + updatecoeffsdispersive[material, pole * 3] * Ty[pole, i, j, k]
                        
                        Ty[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Ty[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ey[i, j, k])
                    Ey[i, j, k] = (updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] 
                                   * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] 
                                   * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi)

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        
                        phi = phi + updatecoeffsdispersive[material, pole * 3] * Tz[pole, i, j, k]
                        
                        Tz[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Tz[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ez[i, j, k])
                    Ez[i, j, k] = (updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] 
                                   * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] 
                                   * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi)

cpdef void update_electric_dispersive_multipole_A_double_complex(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    double[:, ::1] updatecoeffsE,
    double complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    double complex[:, :, :, ::1] Tx,
    double complex[:, :, :, ::1] Ty,
    double complex[:, :, :, ::1] Tz,
    double[:, :, ::1] Ex,
    double[:, :, ::1] Ey,
    double[:, :, ::1] Ez,
    double[:, :, ::1] Hx,
    double[:, :, ::1] Hy,
    double[:, :, ::1] Hz
):
    """Updates the electric field components when dispersive materials 
        (with multiple poles) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                        
                        
                        phi = (phi + creal(updatecoeffsdispersive[material, pole * 3]) 
                               * creal(Tx[pole, i, j, k]))
                        
                        
                        Tx[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Tx[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ex[i, j, k])
                    Ex[i, j, k] = (updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] 
                                   * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] 
                                   * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi)

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        
                        
                        phi = (phi + creal(updatecoeffsdispersive[material, pole * 3]) 
                               * creal(Ty[pole, i, j, k]))
                        
                        
                        Ty[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Ty[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ey[i, j, k])
                    Ey[i, j, k] = (updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] 
                                   * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] 
                                   * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi)

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        
                        
                        phi = (phi + creal(updatecoeffsdispersive[material, pole * 3]) 
                               * creal(Tz[pole, i, j, k]))
                        
                        
                        Tz[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Tz[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ez[i, j, k])
                    Ez[i, j, k] = (updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] 
                                   * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] 
                                   * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi)

cpdef void update_electric_dispersive_multipole_A_float_complex(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float[:, ::1] updatecoeffsE,
    float complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float complex[:, :, :, ::1] Tx,
    float complex[:, :, :, ::1] Ty,
    float complex[:, :, :, ::1] Tz,
    float[:, :, ::1] Ex,
    float[:, :, ::1] Ey,
    float[:, :, ::1] Ez,
    float[:, :, ::1] Hx,
    float[:, :, ::1] Hy,
    float[:, :, ::1] Hz
):
    """Updates the electric field components when dispersive materials 
        (with multiple poles) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                        
                        
                        phi = (phi + crealf(updatecoeffsdispersive[material, pole * 3]) 
                               * crealf(Tx[pole, i, j, k]))
                        
                        
                        Tx[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Tx[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ex[i, j, k])
                    Ex[i, j, k] = (updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] 
                                   * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] 
                                   * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi)

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        
                        
                        phi = (phi + crealf(updatecoeffsdispersive[material, pole * 3]) 
                               * crealf(Ty[pole, i, j, k]))
                        
                        
                        Ty[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Ty[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ey[i, j, k])
                    Ey[i, j, k] = (updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] 
                                   * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] 
                                   * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi)

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        
                        
                        phi = (phi + crealf(updatecoeffsdispersive[material, pole * 3]) 
                               * crealf(Tz[pole, i, j, k]))
                        
                        
                        Tz[pole, i, j, k] = (updatecoeffsdispersive[material, 1 + (pole * 3)] 
                                             * Tz[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ez[i, j, k])
                    Ez[i, j, k] = (updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] 
                                   * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] 
                                   * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi)



###############################################################
# Electric field updates - dispersive materials - multipole B #
###############################################################


cpdef void update_electric_dispersive_multipole_B_double_real(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    double[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    double[:, :, :, ::1] Tx,
    double[:, :, :, ::1] Ty,
    double[:, :, :, ::1] Tz,
    double[:, :, ::1] Ex,
    double[:, :, ::1] Ey,
    double[:, :, ::1] Ez
):
    """Updates a temporary dispersive material array when disperisive materials 
        (with multiple poles) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                        Tx[pole, i, j, k] = (Tx[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ex[i, j, k])

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    for pole in range(maxpoles):
                        Ty[pole, i, j, k] = (Ty[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ey[i, j, k])

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    for pole in range(maxpoles):
                        Tz[pole, i, j, k] = (Tz[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ez[i, j, k])


cpdef void update_electric_dispersive_multipole_B_float_real(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float[:, :, :, ::1] Tx,
    float[:, :, :, ::1] Ty,
    float[:, :, :, ::1] Tz,
    float[:, :, ::1] Ex,
    float[:, :, ::1] Ey,
    float[:, :, ::1] Ez
):
    """Updates a temporary dispersive material array when disperisive materials 
        (with multiple poles) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                        Tx[pole, i, j, k] = (Tx[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ex[i, j, k])

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    for pole in range(maxpoles):
                        Ty[pole, i, j, k] = (Ty[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ey[i, j, k])

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    for pole in range(maxpoles):
                        Tz[pole, i, j, k] = (Tz[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ez[i, j, k])


cpdef void update_electric_dispersive_multipole_B_double_complex(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    double complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    double complex[:, :, :, ::1] Tx,
    double complex[:, :, :, ::1] Ty,
    double complex[:, :, :, ::1] Tz,
    double[:, :, ::1] Ex,
    double[:, :, ::1] Ey,
    double[:, :, ::1] Ez
):
    """Updates a temporary dispersive material array when disperisive materials 
        (with multiple poles) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                        Tx[pole, i, j, k] = (Tx[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ex[i, j, k])

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    for pole in range(maxpoles):
                        Ty[pole, i, j, k] = (Ty[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ey[i, j, k])

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    for pole in range(maxpoles):
                        Tz[pole, i, j, k] = (Tz[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ez[i, j, k])


cpdef void update_electric_dispersive_multipole_B_float_complex(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float complex[:, :, :, ::1] Tx,
    float complex[:, :, :, ::1] Ty,
    float complex[:, :, :, ::1] Tz,
    float[:, :, ::1] Ex,
    float[:, :, ::1] Ey,
    float[:, :, ::1] Ez
):
    """Updates a temporary dispersive material array when disperisive materials 
        (with multiple poles) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                        Tx[pole, i, j, k] = (Tx[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ex[i, j, k])

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    for pole in range(maxpoles):
                        Ty[pole, i, j, k] = (Ty[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ey[i, j, k])

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    for pole in range(maxpoles):
                        Tz[pole, i, j, k] = (Tz[pole, i, j, k] 
                                             - updatecoeffsdispersive[material, 2 + (pole * 3)] 
                                             * Ez[i, j, k])




#################################################################
# Electric field updates - dispersive materials - single pole A #
#################################################################


cpdef void update_electric_dispersive_1pole_A_double_real(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    double[:, ::1] updatecoeffsE,
    double[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    double[:, :, :, ::1] Tx,
    double[:, :, :, ::1] Ty,
    double[:, :, :, ::1] Tz,
    double[:, :, ::1] Ex,
    double[:, :, ::1] Ey,
    double[:, :, ::1] Ez,
    double[:, :, ::1] Hx,
    double[:, :, ::1] Hy,
    double[:, :, ::1] Hz
):
    """Updates the electric field components when dispersive materials 
        (with 1 pole) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                    
                    Tx[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Tx[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ex[i, j, k])
                    Ex[i, j, k] = (updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] 
                                   * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] 
                                   * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi)

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    
                    phi = updatecoeffsdispersive[material, 0] * Ty[0, i, j, k]
                    
                    Ty[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Ty[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ey[i, j, k])
                    Ey[i, j, k] = (updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] 
                                   * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] 
                                   * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi)

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    
                    phi = updatecoeffsdispersive[material, 0] * Tz[0, i, j, k]
                    
                    Tz[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Tz[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ez[i, j, k])
                    Ez[i, j, k] = (updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] 
                                   * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] 
                                   * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi)


cpdef void update_electric_dispersive_1pole_A_float_real(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float[:, ::1] updatecoeffsE,
    float[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float[:, :, :, ::1] Tx,
    float[:, :, :, ::1] Ty,
    float[:, :, :, ::1] Tz,
    float[:, :, ::1] Ex,
    float[:, :, ::1] Ey,
    float[:, :, ::1] Ez,
    float[:, :, ::1] Hx,
    float[:, :, ::1] Hy,
    float[:, :, ::1] Hz
):
    """Updates the electric field components when dispersive materials 
        (with 1 pole) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                    
                    Tx[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Tx[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ex[i, j, k])
                    Ex[i, j, k] = (updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] 
                                   * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] 
                                   * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi)

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    
                    phi = updatecoeffsdispersive[material, 0] * Ty[0, i, j, k]
                    
                    Ty[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Ty[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ey[i, j, k])
                    Ey[i, j, k] = (updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] 
                                   * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] 
                                   * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi)

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    
                    phi = updatecoeffsdispersive[material, 0] * Tz[0, i, j, k]
                    
                    Tz[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Tz[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ez[i, j, k])
                    Ez[i, j, k] = (updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] 
                                   * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] 
                                   * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi)


cpdef void update_electric_dispersive_1pole_A_double_complex(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    double[:, ::1] updatecoeffsE,
    double complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    double complex[:, :, :, ::1] Tx,
    double complex[:, :, :, ::1] Ty,
    double complex[:, :, :, ::1] Tz,
    double[:, :, ::1] Ex,
    double[:, :, ::1] Ey,
    double[:, :, ::1] Ez,
    double[:, :, ::1] Hx,
    double[:, :, ::1] Hy,
    double[:, :, ::1] Hz
):
    """Updates the electric field components when dispersive materials 
        (with 1 pole) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                    
                    
                    phi = (creal(updatecoeffsdispersive[material, 0]) 
                           * creal(Tx[0, i, j, k]))
                    
                    
                    Tx[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Tx[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ex[i, j, k])
                    Ex[i, j, k] = (updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] 
                                   * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] 
                                   * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi)

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    
                    
                    phi = (creal(updatecoeffsdispersive[material, 0]) 
                           * creal(Ty[0, i, j, k]))
                    
                    
                    Ty[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Ty[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ey[i, j, k])
                    Ey[i, j, k] = (updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] 
                                   * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] 
                                   * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi)

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    
                    
                    phi = (creal(updatecoeffsdispersive[material, 0]) 
                           * creal(Tz[0, i, j, k]))
                    
                    
                    Tz[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Tz[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ez[i, j, k])
                    Ez[i, j, k] = (updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] 
                                   * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] 
                                   * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi)


cpdef void update_electric_dispersive_1pole_A_float_complex(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float[:, ::1] updatecoeffsE,
    float complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float complex[:, :, :, ::1] Tx,
    float complex[:, :, :, ::1] Ty,
    float complex[:, :, :, ::1] Tz,
    float[:, :, ::1] Ex,
    float[:, :, ::1] Ey,
    float[:, :, ::1] Ez,
    float[:, :, ::1] Hx,
    float[:, :, ::1] Hy,
    float[:, :, ::1] Hz
):
    """Updates the electric field components when dispersive materials 
        (with 1 pole) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
                    
                    
                    phi = (crealf(updatecoeffsdispersive[material, 0]) 
                           * crealf(Tx[0, i, j, k]))
                    
                    
                    Tx[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Tx[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ex[i, j, k])
                    Ex[i, j, k] = (updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] 
                                   * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] 
                                   * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi)

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    
                    
                    phi = (crealf(updatecoeffsdispersive[material, 0]) 
                           * crealf(Ty[0, i, j, k]))
                    
                    
                    Ty[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Ty[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ey[i, j, k])
                    Ey[i, j, k] = (updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] 
                                   * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] 
                                   * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi)

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    
                    
                    phi = (crealf(updatecoeffsdispersive[material, 0]) 
                           * crealf(Tz[0, i, j, k]))
                    
                    
                    Tz[0, i, j, k] = (updatecoeffsdispersive[material, 1] * Tz[0, i, j, k] 
                                      + updatecoeffsdispersive[material, 2] * Ez[i, j, k])
                    Ez[i, j, k] = (updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] 
                                   * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] 
                                   * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi)




#################################################################
# Electric field updates - dispersive materials - single pole B #
#################################################################


cpdef void update_electric_dispersive_1pole_B_double_real(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    double[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    double[:, :, :, ::1] Tx,
    double[:, :, :, ::1] Ty,
    double[:, :, :, ::1] Tz,
    double[:, :, ::1] Ex,
    double[:, :, ::1] Ey,
    double[:, :, ::1] Ez
):
    """Updates a temporary dispersive material array when disperisive materials 
        (with 1 pole) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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

cpdef void update_electric_dispersive_1pole_B_float_real(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float[:, :, :, ::1] Tx,
    float[:, :, :, ::1] Ty,
    float[:, :, :, ::1] Tz,
    float[:, :, ::1] Ex,
    float[:, :, ::1] Ey,
    float[:, :, ::1] Ez
):
    """Updates a temporary dispersive material array when disperisive materials 
        (with 1 pole) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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

cpdef void update_electric_dispersive_1pole_B_double_complex(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    double complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    double complex[:, :, :, ::1] Tx,
    double complex[:, :, :, ::1] Ty,
    double complex[:, :, :, ::1] Tz,
    double[:, :, ::1] Ex,
    double[:, :, ::1] Ey,
    double[:, :, ::1] Ez
):
    """Updates a temporary dispersive material array when disperisive materials 
        (with 1 pole) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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

cpdef void update_electric_dispersive_1pole_B_float_complex(
    int nx,
    int ny,
    int nz,
    int nthreads,
    int maxpoles,
    float complex[:, ::1] updatecoeffsdispersive,
    np.uint32_t[:, :, :, ::1] ID,
    float complex[:, :, :, ::1] Tx,
    float complex[:, :, :, ::1] Ty,
    float complex[:, :, :, ::1] Tz,
    float[:, :, ::1] Ex,
    float[:, :, ::1] Ey,
    float[:, :, ::1] Ez
):
    """Updates a temporary dispersive material array when disperisive materials 
        (with 1 pole) are present.

    Args:
        nx, ny, nz: int for grid size in cells.
        nthreads: int for number of threads to use.
        maxpoles: int for maximum number of poles.
        updatecoeffs, T, ID, E, H: memoryviews to access to update coeffients, 
                                    temporary, ID and field component arrays.
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
