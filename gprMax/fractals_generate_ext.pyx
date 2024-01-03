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


cpdef void generate_fractal2D(int nx, int ny, int nthreads, float D, np.float64_t[:] weighting, np.float64_t[:] v1, np.complex128_t[:, ::1] A, np.complex128_t[:, ::1] fractalsurface):
    """This function generates a fractal surface for a 2D array.
        
    Args:
        nx, ny (int): Fractal surface size in cells
        nthreads (int): Number of threads to use
        D (float): Fractal dimension
        weighting (memoryview): Access to weighting vector
        v1 (memoryview): Access to positional vector at centre of array, scaled by weighting
        A (memoryview): Access to array containing random numbers (to be convolved with fractal function)
        fractalsurface (memoryview): Access to array containing fractal surface data
    """

    cdef Py_ssize_t i, j
    cdef double v2x, v2y, rr, B

    for i in prange(nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(ny):
                # Positional vector for current position
                v2x = weighting[0] * i
                v2y = weighting[1] * j

                # Calulate norm of v2 - v1
                rr = ((v2x - v1[0])**2 + (v2y - v1[1])**2)**(1/2)

                B = rr**D
                if B == 0:
                    B = 0.9

                fractalsurface[i, j] = A[i, j] / B


cpdef void generate_fractal3D(int nx, int ny, int nz, int nthreads, float D, np.float64_t[:] weighting, np.float64_t[:] v1, np.complex128_t[:, :, ::1] A, np.complex128_t[:, :, ::1] fractalvolume):
    """This function generates a fractal volume for a 3D array.

    Args:
        nx, ny, nz (int): Fractal volume size in cells
        nthreads (int): Number of threads to use
        D (float): Fractal dimension
        weighting (memoryview): Access to weighting vector
        v1 (memoryview): Access to positional vector at centre of array, scaled by weighting
        A (memoryview): Access to array containing random numbers (to be convolved with fractal function)
        fractalvolume (memoryview): Access to array containing fractal volume data
    """

    cdef Py_ssize_t i, j, k
    cdef double v2x, v2y, v2z, rr, B

    for i in prange(nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(ny):
            for k in range(nz):
                # Positional vector for current position
                v2x = weighting[0] * i
                v2y = weighting[1] * j
                v2z = weighting[2] * k

                # Calulate norm of v2 - v1
                rr = ((v2x - v1[0])**2 + (v2y - v1[1])**2 + (v2z - v1[2])**2)**(1/2)
                B = rr**D
                if B == 0:
                    B = 0.9 
                    
                fractalvolume[i, j, k] = A[i, j, k] / B
