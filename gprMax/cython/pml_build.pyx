# cython: cdivision=True
# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.config cimport float_or_double


cpdef pml_average_er_mr(
    int n1,
    int n2,
    int nthreads,
    np.uint32_t[:, :] solid,
    float_or_double[::1] ers,
    float_or_double[::1] mrs
):
    """Calculates average permittivity and permeability in PML slab (based on
        underlying material er and mr from solid array). Used to build PML.

    Args:
        n1, n2: ints for PML size in cells perpendicular to thickness direction.
        nthreads: int for number of threads to use.
        solid: memoryviews to access solid array.
        ers, mrs: memoryviews to access arrays containing permittivity and
                    permeability.

    Returns:
        averageer, averagemr: floats for average permittivity and permeability
                                in PML slab.
    """

    cdef Py_ssize_t m, n
    cdef int numID
    # Sum and average of relative permittivities and permeabilities in PML slab.
    # sumer/summr MUST be zero-initialized: they are accumulated with += inside
    # a prange loop, which Cython/OpenMP treats as a reduction - the reduction
    # combines per-thread partial sums with whatever value the variable held
    # BEFORE the parallel region, not with an implicit zero. Leaving them
    # uninitialized let this function's result depend on leftover stack
    # garbage, producing an intermittent, geometry-dependent PML impedance
    # mismatch (silent reflections) - see the investigation of gprMax/gprMax#435.
    cdef double sumer = 0, summr = 0, averageer, averagemr

    for m in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for n in range(n2):
            numID = solid[m ,n]
            sumer += ers[numID]
            summr += mrs[numID]
    averageer = sumer / (n1 * n2)
    averagemr = summr / (n1 * n2)

    return averageer, averagemr

cpdef pml_sum_er_mr(
    int n1,
    int n2,
    int nthreads,
    np.uint32_t[:, :] solid,
    float_or_double[::1] ers,
    float_or_double[::1] mrs
):
    """Calculates average permittivity and permeability in PML slab (based on
        underlying material er and mr from solid array). Used to build PML.

    Args:
        n1, n2: ints for PML size in cells perpendicular to thickness direction.
        nthreads: int for number of threads to use.
        solid: memoryviews to access solid array.
        ers, mrs: memoryviews to access arrays containing permittivity and
                    permeability.

    Returns:
        averageer, averagemr: floats for average permittivity and permeability
                                in PML slab.
    """

    cdef Py_ssize_t m, n
    cdef int numID
    # Sum and average of relative permittivities and permeabilities in PML
    # slab. sumer/summr MUST be zero-initialized - see the identical fix and
    # explanation in pml_average_er_mr() above (gprMax/gprMax#435).
    cdef double sumer = 0, summr = 0

    for m in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for n in range(n2):
            numID = solid[m ,n]
            sumer += ers[numID]
            summr += mrs[numID]

    return sumer, summr
