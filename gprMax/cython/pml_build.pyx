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
    # Sum and average of relative permittivities and permeabilities in PML slab
    cdef float sumer, summr, averageer, averagemr 

    for m in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for n in range(n2):
            numID = solid[m ,n]
            sumer += ers[numID]
            summr += mrs[numID]
    averageer = sumer / (n1 * n2)
    averagemr = summr / (n1 * n2)

    return averageer, averagemr
