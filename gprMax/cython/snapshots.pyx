# cython: cdivision=True
# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley, 
#                          and Nathan Mannall
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

from cython.parallel import prange

from gprMax.config cimport float_or_double


cpdef void calculate_snapshot_fields(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint isEx,
    bint isEy,
    bint isEz,
    bint isHx,
    bint isHy,
    bint isHz,
    float_or_double[:, :, ::1] Exslice,
    float_or_double[:, :, ::1] Eyslice,
    float_or_double[:, :, ::1] Ezslice,
    float_or_double[:, :, ::1] Hxslice,
    float_or_double[:, :, ::1] Hyslice,
    float_or_double[:, :, ::1] Hzslice,
    float_or_double[:, :, ::1] Exsnap,
    float_or_double[:, :, ::1] Eysnap,
    float_or_double[:, :, ::1] Ezsnap,
    float_or_double[:, :, ::1] Hxsnap,
    float_or_double[:, :, ::1] Hysnap,
    float_or_double[:, :, ::1] Hzsnap,
    int sx=1,
    int sy=1,
    int sz=1
):
    """Calculates electric and magnetic values at points from averaging values
        in cells.

    Args:
        nx, ny, nz: ints for size of snapshot array.
        nthreads: int for number of threads to use.
        is: boolean to determine whether that field snapshot is required.
        slice: memoryviews to access slices of field arrays.
        snap: memoryviews to access snapshot arrays.
        sx, sy, sz: neighbour-offset strides along x, y, z (1 = genuine
            averaging with the +1 neighbour, as in 3D/2D-TM mode; 0 = no
            genuine neighbour exists along that axis, so both terms of any
            pair on that axis collapse to the same index - used for a 2D
            TE-mode model's invariant axis, where there is only ever one
            real field value flanked by forced-zero boundary padding, not
            a second genuine value to average against. Defaults to 1 for
            every axis, reproducing the original (pre-2D-TE-mode) formula
            exactly.
    """

    cdef Py_ssize_t i, j, k

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(ny):
            for k in range(nz):
                # The electric field component value at a point comes from the
                # average of the 4 electric field component values in that cell.
                if isEx:
                    Exsnap[i, j, k] = (Exslice[i, j, k] +
                                       Exslice[i, j + sy, k] +
                                       Exslice[i, j, k + sz] +
                                       Exslice[i, j + sy, k + sz]) / 4
                if isEy:
                    Eysnap[i, j, k] = (Eyslice[i, j, k] +
                                       Eyslice[i + sx, j, k] +
                                       Eyslice[i, j, k + sz] +
                                       Eyslice[i + sx, j, k + sz]) / 4
                if isEz:
                    Ezsnap[i, j, k] = (Ezslice[i, j, k] +
                                       Ezslice[i + sx, j, k] +
                                       Ezslice[i, j + sy, k] +
                                       Ezslice[i + sx, j + sy, k]) / 4

                # The magnetic field component value at a point comes from
                # average of 2 magnetic field component values in that cell and
                # the neighbouring cell.
                if isHx:
                    Hxsnap[i, j, k] = (Hxslice[i, j, k] +
                                       Hxslice[i + sx, j, k]) / 2
                if isHy:
                    Hysnap[i, j, k] = (Hyslice[i, j, k] +
                                       Hyslice[i, j + sy, k]) / 2
                if isHz:
                    Hzsnap[i, j, k] = (Hzslice[i, j, k] +
                                       Hzslice[i, j, k + sz]) / 2
