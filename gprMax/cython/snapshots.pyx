# Copyright (C) 2015-2021: The University of Edinburgh
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

from gprMax.config cimport float_or_double


cpdef void calculate_snapshot_fields(
                    int nx,
                    int ny,
                    int nz,
                    float_or_double[:, :, ::1] sliceEx,
                    float_or_double[:, :, ::1] sliceEy,
                    float_or_double[:, :, ::1] sliceEz,
                    float_or_double[:, :, ::1] sliceHx,
                    float_or_double[:, :, ::1] sliceHy,
                    float_or_double[:, :, ::1] sliceHz,
                    float_or_double[:, :, ::1] snapEx,
                    float_or_double[:, :, ::1] snapEy,
                    float_or_double[:, :, ::1] snapEz,
                    float_or_double[:, :, ::1] snapHx,
                    float_or_double[:, :, ::1] snapHy,
                    float_or_double[:, :, ::1] snapHz
            ):
    """This function calculates electric and magnetic values at points from
        averaging values in cells.

    Args:
        nx, ny, nz (int): Size of snapshot array
        sliceEx, sliceEy, sliceEz,
            sliceHx, sliceHy, sliceHz (memoryview): Access to slices of field arrays
        snapEx, snapEy, snapEz,
            snapHx, snapHy, snapHz (memoryview): Access to snapshot arrays
    """

    cdef Py_ssize_t i, j, k

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # The electric field component value at a point comes from the
                # average of the 4 electric field component values in that cell.
                snapEx[i, j, k] = (sliceEx[i, j, k] + sliceEx[i, j + 1, k] +
                                    sliceEx[i, j, k + 1] + sliceEx[i, j + 1, k + 1]) / 4
                snapEy[i, j, k] = (sliceEy[i, j, k] + sliceEy[i + 1, j, k] +
                                    sliceEy[i, j, k + 1] + sliceEy[i + 1, j, k + 1]) / 4
                snapEz[i, j, k] = (sliceEz[i, j, k] + sliceEz[i + 1, j, k] +
                                    sliceEz[i, j + 1, k] + sliceEz[i + 1, j + 1, k]) / 4

                # The magnetic field component value at a point comes from
                # average of 2 magnetic field component values in that cell and
                # the neighbouring cell.
                snapHx[i, j, k] = (sliceHx[i, j, k] + sliceHx[i + 1, j, k]) / 2
                snapHy[i, j, k] = (sliceHy[i, j, k] + sliceHy[i, j + 1, k]) / 2
                snapHz[i, j, k] = (sliceHz[i, j, k] + sliceHz[i, j, k + 1]) / 2
