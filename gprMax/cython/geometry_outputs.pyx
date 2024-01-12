# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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


cpdef write_lines(
    float xs,
    float ys,
    float zs,
    int nx,
    int ny,
    int nz,
    float dx,
    float dy,
    float dz,
    np.uint32_t[:, :, :, :] ID
):
    """Generates arrays with to be written as lines (cell edges) to a VTK file.

    Args:
        xs, ys, zs: float for starting coordinates of geometry view in metres.
        nx, ny, nz: int for size of the volume in cells.
        dx, dy, dz: float for spatial discretisation of geometry view in metres.
        ID: memoryview of sampled ID array according to geometry view spatial
            discretisation.

    Returns:
        x, y, z: 1D nparrays with coordinates of the vertex of the lines.
        lines: nparray of material IDs for each line (cell edge) required.
    """

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t lc = 0 # Line counter
    cdef Py_ssize_t pc = 0 # Point counter
    cdef Py_ssize_t n_x_lines = 0
    cdef Py_ssize_t n_y_lines = 0
    cdef Py_ssize_t n_z_lines = 0
    cdef Py_ssize_t n_lines = 0
    cdef Py_ssize_t n_points = 0

    n_x_lines = nx * (ny + 1) * (nz + 1)
    n_y_lines = ny * (nx + 1) * (nz + 1)
    n_z_lines = nz * (nx + 1) * (ny + 1)
    n_lines = n_x_lines + n_y_lines + n_z_lines
    n_points = 2 * n_lines # A line is defined by 2 points

    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)
    lines = np.zeros(n_lines)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # x-direction cell edge
                # Material ID of line
                lines[lc] = ID[0][i, j, k]
                # Set the starting point position of the edge
                x[pc], y[pc], z[pc] = i, j, k
                # Next point
                pc += 1
                # Set the end point position of the edge
                x[pc], y[pc], z[pc] = (i + 1), j, k
                # Next point
                pc += 1
                # Next line
                lc += 1

                # y-direction cell edge
                lines[lc] = ID[1, i, j, k]
                x[pc], y[pc], z[pc] = i, j, k
                pc += 1
                x[pc], y[pc], z[pc] = i, (j + 1), k
                pc += 1
                lc += 1

                # z-direction cell edge
                lines[lc] = ID[2, i, j, k]
                x[pc], y[pc], z[pc] = i, j, k
                pc += 1
                x[pc], y[pc], z[pc] = i, j, (k + 1)
                pc += 1
                lc += 1

    x *= dx
    y *= dy
    z *= dz

    x += xs
    y += ys
    z += zs

    return x, y, z, lines
