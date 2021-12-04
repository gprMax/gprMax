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

import numpy as np
cimport numpy as np


cpdef write_lines(int nx, int ny, int nz, int dx, int dy, int dz,
                  np.uint32_t[:, :, :, :] ID):
    """This function generates arrays with to be written as lines (cell edges) 
        to a VTK file.

    Args:
        nx, ny, nz (int): Size of the volume in cells
        dx, dy, dz (int): Spatial discretisation of geometry view in cells
        ID (nparray): Sampled ID array according to geometry view spatial 
                            discretisation

    Returns:
        x, y, z (nparray): 1D arrays with coordinates of the vertex of the lines
        lines (nparray): array of material IDs for each line (cell edge) required
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
                x[pc], y[pc], z[pc] = i * dx, j * dy, k * dz
                # Next point
                pc += 1
                # Set the end point position of the edge
                x[pc], y[pc], z[pc] = (i + 1) * dx, j * dy, k * dz
                # Next point
                pc += 1
                # Next line
                lc += 1

                # y-direction cell edge
                lines[lc] = ID[1, i, j, k]
                x[pc], y[pc], z[pc] = i * dx, j * dy, k * dz
                pc += 1
                x[pc], y[pc], z[pc] = i * dx, (j + 1) * dy, k * dz
                pc += 1
                lc += 1

                # z-direction cell edge
                lines[lc] = ID[2, i, j, k]
                x[pc], y[pc], z[pc] = i * dx, j * dy, k * dz
                pc += 1
                x[pc], y[pc], z[pc] = i * dx, j * dy, (k + 1) * dz
                pc += 1
                lc += 1

    return x, y, z, lines 
