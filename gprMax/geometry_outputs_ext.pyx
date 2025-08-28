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

cimport numpy as np


cpdef void define_fine_geometry(
                    int nx,
                    int ny,
                    int nz,
                    int xs,
                    int xf,
                    int ys,
                    int yf,
                    int zs,
                    int zf,
                    float dx,
                    float dy,
                    float dz,
                    np.uint32_t[:, :, :, :] ID,
                    np.float32_t[:, :] points,
                    np.uint32_t[:, :] x_lines,
                    np.uint32_t[:] x_materials,
                    np.uint32_t[:, :] y_lines,
                    np.uint32_t[:] y_materials,
                    np.uint32_t[:, :] z_lines,
                    np.uint32_t[:] z_materials
            ):

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t label = 0
    cdef Py_ssize_t counter_x = 0
    cdef Py_ssize_t counter_y = 0
    cdef Py_ssize_t counter_z = 0

    cdef int label_x, label_y, label_z

    for i in range(xs, xf + 1):
        for j in range(ys, yf + 1):
            for k in range(zs, zf + 1):
                points[label][0] = i * dx
                points[label][1] = j * dy
                points[label][2] = k * dz
                if i < xf:
                    # x connectivity
                    label_x = label + (ny + 1) * (nz + 1)
                    x_lines[counter_x][0] = label
                    x_lines[counter_x][1] = label_x
                    # material for the line
                    x_materials[counter_x] = ID[0, i, j, k]
                    counter_x += 1
                if j < yf:
                    label_y = label + nz + 1
                    y_lines[counter_y][0] = label
                    y_lines[counter_y][1] = label_y
                    y_materials[counter_y] = ID[1, i, j, k]
                    counter_y += 1
                if k < zf:
                    label_z = label + 1
                    z_lines[counter_z][0] = label
                    z_lines[counter_z][1] = label_z
                    z_materials[counter_z] = ID[2, i, j, k]
                    counter_z += 1

                label = label + 1


cpdef void define_normal_geometry(
                    int xs,
                    int xf,
                    int ys,
                    int yf,
                    int zs,
                    int zf,
                    int dx,
                    int dy,
                    int dz,
                    np.uint32_t[:, :, :] solid,
                    np.int8_t[:, :, :] srcs_pml,
                    np.int8_t[:, :, :] rxs,
                    np.uint32_t[:] solid_geometry,
                    np.int8_t[:] srcs_pml_geometry,
                    np.int8_t[:] rxs_geometry,
            ):

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t counter = 0

    for k in range(zs, zf, dz):
        for j in range(ys, yf, dy):
            for i in range(xs, xf, dx):
                solid_geometry[counter] = solid[i, j, k]
                srcs_pml_geometry[counter] = srcs_pml[i, j, k]
                rxs_geometry[counter] = rxs[i, j, k]
                counter = counter + 1
