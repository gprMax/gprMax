# Copyright (C) 2015-2021: The University of Edinburgh
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


cpdef void cython_update_electric_os(
        np.float64_t[:, :] updatecoeffsE,
        np.uint32_t[:, :, :, :] ID,
        int face,
        int l_l,
        int l_u,
        int m_l,
        int m_u,
        size_t n_l,
        size_t n_u,
        int nwn,
        size_t lookup_id,
        np.float64_t[:, :, :] field,
        np.float64_t[:, :, :] inc_field,
        size_t co,
        int sign_n,
        int sign_f,
        int mid,
        int r,
        int s,
        int nb,
        int nthreads
):
        """
            Args:

            subgrid: (Subgrid)
            n: (String) the normal to the face to update
            nwn: (Int) number of working cell in the normal direction
            to the face
            lookup_id: (Int) id of the H component we wish to update at
            each node
            field: (Numpy array) main grid field to be updated
            inc_field: (Numpy array) incident sub_grid field
            co: (Int) Coefficient used by gprMax update equations which
            is specific to the field component being updated.
            sign_n: (Int) 1 or -1 sign of the incident field on the near face.
            sign_f: (Int) 1 or -1 sign of the incident field on the far face.
            mid: (Bool) is the H node midway along the lower edge?
            r = self.ratio
            s = self.is_os_sep
            nb = self.n_boundary_cells
        """
        # Comments here as as per left and right face

        cdef Py_ssize_t l, m, l_s, m_s, n_s_l, n_s_r, material_e_l, material_e_r, i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3
        cdef int os
        cdef double inc_n, inc_f

        # surface normal index for the subgrid near face h nodes (left i index)
        n_s_l = nb - s * r - r + r // 2
        # surface normal index for the subgrid far face h nodes (right i index)
        n_s_r = nb + nwn + s * r + r // 2
        # OS at the left face
        os = nb - r * s

        # Iterate over a slice of the main grid using dummy indices
        for l in prange(l_l, l_u, nogil=True, schedule='static', num_threads=nthreads):

            # Calculate the subgrid j component of the H nodes
            # i.e. Hz node of the left or right face
            if mid == 1:
                l_s = os + (l - l_l) * r + r // 2
            # i.e. the Hy node of the left or right face
            else:
                l_s = os + (l - l_l) * r

            for m in range(m_l, m_u):

                # Calculate the subgrid k component of the H nodes
                if mid == 1:
                    m_s = os + (m - m_l) * r
                else:
                    m_s = os + (m - m_l) * r + r // 2

                # left and right
                if face == 2:
                    # main grid index
                    i0, j0, k0 = n_l, l, m
                    # equivalent subgrid index
                    i1, j1, k1 = n_s_l, l_s, m_s
                    i2, j2, k2 = n_u, l, m
                    i3, j3, k3 = n_s_r, l_s, m_s
                # front and back
                if face == 3:
                    i0, j0, k0 = l, n_l, m
                    i1, j1, k1 = l_s, n_s_l, m_s
                    i2, j2, k2 = l, n_u, m
                    i3, j3, k3 = l_s, n_s_r, m_s
                # top bottom
                if face == 1:
                    i0, j0, k0 = l, m, n_l
                    i1, j1, k1 = l_s, m_s, n_s_l
                    i2, j2, k2 = l, m, n_u
                    i3, j3, k3 = l_s, m_s, n_s_r
                # Update the left face

                # Get the material at main grid index
                material_e_l = ID[lookup_id, i0, j0, k0]
                # Get the associated indident field from the subgrid
                inc_n = inc_field[i1, j1, k1] * sign_n
                # Update the main grid E field with the corrected H field
                field[i0, j0, k0] += updatecoeffsE[material_e_l, co] * inc_n

                # Update the right face
                material_e_r = ID[lookup_id, i2, j2, k2]
                inc_f = inc_field[i3, j3, k3] * sign_f
                field[i2, j2, k2] += updatecoeffsE[material_e_r, co] * inc_f

cpdef void cython_update_magnetic_os(
        np.float64_t[:, :] updatecoeffsH,
        np.uint32_t[:, :, :, :] ID,
        int face,
        int l_l,
        int l_u,
        int m_l,
        int m_u,
        size_t n_l,
        size_t n_u,
        int nwn,
        size_t lookup_id,
        np.float64_t[:, :, :] field,
        np.float64_t[:, :, :] inc_field,
        size_t co,
        int sign_n,
        int sign_f,
        int mid,
        int r,
        int s,
        int nb,
        int nthreads
):
        """
        int r ratio,
        int s is_os_sep,
        int nb n_boundary_cells
        """

        cdef Py_ssize_t l, m, l_s, m_s, n_s_l, n_s_r, material_e_l, material_e_r, i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3
        cdef int os
        cdef double inc_n, inc_f

        # i index (normal to os) for the subgrid near face e node
        n_s_l = nb - r * s
        # Normal index for the subgrid far face e node
        n_s_r = nb + nwn + s * r

        # os inner index for the sub grid
        os = nb - r * s

        for l in prange(l_l, l_u, nogil=True, schedule='static', num_threads=nthreads):

            # y coord of the Ex field component
            if mid == 1:
                l_s = os + (l - l_l) * r + r // 2
            # y coord of the Ez field component
            else:
                l_s = os + (l - l_l) * r

            for m in range(m_l, m_u):

                # z coordinate of the Ex node in the subgrid
                if mid == 1:
                    m_s = os + (m - m_l) * r
                else:
                    m_s = os + (m - m_l) * r + r // 2

                # associate the given indices with their i, j, k values

                # left and right
                if face == 2:
                    # main grid index
                    i0, j0, k0 = n_l, l, m
                    # equivalent subgrid index
                    i1, j1, k1 = n_s_l, l_s, m_s
                    i2, j2, k2 = n_u, l, m
                    i3, j3, k3 = n_s_r, l_s, m_s
                # front and back
                if face == 3:
                    i0, j0, k0 = l, n_l, m
                    i1, j1, k1 = l_s, n_s_l, m_s
                    i2, j2, k2 = l, n_u, m
                    i3, j3, k3 = l_s, n_s_r, m_s
                # top bottom
                if face == 1:
                    i0, j0, k0 = l, m, n_l
                    i1, j1, k1 = l_s, m_s, n_s_l
                    i2, j2, k2 = l, m, n_u
                    i3, j3, k3 = l_s, m_s, n_s_r

                material_e_l = ID[lookup_id, i0, j0, k0]
                inc_n = inc_field[i1, j1, k1] * sign_n

                # make sure these are the correct grid
                field[i0, j0, k0] += updatecoeffsH[material_e_l, co] * inc_n

                # Far face
                material_e_r = ID[lookup_id, i2, j2, k2]
                inc_f = inc_field[i3, j3, k3] * sign_f
                field[i2, j2, k2] += updatecoeffsH[material_e_r, co] * inc_f

cpdef void cython_update_is(
        int nwx,
        int nwy,
        int nwz,
        np.float64_t[:, :] updatecoeffsE,
        np.uint32_t[:, :, :, :] ID,
        int n,
        int offset,
        int nwl,
        int nwm,
        int nwn,
        int face,
        np.float64_t[:, :, :] field,
        np.float64_t[:, :] inc_field_l,
        np.float64_t[:, :] inc_field_u,
        Py_ssize_t lookup_id,
        int sign_l,
        int sign_u,
        Py_ssize_t co,
        int nthreads
    ):

        cdef Py_ssize_t l, m, i1, j1, k1, i2, j2, k2, field_material_l, field_material_u, inc_i, inc_j
        cdef double inc_l, inc_u, f_l, f_u
        # for inner faces H nodes are 1 cell before n boundary cells
        cdef int n_o = n + offset

        for l in prange(n, nwl + n, nogil=True, schedule='static', num_threads=nthreads):
            for m in range(n, nwm + n):

                # bottom and top
                if face == 1:
                    i1, j1, k1 = l, m, n_o
                    i2, j2, k2 = l, m, n + nwz
                # left and right
                if face == 2:
                    i1, j1, k1 = n_o, l, m
                    i2, j2, k2 = n + nwx, l, m
                # front and back
                if face == 3:
                    i1, j1, k1 = l, n_o, m
                    i2, j2, k2 = l, n + nwy, m

                inc_i = l - n
                inc_j = m - n

                field_material_l = ID[lookup_id, i1, j1, k1]
                inc_l = inc_field_l[inc_i, inc_j]
                # Additional field at i, j, k
                f_l = updatecoeffsE[field_material_l, co] * inc_l * sign_l
                # Set the new value
                field[i1, j1, k1] += f_l

                field_material_u = ID[lookup_id, i2, j2, k2]
                inc_u = inc_field_u[inc_i, inc_j]
                # Additional field at i, j, k
                f_u = updatecoeffsE[field_material_u, co] * inc_u * sign_u
                # Set the new value
                field[i2, j2, k2] += f_u
