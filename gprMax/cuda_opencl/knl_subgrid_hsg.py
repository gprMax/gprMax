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

from string import Template

# HSG SUBGRID - INNER SURFACE (IS) CORRECTION
#
# Ports update_is() from cython/fields_updates_hsg.pyx.
#
# Applies the incident-field correction from the main grid onto the subgrid
# fields at the inner surface. One call covers an opposite face PAIR: the
# lower face at (i1,j1,k1) and the upper face at (i2,j2,k2).
#
# The Cython version is a prange over l with an inner loop over m; here one
# thread handles one (l, m) pair. Each thread writes two distinct elements,
# so no atomics are needed.
#
#   face = 1 - bottom/top   (normal z)
#   face = 2 - left/right   (normal x)
#   face = 3 - front/back   (normal y)
#
# Array dimensions are passed as runtime arguments rather than baked in as
# macros- the subgrid's dimensions differ from the main grid's, and one
# model may hold several subgrids with different sizes.
#
#   coeffs   : updatecoeffsE or updatecoeffsH, shape [nmat, NY_MATCOEFFS]
#   ID       : shape [6, ID_NX, ID_NY, ID_NZ]
#   field    : shape [F_NX, F_NY, F_NZ]
#   inc_l/u  : shape [INC_NX, INC_NY]  (the precursor arrays)

update_is = {
    "args_cuda": Template(
        """
        __global__ void update_is(
            const int nwx,
            const int nwy,
            const int nwz,
            const int n,
            const int offset,
            const int nwl,
            const int nwm,
            const int face,
            const int sign_l,
            const int sign_u,
            const int co,
            const int lookup_id,
            const int NY_MATCOEFFS,
            const int ID_NX,
            const int ID_NY,
            const int ID_NZ,
            const int F_NX,
            const int F_NY,
            const int F_NZ,
            const int INC_NY,
            const $REAL* __restrict__ coeffs,
            const unsigned int* __restrict__ ID,
            $REAL* __restrict__ field,
            const $REAL* __restrict__ inc_l,
            const $REAL* __restrict__ inc_u
        )
        """
    ),
    "args_opencl": Template(
        """
            const int nwx,
            const int nwy,
            const int nwz,
            const int n,
            const int offset,
            const int nwl,
            const int nwm,
            const int face,
            const int sign_l,
            const int sign_u,
            const int co,
            const int lookup_id,
            const int NY_MATCOEFFS,
            const int ID_NX,
            const int ID_NY,
            const int ID_NZ,
            const int F_NX,
            const int F_NY,
            const int F_NZ,
            const int INC_NY,
            __global const $REAL* restrict coeffs,
            __global const unsigned int* restrict ID,
            __global $REAL* restrict field,
            __global const $REAL* restrict inc_l,
            __global const $REAL* restrict inc_u
        """
    ),
    "func": Template(
        """
    // Inner-surface correction for one opposite face pair of an HSG subgrid.
    //
    // Thread mapping: i in [0, nwl*nwm) -> (l, m) with
    //     l = n + i / nwm      (outer loop in the Cython prange)
    //     m = n + i % nwm      (inner loop)
    //
    // Index arithmetic mirrors update_is() in fields_updates_hsg.pyx exactly,
    // including n_o = n + offset for the H nodes that sit one cell inside
    // the boundary.

    $CUDA_IDX

    if (i >= nwl * nwm) return;

    int l = n + i / nwm;
    int m = n + i % nwm;

    // For inner faces H nodes are 1 cell before n boundary cells
    int n_o = n + offset;

    int i1, j1, k1, i2, j2, k2;

    if (face == 1) {            // bottom and top
        i1 = l;   j1 = m;   k1 = n_o;
        i2 = l;   j2 = m;   k2 = n + nwz;
    } else if (face == 2) {     // left and right
        i1 = n_o; j1 = l;   k1 = m;
        i2 = n + nwx; j2 = l; k2 = m;
    } else {                    // face == 3, front and back
        i1 = l;   j1 = n_o; k1 = m;
        i2 = l;   j2 = n + nwy; k2 = m;
    }

    int inc_i = l - n;
    int inc_j = m - n;
    long inc_idx = (long)inc_i * INC_NY + inc_j;

    // Lower face
    // 64-bit index arithmetic: 6*NX*NY*NZ can exceed 2^31 on large grids,
    // matching the Py_ssize_t indexing the Cython version uses.
    long id_l = (((long)lookup_id * ID_NX + i1) * ID_NY + j1) * ID_NZ + k1;
    int mat_l = ID[id_l];
    $REAL f_l = coeffs[(long)mat_l * NY_MATCOEFFS + co] * inc_l[inc_idx] * sign_l;
    field[((long)i1 * F_NY + j1) * F_NZ + k1] += f_l;

    // Upper face
    long id_u = (((long)lookup_id * ID_NX + i2) * ID_NY + j2) * ID_NZ + k2;
    int mat_u = ID[id_u];
    $REAL f_u = coeffs[(long)mat_u * NY_MATCOEFFS + co] * inc_u[inc_idx] * sign_u;
    field[((long)i2 * F_NY + j2) * F_NZ + k2] += f_u;
    """
    ),
}


# HSG SUBGRID - OUTER SURFACE (OS) CORRECTION, ELECTRIC
#
# Ports update_electric_os() from cython/fields_updates_hsg.pyx.
# H nodes sit at half-cell offsets in the normal direction, which is the only
# difference between the electric and magnetic variants.
#
#   coeffs    : updatecoeffsE, shape [nmat, NY_MATCOEFFS]
#   ID, field : MAIN grid arrays
#   inc_field : SUBGRID field array

update_electric_os = {
    "args_cuda": Template(
        """
        __global__ void update_electric_os(
            const int face,
            const int l_l,
            const int l_u,
            const int m_l,
            const int m_u,
            const int n_l,
            const int n_u,
            const int nwn,
            const int lookup_id,
            const int co,
            const int sign_n,
            const int sign_f,
            const int mid,
            const int r,
            const int s,
            const int nb,
            const int NY_MATCOEFFS,
            const int ID_NX,
            const int ID_NY,
            const int ID_NZ,
            const int F_NY,
            const int F_NZ,
            const int S_NY,
            const int S_NZ,
            const $REAL* __restrict__ coeffs,
            const unsigned int* __restrict__ ID,
            $REAL* __restrict__ field,
            const $REAL* __restrict__ inc_field
        )
        """
    ),
    "args_opencl": Template(
        """
            const int face,
            const int l_l,
            const int l_u,
            const int m_l,
            const int m_u,
            const int n_l,
            const int n_u,
            const int nwn,
            const int lookup_id,
            const int co,
            const int sign_n,
            const int sign_f,
            const int mid,
            const int r,
            const int s,
            const int nb,
            const int NY_MATCOEFFS,
            const int ID_NX,
            const int ID_NY,
            const int ID_NZ,
            const int F_NY,
            const int F_NZ,
            const int S_NY,
            const int S_NZ,
            __global const $REAL* restrict coeffs,
            __global const unsigned int* restrict ID,
            __global $REAL* restrict field,
            __global const $REAL* restrict inc_field
        """
    ),
    "func": Template(
        """
    // Outer-surface correction: the subgrid's own fields correct the MAIN
    // grid. Ports update_electric_os() from fields_updates_hsg.pyx.
    //
    // The two grids have different resolutions, so each main-grid face point
    // (l, m) maps to a subgrid node by stepping r cells and offsetting by
    // r/2 for components that sit at cell midpoints (selected by `mid`).
    //
    // Thread mapping: t in [0, L*M) -> (l, m) with
    //     l = l_l + t / M,  m = m_l + t % M,  M = m_u - m_l
    //
    // field/ID index the MAIN grid; inc_field indexes the SUBGRID.

    $CUDA_IDX

    int L = l_u - l_l;
    int M = m_u - m_l;
    if (i >= L * M) return;

    int l = l_l + i / M;
    int m = m_l + i % M;

    // Surface-normal indices of the subgrid nodes (electric staggering)
    int n_s_l = nb - s * r - r + r / 2;
    int n_s_r = nb + nwn + s * r + r / 2;
    // OS at the left face
    int os = nb - r * s;

    int l_s, m_s;
    if (mid == 1) {
        l_s = os + (l - l_l) * r + r / 2;
        m_s = os + (m - m_l) * r;
    } else {
        l_s = os + (l - l_l) * r;
        m_s = os + (m - m_l) * r + r / 2;
    }

    int i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3;

    if (face == 2) {            // left and right
        i0 = n_l;   j0 = l;    k0 = m;
        i1 = n_s_l; j1 = l_s;  k1 = m_s;
        i2 = n_u;   j2 = l;    k2 = m;
        i3 = n_s_r; j3 = l_s;  k3 = m_s;
    } else if (face == 3) {    // front and back
        i0 = l;    j0 = n_l;   k0 = m;
        i1 = l_s;  j1 = n_s_l; k1 = m_s;
        i2 = l;    j2 = n_u;   k2 = m;
        i3 = l_s;  j3 = n_s_r; k3 = m_s;
    } else {                   // face == 1, top and bottom
        i0 = l;    j0 = m;     k0 = n_l;
        i1 = l_s;  j1 = m_s;   k1 = n_s_l;
        i2 = l;    j2 = m;     k2 = n_u;
        i3 = l_s;  j3 = m_s;   k3 = n_s_r;
    }

    // Near face
    long id_n = (((long)lookup_id * ID_NX + i0) * ID_NY + j0) * ID_NZ + k0;
    int mat_n = ID[id_n];
    $REAL inc_n = inc_field[((long)i1 * S_NY + j1) * S_NZ + k1] * sign_n;
    field[((long)i0 * F_NY + j0) * F_NZ + k0] += coeffs[(long)mat_n * NY_MATCOEFFS + co] * inc_n;

    // Far face
    long id_f = (((long)lookup_id * ID_NX + i2) * ID_NY + j2) * ID_NZ + k2;
    int mat_f = ID[id_f];
    $REAL inc_f = inc_field[((long)i3 * S_NY + j3) * S_NZ + k3] * sign_f;
    field[((long)i2 * F_NY + j2) * F_NZ + k2] += coeffs[(long)mat_f * NY_MATCOEFFS + co] * inc_f;
    """
    ),
}


# HSG SUBGRID - OUTER SURFACE (OS) CORRECTION, MAGNETIC
#
# Ports update_magnetic_os() from cython/fields_updates_hsg.pyx.
# E nodes sit on cell boundaries, which is the only difference between the
# electric and magnetic variants.
#
#   coeffs    : updatecoeffsH, shape [nmat, NY_MATCOEFFS]
#   ID, field : MAIN grid arrays
#   inc_field : SUBGRID field array

update_magnetic_os = {
    "args_cuda": Template(
        """
        __global__ void update_magnetic_os(
            const int face,
            const int l_l,
            const int l_u,
            const int m_l,
            const int m_u,
            const int n_l,
            const int n_u,
            const int nwn,
            const int lookup_id,
            const int co,
            const int sign_n,
            const int sign_f,
            const int mid,
            const int r,
            const int s,
            const int nb,
            const int NY_MATCOEFFS,
            const int ID_NX,
            const int ID_NY,
            const int ID_NZ,
            const int F_NY,
            const int F_NZ,
            const int S_NY,
            const int S_NZ,
            const $REAL* __restrict__ coeffs,
            const unsigned int* __restrict__ ID,
            $REAL* __restrict__ field,
            const $REAL* __restrict__ inc_field
        )
        """
    ),
    "args_opencl": Template(
        """
            const int face,
            const int l_l,
            const int l_u,
            const int m_l,
            const int m_u,
            const int n_l,
            const int n_u,
            const int nwn,
            const int lookup_id,
            const int co,
            const int sign_n,
            const int sign_f,
            const int mid,
            const int r,
            const int s,
            const int nb,
            const int NY_MATCOEFFS,
            const int ID_NX,
            const int ID_NY,
            const int ID_NZ,
            const int F_NY,
            const int F_NZ,
            const int S_NY,
            const int S_NZ,
            __global const $REAL* restrict coeffs,
            __global const unsigned int* restrict ID,
            __global $REAL* restrict field,
            __global const $REAL* restrict inc_field
        """
    ),
    "func": Template(
        """
    // Outer-surface correction: the subgrid's own fields correct the MAIN
    // grid. Ports update_magnetic_os() from fields_updates_hsg.pyx.
    //
    // The two grids have different resolutions, so each main-grid face point
    // (l, m) maps to a subgrid node by stepping r cells and offsetting by
    // r/2 for components that sit at cell midpoints (selected by `mid`).
    //
    // Thread mapping: t in [0, L*M) -> (l, m) with
    //     l = l_l + t / M,  m = m_l + t % M,  M = m_u - m_l
    //
    // field/ID index the MAIN grid; inc_field indexes the SUBGRID.

    $CUDA_IDX

    int L = l_u - l_l;
    int M = m_u - m_l;
    if (i >= L * M) return;

    int l = l_l + i / M;
    int m = m_l + i % M;

    // Surface-normal indices of the subgrid nodes (magnetic staggering)
    int n_s_l = nb - r * s;
    int n_s_r = nb + nwn + s * r;
    // OS at the left face
    int os = nb - r * s;

    int l_s, m_s;
    if (mid == 1) {
        l_s = os + (l - l_l) * r + r / 2;
        m_s = os + (m - m_l) * r;
    } else {
        l_s = os + (l - l_l) * r;
        m_s = os + (m - m_l) * r + r / 2;
    }

    int i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3;

    if (face == 2) {            // left and right
        i0 = n_l;   j0 = l;    k0 = m;
        i1 = n_s_l; j1 = l_s;  k1 = m_s;
        i2 = n_u;   j2 = l;    k2 = m;
        i3 = n_s_r; j3 = l_s;  k3 = m_s;
    } else if (face == 3) {    // front and back
        i0 = l;    j0 = n_l;   k0 = m;
        i1 = l_s;  j1 = n_s_l; k1 = m_s;
        i2 = l;    j2 = n_u;   k2 = m;
        i3 = l_s;  j3 = n_s_r; k3 = m_s;
    } else {                   // face == 1, top and bottom
        i0 = l;    j0 = m;     k0 = n_l;
        i1 = l_s;  j1 = m_s;   k1 = n_s_l;
        i2 = l;    j2 = m;     k2 = n_u;
        i3 = l_s;  j3 = m_s;   k3 = n_s_r;
    }

    // Near face
    long id_n = (((long)lookup_id * ID_NX + i0) * ID_NY + j0) * ID_NZ + k0;
    int mat_n = ID[id_n];
    $REAL inc_n = inc_field[((long)i1 * S_NY + j1) * S_NZ + k1] * sign_n;
    field[((long)i0 * F_NY + j0) * F_NZ + k0] += coeffs[(long)mat_n * NY_MATCOEFFS + co] * inc_n;

    // Far face
    long id_f = (((long)lookup_id * ID_NX + i2) * ID_NY + j2) * ID_NZ + k2;
    int mat_f = ID[id_f];
    $REAL inc_f = inc_field[((long)i3 * S_NY + j3) * S_NZ + k3] * sign_f;
    field[((long)i2 * F_NY + j2) * F_NZ + k2] += coeffs[(long)mat_f * NY_MATCOEFFS + co] * inc_f;
    """
    ),
}
