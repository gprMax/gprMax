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

# what is going on in this file (SUMMARY) -
# TF/SF BOUNDARY INJECTION KERNELS -

# Ports- applyTFSFMagnetic(), applyTFSFMagnetic_axial(),
#        applyTFSFElectric(), applyTFSFElectric_axial()
#        from plane_wave.pyx
#
# 24 kernels total-
#   12 standard (homogeneous) - scalar coefficients
#   12 axial    (heterogeneous) - per-cell GID lookup
#
# One kernel per face per field type:
#   x_low_H,  x_high_H,  y_low_H,  y_high_H,  z_low_H,  z_high_H
#   x_low_E,  x_high_E,  y_low_E,  y_high_E,  z_low_E,  z_high_E
#
# Thread layout for all kernels-
#   Thread t handles one cell on the face.
#   x-faces: face patch is (y_stop - y_start + 1) x (z_stop - z_start + 1)
#     t = (j - y_start) * NZ_FACE + (k - z_start)
#   y-faces: face patch is (x_stop - x_start + 1) x (z_stop - z_start + 1)
#     t = (i - x_start) * NZ_FACE + (k - z_start)
#   z-faces: face patch is (x_stop - x_start + 1) x (y_stop - y_start + 1)
#     t = (i - x_start) * NY_FACE + (j - y_start)
#
# Coefficient mapping (from Cython)-
#   Standard-  coef_H_yx = updatecoeffsH[1] (DBx)
#              coef_H_xy = updatecoeffsH[2] (DBy)
#              coef_H_xz = updatecoeffsH[3] (DBz)
#              (same indices for coef_H_zy, coef_H_zx, coef_H_yz)
#   Axial-     GID[4,i,j,k] for Hy, GID[5,i,j,k] for Hz, GID[3,i,j,k] for Hx
#              Column: 1 for x-faces, 2 for y-faces, 3 for z-faces
#
# GID 3D array layout- ID[component, x, y, z] - shape [6, Nx, Ny, Nz]
# Flat index GID[comp, i, j, k] = GID[comp*Nx*Ny*Nz + i*Ny*Nz + j*Nz + k]
# Accessed via IDX4D_ID macro from knl_common_base.tmpl.
#
# updatecoeffsH/E are in constant memory - accessed via IDX2D_MAT macro.
# Both already resident on GPU from existing field update kernels.
# No new allocations needed for 3D side.

# Each kernel shares the same base args with small variations.
# Standard H kernels: scalar coef_H_* args
# Axial H kernels: O_axial int arg + ID pointer, no scalar coefs
# Standard E kernels: scalar coef_E_* args
# Axial E kernels: O_axial int arg + ID pointer, no scalar coefs


def _std_H_args_cuda(name):
    return Template(f"""
        __global__ void __launch_bounds__(256, 4) {name}(
            const int NY_FIELDS,
            const int NZ_FIELDS,
            const int NY_FACE,
            const int NZ_FACE,
            const int x_start, const int x_stop,
            const int y_start, const int y_stop,
            const int z_start, const int z_stop,
            const int m_x, const int m_y, const int m_z,
            const int Ox, const int Oy, const int Oz,
            const $REAL coef_H_1,
            const $REAL coef_H_2,
            $REAL* __restrict__ Hx,
            $REAL* __restrict__ Hy,
            $REAL* __restrict__ Hz,
            const $REAL* __restrict__ E_x,
            const $REAL* __restrict__ E_y,
            const $REAL* __restrict__ E_z
        )
    """)

def _axial_H_args_cuda(name):
    return Template(f"""
        __global__ void __launch_bounds__(256, 4) {name}(
            const int NY_FIELDS,
            const int NZ_FIELDS,
            const int NY_FACE,
            const int NZ_FACE,
            const int x_start, const int x_stop,
            const int y_start, const int y_stop,
            const int z_start, const int z_stop,
            const int m_x, const int m_y, const int m_z,
            const int Ox, const int Oy, const int Oz,
            const int O_axial,
            $REAL* __restrict__ Hx,
            $REAL* __restrict__ Hy,
            $REAL* __restrict__ Hz,
            const $REAL* __restrict__ E_x,
            const $REAL* __restrict__ E_y,
            const $REAL* __restrict__ E_z,
            const unsigned int* __restrict__ ID
        )
    """)

def _std_E_args_cuda(name):
    return Template(f"""
        __global__ void __launch_bounds__(256, 4) {name}(
            const int NY_FIELDS,
            const int NZ_FIELDS,
            const int NY_FACE,
            const int NZ_FACE,
            const int x_start, const int x_stop,
            const int y_start, const int y_stop,
            const int z_start, const int z_stop,
            const int m_x, const int m_y, const int m_z,
            const int Ox, const int Oy, const int Oz,
            const $REAL coef_E_1,
            const $REAL coef_E_2,
            $REAL* __restrict__ Ex,
            $REAL* __restrict__ Ey,
            $REAL* __restrict__ Ez,
            const $REAL* __restrict__ H_x,
            const $REAL* __restrict__ H_y,
            const $REAL* __restrict__ H_z
        )
    """)

def _axial_E_args_cuda(name):
    return Template(f"""
        __global__ void __launch_bounds__(256, 4) {name}(
            const int NY_FIELDS,
            const int NZ_FIELDS,
            const int NY_FACE,
            const int NZ_FACE,
            const int x_start, const int x_stop,
            const int y_start, const int y_stop,
            const int z_start, const int z_stop,
            const int m_x, const int m_y, const int m_z,
            const int Ox, const int Oy, const int Oz,
            const int O_axial,
            $REAL* __restrict__ Ex,
            $REAL* __restrict__ Ey,
            $REAL* __restrict__ Ez,
            const $REAL* __restrict__ H_x,
            const $REAL* __restrict__ H_y,
            const $REAL* __restrict__ H_z,
            const unsigned int* __restrict__ ID
        )
    """)

def _std_H_args_opencl(name):
    return Template(f"""
        const int NY_FIELDS, const int NZ_FIELDS,
        const int NY_FACE, const int NZ_FACE,
        const int x_start, const int x_stop,
        const int y_start, const int y_stop,
        const int z_start, const int z_stop,
        const int m_x, const int m_y, const int m_z,
        const int Ox, const int Oy, const int Oz,
        const $REAL coef_H_1, const $REAL coef_H_2,
        __global $REAL* restrict Hx,
        __global $REAL* restrict Hy,
        __global $REAL* restrict Hz,
        __global const $REAL* restrict E_x,
        __global const $REAL* restrict E_y,
        __global const $REAL* restrict E_z
    """)

def _axial_H_args_opencl(name):
    return Template(f"""
        const int NY_FIELDS, const int NZ_FIELDS,
        const int NY_FACE, const int NZ_FACE,
        const int x_start, const int x_stop,
        const int y_start, const int y_stop,
        const int z_start, const int z_stop,
        const int m_x, const int m_y, const int m_z,
        const int Ox, const int Oy, const int Oz,
        const int O_axial,
        __global $REAL* restrict Hx,
        __global $REAL* restrict Hy,
        __global $REAL* restrict Hz,
        __global const $REAL* restrict E_x,
        __global const $REAL* restrict E_y,
        __global const $REAL* restrict E_z,
        __global const unsigned int* restrict ID
    """)

def _std_E_args_opencl(name):
    return Template(f"""
        const int NY_FIELDS, const int NZ_FIELDS,
        const int NY_FACE, const int NZ_FACE,
        const int x_start, const int x_stop,
        const int y_start, const int y_stop,
        const int z_start, const int z_stop,
        const int m_x, const int m_y, const int m_z,
        const int Ox, const int Oy, const int Oz,
        const $REAL coef_E_1, const $REAL coef_E_2,
        __global $REAL* restrict Ex,
        __global $REAL* restrict Ey,
        __global $REAL* restrict Ez,
        __global const $REAL* restrict H_x,
        __global const $REAL* restrict H_y,
        __global const $REAL* restrict H_z
    """)

def _axial_E_args_opencl(name):
    return Template(f"""
        const int NY_FIELDS, const int NZ_FIELDS,
        const int NY_FACE, const int NZ_FACE,
        const int x_start, const int x_stop,
        const int y_start, const int y_stop,
        const int z_start, const int z_stop,
        const int m_x, const int m_y, const int m_z,
        const int Ox, const int Oy, const int Oz,
        const int O_axial,
        __global $REAL* restrict Ex,
        __global $REAL* restrict Ey,
        __global $REAL* restrict Ez,
        __global const $REAL* restrict H_x,
        __global const $REAL* restrict H_y,
        __global const $REAL* restrict H_z,
        __global const unsigned int* restrict ID
    """)

def _std_H_args_metal(name):
    return Template(f"""
        kernel void {name}(
            device const int& NY_FIELDS, device const int& NZ_FIELDS,
            device const int& NY_FACE, device const int& NZ_FACE,
            device const int& x_start, device const int& x_stop,
            device const int& y_start, device const int& y_stop,
            device const int& z_start, device const int& z_stop,
            device const int& m_x, device const int& m_y, device const int& m_z,
            device const int& Ox, device const int& Oy, device const int& Oz,
            device const $REAL& coef_H_1, device const $REAL& coef_H_2,
            device $REAL* Hx, device $REAL* Hy, device $REAL* Hz,
            device const $REAL* E_x, device const $REAL* E_y, device const $REAL* E_z,
            uint t [[thread_position_in_grid]]
        )
    """)

def _axial_H_args_metal(name):
    return Template(f"""
        kernel void {name}(
            device const int& NY_FIELDS, device const int& NZ_FIELDS,
            device const int& NY_FACE, device const int& NZ_FACE,
            device const int& x_start, device const int& x_stop,
            device const int& y_start, device const int& y_stop,
            device const int& z_start, device const int& z_stop,
            device const int& m_x, device const int& m_y, device const int& m_z,
            device const int& Ox, device const int& Oy, device const int& Oz,
            device const int& O_axial,
            device $REAL* Hx, device $REAL* Hy, device $REAL* Hz,
            device const $REAL* E_x, device const $REAL* E_y, device const $REAL* E_z,
            device const unsigned int* ID,
            uint t [[thread_position_in_grid]]
        )
    """)

def _std_E_args_metal(name):
    return Template(f"""
        kernel void {name}(
            device const int& NY_FIELDS, device const int& NZ_FIELDS,
            device const int& NY_FACE, device const int& NZ_FACE,
            device const int& x_start, device const int& x_stop,
            device const int& y_start, device const int& y_stop,
            device const int& z_start, device const int& z_stop,
            device const int& m_x, device const int& m_y, device const int& m_z,
            device const int& Ox, device const int& Oy, device const int& Oz,
            device const $REAL& coef_E_1, device const $REAL& coef_E_2,
            device $REAL* Ex, device $REAL* Ey, device $REAL* Ez,
            device const $REAL* H_x, device const $REAL* H_y, device const $REAL* H_z,
            uint t [[thread_position_in_grid]]
        )
    """)

def _axial_E_args_metal(name):
    return Template(f"""
        kernel void {name}(
            device const int& NY_FIELDS, device const int& NZ_FIELDS,
            device const int& NY_FACE, device const int& NZ_FACE,
            device const int& x_start, device const int& x_stop,
            device const int& y_start, device const int& y_stop,
            device const int& z_start, device const int& z_stop,
            device const int& m_x, device const int& m_y, device const int& m_z,
            device const int& Ox, device const int& Oy, device const int& Oz,
            device const int& O_axial,
            device $REAL* Ex, device $REAL* Ey, device $REAL* Ez,
            device const $REAL* H_x, device const $REAL* H_y, device const $REAL* H_z,
            device const unsigned int* ID,
            uint t [[thread_position_in_grid]]
        )
    """)


# STANDARD MAGNETIC - X_LOW FACE

# Cython reference (applyTFSFMagnetic, i = x_start):
#   for j in [y_start, y_stop+1):  for k in [z_start, z_stop):
#     index = m_x*(i-Ox) + m_y*(j-Oy) + m_z*(k-Oz)
#     Hy[i-1, j, k] -= coef_H_yx * E_z[index]   coef_H_yx = updatecoeffsH[1]
#   for j in [y_start, y_stop):    for k in [z_start, z_stop+1):
#     Hz[i-1, j, k] += coef_H_zx * E_y[index]   coef_H_zx = updatecoeffsH[1]
#
# Thread t handles one (j,k) pair. Face patch: NY_FACE x NZ_FACE
# Two corrections per thread (Hy and Hz) with different loop bounds.


inject_std_xlow_H = {
    "name": "inject_std_xlow_H",
    "args_cuda":   _std_H_args_cuda("inject_std_xlow_H"),
    "args_opencl": _std_H_args_opencl("inject_std_xlow_H"),
    "args_metal":  _std_H_args_metal("inject_std_xlow_H"),
    "func": Template("""
    // Standard TF/SF magnetic correction - x_low face.
    // Ports applyTFSFMagnetic() x_start section.
    //
    // Corrections at i = x_start:
    //   Hy[i-1, j, k] -= coef_H_yx * E_z[index]   j in [y_start,y_stop], k in [z_start,z_stop)
    //   Hz[i-1, j, k] += coef_H_zx * E_y[index]   j in [y_start,y_stop), k in [z_start,z_stop]
    //
    // coef_H_1 = updatecoeffsH[1] (DBx - passed as kernel arg)
    // coef_H_2 = updatecoeffsH[1] (DBx - same value, kept for clarity)
    // index = m_x*(x_start-Ox) + m_y*(j-Oy) + m_z*(k-Oz)

    $TFSF_IDX

    int j = t / NZ_FACE + y_start;
    int k = t % NZ_FACE + z_start;

    int i = x_start;
    int index;

    // Hy correction - j in [y_start, y_stop], k in [z_start, z_stop)
    if (j <= y_stop && k < z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hy[IDX3D_FIELDS(i-1, j, k)] -= coef_H_1 * E_z[index];
    }

    // Hz correction - j in [y_start, y_stop), k in [z_start, z_stop]
    if (j < y_stop && k <= z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hz[IDX3D_FIELDS(i-1, j, k)] += coef_H_2 * E_y[index];
    }
    """),
}



# STANDARD MAGNETIC - X_HIGH FACE

# Cython (i = x_stop):
#   Hy[i, j, k] += coef_H_yx * E_z[index]   sign FLIPPED vs x_low
#   Hz[i, j, k] -= coef_H_zx * E_y[index]   sign FLIPPED vs x_low


inject_std_xhigh_H = {
    "name": "inject_std_xhigh_H",
    "args_cuda":   _std_H_args_cuda("inject_std_xhigh_H"),
    "args_opencl": _std_H_args_opencl("inject_std_xhigh_H"),
    "args_metal":  _std_H_args_metal("inject_std_xhigh_H"),
    "func": Template("""
    // Standard TF/SF magnetic correction - x_high face.
    // At i = x_stop: signs flipped vs x_low.
    //   Hy[i, j, k] += coef_H_yx * E_z[index]
    //   Hz[i, j, k] -= coef_H_zx * E_y[index]

    $TFSF_IDX

    int j = t / NZ_FACE + y_start;
    int k = t % NZ_FACE + z_start;

    int i = x_stop;
    int index;

    if (j <= y_stop && k < z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hy[IDX3D_FIELDS(i, j, k)] += coef_H_1 * E_z[index];
    }

    if (j < y_stop && k <= z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hz[IDX3D_FIELDS(i, j, k)] -= coef_H_2 * E_y[index];
    }
    """),
}


# STANDARD MAGNETIC - Y_LOW FACE
# Cython (j = y_start):
#   Hx[i, j-1, k] += coef_H_xy * E_z[index]   coef_H_xy = updatecoeffsH[2]
#   Hz[i, j-1, k] -= coef_H_zy * E_x[index]   coef_H_zy = updatecoeffsH[2]


inject_std_ylow_H = {
    "name": "inject_std_ylow_H",
    "args_cuda":   _std_H_args_cuda("inject_std_ylow_H"),
    "args_opencl": _std_H_args_opencl("inject_std_ylow_H"),
    "args_metal":  _std_H_args_metal("inject_std_ylow_H"),
    "func": Template("""
    // Standard TF/SF magnetic correction - y_low face.
    // At j = y_start:
    //   Hx[i, j-1, k] += coef_H_xy * E_z[index]   i in [x_start,x_stop], k in [z_start,z_stop)
    //   Hz[i, j-1, k] -= coef_H_zy * E_x[index]   i in [x_start,x_stop), k in [z_start,z_stop]
    //
    // coef_H_1 = updatecoeffsH[2] (DBy)
    // coef_H_2 = updatecoeffsH[2] (DBy)
    // Thread t: i = t/NZ_FACE + x_start, k = t%NZ_FACE + z_start

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int k = t % NZ_FACE + z_start;

    int j = y_start;
    int index;

    if (i <= x_stop && k < z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hx[IDX3D_FIELDS(i, j-1, k)] += coef_H_1 * E_z[index];
    }

    if (i < x_stop && k <= z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hz[IDX3D_FIELDS(i, j-1, k)] -= coef_H_2 * E_x[index];
    }
    """),
}


# STANDARD MAGNETIC - Y_HIGH FACE

inject_std_yhigh_H = {
    "name": "inject_std_yhigh_H",
    "args_cuda":   _std_H_args_cuda("inject_std_yhigh_H"),
    "args_opencl": _std_H_args_opencl("inject_std_yhigh_H"),
    "args_metal":  _std_H_args_metal("inject_std_yhigh_H"),
    "func": Template("""
    // Standard TF/SF magnetic correction - y_high face.
    // At j = y_stop: signs flipped vs y_low.
    //   Hx[i, j, k] -= coef_H_xy * E_z[index]
    //   Hz[i, j, k] += coef_H_zy * E_x[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int k = t % NZ_FACE + z_start;

    int j = y_stop;
    int index;

    if (i <= x_stop && k < z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hx[IDX3D_FIELDS(i, j, k)] -= coef_H_1 * E_z[index];
    }

    if (i < x_stop && k <= z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hz[IDX3D_FIELDS(i, j, k)] += coef_H_2 * E_x[index];
    }
    """),
}


# STANDARD MAGNETIC - Z_LOW FACE
# Cython (k = z_start):
#   Hy[i, j, k-1] += coef_H_yz * E_x[index]   coef_H_yz = updatecoeffsH[3]
#   Hx[i, j, k-1] -= coef_H_xz * E_y[index]   coef_H_xz = updatecoeffsH[3]
# Thread t: i = t/NY_FACE + x_start, j = t%NY_FACE + y_start


inject_std_zlow_H = {
    "name": "inject_std_zlow_H",
    "args_cuda":   _std_H_args_cuda("inject_std_zlow_H"),
    "args_opencl": _std_H_args_opencl("inject_std_zlow_H"),
    "args_metal":  _std_H_args_metal("inject_std_zlow_H"),
    "func": Template("""
    // Standard TF/SF magnetic correction - z_low face.
    // At k = z_start:
    //   Hy[i, j, k-1] += coef_H_yz * E_x[index]   i in [x_start,x_stop), j in [y_start,y_stop]
    //   Hx[i, j, k-1] -= coef_H_xz * E_y[index]   i in [x_start,x_stop], j in [y_start,y_stop)
    //
    // coef_H_1 = updatecoeffsH[3] (DBz)
    // coef_H_2 = updatecoeffsH[3] (DBz)
    // Thread t: i = t/NY_FACE + x_start, j = t%NY_FACE + y_start

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int j = t % NZ_FACE + y_start;

    int k = z_start;
    int index;

    if (i < x_stop && j <= y_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hy[IDX3D_FIELDS(i, j, k-1)] += coef_H_1 * E_x[index];
    }

    if (i <= x_stop && j < y_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hx[IDX3D_FIELDS(i, j, k-1)] -= coef_H_2 * E_y[index];
    }
    """),
}



# STANDARD MAGNETIC - Z_HIGH FACE


inject_std_zhigh_H = {
    "name": "inject_std_zhigh_H",
    "args_cuda":   _std_H_args_cuda("inject_std_zhigh_H"),
    "args_opencl": _std_H_args_opencl("inject_std_zhigh_H"),
    "args_metal":  _std_H_args_metal("inject_std_zhigh_H"),
    "func": Template("""
    // Standard TF/SF magnetic correction - z_high face.
    // At k = z_stop: signs flipped vs z_low.
    //   Hy[i, j, k] -= coef_H_yz * E_x[index]
    //   Hx[i, j, k] += coef_H_xz * E_y[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int j = t % NZ_FACE + y_start;

    int k = z_stop;
    int index;

    if (i < x_stop && j <= y_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hy[IDX3D_FIELDS(i, j, k)] -= coef_H_1 * E_x[index];
    }

    if (i <= x_stop && j < y_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Hx[IDX3D_FIELDS(i, j, k)] += coef_H_2 * E_y[index];
    }
    """),
}



# STANDARD ELECTRIC - X_LOW FACE

# Cython (applyTFSFElectric, i = x_start):
#   Ez[i, j, k] -= coef_E_zx * H_y[index]   index uses (i-1-Ox) not (i-Ox)!
#   Ey[i, j, k] += coef_E_yx * H_z[index]   same offset (i-1-Ox)
#
# CRITICAL: Electric x-face uses m_x*(i-1-Ox) for x_start - staggered by 1


inject_std_xlow_E = {
    "name": "inject_std_xlow_E",
    "args_cuda":   _std_E_args_cuda("inject_std_xlow_E"),
    "args_opencl": _std_E_args_opencl("inject_std_xlow_E"),
    "args_metal":  _std_E_args_metal("inject_std_xlow_E"),
    "func": Template("""
    // Standard TF/SF electric correction - x_low face.
    // Ports applyTFSFElectric() x_start section.
    //
    // IMPORTANT: index uses (i-1-Ox) not (i-Ox) for x_start face.
    // This is because E and H are staggered by half a cell in the Yee grid.
    //
    //   Ez[i, j, k] -= coef_E_zx * H_y[index]   j in [y_start,y_stop], k in [z_start,z_stop)
    //   Ey[i, j, k] += coef_E_yx * H_z[index]   j in [y_start,y_stop), k in [z_start,z_stop]
    //
    // coef_E_1 = updatecoeffsE[1] (CBx)
    // coef_E_2 = updatecoeffsE[1] (CBx)

    $TFSF_IDX

    int j = t / NZ_FACE + y_start;
    int k = t % NZ_FACE + z_start;

    int i = x_start;
    int index;

    if (j <= y_stop && k < z_stop) {
        index = m_x * (i - 1 - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Ez[IDX3D_FIELDS(i, j, k)] -= coef_E_1 * H_y[index];
    }

    if (j < y_stop && k <= z_stop) {
        index = m_x * (i - 1 - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Ey[IDX3D_FIELDS(i, j, k)] += coef_E_2 * H_z[index];
    }
    """),
}



# STANDARD ELECTRIC - X_HIGH FACE

# x_stop uses m_x*(i-Ox) - normal offset (no -1)
# Signs flipped vs x_low.


inject_std_xhigh_E = {
    "name": "inject_std_xhigh_E",
    "args_cuda":   _std_E_args_cuda("inject_std_xhigh_E"),
    "args_opencl": _std_E_args_opencl("inject_std_xhigh_E"),
    "args_metal":  _std_E_args_metal("inject_std_xhigh_E"),
    "func": Template("""
    // Standard TF/SF electric correction - x_high face.
    // x_stop uses m_x*(i-Ox) - no staggering offset needed here.
    //   Ez[i, j, k] += coef_E_zx * H_y[index]
    //   Ey[i, j, k] -= coef_E_yx * H_z[index]

    $TFSF_IDX

    int j = t / NZ_FACE + y_start;
    int k = t % NZ_FACE + z_start;

    int i = x_stop;
    int index;

    if (j <= y_stop && k < z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Ez[IDX3D_FIELDS(i, j, k)] += coef_E_1 * H_y[index];
    }

    if (j < y_stop && k <= z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Ey[IDX3D_FIELDS(i, j, k)] -= coef_E_2 * H_z[index];
    }
    """),
}



# STANDARD ELECTRIC - Y_LOW FACE

# Cython (j = y_start): index uses m_y*(j-1-Oy) - staggered by 1 in y
#   Ez[i, j, k] += coef_E_zy * H_x[index]   coef_E_zy = updatecoeffsE[2]
#   Ex[i, j, k] -= coef_E_xy * H_z[index]   coef_E_xy = updatecoeffsE[2]


inject_std_ylow_E = {
    "name": "inject_std_ylow_E",
    "args_cuda":   _std_E_args_cuda("inject_std_ylow_E"),
    "args_opencl": _std_E_args_opencl("inject_std_ylow_E"),
    "args_metal":  _std_E_args_metal("inject_std_ylow_E"),
    "func": Template("""
    // Standard TF/SF electric correction - y_low face.
    // index uses m_y*(j-1-Oy) - staggered by 1 in y for y_start face.
    //   Ez[i, j, k] += coef_E_zy * H_x[index]   i in [x_start,x_stop], k in [z_start,z_stop)
    //   Ex[i, j, k] -= coef_E_xy * H_z[index]   i in [x_start,x_stop), k in [z_start,z_stop]
    //
    // coef_E_1 = updatecoeffsE[2] (CBy)
    // coef_E_2 = updatecoeffsE[2] (CBy)

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int k = t % NZ_FACE + z_start;

    int j = y_start;
    int index;

    if (i <= x_stop && k < z_stop) {
        index = m_x * (i - Ox) + m_y * (j - 1 - Oy) + m_z * (k - Oz);
        Ez[IDX3D_FIELDS(i, j, k)] += coef_E_1 * H_x[index];
    }

    if (i < x_stop && k <= z_stop) {
        index = m_x * (i - Ox) + m_y * (j - 1 - Oy) + m_z * (k - Oz);
        Ex[IDX3D_FIELDS(i, j, k)] -= coef_E_2 * H_z[index];
    }
    """),
}



# STANDARD ELECTRIC - Y_HIGH FACE

# y_stop uses m_y*(j-Oy) - normal offset. Signs flipped vs y_low.

inject_std_yhigh_E = {
    "name": "inject_std_yhigh_E",
    "args_cuda":   _std_E_args_cuda("inject_std_yhigh_E"),
    "args_opencl": _std_E_args_opencl("inject_std_yhigh_E"),
    "args_metal":  _std_E_args_metal("inject_std_yhigh_E"),
    "func": Template("""
    // Standard TF/SF electric correction - y_high face.
    // y_stop uses m_y*(j-Oy) - no staggering offset.
    //   Ez[i, j, k] -= coef_E_zy * H_x[index]
    //   Ex[i, j, k] += coef_E_xy * H_z[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int k = t % NZ_FACE + z_start;

    int j = y_stop;
    int index;

    if (i <= x_stop && k < z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Ez[IDX3D_FIELDS(i, j, k)] -= coef_E_1 * H_x[index];
    }

    if (i < x_stop && k <= z_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Ex[IDX3D_FIELDS(i, j, k)] += coef_E_2 * H_z[index];
    }
    """),
}



# STANDARD ELECTRIC - Z_LOW FACE

# Cython (k = z_start): index uses m_z*(k-1-Oz) - staggered by 1 in z
#   Ey[i, j, k] -= coef_E_yz * H_x[index]   coef_E_yz = updatecoeffsE[3]
#   Ex[i, j, k] += coef_E_xz * H_y[index]   coef_E_xz = updatecoeffsE[3]


inject_std_zlow_E = {
    "name": "inject_std_zlow_E",
    "args_cuda":   _std_E_args_cuda("inject_std_zlow_E"),
    "args_opencl": _std_E_args_opencl("inject_std_zlow_E"),
    "args_metal":  _std_E_args_metal("inject_std_zlow_E"),
    "func": Template("""
    // Standard TF/SF electric correction - z_low face.
    // index uses m_z*(k-1-Oz) - staggered by 1 in z for z_start face.
    //   Ey[i, j, k] -= coef_E_yz * H_x[index]   i in [x_start,x_stop], j in [y_start,y_stop)
    //   Ex[i, j, k] += coef_E_xz * H_y[index]   i in [x_start,x_stop), j in [y_start,y_stop]
    //
    // coef_E_1 = updatecoeffsE[3] (CBz)
    // coef_E_2 = updatecoeffsE[3] (CBz)

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int j = t % NZ_FACE + y_start;

    int k = z_start;
    int index;

    if (i <= x_stop && j < y_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - 1 - Oz);
        Ey[IDX3D_FIELDS(i, j, k)] -= coef_E_1 * H_x[index];
    }

    if (i < x_stop && j <= y_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - 1 - Oz);
        Ex[IDX3D_FIELDS(i, j, k)] += coef_E_2 * H_y[index];
    }
    """),
}

# STANDARD ELECTRIC - Z_HIGH FACE

inject_std_zhigh_E = {
    "name": "inject_std_zhigh_E",
    "args_cuda":   _std_E_args_cuda("inject_std_zhigh_E"),
    "args_opencl": _std_E_args_opencl("inject_std_zhigh_E"),
    "args_metal":  _std_E_args_metal("inject_std_zhigh_E"),
    "func": Template("""
    // Standard TF/SF electric correction - z_high face.
    // z_stop uses m_z*(k-Oz) - no staggering. Signs flipped vs z_low.
    //   Ey[i, j, k] += coef_E_yz * H_x[index]
    //   Ex[i, j, k] -= coef_E_xz * H_y[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int j = t % NZ_FACE + y_start;

    int k = z_stop;
    int index;

    if (i <= x_stop && j < y_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Ey[IDX3D_FIELDS(i, j, k)] += coef_E_1 * H_x[index];
    }

    if (i < x_stop && j <= y_stop) {
        index = m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        Ex[IDX3D_FIELDS(i, j, k)] -= coef_E_2 * H_y[index];
    }
    """),
}



# AXIAL MAGNETIC - X_LOW FACE
# Cython (applyTFSFMagnetic_axial, i = x_start):
#   index = O_axial + m_x*(i-Ox) + m_y*(j-Oy) + m_z*(k-Oz)
#   Hy[i-1,j,k] -= updatecoeffsH[GID[4,i-1,j,k], 1] * E_z[index]
#   Hz[i-1,j,k] += updatecoeffsH[GID[5,i-1,j,k], 1] * E_y[index]
#
# GID component 4 = Hy material, component 5 = Hz material. Column = 1 (DBx).
# GID accessed via IDX4D_ID(component, x, y, z) macro.
# updatecoeffsH accessed via IDX2D_MAT(mat_id, col) macro.


inject_axial_xlow_H = {
    "name": "inject_axial_xlow_H",
    "args_cuda":   _axial_H_args_cuda("inject_axial_xlow_H"),
    "args_opencl": _axial_H_args_opencl("inject_axial_xlow_H"),
    "args_metal":  _axial_H_args_metal("inject_axial_xlow_H"),
    "func": Template("""
    // Axial TF/SF magnetic correction - x_low face.
    // Per-cell coefficient lookup via IDX4D_ID and IDX2D_MAT.
    //
    //   index = O_axial + m_x*(i-Ox) + m_y*(j-Oy) + m_z*(k-Oz)
    //   Hy[i-1,j,k] -= updatecoeffsH[GID[4,i-1,j,k], 1] * E_z[index]
    //   Hz[i-1,j,k] += updatecoeffsH[GID[5,i-1,j,k], 1] * E_y[index]

    $TFSF_IDX

    int j = t / NZ_FACE + y_start;
    int k = t % NZ_FACE + z_start;

    int i = x_start;
    int index;
    int mat;

    if (j <= y_stop && k < z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(4, i-1, j, k)];
        Hy[IDX3D_FIELDS(i-1, j, k)] -= updatecoeffsH[IDX2D_MAT(mat, 1)] * E_z[index];
    }

    if (j < y_stop && k <= z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(5, i-1, j, k)];
        Hz[IDX3D_FIELDS(i-1, j, k)] += updatecoeffsH[IDX2D_MAT(mat, 1)] * E_y[index];
    }
    """),
}


inject_axial_xhigh_H = {
    "name": "inject_axial_xhigh_H",
    "args_cuda":   _axial_H_args_cuda("inject_axial_xhigh_H"),
    "args_opencl": _axial_H_args_opencl("inject_axial_xhigh_H"),
    "args_metal":  _axial_H_args_metal("inject_axial_xhigh_H"),
    "func": Template("""
    // Axial TF/SF magnetic correction - x_high face.
    //   Hy[i,j,k] += updatecoeffsH[GID[4,i,j,k], 1] * E_z[index]
    //   Hz[i,j,k] -= updatecoeffsH[GID[5,i,j,k], 1] * E_y[index]

    $TFSF_IDX

    int j = t / NZ_FACE + y_start;
    int k = t % NZ_FACE + z_start;

    int i = x_stop;
    int index;
    int mat;

    if (j <= y_stop && k < z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(4, i, j, k)];
        Hy[IDX3D_FIELDS(i, j, k)] += updatecoeffsH[IDX2D_MAT(mat, 1)] * E_z[index];
    }

    if (j < y_stop && k <= z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(5, i, j, k)];
        Hz[IDX3D_FIELDS(i, j, k)] -= updatecoeffsH[IDX2D_MAT(mat, 1)] * E_y[index];
    }
    """),
}


inject_axial_ylow_H = {
    "name": "inject_axial_ylow_H",
    "args_cuda":   _axial_H_args_cuda("inject_axial_ylow_H"),
    "args_opencl": _axial_H_args_opencl("inject_axial_ylow_H"),
    "args_metal":  _axial_H_args_metal("inject_axial_ylow_H"),
    "func": Template("""
    // Axial TF/SF magnetic correction - y_low face.
    // GID component 3 = Hx, component 5 = Hz. Column = 2 (DBy).
    //   Hx[i,j-1,k] += updatecoeffsH[GID[3,i,j-1,k], 2] * E_z[index]
    //   Hz[i,j-1,k] -= updatecoeffsH[GID[5,i,j-1,k], 2] * E_x[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int k = t % NZ_FACE + z_start;

    int j = y_start;
    int index;
    int mat;

    if (i <= x_stop && k < z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(3, i, j-1, k)];
        Hx[IDX3D_FIELDS(i, j-1, k)] += updatecoeffsH[IDX2D_MAT(mat, 2)] * E_z[index];
    }

    if (i < x_stop && k <= z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(5, i, j-1, k)];
        Hz[IDX3D_FIELDS(i, j-1, k)] -= updatecoeffsH[IDX2D_MAT(mat, 2)] * E_x[index];
    }
    """),
}


inject_axial_yhigh_H = {
    "name": "inject_axial_yhigh_H",
    "args_cuda":   _axial_H_args_cuda("inject_axial_yhigh_H"),
    "args_opencl": _axial_H_args_opencl("inject_axial_yhigh_H"),
    "args_metal":  _axial_H_args_metal("inject_axial_yhigh_H"),
    "func": Template("""
    // Axial TF/SF magnetic correction - y_high face.
    //   Hx[i,j,k] -= updatecoeffsH[GID[3,i,j,k], 2] * E_z[index]
    //   Hz[i,j,k] += updatecoeffsH[GID[5,i,j,k], 2] * E_x[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int k = t % NZ_FACE + z_start;

    int j = y_stop;
    int index;
    int mat;

    if (i <= x_stop && k < z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(3, i, j, k)];
        Hx[IDX3D_FIELDS(i, j, k)] -= updatecoeffsH[IDX2D_MAT(mat, 2)] * E_z[index];
    }

    if (i < x_stop && k <= z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(5, i, j, k)];
        Hz[IDX3D_FIELDS(i, j, k)] += updatecoeffsH[IDX2D_MAT(mat, 2)] * E_x[index];
    }
    """),
}


inject_axial_zlow_H = {
    "name": "inject_axial_zlow_H",
    "args_cuda":   _axial_H_args_cuda("inject_axial_zlow_H"),
    "args_opencl": _axial_H_args_opencl("inject_axial_zlow_H"),
    "args_metal":  _axial_H_args_metal("inject_axial_zlow_H"),
    "func": Template("""
    // Axial TF/SF magnetic correction - z_low face.
    // GID component 4 = Hy, component 3 = Hx. Column = 3 (DBz).
    //   Hy[i,j,k-1] += updatecoeffsH[GID[4,i,j,k-1], 3] * E_x[index]
    //   Hx[i,j,k-1] -= updatecoeffsH[GID[3,i,j,k-1], 3] * E_y[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int j = t % NZ_FACE + y_start;

    int k = z_start;
    int index;
    int mat;

    if (i < x_stop && j <= y_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(4, i, j, k-1)];
        Hy[IDX3D_FIELDS(i, j, k-1)] += updatecoeffsH[IDX2D_MAT(mat, 3)] * E_x[index];
    }

    if (i <= x_stop && j < y_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(3, i, j, k-1)];
        Hx[IDX3D_FIELDS(i, j, k-1)] -= updatecoeffsH[IDX2D_MAT(mat, 3)] * E_y[index];
    }
    """),
}


inject_axial_zhigh_H = {
    "name": "inject_axial_zhigh_H",
    "args_cuda":   _axial_H_args_cuda("inject_axial_zhigh_H"),
    "args_opencl": _axial_H_args_opencl("inject_axial_zhigh_H"),
    "args_metal":  _axial_H_args_metal("inject_axial_zhigh_H"),
    "func": Template("""
    // Axial TF/SF magnetic correction - z_high face.
    //   Hy[i,j,k] -= updatecoeffsH[GID[4,i,j,k], 3] * E_x[index]
    //   Hx[i,j,k] += updatecoeffsH[GID[3,i,j,k], 3] * E_y[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int j = t % NZ_FACE + y_start;

    int k = z_stop;
    int index;
    int mat;

    if (i < x_stop && j <= y_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(4, i, j, k)];
        Hy[IDX3D_FIELDS(i, j, k)] -= updatecoeffsH[IDX2D_MAT(mat, 3)] * E_x[index];
    }

    if (i <= x_stop && j < y_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(3, i, j, k)];
        Hx[IDX3D_FIELDS(i, j, k)] += updatecoeffsH[IDX2D_MAT(mat, 3)] * E_y[index];
    }
    """),
}


# AXIAL ELECTRIC - X_LOW FACE
# Cython (applyTFSFElectric_axial, i = x_start):
#   index = O_axial + m_x*(i-1-Ox) + m_y*(j-Oy) + m_z*(k-Oz)  <-- staggered!
#   Ez[i,j,k] -= updatecoeffsE[GID[2,i,j,k], 1] * H_y[index]  GID comp=2 (Ez mat)
#   Ey[i,j,k] += updatecoeffsE[GID[1,i,j,k], 1] * H_z[index]  GID comp=1 (Ey mat)


inject_axial_xlow_E = {
    "name": "inject_axial_xlow_E",
    "args_cuda":   _axial_E_args_cuda("inject_axial_xlow_E"),
    "args_opencl": _axial_E_args_opencl("inject_axial_xlow_E"),
    "args_metal":  _axial_E_args_metal("inject_axial_xlow_E"),
    "func": Template("""
    // Axial TF/SF electric correction - x_low face.
    // IMPORTANT: index uses (i-1-Ox) staggering for x_start electric face.
    //
    //   Ez[i,j,k] -= updatecoeffsE[GID[2,i,j,k], 1] * H_y[index]
    //   Ey[i,j,k] += updatecoeffsE[GID[1,i,j,k], 1] * H_z[index]

    $TFSF_IDX

    int j = t / NZ_FACE + y_start;
    int k = t % NZ_FACE + z_start;

    int i = x_start;
    int index;
    int mat;

    if (j <= y_stop && k < z_stop) {
        index = O_axial + m_x * (i - 1 - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(2, i, j, k)];
        Ez[IDX3D_FIELDS(i, j, k)] -= updatecoeffsE[IDX2D_MAT(mat, 1)] * H_y[index];
    }

    if (j < y_stop && k <= z_stop) {
        index = O_axial + m_x * (i - 1 - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(1, i, j, k)];
        Ey[IDX3D_FIELDS(i, j, k)] += updatecoeffsE[IDX2D_MAT(mat, 1)] * H_z[index];
    }
    """),
}


inject_axial_xhigh_E = {
    "name": "inject_axial_xhigh_E",
    "args_cuda":   _axial_E_args_cuda("inject_axial_xhigh_E"),
    "args_opencl": _axial_E_args_opencl("inject_axial_xhigh_E"),
    "args_metal":  _axial_E_args_metal("inject_axial_xhigh_E"),
    "func": Template("""
    // Axial TF/SF electric correction - x_high face.
    // x_stop uses (i-Ox) - no staggering. Signs flipped.
    //   Ez[i,j,k] += updatecoeffsE[GID[2,i,j,k], 1] * H_y[index]
    //   Ey[i,j,k] -= updatecoeffsE[GID[1,i,j,k], 1] * H_z[index]

    $TFSF_IDX

    int j = t / NZ_FACE + y_start;
    int k = t % NZ_FACE + z_start;

    int i = x_stop;
    int index;
    int mat;

    if (j <= y_stop && k < z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(2, i, j, k)];
        Ez[IDX3D_FIELDS(i, j, k)] += updatecoeffsE[IDX2D_MAT(mat, 1)] * H_y[index];
    }

    if (j < y_stop && k <= z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(1, i, j, k)];
        Ey[IDX3D_FIELDS(i, j, k)] -= updatecoeffsE[IDX2D_MAT(mat, 1)] * H_z[index];
    }
    """),
}


inject_axial_ylow_E = {
    "name": "inject_axial_ylow_E",
    "args_cuda":   _axial_E_args_cuda("inject_axial_ylow_E"),
    "args_opencl": _axial_E_args_opencl("inject_axial_ylow_E"),
    "args_metal":  _axial_E_args_metal("inject_axial_ylow_E"),
    "func": Template("""
    // Axial TF/SF electric correction - y_low face.
    // index uses (j-1-Oy) staggering for y_start face.
    // GID comp=2 (Ez mat), comp=1 (Ey mat) for Ex correction. Column=2 (CBy).
    //   Ez[i,j,k] += updatecoeffsE[GID[2,i,j,k], 2] * H_x[index]
    //   Ex[i,j,k] -= updatecoeffsE[GID[1,i,j,k], 2] * H_z[index]  (note: GID[1] used in Cython)

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int k = t % NZ_FACE + z_start;

    int j = y_start;
    int index;
    int mat;

    if (i <= x_stop && k < z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - 1 - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(2, i, j, k)];
        Ez[IDX3D_FIELDS(i, j, k)] += updatecoeffsE[IDX2D_MAT(mat, 2)] * H_x[index];
    }

    if (i < x_stop && k <= z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - 1 - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(1, i, j, k)];
        Ex[IDX3D_FIELDS(i, j, k)] -= updatecoeffsE[IDX2D_MAT(mat, 2)] * H_z[index];
    }
    """),
}


inject_axial_yhigh_E = {
    "name": "inject_axial_yhigh_E",
    "args_cuda":   _axial_E_args_cuda("inject_axial_yhigh_E"),
    "args_opencl": _axial_E_args_opencl("inject_axial_yhigh_E"),
    "args_metal":  _axial_E_args_metal("inject_axial_yhigh_E"),
    "func": Template("""
    // Axial TF/SF electric correction - y_high face.
    // y_stop uses (j-Oy) - no staggering. Signs flipped vs y_low.
    // NOTE: Cython uses (j-1-Oy) for Ex correction at y_stop too - preserved here.
    //   Ez[i,j,k] -= updatecoeffsE[GID[2,i,j,k], 2] * H_x[index]
    //   Ex[i,j,k] += updatecoeffsE[GID[1,i,j,k], 2] * H_z[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int k = t % NZ_FACE + z_start;

    int j = y_stop;
    int index;
    int mat;

    if (i <= x_stop && k < z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(2, i, j, k)];
        Ez[IDX3D_FIELDS(i, j, k)] -= updatecoeffsE[IDX2D_MAT(mat, 2)] * H_x[index];
    }

    if (i < x_stop && k <= z_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - 1 - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(1, i, j, k)];
        Ex[IDX3D_FIELDS(i, j, k)] += updatecoeffsE[IDX2D_MAT(mat, 2)] * H_z[index];
    }
    """),
}


inject_axial_zlow_E = {
    "name": "inject_axial_zlow_E",
    "args_cuda":   _axial_E_args_cuda("inject_axial_zlow_E"),
    "args_opencl": _axial_E_args_opencl("inject_axial_zlow_E"),
    "args_metal":  _axial_E_args_metal("inject_axial_zlow_E"),
    "func": Template("""
    // Axial TF/SF electric correction - z_low face.
    // index uses (k-1-Oz) staggering for z_start face.
    // GID comp=1 (Ey mat), comp=0 (Ex mat). Column=3 (CBz).
    //   Ey[i,j,k] -= updatecoeffsE[GID[1,i,j,k], 3] * H_x[index]
    //   Ex[i,j,k] += updatecoeffsE[GID[0,i,j,k], 3] * H_y[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int j = t % NZ_FACE + y_start;

    int k = z_start;
    int index;
    int mat;

    if (i <= x_stop && j < y_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - 1 - Oz);
        mat = ID[IDX4D_ID(1, i, j, k)];
        Ey[IDX3D_FIELDS(i, j, k)] -= updatecoeffsE[IDX2D_MAT(mat, 3)] * H_x[index];
    }

    if (i < x_stop && j <= y_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - 1 - Oz);
        mat = ID[IDX4D_ID(0, i, j, k)];
        Ex[IDX3D_FIELDS(i, j, k)] += updatecoeffsE[IDX2D_MAT(mat, 3)] * H_y[index];
    }
    """),
}


inject_axial_zhigh_E = {
    "name": "inject_axial_zhigh_E",
    "args_cuda":   _axial_E_args_cuda("inject_axial_zhigh_E"),
    "args_opencl": _axial_E_args_opencl("inject_axial_zhigh_E"),
    "args_metal":  _axial_E_args_metal("inject_axial_zhigh_E"),
    "func": Template("""
    // Axial TF/SF electric correction - z_high face.
    // z_stop uses (k-Oz) - no staggering. Signs flipped vs z_low.
    //   Ey[i,j,k] += updatecoeffsE[GID[1,i,j,k], 3] * H_x[index]
    //   Ex[i,j,k] -= updatecoeffsE[GID[0,i,j,k], 3] * H_y[index]

    $TFSF_IDX

    int i = t / NZ_FACE + x_start;
    int j = t % NZ_FACE + y_start;

    int k = z_stop;
    int index;
    int mat;

    if (i <= x_stop && j < y_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(1, i, j, k)];
        Ey[IDX3D_FIELDS(i, j, k)] += updatecoeffsE[IDX2D_MAT(mat, 3)] * H_x[index];
    }

    if (i < x_stop && j <= y_stop) {
        index = O_axial + m_x * (i - Ox) + m_y * (j - Oy) + m_z * (k - Oz);
        mat = ID[IDX4D_ID(0, i, j, k)];
        Ex[IDX3D_FIELDS(i, j, k)] -= updatecoeffsE[IDX2D_MAT(mat, 3)] * H_y[index];
    }
    """),
}
STANDARD_H_KERNELS = [
    inject_std_xlow_H, inject_std_xhigh_H,
    inject_std_ylow_H, inject_std_yhigh_H,
    inject_std_zlow_H, inject_std_zhigh_H,
]

STANDARD_E_KERNELS = [
    inject_std_xlow_E, inject_std_xhigh_E,
    inject_std_ylow_E, inject_std_yhigh_E,
    inject_std_zlow_E, inject_std_zhigh_E,
]

AXIAL_H_KERNELS = [
    inject_axial_xlow_H, inject_axial_xhigh_H,
    inject_axial_ylow_H, inject_axial_yhigh_H,
    inject_axial_zlow_H, inject_axial_zhigh_H,
]

AXIAL_E_KERNELS = [
    inject_axial_xlow_E, inject_axial_xhigh_E,
    inject_axial_ylow_E, inject_axial_yhigh_E,
    inject_axial_zlow_E, inject_axial_zhigh_E,
]
