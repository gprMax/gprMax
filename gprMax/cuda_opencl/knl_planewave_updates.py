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

from string import Template

# DEVICE-RESIDENT SOURCE INITIALISATION

# The delayed waveform values are prepared once during model setup. This
# kernel copies one whole- or half-timestep slice into the first M samples of
# the appropriate auxiliary grid without a host-to-device transfer inside the
# FDTD time loop.
initialise_1d_source = {
    "args_cuda": Template(
        """
        __global__ void initialise_1d_source(
            const int N,
            const int M,
            const int timestep,
            $REAL* __restrict__ fields,
            const $REAL* __restrict__ source_values
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int timestep,
        __global $REAL* restrict fields,
        __global const $REAL* restrict source_values
        """
    ),
    "args_metal": Template(
        """
        kernel void initialise_1d_source(
            device const int& N,
            device const int& M,
            device const int& timestep,
            device $REAL* fields,
            device const $REAL* source_values,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    $CUDA_IDX

    int count = 3 * M;
    if (i < count) {
        int component = i / M;
        int r = i - component * M;
        fields[component * N + r] = source_values[timestep * count + i];
    }
"""
    ),
}


# STANDARD (HOMOGENEOUS) MAGNETIC UPDATE
# Ports: updateMagneticFields() from plane_wave.pyx
# Updates the 1D DPW auxiliary grid H fields for the standard (homogeneous)
# case. The medium is assumed uniform - scalar coefficients from the background
# material are passed directly as kernel arguments.
# Bulk update: parallel over j in [m[3], n-m[3]]
# PML update:  separate kernel (update_1d_magnetic_pml) over p cells


update_1d_magnetic = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_magnetic(
            const int N,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL coef_H_xt,
            const $REAL coef_H_xy,
            const $REAL coef_H_xz,
            const $REAL coef_H_yt,
            const $REAL coef_H_yx,
            const $REAL coef_H_yz,
            const $REAL coef_H_zt,
            const $REAL coef_H_zx,
            const $REAL coef_H_zy,
            $REAL* __restrict__ H_x,
            $REAL* __restrict__ H_y,
            $REAL* __restrict__ H_z,
            const $REAL* __restrict__ E_x,
            const $REAL* __restrict__ E_y,
            const $REAL* __restrict__ E_z
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL coef_H_xt,
        const $REAL coef_H_xy,
        const $REAL coef_H_xz,
        const $REAL coef_H_yt,
        const $REAL coef_H_yx,
        const $REAL coef_H_yz,
        const $REAL coef_H_zt,
        const $REAL coef_H_zx,
        const $REAL coef_H_zy,
        __global $REAL* restrict H_x,
        __global $REAL* restrict H_y,
        __global $REAL* restrict H_z,
        __global const $REAL* restrict E_x,
        __global const $REAL* restrict E_y,
        __global const $REAL* restrict E_z
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_magnetic(
            device const int& N,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& coef_H_xt,
            device const $REAL& coef_H_xy,
            device const $REAL& coef_H_xz,
            device const $REAL& coef_H_yt,
            device const $REAL& coef_H_yx,
            device const $REAL& coef_H_yz,
            device const $REAL& coef_H_zt,
            device const $REAL& coef_H_zx,
            device const $REAL& coef_H_zy,
            device $REAL* H_x,
            device $REAL* H_y,
            device $REAL* H_z,
            device const $REAL* E_x,
            device const $REAL* E_y,
            device const $REAL* E_z,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // 1D DPW auxiliary grid - standard (homogeneous) magnetic field update.
    //
    // Updates H_x, H_y, H_z at each position j in the bulk 1D grid.
    // Scalar coefficients are passed as kernel arguments (same for all cells).
    //
    // Args:
    //   N:    total length of 1D DPW array.
    //   M:    max(m_x, m_y, m_z) - used for loop bounds [M, N-M].
    //   m_x, m_y, m_z: integer direction cosines for DPW angle mapping.
    //   coef_H_*: scalar update coefficients from background material.
    //             coef_H_xt = updatecoeffsH[0] (DA - self decay)
    //             coef_H_xy = updatecoeffsH[2] (DBy)
    //             coef_H_xz = updatecoeffsH[3] (DBz)
    //             coef_H_yx = updatecoeffsH[1] (DBx)
    //             coef_H_yz = updatecoeffsH[3] (DBz)
    //             coef_H_yt = updatecoeffsH[0] (DA)
    //             coef_H_zt = updatecoeffsH[0] (DA)
    //             coef_H_zx = updatecoeffsH[1] (DBx)
    //             coef_H_zy = updatecoeffsH[2] (DBy)
    //   H_x, H_y, H_z: 1D magnetic field arrays - shape [N].
    //   E_x, E_y, E_z: 1D electric field arrays - shape [N].


    $CUDA_IDX

    // Offset thread index into the valid bulk update range [M, N-M)
    int j = i + M;

    // Bounds check - exit threads beyond the bulk region
    if (j >= N - M) return;

    // H_x update: curl of E in yz plane
    // H_x[j] = DA*H_x[j] + DBz*(E_y[j+m_z] - E_y[j]) - DBy*(E_z[j+m_y] - E_z[j])
    H_x[j] = coef_H_xt * H_x[j]
            + coef_H_xz * (E_y[j + m_z] - E_y[j])
            - coef_H_xy * (E_z[j + m_y] - E_z[j]);

    // H_y update: curl of E in xz plane
    // H_y[j] = DA*H_y[j] + DBx*(E_z[j+m_x] - E_z[j]) - DBz*(E_x[j+m_z] - E_x[j])
    H_y[j] = coef_H_yt * H_y[j]
            + coef_H_yx * (E_z[j + m_x] - E_z[j])
            - coef_H_yz * (E_x[j + m_z] - E_x[j]);

    // H_z update: curl of E in xy plane
    // H_z[j] = DA*H_z[j] + DBy*(E_x[j+m_y] - E_x[j]) - DBx*(E_y[j+m_x] - E_y[j])
    H_z[j] = coef_H_zt * H_z[j]
            + coef_H_zy * (E_x[j + m_y] - E_x[j])
            - coef_H_zx * (E_y[j + m_x] - E_y[j]);
    """
    ),
}


# STANDARD (HOMOGENEOUS) MAGNETIC PML UPDATE
# Ports: PML section of updateMagneticFields() from plane_wave.pyx
# Updates PML integral arrays Ix, Iy, Iz and applies PML correction to
# H_x, H_y, H_z at the p cells at the end of the 1D DPW grid.
# Separate from the bulk update because:
# - Only p cells (typically 10-20) vs n cells in bulk
# - Uses different coefficient arrays (rcHx, rcHy, rcHz)
# - Runs sequentially in the Cython - kept as small separate kernel
# Thread i handles PML cell i (i in [0, p))

update_1d_magnetic_pml = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_magnetic_pml(
            const int N,
            const int P,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL coef_H_D,
            const $REAL dx,
            const $REAL dy,
            const $REAL dz,
            $REAL* __restrict__ H_x,
            $REAL* __restrict__ H_y,
            $REAL* __restrict__ H_z,
            const $REAL* __restrict__ E_x,
            const $REAL* __restrict__ E_y,
            const $REAL* __restrict__ E_z,
            $REAL* __restrict__ Ixmzy,
            $REAL* __restrict__ Ixmyz,
            $REAL* __restrict__ Iymzx,
            $REAL* __restrict__ Iymxz,
            $REAL* __restrict__ Izmyx,
            $REAL* __restrict__ Izmxy,
            const $REAL* __restrict__ RAHx,
            const $REAL* __restrict__ RBHx,
            const $REAL* __restrict__ RCHx,
            const $REAL* __restrict__ RDHx,
            const $REAL* __restrict__ RAHy,
            const $REAL* __restrict__ RBHy,
            const $REAL* __restrict__ RCHy,
            const $REAL* __restrict__ RDHy,
            const $REAL* __restrict__ RAHz,
            const $REAL* __restrict__ RBHz,
            const $REAL* __restrict__ RCHz,
            const $REAL* __restrict__ RDHz
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int P,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL coef_H_D,
        const $REAL dx,
        const $REAL dy,
        const $REAL dz,
        __global $REAL* restrict H_x,
        __global $REAL* restrict H_y,
        __global $REAL* restrict H_z,
        __global const $REAL* restrict E_x,
        __global const $REAL* restrict E_y,
        __global const $REAL* restrict E_z,
        __global $REAL* restrict Ixmzy,
        __global $REAL* restrict Ixmyz,
        __global $REAL* restrict Iymzx,
        __global $REAL* restrict Iymxz,
        __global $REAL* restrict Izmyx,
        __global $REAL* restrict Izmxy,
        __global const $REAL* restrict RAHx,
        __global const $REAL* restrict RBHx,
        __global const $REAL* restrict RCHx,
        __global const $REAL* restrict RDHx,
        __global const $REAL* restrict RAHy,
        __global const $REAL* restrict RBHy,
        __global const $REAL* restrict RCHy,
        __global const $REAL* restrict RDHy,
        __global const $REAL* restrict RAHz,
        __global const $REAL* restrict RBHz,
        __global const $REAL* restrict RCHz,
        __global const $REAL* restrict RDHz
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_magnetic_pml(
            device const int& N,
            device const int& P,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& coef_H_D,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device $REAL* H_x,
            device $REAL* H_y,
            device $REAL* H_z,
            device const $REAL* E_x,
            device const $REAL* E_y,
            device const $REAL* E_z,
            device $REAL* Ixmzy,
            device $REAL* Ixmyz,
            device $REAL* Iymzx,
            device $REAL* Iymxz,
            device $REAL* Izmyx,
            device $REAL* Izmxy,
            device const $REAL* RAHx,
            device const $REAL* RBHx,
            device const $REAL* RCHx,
            device const $REAL* RDHx,
            device const $REAL* RAHy,
            device const $REAL* RBHy,
            device const $REAL* RCHy,
            device const $REAL* RDHy,
            device const $REAL* RAHz,
            device const $REAL* RBHz,
            device const $REAL* RCHz,
            device const $REAL* RDHz,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // 1D DPW PML magnetic field update - standard (homogeneous) case.
    //
    // Updates H fields and PML integral arrays at the p PML cells
    // at the end of the 1D DPW grid using the first-order RIPML scheme.
    //
    // Args:
    //   N:   total length of 1D DPW array.
    //   P:   PML thickness in number of cells.
    //   M:   max(m_x, m_y, m_z).
    //   coef_H_D: updatecoeffsH[4] (srcm - source coefficient).
    //   dx, dy, dz: spatial step sizes.
    //   H_x/y/z:   1D magnetic field arrays.
    //   E_x/y/z:   1D electric field arrays.
    //   Ixmzy..Izmxy: PML integral arrays - shape [P] each (rows 2,3 of Ix,Iy,Iz).
    //   RAHx..RDHz: PML coefficients - shape [P] each (rows 0-3 of rcHx,rcHy,rcHz).

    $CUDA_IDX

    // Thread i handles PML cell i - bounds check
    if (i >= P) return;

    // Map PML index i to position in 1D grid (same as Cython: reversed loop)
    // Cython: for i in range(p-1, -1, -1): idx = (n-m[3]) - (p - i)
    int pml_i = P - 1 - i;
    int idx = (N - M) - (P - pml_i);

    $REAL dEzy, dEyz, dEzx, dExz, dEyx, dExy;
    $REAL mxy, mxz, myx, myz, mzx, mzy;

    //  H_x PML correction
    dEzy = (E_z[idx + m_y] - E_z[idx]) / dy;
    dEyz = (E_y[idx + m_z] - E_y[idx]) / dz;

    mxy = RAHx[pml_i] * dEzy + RBHx[pml_i] * Ixmzy[pml_i];
    mxz = RAHx[pml_i] * dEyz + RBHx[pml_i] * Ixmyz[pml_i];

    H_x[idx] = H_x[idx] - coef_H_D * mxy + coef_H_D * mxz;

    Ixmzy[pml_i] = Ixmzy[pml_i] - RCHx[pml_i] * mxy + RDHx[pml_i] * dEzy;
    Ixmyz[pml_i] = Ixmyz[pml_i] - RCHx[pml_i] * mxz + RDHx[pml_i] * dEyz;

    //  H_y PML correction
    dEzx = (E_z[idx + m_x] - E_z[idx]) / dx;
    dExz = (E_x[idx + m_z] - E_x[idx]) / dz;

    myx = RAHy[pml_i] * dEzx + RBHy[pml_i] * Iymzx[pml_i];
    myz = RAHy[pml_i] * dExz + RBHy[pml_i] * Iymxz[pml_i];

    H_y[idx] = H_y[idx] + coef_H_D * myx - coef_H_D * myz;

    Iymzx[pml_i] = Iymzx[pml_i] - RCHy[pml_i] * myx + RDHy[pml_i] * dEzx;
    Iymxz[pml_i] = Iymxz[pml_i] - RCHy[pml_i] * myz + RDHy[pml_i] * dExz;

    //  H_z PML correction
    dEyx = (E_y[idx + m_x] - E_y[idx]) / dx;
    dExy = (E_x[idx + m_y] - E_x[idx]) / dy;

    mzx = RAHz[pml_i] * dEyx + RBHz[pml_i] * Izmyx[pml_i];
    mzy = RAHz[pml_i] * dExy + RBHz[pml_i] * Izmxy[pml_i];

    H_z[idx] = H_z[idx] - coef_H_D * mzx + coef_H_D * mzy;

    Izmyx[pml_i] = Izmyx[pml_i] - RCHz[pml_i] * mzx + RDHz[pml_i] * dEyx;
    Izmxy[pml_i] = Izmxy[pml_i] - RCHz[pml_i] * mzy + RDHz[pml_i] * dExy;
    """
    ),
}


# STANDARD (HOMOGENEOUS) ELECTRIC UPDATE
# Ports: updateElectricFields() from plane_wave.pyx
# Updates the 1D DPW auxiliary grid E fields for the standard case.
# Scalar coefficients from background material passed as kernel arguments.

update_1d_electric = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric(
            const int N,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL coef_E_xt,
            const $REAL coef_E_xy,
            const $REAL coef_E_xz,
            const $REAL coef_E_yt,
            const $REAL coef_E_yx,
            const $REAL coef_E_yz,
            const $REAL coef_E_zt,
            const $REAL coef_E_zx,
            const $REAL coef_E_zy,
            $REAL* __restrict__ E_x,
            $REAL* __restrict__ E_y,
            $REAL* __restrict__ E_z,
            const $REAL* __restrict__ H_x,
            const $REAL* __restrict__ H_y,
            const $REAL* __restrict__ H_z
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL coef_E_xt,
        const $REAL coef_E_xy,
        const $REAL coef_E_xz,
        const $REAL coef_E_yt,
        const $REAL coef_E_yx,
        const $REAL coef_E_yz,
        const $REAL coef_E_zt,
        const $REAL coef_E_zx,
        const $REAL coef_E_zy,
        __global $REAL* restrict E_x,
        __global $REAL* restrict E_y,
        __global $REAL* restrict E_z,
        __global const $REAL* restrict H_x,
        __global const $REAL* restrict H_y,
        __global const $REAL* restrict H_z
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric(
            device const int& N,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& coef_E_xt,
            device const $REAL& coef_E_xy,
            device const $REAL& coef_E_xz,
            device const $REAL& coef_E_yt,
            device const $REAL& coef_E_yx,
            device const $REAL& coef_E_yz,
            device const $REAL& coef_E_zt,
            device const $REAL& coef_E_zx,
            device const $REAL& coef_E_zy,
            device $REAL* E_x,
            device $REAL* E_y,
            device $REAL* E_z,
            device const $REAL* H_x,
            device const $REAL* H_y,
            device const $REAL* H_z,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // 1D DPW auxiliary grid - standard (homogeneous) electric field update.
    // Updates E_x, E_y, E_z at each position j in the bulk 1D grid.
    // Scalar coefficients are passed as kernel arguments (same for all cells).
    //
    // Args:
    //   N:    total length of 1D DPW array.
    //   M:    max(m_x, m_y, m_z).
    //   m_x, m_y, m_z: integer direction cosines.
    //   coef_E_*: scalar update coefficients from background material.
    //             coef_E_xt = updatecoeffsE[0] (CA - self decay)
    //             coef_E_xy = updatecoeffsE[2] (CBy)
    //             coef_E_xz = updatecoeffsE[3] (CBz)
    //             coef_E_yx = updatecoeffsE[1] (CBx)
    //             coef_E_yz = updatecoeffsE[3] (CBz)
    //             coef_E_yt = updatecoeffsE[0] (CA)
    //             coef_E_zt = updatecoeffsE[0] (CA)
    //             coef_E_zx = updatecoeffsE[1] (CBx)
    //             coef_E_zy = updatecoeffsE[2] (CBy)
    //   E_x, E_y, E_z: 1D electric field arrays - shape [N].
    //   H_x, H_y, H_z: 1D magnetic field arrays - shape [N].

    $CUDA_IDX

    // Offset thread index into valid bulk range [M, N-M)
    int j = i + M;

    if (j >= N - M) return;

    // E_x update: curl of H in yz plane
    // E_x[j] = CA*E_x[j] + CBy*(H_z[j]-H_z[j-m_y]) - CBz*(H_y[j]-H_y[j-m_z])
    E_x[j] = coef_E_xt * E_x[j]
            + coef_E_xy * (H_z[j] - H_z[j - m_y])
            - coef_E_xz * (H_y[j] - H_y[j - m_z]);

    // E_y update: curl of H in xz plane
    // E_y[j] = CA*E_y[j] + CBz*(H_x[j]-H_x[j-m_z]) - CBx*(H_z[j]-H_z[j-m_x])
    E_y[j] = coef_E_yt * E_y[j]
            + coef_E_yz * (H_x[j] - H_x[j - m_z])
            - coef_E_yx * (H_z[j] - H_z[j - m_x]);

    // E_z update: curl of H in xy plane
    // E_z[j] = CA*E_z[j] + CBx*(H_y[j]-H_y[j-m_x]) - CBy*(H_x[j]-H_x[j-m_y])
    E_z[j] = coef_E_zt * E_z[j]
            + coef_E_zx * (H_y[j] - H_y[j - m_x])
            - coef_E_zy * (H_x[j] - H_x[j - m_y]);
    """
    ),
}


# STANDARD (HOMOGENEOUS) ELECTRIC PML UPDATE

# Ports- PML section of updateElectricFields() from plane_wave.pyx
# Same structure as magnetic PML but for E fields and rcEx, rcEy, rcEz.

update_1d_electric_pml = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_pml(
            const int N,
            const int P,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL coef_E_D,
            const $REAL dx,
            const $REAL dy,
            const $REAL dz,
            $REAL* __restrict__ E_x,
            $REAL* __restrict__ E_y,
            $REAL* __restrict__ E_z,
            const $REAL* __restrict__ H_x,
            const $REAL* __restrict__ H_y,
            const $REAL* __restrict__ H_z,
            $REAL* __restrict__ Jxmzy,
            $REAL* __restrict__ Jxmyz,
            $REAL* __restrict__ Jymzx,
            $REAL* __restrict__ Jymxz,
            $REAL* __restrict__ Jzmyx,
            $REAL* __restrict__ Jzmxy,
            const $REAL* __restrict__ RAEx,
            const $REAL* __restrict__ RBEx,
            const $REAL* __restrict__ RCEx,
            const $REAL* __restrict__ RDEx,
            const $REAL* __restrict__ RAEy,
            const $REAL* __restrict__ RBEy,
            const $REAL* __restrict__ RCEy,
            const $REAL* __restrict__ RDEy,
            const $REAL* __restrict__ RAEz,
            const $REAL* __restrict__ RBEz,
            const $REAL* __restrict__ RCEz,
            const $REAL* __restrict__ RDEz
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int P,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL coef_E_D,
        const $REAL dx,
        const $REAL dy,
        const $REAL dz,
        __global $REAL* restrict E_x,
        __global $REAL* restrict E_y,
        __global $REAL* restrict E_z,
        __global const $REAL* restrict H_x,
        __global const $REAL* restrict H_y,
        __global const $REAL* restrict H_z,
        __global $REAL* restrict Jxmzy,
        __global $REAL* restrict Jxmyz,
        __global $REAL* restrict Jymzx,
        __global $REAL* restrict Jymxz,
        __global $REAL* restrict Jzmyx,
        __global $REAL* restrict Jzmxy,
        __global const $REAL* restrict RAEx,
        __global const $REAL* restrict RBEx,
        __global const $REAL* restrict RCEx,
        __global const $REAL* restrict RDEx,
        __global const $REAL* restrict RAEy,
        __global const $REAL* restrict RBEy,
        __global const $REAL* restrict RCEy,
        __global const $REAL* restrict RDEy,
        __global const $REAL* restrict RAEz,
        __global const $REAL* restrict RBEz,
        __global const $REAL* restrict RCEz,
        __global const $REAL* restrict RDEz
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_pml(
            device const int& N,
            device const int& P,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& coef_E_D,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device $REAL* E_x,
            device $REAL* E_y,
            device $REAL* E_z,
            device const $REAL* H_x,
            device const $REAL* H_y,
            device const $REAL* H_z,
            device $REAL* Jxmzy,
            device $REAL* Jxmyz,
            device $REAL* Jymzx,
            device $REAL* Jymxz,
            device $REAL* Jzmyx,
            device $REAL* Jzmxy,
            device const $REAL* RAEx,
            device const $REAL* RBEx,
            device const $REAL* RCEx,
            device const $REAL* RDEx,
            device const $REAL* RAEy,
            device const $REAL* RBEy,
            device const $REAL* RCEy,
            device const $REAL* RDEy,
            device const $REAL* RAEz,
            device const $REAL* RBEz,
            device const $REAL* RCEz,
            device const $REAL* RDEz,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // 1D DPW PML electric field update - standard (homogeneous) case.
    //
    // Updates E fields and PML integral arrays at the p PML cells
    // at the end of the 1D DPW grid using the first-order RIPML scheme.
    //
    // Args:
    //   N:   total length of 1D DPW array.
    //   P:   PML thickness in number of cells.
    //   M:   max(m_x, m_y, m_z).
    //   coef_E_D: updatecoeffsE[4] (srce - source coefficient).
    //   dx, dy, dz: spatial step sizes.
    //   E_x/y/z:   1D electric field arrays.
    //   H_x/y/z:   1D magnetic field arrays.
    //   Jxmzy..Jzmxy: PML integral arrays - shape [P] each.
    //   RAEx..RDEz:   PML coefficients - shape [P] each.

    $CUDA_IDX

    if (i >= P) return;

    int pml_i = P - 1 - i;
    int idx = (N - M) - (P - pml_i);

    $REAL dHzy, dHyz, dHzx, dHxz, dHyx, dHxy;
    $REAL jxy, jxz, jyx, jyz, jzx, jzy;

    // E_x PML correction
    dHzy = (H_z[idx] - H_z[idx - m_y]) / dy;
    dHyz = (H_y[idx] - H_y[idx - m_z]) / dz;

    jxy = RAEx[pml_i] * dHzy + RBEx[pml_i] * Jxmzy[pml_i];
    jxz = RAEx[pml_i] * dHyz + RBEx[pml_i] * Jxmyz[pml_i];

    E_x[idx] = E_x[idx] + coef_E_D * jxy - coef_E_D * jxz;

    Jxmzy[pml_i] = Jxmzy[pml_i] - RCEx[pml_i] * jxy + RDEx[pml_i] * dHzy;
    Jxmyz[pml_i] = Jxmyz[pml_i] - RCEx[pml_i] * jxz + RDEx[pml_i] * dHyz;

    //  E_y PML correction
    dHzx = (H_z[idx] - H_z[idx - m_x]) / dx;
    dHxz = (H_x[idx] - H_x[idx - m_z]) / dz;

    jyx = RAEy[pml_i] * dHzx + RBEy[pml_i] * Jymzx[pml_i];
    jyz = RAEy[pml_i] * dHxz + RBEy[pml_i] * Jymxz[pml_i];

    E_y[idx] = E_y[idx] - coef_E_D * jyx + coef_E_D * jyz;

    Jymzx[pml_i] = Jymzx[pml_i] - RCEy[pml_i] * jyx + RDEy[pml_i] * dHzx;
    Jymxz[pml_i] = Jymxz[pml_i] - RCEy[pml_i] * jyz + RDEy[pml_i] * dHxz;

    // E_z PML correction
    dHyx = (H_y[idx] - H_y[idx - m_x]) / dx;
    dHxy = (H_x[idx] - H_x[idx - m_y]) / dy;

    jzx = RAEz[pml_i] * dHyx + RBEz[pml_i] * Jzmyx[pml_i];
    jzy = RAEz[pml_i] * dHxy + RBEz[pml_i] * Jzmxy[pml_i];

    E_z[idx] = E_z[idx] + coef_E_D * jzx - coef_E_D * jzy;

    Jzmyx[pml_i] = Jzmyx[pml_i] - RCEz[pml_i] * jzx + RDEz[pml_i] * dHyx;
    Jzmxy[pml_i] = Jzmxy[pml_i] - RCEz[pml_i] * jzy + RDEz[pml_i] * dHxy;
    """
    ),
}


# NOTE: Axial variants (update_1d_magnetic_axial, update_1d_electric_axial)
# will be added here after standard variants are validated.
# They follow the same dict structure but use per-cell GID lookup-
#   updatecoeffsH[ID[component * N + j], col]
# and require three sequential kernel launches due to source injection
# dependency at origin_axial.


# AXIAL MAGNETIC UPDATE - KERNEL 1 OF 3
# Ports: source grid bulk update section of updateMagneticFields_axial()
#
# Updates H_fields_s (source 1D grid) in parallel over j in [M, N-M).
# Source grid uses fixed background material at GID position 2.
# Coefficients accessed via updatecoeffsH in constant memory.
#
# GID layout: GID[component, position] - shape [6, N] flattened row-major.
# Components: 0=Ex,1=Ey,2=Ez,3=Hx,4=Hy,5=Hz
# Source grid always uses GID[component, 2] - fixed scalar per component.

update_1d_magnetic_axial_source = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_magnetic_axial_source(
            const int N,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            $REAL* __restrict__ H_x_s,
            $REAL* __restrict__ H_y_s,
            $REAL* __restrict__ H_z_s,
            const $REAL* __restrict__ E_x_s,
            const $REAL* __restrict__ E_y_s,
            const $REAL* __restrict__ E_z_s,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        __global $REAL* restrict H_x_s,
        __global $REAL* restrict H_y_s,
        __global $REAL* restrict H_z_s,
        __global const $REAL* restrict E_x_s,
        __global const $REAL* restrict E_y_s,
        __global const $REAL* restrict E_z_s,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_magnetic_axial_source(
            device const int& N,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device $REAL* H_x_s,
            device $REAL* H_y_s,
            device $REAL* H_z_s,
            device const $REAL* E_x_s,
            device const $REAL* E_y_s,
            device const $REAL* E_z_s,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - source grid magnetic field update (Kernel 1 of 3).
    //
    // Updates H_x_s, H_y_s, H_z_s for the source 1D grid.
    // Source grid uses fixed background material at GID position 2:
    //   GID[3, 2] for H_x, GID[4, 2] for H_y, GID[5, 2] for H_z
    // matH is in constant memory - accessed via IDX2D_MAT.
    //
    // GID array layout: [6 components x N positions] row-major.
    // GID[component, j] = GID[component * N + j]
    //
    // Args:
    //   N:   total length of 1D DPW array.
    //   M:   max(m_x, m_y, m_z) - loop bounds [M, N-M).
    //   m_x, m_y, m_z: integer direction cosines.
    //   H_x_s/y_s/z_s: source grid magnetic field arrays - shape [N].
    //   E_x_s/y_s/z_s: source grid electric field arrays - shape [N].
    //   GID: 1D material ID array - shape [6, N] flattened - shape [6*N].

    $CUDA_IDX

    int j = i + M;
    if (j >= N - M) return;

    // Fixed material IDs at position 2 for each H component
    int mat_hx = GID[3 * N + 2];   // GID[3, 2]
    int mat_hy = GID[4 * N + 2];   // GID[4, 2]
    int mat_hz = GID[5 * N + 2];   // GID[5, 2]

    // H_x_s update - equation 8 of Tan & Potter (2010)
    H_x_s[j] = matH[IDX2D_MAT(mat_hx, 0)] * H_x_s[j]
              + matH[IDX2D_MAT(mat_hx, 3)] * (E_y_s[j + m_z] - E_y_s[j])
              - matH[IDX2D_MAT(mat_hx, 2)] * (E_z_s[j + m_y] - E_z_s[j]);

    // H_y_s update
    H_y_s[j] = matH[IDX2D_MAT(mat_hy, 0)] * H_y_s[j]
              + matH[IDX2D_MAT(mat_hy, 1)] * (E_z_s[j + m_x] - E_z_s[j])
              - matH[IDX2D_MAT(mat_hy, 3)] * (E_x_s[j + m_z] - E_x_s[j]);

    // H_z_s update
    H_z_s[j] = matH[IDX2D_MAT(mat_hz, 0)] * H_z_s[j]
              + matH[IDX2D_MAT(mat_hz, 2)] * (E_x_s[j + m_y] - E_x_s[j])
              - matH[IDX2D_MAT(mat_hz, 1)] * (E_y_s[j + m_x] - E_y_s[j]);
    """
    ),
}


# AXIAL MAGNETIC UPDATE - SOURCE GRID PML

# Ports: PML section of source grid in updateMagneticFields_axial()
# Uses rcHx0, rcHy0, rcHz0 and Ix_s, Iy_s, Iz_s.
# Fixed material at GID[component, 2].

update_1d_magnetic_axial_source_pml = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_magnetic_axial_source_pml(
            const int N,
            const int P,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL dx,
            const $REAL dy,
            const $REAL dz,
            $REAL* __restrict__ H_x_s,
            $REAL* __restrict__ H_y_s,
            $REAL* __restrict__ H_z_s,
            const $REAL* __restrict__ E_x_s,
            const $REAL* __restrict__ E_y_s,
            const $REAL* __restrict__ E_z_s,
            $REAL* __restrict__ Ixmzy_s,
            $REAL* __restrict__ Ixmyz_s,
            $REAL* __restrict__ Iymzx_s,
            $REAL* __restrict__ Iymxz_s,
            $REAL* __restrict__ Izmyx_s,
            $REAL* __restrict__ Izmxy_s,
            const $REAL* __restrict__ RAHx0,
            const $REAL* __restrict__ RBHx0,
            const $REAL* __restrict__ RCHx0,
            const $REAL* __restrict__ RDHx0,
            const $REAL* __restrict__ RAHy0,
            const $REAL* __restrict__ RBHy0,
            const $REAL* __restrict__ RCHy0,
            const $REAL* __restrict__ RDHy0,
            const $REAL* __restrict__ RAHz0,
            const $REAL* __restrict__ RBHz0,
            const $REAL* __restrict__ RCHz0,
            const $REAL* __restrict__ RDHz0,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int P,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL dx,
        const $REAL dy,
        const $REAL dz,
        __global $REAL* restrict H_x_s,
        __global $REAL* restrict H_y_s,
        __global $REAL* restrict H_z_s,
        __global const $REAL* restrict E_x_s,
        __global const $REAL* restrict E_y_s,
        __global const $REAL* restrict E_z_s,
        __global $REAL* restrict Ixmzy_s,
        __global $REAL* restrict Ixmyz_s,
        __global $REAL* restrict Iymzx_s,
        __global $REAL* restrict Iymxz_s,
        __global $REAL* restrict Izmyx_s,
        __global $REAL* restrict Izmxy_s,
        __global const $REAL* restrict RAHx0,
        __global const $REAL* restrict RBHx0,
        __global const $REAL* restrict RCHx0,
        __global const $REAL* restrict RDHx0,
        __global const $REAL* restrict RAHy0,
        __global const $REAL* restrict RBHy0,
        __global const $REAL* restrict RCHy0,
        __global const $REAL* restrict RDHy0,
        __global const $REAL* restrict RAHz0,
        __global const $REAL* restrict RBHz0,
        __global const $REAL* restrict RCHz0,
        __global const $REAL* restrict RDHz0,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_magnetic_axial_source_pml(
            device const int& N,
            device const int& P,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device $REAL* H_x_s,
            device $REAL* H_y_s,
            device $REAL* H_z_s,
            device const $REAL* E_x_s,
            device const $REAL* E_y_s,
            device const $REAL* E_z_s,
            device $REAL* Ixmzy_s,
            device $REAL* Ixmyz_s,
            device $REAL* Iymzx_s,
            device $REAL* Iymxz_s,
            device $REAL* Izmyx_s,
            device $REAL* Izmxy_s,
            device const $REAL* RAHx0,
            device const $REAL* RBHx0,
            device const $REAL* RCHx0,
            device const $REAL* RDHx0,
            device const $REAL* RAHy0,
            device const $REAL* RBHy0,
            device const $REAL* RCHy0,
            device const $REAL* RDHy0,
            device const $REAL* RAHz0,
            device const $REAL* RBHz0,
            device const $REAL* RCHz0,
            device const $REAL* RDHz0,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - source grid PML magnetic update.
    //
    // Updates source grid H fields and PML integrals at p PML cells.
    // Uses rcHx0/rcHy0/rcHz0 coefficients (second PML region arrays).
    // Fixed material at GID[component, 2] for srcm coefficient.

    $CUDA_IDX

    if (i >= P) return;

    int pml_i = P - 1 - i;
    int idx = (N - M) - (P - pml_i);

    // Fixed material srcm coefficients for source grid
    int mat_hx = GID[3 * N + 2];
    int mat_hy = GID[4 * N + 2];
    int mat_hz = GID[5 * N + 2];

    $REAL coef_hx_D = matH[IDX2D_MAT(mat_hx, 4)];
    $REAL coef_hy_D = matH[IDX2D_MAT(mat_hy, 4)];
    $REAL coef_hz_D = matH[IDX2D_MAT(mat_hz, 4)];

    $REAL dEzy, dEyz, dEzx, dExz, dEyx, dExy;
    $REAL mxy, mxz, myx, myz, mzx, mzy;

    // H_x_s PML
    dEzy = (E_z_s[idx + m_y] - E_z_s[idx]) / dy;
    dEyz = (E_y_s[idx + m_z] - E_y_s[idx]) / dz;
    mxy = RAHx0[pml_i] * dEzy + RBHx0[pml_i] * Ixmzy_s[pml_i];
    mxz = RAHx0[pml_i] * dEyz + RBHx0[pml_i] * Ixmyz_s[pml_i];
    H_x_s[idx] = H_x_s[idx] - coef_hx_D * mxy + coef_hx_D * mxz;
    Ixmzy_s[pml_i] = Ixmzy_s[pml_i] - RCHx0[pml_i] * mxy + RDHx0[pml_i] * dEzy;
    Ixmyz_s[pml_i] = Ixmyz_s[pml_i] - RCHx0[pml_i] * mxz + RDHx0[pml_i] * dEyz;

    // H_y_s PML
    dEzx = (E_z_s[idx + m_x] - E_z_s[idx]) / dx;
    dExz = (E_x_s[idx + m_z] - E_x_s[idx]) / dz;
    myx = RAHy0[pml_i] * dEzx + RBHy0[pml_i] * Iymzx_s[pml_i];
    myz = RAHy0[pml_i] * dExz + RBHy0[pml_i] * Iymxz_s[pml_i];
    H_y_s[idx] = H_y_s[idx] + coef_hy_D * myx - coef_hy_D * myz;
    Iymzx_s[pml_i] = Iymzx_s[pml_i] - RCHy0[pml_i] * myx + RDHy0[pml_i] * dEzx;
    Iymxz_s[pml_i] = Iymxz_s[pml_i] - RCHy0[pml_i] * myz + RDHy0[pml_i] * dExz;

    // H_z_s PML
    dEyx = (E_y_s[idx + m_x] - E_y_s[idx]) / dx;
    dExy = (E_x_s[idx + m_y] - E_x_s[idx]) / dy;
    mzx = RAHz0[pml_i] * dEyx + RBHz0[pml_i] * Izmyx_s[pml_i];
    mzy = RAHz0[pml_i] * dExy + RBHz0[pml_i] * Izmxy_s[pml_i];
    H_z_s[idx] = H_z_s[idx] - coef_hz_D * mzx + coef_hz_D * mzy;
    Izmyx_s[pml_i] = Izmyx_s[pml_i] - RCHz0[pml_i] * mzx + RDHz0[pml_i] * dEyx;
    Izmxy_s[pml_i] = Izmxy_s[pml_i] - RCHz0[pml_i] * mzy + RDHz0[pml_i] * dExy;
    """
    ),
}


# AXIAL MAGNETIC UPDATE - KERNEL 2 OF 3 (SOURCE INJECTION)
# Ports- single-point source injection in updateMagneticFields_axial()
# H_x[src-2] -= matH[GID[3,src-2], 3] * E_y_s[src-2+m_z]
#             - matH[GID[3,src-2], 2] * E_z_s[src-2+m_y]
# (and same for H_y, H_z)
# Launched with block=(1,1,1), grid=(1,1,1) - single thread.
# Must run AFTER source grid is fully updated (Kernel 1 + source PML).

update_1d_magnetic_axial_inject = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(1, 1) update_1d_magnetic_axial_inject(
            const int N,
            const int src,
            const int m_x,
            const int m_y,
            const int m_z,
            $REAL* __restrict__ H_x,
            $REAL* __restrict__ H_y,
            $REAL* __restrict__ H_z,
            const $REAL* __restrict__ E_x_s,
            const $REAL* __restrict__ E_y_s,
            const $REAL* __restrict__ E_z_s,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int src,
        const int m_x,
        const int m_y,
        const int m_z,
        __global $REAL* restrict H_x,
        __global $REAL* restrict H_y,
        __global $REAL* restrict H_z,
        __global const $REAL* restrict E_x_s,
        __global const $REAL* restrict E_y_s,
        __global const $REAL* restrict E_z_s,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_magnetic_axial_inject(
            device const int& N,
            device const int& src,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device $REAL* H_x,
            device $REAL* H_y,
            device $REAL* H_z,
            device const $REAL* E_x_s,
            device const $REAL* E_y_s,
            device const $REAL* E_z_s,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - source grid injection into main grid (Kernel 2 of 3).
    //
    // Injects source grid E fields into main grid H fields at src-2.
    // Single thread kernel - launched with block=(1,1,1), grid=(1,1,1).
    // Must run after source grid bulk + PML update is complete.
    //
    // Ports:
    //   H_x[src-2] -= matH[GID[3,src-2],3]*E_y_s[src-2+m_z]
    //               - matH[GID[3,src-2],2]*E_z_s[src-2+m_y]
    //   H_y[src-2] -= matH[GID[4,src-2],1]*E_z_s[src-2+m_x]
    //               - matH[GID[4,src-2],3]*E_x_s[src-2+m_z]
    //   H_z[src-2] -= matH[GID[5,src-2],2]*E_x_s[src-2+m_y]
    //               - matH[GID[5,src-2],1]*E_y_s[src-2+m_x]

    // Single thread - no index needed, always runs
    int pos = src - 2;

    int mat_hx = GID[3 * N + pos];   // GID[3, src-2]
    int mat_hy = GID[4 * N + pos];   // GID[4, src-2]
    int mat_hz = GID[5 * N + pos];   // GID[5, src-2]

    H_x[pos] -= matH[IDX2D_MAT(mat_hx, 3)] * E_y_s[pos + m_z]
              - matH[IDX2D_MAT(mat_hx, 2)] * E_z_s[pos + m_y];

    H_y[pos] -= matH[IDX2D_MAT(mat_hy, 1)] * E_z_s[pos + m_x]
              - matH[IDX2D_MAT(mat_hy, 3)] * E_x_s[pos + m_z];

    H_z[pos] -= matH[IDX2D_MAT(mat_hz, 2)] * E_x_s[pos + m_y]
              - matH[IDX2D_MAT(mat_hz, 1)] * E_y_s[pos + m_x];
    """
    ),
}


# AXIAL MAGNETIC UPDATE - KERNEL 3 OF 3 (MAIN GRID BULK)

# Ports: main grid bulk update in updateMagneticFields_axial()
#
# Per-cell material lookup: matH[GID[component, j], col]
# Loop range: j in [M-1, N-M) - note M-1 to include origin point
# Must run AFTER injection kernel (Kernel 2).

update_1d_magnetic_axial_main = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_magnetic_axial_main(
            const int N,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            $REAL* __restrict__ H_x,
            $REAL* __restrict__ H_y,
            $REAL* __restrict__ H_z,
            const $REAL* __restrict__ E_x,
            const $REAL* __restrict__ E_y,
            const $REAL* __restrict__ E_z,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        __global $REAL* restrict H_x,
        __global $REAL* restrict H_y,
        __global $REAL* restrict H_z,
        __global const $REAL* restrict E_x,
        __global const $REAL* restrict E_y,
        __global const $REAL* restrict E_z,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_magnetic_axial_main(
            device const int& N,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device $REAL* H_x,
            device $REAL* H_y,
            device $REAL* H_z,
            device const $REAL* E_x,
            device const $REAL* E_y,
            device const $REAL* E_z,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - main grid magnetic field update (Kernel 3 of 3).
    //
    // Updates H_x, H_y, H_z for the main 1D DPW grid.
    // Per-cell material lookup: matH[GID[component, j], col]
    // Range: j in [M-1, N-M) - includes origin point (M-1) for scattered field.
    // Must run after injection kernel.
    //
    // GID[component, j] = GID[component * N + j]

    $CUDA_IDX

    // Offset into [M-1, N-M) - includes the extra origin point
    int j = i + (M - 1);
    if (j >= N - M) return;

    int mat_hx = GID[3 * N + j];   // GID[3, j]
    int mat_hy = GID[4 * N + j];   // GID[4, j]
    int mat_hz = GID[5 * N + j];   // GID[5, j]

    // H_x update - per-cell coefficients
    H_x[j] = matH[IDX2D_MAT(mat_hx, 0)] * H_x[j]
            + matH[IDX2D_MAT(mat_hx, 3)] * (E_y[j + m_z] - E_y[j])
            - matH[IDX2D_MAT(mat_hx, 2)] * (E_z[j + m_y] - E_z[j]);

    // H_y update
    H_y[j] = matH[IDX2D_MAT(mat_hy, 0)] * H_y[j]
            + matH[IDX2D_MAT(mat_hy, 1)] * (E_z[j + m_x] - E_z[j])
            - matH[IDX2D_MAT(mat_hy, 3)] * (E_x[j + m_z] - E_x[j]);

    // H_z update
    H_z[j] = matH[IDX2D_MAT(mat_hz, 0)] * H_z[j]
            + matH[IDX2D_MAT(mat_hz, 2)] * (E_x[j + m_y] - E_x[j])
            - matH[IDX2D_MAT(mat_hz, 1)] * (E_y[j + m_x] - E_y[j]);
    """
    ),
}


# AXIAL MAGNETIC UPDATE - MAIN GRID PML (END REGION)
# Ports: PML end region of main grid in updateMagneticFields_axial()
# Uses rcHx, rcHy, rcHz and Ix, Iy, Iz.
# Per-cell srcm coefficient: matH[GID[component, idx], 4]


update_1d_magnetic_axial_main_pml_end = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_magnetic_axial_main_pml_end(
            const int N,
            const int P,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL dx,
            const $REAL dy,
            const $REAL dz,
            $REAL* __restrict__ H_x,
            $REAL* __restrict__ H_y,
            $REAL* __restrict__ H_z,
            const $REAL* __restrict__ E_x,
            const $REAL* __restrict__ E_y,
            const $REAL* __restrict__ E_z,
            $REAL* __restrict__ Ixmzy,
            $REAL* __restrict__ Ixmyz,
            $REAL* __restrict__ Iymzx,
            $REAL* __restrict__ Iymxz,
            $REAL* __restrict__ Izmyx,
            $REAL* __restrict__ Izmxy,
            const $REAL* __restrict__ RAHx,
            const $REAL* __restrict__ RBHx,
            const $REAL* __restrict__ RCHx,
            const $REAL* __restrict__ RDHx,
            const $REAL* __restrict__ RAHy,
            const $REAL* __restrict__ RBHy,
            const $REAL* __restrict__ RCHy,
            const $REAL* __restrict__ RDHy,
            const $REAL* __restrict__ RAHz,
            const $REAL* __restrict__ RBHz,
            const $REAL* __restrict__ RCHz,
            const $REAL* __restrict__ RDHz,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int P,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL dx,
        const $REAL dy,
        const $REAL dz,
        __global $REAL* restrict H_x,
        __global $REAL* restrict H_y,
        __global $REAL* restrict H_z,
        __global const $REAL* restrict E_x,
        __global const $REAL* restrict E_y,
        __global const $REAL* restrict E_z,
        __global $REAL* restrict Ixmzy,
        __global $REAL* restrict Ixmyz,
        __global $REAL* restrict Iymzx,
        __global $REAL* restrict Iymxz,
        __global $REAL* restrict Izmyx,
        __global $REAL* restrict Izmxy,
        __global const $REAL* restrict RAHx,
        __global const $REAL* restrict RBHx,
        __global const $REAL* restrict RCHx,
        __global const $REAL* restrict RDHx,
        __global const $REAL* restrict RAHy,
        __global const $REAL* restrict RBHy,
        __global const $REAL* restrict RCHy,
        __global const $REAL* restrict RDHy,
        __global const $REAL* restrict RAHz,
        __global const $REAL* restrict RBHz,
        __global const $REAL* restrict RCHz,
        __global const $REAL* restrict RDHz,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_magnetic_axial_main_pml_end(
            device const int& N,
            device const int& P,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device $REAL* H_x,
            device $REAL* H_y,
            device $REAL* H_z,
            device const $REAL* E_x,
            device const $REAL* E_y,
            device const $REAL* E_z,
            device $REAL* Ixmzy,
            device $REAL* Ixmyz,
            device $REAL* Iymzx,
            device $REAL* Iymxz,
            device $REAL* Izmyx,
            device $REAL* Izmxy,
            device const $REAL* RAHx,
            device const $REAL* RBHx,
            device const $REAL* RCHx,
            device const $REAL* RDHx,
            device const $REAL* RAHy,
            device const $REAL* RBHy,
            device const $REAL* RCHy,
            device const $REAL* RDHy,
            device const $REAL* RAHz,
            device const $REAL* RBHz,
            device const $REAL* RCHz,
            device const $REAL* RDHz,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - main grid PML end region magnetic update.
    //
    // Updates H fields and PML integrals at p cells at END of main grid.
    // Per-cell srcm: matH[GID[component, idx], 4]
    // idx = (N-M) - (P - pml_i)

    $CUDA_IDX

    if (i >= P) return;

    int pml_i = P - 1 - i;
    int idx = (N - M) - (P - pml_i);

    int mat_hx = GID[3 * N + idx];
    int mat_hy = GID[4 * N + idx];
    int mat_hz = GID[5 * N + idx];

    $REAL dEzy, dEyz, dEzx, dExz, dEyx, dExy;
    $REAL mxy, mxz, myx, myz, mzx, mzy;

    // H_x PML end
    dEzy = (E_z[idx + m_y] - E_z[idx]) / dy;
    dEyz = (E_y[idx + m_z] - E_y[idx]) / dz;
    mxy = RAHx[pml_i] * dEzy + RBHx[pml_i] * Ixmzy[pml_i];
    mxz = RAHx[pml_i] * dEyz + RBHx[pml_i] * Ixmyz[pml_i];
    H_x[idx] = H_x[idx] - matH[IDX2D_MAT(mat_hx, 4)] * mxy
                         + matH[IDX2D_MAT(mat_hx, 4)] * mxz;
    Ixmzy[pml_i] = Ixmzy[pml_i] - RCHx[pml_i] * mxy + RDHx[pml_i] * dEzy;
    Ixmyz[pml_i] = Ixmyz[pml_i] - RCHx[pml_i] * mxz + RDHx[pml_i] * dEyz;

    // H_y PML end
    dEzx = (E_z[idx + m_x] - E_z[idx]) / dx;
    dExz = (E_x[idx + m_z] - E_x[idx]) / dz;
    myx = RAHy[pml_i] * dEzx + RBHy[pml_i] * Iymzx[pml_i];
    myz = RAHy[pml_i] * dExz + RBHy[pml_i] * Iymxz[pml_i];
    H_y[idx] = H_y[idx] + matH[IDX2D_MAT(mat_hy, 4)] * myx
                         - matH[IDX2D_MAT(mat_hy, 4)] * myz;
    Iymzx[pml_i] = Iymzx[pml_i] - RCHy[pml_i] * myx + RDHy[pml_i] * dEzx;
    Iymxz[pml_i] = Iymxz[pml_i] - RCHy[pml_i] * myz + RDHy[pml_i] * dExz;

    // H_z PML end
    dEyx = (E_y[idx + m_x] - E_y[idx]) / dx;
    dExy = (E_x[idx + m_y] - E_x[idx]) / dy;
    mzx = RAHz[pml_i] * dEyx + RBHz[pml_i] * Izmyx[pml_i];
    mzy = RAHz[pml_i] * dExy + RBHz[pml_i] * Izmxy[pml_i];
    H_z[idx] = H_z[idx] - matH[IDX2D_MAT(mat_hz, 4)] * mzx
                         + matH[IDX2D_MAT(mat_hz, 4)] * mzy;
    Izmyx[pml_i] = Izmyx[pml_i] - RCHz[pml_i] * mzx + RDHz[pml_i] * dEyx;
    Izmxy[pml_i] = Izmxy[pml_i] - RCHz[pml_i] * mzy + RDHz[pml_i] * dExy;
    """
    ),
}


# AXIAL MAGNETIC UPDATE - MAIN GRID PML (START REGION)
# Ports: PML start region of main grid in updateMagneticFields_axial()
# Uses rcHx0, rcHy0, rcHz0 and Ix0, Iy0, Iz0.
# idx = (P - pml_i) - 1  (start of grid, not end)


update_1d_magnetic_axial_main_pml_start = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_magnetic_axial_main_pml_start(
            const int N,
            const int P,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL dx,
            const $REAL dy,
            const $REAL dz,
            $REAL* __restrict__ H_x,
            $REAL* __restrict__ H_y,
            $REAL* __restrict__ H_z,
            const $REAL* __restrict__ E_x,
            const $REAL* __restrict__ E_y,
            const $REAL* __restrict__ E_z,
            $REAL* __restrict__ Ixmzy0,
            $REAL* __restrict__ Ixmyz0,
            $REAL* __restrict__ Iymzx0,
            $REAL* __restrict__ Iymxz0,
            $REAL* __restrict__ Izmyx0,
            $REAL* __restrict__ Izmxy0,
            const $REAL* __restrict__ RAHx0,
            const $REAL* __restrict__ RBHx0,
            const $REAL* __restrict__ RCHx0,
            const $REAL* __restrict__ RDHx0,
            const $REAL* __restrict__ RAHy0,
            const $REAL* __restrict__ RBHy0,
            const $REAL* __restrict__ RCHy0,
            const $REAL* __restrict__ RDHy0,
            const $REAL* __restrict__ RAHz0,
            const $REAL* __restrict__ RBHz0,
            const $REAL* __restrict__ RCHz0,
            const $REAL* __restrict__ RDHz0,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int P,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL dx,
        const $REAL dy,
        const $REAL dz,
        __global $REAL* restrict H_x,
        __global $REAL* restrict H_y,
        __global $REAL* restrict H_z,
        __global const $REAL* restrict E_x,
        __global const $REAL* restrict E_y,
        __global const $REAL* restrict E_z,
        __global $REAL* restrict Ixmzy0,
        __global $REAL* restrict Ixmyz0,
        __global $REAL* restrict Iymzx0,
        __global $REAL* restrict Iymxz0,
        __global $REAL* restrict Izmyx0,
        __global $REAL* restrict Izmxy0,
        __global const $REAL* restrict RAHx0,
        __global const $REAL* restrict RBHx0,
        __global const $REAL* restrict RCHx0,
        __global const $REAL* restrict RDHx0,
        __global const $REAL* restrict RAHy0,
        __global const $REAL* restrict RBHy0,
        __global const $REAL* restrict RCHy0,
        __global const $REAL* restrict RDHy0,
        __global const $REAL* restrict RAHz0,
        __global const $REAL* restrict RBHz0,
        __global const $REAL* restrict RCHz0,
        __global const $REAL* restrict RDHz0,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_magnetic_axial_main_pml_start(
            device const int& N,
            device const int& P,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device $REAL* H_x,
            device $REAL* H_y,
            device $REAL* H_z,
            device const $REAL* E_x,
            device const $REAL* E_y,
            device const $REAL* E_z,
            device $REAL* Ixmzy0,
            device $REAL* Ixmyz0,
            device $REAL* Iymzx0,
            device $REAL* Iymxz0,
            device $REAL* Izmyx0,
            device $REAL* Izmxy0,
            device const $REAL* RAHx0,
            device const $REAL* RBHx0,
            device const $REAL* RCHx0,
            device const $REAL* RDHx0,
            device const $REAL* RAHy0,
            device const $REAL* RBHy0,
            device const $REAL* RCHy0,
            device const $REAL* RDHy0,
            device const $REAL* RAHz0,
            device const $REAL* RBHz0,
            device const $REAL* RCHz0,
            device const $REAL* RDHz0,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - main grid PML start region magnetic update.
    //
    // Updates H fields and PML integrals at p cells at START of main grid.
    // Uses rcHx0/rcHy0/rcHz0 and Ix0/Iy0/Iz0.
    // Per-cell srcm: matH[GID[component, idx], 4]
    // idx = (P - pml_i) - 1  (start of grid)

    $CUDA_IDX

    if (i >= P) return;

    int pml_i = P - 1 - i;
    int idx = (P - pml_i) - 1;

    int mat_hx = GID[3 * N + idx];
    int mat_hy = GID[4 * N + idx];
    int mat_hz = GID[5 * N + idx];

    $REAL dEzy, dEyz, dEzx, dExz, dEyx, dExy;
    $REAL mxy, mxz, myx, myz, mzx, mzy;

    // H_x PML start
    dEzy = (E_z[idx + m_y] - E_z[idx]) / dy;
    dEyz = (E_y[idx + m_z] - E_y[idx]) / dz;
    mxy = RAHx0[pml_i] * dEzy + RBHx0[pml_i] * Ixmzy0[pml_i];
    mxz = RAHx0[pml_i] * dEyz + RBHx0[pml_i] * Ixmyz0[pml_i];
    H_x[idx] = H_x[idx] - matH[IDX2D_MAT(mat_hx, 4)] * mxy
                         + matH[IDX2D_MAT(mat_hx, 4)] * mxz;
    Ixmzy0[pml_i] = Ixmzy0[pml_i] - RCHx0[pml_i] * mxy + RDHx0[pml_i] * dEzy;
    Ixmyz0[pml_i] = Ixmyz0[pml_i] - RCHx0[pml_i] * mxz + RDHx0[pml_i] * dEyz;

    // H_y PML start
    dEzx = (E_z[idx + m_x] - E_z[idx]) / dx;
    dExz = (E_x[idx + m_z] - E_x[idx]) / dz;
    myx = RAHy0[pml_i] * dEzx + RBHy0[pml_i] * Iymzx0[pml_i];
    myz = RAHy0[pml_i] * dExz + RBHy0[pml_i] * Iymxz0[pml_i];
    H_y[idx] = H_y[idx] + matH[IDX2D_MAT(mat_hy, 4)] * myx
                         - matH[IDX2D_MAT(mat_hy, 4)] * myz;
    Iymzx0[pml_i] = Iymzx0[pml_i] - RCHy0[pml_i] * myx + RDHy0[pml_i] * dEzx;
    Iymxz0[pml_i] = Iymxz0[pml_i] - RCHy0[pml_i] * myz + RDHy0[pml_i] * dExz;

    // H_z PML start
    dEyx = (E_y[idx + m_x] - E_y[idx]) / dx;
    dExy = (E_x[idx + m_y] - E_x[idx]) / dy;
    mzx = RAHz0[pml_i] * dEyx + RBHz0[pml_i] * Izmyx0[pml_i];
    mzy = RAHz0[pml_i] * dExy + RBHz0[pml_i] * Izmxy0[pml_i];
    H_z[idx] = H_z[idx] - matH[IDX2D_MAT(mat_hz, 4)] * mzx
                         + matH[IDX2D_MAT(mat_hz, 4)] * mzy;
    Izmyx0[pml_i] = Izmyx0[pml_i] - RCHz0[pml_i] * mzx + RDHz0[pml_i] * dEyx;
    Izmxy0[pml_i] = Izmxy0[pml_i] - RCHz0[pml_i] * mzy + RDHz0[pml_i] * dExy;
    """
    ),
}


# AXIAL ELECTRIC UPDATE KERNELS
# Mirrors the magnetic axial structure exactly but for E fields.
# Three sequential kernel launches:
#   1. update_1d_electric_axial_source      (source grid bulk + PML)
#   2. update_1d_electric_axial_inject      (single point injection)
#   3. update_1d_electric_axial_main        (main grid bulk + two PML regions)
#
# Key differences from magnetic axial:
#   - Uses matE instead of matH
#   - GID components 0,1,2 (Ex,Ey,Ez) instead of 3,4,5 (Hx,Hy,Hz)
#   - Injection at src-1 instead of src-2
#   - Electric PML start region: idx = (P - pml_i) not (P - pml_i) - 1
#   - E update curl formula uses H neighbors (H[j] - H[j-m]) not (H[j+m] - H[j])


update_1d_electric_axial_source = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_axial_source(
            const int N,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            $REAL* __restrict__ E_x_s,
            $REAL* __restrict__ E_y_s,
            $REAL* __restrict__ E_z_s,
            const $REAL* __restrict__ H_x_s,
            const $REAL* __restrict__ H_y_s,
            const $REAL* __restrict__ H_z_s,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        __global $REAL* restrict E_x_s,
        __global $REAL* restrict E_y_s,
        __global $REAL* restrict E_z_s,
        __global const $REAL* restrict H_x_s,
        __global const $REAL* restrict H_y_s,
        __global const $REAL* restrict H_z_s,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_axial_source(
            device const int& N,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device $REAL* E_x_s,
            device $REAL* E_y_s,
            device $REAL* E_z_s,
            device const $REAL* H_x_s,
            device const $REAL* H_y_s,
            device const $REAL* H_z_s,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW -source grid electric field update (Kernel 1 of 3).
    //
    // Updates E_x_s, E_y_s, E_z_s for source 1D grid.
    // Fixed material at GID[0,2], GID[1,2], GID[2,2] for Ex, Ey, Ez.
    // Reference: Equation 9 of Tan & Potter (2010).

    $CUDA_IDX

    int j = i + M;
    if (j >= N - M) return;

    int mat_ex = GID[0 * N + 2];   // GID[0, 2]
    int mat_ey = GID[1 * N + 2];   // GID[1, 2]
    int mat_ez = GID[2 * N + 2];   // GID[2, 2]

    // E_x_s update: curl of H_s
    E_x_s[j] = matE[IDX2D_MAT(mat_ex, 0)] * E_x_s[j]
              + matE[IDX2D_MAT(mat_ex, 2)] * (H_z_s[j] - H_z_s[j - m_y])
              - matE[IDX2D_MAT(mat_ex, 3)] * (H_y_s[j] - H_y_s[j - m_z]);

    // E_y_s update
    E_y_s[j] = matE[IDX2D_MAT(mat_ey, 0)] * E_y_s[j]
              + matE[IDX2D_MAT(mat_ey, 3)] * (H_x_s[j] - H_x_s[j - m_z])
              - matE[IDX2D_MAT(mat_ey, 1)] * (H_z_s[j] - H_z_s[j - m_x]);

    // E_z_s update
    E_z_s[j] = matE[IDX2D_MAT(mat_ez, 0)] * E_z_s[j]
              + matE[IDX2D_MAT(mat_ez, 1)] * (H_y_s[j] - H_y_s[j - m_x])
              - matE[IDX2D_MAT(mat_ez, 2)] * (H_x_s[j] - H_x_s[j - m_y]);
    """
    ),
}


update_1d_electric_axial_source_pml = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_axial_source_pml(
            const int N,
            const int P,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL dx,
            const $REAL dy,
            const $REAL dz,
            $REAL* __restrict__ E_x_s,
            $REAL* __restrict__ E_y_s,
            $REAL* __restrict__ E_z_s,
            const $REAL* __restrict__ H_x_s,
            const $REAL* __restrict__ H_y_s,
            const $REAL* __restrict__ H_z_s,
            $REAL* __restrict__ Ixjzy_s,
            $REAL* __restrict__ Ixjyz_s,
            $REAL* __restrict__ Iyjzx_s,
            $REAL* __restrict__ Iyjxz_s,
            $REAL* __restrict__ Izjyx_s,
            $REAL* __restrict__ Izjxy_s,
            const $REAL* __restrict__ RAEx0,
            const $REAL* __restrict__ RBEx0,
            const $REAL* __restrict__ RCEx0,
            const $REAL* __restrict__ RDEx0,
            const $REAL* __restrict__ RAEy0,
            const $REAL* __restrict__ RBEy0,
            const $REAL* __restrict__ RCEy0,
            const $REAL* __restrict__ RDEy0,
            const $REAL* __restrict__ RAEz0,
            const $REAL* __restrict__ RBEz0,
            const $REAL* __restrict__ RCEz0,
            const $REAL* __restrict__ RDEz0,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int P,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL dx,
        const $REAL dy,
        const $REAL dz,
        __global $REAL* restrict E_x_s,
        __global $REAL* restrict E_y_s,
        __global $REAL* restrict E_z_s,
        __global const $REAL* restrict H_x_s,
        __global const $REAL* restrict H_y_s,
        __global const $REAL* restrict H_z_s,
        __global $REAL* restrict Ixjzy_s,
        __global $REAL* restrict Ixjyz_s,
        __global $REAL* restrict Iyjzx_s,
        __global $REAL* restrict Iyjxz_s,
        __global $REAL* restrict Izjyx_s,
        __global $REAL* restrict Izjxy_s,
        __global const $REAL* restrict RAEx0,
        __global const $REAL* restrict RBEx0,
        __global const $REAL* restrict RCEx0,
        __global const $REAL* restrict RDEx0,
        __global const $REAL* restrict RAEy0,
        __global const $REAL* restrict RBEy0,
        __global const $REAL* restrict RCEy0,
        __global const $REAL* restrict RDEy0,
        __global const $REAL* restrict RAEz0,
        __global const $REAL* restrict RBEz0,
        __global const $REAL* restrict RCEz0,
        __global const $REAL* restrict RDEz0,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_axial_source_pml(
            device const int& N,
            device const int& P,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device $REAL* E_x_s,
            device $REAL* E_y_s,
            device $REAL* E_z_s,
            device const $REAL* H_x_s,
            device const $REAL* H_y_s,
            device const $REAL* H_z_s,
            device $REAL* Ixjzy_s,
            device $REAL* Ixjyz_s,
            device $REAL* Iyjzx_s,
            device $REAL* Iyjxz_s,
            device $REAL* Izjyx_s,
            device $REAL* Izjxy_s,
            device const $REAL* RAEx0,
            device const $REAL* RBEx0,
            device const $REAL* RCEx0,
            device const $REAL* RDEx0,
            device const $REAL* RAEy0,
            device const $REAL* RBEy0,
            device const $REAL* RCEy0,
            device const $REAL* RDEy0,
            device const $REAL* RAEz0,
            device const $REAL* RBEz0,
            device const $REAL* RCEz0,
            device const $REAL* RDEz0,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - source grid PML electric update.
    // Uses rcEx0/rcEy0/rcEz0 and Ix_s/Iy_s/Iz_s (rows 0,1).
    // Fixed material at GID[0/1/2, 2] for srce coefficient.

    $CUDA_IDX

    if (i >= P) return;

    int pml_i = P - 1 - i;
    int idx = (N - M) - (P - pml_i);

    int mat_ex = GID[0 * N + 2];
    int mat_ey = GID[1 * N + 2];
    int mat_ez = GID[2 * N + 2];

    $REAL dHzy, dHyz, dHzx, dHxz, dHyx, dHxy;
    $REAL jxy, jxz, jyx, jyz, jzx, jzy;

    // E_x_s PML
    dHzy = (H_z_s[idx] - H_z_s[idx - m_y]) / dy;
    dHyz = (H_y_s[idx] - H_y_s[idx - m_z]) / dz;
    jxy = RAEx0[pml_i] * dHzy + RBEx0[pml_i] * Ixjzy_s[pml_i];
    jxz = RAEx0[pml_i] * dHyz + RBEx0[pml_i] * Ixjyz_s[pml_i];
    E_x_s[idx] = E_x_s[idx] + matE[IDX2D_MAT(mat_ex, 4)] * jxy
                             - matE[IDX2D_MAT(mat_ex, 4)] * jxz;
    Ixjzy_s[pml_i] = Ixjzy_s[pml_i] - RCEx0[pml_i] * jxy + RDEx0[pml_i] * dHzy;
    Ixjyz_s[pml_i] = Ixjyz_s[pml_i] - RCEx0[pml_i] * jxz + RDEx0[pml_i] * dHyz;

    // E_y_s PML
    dHzx = (H_z_s[idx] - H_z_s[idx - m_x]) / dx;
    dHxz = (H_x_s[idx] - H_x_s[idx - m_z]) / dz;
    jyx = RAEy0[pml_i] * dHzx + RBEy0[pml_i] * Iyjzx_s[pml_i];
    jyz = RAEy0[pml_i] * dHxz + RBEy0[pml_i] * Iyjxz_s[pml_i];
    E_y_s[idx] = E_y_s[idx] - matE[IDX2D_MAT(mat_ey, 4)] * jyx
                             + matE[IDX2D_MAT(mat_ey, 4)] * jyz;
    Iyjzx_s[pml_i] = Iyjzx_s[pml_i] - RCEy0[pml_i] * jyx + RDEy0[pml_i] * dHzx;
    Iyjxz_s[pml_i] = Iyjxz_s[pml_i] - RCEy0[pml_i] * jyz + RDEy0[pml_i] * dHxz;

    // E_z_s PML
    dHyx = (H_y_s[idx] - H_y_s[idx - m_x]) / dx;
    dHxy = (H_x_s[idx] - H_x_s[idx - m_y]) / dy;
    jzx = RAEz0[pml_i] * dHyx + RBEz0[pml_i] * Izjyx_s[pml_i];
    jzy = RAEz0[pml_i] * dHxy + RBEz0[pml_i] * Izjxy_s[pml_i];
    E_z_s[idx] = E_z_s[idx] + matE[IDX2D_MAT(mat_ez, 4)] * jzx
                             - matE[IDX2D_MAT(mat_ez, 4)] * jzy;
    Izjyx_s[pml_i] = Izjyx_s[pml_i] - RCEz0[pml_i] * jzx + RDEz0[pml_i] * dHyx;
    Izjxy_s[pml_i] = Izjxy_s[pml_i] - RCEz0[pml_i] * jzy + RDEz0[pml_i] * dHxy;
    """
    ),
}


update_1d_electric_axial_inject = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(1, 1) update_1d_electric_axial_inject(
            const int N,
            const int src,
            const int m_x,
            const int m_y,
            const int m_z,
            $REAL* __restrict__ E_x,
            $REAL* __restrict__ E_y,
            $REAL* __restrict__ E_z,
            const $REAL* __restrict__ H_x_s,
            const $REAL* __restrict__ H_y_s,
            const $REAL* __restrict__ H_z_s,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int src,
        const int m_x,
        const int m_y,
        const int m_z,
        __global $REAL* restrict E_x,
        __global $REAL* restrict E_y,
        __global $REAL* restrict E_z,
        __global const $REAL* restrict H_x_s,
        __global const $REAL* restrict H_y_s,
        __global const $REAL* restrict H_z_s,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_axial_inject(
            device const int& N,
            device const int& src,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device $REAL* E_x,
            device $REAL* E_y,
            device $REAL* E_z,
            device const $REAL* H_x_s,
            device const $REAL* H_y_s,
            device const $REAL* H_z_s,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - source injection into main E grid (Kernel 2 of 3).
    //
    // Injects source grid H fields into main grid E fields at src-1.
    // Single thread kernel - launched with block=(1,1,1), grid=(1,1,1).
    //
    // Ports:
    //   E_x[src-1] -= matE[GID[0,src-1],2]*H_z_s[src-1-m_y]
    //               - matE[GID[0,src-1],3]*H_y_s[src-1-m_z]
    //   E_y[src-1] -= matE[GID[1,src-1],3]*H_x_s[src-1-m_z]
    //               - matE[GID[1,src-1],1]*H_z_s[src-1-m_x]
    //   E_z[src-1] -= matE[GID[2,src-1],1]*H_y_s[src-1-m_x]
    //               - matE[GID[2,src-1],2]*H_x_s[src-1-m_y]

    int pos = src - 1;

    int mat_ex = GID[0 * N + pos];
    int mat_ey = GID[1 * N + pos];
    int mat_ez = GID[2 * N + pos];

    E_x[pos] -= matE[IDX2D_MAT(mat_ex, 2)] * H_z_s[pos - m_y]
              - matE[IDX2D_MAT(mat_ex, 3)] * H_y_s[pos - m_z];

    E_y[pos] -= matE[IDX2D_MAT(mat_ey, 3)] * H_x_s[pos - m_z]
              - matE[IDX2D_MAT(mat_ey, 1)] * H_z_s[pos - m_x];

    E_z[pos] -= matE[IDX2D_MAT(mat_ez, 1)] * H_y_s[pos - m_x]
              - matE[IDX2D_MAT(mat_ez, 2)] * H_x_s[pos - m_y];
    """
    ),
}


update_1d_electric_axial_main = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_axial_main(
            const int N,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            $REAL* __restrict__ E_x,
            $REAL* __restrict__ E_y,
            $REAL* __restrict__ E_z,
            const $REAL* __restrict__ H_x,
            const $REAL* __restrict__ H_y,
            const $REAL* __restrict__ H_z,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        __global $REAL* restrict E_x,
        __global $REAL* restrict E_y,
        __global $REAL* restrict E_z,
        __global const $REAL* restrict H_x,
        __global const $REAL* restrict H_y,
        __global const $REAL* restrict H_z,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_axial_main(
            device const int& N,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device $REAL* E_x,
            device $REAL* E_y,
            device $REAL* E_z,
            device const $REAL* H_x,
            device const $REAL* H_y,
            device const $REAL* H_z,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - main grid electric field update (Kernel 3 of 3).
    //
    // Per-cell material lookup: matE[GID[component, j], col]
    // Range: j in [M, N-M).
    // Must run after injection kernel.

    $CUDA_IDX

    int j = i + M;
    if (j >= N - M) return;

    int mat_ex = GID[0 * N + j];
    int mat_ey = GID[1 * N + j];
    int mat_ez = GID[2 * N + j];

    E_x[j] = matE[IDX2D_MAT(mat_ex, 0)] * E_x[j]
            + matE[IDX2D_MAT(mat_ex, 2)] * (H_z[j] - H_z[j - m_y])
            - matE[IDX2D_MAT(mat_ex, 3)] * (H_y[j] - H_y[j - m_z]);

    E_y[j] = matE[IDX2D_MAT(mat_ey, 0)] * E_y[j]
            + matE[IDX2D_MAT(mat_ey, 3)] * (H_x[j] - H_x[j - m_z])
            - matE[IDX2D_MAT(mat_ey, 1)] * (H_z[j] - H_z[j - m_x]);

    E_z[j] = matE[IDX2D_MAT(mat_ez, 0)] * E_z[j]
            + matE[IDX2D_MAT(mat_ez, 1)] * (H_y[j] - H_y[j - m_x])
            - matE[IDX2D_MAT(mat_ez, 2)] * (H_x[j] - H_x[j - m_y]);
    """
    ),
}


update_1d_electric_axial_main_pml_end = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_axial_main_pml_end(
            const int N,
            const int P,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL dx,
            const $REAL dy,
            const $REAL dz,
            $REAL* __restrict__ E_x,
            $REAL* __restrict__ E_y,
            $REAL* __restrict__ E_z,
            const $REAL* __restrict__ H_x,
            const $REAL* __restrict__ H_y,
            const $REAL* __restrict__ H_z,
            $REAL* __restrict__ Ixjzy,
            $REAL* __restrict__ Ixjyz,
            $REAL* __restrict__ Iyjzx,
            $REAL* __restrict__ Iyjxz,
            $REAL* __restrict__ Izjyx,
            $REAL* __restrict__ Izjxy,
            const $REAL* __restrict__ RAEx,
            const $REAL* __restrict__ RBEx,
            const $REAL* __restrict__ RCEx,
            const $REAL* __restrict__ RDEx,
            const $REAL* __restrict__ RAEy,
            const $REAL* __restrict__ RBEy,
            const $REAL* __restrict__ RCEy,
            const $REAL* __restrict__ RDEy,
            const $REAL* __restrict__ RAEz,
            const $REAL* __restrict__ RBEz,
            const $REAL* __restrict__ RCEz,
            const $REAL* __restrict__ RDEz,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int P,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL dx,
        const $REAL dy,
        const $REAL dz,
        __global $REAL* restrict E_x,
        __global $REAL* restrict E_y,
        __global $REAL* restrict E_z,
        __global const $REAL* restrict H_x,
        __global const $REAL* restrict H_y,
        __global const $REAL* restrict H_z,
        __global $REAL* restrict Ixjzy,
        __global $REAL* restrict Ixjyz,
        __global $REAL* restrict Iyjzx,
        __global $REAL* restrict Iyjxz,
        __global $REAL* restrict Izjyx,
        __global $REAL* restrict Izjxy,
        __global const $REAL* restrict RAEx,
        __global const $REAL* restrict RBEx,
        __global const $REAL* restrict RCEx,
        __global const $REAL* restrict RDEx,
        __global const $REAL* restrict RAEy,
        __global const $REAL* restrict RBEy,
        __global const $REAL* restrict RCEy,
        __global const $REAL* restrict RDEy,
        __global const $REAL* restrict RAEz,
        __global const $REAL* restrict RBEz,
        __global const $REAL* restrict RCEz,
        __global const $REAL* restrict RDEz,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_axial_main_pml_end(
            device const int& N,
            device const int& P,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device $REAL* E_x,
            device $REAL* E_y,
            device $REAL* E_z,
            device const $REAL* H_x,
            device const $REAL* H_y,
            device const $REAL* H_z,
            device $REAL* Ixjzy,
            device $REAL* Ixjyz,
            device $REAL* Iyjzx,
            device $REAL* Iyjxz,
            device $REAL* Izjyx,
            device $REAL* Izjxy,
            device const $REAL* RAEx,
            device const $REAL* RBEx,
            device const $REAL* RCEx,
            device const $REAL* RDEx,
            device const $REAL* RAEy,
            device const $REAL* RBEy,
            device const $REAL* RCEy,
            device const $REAL* RDEy,
            device const $REAL* RAEz,
            device const $REAL* RBEz,
            device const $REAL* RCEz,
            device const $REAL* RDEz,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - main grid PML end region electric update.
    // Per-cell srce: matE[GID[component, idx], 4]
    // idx = (N-M) - (P - pml_i)

    $CUDA_IDX

    if (i >= P) return;

    int pml_i = P - 1 - i;
    int idx = (N - M) - (P - pml_i);

    int mat_ex = GID[0 * N + idx];
    int mat_ey = GID[1 * N + idx];
    int mat_ez = GID[2 * N + idx];

    $REAL dHzy, dHyz, dHzx, dHxz, dHyx, dHxy;
    $REAL jxy, jxz, jyx, jyz, jzx, jzy;

    // E_x PML end
    dHzy = (H_z[idx] - H_z[idx - m_y]) / dy;
    dHyz = (H_y[idx] - H_y[idx - m_z]) / dz;
    jxy = RAEx[pml_i] * dHzy + RBEx[pml_i] * Ixjzy[pml_i];
    jxz = RAEx[pml_i] * dHyz + RBEx[pml_i] * Ixjyz[pml_i];
    E_x[idx] = E_x[idx] + matE[IDX2D_MAT(mat_ex, 4)] * jxy
                         - matE[IDX2D_MAT(mat_ex, 4)] * jxz;
    Ixjzy[pml_i] = Ixjzy[pml_i] - RCEx[pml_i] * jxy + RDEx[pml_i] * dHzy;
    Ixjyz[pml_i] = Ixjyz[pml_i] - RCEx[pml_i] * jxz + RDEx[pml_i] * dHyz;

    // E_y PML end
    dHzx = (H_z[idx] - H_z[idx - m_x]) / dx;
    dHxz = (H_x[idx] - H_x[idx - m_z]) / dz;
    jyx = RAEy[pml_i] * dHzx + RBEy[pml_i] * Iyjzx[pml_i];
    jyz = RAEy[pml_i] * dHxz + RBEy[pml_i] * Iyjxz[pml_i];
    E_y[idx] = E_y[idx] - matE[IDX2D_MAT(mat_ey, 4)] * jyx
                         + matE[IDX2D_MAT(mat_ey, 4)] * jyz;
    Iyjzx[pml_i] = Iyjzx[pml_i] - RCEy[pml_i] * jyx + RDEy[pml_i] * dHzx;
    Iyjxz[pml_i] = Iyjxz[pml_i] - RCEy[pml_i] * jyz + RDEy[pml_i] * dHxz;

    // E_z PML end
    dHyx = (H_y[idx] - H_y[idx - m_x]) / dx;
    dHxy = (H_x[idx] - H_x[idx - m_y]) / dy;
    jzx = RAEz[pml_i] * dHyx + RBEz[pml_i] * Izjyx[pml_i];
    jzy = RAEz[pml_i] * dHxy + RBEz[pml_i] * Izjxy[pml_i];
    E_z[idx] = E_z[idx] + matE[IDX2D_MAT(mat_ez, 4)] * jzx
                         - matE[IDX2D_MAT(mat_ez, 4)] * jzy;
    Izjyx[pml_i] = Izjyx[pml_i] - RCEz[pml_i] * jzx + RDEz[pml_i] * dHyx;
    Izjxy[pml_i] = Izjxy[pml_i] - RCEz[pml_i] * jzy + RDEz[pml_i] * dHxy;
    """
    ),
}


update_1d_electric_axial_main_pml_start = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_axial_main_pml_start(
            const int N,
            const int P,
            const int m_x,
            const int m_y,
            const int m_z,
            const $REAL dx,
            const $REAL dy,
            const $REAL dz,
            $REAL* __restrict__ E_x,
            $REAL* __restrict__ E_y,
            $REAL* __restrict__ E_z,
            const $REAL* __restrict__ H_x,
            const $REAL* __restrict__ H_y,
            const $REAL* __restrict__ H_z,
            $REAL* __restrict__ Ixjzy0,
            $REAL* __restrict__ Ixjyz0,
            $REAL* __restrict__ Iyjzx0,
            $REAL* __restrict__ Iyjxz0,
            $REAL* __restrict__ Izjyx0,
            $REAL* __restrict__ Izjxy0,
            const $REAL* __restrict__ RAEx0,
            const $REAL* __restrict__ RBEx0,
            const $REAL* __restrict__ RCEx0,
            const $REAL* __restrict__ RDEx0,
            const $REAL* __restrict__ RAEy0,
            const $REAL* __restrict__ RBEy0,
            const $REAL* __restrict__ RCEy0,
            const $REAL* __restrict__ RDEy0,
            const $REAL* __restrict__ RAEz0,
            const $REAL* __restrict__ RBEz0,
            const $REAL* __restrict__ RCEz0,
            const $REAL* __restrict__ RDEz0,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matH,
            const $REAL* __restrict__ matE
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int P,
        const int m_x,
        const int m_y,
        const int m_z,
        const $REAL dx,
        const $REAL dy,
        const $REAL dz,
        __global $REAL* restrict E_x,
        __global $REAL* restrict E_y,
        __global $REAL* restrict E_z,
        __global const $REAL* restrict H_x,
        __global const $REAL* restrict H_y,
        __global const $REAL* restrict H_z,
        __global $REAL* restrict Ixjzy0,
        __global $REAL* restrict Ixjyz0,
        __global $REAL* restrict Iyjzx0,
        __global $REAL* restrict Iyjxz0,
        __global $REAL* restrict Izjyx0,
        __global $REAL* restrict Izjxy0,
        __global const $REAL* restrict RAEx0,
        __global const $REAL* restrict RBEx0,
        __global const $REAL* restrict RCEx0,
        __global const $REAL* restrict RDEx0,
        __global const $REAL* restrict RAEy0,
        __global const $REAL* restrict RBEy0,
        __global const $REAL* restrict RCEy0,
        __global const $REAL* restrict RDEy0,
        __global const $REAL* restrict RAEz0,
        __global const $REAL* restrict RBEz0,
        __global const $REAL* restrict RCEz0,
        __global const $REAL* restrict RDEz0,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matH,
        __global const $REAL* restrict matE
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_axial_main_pml_start(
            device const int& N,
            device const int& P,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device $REAL* E_x,
            device $REAL* E_y,
            device $REAL* E_z,
            device const $REAL* H_x,
            device const $REAL* H_y,
            device const $REAL* H_z,
            device $REAL* Ixjzy0,
            device $REAL* Ixjyz0,
            device $REAL* Iyjzx0,
            device $REAL* Iyjxz0,
            device $REAL* Izjyx0,
            device $REAL* Izjxy0,
            device const $REAL* RAEx0,
            device const $REAL* RBEx0,
            device const $REAL* RCEx0,
            device const $REAL* RDEx0,
            device const $REAL* RAEy0,
            device const $REAL* RBEy0,
            device const $REAL* RCEy0,
            device const $REAL* RDEy0,
            device const $REAL* RAEz0,
            device const $REAL* RBEz0,
            device const $REAL* RCEz0,
            device const $REAL* RDEz0,
            device const unsigned int* GID,
            device const $REAL* matH,
            device const $REAL* matE,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Axial 1D DPW - main grid PML start region electric update.
    // Uses rcEx0/rcEy0/rcEz0 and Ix0/Iy0/Iz0.
    // Per-cell srce: matE[GID[component, idx], 4]
    // IMPORTANT: idx = (P - pml_i)  - NOT (P - pml_i) - 1
    // This matches the Cython: idx = (p - i) for the electric start region.

    $CUDA_IDX

    if (i >= P) return;

    int pml_i = P - 1 - i;
    int idx = P - pml_i;   // electric start: idx = (p - i) in Cython

    int mat_ex = GID[0 * N + idx];
    int mat_ey = GID[1 * N + idx];
    int mat_ez = GID[2 * N + idx];

    $REAL dHzy, dHyz, dHzx, dHxz, dHyx, dHxy;
    $REAL jxy, jxz, jyx, jyz, jzx, jzy;

    // E_x PML start
    dHzy = (H_z[idx] - H_z[idx - m_y]) / dy;
    dHyz = (H_y[idx] - H_y[idx - m_z]) / dz;
    jxy = RAEx0[pml_i] * dHzy + RBEx0[pml_i] * Ixjzy0[pml_i];
    jxz = RAEx0[pml_i] * dHyz + RBEx0[pml_i] * Ixjyz0[pml_i];
    E_x[idx] = E_x[idx] + matE[IDX2D_MAT(mat_ex, 4)] * jxy
                         - matE[IDX2D_MAT(mat_ex, 4)] * jxz;
    Ixjzy0[pml_i] = Ixjzy0[pml_i] - RCEx0[pml_i] * jxy + RDEx0[pml_i] * dHzy;
    Ixjyz0[pml_i] = Ixjyz0[pml_i] - RCEx0[pml_i] * jxz + RDEx0[pml_i] * dHyz;

    // E_y PML start
    dHzx = (H_z[idx] - H_z[idx - m_x]) / dx;
    dHxz = (H_x[idx] - H_x[idx - m_z]) / dz;
    jyx = RAEy0[pml_i] * dHzx + RBEy0[pml_i] * Iyjzx0[pml_i];
    jyz = RAEy0[pml_i] * dHxz + RBEy0[pml_i] * Iyjxz0[pml_i];
    E_y[idx] = E_y[idx] - matE[IDX2D_MAT(mat_ey, 4)] * jyx
                         + matE[IDX2D_MAT(mat_ey, 4)] * jyz;
    Iyjzx0[pml_i] = Iyjzx0[pml_i] - RCEy0[pml_i] * jyx + RDEy0[pml_i] * dHzx;
    Iyjxz0[pml_i] = Iyjxz0[pml_i] - RCEy0[pml_i] * jyz + RDEy0[pml_i] * dHxz;

    // E_z PML start
    dHyx = (H_y[idx] - H_y[idx - m_x]) / dx;
    dHxy = (H_x[idx] - H_x[idx - m_y]) / dy;
    jzx = RAEz0[pml_i] * dHyx + RBEz0[pml_i] * Izjyx0[pml_i];
    jzy = RAEz0[pml_i] * dHxy + RBEz0[pml_i] * Izjxy0[pml_i];
    E_z[idx] = E_z[idx] + matE[IDX2D_MAT(mat_ez, 4)] * jzx
                         - matE[IDX2D_MAT(mat_ez, 4)] * jzy;
    Izjyx0[pml_i] = Izjyx0[pml_i] - RCEz0[pml_i] * jzx + RDEz0[pml_i] * dHyx;
    Izjxy0[pml_i] = Izjxy0[pml_i] - RCEz0[pml_i] * jzy + RDEz0[pml_i] * dHxy;
    """
    ),
}


# DISPERSIVE STANDARD ELECTRIC UPDATE (bulk) - part A
#
# Ports bulk loop of updateElectricFields_dispersive() from plane_wave.pyx.
# Recursive convolution polarization terms T_x/T_y/T_z shape [num_poles, N]
# (flattened, row stride N). coef_D is the row of updatecoeffsdispersive for
# the (single) source material - length 3*num_poles.
# PML region is identical to the non-dispersive case - reuse
# update_1d_electric_pml. The final T subtraction is in part B and must be
# launched AFTER the PML kernel (matches Cython ordering).

update_1d_electric_dispersive_A = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_dispersive_A(
            const int N,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const int num_poles,
            const $REAL coef_E_t,
            const $REAL coef_E_x,
            const $REAL coef_E_y,
            const $REAL coef_E_z,
            const $REAL coef_E_D,
            const $COMPLEX* __restrict__ coef_D,
            $COMPLEX* __restrict__ T_x,
            $COMPLEX* __restrict__ T_y,
            $COMPLEX* __restrict__ T_z,
            $REAL* __restrict__ E_x,
            $REAL* __restrict__ E_y,
            $REAL* __restrict__ E_z,
            const $REAL* __restrict__ H_x,
            const $REAL* __restrict__ H_y,
            const $REAL* __restrict__ H_z
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const int num_poles,
        const $REAL coef_E_t,
        const $REAL coef_E_x,
        const $REAL coef_E_y,
        const $REAL coef_E_z,
        const $REAL coef_E_D,
        __global const $COMPLEX* restrict coef_D,
        __global $COMPLEX* restrict T_x,
        __global $COMPLEX* restrict T_y,
        __global $COMPLEX* restrict T_z,
        __global $REAL* restrict E_x,
        __global $REAL* restrict E_y,
        __global $REAL* restrict E_z,
        __global const $REAL* restrict H_x,
        __global const $REAL* restrict H_y,
        __global const $REAL* restrict H_z
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_dispersive_A(
            device const int& N,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const int& num_poles,
            device const $REAL& coef_E_t,
            device const $REAL& coef_E_x,
            device const $REAL& coef_E_y,
            device const $REAL& coef_E_z,
            device const $REAL& coef_E_D,
            device const $COMPLEX* coef_D,
            device $COMPLEX* T_x,
            device $COMPLEX* T_y,
            device $COMPLEX* T_z,
            device $REAL* E_x,
            device $REAL* E_y,
            device $REAL* E_z,
            device const $REAL* H_x,
            device const $REAL* H_y,
            device const $REAL* H_z,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Dispersive standard 1D DPW electric bulk update.
    // Equation 9 of Tan & Potter modified for dispersive materials.
    // coef_E_t = updatecoeffsE[0] (CA), coef_E_x/y/z = updatecoeffsE[1/2/3],
    // coef_E_D = updatecoeffsE[4].
    // T arrays flattened [num_poles, N]: element (pole, j) at T[pole*N + j].
    // coef_D layout per pole: [phi-coef, T-decay, T-source] = coef_D[3*pole + 0/1/2]

    $CUDA_IDX

    int j = i + M;
    if (j >= N - M) return;

    $REAL phi;
    int pole;

    phi = 0;
    for (pole = 0; pole < num_poles; pole++) {
        phi += GPRMAX_CREAL(GPRMAX_CMUL(coef_D[pole * 3], T_x[pole * N + j]));
        T_x[pole * N + j] = GPRMAX_CADD(GPRMAX_CMUL(coef_D[1 + pole * 3], T_x[pole * N + j]),
                            GPRMAX_CRMUL(coef_D[2 + pole * 3], E_x[j]));
    }
    E_x[j] = coef_E_t * E_x[j]
            + coef_E_y * (H_z[j] - H_z[j - m_y])
            - coef_E_z * (H_y[j] - H_y[j - m_z])
            - coef_E_D * phi;

    phi = 0;
    for (pole = 0; pole < num_poles; pole++) {
        phi += GPRMAX_CREAL(GPRMAX_CMUL(coef_D[pole * 3], T_y[pole * N + j]));
        T_y[pole * N + j] = GPRMAX_CADD(GPRMAX_CMUL(coef_D[1 + pole * 3], T_y[pole * N + j]),
                            GPRMAX_CRMUL(coef_D[2 + pole * 3], E_y[j]));
    }
    E_y[j] = coef_E_t * E_y[j]
            + coef_E_z * (H_x[j] - H_x[j - m_z])
            - coef_E_x * (H_z[j] - H_z[j - m_x])
            - coef_E_D * phi;

    phi = 0;
    for (pole = 0; pole < num_poles; pole++) {
        phi += GPRMAX_CREAL(GPRMAX_CMUL(coef_D[pole * 3], T_z[pole * N + j]));
        T_z[pole * N + j] = GPRMAX_CADD(GPRMAX_CMUL(coef_D[1 + pole * 3], T_z[pole * N + j]),
                            GPRMAX_CRMUL(coef_D[2 + pole * 3], E_z[j]));
    }
    E_z[j] = coef_E_t * E_z[j]
            + coef_E_x * (H_y[j] - H_y[j - m_x])
            - coef_E_y * (H_x[j] - H_x[j - m_y])
            - coef_E_D * phi;
    """
    ),
}


# DISPERSIVE STANDARD ELECTRIC UPDATE (final T subtraction) - part B
#
# Ports final loop of updateElectricFields_dispersive() from plane_wave.pyx.
# Must run after the PML kernel (uses post-PML E values).

update_1d_electric_dispersive_B = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_dispersive_B(
            const int N,
            const int M,
            const int num_poles,
            const $COMPLEX* __restrict__ coef_D,
            $COMPLEX* __restrict__ T_x,
            $COMPLEX* __restrict__ T_y,
            $COMPLEX* __restrict__ T_z,
            const $REAL* __restrict__ E_x,
            const $REAL* __restrict__ E_y,
            const $REAL* __restrict__ E_z
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int num_poles,
        __global const $COMPLEX* restrict coef_D,
        __global $COMPLEX* restrict T_x,
        __global $COMPLEX* restrict T_y,
        __global $COMPLEX* restrict T_z,
        __global const $REAL* restrict E_x,
        __global const $REAL* restrict E_y,
        __global const $REAL* restrict E_z
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_dispersive_B(
            device const int& N,
            device const int& M,
            device const int& num_poles,
            device const $COMPLEX* coef_D,
            device $COMPLEX* T_x,
            device $COMPLEX* T_y,
            device $COMPLEX* T_z,
            device const $REAL* E_x,
            device const $REAL* E_y,
            device const $REAL* E_z,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Dispersive standard 1D DPW - final polarization term correction.
    // T[pole, j] -= coef_D[2 + 3*pole] * E[j] using post-PML E values.

    $CUDA_IDX

    int j = i + M;
    if (j >= N - M) return;

    int pole;
    for (pole = 0; pole < num_poles; pole++) {
        T_x[pole * N + j] = GPRMAX_CSUB(T_x[pole * N + j], GPRMAX_CRMUL(coef_D[2 + pole * 3], E_x[j]));
        T_y[pole * N + j] = GPRMAX_CSUB(T_y[pole * N + j], GPRMAX_CRMUL(coef_D[2 + pole * 3], E_y[j]));
        T_z[pole * N + j] = GPRMAX_CSUB(T_z[pole * N + j], GPRMAX_CRMUL(coef_D[2 + pole * 3], E_z[j]));
    }
    """
    ),
}


# DISPERSIVE AXIAL ELECTRIC UPDATE - source grid bulk
#
# Ports source-grid bulk loop of updateElectricFields_dispersive_axial()
# from plane_wave.pyx. Source grid runs in the source material whose ID is
# taken at fixed position GID[component, 2] (per Cython reference).
# matD is the full updatecoeffsdispersive array [nmats, 3*num_poles].

update_1d_electric_dispersive_axial_source = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_dispersive_axial_source(
            const int N,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const int num_poles,
            $COMPLEX* __restrict__ T_x_s,
            $COMPLEX* __restrict__ T_y_s,
            $COMPLEX* __restrict__ T_z_s,
            $REAL* __restrict__ E_x_s,
            $REAL* __restrict__ E_y_s,
            $REAL* __restrict__ E_z_s,
            const $REAL* __restrict__ H_x_s,
            const $REAL* __restrict__ H_y_s,
            const $REAL* __restrict__ H_z_s,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matE,
            const $COMPLEX* __restrict__ matD
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const int num_poles,
        __global $COMPLEX* restrict T_x_s,
        __global $COMPLEX* restrict T_y_s,
        __global $COMPLEX* restrict T_z_s,
        __global $REAL* restrict E_x_s,
        __global $REAL* restrict E_y_s,
        __global $REAL* restrict E_z_s,
        __global const $REAL* restrict H_x_s,
        __global const $REAL* restrict H_y_s,
        __global const $REAL* restrict H_z_s,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matE,
        __global const $COMPLEX* restrict matD
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_dispersive_axial_source(
            device const int& N,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const int& num_poles,
            device $COMPLEX* T_x_s,
            device $COMPLEX* T_y_s,
            device $COMPLEX* T_z_s,
            device $REAL* E_x_s,
            device $REAL* E_y_s,
            device $REAL* E_z_s,
            device const $REAL* H_x_s,
            device const $REAL* H_y_s,
            device const $REAL* H_z_s,
            device const unsigned int* GID,
            device const $REAL* matE,
            device const $COMPLEX* matD,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Dispersive axial 1D DPW - source grid electric bulk update.
    // Material fixed per component at GID[component, 2] (source material).
    // T arrays flattened [num_poles, N].

    $CUDA_IDX

    int j = i + M;
    if (j >= N - M) return;

    $REAL phi;
    int pole;
    int mat;

    mat = GID[0 * N + 2];
    phi = 0;
    for (pole = 0; pole < num_poles; pole++) {
        phi += GPRMAX_CREAL(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, pole * 3)], T_x_s[pole * N + j]));
        T_x_s[pole * N + j] = GPRMAX_CADD(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, 1 + pole * 3)], T_x_s[pole * N + j]),
                            GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat, 2 + pole * 3)], E_x_s[j]));
    }
    E_x_s[j] = matE[IDX2D_MAT(mat, 0)] * E_x_s[j]
              + matE[IDX2D_MAT(mat, 2)] * (H_z_s[j] - H_z_s[j - m_y])
              - matE[IDX2D_MAT(mat, 3)] * (H_y_s[j] - H_y_s[j - m_z])
              - matE[IDX2D_MAT(mat, 4)] * phi;

    mat = GID[1 * N + 2];
    phi = 0;
    for (pole = 0; pole < num_poles; pole++) {
        phi += GPRMAX_CREAL(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, pole * 3)], T_y_s[pole * N + j]));
        T_y_s[pole * N + j] = GPRMAX_CADD(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, 1 + pole * 3)], T_y_s[pole * N + j]),
                            GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat, 2 + pole * 3)], E_y_s[j]));
    }
    E_y_s[j] = matE[IDX2D_MAT(mat, 0)] * E_y_s[j]
              + matE[IDX2D_MAT(mat, 3)] * (H_x_s[j] - H_x_s[j - m_z])
              - matE[IDX2D_MAT(mat, 1)] * (H_z_s[j] - H_z_s[j - m_x])
              - matE[IDX2D_MAT(mat, 4)] * phi;

    mat = GID[2 * N + 2];
    phi = 0;
    for (pole = 0; pole < num_poles; pole++) {
        phi += GPRMAX_CREAL(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, pole * 3)], T_z_s[pole * N + j]));
        T_z_s[pole * N + j] = GPRMAX_CADD(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, 1 + pole * 3)], T_z_s[pole * N + j]),
                            GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat, 2 + pole * 3)], E_z_s[j]));
    }
    E_z_s[j] = matE[IDX2D_MAT(mat, 0)] * E_z_s[j]
              + matE[IDX2D_MAT(mat, 1)] * (H_y_s[j] - H_y_s[j - m_x])
              - matE[IDX2D_MAT(mat, 2)] * (H_x_s[j] - H_x_s[j - m_y])
              - matE[IDX2D_MAT(mat, 4)] * phi;
    """
    ),
}


# DISPERSIVE AXIAL ELECTRIC UPDATE - source grid final T subtraction

update_1d_electric_dispersive_axial_source_T = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_dispersive_axial_source_T(
            const int N,
            const int M,
            const int num_poles,
            $COMPLEX* __restrict__ T_x_s,
            $COMPLEX* __restrict__ T_y_s,
            $COMPLEX* __restrict__ T_z_s,
            const $REAL* __restrict__ E_x_s,
            const $REAL* __restrict__ E_y_s,
            const $REAL* __restrict__ E_z_s,
            const unsigned int* __restrict__ GID,
            const $COMPLEX* __restrict__ matD
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int num_poles,
        __global $COMPLEX* restrict T_x_s,
        __global $COMPLEX* restrict T_y_s,
        __global $COMPLEX* restrict T_z_s,
        __global const $REAL* restrict E_x_s,
        __global const $REAL* restrict E_y_s,
        __global const $REAL* restrict E_z_s,
        __global const unsigned int* restrict GID,
        __global const $COMPLEX* restrict matD
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_dispersive_axial_source_T(
            device const int& N,
            device const int& M,
            device const int& num_poles,
            device $COMPLEX* T_x_s,
            device $COMPLEX* T_y_s,
            device $COMPLEX* T_z_s,
            device const $REAL* E_x_s,
            device const $REAL* E_y_s,
            device const $REAL* E_z_s,
            device const unsigned int* GID,
            device const $COMPLEX* matD,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Dispersive axial 1D DPW - source grid final polarization correction.
    // Runs after the source-grid PML kernel (uses post-PML E values).

    $CUDA_IDX

    int j = i + M;
    if (j >= N - M) return;

    int pole;
    int mat_x = GID[0 * N + 2];
    int mat_y = GID[1 * N + 2];
    int mat_z = GID[2 * N + 2];

    for (pole = 0; pole < num_poles; pole++) {
        T_x_s[pole * N + j] = GPRMAX_CSUB(T_x_s[pole * N + j], GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat_x, 2 + pole * 3)], E_x_s[j]));
        T_y_s[pole * N + j] = GPRMAX_CSUB(T_y_s[pole * N + j], GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat_y, 2 + pole * 3)], E_y_s[j]));
        T_z_s[pole * N + j] = GPRMAX_CSUB(T_z_s[pole * N + j], GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat_z, 2 + pole * 3)], E_z_s[j]));
    }
    """
    ),
}


# DISPERSIVE AXIAL ELECTRIC UPDATE - main grid bulk
#
# Per-cell material lookup GID[component, j] - heterogeneous media.

update_1d_electric_dispersive_axial_main = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_dispersive_axial_main(
            const int N,
            const int M,
            const int m_x,
            const int m_y,
            const int m_z,
            const int num_poles,
            $COMPLEX* __restrict__ T_x,
            $COMPLEX* __restrict__ T_y,
            $COMPLEX* __restrict__ T_z,
            $REAL* __restrict__ E_x,
            $REAL* __restrict__ E_y,
            $REAL* __restrict__ E_z,
            const $REAL* __restrict__ H_x,
            const $REAL* __restrict__ H_y,
            const $REAL* __restrict__ H_z,
            const unsigned int* __restrict__ GID,
            const $REAL* __restrict__ matE,
            const $COMPLEX* __restrict__ matD
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int m_x,
        const int m_y,
        const int m_z,
        const int num_poles,
        __global $COMPLEX* restrict T_x,
        __global $COMPLEX* restrict T_y,
        __global $COMPLEX* restrict T_z,
        __global $REAL* restrict E_x,
        __global $REAL* restrict E_y,
        __global $REAL* restrict E_z,
        __global const $REAL* restrict H_x,
        __global const $REAL* restrict H_y,
        __global const $REAL* restrict H_z,
        __global const unsigned int* restrict GID,
        __global const $REAL* restrict matE,
        __global const $COMPLEX* restrict matD
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_dispersive_axial_main(
            device const int& N,
            device const int& M,
            device const int& m_x,
            device const int& m_y,
            device const int& m_z,
            device const int& num_poles,
            device $COMPLEX* T_x,
            device $COMPLEX* T_y,
            device $COMPLEX* T_z,
            device $REAL* E_x,
            device $REAL* E_y,
            device $REAL* E_z,
            device const $REAL* H_x,
            device const $REAL* H_y,
            device const $REAL* H_z,
            device const unsigned int* GID,
            device const $REAL* matE,
            device const $COMPLEX* matD,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Dispersive axial 1D DPW - main grid electric bulk update.
    // Per-cell material lookup GID[component, j] for layered/heterogeneous media.
    // Must run after the injection kernel.

    $CUDA_IDX

    int j = i + M;
    if (j >= N - M) return;

    $REAL phi;
    int pole;
    int mat;

    mat = GID[0 * N + j];
    phi = 0;
    for (pole = 0; pole < num_poles; pole++) {
        phi += GPRMAX_CREAL(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, pole * 3)], T_x[pole * N + j]));
        T_x[pole * N + j] = GPRMAX_CADD(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, 1 + pole * 3)], T_x[pole * N + j]),
                            GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat, 2 + pole * 3)], E_x[j]));
    }
    E_x[j] = matE[IDX2D_MAT(mat, 0)] * E_x[j]
            + matE[IDX2D_MAT(mat, 2)] * (H_z[j] - H_z[j - m_y])
            - matE[IDX2D_MAT(mat, 3)] * (H_y[j] - H_y[j - m_z])
            - matE[IDX2D_MAT(mat, 4)] * phi;

    mat = GID[1 * N + j];
    phi = 0;
    for (pole = 0; pole < num_poles; pole++) {
        phi += GPRMAX_CREAL(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, pole * 3)], T_y[pole * N + j]));
        T_y[pole * N + j] = GPRMAX_CADD(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, 1 + pole * 3)], T_y[pole * N + j]),
                            GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat, 2 + pole * 3)], E_y[j]));
    }
    E_y[j] = matE[IDX2D_MAT(mat, 0)] * E_y[j]
            + matE[IDX2D_MAT(mat, 3)] * (H_x[j] - H_x[j - m_z])
            - matE[IDX2D_MAT(mat, 1)] * (H_z[j] - H_z[j - m_x])
            - matE[IDX2D_MAT(mat, 4)] * phi;

    mat = GID[2 * N + j];
    phi = 0;
    for (pole = 0; pole < num_poles; pole++) {
        phi += GPRMAX_CREAL(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, pole * 3)], T_z[pole * N + j]));
        T_z[pole * N + j] = GPRMAX_CADD(GPRMAX_CMUL(matD[IDX2D_MATDISP(mat, 1 + pole * 3)], T_z[pole * N + j]),
                            GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat, 2 + pole * 3)], E_z[j]));
    }
    E_z[j] = matE[IDX2D_MAT(mat, 0)] * E_z[j]
            + matE[IDX2D_MAT(mat, 1)] * (H_y[j] - H_y[j - m_x])
            - matE[IDX2D_MAT(mat, 2)] * (H_x[j] - H_x[j - m_y])
            - matE[IDX2D_MAT(mat, 4)] * phi;
    """
    ),
}


# DISPERSIVE AXIAL ELECTRIC UPDATE - main grid final T subtraction

update_1d_electric_dispersive_axial_main_T = {
    "args_cuda": Template(
        """
        __global__ void __launch_bounds__(256, 4) update_1d_electric_dispersive_axial_main_T(
            const int N,
            const int M,
            const int num_poles,
            $COMPLEX* __restrict__ T_x,
            $COMPLEX* __restrict__ T_y,
            $COMPLEX* __restrict__ T_z,
            const $REAL* __restrict__ E_x,
            const $REAL* __restrict__ E_y,
            const $REAL* __restrict__ E_z,
            const unsigned int* __restrict__ GID,
            const $COMPLEX* __restrict__ matD
        )
        """
    ),
    "args_opencl": Template(
        """
        const int N,
        const int M,
        const int num_poles,
        __global $COMPLEX* restrict T_x,
        __global $COMPLEX* restrict T_y,
        __global $COMPLEX* restrict T_z,
        __global const $REAL* restrict E_x,
        __global const $REAL* restrict E_y,
        __global const $REAL* restrict E_z,
        __global const unsigned int* restrict GID,
        __global const $COMPLEX* restrict matD
        """
    ),
    "args_metal": Template(
        """
        kernel void update_1d_electric_dispersive_axial_main_T(
            device const int& N,
            device const int& M,
            device const int& num_poles,
            device $COMPLEX* T_x,
            device $COMPLEX* T_y,
            device $COMPLEX* T_z,
            device const $REAL* E_x,
            device const $REAL* E_y,
            device const $REAL* E_z,
            device const unsigned int* GID,
            device const $COMPLEX* matD,
            uint i [[thread_position_in_grid]]
        )
        """
    ),
    "func": Template(
        """
    // Dispersive axial 1D DPW - main grid final polarization correction.
    // Per-cell material lookup. Runs after both main-grid PML kernels
    // (uses post-PML E values).

    $CUDA_IDX

    int j = i + M;
    if (j >= N - M) return;

    int pole;
    int mat_x = GID[0 * N + j];
    int mat_y = GID[1 * N + j];
    int mat_z = GID[2 * N + j];

    for (pole = 0; pole < num_poles; pole++) {
        T_x[pole * N + j] = GPRMAX_CSUB(T_x[pole * N + j], GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat_x, 2 + pole * 3)], E_x[j]));
        T_y[pole * N + j] = GPRMAX_CSUB(T_y[pole * N + j], GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat_y, 2 + pole * 3)], E_y[j]));
        T_z[pole * N + j] = GPRMAX_CSUB(T_z[pole * N + j], GPRMAX_CRMUL(matD[IDX2D_MATDISP(mat_z, 2 + pole * 3)], E_z[j]));
    }
    """
    ),
}
