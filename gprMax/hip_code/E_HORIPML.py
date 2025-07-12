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

x_args = {
    "hip": Template(
        """
                    __global__ void $FUNC(int xs,
                                        int xf,
                                        int ys,
                                        int yf,
                                        int zs,
                                        int zf,
                                        int NX_PHI1,
                                        int NY_PHI1,
                                        int NZ_PHI1,
                                        int NX_PHI2,
                                        int NY_PHI2,
                                        int NZ_PHI2,
                                        int NY_R,
                                        const unsigned int* __restrict__ ID,
                                        const $REAL* __restrict__ Ex,
                                        $REAL *Ey,
                                        $REAL *Ez,
                                        const $REAL* __restrict__ Hx,
                                        const $REAL* __restrict__ Hy,
                                        const $REAL* __restrict__ Hz,
                                        $REAL *PHI1,
                                        $REAL *PHI2,
                                        const $REAL* __restrict__ RA,
                                        const $REAL* __restrict__ RB,
                                        const $REAL* __restrict__ RE,
                                        const $REAL* __restrict__ RF,
                                        $REAL d,
                                        const $REAL* __restrict__ updatecoeffsE,
                                        const $REAL* __restrict__ updatecoeffsH)
                    """
    )
}

y_args = {
    "hip": Template(
        """
                    __global__ void $FUNC(int xs,
                                        int xf,
                                        int ys,
                                        int yf,
                                        int zs,
                                        int zf,
                                        int NX_PHI1,
                                        int NY_PHI1,
                                        int NZ_PHI1,
                                        int NX_PHI2,
                                        int NY_PHI2,
                                        int NZ_PHI2,
                                        int NY_R,
                                        const unsigned int* __restrict__ ID,
                                        $REAL *Ex,
                                        const $REAL* __restrict__ Ey,
                                        $REAL *Ez,
                                        const $REAL* __restrict__ Hx,
                                        const $REAL* __restrict__ Hy,
                                        const $REAL* __restrict__ Hz,
                                        $REAL *PHI1,
                                        $REAL *PHI2,
                                        const $REAL* __restrict__ RA,
                                        const $REAL* __restrict__ RB,
                                        const $REAL* __restrict__ RE,
                                        const $REAL* __restrict__ RF,
                                        $REAL d,
                                        const $REAL* __restrict__ updatecoeffsE,
                                        const $REAL* __restrict__ updatecoeffsH)
                    """
    )
}

z_args = {
    "hip": Template(
        """
                    __global__ void $FUNC(int xs,
                                        int xf,
                                        int ys,
                                        int yf,
                                        int zs,
                                        int zf,
                                        int NX_PHI1,
                                        int NY_PHI1,
                                        int NZ_PHI1,
                                        int NX_PHI2,
                                        int NY_PHI2,
                                        int NZ_PHI2,
                                        int NY_R,
                                        const unsigned int* __restrict__ ID,
                                        $REAL *Ex,
                                        $REAL *Ey,
                                        const $REAL* __restrict__ Ez,
                                        const $REAL* __restrict__ Hx,
                                        const $REAL* __restrict__ Hy,
                                        const $REAL* __restrict__ Hz,
                                        $REAL *PHI1,
                                        $REAL *PHI2,
                                        const $REAL* __restrict__ RA,
                                        const $REAL* __restrict__ RB,
                                        const $REAL* __restrict__ RE,
                                        const $REAL* __restrict__ RF,
                                        $REAL d,
                                        const $REAL* __restrict__ updatecoeffsE,
                                        const $REAL* __restrict__ updatecoeffsH)
                    """
    )
}

order1_xminus = {
    "args_hip": x_args["hip"],
    "func": Template(
        """
    //  This function updates the Ey and Ez field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA01, RB0, RE0, RF0, dHy, dHz;
    $REAL dx = d;
    int ii, jj, kk, materialEy, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - i1;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,i1)] - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                     (RA01 * dHz + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - i2;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,i2)] - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                     (RA01 * dHy + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
"""
    ),
}

order2_xminus = {
    "args_hip": x_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ey and Ez field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHy, dHz;
    $REAL dx = d;
    int ii, jj, kk, materialEy, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - i1;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,i1)];
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RA1 = RA[IDX2D_R(1,i1)];
        RB1 = RB[IDX2D_R(1,i1)];
        RE1 = RE[IDX2D_R(1,i1)];
        RF1 = RF[IDX2D_R(1,i1)];
        RA01 = RA[IDX2D_R(0,i1)] * RA[IDX2D_R(1,i1)] - 1;

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                     (RA01 * dHz + RA1 * RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] +
                                      RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] - RF1 *
                                       (RA0 * dHz + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - i2;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,i2)];
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RA1 = RA[IDX2D_R(1,i2)];
        RB1 = RB[IDX2D_R(1,i2)];
        RE1 = RE[IDX2D_R(1,i2)];
        RF1 = RF[IDX2D_R(1,i2)];
        RA01 = RA[IDX2D_R(0,i2)] * RA[IDX2D_R(1,i2)] - 1;

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                     (RA01 * dHy + RA1 * RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] +
                                      RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] - RF1 *
                                       (RA0 * dHy + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
"""
    ),
}

order1_xplus = {
    "args_hip": x_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ey and Ez field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA01, RB0, RE0, RF0, dHy, dHz;
    $REAL dx = d;
    int ii, jj, kk, materialEy, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,i1)] - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                     (RA01 * dHz + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,i2)] - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                     (RA01 * dHy + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
"""
    ),
}

order2_xplus = {
    "args_hip": x_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ey and Ez field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHy, dHz;
    $REAL dx = d;
    int ii, jj, kk, materialEy, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,i1)];
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RA1 = RA[IDX2D_R(1,i1)];
        RB1 = RB[IDX2D_R(1,i1)];
        RE1 = RE[IDX2D_R(1,i1)];
        RF1 = RF[IDX2D_R(1,i1)];
        RA01 = RA[IDX2D_R(0,i1)] * RA[IDX2D_R(1,i1)] - 1;

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                     (RA01 * dHz + RA1 * RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] - RF1 *
                                       (RA0 * dHz + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,i2)];
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RA1 = RA[IDX2D_R(1,i2)];
        RB1 = RB[IDX2D_R(1,i2)];
        RE1 = RE[IDX2D_R(1,i2)];
        RF1 = RF[IDX2D_R(1,i2)];
        RA01 = RA[IDX2D_R(0,i2)] * RA[IDX2D_R(1,i2)] - 1;

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                     (RA01 * dHy + RA1 * RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] +
                                      RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] - RF1 *
                                       (RA0 * dHy + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
"""
    ),
}

order1_yminus = {
    "args_hip": y_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ex and Ez field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA01, RB0, RE0, RF0, dHx, dHz;
    $REAL dy = d;
    int ii, jj, kk, materialEx, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - j1;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,j1)] - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                     (RA01 * dHz + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - j2;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,j2)] - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                     (RA01 * dHx + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
"""
    ),
}

order2_yminus = {
    "args_hip": y_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ex and Ez field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHx, dHz;
    $REAL dy = d;
    int ii, jj, kk, materialEx, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - j1;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,j1)];
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RA1 = RA[IDX2D_R(1,j1)];
        RB1 = RB[IDX2D_R(1,j1)];
        RE1 = RE[IDX2D_R(1,j1)];
        RF1 = RF[IDX2D_R(1,j1)];
        RA01 = RA[IDX2D_R(0,j1)] * RA[IDX2D_R(1,j1)] - 1;

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                     (RA01 * dHz + RA1 * RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] +
                                      RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] - RF1 *
                                       (RA0 * dHz + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - j2;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,j2)];
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RA1 = RA[IDX2D_R(1,j2)];
        RB1 = RB[IDX2D_R(1,j2)];
        RE1 = RE[IDX2D_R(1,j2)];
        RF1 = RF[IDX2D_R(1,j2)];
        RA01 = RA[IDX2D_R(0,j2)] * RA[IDX2D_R(1,j2)] - 1;

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                     (RA01 * dHx + RA1 * RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] +
                                      RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] - RF1 *
                                       (RA0 * dHx + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
"""
    ),
}

order1_yplus = {
    "args_hip": y_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ex and Ez field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA01, RB0, RE0, RF0, dHx, dHz;
    $REAL dy = d;
    int ii, jj, kk, materialEx, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,j1)] - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                     (RA01 * dHz + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,j2)] - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                     (RA01 * dHx + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
"""
    ),
}

order2_yplus = {
    "args_hip": y_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ex and Ez field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHx, dHz;
    $REAL dy = d;
    int ii, jj, kk, materialEx, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,j1)];
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RA1 = RA[IDX2D_R(1,j1)];
        RB1 = RB[IDX2D_R(1,j1)];
        RE1 = RE[IDX2D_R(1,j1)];
        RF1 = RF[IDX2D_R(1,j1)];
        RA01 = RA[IDX2D_R(0,j1)] * RA[IDX2D_R(1,j1)] - 1;

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                     (RA01 * dHz + RA1 * RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] +
                                      RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] - RF1 *
                                       (RA0 * dHz + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,j2)];
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RA1 = RA[IDX2D_R(1,j2)];
        RB1 = RB[IDX2D_R(1,j2)];
        RE1 = RE[IDX2D_R(1,j2)];
        RF1 = RF[IDX2D_R(1,j2)];
        RA01 = RA[IDX2D_R(0,j2)] * RA[IDX2D_R(1,j2)] - 1;

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                     (RA01 * dHx + RA1 * RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] +
                                      RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] - RF1 *
                                       (RA0 * dHx + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
"""
    ),
}

order1_zminus = {
    "args_hip": z_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ex and Ey field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA01, RB0, RE0, RF0, dHx, dHy;
    $REAL dz = d;
    int ii, jj, kk, materialEx, materialEy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - k1;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,k1)] - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                     (RA01 * dHy + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - k2;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,k2)] - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                     (RA01 * dHx + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
"""
    ),
}

order2_zminus = {
    "args_hip": z_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ex and Ey field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHx, dHy;
    $REAL dz = d;
    int ii, jj, kk, materialEx, materialEy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - k1;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,k1)];
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RA1 = RA[IDX2D_R(1,k1)];
        RB1 = RB[IDX2D_R(1,k1)];
        RE1 = RE[IDX2D_R(1,k1)];
        RF1 = RF[IDX2D_R(1,k1)];
        RA01 = RA[IDX2D_R(0,k1)] * RA[IDX2D_R(1,k1)] - 1;

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                     (RA01 * dHy + RA1 * RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] +
                                      RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] - RF1 *
                                       (RA0 * dHy + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - k2;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,k2)];
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RA1 = RA[IDX2D_R(1,k2)];
        RB1 = RB[IDX2D_R(1,k2)];
        RE1 = RE[IDX2D_R(1,k2)];
        RF1 = RF[IDX2D_R(1,k2)];
        RA01 = RA[IDX2D_R(0,k2)] * RA[IDX2D_R(1,k2)] - 1;

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                     (RA01 * dHx + RA1 * RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] +
                                      RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] - RF1 *
                                       (RA0 * dHx + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
"""
    ),
}

order1_zplus = {
    "args_hip": z_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ex and Ey field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA01, RB0, RE0, RF0, dHx, dHy;
    $REAL dz = d;
    int ii, jj, kk, materialEx, materialEy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,k1)] - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                     (RA01 * dHy + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[IDX2D_R(0,k2)] - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                     (RA01 * dHx + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
"""
    ),
}

order2_zplus = {
    "args_hip": z_args["hip"],
    
    "func": Template(
        """
    //  This function updates the Ex and Ey field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHx, dHy;
    $REAL dz = d;
    int ii, jj, kk, materialEx, materialEy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,k1)];
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RA1 = RA[IDX2D_R(1,k1)];
        RB1 = RB[IDX2D_R(1,k1)];
        RE1 = RE[IDX2D_R(1,k1)];
        RF1 = RF[IDX2D_R(1,k1)];
        RA01 = RA[IDX2D_R(0,k1)] * RA[IDX2D_R(1,k1)] - 1;

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                     (RA01 * dHy + RA1 * RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] +
                                      RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] - RF1 *
                                       (RA0 * dHy + RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[IDX2D_R(0,k2)];
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RA1 = RA[IDX2D_R(1,k2)];
        RB1 = RB[IDX2D_R(1,k2)];
        RE1 = RE[IDX2D_R(1,k2)];
        RF1 = RF[IDX2D_R(1,k2)];
        RA01 = RA[IDX2D_R(0,k2)] * RA[IDX2D_R(1,k2)] - 1;

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                     (RA01 * dHx + RA1 * RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] +
                                      RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] - RF1 *
                                       (RA0 * dHx + RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
"""
    ),
}
