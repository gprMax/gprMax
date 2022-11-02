# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
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

from string import Template

x_args = {'cuda': Template("""
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
                                        const $REAL* __restrict__ Ex, $REAL *Ey, 
                                        $REAL *Ez, 
                                        const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *PHI1, 
                                        $REAL *PHI2, 
                                        const $REAL* __restrict__ RA, const $REAL* __restrict__ RB, const $REAL* __restrict__ RE, const $REAL* __restrict__ RF, $REAL d)
                    """),
          'opencl': Template("""
                        int xs,
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
                        __global const unsigned int* restrict ID,
                        __global const $REAL* restrict Ex,
                        __global $REAL *Ey,
                        __global $REAL *Ez,
                        __global const $REAL* restrict Hx,
                        __global const $REAL* restrict Hy,
                        __global const $REAL* restrict Hz,
                        __global $REAL *PHI1,
                        __global $REAL *PHI2,
                        __global const $REAL* restrict RA,
                        __global const $REAL* restrict RB,
                        __global const $REAL* restrict RE,
                        __global const $REAL* restrict RF,
                        $REAL d
                    """)
             }

y_args = {'cuda': Template("""
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
                                        const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *PHI1, 
                                        $REAL *PHI2, 
                                        const $REAL* __restrict__ RA, const $REAL* __restrict__ RB, const $REAL* __restrict__ RE, const $REAL* __restrict__ RF, $REAL d)
                    """),
          'opencl': Template("""
                        int xs,
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
                        __global const unsigned int* restrict ID,
                        __global $REAL *Ex,
                        __global const $REAL* restrict Ey,
                        __global $REAL *Ez,
                        __global const $REAL* restrict Hx,
                        __global const $REAL* restrict Hy,
                        __global const $REAL* restrict Hz,
                        __global $REAL *PHI1,
                        __global $REAL *PHI2,
                        __global const $REAL* restrict RA,
                        __global const $REAL* restrict RB,
                        __global const $REAL* restrict RE,
                        __global const $REAL* restrict RF,
                        $REAL d
                    """)
         }

z_args = {'cuda': Template("""
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
                                        const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *PHI1, 
                                        $REAL *PHI2, 
                                        const $REAL* __restrict__ RA, const $REAL* __restrict__ RB, const $REAL* __restrict__ RE, const $REAL* __restrict__ RF, $REAL d)
                    """),
          'opencl': Template("""
                        int xs,
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
                        __global const unsigned int* restrict ID,
                        __global $REAL *Ex,
                        __global $REAL *Ey,
                        __global const $REAL* restrict Ez,
                        __global const $REAL* restrict Hx,
                        __global const $REAL* restrict Hy,
                        __global const $REAL* restrict Hz,
                        __global $REAL *PHI1,
                        __global $REAL *PHI2,
                        __global const $REAL* restrict RA,
                        __global const $REAL* restrict RB,
                        __global const $REAL* restrict RE,
                        __global const $REAL* restrict RF,
                        $REAL d
                    """)
             }

order1_xminus = {'args_cuda': x_args['cuda'],
                 'args_opencl': x_args['opencl'], 
                 'func': Template("""
    //  This function updates the Ey and Ez field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dHy, dHz;
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
        IRA = 1 / RA[IDX2D_R(0,i1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RC0 = IRA * RB0 * RF0;

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                     (IRA1 * dHz - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dHz - 
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - i2;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,i2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RC0 = IRA * RB0 * RF0;

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                     (IRA1 * dHy - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dHy - 
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
""")}

order2_xminus = {'args_cuda': x_args['cuda'],
                 'args_opencl': x_args['opencl'], 
                 'func': Template("""
    //  This function updates the Ey and Ez field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dHy, dHz;
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
        IRA = 1 / (RA[IDX2D_R(0,i1)] + RA[IDX2D_R(1,i1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,i1)];
        RE1 = RE[IDX2D_R(1,i1)];
        RF1 = RF[IDX2D_R(1,i1)];
        RC1 = IRA * RF1;

        // Ey
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                     (IRA1 * dHz - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dHz - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dHz - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - i2;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,i2)] + RA[IDX2D_R(1,i2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,i2)];
        RE1 = RE[IDX2D_R(1,i2)];
        RF1 = RF[IDX2D_R(1,i2)];
        RC1 = IRA * RF1;

        // Ez
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                     (IRA1 * dHy - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dHy - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dHy - Psi2);
    }
""")}

order1_xplus = {'args_cuda': x_args['cuda'],
                'args_opencl': x_args['opencl'], 
                'func': Template("""
    //  This function updates the Ey and Ez field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dHy, dHz;
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
        IRA = 1 / RA[IDX2D_R(0,i1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RC0 = IRA * RB0 * RF0;

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                     (IRA1 * dHz - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dHz - 
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,i2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RC0 = IRA * RB0 * RF0;

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                     (IRA1 * dHy - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dHy - 
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
""")}

order2_xplus = {'args_cuda': x_args['cuda'],
                'args_opencl': x_args['opencl'], 
                'func': Template("""
    //  This function updates the Ey and Ez field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dHy, dHz;
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
        IRA = 1 / (RA[IDX2D_R(0,i1)] + RA[IDX2D_R(1,i1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,i1)];
        RE1 = RE[IDX2D_R(1,i1)];
        RF1 = RF[IDX2D_R(1,i1)];
        RC1 = IRA * RF1;

        // Ey
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                     (IRA1 * dHz - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dHz - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dHz - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,i2)] + RA[IDX2D_R(1,i2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,i2)];
        RE1 = RE[IDX2D_R(1,i2)];
        RF1 = RF[IDX2D_R(1,i2)];
        RC1 = IRA * RF1;

        // Ez
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                     (IRA1 * dHy - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dHy - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dHy - Psi2);
    }
""")}

order1_yminus = {'args_cuda': y_args['cuda'],
                 'args_opencl': y_args['opencl'], 
                 'func': Template("""
    //  This function updates the Ex and Ez field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dHx, dHz;
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
        IRA = 1 / RA[IDX2D_R(0,j1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RC0 = IRA * RB0 * RF0;

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                     (IRA1 * dHz - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dHz - 
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - j2;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,j2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RC0 = IRA * RB0 * RF0;

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                     (IRA1 * dHx - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dHx - 
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
""")}

order2_yminus = {'args_cuda': y_args['cuda'],
                 'args_opencl': y_args['opencl'], 
                 'func': Template("""
    //  This function updates the Ex and Ez field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dHx, dHz;
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
        IRA = 1 / (RA[IDX2D_R(0,j1)] + RA[IDX2D_R(1,j1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,j1)];
        RE1 = RE[IDX2D_R(1,j1)];
        RF1 = RF[IDX2D_R(1,j1)];
        RC1 = IRA * RF1;

        // Ex
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                     (IRA1 * dHz - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dHz - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dHz - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - j2;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,j2)] + RA[IDX2D_R(1,j2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,j2)];
        RE1 = RE[IDX2D_R(1,j2)];
        RF1 = RF[IDX2D_R(1,j2)];
        RC1 = IRA * RF1;

        // Ez
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                     (IRA1 * dHx - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dHx - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dHx - Psi2);
    }
""")}

order1_yplus = {'args_cuda': y_args['cuda'],
                'args_opencl': y_args['opencl'], 
                'func': Template("""
    //  This function updates the Ex and Ez field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dHx, dHz;
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
        IRA = 1 / RA[IDX2D_R(0,j1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RC0 = IRA * RB0 * RF0;

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                     (IRA1 * dHz - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dHz - 
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,j2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RC0 = IRA * RB0 * RF0;

        // Ez
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                     (IRA1 * dHx - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dHx - 
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
""")}

order2_yplus = {'args_cuda': y_args['cuda'],
                'args_opencl': y_args['opencl'], 
                'func': Template("""
    //  This function updates the Ex and Ez field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dHx, dHz;
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
        IRA = 1 / (RA[IDX2D_R(0,j1)] + RA[IDX2D_R(1,j1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,j1)];
        RE1 = RE[IDX2D_R(1,j1)];
        RF1 = RF[IDX2D_R(1,j1)];
        RC1 = IRA * RF1;

        // Ex
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[IDX3D_FIELDS(ii,jj,kk)] - Hz[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                     (IRA1 * dHz - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dHz - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dHz - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,j2)] + RA[IDX2D_R(1,j2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,j2)];
        RE1 = RE[IDX2D_R(1,j2)];
        RF1 = RF[IDX2D_R(1,j2)];
        RC1 = IRA * RF1;

        // Ez
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialEz = ID[IDX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[IDX3D_FIELDS(ii,jj,kk)] = Ez[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                     (IRA1 * dHx - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dHx - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dHx - Psi2);
    }
""")}

order1_zminus = {'args_cuda': z_args['cuda'],
                 'args_opencl': z_args['opencl'], 
                 'func': Template("""
    //  This function updates the Ex and Ey field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dHx, dHy;
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
        IRA = 1 / RA[IDX2D_R(0,k1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RC0 = IRA * RB0 * RF0;

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                     (IRA1 * dHy - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dHy - 
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - k2;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,k2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RC0 = IRA * RB0 * RF0;

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                     (IRA1 * dHx - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dHx - 
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
""")}

order2_zminus = {'args_cuda': z_args['cuda'],
                 'args_opencl': z_args['opencl'], 
                 'func': Template("""
    //  This function updates the Ex and Ey field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dHx, dHy;
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
        IRA = 1 / (RA[IDX2D_R(0,k1)] + RA[IDX2D_R(1,k1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,k1)];
        RE1 = RE[IDX2D_R(1,k1)];
        RF1 = RF[IDX2D_R(1,k1)];
        RC1 = IRA * RF1;

        // Ex
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                     (IRA1 * dHy - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dHy - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dHy - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - k2;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,k2)] + RA[IDX2D_R(1,k2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,k2)];
        RE1 = RE[IDX2D_R(1,k2)];
        RF1 = RF[IDX2D_R(1,k2)];
        RC1 = IRA * RF1;

        // Ey
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                     (IRA1 * dHx - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dHx - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dHx - Psi2);
    }
""")}

order1_zplus = {'args_cuda': z_args['cuda'],
                'args_opencl': z_args['opencl'], 
                'func': Template("""
    //  This function updates the Ex and Ey field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dHx, dHy;
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
        IRA = 1 / RA[IDX2D_R(0,k1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RC0 = IRA * RB0 * RF0;

        // Ex
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                     (IRA1 * dHy - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dHy - 
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,k2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RC0 = IRA * RB0 * RF0;

        // Ey
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                     (IRA1 * dHx - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dHx - 
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
""")}

order2_zplus = {'args_cuda': z_args['cuda'],
                'args_opencl': z_args['opencl'], 
                'func': Template("""
    //  This function updates the Ex and Ey field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    $CUDA_IDX

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

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dHx, dHy;
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
        IRA = 1 / (RA[IDX2D_R(0,k1)] + RA[IDX2D_R(1,k1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,k1)];
        RE1 = RE[IDX2D_R(1,k1)];
        RF1 = RF[IDX2D_R(1,k1)];
        RC1 = IRA * RF1;

        // Ex
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialEx = ID[IDX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[IDX3D_FIELDS(ii,jj,kk)] - Hy[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[IDX3D_FIELDS(ii,jj,kk)] = Ex[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                     (IRA1 * dHy - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dHy - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dHy - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,k2)] + RA[IDX2D_R(1,k2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,k2)];
        RE1 = RE[IDX2D_R(1,k2)];
        RF1 = RF[IDX2D_R(1,k2)];
        RC1 = IRA * RF1;

        // Ey
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialEy = ID[IDX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[IDX3D_FIELDS(ii,jj,kk)] - Hx[IDX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[IDX3D_FIELDS(ii,jj,kk)] = Ey[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                     (IRA1 * dHx - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dHx - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dHx - Psi2);
    }
""")}