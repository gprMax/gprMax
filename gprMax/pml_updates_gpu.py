# Copyright (C) 2015-2018: The University of Edinburgh
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

from string import Template

kernels_template_pml = Template("""

// Macros for converting subscripts to linear index:
#define INDEX2D_R(m, n) (m)*($NY_R)+(n)
#define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
#define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
#define INDEX4D_EPHI1(p, i, j, k) (p)*(NX_EPHI1)*(NY_EPHI1)*(NZ_EPHI1)+(i)*(NY_EPHI1)*(NZ_EPHI1)+(j)*(NZ_EPHI1)+(k)
#define INDEX4D_EPHI2(p, i, j, k) (p)*(NX_EPHI2)*(NY_EPHI2)*(NZ_EPHI2)+(i)*(NY_EPHI2)*(NZ_EPHI2)+(j)*(NZ_EPHI2)+(k)
#define INDEX4D_HPHI1(p, i, j, k) (p)*(NX_HPHI1)*(NY_HPHI1)*(NZ_HPHI1)+(i)*(NY_HPHI1)*(NZ_HPHI1)+(j)*(NZ_HPHI1)+(k)
#define INDEX4D_HPHI2(p, i, j, k) (p)*(NX_HPHI2)*(NY_HPHI2)*(NZ_HPHI2)+(i)*(NY_HPHI2)*(NZ_HPHI2)+(j)*(NZ_HPHI2)+(k)

// Material coefficients (read-only) in constant memory (64KB)
__device__ __constant__ $REAL updatecoeffsE[$N_updatecoeffsE];
__device__ __constant__ $REAL updatecoeffsH[$N_updatecoeffsH];

//////////////////////////////////////////////////////////
// Electric field PML updates - 1st order - xminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_1order_electric_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, $REAL *Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ey and Ez field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

    $REAL RA0, RB0, RE0, RF0, dHy, dHz;
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
        RA0 = ERA[INDEX2D_R(0,i1)] - 1;
        RB0 = ERB[INDEX2D_R(0,i1)];
        RE0 = ERE[INDEX2D_R(0,i1)];
        RF0 = ERF[INDEX2D_R(0,i1)];

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA0 * dHz + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - i2;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,i2)] - 1;
        RB0 = ERB[INDEX2D_R(0,i2)];
        RE0 = ERE[INDEX2D_R(0,i2)];
        RF0 = ERF[INDEX2D_R(0,i2)];

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA0 * dHy + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
}


/////////////////////////////////////////////////////////
// Electric field PML updates - 1st order - xplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_1order_electric_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, $REAL *Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ey and Ez field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

    $REAL RA0, RB0, RE0, RF0, dHy, dHz;
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
        RA0 = ERA[INDEX2D_R(0,i1)] - 1;
        RB0 = ERB[INDEX2D_R(0,i1)];
        RE0 = ERE[INDEX2D_R(0,i1)];
        RF0 = ERF[INDEX2D_R(0,i1)];

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA0 * dHz + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,i2)] - 1;
        RB0 = ERB[INDEX2D_R(0,i2)];
        RE0 = ERE[INDEX2D_R(0,i2)];
        RF0 = ERF[INDEX2D_R(0,i2)];

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA0 * dHy + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
}


//////////////////////////////////////////////////////////
// Electric field PML updates - 1st order - yminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_1order_electric_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, $REAL *Ex, const $REAL* __restrict__ Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ex and Ez field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

    $REAL RA0, RB0, RE0, RF0, dHx, dHz;
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
        RA0 = ERA[INDEX2D_R(0,j1)] - 1;
        RB0 = ERB[INDEX2D_R(0,j1)];
        RE0 = ERE[INDEX2D_R(0,j1)];
        RF0 = ERF[INDEX2D_R(0,j1)];

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA0 * dHz + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - j2;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,j2)] - 1;
        RB0 = ERB[INDEX2D_R(0,j2)];
        RE0 = ERE[INDEX2D_R(0,j2)];
        RF0 = ERF[INDEX2D_R(0,j2)];

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA0 * dHx + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


/////////////////////////////////////////////////////////
// Electric field PML updates - 1st order - yplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_1order_electric_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, $REAL *Ex, const $REAL* __restrict__ Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ex and Ez field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

    $REAL RA0, RB0, RE0, RF0, dHx, dHz;
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
        RA0 = ERA[INDEX2D_R(0,j1)] - 1;
        RB0 = ERB[INDEX2D_R(0,j1)];
        RE0 = ERE[INDEX2D_R(0,j1)];
        RF0 = ERF[INDEX2D_R(0,j1)];

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA0 * dHz + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,j2)] - 1;
        RB0 = ERB[INDEX2D_R(0,j2)];
        RE0 = ERE[INDEX2D_R(0,j2)];
        RF0 = ERF[INDEX2D_R(0,j2)];

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA0 * dHx + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


//////////////////////////////////////////////////////////
// Electric field PML updates - 1st order - zminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_1order_electric_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, $REAL *Ex, $REAL *Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ex and Ey field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

    $REAL RA0, RB0, RE0, RF0, dHx, dHy;
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
        RA0 = ERA[INDEX2D_R(0,k1)] - 1;
        RB0 = ERB[INDEX2D_R(0,k1)];
        RE0 = ERE[INDEX2D_R(0,k1)];
        RF0 = ERF[INDEX2D_R(0,k1)];

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA0 * dHy + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + xs;
        kk = zf - k2;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,k2)] - 1;
        RB0 = ERB[INDEX2D_R(0,k2)];
        RE0 = ERE[INDEX2D_R(0,k2)];
        RF0 = ERF[INDEX2D_R(0,k2)];

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA0 * dHx + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


/////////////////////////////////////////////////////////
// Electric field PML updates - 1st order - zplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_1order_electric_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, $REAL *Ex, $REAL *Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ex and Ey field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      updatecoeffs, ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

    $REAL RA0, RB0, RE0, RF0, dHx, dHy;
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
        RA0 = ERA[INDEX2D_R(0,k1)] - 1;
        RB0 = ERB[INDEX2D_R(0,k1)];
        RE0 = ERE[INDEX2D_R(0,k1)];
        RF0 = ERF[INDEX2D_R(0,k1)];

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA0 * dHy + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,k2)] - 1;
        RB0 = ERB[INDEX2D_R(0,k2)];
        RE0 = ERE[INDEX2D_R(0,k2)];
        RF0 = ERF[INDEX2D_R(0,k2)];

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA0 * dHx + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


//////////////////////////////////////////////////////////
// Magnetic field PML updates - 1st order - xminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_1order_magnetic_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, $REAL *Hy, $REAL *Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hy and Hz field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, dEy, dEz;
    $REAL dx = d;
    int ii, jj, kk, materialHy, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - (i1 + 1);
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,i1)] - 1;
        RB0 = HRB[INDEX2D_R(0,i1)];
        RE0 = HRE[INDEX2D_R(0,i1)];
        RF0 = HRF[INDEX2D_R(0,i1)];

        // Hy
        materialHy = ID[INDEX4D_ID(4,ii,jj,kk)];
        dEz = (Ez[INDEX3D_FIELDS(ii+1,jj,kk)] - Ez[INDEX3D_FIELDS(ii,jj,kk)]) / dx;
        Hy[INDEX3D_FIELDS(ii,jj,kk)] = Hy[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHy,4)] * (RA0 * dEz + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - (i2 + 1);
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,i2)] - 1;
        RB0 = HRB[INDEX2D_R(0,i2)];
        RE0 = HRE[INDEX2D_R(0,i2)];
        RF0 = HRF[INDEX2D_R(0,i2)];

        // Hz
        materialHz = ID[INDEX4D_ID(5,ii,jj,kk)];
        dEy = (Ey[INDEX3D_FIELDS(ii+1,jj,kk)] - Ey[INDEX3D_FIELDS(ii,jj,kk)]) / dx;
        Hz[INDEX3D_FIELDS(ii,jj,kk)] = Hz[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHz,4)] * (RA0 * dEy + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEy;
    }
}


/////////////////////////////////////////////////////////
// Magnetic field PML updates - 1st order - xplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_1order_magnetic_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, $REAL *Hy, $REAL *Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hy and Hz field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, dEy, dEz;
    $REAL dx = d;
    int ii, jj, kk, materialHy, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,i1)] - 1;
        RB0 = HRB[INDEX2D_R(0,i1)];
        RE0 = HRE[INDEX2D_R(0,i1)];
        RF0 = HRF[INDEX2D_R(0,i1)];

        // Hy
        materialHy = ID[INDEX4D_ID(4,ii,jj,kk)];
        dEz = (Ez[INDEX3D_FIELDS(ii+1,jj,kk)] - Ez[INDEX3D_FIELDS(ii,jj,kk)]) / dx;
        Hy[INDEX3D_FIELDS(ii,jj,kk)] = Hy[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHy,4)] * (RA0 * dEz + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,i2)] - 1;
        RB0 = HRB[INDEX2D_R(0,i2)];
        RE0 = HRE[INDEX2D_R(0,i2)];
        RF0 = HRF[INDEX2D_R(0,i2)];

        // Hz
        materialHz = ID[INDEX4D_ID(5,ii,jj,kk)];
        dEy = (Ey[INDEX3D_FIELDS(ii+1,jj,kk)] - Ey[INDEX3D_FIELDS(ii,jj,kk)]) / dx;
        Hz[INDEX3D_FIELDS(ii,jj,kk)] = Hz[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHz,4)] * (RA0 * dEy + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEy;
    }
}


//////////////////////////////////////////////////////////
// Magnetic field PML updates - 1st order - yminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_1order_magnetic_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, $REAL *Hx, const $REAL* __restrict__ Hy, $REAL *Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hx and Hz field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, dEx, dEz;
    $REAL dy = d;
    int ii, jj, kk, materialHx, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - (j1 + 1);
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,j1)] - 1;
        RB0 = HRB[INDEX2D_R(0,j1)];
        RE0 = HRE[INDEX2D_R(0,j1)];
        RF0 = HRF[INDEX2D_R(0,j1)];

        // Hx
        materialHx = ID[INDEX4D_ID(3,ii,jj,kk)];
        dEz = (Ez[INDEX3D_FIELDS(ii,jj+1,kk)] - Ez[INDEX3D_FIELDS(ii,jj,kk)]) / dy;
        Hx[INDEX3D_FIELDS(ii,jj,kk)] = Hx[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHx,4)] * (RA0 * dEz + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - (j2 + 1);
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,j2)] - 1;
        RB0 = HRB[INDEX2D_R(0,j2)];
        RE0 = HRE[INDEX2D_R(0,j2)];
        RF0 = HRF[INDEX2D_R(0,j2)];

        // Hz
        materialHz = ID[INDEX4D_ID(5,ii,jj,kk)];
        dEx = (Ex[INDEX3D_FIELDS(ii,jj+1,kk)] - Ex[INDEX3D_FIELDS(ii,jj,kk)]) / dy;
        Hz[INDEX3D_FIELDS(ii,jj,kk)] = Hz[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHz,4)] * (RA0 * dEx + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEx;
    }
}


/////////////////////////////////////////////////////////
// Magnetic field PML updates - 1st order - yplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_1order_magnetic_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, $REAL *Hx, const $REAL* __restrict__ Hy, $REAL *Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hx and Hz field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, dEx, dEz;
    $REAL dy = d;
    int ii, jj, kk, materialHx, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,j1)] - 1;
        RB0 = HRB[INDEX2D_R(0,j1)];
        RE0 = HRE[INDEX2D_R(0,j1)];
        RF0 = HRF[INDEX2D_R(0,j1)];

        // Hx
        materialHx = ID[INDEX4D_ID(3,ii,jj,kk)];
        dEz = (Ez[INDEX3D_FIELDS(ii,jj+1,kk)] - Ez[INDEX3D_FIELDS(ii,jj,kk)]) / dy;
        Hx[INDEX3D_FIELDS(ii,jj,kk)] = Hx[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHx,4)] * (RA0 * dEz + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,j2)] - 1;
        RB0 = HRB[INDEX2D_R(0,j2)];
        RE0 = HRE[INDEX2D_R(0,j2)];
        RF0 = HRF[INDEX2D_R(0,j2)];

        // Hz
        materialHz = ID[INDEX4D_ID(5,ii,jj,kk)];
        dEx = (Ex[INDEX3D_FIELDS(ii,jj+1,kk)] - Ex[INDEX3D_FIELDS(ii,jj,kk)]) / dy;
        Hz[INDEX3D_FIELDS(ii,jj,kk)] = Hz[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHz,4)] * (RA0 * dEx + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEx;
    }
}


//////////////////////////////////////////////////////////
// Magnetic field PML updates - 1st order - zminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_1order_magnetic_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, $REAL *Hx, $REAL *Hy, const $REAL* __restrict__ Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hx and Hy field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, dEx, dEy;
    $REAL dz = d;
    int ii, jj, kk, materialHx, materialHy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - (k1 + 1);

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,k1)] - 1;
        RB0 = HRB[INDEX2D_R(0,k1)];
        RE0 = HRE[INDEX2D_R(0,k1)];
        RF0 = HRF[INDEX2D_R(0,k1)];

        // Hx
        materialHx = ID[INDEX4D_ID(3,ii,jj,kk)];
        dEy = (Ey[INDEX3D_FIELDS(ii,jj,kk+1)] - Ey[INDEX3D_FIELDS(ii,jj,kk)]) / dz;
        Hx[INDEX3D_FIELDS(ii,jj,kk)] = Hx[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHx,4)] * (RA0 * dEy + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - (k2 + 1);

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,k2)] - 1;
        RB0 = HRB[INDEX2D_R(0,k2)];
        RE0 = HRE[INDEX2D_R(0,k2)];
        RF0 = HRF[INDEX2D_R(0,k2)];

        // Hy
        materialHy = ID[INDEX4D_ID(4,ii,jj,kk)];
        dEx = (Ex[INDEX3D_FIELDS(ii,jj,kk+1)] - Ex[INDEX3D_FIELDS(ii,jj,kk)]) / dz;
        Hy[INDEX3D_FIELDS(ii,jj,kk)] = Hy[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHy,4)] * (RA0 * dEx + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEx;
    }
}


/////////////////////////////////////////////////////////
// Magnetic field PML updates - 1st order - zplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_1order_magnetic_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, $REAL *Hx, $REAL *Hy, const $REAL* __restrict__ Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hx and Hy field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, dEx, dEy;
    $REAL dz = d;
    int ii, jj, kk, materialHx, materialHy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,k1)] - 1;
        RB0 = HRB[INDEX2D_R(0,k1)];
        RE0 = HRE[INDEX2D_R(0,k1)];
        RF0 = HRF[INDEX2D_R(0,k1)];

        // Hx
        materialHx = ID[INDEX4D_ID(3,ii,jj,kk)];
        dEy = (Ey[INDEX3D_FIELDS(ii,jj,kk+1)] - Ey[INDEX3D_FIELDS(ii,jj,kk)]) / dz;
        Hx[INDEX3D_FIELDS(ii,jj,kk)] = Hx[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHx,4)] * (RA0 * dEy + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,k2)] - 1;
        RB0 = HRB[INDEX2D_R(0,k2)];
        RE0 = HRE[INDEX2D_R(0,k2)];
        RF0 = HRF[INDEX2D_R(0,k2)];

        // Hy
        materialHy = ID[INDEX4D_ID(4,ii,jj,kk)];
        dEx = (Ex[INDEX3D_FIELDS(ii,jj,kk+1)] - Ex[INDEX3D_FIELDS(ii,jj,kk)]) / dz;
        Hy[INDEX3D_FIELDS(ii,jj,kk)] = Hy[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHy,4)] * (RA0 * dEx + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEx;
    }
}


//////////////////////////////////////////////////////////
// Electric field PML updates - 2nd order - xminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_2order_electric_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, $REAL *Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ey and Ez field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

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
        RA0 = ERA[INDEX2D_R(0,i1)];
        RB0 = ERB[INDEX2D_R(0,i1)];
        RE0 = ERE[INDEX2D_R(0,i1)];
        RF0 = ERF[INDEX2D_R(0,i1)];
        RA1 = ERA[INDEX2D_R(1,i1)];
        RB1 = ERB[INDEX2D_R(1,i1)];
        RE1 = ERE[INDEX2D_R(1,i1)];
        RF1 = ERF[INDEX2D_R(1,i1)];
        RA01 = ERA[INDEX2D_R(0,i1)] * ERA[INDEX2D_R(1,i1)] - 1;

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHz + RA1 * RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] + RB1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] = RE1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHz + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - i2;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,i2)];
        RB0 = ERB[INDEX2D_R(0,i2)];
        RE0 = ERE[INDEX2D_R(0,i2)];
        RF0 = ERF[INDEX2D_R(0,i2)];
        RA1 = ERA[INDEX2D_R(1,i2)];
        RB1 = ERB[INDEX2D_R(1,i2)];
        RE1 = ERE[INDEX2D_R(1,i2)];
        RF1 = ERF[INDEX2D_R(1,i2)];
        RA01 = ERA[INDEX2D_R(0,i2)] * ERA[INDEX2D_R(1,i2)] - 1;

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHy + RA1 * RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] + RB1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] = RE1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHy + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
}


/////////////////////////////////////////////////////////
// Electric field PML updates - 2nd order - xplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_2order_electric_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, $REAL *Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ey and Ez field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

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
        RA0 = ERA[INDEX2D_R(0,i1)];
        RB0 = ERB[INDEX2D_R(0,i1)];
        RE0 = ERE[INDEX2D_R(0,i1)];
        RF0 = ERF[INDEX2D_R(0,i1)];
        RA1 = ERA[INDEX2D_R(1,i1)];
        RB1 = ERB[INDEX2D_R(1,i1)];
        RE1 = ERE[INDEX2D_R(1,i1)];
        RF1 = ERF[INDEX2D_R(1,i1)];
        RA01 = ERA[INDEX2D_R(0,i1)] * ERA[INDEX2D_R(1,i1)] - 1;

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHz + RA1 * RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] + RB1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] = RE1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHz + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,i2)];
        RB0 = ERB[INDEX2D_R(0,i2)];
        RE0 = ERE[INDEX2D_R(0,i2)];
        RF0 = ERF[INDEX2D_R(0,i2)];
        RA1 = ERA[INDEX2D_R(1,i2)];
        RB1 = ERB[INDEX2D_R(1,i2)];
        RE1 = ERE[INDEX2D_R(1,i2)];
        RF1 = ERF[INDEX2D_R(1,i2)];
        RA01 = ERA[INDEX2D_R(0,i2)] * ERA[INDEX2D_R(1,i2)] - 1;

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHy + RA1 * RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] + RB1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] = RE1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHy + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
}


//////////////////////////////////////////////////////////
// Electric field PML updates - 2nd order - yminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_2order_electric_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, $REAL *Ex, const $REAL* __restrict__ Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ex and Ez field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

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
        RA0 = ERA[INDEX2D_R(0,j1)];
        RB0 = ERB[INDEX2D_R(0,j1)];
        RE0 = ERE[INDEX2D_R(0,j1)];
        RF0 = ERF[INDEX2D_R(0,j1)];
        RA1 = ERA[INDEX2D_R(1,j1)];
        RB1 = ERB[INDEX2D_R(1,j1)];
        RE1 = ERE[INDEX2D_R(1,j1)];
        RF1 = ERF[INDEX2D_R(1,j1)];
        RA01 = ERA[INDEX2D_R(0,j1)] * ERA[INDEX2D_R(1,j1)] - 1;

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHz + RA1 * RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] + RB1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] = RE1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHz + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - j2;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,j2)];
        RB0 = ERB[INDEX2D_R(0,j2)];
        RE0 = ERE[INDEX2D_R(0,j2)];
        RF0 = ERF[INDEX2D_R(0,j2)];
        RA1 = ERA[INDEX2D_R(1,j2)];
        RB1 = ERB[INDEX2D_R(1,j2)];
        RE1 = ERE[INDEX2D_R(1,j2)];
        RF1 = ERF[INDEX2D_R(1,j2)];
        RA01 = ERA[INDEX2D_R(0,j2)] * ERA[INDEX2D_R(1,j2)] - 1;

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHx + RA1 * RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] + RB1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] = RE1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHx + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


/////////////////////////////////////////////////////////
// Electric field PML updates - 2nd order - yplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_2order_electric_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, $REAL *Ex, const $REAL* __restrict__ Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ex and Ez field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

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
        RA0 = ERA[INDEX2D_R(0,j1)];
        RB0 = ERB[INDEX2D_R(0,j1)];
        RE0 = ERE[INDEX2D_R(0,j1)];
        RF0 = ERF[INDEX2D_R(0,j1)];
        RA1 = ERA[INDEX2D_R(1,j1)];
        RB1 = ERB[INDEX2D_R(1,j1)];
        RE1 = ERE[INDEX2D_R(1,j1)];
        RF1 = ERF[INDEX2D_R(1,j1)];
        RA01 = ERA[INDEX2D_R(0,j1)] * ERA[INDEX2D_R(1,j1)] - 1;

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHz + RA1 * RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] + RB1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] = RE1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHz + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,j2)];
        RB0 = ERB[INDEX2D_R(0,j2)];
        RE0 = ERE[INDEX2D_R(0,j2)];
        RF0 = ERF[INDEX2D_R(0,j2)];
        RA1 = ERA[INDEX2D_R(1,j2)];
        RB1 = ERB[INDEX2D_R(1,j2)];
        RE1 = ERE[INDEX2D_R(1,j2)];
        RF1 = ERF[INDEX2D_R(1,j2)];
        RA01 = ERA[INDEX2D_R(0,j2)] * ERA[INDEX2D_R(1,j2)] - 1;

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHx + RA1 * RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] + RB1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] = RE1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHx + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


//////////////////////////////////////////////////////////
// Electric field PML updates - 2nd order - zminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_2order_electric_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, $REAL *Ex, $REAL *Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ex and Ey field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

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
        RA0 = ERA[INDEX2D_R(0,k1)];
        RB0 = ERB[INDEX2D_R(0,k1)];
        RE0 = ERE[INDEX2D_R(0,k1)];
        RF0 = ERF[INDEX2D_R(0,k1)];
        RA1 = ERA[INDEX2D_R(1,k1)];
        RB1 = ERB[INDEX2D_R(1,k1)];
        RE1 = ERE[INDEX2D_R(1,k1)];
        RF1 = ERF[INDEX2D_R(1,k1)];
        RA01 = ERA[INDEX2D_R(0,k1)] * ERA[INDEX2D_R(1,k1)] - 1;

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHy + RA1 * RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] + RB1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] = RE1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHy + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + xs;
        kk = zf - k2;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,k2)];
        RB0 = ERB[INDEX2D_R(0,k2)];
        RE0 = ERE[INDEX2D_R(0,k2)];
        RF0 = ERF[INDEX2D_R(0,k2)];
        RA1 = ERA[INDEX2D_R(1,k2)];
        RB1 = ERB[INDEX2D_R(1,k2)];
        RE1 = ERE[INDEX2D_R(1,k2)];
        RF1 = ERF[INDEX2D_R(1,k2)];
        RA01 = ERA[INDEX2D_R(0,k2)] * ERA[INDEX2D_R(1,k2)] - 1;

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHx + RA1 * RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] + RB1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] = RE1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHx + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


/////////////////////////////////////////////////////////
// Electric field PML updates - 2nd order - zplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_2order_electric_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_EPHI1, int NY_EPHI1, int NZ_EPHI1, int NX_EPHI2, int NY_EPHI2, int NZ_EPHI2, const unsigned int* __restrict__ ID, $REAL *Ex, $REAL *Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz, $REAL *EPhi1, $REAL *EPhi2, const $REAL* __restrict__ ERA, const $REAL* __restrict__ ERB, const $REAL* __restrict__ ERE, const $REAL* __restrict__ ERF, $REAL d) {

    //  This function updates the Ex and Ey field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_EPHI, NY_EPHI, NZ_EPHI: Dimensions of EPhi1 and EPhi2 PML arrays
    //      updatecoeffs, ID, E, H: Access to ID and field component arrays
    //      EPhi, ERA, ERB, ERE, ERF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML EPhi1 (4D) arrays
    int p1 = idx / (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1);
    int i1 = (idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) / (NY_EPHI1 * NZ_EPHI1);
    int j1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) / NZ_EPHI1;
    int k1 = ((idx % (NX_EPHI1 * NY_EPHI1 * NZ_EPHI1)) % (NY_EPHI1 * NZ_EPHI1)) % NZ_EPHI1;

    // Convert the linear index to subscripts for PML EPhi2 (4D) arrays
    int p2 = idx / (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2);
    int i2 = (idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) / (NY_EPHI2 * NZ_EPHI2);
    int j2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) / NZ_EPHI2;
    int k2 = ((idx % (NX_EPHI2 * NY_EPHI2 * NZ_EPHI2)) % (NY_EPHI2 * NZ_EPHI2)) % NZ_EPHI2;

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
        RA0 = ERA[INDEX2D_R(0,k1)];
        RB0 = ERB[INDEX2D_R(0,k1)];
        RE0 = ERE[INDEX2D_R(0,k1)];
        RF0 = ERF[INDEX2D_R(0,k1)];
        RA1 = ERA[INDEX2D_R(1,k1)];
        RB1 = ERB[INDEX2D_R(1,k1)];
        RE1 = ERE[INDEX2D_R(1,k1)];
        RF1 = ERF[INDEX2D_R(1,k1)];
        RA01 = ERA[INDEX2D_R(0,k1)] * ERA[INDEX2D_R(1,k1)] - 1;

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHy + RA1 * RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] + RB1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] = RE1 * EPhi1[INDEX4D_EPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHy + RB0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)]);
        EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] = RE0 * EPhi1[INDEX4D_EPHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = ERA[INDEX2D_R(0,k2)];
        RB0 = ERB[INDEX2D_R(0,k2)];
        RE0 = ERE[INDEX2D_R(0,k2)];
        RF0 = ERF[INDEX2D_R(0,k2)];
        RA1 = ERA[INDEX2D_R(1,k2)];
        RB1 = ERB[INDEX2D_R(1,k2)];
        RE1 = ERE[INDEX2D_R(1,k2)];
        RF1 = ERF[INDEX2D_R(1,k2)];
        RA01 = ERA[INDEX2D_R(0,k2)] * ERA[INDEX2D_R(1,k2)] - 1;

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHx + RA1 * RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] + RB1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] = RE1 * EPhi2[INDEX4D_EPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHx + RB0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)]);
        EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] = RE0 * EPhi2[INDEX4D_EPHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


//////////////////////////////////////////////////////////
// Magnetic field PML updates - 2nd order - xminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_2order_magnetic_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, $REAL *Hy, $REAL *Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hy and Hz field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dEy, dEz;
    $REAL dx = d;
    int ii, jj, kk, materialHy, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - (i1 + 1);
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,i1)];
        RB0 = HRB[INDEX2D_R(0,i1)];
        RE0 = HRE[INDEX2D_R(0,i1)];
        RF0 = HRF[INDEX2D_R(0,i1)];
        RA1 = HRA[INDEX2D_R(1,i1)];
        RB1 = HRB[INDEX2D_R(1,i1)];
        RE1 = HRE[INDEX2D_R(1,i1)];
        RF1 = HRF[INDEX2D_R(1,i1)];
        RA01 = HRA[INDEX2D_R(0,i1)] * HRA[INDEX2D_R(1,i1)] - 1;

        // Hy
        materialHy = ID[INDEX4D_ID(4,ii,jj,kk)];
        dEz = (Ez[INDEX3D_FIELDS(ii+1,jj,kk)] - Ez[INDEX3D_FIELDS(ii,jj,kk)]) / dx;
        Hy[INDEX3D_FIELDS(ii,jj,kk)] = Hy[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHy,4)] * (RA01 * dEz + RA1 * RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] + RB1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] = RE1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dEz + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - (i2 + 1);
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,i2)];
        RB0 = HRB[INDEX2D_R(0,i2)];
        RE0 = HRE[INDEX2D_R(0,i2)];
        RF0 = HRF[INDEX2D_R(0,i2)];
        RA1 = HRA[INDEX2D_R(1,i2)];
        RB1 = HRB[INDEX2D_R(1,i2)];
        RE1 = HRE[INDEX2D_R(1,i2)];
        RF1 = HRF[INDEX2D_R(1,i2)];
        RA01 = HRA[INDEX2D_R(0,i2)] * HRA[INDEX2D_R(1,i2)] - 1;

        // Hz
        materialHz = ID[INDEX4D_ID(5,ii,jj,kk)];
        dEy = (Ey[INDEX3D_FIELDS(ii+1,jj,kk)] - Ey[INDEX3D_FIELDS(ii,jj,kk)]) / dx;
        Hz[INDEX3D_FIELDS(ii,jj,kk)] = Hz[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHz,4)] * (RA01 * dEy + RA1 * RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] + RB1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] = RE1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dEy + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEy;
    }
}


/////////////////////////////////////////////////////////
// Magnetic field PML updates - 2nd order - xplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_2order_magnetic_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, $REAL *Hy, $REAL *Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hy and Hz field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dEy, dEz;
    $REAL dx = d;
    int ii, jj, kk, materialHy, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,i1)];
        RB0 = HRB[INDEX2D_R(0,i1)];
        RE0 = HRE[INDEX2D_R(0,i1)];
        RF0 = HRF[INDEX2D_R(0,i1)];
        RA1 = HRA[INDEX2D_R(1,i1)];
        RB1 = HRB[INDEX2D_R(1,i1)];
        RE1 = HRE[INDEX2D_R(1,i1)];
        RF1 = HRF[INDEX2D_R(1,i1)];
        RA01 = HRA[INDEX2D_R(0,i1)] * HRA[INDEX2D_R(1,i1)] - 1;

        // Hy
        materialHy = ID[INDEX4D_ID(4,ii,jj,kk)];
        dEz = (Ez[INDEX3D_FIELDS(ii+1,jj,kk)] - Ez[INDEX3D_FIELDS(ii,jj,kk)]) / dx;
        Hy[INDEX3D_FIELDS(ii,jj,kk)] = Hy[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHy,4)] * (RA01 * dEz + RA1 * RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] + RB1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] = RE1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dEz + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,i2)];
        RB0 = HRB[INDEX2D_R(0,i2)];
        RE0 = HRE[INDEX2D_R(0,i2)];
        RF0 = HRF[INDEX2D_R(0,i2)];
        RA1 = HRA[INDEX2D_R(1,i2)];
        RB1 = HRB[INDEX2D_R(1,i2)];
        RE1 = HRE[INDEX2D_R(1,i2)];
        RF1 = HRF[INDEX2D_R(1,i2)];
        RA01 = HRA[INDEX2D_R(0,i2)] * HRA[INDEX2D_R(1,i2)] - 1;

        // Hz
        materialHz = ID[INDEX4D_ID(5,ii,jj,kk)];
        dEy = (Ey[INDEX3D_FIELDS(ii+1,jj,kk)] - Ey[INDEX3D_FIELDS(ii,jj,kk)]) / dx;
        Hz[INDEX3D_FIELDS(ii,jj,kk)] = Hz[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHz,4)] * (RA01 * dEy + RA1 * RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] + RB1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] = RE1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dEy + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEy;
    }
}


//////////////////////////////////////////////////////////
// Magnetic field PML updates - 2nd order - yminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_2order_magnetic_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, $REAL *Hx, const $REAL* __restrict__ Hy, $REAL *Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hx and Hz field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dEx, dEz;
    $REAL dy = d;
    int ii, jj, kk, materialHx, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - (j1 + 1);
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,j1)];
        RB0 = HRB[INDEX2D_R(0,j1)];
        RE0 = HRE[INDEX2D_R(0,j1)];
        RF0 = HRF[INDEX2D_R(0,j1)];
        RA1 = HRA[INDEX2D_R(1,j1)];
        RB1 = HRB[INDEX2D_R(1,j1)];
        RE1 = HRE[INDEX2D_R(1,j1)];
        RF1 = HRF[INDEX2D_R(1,j1)];
        RA01 = HRA[INDEX2D_R(0,j1)] * HRA[INDEX2D_R(1,j1)] - 1;

        // Hx
        materialHx = ID[INDEX4D_ID(3,ii,jj,kk)];
        dEz = (Ez[INDEX3D_FIELDS(ii,jj+1,kk)] - Ez[INDEX3D_FIELDS(ii,jj,kk)]) / dy;
        Hx[INDEX3D_FIELDS(ii,jj,kk)] = Hx[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHx,4)] * (RA01 * dEz + RA1 * RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] + RB1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] = RE1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dEz + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - (j2 + 1);
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,j2)];
        RB0 = HRB[INDEX2D_R(0,j2)];
        RE0 = HRE[INDEX2D_R(0,j2)];
        RF0 = HRF[INDEX2D_R(0,j2)];
        RA1 = HRA[INDEX2D_R(1,j2)];
        RB1 = HRB[INDEX2D_R(1,j2)];
        RE1 = HRE[INDEX2D_R(1,j2)];
        RF1 = HRF[INDEX2D_R(1,j2)];
        RA01 = HRA[INDEX2D_R(0,j2)] * HRA[INDEX2D_R(1,j2)] - 1;

        // Hz
        materialHz = ID[INDEX4D_ID(5,ii,jj,kk)];
        dEx = (Ex[INDEX3D_FIELDS(ii,jj+1,kk)] - Ex[INDEX3D_FIELDS(ii,jj,kk)]) / dy;
        Hz[INDEX3D_FIELDS(ii,jj,kk)] = Hz[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHz,4)] * (RA01 * dEx + RA1 * RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] + RB1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] = RE1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dEx + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEx;
    }
}


/////////////////////////////////////////////////////////
// Magnetic field PML updates - 2nd order - yplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_2order_magnetic_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, $REAL *Hx, const $REAL* __restrict__ Hy, $REAL *Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hx and Hz field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dEx, dEz;
    $REAL dy = d;
    int ii, jj, kk, materialHx, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,j1)];
        RB0 = HRB[INDEX2D_R(0,j1)];
        RE0 = HRE[INDEX2D_R(0,j1)];
        RF0 = HRF[INDEX2D_R(0,j1)];
        RA1 = HRA[INDEX2D_R(1,j1)];
        RB1 = HRB[INDEX2D_R(1,j1)];
        RE1 = HRE[INDEX2D_R(1,j1)];
        RF1 = HRF[INDEX2D_R(1,j1)];
        RA01 = HRA[INDEX2D_R(0,j1)] * HRA[INDEX2D_R(1,j1)] - 1;

        // Hx
        materialHx = ID[INDEX4D_ID(3,ii,jj,kk)];
        dEz = (Ez[INDEX3D_FIELDS(ii,jj+1,kk)] - Ez[INDEX3D_FIELDS(ii,jj,kk)]) / dy;
        Hx[INDEX3D_FIELDS(ii,jj,kk)] = Hx[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHx,4)] * (RA01 * dEz + RA1 * RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] + RB1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] = RE1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dEz + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,j2)];
        RB0 = HRB[INDEX2D_R(0,j2)];
        RE0 = HRE[INDEX2D_R(0,j2)];
        RF0 = HRF[INDEX2D_R(0,j2)];
        RA1 = HRA[INDEX2D_R(1,j2)];
        RB1 = HRB[INDEX2D_R(1,j2)];
        RE1 = HRE[INDEX2D_R(1,j2)];
        RF1 = HRF[INDEX2D_R(1,j2)];
        RA01 = HRA[INDEX2D_R(0,j2)] * HRA[INDEX2D_R(1,j2)] - 1;

        // Hz
        materialHz = ID[INDEX4D_ID(5,ii,jj,kk)];
        dEx = (Ex[INDEX3D_FIELDS(ii,jj+1,kk)] - Ex[INDEX3D_FIELDS(ii,jj,kk)]) / dy;
        Hz[INDEX3D_FIELDS(ii,jj,kk)] = Hz[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHz,4)] * (RA01 * dEx + RA1 * RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] + RB1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] = RE1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dEx + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEx;
    }
}


//////////////////////////////////////////////////////////
// Magnetic field PML updates - 2nd order - zminus slab //
//////////////////////////////////////////////////////////

__global__ void update_pml_2order_magnetic_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, $REAL *Hx, $REAL *Hy, const $REAL* __restrict__ Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hx and Hy field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dEx, dEy;
    $REAL dz = d;
    int ii, jj, kk, materialHx, materialHy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - (k1 + 1);

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,k1)];
        RB0 = HRB[INDEX2D_R(0,k1)];
        RE0 = HRE[INDEX2D_R(0,k1)];
        RF0 = HRF[INDEX2D_R(0,k1)];
        RA1 = HRA[INDEX2D_R(1,k1)];
        RB1 = HRB[INDEX2D_R(1,k1)];
        RE1 = HRE[INDEX2D_R(1,k1)];
        RF1 = HRF[INDEX2D_R(1,k1)];
        RA01 = HRA[INDEX2D_R(0,k1)] * HRA[INDEX2D_R(1,k1)] - 1;

        // Hx
        materialHx = ID[INDEX4D_ID(3,ii,jj,kk)];
        dEy = (Ey[INDEX3D_FIELDS(ii,jj,kk+1)] - Ey[INDEX3D_FIELDS(ii,jj,kk)]) / dz;
        Hx[INDEX3D_FIELDS(ii,jj,kk)] = Hx[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHx,4)] * (RA01 * dEy + RA1 * RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] + RB1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] = RE1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dEy + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - (k2 + 1);

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,k2)];
        RB0 = HRB[INDEX2D_R(0,k2)];
        RE0 = HRE[INDEX2D_R(0,k2)];
        RF0 = HRF[INDEX2D_R(0,k2)];
        RA1 = HRA[INDEX2D_R(1,k2)];
        RB1 = HRB[INDEX2D_R(1,k2)];
        RE1 = HRE[INDEX2D_R(1,k2)];
        RF1 = HRF[INDEX2D_R(1,k2)];
        RA01 = HRA[INDEX2D_R(0,k2)] * HRA[INDEX2D_R(1,k2)] - 1;

        // Hy
        materialHy = ID[INDEX4D_ID(4,ii,jj,kk)];
        dEx = (Ex[INDEX3D_FIELDS(ii,jj,kk+1)] - Ex[INDEX3D_FIELDS(ii,jj,kk)]) / dz;
        Hy[INDEX3D_FIELDS(ii,jj,kk)] = Hy[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHy,4)] * (RA01 * dEx + RA1 * RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] + RB1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] = RE1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dEx + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEx;
    }
}


/////////////////////////////////////////////////////////
// Magnetic field PML updates - 2nd order - zplus slab //
/////////////////////////////////////////////////////////

__global__ void update_pml_2order_magnetic_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_HPHI1, int NY_HPHI1, int NZ_HPHI1, int NX_HPHI2, int NY_HPHI2, int NZ_HPHI2, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, $REAL *Hx, $REAL *Hy, const $REAL* __restrict__ Hz, $REAL *HPhi1, $REAL *HPhi2, const $REAL* __restrict__ HRA, const $REAL* __restrict__ HRB, const $REAL* __restrict__ HRE, const $REAL* __restrict__ HRF, $REAL d) {

    //  This function updates the Hx and Hy field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_HPHI, NY_HPHI, NZ_HPHI: Dimensions of HPhi1 and HPhi2 PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      HPhi, HRA, HRB, HRE, HRF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML HPhi1 (4D) arrays
    int p1 = idx / (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1);
    int i1 = (idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) / (NY_HPHI1 * NZ_HPHI1);
    int j1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) / NZ_HPHI1;
    int k1 = ((idx % (NX_HPHI1 * NY_HPHI1 * NZ_HPHI1)) % (NY_HPHI1 * NZ_HPHI1)) % NZ_HPHI1;

    // Convert the linear index to subscripts for PML HPhi2 (4D) arrays
    int p2 = idx / (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2);
    int i2 = (idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) / (NY_HPHI2 * NZ_HPHI2);
    int j2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) / NZ_HPHI2;
    int k2 = ((idx % (NX_HPHI2 * NY_HPHI2 * NZ_HPHI2)) % (NY_HPHI2 * NZ_HPHI2)) % NZ_HPHI2;

    $REAL RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dEx, dEy;
    $REAL dz = d;
    int ii, jj, kk, materialHx, materialHy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,k1)];
        RB0 = HRB[INDEX2D_R(0,k1)];
        RE0 = HRE[INDEX2D_R(0,k1)];
        RF0 = HRF[INDEX2D_R(0,k1)];
        RA1 = HRA[INDEX2D_R(1,k1)];
        RB1 = HRB[INDEX2D_R(1,k1)];
        RE1 = HRE[INDEX2D_R(1,k1)];
        RF1 = HRF[INDEX2D_R(1,k1)];
        RA01 = HRA[INDEX2D_R(0,k1)] * HRA[INDEX2D_R(1,k1)] - 1;

        // Hx
        materialHx = ID[INDEX4D_ID(3,ii,jj,kk)];
        dEy = (Ey[INDEX3D_FIELDS(ii,jj,kk+1)] - Ey[INDEX3D_FIELDS(ii,jj,kk)]) / dz;
        Hx[INDEX3D_FIELDS(ii,jj,kk)] = Hx[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[INDEX2D_MAT(materialHx,4)] * (RA01 * dEy + RA1 * RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] + RB1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] = RE1 * HPhi1[INDEX4D_HPHI1(1,i1,j1,k1)] - RF1 * (RA0 * dEy + RB0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)]);
        HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] = RE0 * HPhi1[INDEX4D_HPHI1(0,i1,j1,k1)] - RF0 * dEy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = HRA[INDEX2D_R(0,k2)];
        RB0 = HRB[INDEX2D_R(0,k2)];
        RE0 = HRE[INDEX2D_R(0,k2)];
        RF0 = HRF[INDEX2D_R(0,k2)];
        RA1 = HRA[INDEX2D_R(1,k2)];
        RB1 = HRB[INDEX2D_R(1,k2)];
        RE1 = HRE[INDEX2D_R(1,k2)];
        RF1 = HRF[INDEX2D_R(1,k2)];
        RA01 = HRA[INDEX2D_R(0,k2)] * HRA[INDEX2D_R(1,k2)] - 1;

        // Hy
        materialHy = ID[INDEX4D_ID(4,ii,jj,kk)];
        dEx = (Ex[INDEX3D_FIELDS(ii,jj,kk+1)] - Ex[INDEX3D_FIELDS(ii,jj,kk)]) / dz;
        Hy[INDEX3D_FIELDS(ii,jj,kk)] = Hy[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[INDEX2D_MAT(materialHy,4)] * (RA01 * dEx + RA1 * RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] + RB1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)]);
        HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] = RE1 * HPhi2[INDEX4D_HPHI2(1,i2,j2,k2)] - RF1 * (RA0 * dEx + RB0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)]);        HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] = RE0 * HPhi2[INDEX4D_HPHI2(0,i2,j2,k2)] - RF0 * dEx;
    }
}

""")
