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

from string import Template

kernels_template_fields = Template("""

#include <pycuda-complex.hpp>

// Macros for converting subscripts to linear index:
#define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
#define INDEX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
#define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
#define INDEX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)

// Material coefficients (read-only) in constant memory (64KB)_
__device__ __constant__ $REAL updatecoeffsE[$N_updatecoeffsE];
__device__ __constant__ $REAL updatecoeffsH[$N_updatecoeffsH];

/////////////////////////////////////////////////
// Electric field updates - standard materials //
/////////////////////////////////////////////////

__global__ void update_e(int NX, int NY, int NZ, const unsigned int* __restrict__ ID, $REAL *Ex, $REAL *Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz) {

    //  This function updates electric field values.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain
    //      ID, E, H: Access to ID and field component arrays

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for 3D field arrays
    int i = idx / ($NY_FIELDS * $NZ_FIELDS);
    int j = (idx % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int k = (idx % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int i_ID = (idx % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int j_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int k_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Ex component
    if ((NY != 1 || NZ != 1) && i >= 0 && i < NX && j > 0 && j < NY && k > 0 && k < NZ) {
        int materialEx = ID[INDEX4D_ID(0,i_ID,j_ID,k_ID)];
        Ex[INDEX3D_FIELDS(i,j,k)] = updatecoeffsE[INDEX2D_MAT(materialEx,0)] * Ex[INDEX3D_FIELDS(i,j,k)] + updatecoeffsE[INDEX2D_MAT(materialEx,2)] * (Hz[INDEX3D_FIELDS(i,j,k)] - Hz[INDEX3D_FIELDS(i,j-1,k)]) - updatecoeffsE[INDEX2D_MAT(materialEx,3)] * (Hy[INDEX3D_FIELDS(i,j,k)] - Hy[INDEX3D_FIELDS(i,j,k-1)]);
    }

    // Ey component
    if ((NX != 1 || NZ != 1) && i > 0 && i < NX && j >= 0 && j < NY && k > 0 && k < NZ) {
        int materialEy = ID[INDEX4D_ID(1,i_ID,j_ID,k_ID)];
        Ey[INDEX3D_FIELDS(i,j,k)] = updatecoeffsE[INDEX2D_MAT(materialEy,0)] * Ey[INDEX3D_FIELDS(i,j,k)] + updatecoeffsE[INDEX2D_MAT(materialEy,3)] * (Hx[INDEX3D_FIELDS(i,j,k)] - Hx[INDEX3D_FIELDS(i,j,k-1)]) - updatecoeffsE[INDEX2D_MAT(materialEy,1)] * (Hz[INDEX3D_FIELDS(i,j,k)] - Hz[INDEX3D_FIELDS(i-1,j,k)]);
    }

    // Ez component
    if ((NX != 1 || NY != 1) && i > 0 && i < NX && j > 0 && j < NY && k >= 0 && k < NZ) {
        int materialEz = ID[INDEX4D_ID(2,i_ID,j_ID,k_ID)];
        Ez[INDEX3D_FIELDS(i,j,k)] = updatecoeffsE[INDEX2D_MAT(materialEz,0)] * Ez[INDEX3D_FIELDS(i,j,k)] + updatecoeffsE[INDEX2D_MAT(materialEz,1)] * (Hy[INDEX3D_FIELDS(i,j,k)] - Hy[INDEX3D_FIELDS(i-1,j,k)]) - updatecoeffsE[INDEX2D_MAT(materialEz,2)] * (Hx[INDEX3D_FIELDS(i,j,k)] - Hx[INDEX3D_FIELDS(i,j-1,k)]);
    }
}

///////////////////////////////////////////////////
// Electric field updates - dispersive materials //
///////////////////////////////////////////////////

__global__ void update_e_dispersive_A(int NX, int NY, int NZ, int MAXPOLES, const $COMPLEX* __restrict__ updatecoeffsdispersive, $COMPLEX *Tx, $COMPLEX *Ty, $COMPLEX *Tz, const unsigned int* __restrict__ ID, $REAL *Ex, $REAL *Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz) {

    //  This function is part A of updates to electric field values when dispersive materials (with multiple poles) are present.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain
    //      MAXPOLES: Maximum number of dispersive material poles present in model
    //      updatedispersivecoeffs, T, ID, E, H: Access to update coefficients, dispersive, ID and field component arrays

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for 3D field arrays
    int i = idx / ($NY_FIELDS * $NZ_FIELDS);
    int j = (idx % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int k = (idx % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int i_ID = (idx % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int j_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int k_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Convert the linear index to subscripts for 4D dispersive array
    int i_T = (idx % ($NX_T * $NY_T * $NZ_T)) / ($NY_T * $NZ_T);
    int j_T = ((idx % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) / $NZ_T;
    int k_T = ((idx % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) % $NZ_T;

    // Ex component
    if ((NY != 1 || NZ != 1) && i >= 0 && i < NX && j > 0 && j < NY && k > 0 && k < NZ) {
        int materialEx = ID[INDEX4D_ID(0,i_ID,j_ID,k_ID)];
        $REAL phi = 0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            phi = phi + updatecoeffsdispersive[INDEX2D_MATDISP(materialEx,pole*3)].real() * Tx[INDEX4D_T(pole,i_T,j_T,k_T)].real();
            Tx[INDEX4D_T(pole,i_T,j_T,k_T)] = updatecoeffsdispersive[INDEX2D_MATDISP(materialEx,1+(pole*3))] * Tx[INDEX4D_T(pole,i_T,j_T,k_T)] + updatecoeffsdispersive[INDEX2D_MATDISP(materialEx,2+(pole*3))] * Ex[INDEX3D_FIELDS(i,j,k)];
        }
        Ex[INDEX3D_FIELDS(i,j,k)] = updatecoeffsE[INDEX2D_MAT(materialEx,0)] * Ex[INDEX3D_FIELDS(i,j,k)] + updatecoeffsE[INDEX2D_MAT(materialEx,2)] * (Hz[INDEX3D_FIELDS(i,j,k)] - Hz[INDEX3D_FIELDS(i,j-1,k)]) - updatecoeffsE[INDEX2D_MAT(materialEx,3)] * (Hy[INDEX3D_FIELDS(i,j,k)] - Hy[INDEX3D_FIELDS(i,j,k-1)]) - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * phi;
    }

    // Ey component
    if ((NX != 1 || NZ != 1) && i > 0 && i < NX && j >= 0 && j < NY && k > 0 && k < NZ) {
        int materialEy = ID[INDEX4D_ID(1,i_ID,j_ID,k_ID)];
        $REAL phi = 0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            phi = phi + updatecoeffsdispersive[INDEX2D_MATDISP(materialEy,pole*3)].real() * Ty[INDEX4D_T(pole,i_T,j_T,k_T)].real();
            Ty[INDEX4D_T(pole,i_T,j_T,k_T)] = updatecoeffsdispersive[INDEX2D_MATDISP(materialEy,1+(pole*3))] * Ty[INDEX4D_T(pole,i_T,j_T,k_T)] + updatecoeffsdispersive[INDEX2D_MATDISP(materialEy,2+(pole*3))] * Ey[INDEX3D_FIELDS(i,j,k)];
        }
        Ey[INDEX3D_FIELDS(i,j,k)] = updatecoeffsE[INDEX2D_MAT(materialEy,0)] * Ey[INDEX3D_FIELDS(i,j,k)] + updatecoeffsE[INDEX2D_MAT(materialEy,3)] * (Hx[INDEX3D_FIELDS(i,j,k)] - Hx[INDEX3D_FIELDS(i,j,k-1)]) - updatecoeffsE[INDEX2D_MAT(materialEy,1)] * (Hz[INDEX3D_FIELDS(i,j,k)] - Hz[INDEX3D_FIELDS(i-1,j,k)]) - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * phi;
    }

    // Ez component
    if ((NX != 1 || NY != 1) && i > 0 && i < NX && j > 0 && j < NY && k >= 0 && k < NZ) {
        int materialEz = ID[INDEX4D_ID(2,i_ID,j_ID,k_ID)];
        $REAL phi = 0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            phi = phi + updatecoeffsdispersive[INDEX2D_MATDISP(materialEz,pole*3)].real() * Tz[INDEX4D_T(pole,i_T,j_T,k_T)].real();
            Tz[INDEX4D_T(pole,i_T,j_T,k_T)] = updatecoeffsdispersive[INDEX2D_MATDISP(materialEz,1+(pole*3))] * Tz[INDEX4D_T(pole,i_T,j_T,k_T)] + updatecoeffsdispersive[INDEX2D_MATDISP(materialEz,2+(pole*3))] * Ez[INDEX3D_FIELDS(i,j,k)];
        }
        Ez[INDEX3D_FIELDS(i,j,k)] = updatecoeffsE[INDEX2D_MAT(materialEz,0)] * Ez[INDEX3D_FIELDS(i,j,k)] + updatecoeffsE[INDEX2D_MAT(materialEz,1)] * (Hy[INDEX3D_FIELDS(i,j,k)] - Hy[INDEX3D_FIELDS(i-1,j,k)]) - updatecoeffsE[INDEX2D_MAT(materialEz,2)] * (Hx[INDEX3D_FIELDS(i,j,k)] - Hx[INDEX3D_FIELDS(i,j-1,k)]) - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * phi;
    }
}

__global__ void update_e_dispersive_B(int NX, int NY, int NZ, int MAXPOLES, const $COMPLEX* __restrict__ updatecoeffsdispersive, $COMPLEX *Tx, $COMPLEX *Ty, $COMPLEX *Tz, const unsigned int* __restrict__ ID, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez) {

    //  This function is part B which updates the dispersive field arrays when dispersive materials (with multiple poles) are present.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain
    //      MAXPOLES: Maximum number of dispersive material poles present in model
    //      updatedispersivecoeffs, T, ID, E, H: Access to update coefficients, dispersive, ID and field component arrays

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for 3D field arrays
    int i = idx / ($NY_FIELDS * $NZ_FIELDS);
    int j = (idx % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int k = (idx % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int i_ID = (idx % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int j_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int k_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Convert the linear index to subscripts for 4D dispersive array
    int i_T = (idx % ($NX_T * $NY_T * $NZ_T)) / ($NY_T * $NZ_T);
    int j_T = ((idx % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) / $NZ_T;
    int k_T = ((idx % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) % $NZ_T;

    // Ex component
    if ((NY != 1 || NZ != 1) && i >= 0 && i < NX && j > 0 && j < NY && k > 0 && k < NZ) {
        int materialEx = ID[INDEX4D_ID(0,i_ID,j_ID,k_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            Tx[INDEX4D_T(pole,i_T,j_T,k_T)] = Tx[INDEX4D_T(pole,i_T,j_T,k_T)] - updatecoeffsdispersive[INDEX2D_MATDISP(materialEx,2+(pole*3))] * Ex[INDEX3D_FIELDS(i,j,k)];
        }
    }

    // Ey component
    if ((NX != 1 || NZ != 1) && i > 0 && i < NX && j >= 0 && j < NY && k > 0 && k < NZ) {
        int materialEy = ID[INDEX4D_ID(1,i_ID,j_ID,k_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            Ty[INDEX4D_T(pole,i_T,j_T,k_T)] = Ty[INDEX4D_T(pole,i_T,j_T,k_T)] - updatecoeffsdispersive[INDEX2D_MATDISP(materialEy,2+(pole*3))] * Ey[INDEX3D_FIELDS(i,j,k)];
        }
    }

    // Ez component
    if ((NX != 1 || NY != 1) && i > 0 && i < NX && j > 0 && j < NY && k >= 0 && k < NZ) {
        int materialEz = ID[INDEX4D_ID(2,i_ID,j_ID,k_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            Tz[INDEX4D_T(pole,i_T,j_T,k_T)] = Tz[INDEX4D_T(pole,i_T,j_T,k_T)] - updatecoeffsdispersive[INDEX2D_MATDISP(materialEz,2+(pole*3))] * Ez[INDEX3D_FIELDS(i,j,k)];
        }
    }
}


////////////////////////////
// Magnetic field updates //
////////////////////////////

__global__ void update_h(int NX, int NY, int NZ, const unsigned int* __restrict__ ID, $REAL *Hx, $REAL *Hy, $REAL *Hz, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez) {

    //  This function updates magnetic field values.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain
    //      ID, E, H: Access to ID and field component arrays

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for 3D field arrays
    int i = idx / ($NY_FIELDS * $NZ_FIELDS);
    int j = (idx % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int k = (idx % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int i_ID = (idx % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int j_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int k_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Hx component
    if (NX != 1 && i > 0 && i < NX && j >= 0 && j < NY && k >= 0 && k < NZ) {
        int materialHx = ID[INDEX4D_ID(3,i_ID,j_ID,k_ID)];
        Hx[INDEX3D_FIELDS(i,j,k)] = updatecoeffsH[INDEX2D_MAT(materialHx,0)] * Hx[INDEX3D_FIELDS(i,j,k)] - updatecoeffsH[INDEX2D_MAT(materialHx,2)] * (Ez[INDEX3D_FIELDS(i,j+1,k)] - Ez[INDEX3D_FIELDS(i,j,k)]) + updatecoeffsH[INDEX2D_MAT(materialHx,3)] * (Ey[INDEX3D_FIELDS(i,j,k+1)] - Ey[INDEX3D_FIELDS(i,j,k)]);
    }

    // Hy component
    if (NY != 1 && i >= 0 && i < NX && j > 0 && j < NY && k >= 0 && k < NZ) {
        int materialHy = ID[INDEX4D_ID(4,i_ID,j_ID,k_ID)];
        Hy[INDEX3D_FIELDS(i,j,k)] = updatecoeffsH[INDEX2D_MAT(materialHy,0)] * Hy[INDEX3D_FIELDS(i,j,k)] - updatecoeffsH[INDEX2D_MAT(materialHy,3)] * (Ex[INDEX3D_FIELDS(i,j,k+1)] - Ex[INDEX3D_FIELDS(i,j,k)]) + updatecoeffsH[INDEX2D_MAT(materialHy,1)] * (Ez[INDEX3D_FIELDS(i+1,j,k)] - Ez[INDEX3D_FIELDS(i,j,k)]);
    }

    // Hz component
    if (NZ != 1 && i >= 0 && i < NX && j >= 0 && j < NY && k > 0 && k < NZ) {
        int materialHz = ID[INDEX4D_ID(5,i_ID,j_ID,k_ID)];
        Hz[INDEX3D_FIELDS(i,j,k)] = updatecoeffsH[INDEX2D_MAT(materialHz,0)] * Hz[INDEX3D_FIELDS(i,j,k)] - updatecoeffsH[INDEX2D_MAT(materialHz,1)] * (Ey[INDEX3D_FIELDS(i+1,j,k)] - Ey[INDEX3D_FIELDS(i,j,k)]) + updatecoeffsH[INDEX2D_MAT(materialHz,2)] * (Ex[INDEX3D_FIELDS(i,j+1,k)] - Ex[INDEX3D_FIELDS(i,j,k)]);
    }
}

""")
