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

update_electric = {
    'args_cuda': Template("""
                __global__ void update_electric(int NX,
                                                int NY,
                                                int NZ, 
                                                const unsigned int* __restrict__ ID, 
                                                $REAL *Ex, 
                                                $REAL *Ey, 
                                                $REAL *Ez, 
                                                const $REAL* __restrict__ Hx, 
                                                const $REAL* __restrict__ Hy, 
                                                const $REAL* __restrict__ Hz)
                    """),
    'args_opencl': Template("""
                        int NX,
                        int NY,
                        int NZ,
                        __global const unsigned int* restrict ID,
                        __global $REAL *Ex,
                        __global $REAL *Ey,
                        __global $REAL *Ez,
                        __global const $REAL * restrict Hx,
                        __global const $REAL * restrict Hy,
                        __global const $REAL * restrict Hz
                    """),
    'func': Template("""
    // Electric field updates - normal materials.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain.
    //      ID, E, H: Access to ID and field component arrays.

    $CUDA_IDX

    // Convert the linear index to subscripts for 3D field arrays
    int x = i / ($NY_FIELDS * $NZ_FIELDS);
    int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Ex component
    if ((NY != 1 || NZ != 1) && x >= 0 && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
        int materialEx = ID[IDX4D_ID(0,x_ID,y_ID,z_ID)];
        Ex[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEx,0)] * Ex[IDX3D_FIELDS(x,y,z)] + 
                                    updatecoeffsE[IDX2D_MAT(materialEx,2)] * (Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x,y-1,z)]) - 
                                updatecoeffsE[IDX2D_MAT(materialEx,3)] * (Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x,y,z-1)]);
    }

    // Ey component
    if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
        int materialEy = ID[IDX4D_ID(1,x_ID,y_ID,z_ID)];
        Ey[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEy,0)] * Ey[IDX3D_FIELDS(x,y,z)] + 
                                    updatecoeffsE[IDX2D_MAT(materialEy,3)] * (Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y,z-1)]) - 
                                    updatecoeffsE[IDX2D_MAT(materialEy,1)] * (Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x-1,y,z)]);
    }

    // Ez component
    if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
        int materialEz = ID[IDX4D_ID(2,x_ID,y_ID,z_ID)];
        Ez[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEz,0)] * Ez[IDX3D_FIELDS(x,y,z)] + 
                                    updatecoeffsE[IDX2D_MAT(materialEz,1)] * (Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x-1,y,z)]) - 
                                    updatecoeffsE[IDX2D_MAT(materialEz,2)] * (Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y-1,z)]);
    }
    """)}

update_magnetic = {
    'args_cuda': Template("""
                __global__ void update_magnetic(int NX,
                                                int NY,
                                                int NZ, 
                                                const unsigned int* __restrict__ ID, 
                                                $REAL *Hx, 
                                                $REAL *Hy, 
                                                $REAL *Hz, 
                                                const $REAL* __restrict__ Ex, 
                                                const $REAL* __restrict__ Ey, 
                                                const $REAL* __restrict__ Ez)
                    """),
    'args_opencl': Template("""
                        int NX,
                        int NY,
                        int NZ,
                        __global const unsigned int* restrict ID,
                        __global $REAL *Hx,
                        __global $REAL *Hy,
                        __global $REAL *Hz,
                        __global const $REAL * restrict Ex,
                        __global const $REAL * restrict Ey,
                        __global const $REAL * restrict Ez
                    """),
    'func': Template("""
    // Magnetic field updates - normal materials.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain.
    //      ID, E, H: Access to ID and field component arrays.

    $CUDA_IDX
    
    // Convert the linear index to subscripts for 3D field arrays
    int x = i / ($NY_FIELDS * $NZ_FIELDS);
    int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Hx component
    if (NX != 1 && x > 0 && x < NX && y >= 0 && y < NY && z >= 0 && z < NZ) {
        int materialHx = ID[IDX4D_ID(3,x_ID,y_ID,z_ID)];
        Hx[IDX3D_FIELDS(x,y,z)] = updatecoeffsH[IDX2D_MAT(materialHx,0)] * Hx[IDX3D_FIELDS(x,y,z)] - 
                                    updatecoeffsH[IDX2D_MAT(materialHx,2)] * (Ez[IDX3D_FIELDS(x,y+1,z)] - Ez[IDX3D_FIELDS(x,y,z)]) + 
                                    updatecoeffsH[IDX2D_MAT(materialHx,3)] * (Ey[IDX3D_FIELDS(x,y,z+1)] - Ey[IDX3D_FIELDS(x,y,z)]);
    }

    // Hy component
    if (NY != 1 && x >= 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
        int materialHy = ID[IDX4D_ID(4,x_ID,y_ID,z_ID)];
        Hy[IDX3D_FIELDS(x,y,z)] = updatecoeffsH[IDX2D_MAT(materialHy,0)] * Hy[IDX3D_FIELDS(x,y,z)] - 
                                    updatecoeffsH[IDX2D_MAT(materialHy,3)] * (Ex[IDX3D_FIELDS(x,y,z+1)] - Ex[IDX3D_FIELDS(x,y,z)]) + 
                                    updatecoeffsH[IDX2D_MAT(materialHy,1)] * (Ez[IDX3D_FIELDS(x+1,y,z)] - Ez[IDX3D_FIELDS(x,y,z)]);
    }

    // Hz component
    if (NZ != 1 && x >= 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
        int materialHz = ID[IDX4D_ID(5,x_ID,y_ID,z_ID)];
        Hz[IDX3D_FIELDS(x,y,z)] = updatecoeffsH[IDX2D_MAT(materialHz,0)] * Hz[IDX3D_FIELDS(x,y,z)] - 
                                    updatecoeffsH[IDX2D_MAT(materialHz,1)] * (Ey[IDX3D_FIELDS(x+1,y,z)] - Ey[IDX3D_FIELDS(x,y,z)]) + 
                                    updatecoeffsH[IDX2D_MAT(materialHz,2)] * (Ex[IDX3D_FIELDS(x,y+1,z)] - Ex[IDX3D_FIELDS(x,y,z)]);
    }
    """)}

update_electric_dispersive_A = {
    'args_cuda': Template("""
                __global__ void update_electric_dispersive_A(int NX,
                                    int NY,
                                    int NZ,
                                    int MAXPOLES, 
                                    const $COMPLEX* __restrict__ updatecoeffsdispersive, 
                                    $COMPLEX *Tx, 
                                    $COMPLEX *Ty, 
                                    $COMPLEX *Tz, 
                                    const unsigned int* __restrict__ ID, 
                                    $REAL *Ex, 
                                    $REAL *Ey, 
                                    $REAL *Ez, 
                                    const $REAL* __restrict__ Hx, 
                                    const $REAL* __restrict__ Hy, 
                                    const $REAL* __restrict__ Hz)
                    """),
    'args_opencl': Template("""
                        int NX,
                        int NY,
                        int NZ,
                        int MAXPOLES,
                        __global const $COMPLEX* restrict updatecoeffsdispersive,
                        __global $COMPLEX *Tx,
                        __global $COMPLEX *Ty,
                        __global $COMPLEX *Tz,
                        __global const unsigned int* restrict ID,
                        __global $REAL *Ex,
                        __global $REAL *Ey,
                        __global $REAL *Ez,
                        __global const $REAL* restrict Hx,
                        __global const $REAL* restrict Hy,
                        __global const $REAL* restrict Hz
                    """),
    'func': Template("""
    // Electric field updates - dispersive materials - part A of updates to electric
    //                              field values when dispersive materials 
    //                              (with multiple poles) are present.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain.
    //      MAXPOLES: Maximum number of dispersive material poles present in model.
    //      updatedispersivecoeffs, T, ID, E, H: Access to update coefficients, 
    //                                              dispersive, ID and field 
    //                                              component arrays.


    // Convert the linear index to subscripts for 3D field arrays
    int x = i / ($NY_FIELDS * $NZ_FIELDS);
    int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Convert the linear index to subscripts for 4D dispersive array
    int x_T = (i % ($NX_T * $NY_T * $NZ_T)) / ($NY_T * $NZ_T);
    int y_T = ((i % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) / $NZ_T;
    int z_T = ((i % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) % $NZ_T;

    $CUDA_IDX

    // Ex component
    if ((NY != 1 || NZ != 1) && x >= 0 && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
        int materialEx = ID[IDX4D_ID(0,x_ID,y_ID,z_ID)];
        $REAL phi = 0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            phi = phi + updatecoeffsdispersive[IDX2D_MATDISP(materialEx,pole*3)]$REALFUNC * Tx[IDX4D_T(pole,x_T,y_T,z_T)]$REALFUNC;
            Tx[IDX4D_T(pole,x_T,y_T,z_T)] = updatecoeffsdispersive[IDX2D_MATDISP(materialEx,1+(pole*3))] * Tx[IDX4D_T(pole,x_T,y_T,z_T)] + 
                                            updatecoeffsdispersive[IDX2D_MATDISP(materialEx,2+(pole*3))] * Ex[IDX3D_FIELDS(x,y,z)];
        }
        Ex[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEx,0)] * Ex[IDX3D_FIELDS(x,y,z)] + 
                                    updatecoeffsE[IDX2D_MAT(materialEx,2)] * (Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x,y-1,z)]) - 
                                    updatecoeffsE[IDX2D_MAT(materialEx,3)] * (Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x,y,z-1)]) - 
                                    updatecoeffsE[IDX2D_MAT(materialEx,4)] * phi;
    }

    // Ey component
    if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
        int materialEy = ID[IDX4D_ID(1,x_ID,y_ID,z_ID)];
        $REAL phi = 0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            phi = phi + updatecoeffsdispersive[IDX2D_MATDISP(materialEy,pole*3)]$REALFUNC * Ty[IDX4D_T(pole,x_T,y_T,z_T)]$REALFUNC;
            Ty[IDX4D_T(pole,x_T,y_T,z_T)] = updatecoeffsdispersive[IDX2D_MATDISP(materialEy,1+(pole*3))] * Ty[IDX4D_T(pole,x_T,y_T,z_T)] + 
                                            updatecoeffsdispersive[IDX2D_MATDISP(materialEy,2+(pole*3))] * Ey[IDX3D_FIELDS(x,y,z)];
        }
        Ey[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEy,0)] * Ey[IDX3D_FIELDS(x,y,z)] + 
                                    updatecoeffsE[IDX2D_MAT(materialEy,3)] * (Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y,z-1)]) - 
                                    updatecoeffsE[IDX2D_MAT(materialEy,1)] * (Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x-1,y,z)]) - 
                                    updatecoeffsE[IDX2D_MAT(materialEy,4)] * phi;
    }

    // Ez component
    if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
        int materialEz = ID[IDX4D_ID(2,x_ID,y_ID,z_ID)];
        $REAL phi = 0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            phi = phi + updatecoeffsdispersive[IDX2D_MATDISP(materialEz,pole*3)]$REALFUNC * Tz[IDX4D_T(pole,x_T,y_T,z_T)]$REALFUNC;
            Tz[IDX4D_T(pole,x_T,y_T,z_T)] = updatecoeffsdispersive[IDX2D_MATDISP(materialEz,1+(pole*3))] * Tz[IDX4D_T(pole,x_T,y_T,z_T)] + 
                                            updatecoeffsdispersive[IDX2D_MATDISP(materialEz,2+(pole*3))] * Ez[IDX3D_FIELDS(x,y,z)];
        }
        Ez[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEz,0)] * Ez[IDX3D_FIELDS(x,y,z)] + 
                                    updatecoeffsE[IDX2D_MAT(materialEz,1)] * (Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x-1,y,z)]) - 
                                    updatecoeffsE[IDX2D_MAT(materialEz,2)] * (Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y-1,z)]) - 
                                    updatecoeffsE[IDX2D_MAT(materialEz,4)] * phi;
    }
    """)}

update_electric_dispersive_B = {
    'args_cuda': Template("""
                __global__ void update_electric_dispersive_B(int NX,
                                    int NY,
                                    int NZ,
                                    int MAXPOLES, 
                                    const $COMPLEX* __restrict__ updatecoeffsdispersive, 
                                    $COMPLEX *Tx, 
                                    $COMPLEX *Ty, 
                                    $COMPLEX *Tz, 
                                    const unsigned int* __restrict__ ID, 
                                    const $REAL* __restrict__ Ex, 
                                    const $REAL* __restrict__ Ey, 
                                    const $REAL* __restrict__ Ez)
                    """),
    'args_opencl': Template("""
                        int NX,
                        int NY,
                        int NZ,
                        int MAXPOLES,
                        __global const $COMPLEX* restrict updatecoeffsdispersive,
                        __global $COMPLEX *Tx,
                        __global $COMPLEX *Ty,
                        __global $COMPLEX *Tz,
                        __global const unsigned int* restrict ID,
                        __global const $REAL* restrict Ex,
                        __global const $REAL* restrict Ey,
                        __global const $REAL* restrict Ez
                    """),
    'func': Template("""
    // Electric field updates - dispersive materials - part B of updates to electric
    //                              field values when dispersive materials 
    //                              (with multiple poles) are present.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain.
    //      MAXPOLES: Maximum number of dispersive material poles present in model.
    //      updatedispersivecoeffs, T, ID, E, H: Access to update coefficients, 
    //                                              dispersive, ID and field 
    //                                              component arrays.


    // Convert the linear index to subscripts for 3D field arrays
    int x = i / ($NY_FIELDS * $NZ_FIELDS);
    int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Convert the linear index to subscripts for 4D dispersive array
    int x_T = (i % ($NX_T * $NY_T * $NZ_T)) / ($NY_T * $NZ_T);
    int y_T = ((i % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) / $NZ_T;
    int z_T = ((i % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) % $NZ_T;

    $CUDA_IDX
    
    // Ex component
    if ((NY != 1 || NZ != 1) && x >= 0 && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
        int materialEx = ID[IDX4D_ID(0,x_ID,y_ID,z_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            Tx[IDX4D_T(pole,x_T,y_T,z_T)] = Tx[IDX4D_T(pole,x_T,y_T,z_T)] - 
                                                updatecoeffsdispersive[IDX2D_MATDISP(materialEx,2+(pole*3))] * Ex[IDX3D_FIELDS(x,y,z)];
        }
    }

    // Ey component
    if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
        int materialEy = ID[IDX4D_ID(1,x_ID,y_ID,z_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            Ty[IDX4D_T(pole,x_T,y_T,z_T)] = Ty[IDX4D_T(pole,x_T,y_T,z_T)] - 
                                                updatecoeffsdispersive[IDX2D_MATDISP(materialEy,2+(pole*3))] * Ey[IDX3D_FIELDS(x,y,z)];
        }
    }

    // Ez component
    if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
        int materialEz = ID[IDX4D_ID(2,x_ID,y_ID,z_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            Tz[IDX4D_T(pole,x_T,y_T,z_T)] = Tz[IDX4D_T(pole,x_T,y_T,z_T)] - 
                                                updatecoeffsdispersive[IDX2D_MATDISP(materialEz,2+(pole*3))] * Ez[IDX3D_FIELDS(x,y,z)];
        }
    }
    """)}