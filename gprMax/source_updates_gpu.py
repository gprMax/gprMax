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

kernels_template_sources = Template("""

// Macros for converting subscripts to linear index:
#define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
#define INDEX2D_SRCINFO(m, n) (m)*$NY_SRCINFO+(n)
#define INDEX2D_SRCWAVES(m, n) (m)*($NY_SRCWAVES)+(n)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
#define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)

// Material coefficients (read-only) in constant memory (64KB)
__device__ __constant__ $REAL updatecoeffsE[$N_updatecoeffsE];
__device__ __constant__ $REAL updatecoeffsH[$N_updatecoeffsH];

///////////////////////////////////////////
// Hertzian dipole electric field update //
///////////////////////////////////////////

__global__ void update_hertzian_dipole(int NHERTZDIPOLE, int iteration, $REAL dx, $REAL dy, $REAL dz, const int* __restrict__ srcinfo1, const $REAL* __restrict__ srcinfo2, const $REAL* __restrict__ srcwaveforms, const unsigned int* __restrict__ ID, $REAL *Ex, $REAL *Ey, $REAL *Ez) {

    //  This function updates electric field values for Hertzian dipole sources.
    //
    //  Args:
    //      NHERTZDIPOLE: Total number of Hertzian dipoles in the model
    //      iteration: Iteration number of simulation
    //      dx, dy, dz: Spatial discretisations
    //      srcinfo1: Source cell coordinates and polarisation information
    //      srcinfo2: Other source information, e.g. length, resistance etc...
    //      srcwaveforms: Source waveform values
    //      ID, E: Access to ID and field component arrays

    // Obtain the linear index corresponding to the current thread and use for each receiver
    int src = blockIdx.x * blockDim.x + threadIdx.x;

    if (src < NHERTZDIPOLE) {

        $REAL dl;
        int i, j, k, polarisation;

        i = srcinfo1[INDEX2D_SRCINFO(src,0)];
        j = srcinfo1[INDEX2D_SRCINFO(src,1)];
        k = srcinfo1[INDEX2D_SRCINFO(src,2)];
        polarisation = srcinfo1[INDEX2D_SRCINFO(src,3)];
        dl = srcinfo2[src];

        // 'x' polarised source
        if (polarisation == 0) {
            int materialEx = ID[INDEX4D_ID(0,i,j,k)];
            Ex[INDEX3D_FIELDS(i,j,k)] = Ex[INDEX3D_FIELDS(i,j,k)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * dl * (1 / (dx * dy * dz));
        }

        // 'y' polarised source
        else if (polarisation == 1) {
            int materialEy = ID[INDEX4D_ID(1,i,j,k)];
            Ey[INDEX3D_FIELDS(i,j,k)] = Ey[INDEX3D_FIELDS(i,j,k)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * dl * (1 / (dx * dy * dz));
        }

        // 'z' polarised source
        else if (polarisation == 2) {
            int materialEz = ID[INDEX4D_ID(2,i,j,k)];
            Ez[INDEX3D_FIELDS(i,j,k)] = Ez[INDEX3D_FIELDS(i,j,k)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * dl * (1 / (dx * dy * dz));
        }
    }
}


///////////////////////////////////////////
// Magnetic dipole magnetic field update //
///////////////////////////////////////////

__global__ void update_magnetic_dipole(int NMAGDIPOLE, int iteration, $REAL dx, $REAL dy, $REAL dz, const int* __restrict__ srcinfo1, const $REAL* __restrict__ srcinfo2, const $REAL* __restrict__ srcwaveforms, const unsigned int* __restrict__ ID, $REAL *Hx, $REAL *Hy, $REAL *Hz) {

    //  This function updates magnetic field values for magnetic dipole sources.
    //
    //  Args:
    //      NMAGDIPOLE: Total number of magnetic dipoles in the model
    //      iteration: Iteration number of simulation
    //      dx, dy, dz: Spatial discretisations
    //      srcinfo1: Source cell coordinates and polarisation information
    //      srcinfo2: Other source information, e.g. length, resistance etc...
    //      srcwaveforms: Source waveform values
    //      ID, H: Access to ID and field component arrays

    // Obtain the linear index corresponding to the current thread and use for each receiver
    int src = blockIdx.x * blockDim.x + threadIdx.x;

    if (src < NMAGDIPOLE) {

        int i, j, k, polarisation;

        i = srcinfo1[INDEX2D_SRCINFO(src,0)];
        j = srcinfo1[INDEX2D_SRCINFO(src,1)];
        k = srcinfo1[INDEX2D_SRCINFO(src,2)];
        polarisation = srcinfo1[INDEX2D_SRCINFO(src,3)];

        // 'x' polarised source
        if (polarisation == 0) {
            int materialHx = ID[INDEX4D_ID(3,i,j,k)];
            Hx[INDEX3D_FIELDS(i,j,k)] = Hx[INDEX3D_FIELDS(i,j,k)] - updatecoeffsH[INDEX2D_MAT(materialHx,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * (1 / (dx * dy * dz));
        }

        // 'y' polarised source
        else if (polarisation == 1) {
            int materialHy = ID[INDEX4D_ID(4,i,j,k)];
            Hy[INDEX3D_FIELDS(i,j,k)] = Hy[INDEX3D_FIELDS(i,j,k)] - updatecoeffsH[INDEX2D_MAT(materialHy,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * (1 / (dx * dy * dz));
        }

        // 'z' polarised source
        else if (polarisation == 2) {
            int materialHz = ID[INDEX4D_ID(5,i,j,k)];
            Hz[INDEX3D_FIELDS(i,j,k)] = Hz[INDEX3D_FIELDS(i,j,k)] - updatecoeffsH[INDEX2D_MAT(materialHz,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * (1 / (dx * dy * dz));
        }
    }
}


//////////////////////////////////////////
// Voltage source electric field update //
//////////////////////////////////////////

__global__ void update_voltage_source(int NVOLTSRC, int iteration, $REAL dx, $REAL dy, $REAL dz, const int* __restrict__ srcinfo1, const $REAL* __restrict__ srcinfo2, const $REAL* __restrict__ srcwaveforms, const unsigned int* __restrict__ ID, $REAL *Ex, $REAL *Ey, $REAL *Ez) {

    //  This function updates electric field values for voltage sources.
    //
    //  Args:
    //      NVOLTSRC: Total number of voltage sources in the model
    //      iteration: Iteration number of simulation
    //      dx, dy, dz: Spatial discretisations
    //      srcinfo1: Source cell coordinates and polarisation information
    //      srcinfo2: Other source information, e.g. length, resistance etc...
    //      srcwaveforms: Source waveform values
    //      ID, E: Access to ID and field component arrays

    // Obtain the linear index corresponding to the current thread and use for each receiver
    int src = blockIdx.x * blockDim.x + threadIdx.x;

    if (src < NVOLTSRC) {

        $REAL resistance;
        int i, j, k, polarisation;

        i = srcinfo1[INDEX2D_SRCINFO(src,0)];
        j = srcinfo1[INDEX2D_SRCINFO(src,1)];
        k = srcinfo1[INDEX2D_SRCINFO(src,2)];
        polarisation = srcinfo1[INDEX2D_SRCINFO(src,3)];
        resistance = srcinfo2[src];

        // 'x' polarised source
        if (polarisation == 0) {
            if (resistance != 0) {
                int materialEx = ID[INDEX4D_ID(0,i,j,k)];
                Ex[INDEX3D_FIELDS(i,j,k)] = Ex[INDEX3D_FIELDS(i,j,k)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * (1 / (resistance * dy * dz));
            }
            else {
                Ex[INDEX3D_FIELDS(i,j,k)] = -1 * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] / dx;
            }
        }

        // 'y' polarised source
        else if (polarisation == 1) {
            if (resistance != 0) {
                int materialEy = ID[INDEX4D_ID(1,i,j,k)];
                Ey[INDEX3D_FIELDS(i,j,k)] = Ey[INDEX3D_FIELDS(i,j,k)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * (1 / (resistance * dx * dz));
            }
            else {
                Ey[INDEX3D_FIELDS(i,j,k)] = -1 * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] / dy;
            }
        }

        // 'z' polarised source
        else if (polarisation == 2) {
            if (resistance != 0) {
                int materialEz = ID[INDEX4D_ID(2,i,j,k)];
                Ez[INDEX3D_FIELDS(i,j,k)] = Ez[INDEX3D_FIELDS(i,j,k)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * (1 / (resistance * dx * dy));
            }
            else {
                Ez[INDEX3D_FIELDS(i,j,k)] = -1 * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] / dz;
            }
        }
    }
}


""")
