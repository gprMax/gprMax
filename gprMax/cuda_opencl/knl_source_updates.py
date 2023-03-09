# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
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

update_hertzian_dipole = {'args_cuda': Template("""
                                        __global__ void update_hertzian_dipole(int NHERTZDIPOLE, 
                                            int iteration, 
                                            $REAL dx, 
                                            $REAL dy, 
                                            $REAL dz, 
                                            const int* __restrict__ srcinfo1, 
                                            const $REAL* __restrict__ srcinfo2, 
                                            const $REAL* __restrict__ srcwaveforms, 
                                            const unsigned int* __restrict__ ID, 
                                            $REAL *Ex, 
                                            $REAL *Ey, 
                                            $REAL *Ez)
                                        """),
                          'args_opencl': Template("""
                                            int NHERTZDIPOLE,
                                            int iteration,
                                            $REAL dx,
                                            $REAL dy,
                                            $REAL dz,
                                            __global const int* restrict srcinfo1,
                                            __global const $REAL* restrict srcinfo2,
                                            __global const $REAL* restrict srcwaveforms,
                                            __global const unsigned int* restrict ID,
                                            __global $REAL *Ex,
                                            __global $REAL *Ey,
                                            __global $REAL *Ez
                                        """),
                          'func': Template("""
    // Updates electric field values for Hertzian dipole sources.
    //
    //  Args:
    //      NHERTZDIPOLE: Total number of Hertzian dipoles in the model.
    //      iteration: Iteration number of simulation.
    //      dx, dy, dz: Spatial discretisations.
    //      srcinfo1: Source cell coordinates and polarisation information.
    //      srcinfo2: Other source information, e.g. length, resistance etc...
    //      srcwaveforms: Source waveform values.
    //      ID, E: Access to ID and field component arrays.

    $CUDA_IDX

    if (i < NHERTZDIPOLE) {

        $REAL dl;
        int x, y, z, polarisation;

        x = srcinfo1[IDX2D_SRCINFO(i,0)];
        y = srcinfo1[IDX2D_SRCINFO(i,1)];
        z = srcinfo1[IDX2D_SRCINFO(i,2)];
        polarisation = srcinfo1[IDX2D_SRCINFO(i,3)];
        dl = srcinfo2[i];

        // 'x' polarised source
        if (polarisation == 0) {
            int materialEx = ID[IDX4D_ID(0,x,y,z)];
            Ex[IDX3D_FIELDS(x,y,z)] = Ex[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                        srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * dl * (1 / (dx * dy * dz));
        }

        // 'y' polarised source
        else if (polarisation == 1) {
            int materialEy = ID[IDX4D_ID(1,x,y,z)];
            Ey[IDX3D_FIELDS(x,y,z)] = Ey[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                        srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * dl * (1 / (dx * dy * dz));
        }

        // 'z' polarised source
        else if (polarisation == 2) {
            int materialEz = ID[IDX4D_ID(2,x,y,z)];
            Ez[IDX3D_FIELDS(x,y,z)] = Ez[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                        srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * dl * (1 / (dx * dy * dz));
        }
    }
""")
}

update_magnetic_dipole = {'args_cuda': Template("""
                                        __global__ void update_magnetic_dipole(int NMAGDIPOLE, 
                                            int iteration, 
                                            $REAL dx, 
                                            $REAL dy, 
                                            $REAL dz, 
                                            const int* __restrict__ srcinfo1, 
                                            const $REAL* __restrict__ srcinfo2, 
                                            const $REAL* __restrict__ srcwaveforms, 
                                            const unsigned int* __restrict__ ID, 
                                            $REAL *Hx, 
                                            $REAL *Hy, 
                                            $REAL *Hz)
                                        """),
                          'args_opencl': Template("""
                                            int NMAGDIPOLE,
                                            int iteration,
                                            $REAL dx,
                                            $REAL dy,
                                            $REAL dz,
                                            __global const int* restrict rcinfo1,
                                            __global const $REAL* restrict rcinfo2,
                                            __global const $REAL* restrict rcwaveforms,
                                            __global const unsigned int* estrict ID,
                                            __global $REAL *Hx,
                                            __global $REAL *Hy,
                                            __global $REAL *Hz
                                        """),
                          'func': Template("""
    // Updates electric field values for Hertzian dipole sources.
    //
    //  Args:
    //      NMAGDIPOLE: Total number of magnetic dipoles in the model.
    //      iteration: Iteration number of simulation.
    //      dx, dy, dz: Spatial discretisations.
    //      srcinfo1: Source cell coordinates and polarisation information.
    //      srcinfo2: Other source information, e.g. length, resistance etc...
    //      srcwaveforms: Source waveform values.
    //      ID, H: Access to ID and field component arrays.

    $CUDA_IDX

    if (i < NMAGDIPOLE) {

        int x, y, z, polarisation;

        x = srcinfo1[IDX2D_SRCINFO(i,0)];
        y = srcinfo1[IDX2D_SRCINFO(i,1)];
        z = srcinfo1[IDX2D_SRCINFO(i,2)];
        polarisation = srcinfo1[IDX2D_SRCINFO(i,3)];

        // 'x' polarised source
        if (polarisation == 0) {
            int materialHx = ID[IDX4D_ID(3,x,y,z)];
            Hx[IDX3D_FIELDS(x,y,z)] = Hx[IDX3D_FIELDS(x,y,z)] - updatecoeffsH[IDX2D_MAT(materialHx,4)] * 
                                        srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (dx * dy * dz));
        }

        // 'y' polarised source
        else if (polarisation == 1) {
            int materialHy = ID[IDX4D_ID(4,x,y,z)];
            Hy[IDX3D_FIELDS(x,y,z)] = Hy[IDX3D_FIELDS(x,y,z)] - updatecoeffsH[IDX2D_MAT(materialHy,4)] * 
                                        srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (dx * dy * dz));
        }

        // 'z' polarised source
        else if (polarisation == 2) {
            int materialHz = ID[IDX4D_ID(5,x,y,z)];
            Hz[IDX3D_FIELDS(x,y,z)] = Hz[IDX3D_FIELDS(x,y,z)] - updatecoeffsH[IDX2D_MAT(materialHz,4)] * 
                                        srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (dx * dy * dz));
        }
    }
""")
}

update_voltage_source = {'args_cuda': Template("""
                                        __global__ void update_voltage_source(int NVOLTSRC, 
                                            int iteration, 
                                            $REAL dx, 
                                            $REAL dy, 
                                            $REAL dz, 
                                            const int* __restrict__ srcinfo1, 
                                            const $REAL* __restrict__ srcinfo2, 
                                            const $REAL* __restrict__ srcwaveforms, 
                                            const unsigned int* __restrict__ ID, 
                                            $REAL *Ex, 
                                            $REAL *Ey, 
                                            $REAL *Ez)
                                        """),
                          'args_opencl': Template("""
                                            int NVOLTSRC,
                                            int iteration,
                                            $REAL dx,
                                            $REAL dy,
                                            $REAL dz,
                                            __global const int* restrict rcinfo1,
                                            __global const $REAL* restrict rcinfo2,
                                            __global const $REAL* restrict rcwaveforms,
                                            __global const unsigned int* estrict ID,
                                            __global $REAL *Ex,
                                            __global $REAL *Ey,
                                            __global $REAL *Ez
                                        """),
                          'func': Template("""
    // Updates electric field values for voltage sources.
    //
    //  Args:
    //      NVOLTSRC: Total number of voltage sources in the model.
    //      iteration: Iteration number of simulation.
    //      dx, dy, dz: Spatial discretisations.
    //      srcinfo1: Source cell coordinates and polarisation information.
    //      srcinfo2: Other source information, e.g. length, resistance etc...
    //      srcwaveforms: Source waveform values.
    //      ID, E: Access to ID and field component arrays.

    $CUDA_IDX

    if (i < NVOLTSRC) {

        $REAL resistance;
        int x, y, z, polarisation;

        x = srcinfo1[IDX2D_SRCINFO(i,0)];
        y = srcinfo1[IDX2D_SRCINFO(i,1)];
        z = srcinfo1[IDX2D_SRCINFO(i,2)];
        polarisation = srcinfo1[IDX2D_SRCINFO(i,3)];
        resistance = srcinfo2[i];

        // 'x' polarised source
        if (polarisation == 0) {
            if (resistance != 0) {
                int materialEx = ID[IDX4D_ID(0,x,y,z)];
                Ex[IDX3D_FIELDS(x,y,z)] = Ex[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] * 
                                            srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (resistance * dy * dz));
            }
            else {
                Ex[IDX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[IDX2D_SRCWAVES(i,iteration)] / dx;
            }
        }

        // 'y' polarised source
        else if (polarisation == 1) {
            if (resistance != 0) {
                int materialEy = ID[IDX4D_ID(1,x,y,z)];
                Ey[IDX3D_FIELDS(x,y,z)] = Ey[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] * 
                                            srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (resistance * dx * dz));
            }
            else {
                Ey[IDX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[IDX2D_SRCWAVES(i,iteration)] / dy;
            }
        }

        // 'z' polarised source
        else if (polarisation == 2) {
            if (resistance != 0) {
                int materialEz = ID[IDX4D_ID(2,x,y,z)];
                Ez[IDX3D_FIELDS(x,y,z)] = Ez[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] * 
                                            srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (resistance * dx * dy));
            }
            else {
                Ez[IDX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[IDX2D_SRCWAVES(i,iteration)] / dz;
            }
        }
    }
""")
}