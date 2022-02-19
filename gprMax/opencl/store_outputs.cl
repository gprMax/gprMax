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


#define INDEX2D_RXCOORDS(m,n) (m)*({{NY_RXCOORDS}}) + (n)
#define INDEX3D_RXS(i,j,k) (i)*({{NY_RXS}})*({{NZ_RXS}}) + (j)*({{NZ_RXS}}) + (k)
#define INDEX3D_FIELDS(i,j,k) (i)*({{NY_FIELDS}})*({{NZ_FIELDS}}) + (j)*({{NZ_FIELDS}}) + (k)

__kernel void store_outputs(int NRX, int iteration, 
                            __global const int* restrict rxcoords, 
                            __global {{REAL}} *rxs, 
                            __global const {{REAL}}* restrict Ex, 
                            __global const {{REAL}}* restrict Ey, 
                            __global const {{REAL}}* restrict Ez, 
                            __global const {{REAL}}* restrict Hx, 
                            __global const {{REAL}}* restrict Hy, 
                            __global const {{REAL}}* restrict Hz) {

    // This function stores field component values for every receiver in the model.
    //
    // Args: 
    //    NRX: total number of receivers in the model.
    //    rxs: array to store field components for receivers - rows 
    //          are field components; columns are iterations; pages are receiver

    // Obtain linear index corresponding to the current work item
    int rx = get_global_id(2) * get_global_size(0) * get_global_size(1) + 
             get_global_id(1) * get_global_size(0) + get_global_id(0);

    int i,j,k;

    if (rx < NRX) {
        i = rxcoords[INDEX2D_RXCOORDS(rx,0)];
        j = rxcoords[INDEX2D_RXCOORDS(rx,1)];
        k = rxcoords[INDEX2D_RXCOORDS(rx,2)];
        rxs[INDEX3D_RXS(0,iteration,rx)] = Ex[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(1,iteration,rx)] = Ey[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(2,iteration,rx)] = Ez[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(3,iteration,rx)] = Hx[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(4,iteration,rx)] = Hy[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(5,iteration,rx)] = Hz[INDEX3D_FIELDS(i,j,k)];
    }
}

