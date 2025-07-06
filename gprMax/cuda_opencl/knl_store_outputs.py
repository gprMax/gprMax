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

store_outputs = {
    "args_cuda": Template(
        """
                                __global__ void store_outputs(int NRX,
                                                    int iteration,
                                                    const int* __restrict__ rxcoords,
                                                    $REAL *rxs,
                                                    const $REAL* __restrict__ Ex,
                                                    const $REAL* __restrict__ Ey,
                                                    const $REAL* __restrict__ Ez,
                                                    const $REAL* __restrict__ Hx,
                                                    const $REAL* __restrict__ Hy,
                                                    const $REAL* __restrict__ Hz)
                                """
    ),
    "args_opencl": Template(
        """
                                    int NRX,
                                    int iteration,
                                    __global const int* restrict rxcoords,
                                    __global $REAL *rxs,
                                    __global const $REAL* restrict Ex,
                                    __global const $REAL* restrict Ey,
                                    __global const $REAL* restrict Ez,
                                    __global const $REAL* restrict Hx,
                                    __global const $REAL* restrict Hy,
                                    __global const $REAL* restrict Hz
                                    """
    ),
    "args_metal": Template(
        """
                               kernel void store_outputs(device const int& NRX,
                                                    device const int& iteration,
                                                    device const int* rxcoords,
                                                    device float* rxs,
                                                    device const float* Ex,
                                                    device const float* Ey,
                                                    device const float* Ez,
                                                    device const float* Hx,
                                                    device const float* Hy,
                                                    device const float* Hz,
                                                    uint i [[thread_position_in_grid]])
                                """
    ),
    "func": Template(
        """
    // Stores field component values for every receiver in the model.
    //
    // Args:
    //    NRX: total number of receivers in the model.
    //    rxs: array to store field components for receivers - rows
    //          are field components; columns are iterations; pages are receiver.

    $CUDA_IDX

    if (i < NRX) {
        int x, y, z;
        x = rxcoords[IDX2D_RXCOORDS(i,0)];
        y = rxcoords[IDX2D_RXCOORDS(i,1)];
        z = rxcoords[IDX2D_RXCOORDS(i,2)];
        rxs[IDX3D_RXS(0,iteration,i)] = Ex[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(1,iteration,i)] = Ey[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(2,iteration,i)] = Ez[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(3,iteration,i)] = Hx[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(4,iteration,i)] = Hy[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(5,iteration,i)] = Hz[IDX3D_FIELDS(x,y,z)];
    }
"""
    ),
}
