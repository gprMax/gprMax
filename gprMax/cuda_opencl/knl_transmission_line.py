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

"""Shared CUDA/OpenCL/Metal kernels for transmission-line feeds.

One work-item owns one complete one-dimensional line. The coupled line has
only a small number of active cells, so a serial loop within a work-item is
both cheaper and easier to keep numerically consistent with the CPU source
than coordinating a workgroup for each line.
"""

from string import Template

update_transmission_line_magnetic = {
    "args_cuda": Template(
        """
        __global__ void update_transmission_line_magnetic(
            int NTL,
            int iteration,
            $REAL dx,
            $REAL dy,
            $REAL dz,
            $REAL line_coefficient,
            const int* __restrict__ tl_info,
            const $REAL* __restrict__ resistance,
            const $REAL* __restrict__ waveform_half,
            const $REAL* __restrict__ voltage,
            $REAL* current,
            $REAL* Vtotal,
            $REAL* Itotal,
            const $REAL* __restrict__ Hx,
            const $REAL* __restrict__ Hy,
            const $REAL* __restrict__ Hz)
        """
    ),
    "args_opencl": Template(
        """
            int NTL,
            int iteration,
            $REAL dx,
            $REAL dy,
            $REAL dz,
            $REAL line_coefficient,
            __global const int* restrict tl_info,
            __global const $REAL* restrict resistance,
            __global const $REAL* restrict waveform_half,
            __global const $REAL* restrict voltage,
            __global $REAL* current,
            __global $REAL* Vtotal,
            __global $REAL* Itotal,
            __global const $REAL* restrict Hx,
            __global const $REAL* restrict Hy,
            __global const $REAL* restrict Hz
        """
    ),
    "args_metal": Template(
        """
        kernel void update_transmission_line_magnetic(
            device const int& NTL,
            device const int& iteration,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device const $REAL& line_coefficient,
            device const int* tl_info,
            device const $REAL* resistance,
            device const $REAL* waveform_half,
            device const $REAL* voltage,
            device $REAL* current,
            device $REAL* Vtotal,
            device $REAL* Itotal,
            device const $REAL* Hx,
            device const $REAL* Hy,
            device const $REAL* Hz,
            uint i [[thread_position_in_grid]])
        """
    ),
    "func": Template(
        """
    // tl_info columns: x, y, z, polarisation, state offset, line length,
    // source position, antenna position, first active iteration, last active
    // iteration. One work-item advances one complete line.

    $CUDA_IDX

    if (i < NTL) {
        int x = tl_info[i * $NY_TLINFO + 0];
        int y = tl_info[i * $NY_TLINFO + 1];
        int z = tl_info[i * $NY_TLINFO + 2];
        int polarisation = tl_info[i * $NY_TLINFO + 3];
        int offset = tl_info[i * $NY_TLINFO + 4];
        int nl = tl_info[i * $NY_TLINFO + 5];
        int srcpos = tl_info[i * $NY_TLINFO + 6];
        int antpos = tl_info[i * $NY_TLINFO + 7];
        int first_active = tl_info[i * $NY_TLINFO + 8];
        int last_active = tl_info[i * $NY_TLINFO + 9];
        int antenna = offset + antpos;

        // CPU output is sampled at the start of the time step. The line state
        // has not changed during the bulk H update, so storing it here is
        // equivalent and avoids an additional device-kernel launch.
        Vtotal[i * $NY_TLOUTPUTS + iteration] = voltage[antenna];
        Itotal[i * $NY_TLOUTPUTS + iteration] = current[antenna];

        if (iteration >= first_active && iteration <= last_active) {
            // Replace the terminal current with the Ampere-loop current from
            // the four magnetic Yee edges surrounding the source E edge.
            if (polarisation == 0) {
                if (y == 0 || z == 0) {
                    current[antenna] = 0;
                }
                else {
                    current[antenna] =
                        dy * (Hy[IDX3D_FIELDS(x,y,z-1)] - Hy[IDX3D_FIELDS(x,y,z)]) +
                        dz * (Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x,y-1,z)]);
                }
            }
            else if (polarisation == 1) {
                if (x == 0 || z == 0) {
                    current[antenna] = 0;
                }
                else {
                    current[antenna] =
                        dx * (Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y,z-1)]) +
                        dz * (Hz[IDX3D_FIELDS(x-1,y,z)] - Hz[IDX3D_FIELDS(x,y,z)]);
                }
            }
            else {
                if (x == 0 || y == 0) {
                    current[antenna] = 0;
                }
                else {
                    current[antenna] =
                        dx * (Hx[IDX3D_FIELDS(x,y-1,z)] - Hx[IDX3D_FIELDS(x,y,z)]) +
                        dy * (Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x-1,y,z)]);
                }
            }

            // Advance all non-terminal transmission-line current samples.
            for (int linepos = 0; linepos < nl - 1; linepos++) {
                int pos = offset + linepos;
                current[pos] -= (line_coefficient / resistance[i]) *
                    (voltage[pos + 1] - voltage[pos]);
            }
            current[offset + srcpos - 1] +=
                (line_coefficient / resistance[i]) *
                waveform_half[i * $NY_TLWAVES + iteration];
        }
    }
"""
    ),
}


update_transmission_line_electric = {
    "args_cuda": Template(
        """
        __global__ void update_transmission_line_electric(
            int NTL,
            int iteration,
            $REAL dx,
            $REAL dy,
            $REAL dz,
            $REAL line_coefficient,
            $REAL abc_coefficient,
            const int* __restrict__ tl_info,
            const $REAL* __restrict__ resistance,
            const $REAL* __restrict__ waveform_whole,
            $REAL* voltage,
            const $REAL* __restrict__ current,
            $REAL* abcv0,
            $REAL* abcv1,
            $REAL* Ex,
            $REAL* Ey,
            $REAL* Ez)
        """
    ),
    "args_opencl": Template(
        """
            int NTL,
            int iteration,
            $REAL dx,
            $REAL dy,
            $REAL dz,
            $REAL line_coefficient,
            $REAL abc_coefficient,
            __global const int* restrict tl_info,
            __global const $REAL* restrict resistance,
            __global const $REAL* restrict waveform_whole,
            __global $REAL* voltage,
            __global const $REAL* restrict current,
            __global $REAL* abcv0,
            __global $REAL* abcv1,
            __global $REAL* Ex,
            __global $REAL* Ey,
            __global $REAL* Ez
        """
    ),
    "args_metal": Template(
        """
        kernel void update_transmission_line_electric(
            device const int& NTL,
            device const int& iteration,
            device const $REAL& dx,
            device const $REAL& dy,
            device const $REAL& dz,
            device const $REAL& line_coefficient,
            device const $REAL& abc_coefficient,
            device const int* tl_info,
            device const $REAL* resistance,
            device const $REAL* waveform_whole,
            device $REAL* voltage,
            device const $REAL* current,
            device $REAL* abcv0,
            device $REAL* abcv1,
            device $REAL* Ex,
            device $REAL* Ey,
            device $REAL* Ez,
            uint i [[thread_position_in_grid]])
        """
    ),
    "func": Template(
        """
    $CUDA_IDX

    if (i < NTL) {
        int x = tl_info[i * $NY_TLINFO + 0];
        int y = tl_info[i * $NY_TLINFO + 1];
        int z = tl_info[i * $NY_TLINFO + 2];
        int polarisation = tl_info[i * $NY_TLINFO + 3];
        int offset = tl_info[i * $NY_TLINFO + 4];
        int nl = tl_info[i * $NY_TLINFO + 5];
        int srcpos = tl_info[i * $NY_TLINFO + 6];
        int antpos = tl_info[i * $NY_TLINFO + 7];
        int first_active = tl_info[i * $NY_TLINFO + 8];
        int last_active = tl_info[i * $NY_TLINFO + 9];

        if (iteration >= first_active && iteration <= last_active) {
            for (int linepos = 1; linepos < nl; linepos++) {
                int pos = offset + linepos;
                voltage[pos] -= resistance[i] * line_coefficient *
                    (current[pos] - current[pos - 1]);
            }
            voltage[offset + srcpos] +=
                line_coefficient * waveform_whole[i * $NY_TLWAVES + iteration];

            // First-order absorbing boundary at the remote end of the line.
            $REAL boundary_voltage = abc_coefficient *
                (voltage[offset + 1] - abcv0[i]) + abcv1[i];
            voltage[offset] = boundary_voltage;
            abcv0[i] = boundary_voltage;
            abcv1[i] = voltage[offset + 1];

            $REAL terminal_voltage = voltage[offset + antpos];
            if (polarisation == 0) {
                Ex[IDX3D_FIELDS(x,y,z)] = -terminal_voltage / dx;
            }
            else if (polarisation == 1) {
                Ey[IDX3D_FIELDS(x,y,z)] = -terminal_voltage / dy;
            }
            else {
                Ez[IDX3D_FIELDS(x,y,z)] = -terminal_voltage / dz;
            }
        }
    }
"""
    ),
}
