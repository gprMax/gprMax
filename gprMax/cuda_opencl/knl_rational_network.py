# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Shared CUDA/OpenCL/Metal kernel for sparse rational-network terminals."""

from string import Template

update_rational_network = {
    "args_cuda": Template(
        """
        __global__ void update_rational_network(
            int NTERMINALS,
            int iteration,
            $REAL dt,
            const int* __restrict__ info,
            const $REAL* __restrict__ params,
            const $REAL* __restrict__ waveform_whole,
            const $REAL* __restrict__ waveform_half,
            $REAL* voltage,
            $REAL* current,
            const $REAL* __restrict__ exp_half_real,
            const $REAL* __restrict__ exp_half_imag,
            const $REAL* __restrict__ coeff_half_new_real,
            const $REAL* __restrict__ coeff_half_new_imag,
            const $REAL* __restrict__ coeff_half_old_real,
            const $REAL* __restrict__ coeff_half_old_imag,
            const $REAL* __restrict__ exp_full_real,
            const $REAL* __restrict__ exp_full_imag,
            const $REAL* __restrict__ coeff_full_new_real,
            const $REAL* __restrict__ coeff_full_new_imag,
            const $REAL* __restrict__ coeff_full_old_real,
            const $REAL* __restrict__ coeff_full_old_imag,
            $REAL* state_real,
            $REAL* state_imag,
            $REAL* Ex,
            $REAL* Ey,
            $REAL* Ez)
        """
    ),
    "args_opencl": Template(
        """
            int NTERMINALS,
            int iteration,
            $REAL dt,
            __global const int* restrict info,
            __global const $REAL* restrict params,
            __global const $REAL* restrict waveform_whole,
            __global const $REAL* restrict waveform_half,
            __global $REAL* voltage,
            __global $REAL* current,
            __global const $REAL* restrict exp_half_real,
            __global const $REAL* restrict exp_half_imag,
            __global const $REAL* restrict coeff_half_new_real,
            __global const $REAL* restrict coeff_half_new_imag,
            __global const $REAL* restrict coeff_half_old_real,
            __global const $REAL* restrict coeff_half_old_imag,
            __global const $REAL* restrict exp_full_real,
            __global const $REAL* restrict exp_full_imag,
            __global const $REAL* restrict coeff_full_new_real,
            __global const $REAL* restrict coeff_full_new_imag,
            __global const $REAL* restrict coeff_full_old_real,
            __global const $REAL* restrict coeff_full_old_imag,
            __global $REAL* state_real,
            __global $REAL* state_imag,
            __global $REAL* Ex,
            __global $REAL* Ey,
            __global $REAL* Ez
        """
    ),
    "args_metal": Template(
        """
        kernel void update_rational_network(
            device const int& NTERMINALS,
            device const int& iteration,
            device const $REAL& dt,
            device const int* info,
            device const $REAL* params,
            device const $REAL* waveform_whole,
            device const $REAL* waveform_half,
            device $REAL* voltage,
            device $REAL* current,
            device const $REAL* exp_half_real,
            device const $REAL* exp_half_imag,
            device const $REAL* coeff_half_new_real,
            device const $REAL* coeff_half_new_imag,
            device const $REAL* coeff_half_old_real,
            device const $REAL* coeff_half_old_imag,
            device const $REAL* exp_full_real,
            device const $REAL* exp_full_imag,
            device const $REAL* coeff_full_new_real,
            device const $REAL* coeff_full_new_imag,
            device const $REAL* coeff_full_old_real,
            device const $REAL* coeff_full_old_imag,
            device $REAL* state_real,
            device $REAL* state_imag,
            device $REAL* Ex,
            device $REAL* Ey,
            device $REAL* Ez,
            uint i [[thread_position_in_grid]])
        """
    ),
    "func": Template(
        """
    $CUDA_IDX

    if (i < NTERMINALS) {
        int x = info[i * $NY_RNINFO + 0];
        int y = info[i * $NY_RNINFO + 1];
        int z = info[i * $NY_RNINFO + 2];
        int polarisation = info[i * $NY_RNINFO + 3];
        int pole_offset = info[i * $NY_RNINFO + 4];
        int pole_count = info[i * $NY_RNINFO + 5];

        $REAL dl = params[i * $NY_RNPARAMS + 0];
        $REAL area = params[i * $NY_RNPARAMS + 1];
        $REAL source_coefficient = params[i * $NY_RNPARAMS + 2];
        $REAL denominator = params[i * $NY_RNPARAMS + 3];
        $REAL alpha = params[i * $NY_RNPARAMS + 4];
        $REAL conductance = params[i * $NY_RNPARAMS + 5];
        $REAL capacitance = params[i * $NY_RNPARAMS + 6];

        int whole_offset = i * $NY_RNWAVEWHOLE;
        int half_offset = i * $NY_RNWAVEHALF;
        int voltage_offset = i * $NY_RNVOLTAGE;
        int current_offset = i * $NY_RNCURRENT;
        $REAL voltage_old = voltage[voltage_offset + iteration];
        $REAL generator_old = waveform_whole[whole_offset + iteration];
        $REAL generator_new = waveform_whole[whole_offset + iteration + 1];
        $REAL generator_half = waveform_half[half_offset + iteration];
        $REAL input_old = voltage_old - generator_old;
        $REAL history =
            (conductance / ($REAL)2.0 - capacitance / dt) * voltage_old
            - conductance * generator_half
            - capacitance * (generator_new - generator_old) / dt;

        for (int local_pole = 0; local_pole < pole_count; local_pole++) {
            int p = pole_offset + local_pole;
            $REAL half_real =
                exp_half_real[p] * state_real[p]
                - exp_half_imag[p] * state_imag[p]
                + coeff_half_old_real[p] * input_old
                - coeff_half_new_real[p] * generator_new;
            history += half_real;
        }

        int field_index = IDX3D_FIELDS(x,y,z);
        $REAL electric;
        if (polarisation == 0) {
            electric = (Ex[field_index] + source_coefficient * history / area) / denominator;
            Ex[field_index] = electric;
        }
        else if (polarisation == 1) {
            electric = (Ey[field_index] + source_coefficient * history / area) / denominator;
            Ey[field_index] = electric;
        }
        else {
            electric = (Ez[field_index] + source_coefficient * history / area) / denominator;
            Ez[field_index] = electric;
        }

        $REAL voltage_new = -dl * electric;
        $REAL input_new = voltage_new - generator_new;
        voltage[voltage_offset + iteration + 1] = voltage_new;
        current[current_offset + iteration] = alpha * voltage_new + history;

        for (int local_pole = 0; local_pole < pole_count; local_pole++) {
            int p = pole_offset + local_pole;
            $REAL old_real = state_real[p];
            $REAL old_imag = state_imag[p];
            state_real[p] =
                exp_full_real[p] * old_real - exp_full_imag[p] * old_imag
                + coeff_full_new_real[p] * input_new
                + coeff_full_old_real[p] * input_old;
            state_imag[p] =
                exp_full_real[p] * old_imag + exp_full_imag[p] * old_real
                + coeff_full_new_imag[p] * input_new
                + coeff_full_old_imag[p] * input_old;
        }
    }
"""
    ),
}
