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
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""CUDA/OpenCL/Metal kernel for the corrected Hyun magnetic-frill source.

The host supplies the fully compiled feed stencil produced by
``MagneticFrillSource.finalise_setup``. Consequently this shared kernel does
not reinterpret Yee geometry or material IDs: its current weights and source
gains already include PMC image completion, the Mäkinen ``k_H`` projection,
Hyun's radius-dependent ``F`` factor, and the local cell dimensions.
"""

from string import Template

update_magnetic_frill_source = {
    "args_cuda": Template(
        """
        __global__ void update_magnetic_frill_source(
            int NFRILL,
            int iteration,
            const int* __restrict__ frill_term_counts,
            const int* __restrict__ frill_term_info,
            const $REAL* __restrict__ frill_term_params,
            const $REAL* __restrict__ frill_params,
            $REAL* frill_state,
            const $REAL* __restrict__ frill_waveform,
            $REAL* Vinc,
            $REAL* Vtotal,
            $REAL* Itot,
            $REAL* Hx,
            $REAL* Hy,
            $REAL* Hz)
        """
    ),
    "args_opencl": Template(
        """
            int NFRILL,
            int iteration,
            __global const int* restrict frill_term_counts,
            __global const int* restrict frill_term_info,
            __global const $REAL* restrict frill_term_params,
            __global const $REAL* restrict frill_params,
            __global $REAL* frill_state,
            __global const $REAL* restrict frill_waveform,
            __global $REAL* Vinc,
            __global $REAL* Vtotal,
            __global $REAL* Itot,
            __global $REAL* Hx,
            __global $REAL* Hy,
            __global $REAL* Hz
        """
    ),
    "args_metal": Template(
        """
        kernel void update_magnetic_frill_source(
            device const int& NFRILL,
            device const int& iteration,
            device const int* frill_term_counts,
            device const int* frill_term_info,
            device const $REAL* frill_term_params,
            device const $REAL* frill_params,
            device $REAL* frill_state,
            device const $REAL* frill_waveform,
            device $REAL* Vinc,
            device $REAL* Vtotal,
            device $REAL* Itot,
            device $REAL* Hx,
            device $REAL* Hy,
            device $REAL* Hz,
            uint i [[thread_position_in_grid]])
        """
    ),
    "func": Template(
        """
    // frill_term_info columns: component (0=Hx, 1=Hy, 2=Hz), x, y, z.
    // frill_term_params columns: current-loop weight, source gain.
    // frill_params columns: Z0, feed-cell self-admittance G, theta.

    $CUDA_IDX

    if (i < NFRILL) {
        $REAL current_bulk = ($REAL)0;
        int nterms = frill_term_counts[i];

        for (int term = 0; term < nterms; term++) {
            int term_index = i * $MAX_FRILLTERMS + term;
            int info_index = term_index * $NY_FRILLTERMINFO;
            int param_index = term_index * $NY_FRILLTERMPARAMS;
            int component = frill_term_info[info_index + 0];
            int x = frill_term_info[info_index + 1];
            int y = frill_term_info[info_index + 2];
            int z = frill_term_info[info_index + 3];
            $REAL weight = frill_term_params[param_index + 0];
            int field_index = IDX3D_FIELDS(x,y,z);

            if (component == 0) {
                current_bulk += weight * Hx[field_index];
            }
            else if (component == 1) {
                current_bulk += weight * Hy[field_index];
            }
            else {
                current_bulk += weight * Hz[field_index];
            }
        }

        int source_index = i * $NY_FRILLPARAMS;
        int output_index = i * $NY_FRILLOUT + iteration;
        $REAL Z0 = frill_params[source_index + 0];
        $REAL G = frill_params[source_index + 1];
        $REAL theta = frill_params[source_index + 2];
        $REAL previous_current = frill_state[i];
        $REAL Vinc_value = ($REAL)0.5
            * frill_waveform[i * $NY_FRILLWAVES + iteration];
        $REAL zeta = G * Z0;

        // Hyun's time-average terminal equation is implicit because this
        // frill deposit changes the current measured around the same feed
        // edge. Solve that scalar feedback relation in closed form.
        $REAL current_new = (
            current_bulk
            + ($REAL)2 * G * Vinc_value
            - zeta * (($REAL)1 - theta) * previous_current
        ) / (($REAL)1 + zeta * theta);
        $REAL current_centred = (
            (($REAL)1 - theta) * previous_current + theta * current_new
        );
        $REAL V_ab = ($REAL)2 * Vinc_value - Z0 * current_centred;

        Vinc[output_index] = Vinc_value;
        Itot[output_index] = current_centred;
        Vtotal[output_index] = V_ab;

        for (int term = 0; term < nterms; term++) {
            int term_index = i * $MAX_FRILLTERMS + term;
            int info_index = term_index * $NY_FRILLTERMINFO;
            int param_index = term_index * $NY_FRILLTERMPARAMS;
            int component = frill_term_info[info_index + 0];
            int x = frill_term_info[info_index + 1];
            int y = frill_term_info[info_index + 2];
            int z = frill_term_info[info_index + 3];
            $REAL source_gain = frill_term_params[param_index + 1];
            int field_index = IDX3D_FIELDS(x,y,z);

            if (component == 0) {
                Hx[field_index] += source_gain * V_ab;
            }
            else if (component == 1) {
                Hy[field_index] += source_gain * V_ab;
            }
            else {
                Hz[field_index] += source_gain * V_ab;
            }
        }

        // This state remains active after the requested waveform interval;
        // the waveform becomes zero but the passive Z0 terminal persists.
        frill_state[i] = current_new;
    }
"""
    ),
}
