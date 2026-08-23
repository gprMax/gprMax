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

"""Runtime kernels for device-resident KSIR and Love-current transforms."""

from string import Template

accumulate_ntff = {
    "args_cuda": Template(
        """
extern "C" __global__ void accumulate_ntff(
    int total, int npatches,
    const int* __restrict__ inside_index,
    const int* __restrict__ outside_index,
    const $REAL* __restrict__ multiplier_real,
    const $REAL* __restrict__ multiplier_imag,
    const $REAL* __restrict__ field,
    $REAL* inside_real, $REAL* inside_imag,
    $REAL* outside_real, $REAL* outside_imag)
"""
    ),
    "args_opencl": Template(
        """
__kernel void accumulate_ntff(
    int total, int npatches,
    __global const int* restrict inside_index,
    __global const int* restrict outside_index,
    __global const $REAL* restrict multiplier_real,
    __global const $REAL* restrict multiplier_imag,
    int field_offset,
    __global const $REAL* restrict field,
    __global $REAL* inside_real, __global $REAL* inside_imag,
    __global $REAL* outside_real, __global $REAL* outside_imag)
"""
    ),
    "args_metal": Template(
        """
kernel void accumulate_ntff(
    device const int& total, device const int& npatches,
    device const int* inside_index,
    device const int* outside_index,
    device const $REAL* multiplier_real,
    device const $REAL* multiplier_imag,
    device const $REAL* field,
    device $REAL* inside_real, device $REAL* inside_imag,
    device $REAL* outside_real, device $REAL* outside_imag,
    uint i [[thread_position_in_grid]])
"""
    ),
    "func": Template(
        """
    if (i < total) {
        int frequency = i / npatches;
        int patch = i - frequency * npatches;
        $REAL inside_value = field[$FIELD_OFFSET + inside_index[patch]];
        $REAL outside_value = field[$FIELD_OFFSET + outside_index[patch]];
        $REAL real_weight = multiplier_real[frequency];
        $REAL imag_weight = multiplier_imag[frequency];
        inside_real[i] += real_weight * inside_value;
        inside_imag[i] += imag_weight * inside_value;
        outside_real[i] += real_weight * outside_value;
        outside_imag[i] += imag_weight * outside_value;
    }
"""
    ),
}


def build_ntff_kernel_source(backend: str, c_real: str) -> str:
    """Render a standalone NTFF kernel for CUDA, OpenCL, or Metal."""

    try:
        declaration = accumulate_ntff[f"args_{backend}"].substitute(REAL=c_real)
    except KeyError as exc:
        raise ValueError("backend must be 'cuda', 'opencl', or 'metal'") from exc

    if backend == "cuda":
        index = "int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        preamble = ""
    elif backend == "opencl":
        index = "int i = get_global_id(0);\n"
        preamble = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" if c_real == "double" else ""
    else:
        index = ""
        preamble = "#include <metal_stdlib>\nusing namespace metal;\n"

    field_offset = "field_offset" if backend == "opencl" else "0"
    body = accumulate_ntff["func"].substitute(REAL=c_real, FIELD_OFFSET=field_offset)
    return f"{preamble}{declaration}{{\n{index}{body}}}\n"


gather_time_domain_ntff = {
    "args_cuda": Template(
        """
extern "C" __global__ void gather_time_domain_ntff(
    int npatches,
    const int* __restrict__ inside_index,
    const int* __restrict__ outside_index,
    const $REAL* __restrict__ normal_spacing,
    const $REAL* __restrict__ field,
    $REAL* surface_value,
    $REAL* normal_derivative)
"""
    ),
    "args_opencl": Template(
        """
__kernel void gather_time_domain_ntff(
    int npatches,
    __global const int* restrict inside_index,
    __global const int* restrict outside_index,
    __global const $REAL* restrict normal_spacing,
    __global const $REAL* restrict field,
    __global $REAL* surface_value,
    __global $REAL* normal_derivative)
"""
    ),
    "args_metal": Template(
        """
kernel void gather_time_domain_ntff(
    device const int& npatches,
    device const int* inside_index,
    device const int* outside_index,
    device const $REAL* normal_spacing,
    device const $REAL* field,
    device $REAL* surface_value,
    device $REAL* normal_derivative,
    uint patch [[thread_position_in_grid]])
"""
    ),
    "func": Template(
        """
    $INDEX
    if (patch < npatches) {
        $REAL inside_value = field[inside_index[patch]];
        $REAL outside_value = field[outside_index[patch]];
        surface_value[patch] = ($REAL)0.5 * (inside_value + outside_value);
        normal_derivative[patch] =
            (outside_value - inside_value) / normal_spacing[patch];
    }
"""
    ),
}


deposit_time_domain_ntff = {
    "args_cuda": Template(
        """
extern "C" __global__ void deposit_time_domain_ntff(
    int total,
    int neffective_patches,
    int output_length,
    int sample_index,
    const $REAL* __restrict__ surface_value,
    const $REAL* __restrict__ normal_derivative,
    const $REAL* __restrict__ time_surface_a,
    const $REAL* __restrict__ time_surface_b,
    const $REAL* __restrict__ time_surface_c,
    $REAL coefficient_a,
    $REAL coefficient_b,
    $REAL coefficient_c,
    const $REAL* __restrict__ normal_derivative_weight,
    const $REAL* __restrict__ field_weight,
    const $REAL* __restrict__ time_derivative_weight,
    const int* __restrict__ source_patch_index,
    const int* __restrict__ integer_delay,
    const $REAL* __restrict__ fractional_delay,
    const int* __restrict__ time_origin_steps,
    $REAL* output)
"""
    ),
    "args_opencl": Template(
        """
__kernel void deposit_time_domain_ntff(
    int npoints,
    int neffective_patches,
    int output_length,
    int sample_index,
    __global const $REAL* restrict surface_value,
    __global const $REAL* restrict normal_derivative,
    __global const $REAL* restrict time_surface_a,
    __global const $REAL* restrict time_surface_b,
    __global const $REAL* restrict time_surface_c,
    $REAL coefficient_a,
    $REAL coefficient_b,
    $REAL coefficient_c,
    __global const $REAL* restrict normal_derivative_weight,
    __global const $REAL* restrict field_weight,
    __global const $REAL* restrict time_derivative_weight,
    __global const int* restrict source_patch_index,
    __global const int* restrict integer_delay,
    __global const $REAL* restrict fractional_delay,
    __global const int* restrict time_origin_steps,
    __global $REAL* output)
"""
    ),
    "args_metal": Template(
        """
kernel void deposit_time_domain_ntff(
    device const int& npoints,
    device const int& neffective_patches,
    device const int& output_length,
    device const int& sample_index,
    device const $REAL* surface_value,
    device const $REAL* normal_derivative,
    device const $REAL* time_surface_a,
    device const $REAL* time_surface_b,
    device const $REAL* time_surface_c,
    device const $REAL& coefficient_a,
    device const $REAL& coefficient_b,
    device const $REAL& coefficient_c,
    device const $REAL* normal_derivative_weight,
    device const $REAL* field_weight,
    device const $REAL* time_derivative_weight,
    device const int* source_patch_index,
    device const int* integer_delay,
    device const $REAL* fractional_delay,
    device const int* time_origin_steps,
    device $REAL* output,
    uint point [[thread_position_in_grid]])
"""
    ),
    "func_cuda": Template(
        """
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int point = i / neffective_patches;
        int patch = i - point * neffective_patches;
        int source_patch = source_patch_index[patch];
        int destination = sample_index - time_origin_steps[point]
            + integer_delay[i];
        $REAL time_derivative = coefficient_a * time_surface_a[source_patch]
            + coefficient_b * time_surface_b[source_patch]
            + coefficient_c * time_surface_c[source_patch];
        $REAL contribution = normal_derivative_weight[i]
                * normal_derivative[source_patch]
            + field_weight[i] * surface_value[source_patch]
            + time_derivative_weight[i] * time_derivative;
        $REAL fraction = fractional_delay[i];
        ksir_atomic_add(
            &output[point * output_length + destination],
            (($REAL)1 - fraction) * contribution);
        ksir_atomic_add(
            &output[point * output_length + destination + 1],
            fraction * contribution);
    }
"""
    ),
    "func_portable": Template(
        """
    $INDEX
    if (point < npoints) {
        int point_offset = point * neffective_patches;
        int output_offset = point * output_length;
        for (int patch = 0; patch < neffective_patches; patch++) {
            int i = point_offset + patch;
            int source_patch = source_patch_index[patch];
            int destination = sample_index - time_origin_steps[point]
                + integer_delay[i];
            $REAL time_derivative = coefficient_a * time_surface_a[source_patch]
                + coefficient_b * time_surface_b[source_patch]
                + coefficient_c * time_surface_c[source_patch];
            $REAL contribution = normal_derivative_weight[i]
                    * normal_derivative[source_patch]
                + field_weight[i] * surface_value[source_patch]
                + time_derivative_weight[i] * time_derivative;
            $REAL fraction = fractional_delay[i];
            output[output_offset + destination] +=
                (($REAL)1 - fraction) * contribution;
            output[output_offset + destination + 1] += fraction * contribution;
        }
    }
"""
    ),
}


_CUDA_FLOAT_ATOMIC_PREAMBLE = r"""
__device__ inline void ksir_atomic_add(float* address, float value)
{
    atomicAdd(address, value);
}
"""


_CUDA_DOUBLE_ATOMIC_PREAMBLE = r"""
__device__ inline void ksir_atomic_add(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, value);
#else
    unsigned long long int* integer_address =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *integer_address;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            integer_address,
            assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}
"""


def build_time_domain_ntff_kernel_source(c_real: str, backend: str = "cuda") -> str:
    """Render advanced-time gather and delayed-deposition kernels."""

    if c_real not in ("float", "double"):
        raise ValueError("c_real must be 'float' or 'double'")
    if backend not in ("cuda", "opencl", "metal"):
        raise ValueError("backend must be 'cuda', 'opencl', or 'metal'")

    if backend == "cuda":
        preamble = (
            _CUDA_FLOAT_ATOMIC_PREAMBLE if c_real == "float" else _CUDA_DOUBLE_ATOMIC_PREAMBLE
        )
        gather_index = "int patch = blockIdx.x * blockDim.x + threadIdx.x;"
        deposit_body = deposit_time_domain_ntff["func_cuda"].substitute(REAL=c_real)
    elif backend == "opencl":
        preamble = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" if c_real == "double" else ""
        gather_index = "int patch = get_global_id(0);"
        deposit_body = deposit_time_domain_ntff["func_portable"].substitute(
            REAL=c_real, INDEX="int point = get_global_id(0);"
        )
    else:
        preamble = "#include <metal_stdlib>\nusing namespace metal;\n"
        gather_index = ""
        deposit_body = deposit_time_domain_ntff["func_portable"].substitute(REAL=c_real, INDEX="")

    gather_declaration = gather_time_domain_ntff[f"args_{backend}"].substitute(REAL=c_real)
    gather_body = gather_time_domain_ntff["func"].substitute(REAL=c_real, INDEX=gather_index)
    deposit_declaration = deposit_time_domain_ntff[f"args_{backend}"].substitute(REAL=c_real)
    return (
        f"{preamble}\n{gather_declaration}{{\n{gather_body}}}\n"
        f"{deposit_declaration}{{\n{deposit_body}}}\n"
    )


_EQUIVALENT_CURRENT_GATHER_ARGS = {
    "cuda": Template(
        """
extern "C" __global__ void gather_equivalent_current_time(
    int npatches, int max_samples,
    const int* __restrict__ count_x,
    const int* __restrict__ count_y,
    const int* __restrict__ count_z,
    const int* __restrict__ stencil_x,
    const int* __restrict__ stencil_y,
    const int* __restrict__ stencil_z,
    const $REAL* __restrict__ normals,
    $REAL scale,
    const $REAL* __restrict__ field_x,
    const $REAL* __restrict__ field_y,
    const $REAL* __restrict__ field_z,
    $REAL* current)
"""
    ),
    "opencl": Template(
        """
__kernel void gather_equivalent_current_time(
    int npatches, int max_samples,
    __global const int* restrict count_x,
    __global const int* restrict count_y,
    __global const int* restrict count_z,
    __global const int* restrict stencil_x,
    __global const int* restrict stencil_y,
    __global const int* restrict stencil_z,
    __global const $REAL* restrict normals,
    $REAL scale,
    __global const $REAL* restrict field_x,
    __global const $REAL* restrict field_y,
    __global const $REAL* restrict field_z,
    __global $REAL* current)
"""
    ),
    "metal": Template(
        """
kernel void gather_equivalent_current_time(
    device const int& npatches, device const int& max_samples,
    device const int* count_x,
    device const int* count_y,
    device const int* count_z,
    device const int* stencil_x,
    device const int* stencil_y,
    device const int* stencil_z,
    device const $REAL* normals,
    device const $REAL& scale,
    device const $REAL* field_x,
    device const $REAL* field_y,
    device const $REAL* field_z,
    device $REAL* current,
    uint patch [[thread_position_in_grid]])
"""
    ),
}


_EQUIVALENT_CURRENT_GATHER_BODY = Template(
    r"""
    $INDEX
    if (patch < npatches) {
        $REAL values[3] = {($REAL)0, ($REAL)0, ($REAL)0};
        int offset = patch * max_samples;
        for (int sample = 0; sample < count_x[patch]; sample++)
            values[0] += field_x[stencil_x[offset + sample]];
        for (int sample = 0; sample < count_y[patch]; sample++)
            values[1] += field_y[stencil_y[offset + sample]];
        for (int sample = 0; sample < count_z[patch]; sample++)
            values[2] += field_z[stencil_z[offset + sample]];
        if (count_x[patch] > 0) values[0] /= ($REAL)count_x[patch];
        if (count_y[patch] > 0) values[1] /= ($REAL)count_y[patch];
        if (count_z[patch] > 0) values[2] /= ($REAL)count_z[patch];
        $REAL nx = normals[patch * 3];
        $REAL ny = normals[patch * 3 + 1];
        $REAL nz = normals[patch * 3 + 2];
        current[patch * 3] = scale * (ny * values[2] - nz * values[1]);
        current[patch * 3 + 1] = scale * (nz * values[0] - nx * values[2]);
        current[patch * 3 + 2] = scale * (nx * values[1] - ny * values[0]);
    }
"""
)


_EQUIVALENT_CURRENT_DEPOSIT_ARGS = {
    "cuda": Template(
        """
extern "C" __global__ void deposit_equivalent_current_time(
    int ndirections, int npatches, int output_length,
    int sample_index, int time_origin_step, $REAL inverse_dt,
    const $REAL* __restrict__ current,
    const $REAL* __restrict__ previous,
    const $REAL* __restrict__ theta_basis,
    const $REAL* __restrict__ phi_basis,
    const int* __restrict__ integer_delay,
    const $REAL* __restrict__ fractional_delay,
    const $REAL* __restrict__ area_weights,
    $REAL* output_theta, $REAL* output_phi)
"""
    ),
    "opencl": Template(
        """
__kernel void deposit_equivalent_current_time(
    int ndirections, int npatches, int output_length,
    int sample_index, int time_origin_step, $REAL inverse_dt,
    __global const $REAL* restrict current,
    __global const $REAL* restrict previous,
    __global const $REAL* restrict theta_basis,
    __global const $REAL* restrict phi_basis,
    __global const int* restrict integer_delay,
    __global const $REAL* restrict fractional_delay,
    __global const $REAL* restrict area_weights,
    __global $REAL* output_theta,
    __global $REAL* output_phi)
"""
    ),
    "metal": Template(
        """
kernel void deposit_equivalent_current_time(
    device const int& ndirections, device const int& npatches,
    device const int& output_length, device const int& sample_index,
    device const int& time_origin_step, device const $REAL& inverse_dt,
    device const $REAL* current,
    device const $REAL* previous,
    device const $REAL* theta_basis,
    device const $REAL* phi_basis,
    device const int* integer_delay,
    device const $REAL* fractional_delay,
    device const $REAL* area_weights,
    device $REAL* output_theta,
    device $REAL* output_phi,
    uint direction [[thread_position_in_grid]])
"""
    ),
}


_EQUIVALENT_CURRENT_DEPOSIT_BODY = Template(
    r"""
    $INDEX
    if (direction < ndirections) {
        int basis = direction * 3;
        int delay = direction * npatches;
        int output = direction * output_length;
        for (int patch = 0; patch < npatches; patch++) {
            int vector = patch * 3;
            $REAL dx = (current[vector] - previous[vector]) * inverse_dt;
            $REAL dy = (current[vector + 1] - previous[vector + 1]) * inverse_dt;
            $REAL dz = (current[vector + 2] - previous[vector + 2]) * inverse_dt;
            $REAL theta_value = dx * theta_basis[basis]
                + dy * theta_basis[basis + 1] + dz * theta_basis[basis + 2];
            $REAL phi_value = dx * phi_basis[basis]
                + dy * phi_basis[basis + 1] + dz * phi_basis[basis + 2];
            int destination = sample_index + integer_delay[delay + patch]
                - time_origin_step;
            $REAL fraction = fractional_delay[delay + patch];
            $REAL area = area_weights[patch];
            output_theta[output + destination] +=
                (($REAL)1 - fraction) * area * theta_value;
            output_theta[output + destination + 1] +=
                fraction * area * theta_value;
            output_phi[output + destination] +=
                (($REAL)1 - fraction) * area * phi_value;
            output_phi[output + destination + 1] += fraction * area * phi_value;
        }
    }
"""
)


def build_equivalent_current_time_kernel_source(c_real: str, backend: str) -> str:
    """Render the 1997 Love-current gather and delayed-deposition kernels."""

    if c_real not in ("float", "double"):
        raise ValueError("c_real must be 'float' or 'double'")
    if backend not in ("cuda", "opencl", "metal"):
        raise ValueError("backend must be 'cuda', 'opencl', or 'metal'")
    if backend == "cuda":
        preamble = ""
        gather_index = "int patch = blockIdx.x * blockDim.x + threadIdx.x;"
        deposit_index = "int direction = blockIdx.x * blockDim.x + threadIdx.x;"
    elif backend == "opencl":
        preamble = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" if c_real == "double" else ""
        gather_index = "int patch = get_global_id(0);"
        deposit_index = "int direction = get_global_id(0);"
    else:
        preamble = "#include <metal_stdlib>\nusing namespace metal;\n"
        gather_index = ""
        deposit_index = ""
    gather = _EQUIVALENT_CURRENT_GATHER_ARGS[backend].substitute(REAL=c_real)
    gather_body = _EQUIVALENT_CURRENT_GATHER_BODY.substitute(
        REAL=c_real, INDEX=gather_index
    )
    deposit = _EQUIVALENT_CURRENT_DEPOSIT_ARGS[backend].substitute(REAL=c_real)
    deposit_body = _EQUIVALENT_CURRENT_DEPOSIT_BODY.substitute(
        REAL=c_real, INDEX=deposit_index
    )
    return f"{preamble}{gather}{{\n{gather_body}}}\n{deposit}{{\n{deposit_body}}}\n"
