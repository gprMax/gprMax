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

"""Runtime kernels for sparse device-resident SAR field DFTs."""

from string import Template

_ACCUMULATE_SAR = {
    "cuda": Template(
        """
extern "C" __global__ void accumulate_sar(
    int total, int nedges,
    const int* __restrict__ edge_index,
    const $REAL* __restrict__ multiplier_real,
    const $REAL* __restrict__ multiplier_imag,
    const $REAL* __restrict__ field,
    $REAL* output_real, $REAL* output_imag)
"""
    ),
    "opencl": Template(
        """
__kernel void accumulate_sar(
    int total, int nedges,
    __global const int* restrict edge_index,
    __global const $REAL* restrict multiplier_real,
    __global const $REAL* restrict multiplier_imag,
    int field_offset,
    __global const $REAL* restrict field,
    __global $REAL* output_real,
    __global $REAL* output_imag)
"""
    ),
    "metal": Template(
        """
kernel void accumulate_sar(
    device const int& total, device const int& nedges,
    device const int* edge_index,
    device const $REAL* multiplier_real,
    device const $REAL* multiplier_imag,
    device const $REAL* field,
    device $REAL* output_real,
    device $REAL* output_imag,
    uint i [[thread_position_in_grid]])
"""
    ),
}


def build_sar_kernel_source(backend: str, c_real: str) -> str:
    """Render the sparse SAR DFT kernel for one accelerator backend."""

    if backend not in _ACCUMULATE_SAR:
        raise ValueError("backend must be 'cuda', 'opencl', or 'metal'")
    declaration = _ACCUMULATE_SAR[backend].substitute(REAL=c_real)
    if backend == "cuda":
        preamble = ""
        index = "int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        field_offset = "0"
    elif backend == "opencl":
        preamble = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" if c_real == "double" else ""
        index = "int i = get_global_id(0);\n"
        field_offset = "field_offset"
    else:
        preamble = "#include <metal_stdlib>\nusing namespace metal;\n"
        index = ""
        field_offset = "0"
    body = f"""
    if (i < total) {{
        int frequency = i / nedges;
        int edge = i - frequency * nedges;
        $REAL value = field[{field_offset} + edge_index[edge]];
        output_real[i] += multiplier_real[frequency] * value;
        output_imag[i] += multiplier_imag[frequency] * value;
    }}
""".replace(
        "$REAL", c_real
    )
    return f"{preamble}{declaration}{{\n{index}{body}}}\n"
