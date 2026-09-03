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

"""Backend-neutral plane-wave and OpenCL NTFF kernel-source regressions."""

import re
from types import SimpleNamespace

import numpy as np

from gprMax.cuda_opencl import knl_planewave_updates, knl_tfsf_injection
from gprMax.cuda_opencl.knl_ntff import build_ntff_kernel_source
from gprMax.grid.metal_grid import MetalBufferView
from gprMax.updates.metal_plane_waves import (
    _MetalElementwiseKernel,
    _prepare_planewave_kernel_source,
    _render_planewave_kernel_source,
)


def _plane_wave_kernel_specs(module):
    return [
        value
        for value in vars(module).values()
        if isinstance(value, dict) and {"args_metal", "func"}.issubset(value)
    ]


def _metal_parameter_names(declaration):
    """Return the parameter names from a rendered Metal kernel declaration."""

    parameters = declaration[declaration.find("(") + 1 : declaration.rfind(")")]
    names = []
    for parameter in parameters.split(","):
        parameter = re.sub(r"\[\[.*?\]\]", "", parameter).strip()
        match = re.search(r"([A-Za-z_]\w*)\s*$", parameter)
        assert match is not None, f"Could not parse Metal parameter: {parameter}"
        names.append(match.group(1))
    return names


def test_tfsf_kernels_use_backend_index_substitution():
    kernels = (
        knl_tfsf_injection.STANDARD_H_KERNELS
        + knl_tfsf_injection.STANDARD_E_KERNELS
        + knl_tfsf_injection.AXIAL_H_KERNELS
        + knl_tfsf_injection.AXIAL_E_KERNELS
    )

    assert len(kernels) == 24
    for kernel in kernels:
        body = kernel["func"].template
        assert "$TFSF_IDX" in body
        assert "blockIdx.x" not in body


def test_axial_opencl_kernels_receive_material_coefficients():
    names = (
        "update_1d_magnetic_axial_source",
        "update_1d_magnetic_axial_source_pml",
        "update_1d_magnetic_axial_inject",
        "update_1d_magnetic_axial_main",
        "update_1d_magnetic_axial_main_pml_end",
        "update_1d_magnetic_axial_main_pml_start",
        "update_1d_electric_axial_source",
        "update_1d_electric_axial_source_pml",
        "update_1d_electric_axial_inject",
        "update_1d_electric_axial_main",
        "update_1d_electric_axial_main_pml_end",
        "update_1d_electric_axial_main_pml_start",
    )

    for name in names:
        arguments = getattr(knl_planewave_updates, name)["args_opencl"].template
        assert "matH" in arguments
        assert "matE" in arguments


def test_all_plane_wave_templates_render_for_metal():
    specs = _plane_wave_kernel_specs(knl_planewave_updates)
    specs += (
        knl_tfsf_injection.STANDARD_H_KERNELS
        + knl_tfsf_injection.STANDARD_E_KERNELS
        + knl_tfsf_injection.AXIAL_H_KERNELS
        + knl_tfsf_injection.AXIAL_E_KERNELS
    )

    assert len(specs) > 24
    for specification in specs:
        declaration = specification["args_metal"].substitute(
            {"REAL": "float", "COMPLEX": "metal::complex<float>"}
        )
        body = specification["func"].substitute({"CUDA_IDX": "", "TFSF_IDX": "", "REAL": "float"})
        source = _render_planewave_kernel_source(
            specification, "", "float", "metal::complex<float>"
        )
        prepared_source, packed_scalar_count = _prepare_planewave_kernel_source(
            specification, "", "float", "metal::complex<float>"
        )

        parameter_names = _metal_parameter_names(declaration)

        assert source == prepared_source
        assert "$" not in declaration
        assert "$" not in body
        assert "thread_position_in_grid" in declaration
        assert "blockIdx" not in body
        assert len(parameter_names) == len(set(parameter_names))

        resource_count = len(parameter_names) - 1
        if resource_count <= 31:
            assert packed_scalar_count == 0
            assert source == f"\n{declaration}{{\n{body}}}\n"
        else:
            assert packed_scalar_count > 0
            rendered_declaration = re.search(
                r"kernel\s+void\s+\w+\s*\(.*?\)", source, re.DOTALL
            ).group(0)
            rendered_parameter_names = _metal_parameter_names(rendered_declaration)
            assert len(rendered_parameter_names) - 1 <= 31
            assert rendered_parameter_names[0] == "_metal_scalars"
            for name in parameter_names[:packed_scalar_count]:
                assert f"_metal_scalars.{name}" in source
        if "$TFSF_IDX" in specification["func"].template:
            assert parameter_names.count("t") == 1
            assert "int t =" not in body


class _FakeEncoder:
    def __init__(self):
        self.buffers = {}
        self.scalars = {}
        self.dispatched = None

    def setComputePipelineState_(self, pipeline):
        self.pipeline = pipeline

    def setBuffer_offset_atIndex_(self, buffer, offset, index):
        self.buffers[index] = (buffer, offset)

    def setBytes_length_atIndex_(self, value, length, index):
        self.scalars[index] = (bytes(value), length)

    def dispatchThreads_threadsPerThreadgroup_(self, threads, group):
        self.dispatched = (threads, group)

    def endEncoding(self):
        pass


class _FakeCommand:
    def __init__(self):
        self.encoder = _FakeEncoder()

    def computeCommandEncoder(self):
        return self.encoder

    def commit(self):
        pass

    def waitUntilCompleted(self):
        pass


def test_metal_plane_wave_adapter_preserves_buffer_offsets_and_scalar_types():
    command = _FakeCommand()
    owner = SimpleNamespace(
        cmdqueue=SimpleNamespace(commandBuffer=lambda: command),
        metal=SimpleNamespace(MTLSizeMake=lambda x, y, z: (x, y, z)),
    )
    pipeline = SimpleNamespace(maxTotalThreadsPerThreadgroup=lambda: 64)
    kernel = _MetalElementwiseKernel(owner, pipeline)
    raw_buffer = SimpleNamespace(contents=lambda: None, length=lambda: 128)
    view = MetalBufferView(raw_buffer, 24)

    kernel(np.int32(7), view, range=slice(0, 13))

    assert command.encoder.scalars[0][1] == np.dtype(np.int32).itemsize
    assert command.encoder.buffers[1] == (raw_buffer, 24)
    assert command.encoder.dispatched == ((13, 1, 1), (13, 1, 1))


def test_metal_plane_wave_adapter_packs_scalar_prefix_into_one_buffer_slot():
    command = _FakeCommand()
    owner = SimpleNamespace(
        cmdqueue=SimpleNamespace(commandBuffer=lambda: command),
        metal=SimpleNamespace(MTLSizeMake=lambda x, y, z: (x, y, z)),
    )
    pipeline = SimpleNamespace(maxTotalThreadsPerThreadgroup=lambda: 64)
    kernel = _MetalElementwiseKernel(owner, pipeline, packed_scalar_count=2)
    raw_buffer = SimpleNamespace(contents=lambda: None, length=lambda: 128)

    kernel(np.int32(7), np.float32(1.25), raw_buffer, range=slice(0, 13))

    packed, packed_length = command.encoder.scalars[0]
    assert packed_length == 8
    assert np.frombuffer(packed[:4], dtype=np.int32)[0] == 7
    assert np.frombuffer(packed[4:], dtype=np.float32)[0] == np.float32(1.25)
    assert command.encoder.buffers[1] == (raw_buffer, 0)


def test_opencl_ntff_source_accepts_offset_field_views():
    opencl = build_ntff_kernel_source("opencl", "float")
    cuda = build_ntff_kernel_source("cuda", "float")

    assert "int field_offset" in opencl
    assert "field[field_offset + inside_index[patch]]" in opencl
    assert "field[0 + inside_index[patch]]" in cuda
