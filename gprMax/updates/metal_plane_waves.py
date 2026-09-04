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

"""Metal launch adapter for the shared discrete-plane-wave kernels."""

import re

import numpy as np

from gprMax import config
from gprMax.grid.metal_grid import MetalBufferView
from gprMax.updates.opencl_updates import OpenCLUpdates


_METAL_MAX_BUFFER_ARGUMENTS = 31


def _prepare_planewave_kernel_source(specification, knl_common, c_real, c_complex):
    """Render a plane-wave kernel and pack scalars if Metal requires it.

    CUDA and OpenCL accept the flat auxiliary-grid PML signatures, but some
    of them contain more resources than Metal's 31-buffer limit. Their scalar
    arguments form a contiguous prefix, so Metal can receive that prefix as a
    single constant structure while keeping the shared numerical kernel body
    and the backend-neutral launch sequence unchanged.

    Returns:
        tuple[str, int]: Rendered Metal source and the number of leading host
        arguments packed into the constant structure (zero when packing is
        unnecessary).
    """

    declaration = specification["args_metal"].substitute(
        {"REAL": c_real, "COMPLEX": c_complex}
    )
    body = specification["func"].substitute(
        {"CUDA_IDX": "", "TFSF_IDX": "", "REAL": c_real}
    )

    function_match = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", declaration)
    if function_match is None:
        raise ValueError("Could not identify the Metal plane-wave kernel declaration")
    function_name = function_match.group(1)
    parameters_start = declaration.find("(", function_match.start()) + 1
    parameters_end = declaration.rfind(")")
    parameters = [
        parameter.strip()
        for parameter in declaration[parameters_start:parameters_end].split(",")
        if parameter.strip()
    ]
    resource_parameters = [
        parameter for parameter in parameters if "thread_position_in_grid" not in parameter
    ]
    if len(resource_parameters) <= _METAL_MAX_BUFFER_ARGUMENTS:
        return f"{knl_common}\n{declaration}{{\n{body}}}\n", 0

    scalar_fields = []
    for parameter in resource_parameters:
        scalar_match = re.fullmatch(
            r"device\s+const\s+(.+?)\s*&\s*([A-Za-z_]\w*)", parameter
        )
        if scalar_match is None:
            break
        scalar_fields.append(scalar_match.groups())

    packed_scalar_count = len(scalar_fields)
    packed_resource_count = 1 + len(resource_parameters) - packed_scalar_count
    if packed_scalar_count == 0 or packed_resource_count > _METAL_MAX_BUFFER_ARGUMENTS:
        raise ValueError(
            f"Metal plane-wave kernel {function_name} requires "
            f"{len(resource_parameters)} buffer arguments and cannot be reduced below "
            f"the {_METAL_MAX_BUFFER_ARGUMENTS}-buffer limit by packing its scalar prefix"
        )

    struct_name = f"_MetalScalars_{function_name}"
    struct_fields = "\n".join(f"    {field_type} {name};" for field_type, name in scalar_fields)
    aliases = "\n".join(
        f"    const {field_type} {name} = _metal_scalars.{name};"
        for field_type, name in scalar_fields
    )
    packed_parameters = [
        f"constant {struct_name}& _metal_scalars",
        *parameters[packed_scalar_count:],
    ]
    packed_declaration = (
        declaration[:parameters_start]
        + "\n            "
        + ",\n            ".join(packed_parameters)
        + declaration[parameters_end:]
    )
    packed_struct = f"struct {struct_name} {{\n{struct_fields}\n}};"
    source = f"{knl_common}\n{packed_struct}\n{packed_declaration}{{\n{aliases}\n{body}}}\n"
    return source, packed_scalar_count


def _render_planewave_kernel_source(specification, knl_common, c_real, c_complex):
    """Render a shared plane-wave kernel for Metal."""

    return _prepare_planewave_kernel_source(
        specification, knl_common, c_real, c_complex
    )[0]


class _MetalElementwiseKernel:
    """Small callable matching the PyOpenCL ElementwiseKernel launch API."""

    def __init__(self, owner, pipeline, packed_scalar_count=0):
        self.owner = owner
        self.pipeline = pipeline
        self.packed_scalar_count = packed_scalar_count

    @staticmethod
    def _buffer_and_offset(value):
        if isinstance(value, MetalBufferView):
            return value.buffer, value.offset
        if hasattr(value, "contents") and hasattr(value, "length"):
            return value, 0
        return None

    def __call__(self, *arguments, range=None):
        if range is None:
            raise ValueError("Metal elementwise kernels require an explicit range")
        count = int(range.stop) - int(range.start or 0)
        if count <= 0:
            return
        command = self.owner.cmdqueue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        argument_offset = 0
        metal_index = 0
        if self.packed_scalar_count:
            packed_scalars = []
            for argument in arguments[: self.packed_scalar_count]:
                scalar = np.asarray(argument)
                if scalar.ndim != 0:
                    raise TypeError("Packed Metal plane-wave arguments must be scalars")
                packed_scalars.append(scalar.tobytes())
            packed_data = b"".join(packed_scalars)
            encoder.setBytes_length_atIndex_(packed_data, len(packed_data), metal_index)
            argument_offset = self.packed_scalar_count
            metal_index += 1

        for index, argument in enumerate(arguments[argument_offset:], start=metal_index):
            view = self._buffer_and_offset(argument)
            if view is not None:
                encoder.setBuffer_offset_atIndex_(view[0], view[1], index)
            else:
                scalar = np.asarray(argument)
                encoder.setBytes_length_atIndex_(scalar.tobytes(), scalar.dtype.itemsize, index)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.owner.metal.MTLSizeMake(count, 1, 1),
            self.owner.metal.MTLSizeMake(
                min(count, self.pipeline.maxTotalThreadsPerThreadgroup()), 1, 1
            ),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()


class MetalPlaneWaveController(OpenCLUpdates):
    """Reuse backend-neutral OpenCL plane-wave sequencing with Metal launches."""

    def __init__(self, updates):
        # Deliberately do not call OpenCLUpdates.__init__. Its plane-wave
        # orchestration is backend-neutral once arrays and elementwise
        # launches expose the small interfaces provided here.
        self.grid = updates.grid
        self.dev = updates.dev
        self.cmdqueue = updates.cmdqueue
        self.queue = updates.cmdqueue
        self.metal = updates.metal
        self.opts = updates.opts
        self.knl_common = updates.knl_common
        self._planewave_kernel_cache = {}
        self._set_planewave_knls()

    def _planewave_kernel(self, specification, name):
        try:
            return self._planewave_kernel_cache[name]
        except KeyError:
            pass
        c_real = config.sim_config.dtypes["C_float_or_double"]
        source, packed_scalar_count = _prepare_planewave_kernel_source(
            specification,
            self.knl_common,
            c_real,
            self.grid.dispersiveCdtype,
        )
        library, error = self.dev.newLibraryWithSource_options_error_(source, self.opts, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal plane-wave kernel {name}: {error}")
        function = library.newFunctionWithName_(name)
        pipeline = self.dev.newComputePipelineStateWithFunction_error_(function, None)[0]
        kernel = _MetalElementwiseKernel(self, pipeline, packed_scalar_count)
        self._planewave_kernel_cache[name] = kernel
        return kernel
