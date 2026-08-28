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

import numpy as np

from gprMax import config
from gprMax.grid.metal_grid import MetalBufferView
from gprMax.updates.opencl_updates import OpenCLUpdates


def _render_planewave_kernel_source(specification, knl_common, c_real, c_complex):
    """Render a shared plane-wave kernel for Metal."""

    declaration = specification["args_metal"].substitute({"REAL": c_real, "COMPLEX": c_complex})
    body = specification["func"].substitute({"CUDA_IDX": "", "TFSF_IDX": "", "REAL": c_real})
    return f"{knl_common}\n{declaration}{{\n{body}}}\n"


class _MetalElementwiseKernel:
    """Small callable matching the PyOpenCL ElementwiseKernel launch API."""

    def __init__(self, owner, pipeline):
        self.owner = owner
        self.pipeline = pipeline

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
        for index, argument in enumerate(arguments):
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
        source = _render_planewave_kernel_source(
            specification,
            self.knl_common,
            c_real,
            config.get_model_config().materials["dispersiveCdtype"],
        )
        library, error = self.dev.newLibraryWithSource_options_error_(source, self.opts, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal plane-wave kernel {name}: {error}")
        function = library.newFunctionWithName_(name)
        pipeline = self.dev.newComputePipelineStateWithFunction_error_(function, None)[0]
        kernel = _MetalElementwiseKernel(self, pipeline)
        self._planewave_kernel_cache[name] = kernel
        return kernel
