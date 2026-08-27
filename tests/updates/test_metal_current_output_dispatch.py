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

"""Metal receiver-current dispatch contract without requiring Metal hardware."""

from types import SimpleNamespace

import numpy as np

from gprMax import config
from gprMax.updates.metal_updates import MetalUpdates


class _Buffer:
    pass


class _Device:
    def newBufferWithBytes_length_options_(self, data, length, options):
        return _Buffer()


class _Encoder:
    def __init__(self):
        self.pso = None
        self.buffers = {}
        self.dispatched = False
        self.ended = False

    def setComputePipelineState_(self, pso):
        self.pso = pso

    def setBuffer_offset_atIndex_(self, buffer, offset, index):
        self.buffers[index] = buffer

    def dispatchThreads_threadsPerThreadgroup_(self, grid_size, group_size):
        self.dispatched = True

    def endEncoding(self):
        self.ended = True


class _CommandBuffer:
    def __init__(self):
        self.encoders = []
        self.committed = False
        self.waited = False

    def computeCommandEncoder(self):
        encoder = _Encoder()
        self.encoders.append(encoder)
        return encoder

    def commit(self):
        self.committed = True

    def waitUntilCompleted(self):
        self.waited = True


class _Queue:
    def __init__(self):
        self.buffer = None

    def commandBuffer(self):
        self.buffer = _CommandBuffer()
        return self.buffer


class _PSO:
    def maxTotalThreadsPerThreadgroup(self):
        return 64


class _Metal:
    @staticmethod
    def MTLSizeMake(x, y, z):
        return (x, y, z)


def test_current_output_uses_second_encoder_and_all_ten_kernel_arguments(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"float_or_double": np.float32}),
    )
    updates = MetalUpdates.__new__(MetalUpdates)
    updates.dev = _Device()
    updates.cmdqueue = _Queue()
    updates.metal = _Metal()
    updates.pso_store_outputs = _PSO()
    updates.pso_store_current_outputs = _PSO()
    updates.nrxcurrent = 2
    updates.rxcoords_dev = _Buffer()
    updates.rxs_dev = _Buffer()
    updates.rxcurrentinfo_dev = _Buffer()
    updates.rxcurrents_dev = _Buffer()

    fields = {name: _Buffer() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    updates.grid = SimpleNamespace(
        rxs=(object(),),
        dx=1e-3,
        dy=2e-3,
        dz=3e-3,
        **{f"{name}_dev": value for name, value in fields.items()},
    )

    updates.store_outputs(iteration=3)

    command = updates.cmdqueue.buffer
    assert command.committed and command.waited
    assert len(command.encoders) == 2
    current = command.encoders[1]
    assert current.pso is updates.pso_store_current_outputs
    assert current.dispatched and current.ended
    assert set(current.buffers) == set(range(10))
    assert current.buffers[2] is updates.rxcurrentinfo_dev
    assert current.buffers[3] is updates.rxcurrents_dev
    assert current.buffers[7] is fields["Hx"]
    assert current.buffers[8] is fields["Hy"]
    assert current.buffers[9] is fields["Hz"]
