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

"""Regression test for the Metal voltage-source kernel argument
misalignment (gprMax/updates/metal_updates.py's update_electric_sources(),
voltage-source branch) - found by an external Codex review.

knl_source_updates.update_voltage_source's args_metal signature declares
12 arguments: NVOLTSRC, iteration, dx, dy, dz, srcinfo1, srcinfo2,
srcwaveforms, ID, Ex, Ey, Ez (indices 0-11). The dispatch code only bound
8 buffers, at the wrong indices - dx/dy/dz and ID were never bound at
all, and everything from srcinfo1 onward was shifted 4 slots earlier
(srcinfo1 landed at index 2 instead of 5, Ex at index 5 instead of 9,
etc.) - so the kernel would read/write completely wrong buffers for
every voltage source on Metal.

Fixed by adding the missing dx/dy/dz/ID bindings and moving every
subsequent buffer to its correct index, mirroring the already-correct
Hertzian-dipole/magnetic-dipole dispatch pattern in this same file.

Real Apple Metal hardware/PyObjC isn't available in this environment, so
this test exercises update_electric_sources()'s voltage-source branch
directly against a MetalUpdates instance built with __new__ (bypassing
__init__) and fake Metal API stand-ins, checking the buffer-index-to-
kernel-argument contract directly.
"""
from types import SimpleNamespace

import numpy as np

from gprMax import config
from gprMax.updates.metal_updates import MetalUpdates


class _FakeBuffer:
    def __init__(self, data=b""):
        self.data = bytes(data)


class _FakeDevice:
    def newBufferWithBytes_length_options_(self, data, length, options):
        return _FakeBuffer(data)


class _FakeEncoder:
    def __init__(self):
        self.pso = None
        self.buffers = {}
        self.dispatched = False
        self.ended = False

    def setComputePipelineState_(self, pso):
        self.pso = pso

    def setBuffer_offset_atIndex_(self, buf, offset, index):
        self.buffers[index] = buf

    def dispatchThreads_threadsPerThreadgroup_(self, grid_size, group_size):
        self.dispatched = True

    def endEncoding(self):
        self.ended = True


class _FakeCommandBuffer:
    def __init__(self):
        self.encoder = _FakeEncoder()
        self.committed = False
        self.waited = False

    def computeCommandEncoder(self):
        return self.encoder

    def commit(self):
        self.committed = True

    def waitUntilCompleted(self):
        self.waited = True


class _FakeCommandQueue:
    def __init__(self):
        self.buffers_created = []

    def commandBuffer(self):
        buf = _FakeCommandBuffer()
        self.buffers_created.append(buf)
        return buf


class _FakePSO:
    def maxTotalThreadsPerThreadgroup(self):
        return 64


class _FakeMetalModule:
    def MTLSizeMake(self, x, y, z):
        return (x, y, z)


class _FakeVoltageSource:
    pass


def test_update_electric_sources_voltage_source_binds_all_12_args_correctly(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"float_or_double": np.float32}),
    )
    updates = MetalUpdates.__new__(MetalUpdates)
    updates.dev = _FakeDevice()
    updates.cmdqueue = _FakeCommandQueue()
    updates.metal = _FakeMetalModule()
    updates.pso_voltage_source = _FakePSO()
    updates.srcinfo1_voltage_dev = _FakeBuffer(b"srcinfo1")
    updates.srcinfo2_voltage_dev = _FakeBuffer(b"srcinfo2")
    updates.srcwaves_voltage_dev = _FakeBuffer(b"srcwaves")

    class _Grid:
        voltagesources = [_FakeVoltageSource()]
        hertziandipoles = []
        dx = dy = dz = 1e-3
        iteration = 0
        ID_dev = _FakeBuffer(b"id")
        Ex_dev = _FakeBuffer(b"ex")
        Ey_dev = _FakeBuffer(b"ey")
        Ez_dev = _FakeBuffer(b"ez")

    updates.grid = _Grid()

    updates.update_electric_sources(iteration=1)

    assert len(updates.cmdqueue.buffers_created) == 1
    encoder = updates.cmdqueue.buffers_created[0].encoder
    assert encoder.pso is updates.pso_voltage_source
    assert encoder.dispatched
    assert encoder.ended

    # Buffer index contract must match knl_source_updates.update_voltage_source's
    # args_metal signature: NVOLTSRC, iteration, dx, dy, dz, srcinfo1,
    # srcinfo2, srcwaveforms, ID, Ex, Ey, Ez (indices 0-11) - all 12 must
    # be bound, not just 8.
    assert set(encoder.buffers.keys()) == set(range(12))
    assert encoder.buffers[5] is updates.srcinfo1_voltage_dev
    assert encoder.buffers[6] is updates.srcinfo2_voltage_dev
    assert encoder.buffers[7] is updates.srcwaves_voltage_dev
    assert encoder.buffers[8] is updates.grid.ID_dev
    assert encoder.buffers[9] is updates.grid.Ex_dev
    assert encoder.buffers[10] is updates.grid.Ey_dev
    assert encoder.buffers[11] is updates.grid.Ez_dev
