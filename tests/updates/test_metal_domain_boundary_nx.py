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

"""Regression test for the Metal domain-boundary field corruption bug
(gprMax/updates/metal_updates.py's update_magnetic()/update_electric_a()) -
found in the earlier deep bug audit.

knl_fields_updates.py's update_electric/update_magnetic kernels take NX,
NY, NZ as their first three arguments and use them directly as
bounds-check comparisons (e.g. "x < NX"), meaning NX/NY/NZ must be the
raw cell count - CUDA/OpenCL correctly pass np.int32(self.grid.nx) (no
+1). Metal was passing self.grid.nx + 1 (the *field-array* dimension)
instead, letting every boundary-plane bounds check admit one extra plane
it shouldn't - corrupting domain-boundary field values on every Metal
run, and doubling the intentional single-plane H-wall-symmetry widening
into a two-plane widening on Metal specifically.

Fixed by dropping the "+ 1" at all three call sites (update_magnetic, and
both branches of update_electric_a).

Real Apple Metal hardware/PyObjC isn't available in this environment, so
this test exercises update_magnetic()/update_electric_a() directly
against a MetalUpdates instance built with __new__ (bypassing __init__)
and fake Metal API stand-ins, checking the exact int32 bytes passed via
setBytes_length_atIndex_ for indices 0/1/2 (NX/NY/NZ).
"""
import numpy as np

from gprMax import config
from gprMax.updates.metal_updates import MetalUpdates


class _FakeBuffer:
    pass


class _FakeEncoder:
    def __init__(self):
        self.pso = None
        self.bytes_args = {}
        self.buffers = {}
        self.dispatched = False
        self.ended = False

    def setComputePipelineState_(self, pso):
        self.pso = pso

    def setBytes_length_atIndex_(self, data, length, index):
        self.bytes_args[index] = np.frombuffer(data, dtype=np.int32)[0]

    def setBuffer_offset_atIndex_(self, buf, offset, index):
        self.buffers[index] = buf

    def dispatchThreads_threadsPerThreadgroup_(self, grid_size, group_size):
        self.dispatched = True

    def endEncoding(self):
        self.ended = True


class _FakeCommandBuffer:
    def __init__(self):
        self.encoder = _FakeEncoder()

    def computeCommandEncoder(self):
        return self.encoder

    def commit(self):
        pass

    def waitUntilCompleted(self):
        pass


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


def _make_updates(monkeypatch, maxpoles=0):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: type("_MC", (), {"materials": {"maxpoles": maxpoles}})(),
    )

    updates = MetalUpdates.__new__(MetalUpdates)
    updates.cmdqueue = _FakeCommandQueue()
    updates.metal = _FakeMetalModule()
    updates.psoE = "psoE"
    updates.psoH = "psoH"
    updates.pso_dispersive_a = _FakePSO()
    updates.pso_dispersive_b = _FakePSO()

    class _Grid:
        nx, ny, nz = 10, 12, 14
        tptg = "tptg"
        tgs = "tgs"
        ID_dev = _FakeBuffer()
        Ex_dev = _FakeBuffer()
        Ey_dev = _FakeBuffer()
        Ez_dev = _FakeBuffer()
        Hx_dev = _FakeBuffer()
        Hy_dev = _FakeBuffer()
        Hz_dev = _FakeBuffer()
        updatecoeffsdispersive_dev = _FakeBuffer()
        Tx_dev = _FakeBuffer()
        Ty_dev = _FakeBuffer()
        Tz_dev = _FakeBuffer()

    updates.grid = _Grid()
    updates.grid.maxpoles = maxpoles
    return updates


def test_update_magnetic_passes_raw_cell_count_not_field_array_dimension(monkeypatch):
    updates = _make_updates(monkeypatch)

    updates.update_magnetic()

    encoder = updates.cmdqueue.buffers_created[0].encoder
    assert encoder.bytes_args[0] == updates.grid.nx
    assert encoder.bytes_args[1] == updates.grid.ny
    assert encoder.bytes_args[2] == updates.grid.nz


def test_update_electric_a_non_dispersive_passes_raw_cell_count(monkeypatch):
    updates = _make_updates(monkeypatch, maxpoles=0)

    updates.update_electric_a()

    encoder = updates.cmdqueue.buffers_created[0].encoder
    assert encoder.bytes_args[0] == updates.grid.nx
    assert encoder.bytes_args[1] == updates.grid.ny
    assert encoder.bytes_args[2] == updates.grid.nz


def test_update_electric_a_dispersive_passes_raw_cell_count(monkeypatch):
    """maxpoles > 0 now dispatches the real dispersive kernel (see
    test_metal_dispersive_dispatch.py) rather than a non-dispersive
    fallback - confirm the same NX/NY/NZ fix applies there too."""
    updates = _make_updates(monkeypatch, maxpoles=1)

    updates.update_electric_a()

    encoder = updates.cmdqueue.buffers_created[0].encoder
    assert encoder.bytes_args[0] == updates.grid.nx
    assert encoder.bytes_args[1] == updates.grid.ny
    assert encoder.bytes_args[2] == updates.grid.nz
