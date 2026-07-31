"""Regression test for gprMax/updates/metal_updates.py's
MetalUpdates.update_magnetic_sources() - found by an external Codex review
and confirmed by reading the code: the method checked
`if self.grid.magneticdipoles:` but the body was a bare `pass` with a
"TODO: Implement Metal compute pipeline execution" comment - magnetic
dipole sources were silently never excited on Metal, despite the
underlying kernel (knl_source_updates.update_magnetic_dipole, with a full
args_metal template) already being compiled into a pipeline state object
by _set_src_knls().

Fixed by implementing the dispatch, mirroring update_electric_sources()'s
existing (working) Hertzian-dipole dispatch pattern exactly - same 12
kernel arguments (NMAGDIPOLE, iteration, dx, dy, dz, srcinfo1, srcinfo2,
srcwaveforms, ID, then the three field components) at the same buffer
indices 0-11, just Hx/Hy/Hz instead of Ex/Ey/Ez.

Real Apple Metal hardware/PyObjC isn't available in this environment, so
this test exercises update_magnetic_sources() directly against a
MetalUpdates instance built with __new__ (bypassing __init__, which
imports the real "Metal" module) and fake stand-ins for the Metal API
objects it calls (command queue/buffer/encoder, device, pipeline state),
recording every setBuffer_offset_atIndex_ call so the buffer-index
contract against the kernel signature can be checked directly.
"""
from types import SimpleNamespace

import numpy as np

from gprMax import config
from gprMax.updates.metal_updates import MetalUpdates


class _FakeBuffer:
    def __init__(self, data):
        self.data = data


class _FakeDevice:
    def newBufferWithBytes_length_options_(self, data, length, options):
        return _FakeBuffer(bytes(data))


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


class _FakeMagneticDipole:
    pass


def _make_updates(magnetic_dipole_list):
    updates = MetalUpdates.__new__(MetalUpdates)
    updates.dev = _FakeDevice()
    updates.cmdqueue = _FakeCommandQueue()
    updates.metal = _FakeMetalModule()
    updates.pso_magnetic_dipole = _FakePSO()
    updates.srcinfo1_magnetic_dev = _FakeBuffer(b"srcinfo1")
    updates.srcinfo2_magnetic_dev = _FakeBuffer(b"srcinfo2")
    updates.srcwaves_magnetic_dev = _FakeBuffer(b"srcwaves")

    class _Grid:
        magneticdipoles = magnetic_dipole_list
        magneticfrillsources = []
        dx = dy = dz = 1e-3
        ID_dev = _FakeBuffer(b"id")
        Hx_dev = _FakeBuffer(b"hx")
        Hy_dev = _FakeBuffer(b"hy")
        Hz_dev = _FakeBuffer(b"hz")

    updates.grid = _Grid()
    return updates


def test_update_magnetic_sources_dispatches_kernel_with_correct_buffer_order(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"float_or_double": np.float32}),
    )
    updates = _make_updates([_FakeMagneticDipole()])

    updates.update_magnetic_sources(iteration=1)

    assert len(updates.cmdqueue.buffers_created) == 1
    encoder = updates.cmdqueue.buffers_created[0].encoder
    assert encoder.pso is updates.pso_magnetic_dipole
    assert encoder.dispatched
    assert encoder.ended
    assert updates.cmdqueue.buffers_created[0].committed
    assert updates.cmdqueue.buffers_created[0].waited

    # Buffer index contract must match knl_source_updates.update_magnetic_dipole's
    # args_metal signature: N, iteration, dx, dy, dz, srcinfo1, srcinfo2,
    # srcwaveforms, ID, Hx, Hy, Hz (indices 0-11).
    assert set(encoder.buffers.keys()) == set(range(12))
    assert encoder.buffers[5] is updates.srcinfo1_magnetic_dev
    assert encoder.buffers[6] is updates.srcinfo2_magnetic_dev
    assert encoder.buffers[7] is updates.srcwaves_magnetic_dev
    assert encoder.buffers[8] is updates.grid.ID_dev
    assert encoder.buffers[9] is updates.grid.Hx_dev
    assert encoder.buffers[10] is updates.grid.Hy_dev
    assert encoder.buffers[11] is updates.grid.Hz_dev


def test_update_magnetic_sources_no_op_without_magnetic_dipoles():
    updates = _make_updates([])

    updates.update_magnetic_sources(iteration=1)

    assert updates.cmdqueue.buffers_created == []
