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

"""Regression test for Metal snapshot support in
gprMax/updates/metal_updates.py - found by an external Codex review,
confirmed as a second, distinct (and currently-masked) bug from the
already-documented "Snapshots are non-functional" one:

1. `_set_snapshot_knl()` was never called (commented out in __init__) and,
   even if it had been, its body was verbatim-copied OpenCL code
   (self.queue/self.ctx/self.elwiseknl - none of which exist on
   MetalUpdates) plus a wrong call (`htod_snapshot_array(self.grid, ...)`
   instead of `htod_snapshot_array(self.grid.snapshots, ...)`). This meant
   store_snapshots() crashed with AttributeError on the very first
   iteration any snapshot triggered - masking bug 2 below.
2. store_snapshots()/finalise() both called .get() on the snapshot
   buffers - the CUDA/OpenCL array API. Metal buffers (MTLBuffer) have no
   .get(); they are read back via .contents().as_buffer(size), as used
   correctly elsewhere in this file (e.g. dtoh_rx_array's Metal branch).

Fixed by: rewriting _set_snapshot_knl() to build the kernel via the
existing _build_knl()/pipeline-state-object pattern (matching
_set_src_knls), uncommenting its call in __init__, rewriting
store_snapshots() to dispatch via a real Metal command
buffer/encoder (matching the working Hertzian-dipole/magnetic-dipole
dispatch pattern) instead of an OpenCL elementwise-kernel call, and adding
_metal_snapshot_buffers_to_numpy() (a single conversion helper) used by
both store_snapshots()'s immediate-readback path and finalise().

Real Apple Metal hardware/PyObjC isn't available in this environment, so
this test builds a MetalUpdates instance with __new__ (bypassing
__init__) and fake stand-ins for the Metal API objects involved.
"""
import numpy as np

from gprMax import config
from gprMax.snapshots import Snapshot
from gprMax.updates import metal_updates as metal_updates_mod
from gprMax.updates.metal_updates import MetalUpdates


class _FakeBuffer:
    def __init__(self, data=b""):
        self.data = bytes(data)

    def contents(self):
        return self

    def as_buffer(self, size):
        return self.data


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


class _FakeSnap:
    def __init__(self, time):
        self.time = time
        self.xs = self.ys = self.zs = 0
        self.nx = self.ny = self.nz = 2
        self.dx = self.dy = self.dz = 1
        self.snapfields = {}


def _make_updates(snapshots, snapsgpu2cpu, monkeypatch):
    monkeypatch.setattr(Snapshot, "nx_max", 2)
    monkeypatch.setattr(Snapshot, "ny_max", 2)
    monkeypatch.setattr(Snapshot, "nz_max", 2)
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.dtypes = {"float_or_double": np.float64}
    monkeypatch.setattr(config, "get_model_config", lambda: type(
        "_MC", (), {"device": {"snapsgpu2cpu": snapsgpu2cpu}, "mode": "3D"}
    )())

    updates = MetalUpdates.__new__(MetalUpdates)
    updates.dev = _FakeDevice()
    updates.cmdqueue = _FakeCommandQueue()
    updates.metal = _FakeMetalModule()
    updates.pso_store_snapshot = _FakePSO()

    numsnaps = 1 if snapsgpu2cpu else len(snapshots)
    shape = (numsnaps, 2, 2, 2)
    known = {
        name: np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + offset
        for offset, name in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
    }
    for name, arr in known.items():
        setattr(updates, f"snap{name}_dev", _FakeBuffer(arr.tobytes()))

    class _Grid:
        pass

    grid = _Grid()
    grid.snapshots = snapshots
    grid.magneticfrillsources = []
    grid.Ex_dev = _FakeBuffer()
    grid.Ey_dev = _FakeBuffer()
    grid.Ez_dev = _FakeBuffer()
    grid.Hx_dev = _FakeBuffer()
    grid.Hy_dev = _FakeBuffer()
    grid.Hz_dev = _FakeBuffer()
    updates.grid = grid

    return updates, known


def test_store_snapshots_dispatches_with_correct_buffer_order(monkeypatch):
    snap = _FakeSnap(time=1)
    updates, _ = _make_updates([snap], snapsgpu2cpu=False, monkeypatch=monkeypatch)

    updates.store_snapshots(iteration=0)

    assert len(updates.cmdqueue.buffers_created) == 1
    encoder = updates.cmdqueue.buffers_created[0].encoder
    assert encoder.pso is updates.pso_store_snapshot
    assert encoder.dispatched
    assert encoder.ended

    # 13 scalar args (p, xs, ys, zs, nx, ny, nz, dx, dy, dz, sx, sy, sz) at
    # indices 0-12, then 12 field/snapshot buffers at indices 13-24,
    # matching knl_snapshots.store_snapshot's args_metal signature (nx/ny/nz
    # are this snapshot's own local sample counts, not the absolute finish
    # coordinate - see project_deep_bug_audit.md's "GPU snapshot indexing"
    # entry for why comparing against an absolute coordinate was wrong).
    assert set(encoder.buffers.keys()) == set(range(25))
    assert encoder.buffers[13] is updates.grid.Ex_dev
    assert encoder.buffers[14] is updates.grid.Ey_dev
    assert encoder.buffers[15] is updates.grid.Ez_dev
    assert encoder.buffers[16] is updates.grid.Hx_dev
    assert encoder.buffers[17] is updates.grid.Hy_dev
    assert encoder.buffers[18] is updates.grid.Hz_dev
    assert encoder.buffers[19] is updates.snapEx_dev
    assert encoder.buffers[20] is updates.snapEy_dev
    assert encoder.buffers[21] is updates.snapEz_dev
    assert encoder.buffers[22] is updates.snapHx_dev
    assert encoder.buffers[23] is updates.snapHy_dev
    assert encoder.buffers[24] is updates.snapHz_dev


def test_store_snapshots_skips_untriggered_snapshot(monkeypatch):
    snap = _FakeSnap(time=5)
    updates, _ = _make_updates([snap], snapsgpu2cpu=False, monkeypatch=monkeypatch)

    updates.store_snapshots(iteration=0)

    assert updates.cmdqueue.buffers_created == []


def test_store_snapshots_immediate_readback_when_snapsgpu2cpu(monkeypatch):
    snap = _FakeSnap(time=1)
    updates, known = _make_updates([snap], snapsgpu2cpu=True, monkeypatch=monkeypatch)

    updates.store_snapshots(iteration=0)

    assert np.array_equal(snap.snapfields["Ex"], known["Ex"][0, 0:2, 0:2, 0:2])
    assert np.array_equal(snap.snapfields["Hz"], known["Hz"][0, 0:2, 0:2, 0:2])


def test_set_snapshot_knl_passes_snapshots_list_not_grid(monkeypatch):
    """The original bug called htod_snapshot_array(self.grid, self.queue) -
    passing the grid object itself (and a nonexistent self.queue attribute)
    instead of self.grid.snapshots. Confirm the fixed call site passes the
    snapshots list specifically."""
    monkeypatch.setattr(Snapshot, "nx_max", 2)
    monkeypatch.setattr(Snapshot, "ny_max", 2)
    monkeypatch.setattr(Snapshot, "nz_max", 2)

    recorded = {}

    def _fake_htod_snapshot_array(snapshots, dev):
        recorded["snapshots"] = snapshots
        return (
            _FakeBuffer(),
            _FakeBuffer(),
            _FakeBuffer(),
            _FakeBuffer(),
            _FakeBuffer(),
            _FakeBuffer(),
        )

    monkeypatch.setattr(
        metal_updates_mod, "htod_snapshot_array", _fake_htod_snapshot_array
    )

    class _FakeLib:
        def newFunctionWithName_(self, name):
            return f"function:{name}"

    class _FakeDeviceForBuild(_FakeDevice):
        def newLibraryWithSource_options_error_(self, source, opts, error):
            return _FakeLib(), None

        def newComputePipelineStateWithFunction_error_(self, func, error):
            return (_FakePSO(), None)

    updates = MetalUpdates.__new__(MetalUpdates)
    updates.dev = _FakeDeviceForBuild()
    updates.opts = None
    updates.knl_common = ""
    updates.subs_func = {
        "REAL": "float",
        "CUDA_IDX": "",
        "NX_FIELDS": 1,
        "NY_FIELDS": 1,
        "NZ_FIELDS": 1,
        "NX_ID": 1,
        "NY_ID": 1,
        "NZ_ID": 1,
    }
    updates.subs_name_args = {"REAL": "float", "COMPLEX": "float2"}

    class _Grid:
        pass

    sentinel_snapshots = [_FakeSnap(time=1)]
    grid = _Grid()
    grid.snapshots = sentinel_snapshots
    updates.grid = grid

    updates._set_snapshot_knl()

    assert recorded["snapshots"] is sentinel_snapshots
    assert updates.pso_store_snapshot is not None


def test_finalise_populates_each_snapshot_from_its_own_page(monkeypatch):
    snap0 = _FakeSnap(time=1)
    snap1 = _FakeSnap(time=2)
    updates, known = _make_updates(
        [snap0, snap1], snapsgpu2cpu=False, monkeypatch=monkeypatch
    )
    updates.grid.rxs = []

    updates.finalise()

    assert np.array_equal(snap0.snapfields["Ex"], known["Ex"][0, 0:2, 0:2, 0:2])
    assert np.array_equal(snap1.snapfields["Ex"], known["Ex"][1, 0:2, 0:2, 0:2])
    assert not np.array_equal(snap0.snapfields["Ex"], snap1.snapfields["Ex"])
