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

"""Regression tests for the GPU snapshot indexing/ordering bugs found while
verifying the 2D TE-mode snapshot-averaging fix on real CUDA hardware (see
project_deep_bug_audit.md's "GPU snapshot indexing" entry for the full
writeup). Three independent, compounding bugs, all in the shared
CUDA/OpenCL/Metal snapshot pipeline:

1. store_snapshot's kernel (knl_snapshots.py) compared its own local,
   0-based snaps-array thread index directly against the snapshot's
   *absolute* finish coordinate (xf/yf/zf) - only happened to work for a
   snapshot starting exactly at the grid origin; any other position
   silently truncated/misaligned the output. Fixed by passing the
   snapshot's own already-correct local sample count (nx/ny/nz) instead
   of the absolute finish coordinate, and comparing the local thread
   index against that.
2. dtoh_snapshot_array() (gprMax/snapshots.py) had the identical bug on
   the host side - slicing the device-to-host-copied buffer with
   snap.xs:snap.xf (absolute) instead of 0:snap.nx (local). Fixed
   alongside (1).
3. Snapshot.nx_max/ny_max/nz_max (class attributes baked into the shared
   IDX4D_SNAPS indexing macro by _set_macros(), which always runs before
   _set_snapshot_knl()) default to 0 and were only ever updated inside
   htod_snapshot_array() - called *by* _set_snapshot_knl(), i.e. after
   _set_macros() already rendered the (stale, zeroed) macro. This
   collapsed IDX4D_SNAPS's indexing arithmetic down to effectively just
   the z-index for the first snapshot-using model in any process,
   causing every thread to race on a handful of memory locations. Fixed
   by extracting update_snapshot_max_dims() and calling it before
   _set_macros() in all three backends' __init__.

All three were found and fixed together; verified end-to-end (real
CUDA hardware, TITAN RTX) that a 3D and a 2D TEz snapshot both now match
the CPU/Cython reference to within expected single-precision
cross-platform floating-point noise (~1e-4 relative), for a
non-origin snapshot region - the case that was broken before. These
tests cover the parts that can be verified without real hardware: the
dispatch argument order/contract for CUDA and OpenCL (mirroring the
existing Metal dispatch tests), and update_snapshot_max_dims() in
isolation.
"""
from types import SimpleNamespace

import numpy as np

from gprMax.cuda_opencl import knl_snapshots
from gprMax.snapshots import Snapshot, update_snapshot_max_dims


class _FakeSnapshot:
    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz


def test_update_snapshot_max_dims_tracks_the_largest_requested_snapshot(monkeypatch):
    monkeypatch.setattr(Snapshot, "nx_max", 0)
    monkeypatch.setattr(Snapshot, "ny_max", 0)
    monkeypatch.setattr(Snapshot, "nz_max", 0)

    update_snapshot_max_dims([_FakeSnapshot(5, 20, 3), _FakeSnapshot(12, 4, 30)])

    assert Snapshot.nx_max == 12
    assert Snapshot.ny_max == 20
    assert Snapshot.nz_max == 30


def test_update_snapshot_max_dims_is_a_noop_on_empty_list(monkeypatch):
    monkeypatch.setattr(Snapshot, "nx_max", 7)
    monkeypatch.setattr(Snapshot, "ny_max", 8)
    monkeypatch.setattr(Snapshot, "nz_max", 9)

    update_snapshot_max_dims([])

    assert (Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max) == (7, 8, 9)


def test_gpu_snapshot_collocation_uses_the_snapshot_cell_stride():
    """GPU collocation must match the CPU's strided ``GridView`` corners."""

    body = knl_snapshots.store_snapshot["func"].substitute(
        {
            "CUDA_IDX": "",
            "REAL": "float",
            "NX_SNAPS": 1,
            "NY_SNAPS": 1,
            "NZ_SNAPS": 1,
        }
    )

    assert "xx+sx*dx" in body
    assert "yy+sy*dy" in body
    assert "zz+sz*dz" in body


# ---------------------------------------------------------------------------
# CUDA dispatch argument order
# ---------------------------------------------------------------------------


class _FakeGpuData:
    """Stand-in for a pycuda GPUArray's .gpudata pointer attribute."""


class _FakeGpuArray:
    def __init__(self, name):
        self.gpudata = f"gpudata:{name}"


class _FakeSnap:
    def __init__(self, time):
        self.time = time
        self.xs, self.ys, self.zs = 3, 4, 5
        self.nx, self.ny, self.nz = 6, 7, 8
        self.dx = self.dy = self.dz = 1


def _make_cuda_updates(monkeypatch, snapshots):
    import gprMax.config as config
    from gprMax.updates.cuda_updates import CUDAUpdates

    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: type(
            "_MC", (), {"device": {"snapsgpu2cpu": False}, "mode": "3D"}
        )(),
    )

    updates = CUDAUpdates.__new__(CUDAUpdates)
    recorded_calls = []

    def fake_store_snapshot_dev(*args, **kwargs):
        recorded_calls.append((args, kwargs))

    updates.store_snapshot_dev = fake_store_snapshot_dev

    class _Grid:
        pass

    grid = _Grid()
    grid.snapshots = snapshots
    for name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        setattr(grid, f"{name}_dev", _FakeGpuArray(name))
    updates.grid = grid
    for name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        setattr(updates, f"snap{name}_dev", _FakeGpuArray(f"snap{name}"))

    return updates, recorded_calls


def test_cuda_store_snapshots_passes_local_sample_counts_not_absolute_finish(monkeypatch):
    snap = _FakeSnap(time=1)
    updates, calls = _make_cuda_updates(monkeypatch, [snap])

    updates.store_snapshots(iteration=0)

    assert len(calls) == 1
    args, kwargs = calls[0]
    # p, xs, ys, zs, nx, ny, nz, dx, dy, dz, sx, sy, sz, then 12 buffers.
    values = [int(a) for a in args[:13]]
    assert values == [0, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1]
    assert args[13] == "gpudata:Ex"
    assert args[18] == "gpudata:Hz"
    assert args[19] == "gpudata:snapEx"
    assert args[24] == "gpudata:snapHz"
    assert kwargs["block"] is Snapshot.tpb
    assert kwargs["grid"] is Snapshot.bpg


def test_cuda_store_snapshots_skips_untriggered_snapshot(monkeypatch):
    snap = _FakeSnap(time=5)
    updates, calls = _make_cuda_updates(monkeypatch, [snap])

    updates.store_snapshots(iteration=0)

    assert calls == []


# ---------------------------------------------------------------------------
# OpenCL dispatch argument order
# ---------------------------------------------------------------------------


def _make_opencl_updates(monkeypatch, snapshots):
    import gprMax.config as config
    from gprMax.updates.opencl_updates import OpenCLUpdates

    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: type(
            "_MC", (), {"device": {"snapsgpu2cpu": False}, "mode": "3D"}
        )(),
    )

    updates = OpenCLUpdates.__new__(OpenCLUpdates)
    recorded_calls = []

    def fake_store_snapshot_dev(*args, **kwargs):
        recorded_calls.append((args, kwargs))

    updates.store_snapshot_dev = fake_store_snapshot_dev

    class _Grid:
        pass

    grid = _Grid()
    grid.snapshots = snapshots
    for name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        setattr(grid, f"{name}_dev", f"dev:{name}")
    updates.grid = grid
    for name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        setattr(updates, f"snap{name}_dev", f"dev:snap{name}")

    return updates, recorded_calls


def test_opencl_store_snapshots_passes_local_sample_counts_not_absolute_finish(monkeypatch):
    snap = _FakeSnap(time=1)
    updates, calls = _make_opencl_updates(monkeypatch, [snap])

    updates.store_snapshots(iteration=0)

    assert len(calls) == 1
    args, kwargs = calls[0]
    values = [int(a) for a in args[:13]]
    assert values == [0, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1]
    assert args[13] == "dev:Ex"
    assert args[18] == "dev:Hz"
    assert args[19] == "dev:snapEx"
    assert args[24] == "dev:snapHz"


def test_opencl_snapshot_kernel_substitutes_real_type(monkeypatch):
    import gprMax.config as config
    import gprMax.updates.opencl_updates as opencl_updates
    from gprMax.updates.opencl_updates import OpenCLUpdates

    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={"C_float_or_double": "double"},
            devices={"compiler_opts": []},
        ),
    )
    monkeypatch.setattr(
        opencl_updates,
        "htod_snapshot_array",
        lambda snapshots, queue: tuple(f"snap:{name}" for name in "Ex Ey Ez Hx Hy Hz".split()),
    )

    captured = {}
    updates = OpenCLUpdates.__new__(OpenCLUpdates)
    updates.grid = SimpleNamespace(snapshots=[object()])
    updates.queue = object()
    updates.ctx = object()
    updates.knl_common = ""

    def fake_elementwise(context, arguments, body, name, **kwargs):
        captured.update(arguments=arguments, body=body, name=name)
        return object()

    updates.elwiseknl = fake_elementwise
    updates._set_snapshot_knl()

    assert captured["name"] == "store_snapshot"
    assert "$" not in captured["arguments"]
    assert "$" not in captured["body"]
    assert "(double)0.25" in captured["body"]
