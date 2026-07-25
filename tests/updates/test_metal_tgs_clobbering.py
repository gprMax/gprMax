"""Regression test for the Metal threadgroup-size clobbering bug
(gprMax/updates/metal_updates.py's _set_rx_knl()).

_set_field_knls() sizes self.grid.tgs/tptg for the field-update pipeline
(psoE) and this is reused by every bulk electric/magnetic/PML dispatch
(update_electric_a/b, update_magnetic, MetalPML.update_electric/magnetic -
all dispatch via self.grid.tptg/self.grid.tgs). _set_rx_knl() used to call
self.grid.set_thread_group_size(self.pso_store_outputs) unconditionally
right after building the (much smaller, one-thread-per-receiver)
receiver-storage pipeline, overwriting that field-sized value with the rx
pipeline's own limit - for any model with at least one receiver (nearly
all real models), every subsequent field/PML dispatch used a threadgroup
size derived from the wrong pipeline. store_outputs() itself never reads
self.grid.tgs/tptg at all (it computes its own size inline from
self.pso_store_outputs.maxTotalThreadsPerThreadgroup() at dispatch time),
so the call served no purpose other than clobbering.

Fixed by removing the two grid.set_threads_per_thread_group()/
set_thread_group_size() calls from _set_rx_knl() entirely.

Real Apple Metal hardware/PyObjC isn't available in this environment, so
this test builds a MetalUpdates instance with __new__ (bypassing
__init__) and exercises _set_rx_knl() directly against a fake grid that
records every set_thread_group_size()/set_threads_per_thread_group() call.
"""
import numpy as np

from gprMax import config
from gprMax.updates.metal_updates import MetalUpdates


class _FakeBuffer:
    pass


class _FakeLib:
    def newFunctionWithName_(self, name):
        return f"function:{name}"


class _FakePSO:
    def maxTotalThreadsPerThreadgroup(self):
        return 64


class _FakeDevice:
    def newBufferWithBytes_length_options_(self, data, length, options):
        return _FakeBuffer()

    def newLibraryWithSource_options_error_(self, source, opts, error):
        return _FakeLib(), None

    def newComputePipelineStateWithFunction_error_(self, func, error):
        return (_FakePSO(), None)


class _FakeRx:
    xcoord = ycoord = zcoord = 0


class _FakeGrid:
    def __init__(self):
        self.rxs = [_FakeRx()]
        self.iterations = 10
        self.nx = self.ny = self.nz = 4
        self.tgs = "field-sized-tgs"
        self.tptg = "field-sized-tptg"
        self.set_thread_group_size_calls = []
        self.set_threads_per_thread_group_calls = 0

    def set_thread_group_size(self, pso):
        self.set_thread_group_size_calls.append(pso)
        self.tgs = f"clobbered-by-{pso}"

    def set_threads_per_thread_group(self):
        self.set_threads_per_thread_group_calls += 1


def test_set_rx_knl_does_not_touch_grid_thread_group_size(monkeypatch):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.dtypes = {"float_or_double": np.float64, "C_float_or_double": "double"}
    config.sim_config.general = {"solver": "metal"}
    monkeypatch.setattr(
        config, "get_model_config", lambda: type("_MC", (), {"device": {"dev": _FakeDevice()}})()
    )

    updates = MetalUpdates.__new__(MetalUpdates)
    updates.dev = _FakeDevice()
    updates.opts = None
    updates.knl_common = ""
    updates.subs_func = {"REAL": "double", "CUDA_IDX": ""}
    updates.subs_name_args = {"REAL": "double"}
    updates.grid = _FakeGrid()

    original_tgs = updates.grid.tgs
    original_tptg = updates.grid.tptg

    updates._set_rx_knl()

    assert updates.grid.set_thread_group_size_calls == []
    assert updates.grid.set_threads_per_thread_group_calls == 0
    assert updates.grid.tgs == original_tgs
    assert updates.grid.tptg == original_tptg
    assert updates.pso_store_outputs is not None
