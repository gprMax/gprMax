"""Regression test for gprMax/sources.py's htod_src_arrays() - GitHub issue
#581: two unconditional lines immediately after the VoltageSource if/else
(resistance != 0 -> soft source -> waveformvalues_halfdt; resistance == 0 ->
hard source -> waveformvalues_wholedt) overwrote that selection every time,
so srcwaves always ended up as waveformvalues_halfdt regardless of
resistance - on every GPU backend (CUDA, OpenCL, Metal), a hard
(zero-resistance) VoltageSource silently used the wrong waveform array,
inconsistent with the CPU path's own VoltageSource.update_electric(), which
correctly uses waveformvalues_wholedt for resistance == 0 (see the `if
self.resistance != 0: ... else: Ex[...] = -1 * self.waveformvalues_wholedt[...]`
branch there).

Uses pycuda's gpuarray.to_gpu (monkeypatched to a no-op passthrough) so this
exercises the actual "cuda" solver branch of htod_src_arrays() without
requiring a live CUDA device/context - portable to machines without a GPU,
while still covering the exact code path the bug was in (not just the
CPU-only array-prep logic in isolation).
"""
import numpy as np
import pytest

pytest.importorskip("pycuda")

import gprMax.config as config
import gprMax.sources as sources_mod


class VoltageSource:
    """Duck-typed stand-in - htod_src_arrays() dispatches on
    src.__class__.__name__ == "VoltageSource" (a string check, not
    isinstance), so naming this class exactly that is sufficient to
    exercise the real branch without a full Scene/grid setup."""

    def __init__(self, resistance, halfdt, wholedt):
        self.xcoord, self.ycoord, self.zcoord = 1, 2, 3
        self.polarisation = "z"
        self.resistance = resistance
        self.waveformvalues_halfdt = halfdt
        self.waveformvalues_wholedt = wholedt


class _DummyGrid:
    iterations = 4


class _PassthroughGPUArray:
    """Stand-in for a pycuda GPUArray that just wraps the host array,
    avoiding any real device allocation/context requirement."""

    def __init__(self, host_array):
        self._host_array = host_array

    def get(self):
        return self._host_array


def test_voltage_source_uses_wholedt_for_hard_source_and_halfdt_for_soft_source(monkeypatch):
    import pycuda.gpuarray as gpuarray

    monkeypatch.setattr(gpuarray, "to_gpu", lambda ary: _PassthroughGPUArray(ary))
    monkeypatch.setattr(
        config, "sim_config", type("_SC", (), {})()
    )
    config.sim_config.general = {"solver": "cuda"}
    config.sim_config.dtypes = {"float_or_double": np.float64}

    hard = VoltageSource(resistance=0, halfdt=np.full(5, 111.0), wholedt=np.full(5, 222.0))
    soft = VoltageSource(resistance=75.0, halfdt=np.full(5, 333.0), wholedt=np.full(5, 444.0))

    _, srcinfo2_dev, srcwaves_dev = sources_mod.htod_src_arrays([hard, soft], _DummyGrid())
    srcinfo2 = srcinfo2_dev.get()
    srcwaves = srcwaves_dev.get()

    assert np.allclose(srcwaves[0], 222.0), "hard (zero-resistance) source must use waveformvalues_wholedt"
    assert np.allclose(srcwaves[1], 333.0), "soft (resistive) source must use waveformvalues_halfdt"
    assert srcinfo2[0] == 0
    assert srcinfo2[1] == 75.0
