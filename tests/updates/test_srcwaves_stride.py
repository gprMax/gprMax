"""Regression test for the NY_SRCWAVES stride bug (Codex-reported,
"Critical"): CUDA, OpenCL, and Metal's _set_macros() all baked
NY_SRCWAVES=grid.iterations into the shared IDX2D_SRCWAVES(m, n) macro
(gprMax/cuda_opencl/knl_common_base.tmpl:
`#define IDX2D_SRCWAVES(m, n) (m)*(NY_SRCWAVES)+(n)`), but
gprMax/sources.py's htod_src_arrays() actually allocates the srcwaves host
array with shape (len(sources), G.iterations + 1) - one extra column
holding a half-timestep sample.

Since the macro's row stride (grid.iterations) was one column short of the
array's real row length (grid.iterations + 1), only the first source (row
0) read correctly. Every source after that read shifted: row m's data
really starts at m*(iterations+1), but the kernel computed m*iterations -
a growing per-row offset (source 2 off by 1 element, source 3 by 2, etc).
Verified end-to-end on real CUDA hardware (2 Hertzian dipoles with
different waveforms/amplitudes, one receiver near each): with the bug
reverted, receiver 2's trace was cpu[0:6] then cpu[7:] shifted one sample
early - an exact match for the reported mechanism. With the fix, CUDA vs
CPU relative error is ~1e-5 (ordinary float32 GPU noise).

Fixed by changing NY_SRCWAVES to grid.iterations + 1 in all three
backends' _set_macros() (gprMax/updates/cuda_updates.py,
opencl_updates.py, metal_updates.py) - plus a second, otherwise-inert
occurrence in CUDAUpdates._set_src_knls() that sets the same key on
self.subs_func (dead for kernel-building today, since
knl_source_updates.py's templates use the already-expanded IDX2D_SRCWAVES
macro rather than a $NY_SRCWAVES placeholder - fixed anyway so it can't
silently reintroduce the bug if that ever changes).

This test renders each backend's real Jinja kernel-common template via
_set_macros() against a minimal fake grid (bypassing __init__ with
__new__, matching the existing test_metal_dispersive_dispatch.py /
test_metal_tgs_clobbering.py pattern) and asserts the row-stride literal
baked into the IDX2D_SRCWAVES macro equals iterations + 1, not
iterations.
"""
import re

import numpy as np
import pytest
from jinja2 import Environment, PackageLoader

from gprMax import config
from gprMax.updates.cuda_updates import CUDAUpdates
from gprMax.updates.metal_updates import MetalUpdates
from gprMax.updates.opencl_updates import OpenCLUpdates

ITERATIONS = 10


class _FakeGrid:
    nx = ny = nz = 4
    iterations = ITERATIONS
    rxs = []
    updatecoeffsE = np.zeros((3, 5))
    updatecoeffsH = np.zeros((3, 5))
    ID = np.zeros((6, 5, 5, 5))


def _srcwaves_stride(knl_common: str) -> int:
    match = re.search(r"#define IDX2D_SRCWAVES\(m, n\) \(m\)\*\((\d+)\)\+\(n\)", knl_common)
    assert match, f"IDX2D_SRCWAVES macro not found in rendered template:\n{knl_common}"
    return int(match.group(1))


@pytest.fixture(autouse=True)
def _fake_model_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: type("_MC", (), {"materials": {"maxpoles": 0}})(),
    )
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.dtypes = {"C_float_or_double": "float"}


def test_cuda_srcwaves_stride_matches_htod_src_arrays_row_length():
    updates = CUDAUpdates.__new__(CUDAUpdates)
    updates.grid = _FakeGrid()
    updates.env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))
    updates._set_macros()

    assert _srcwaves_stride(updates.knl_common) == ITERATIONS + 1


def test_opencl_srcwaves_stride_matches_htod_src_arrays_row_length():
    updates = OpenCLUpdates.__new__(OpenCLUpdates)
    updates.grid = _FakeGrid()
    updates.env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))
    updates._set_macros()

    assert _srcwaves_stride(updates.knl_common) == ITERATIONS + 1


def test_metal_srcwaves_stride_matches_htod_src_arrays_row_length():
    updates = MetalUpdates.__new__(MetalUpdates)
    updates.grid = _FakeGrid()
    updates.env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))
    updates._set_macros()

    assert _srcwaves_stride(updates.knl_common) == ITERATIONS + 1


def test_htod_src_arrays_row_length_is_iterations_plus_one():
    """Sanity check on the array side of the contract: if sources.py's
    allocation shape ever changes, this test (not just the macro-side
    ones above) should fail first. Needs a real CUDA context to allocate
    device memory (available in this environment); skipped elsewhere."""
    pytest.importorskip("pycuda")
    import pycuda.autoinit  # noqa: F401 - establishes a CUDA context
    from gprMax.sources import htod_src_arrays

    class _Src:
        xcoord = ycoord = zcoord = 0
        polarisation = "z"
        __class__ = type("HertzianDipole", (), {})

    class _Grid:
        iterations = ITERATIONS

    import gprMax.config as cfg

    orig_sim_config = cfg.sim_config
    cfg.sim_config = type("_SC", (), {})()
    cfg.sim_config.dtypes = {"float_or_double": np.float64}
    cfg.sim_config.general = {"solver": "cuda"}
    try:
        src = _Src()
        src.dl = 1.0
        src.waveformvalues_halfdt = np.zeros(ITERATIONS + 1)
        srcinfo1_dev, srcinfo2_dev, srcwaves_dev = htod_src_arrays([src], _Grid())
    finally:
        cfg.sim_config = orig_sim_config

    assert srcwaves_dev.shape[1] == ITERATIONS + 1
