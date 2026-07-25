"""Regression tests: two GPU-restriction guards in
gprMax/user_objects/cmds_multiuse.py listed "cuda"/"opencl" but omitted
"metal" (Codex-reported), even though Metal has no support for either
feature they gate:

1. TransmissionLine._validate_parameters() rejects transmission lines on
   CUDA/OpenCL (no update path exists for them there), but Metal - which
   also has zero transmission-line handling in
   gprMax/updates/metal_updates.py - fell through and was silently
   accepted. A Metal transmission-line model would run with no excitation
   at all, and meaningless Vtotal/Itotal outputs.
2. Rx.build()'s allowable-outputs check restricts CUDA/OpenCL receivers to
   the 6 field components the shared GPU kernel
   (gprMax/cuda_opencl/knl_store_outputs.py) actually writes, but Metal
   (which uses that exact same shared kernel/args_metal template) fell
   through to the full CPU list including Ix/Iy/Iz - accepted at parse
   time, then raised a late, confusing ValueError from
   Rx.allowableoutputs_dev.index() during output finalisation instead of
   a clean upfront rejection.

Both are fixed by adding "metal" to the respective solver-name lists.
Follows the established pattern (see tests/test_receivers_dtoh.py) of
replacing config.sim_config wholesale to fake a non-CPU solver without
needing real GPU/Metal hardware present.
"""
import numpy as np
import pytest

import gprMax.config as config
from gprMax.user_objects.cmds_multiuse import Rx, TransmissionLine


def _set_solver(monkeypatch, solver):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": solver}
    config.sim_config.dtypes = {"float_or_double": np.float64}


@pytest.mark.parametrize("solver", ["cuda", "opencl", "metal"])
def test_transmission_line_rejected_on_every_gpu_solver(monkeypatch, solver):
    _set_solver(monkeypatch, solver)

    tl = TransmissionLine(
        p1=(0.01, 0.01, 0.01), polarisation="x", resistance=50, waveform_id="w"
    )

    with pytest.raises(ValueError, match="cannot currently be used"):
        tl._validate_parameters(grid=None)


@pytest.mark.parametrize("solver", ["cuda", "opencl", "metal"])
def test_current_output_rejected_on_every_gpu_solver(monkeypatch, solver):
    _set_solver(monkeypatch, solver)

    rx = Rx(p1=(0.01, 0.01, 0.01), id="r1", outputs=["Ix"])

    with pytest.raises(ValueError, match="not allowable"):
        rx._create_receiver(grid=None, coord=np.array([1, 1, 1], dtype=np.int32))


def test_current_output_still_allowed_on_cpu(monkeypatch):
    _set_solver(monkeypatch, "cpu")

    class _Grid:
        iterations = 5

    rx = Rx(p1=(0.01, 0.01, 0.01), id="r1", outputs=["Ix"])
    r = rx._create_receiver(grid=_Grid(), coord=np.array([1, 1, 1], dtype=np.int32))

    assert "Ix" in r.outputs
