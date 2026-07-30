"""End-to-end CUDA/CPU parity for Lorentz and Drude dispersive materials.

PR #721 ("Fix Lorentz and Drude dispersive updates") corrected the complex
current-density coupling used for multi-pole Lorentz/Drude materials from
`Re(a) * Re(T)` to `Re(a * T)` (the imaginary cross-term must not be
discarded by taking real parts before multiplying), and ported the fix
across CPU, CUDA, OpenCL and Metal kernels. The PR's own testing explicitly
notes GPU hardware was not available to verify this on real devices, and no
existing test in this repository runs an actual Lorentz or Drude material
through a real time-stepped solve on real GPU hardware (the closest test,
test_dispersive_gpu_kernel_source.py, only checks generated kernel source
text, not execution/results; test_complex_permittivity.py only unit-tests
the analytical calculate_er() helper).

Closes that gap directly: runs the same Lorentz/Drude scene on CPU and on
real CUDA hardware and checks they agree - if the CUDA kernel's complex
arithmetic were still doing Re(a)*Re(T) instead of Re(a*T), this would show
up as a real, measurable mismatch against the (already independently
verified, see test_complex_permittivity.py) CPU dispersive path.
"""

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax

try:
    import pycuda.driver as _cuda_driver

    _cuda_driver.init()
    HAS_CUDA = _cuda_driver.Device.count() > 0
except Exception:
    HAS_CUDA = False


def _lorentz_scene():
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.Material(er=3, se=0.001, mr=1, sm=0, id="lorentz_mat"))
    scene.add(
        gprMax.AddLorentzDispersion(
            poles=1, er_delta=[2.0], omega=[2e10], delta=[5e9], material_ids=["lorentz_mat"]
        )
    )
    scene.add(gprMax.Box(p1=(0.0, 0.0, 0.0), p2=(0.02, 0.02, 0.02), material_id="lorentz_mat"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.006, 0.01, 0.01), waveform_id="w"))
    scene.add(gprMax.Rx(p1=(0.014, 0.01, 0.01)))
    return scene


def _drude_scene():
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.Material(er=3, se=0.001, mr=1, sm=0, id="drude_mat"))
    scene.add(
        gprMax.AddDrudeDispersion(
            poles=1, omega=[2e10], alpha=[5e9], material_ids=["drude_mat"]
        )
    )
    scene.add(gprMax.Box(p1=(0.0, 0.0, 0.0), p2=(0.02, 0.02, 0.02), material_id="drude_mat"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.006, 0.01, 0.01), waveform_id="w"))
    scene.add(gprMax.Rx(p1=(0.014, 0.01, 0.01)))
    return scene


def _run_cpu_vs_cuda(scene_factory, tmp_path, label, precision):
    cpu_path = tmp_path / f"cpu_{label}_{precision}"
    cuda_path = tmp_path / f"cuda_{label}_{precision}"
    gprMax.run(
        scenes=[scene_factory()], n=1, outputfile=cpu_path,
        hide_progress_bars=True, cpu_precision=precision,
    )
    gprMax.run(
        scenes=[scene_factory()], n=1, outputfile=cuda_path,
        hide_progress_bars=True, gpu=[0], gpu_precision=precision,
    )

    with h5py.File(str(cpu_path) + ".h5", "r") as f:
        cpu = {comp: f[f"rxs/rx1/{comp}"][:] for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    with h5py.File(str(cuda_path) + ".h5", "r") as f:
        cuda = {comp: f[f"rxs/rx1/{comp}"][:] for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}

    assert np.max(np.abs(cpu["Ez"])) > 1e-3, "CPU Ez is degenerate (all-zero) - wave never arrived"

    # Scale atol off the single dominant field magnitude across ALL
    # components, not each component's own max: by symmetry (z-polarised
    # dipole, receiver on the x-axis), Hz is exactly zero here and its own
    # max is pure floating-point noise (~1e-16) - using that as its own
    # tolerance scale would make an utterly meaningless noise-vs-noise
    # comparison "fail" despite every physically real component agreeing
    # to machine precision.
    overall_scale = max(float(np.max(np.abs(v))) for v in cpu.values())
    tolerance = 2e-5 if precision == "single" else 2e-10
    for comp in cpu:
        assert np.isfinite(cuda[comp]).all(), f"{comp} contains NaN/inf on CUDA"
        assert_allclose(cuda[comp], cpu[comp], rtol=tolerance, atol=tolerance * overall_scale)


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_lorentz_material_matches_cpu(tmp_path, precision):
    _run_cpu_vs_cuda(_lorentz_scene, tmp_path, "lorentz", precision)


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_drude_material_matches_cpu(tmp_path, precision):
    _run_cpu_vs_cuda(_drude_scene, tmp_path, "drude", precision)
