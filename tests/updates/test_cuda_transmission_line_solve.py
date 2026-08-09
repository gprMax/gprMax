"""End-to-end CUDA/CPU parity for a device-resident transmission line."""

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

try:
    import pycuda.driver as _cuda_driver

    _cuda_driver.init()
    HAS_CUDA = _cuda_driver.Device.count() > 0
except Exception:
    HAS_CUDA = False


def _scene():
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.012, 0.012, 0.012)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=2e10, id="w"))
    scene.add(
        gprMax.TransmissionLine(
            polarisation="z",
            p1=(0.006, 0.006, 0.006),
            resistance=50,
            waveform_id="w",
        )
    )
    scene.add(gprMax.Rx(p1=(0.006, 0.006, 0.006), id="rx"))
    return scene


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_transmission_line_matches_cpu(tmp_path, gpu_device, precision):
    cpu_path = tmp_path / f"cpu_tl_{precision}"
    cuda_path = tmp_path / f"cuda_tl_{precision}"
    gprMax.run(
        scenes=[_scene()],
        n=1,
        outputfile=cpu_path,
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[_scene()],
        n=1,
        outputfile=cuda_path,
        hide_progress_bars=True,
        gpu=[gpu_device],
        gpu_precision=precision,
    )

    with h5py.File(str(cpu_path) + ".h5", "r") as output:
        cpu = {name: output[f"tls/tl1/{name}"][:] for name in ("Vinc", "Iinc", "Vtotal", "Itotal")}
        cpu["frequency"] = output["tls/tl1/frequency"][:]
        cpu["S11"] = output["tls/tl1/S11"][:]
        cpu["Zin"] = output["tls/tl1/Zin"][:]
        cpu["valid"] = output["tls/tl1/valid_Zin"][:].astype(bool)
        cpu["Ez"] = output["rxs/rx1/Ez"][:]
    with h5py.File(str(cuda_path) + ".h5", "r") as output:
        cuda = {name: output[f"tls/tl1/{name}"][:] for name in ("Vinc", "Iinc", "Vtotal", "Itotal")}
        cuda["frequency"] = output["tls/tl1/frequency"][:]
        cuda["S11"] = output["tls/tl1/S11"][:]
        cuda["Zin"] = output["tls/tl1/Zin"][:]
        cuda["valid"] = output["tls/tl1/valid_Zin"][:].astype(bool)
        cuda["Ez"] = output["rxs/rx1/Ez"][:]

    assert np.max(np.abs(cpu["Vtotal"])) > 1e-3
    for name in ("Vinc", "Iinc", "Vtotal", "Itotal", "Ez"):
        scale = max(float(np.max(np.abs(cpu[name]))), 1e-12)
        assert np.isfinite(cuda[name]).all()
        tolerance = 2e-5 if precision == "single" else 2e-12
        assert_allclose(cuda[name], cpu[name], rtol=tolerance, atol=tolerance * scale)

    assert_allclose(cuda["frequency"], cpu["frequency"], rtol=0, atol=0)
    valid = cpu["valid"] & cuda["valid"]
    assert valid.any()
    tolerance = 4e-5 if precision == "single" else 4e-12
    assert_allclose(cuda["S11"][valid], cpu["S11"][valid], rtol=tolerance, atol=tolerance)
    impedance_scale = max(float(np.max(np.abs(cpu["Zin"][valid]))), 1.0)
    # Near an open circuit, Zin = Z0(1 + S11)/(1 - S11) amplifies a small
    # single-precision S11 difference. Keep the tighter comparison on S11
    # above and allow the corresponding conditioning in this secondary value.
    impedance_tolerance = 2e-4 if precision == "single" else 4e-12
    assert_allclose(
        cuda["Zin"][valid],
        cpu["Zin"][valid],
        rtol=impedance_tolerance,
        atol=impedance_tolerance * impedance_scale,
    )
