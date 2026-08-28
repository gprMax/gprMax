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

"""End-to-end CPU/GPU parity for a device-resident transmission line."""

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

try:
    import pyopencl as _cl

    HAS_OPENCL = bool(_cl.get_platforms())
except Exception:
    HAS_OPENCL = False


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


@pytest.mark.parametrize("backend", ["cuda", "opencl"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_device_transmission_line_matches_cpu(tmp_path, request, backend, precision):
    if backend == "cuda" and not HAS_CUDA:
        pytest.skip("No CUDA device/pycuda available")
    if backend == "opencl" and not HAS_OPENCL:
        pytest.skip("No OpenCL platform/pyopencl available")

    if backend == "cuda":
        device_options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        device_options = {"opencl": [request.getfixturevalue("opencl_device")]}

    cpu_path = tmp_path / f"cpu_tl_{precision}"
    device_path = tmp_path / f"{backend}_tl_{precision}"
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
        outputfile=device_path,
        hide_progress_bars=True,
        gpu_precision=precision,
        **device_options,
    )

    with h5py.File(str(cpu_path) + ".h5", "r") as output:
        cpu = {name: output[f"tls/tl1/{name}"][:] for name in ("Vinc", "Iinc", "Vtotal", "Itotal")}
        cpu["frequency"] = output["tls/tl1/frequency"][:]
        cpu["S11"] = output["tls/tl1/S11"][:]
        cpu["Zin"] = output["tls/tl1/Zin"][:]
        cpu["valid"] = output["tls/tl1/valid_Zin"][:].astype(bool)
        cpu["Ez"] = output["rxs/rx1/Ez"][:]
    with h5py.File(str(device_path) + ".h5", "r") as output:
        device = {name: output[f"tls/tl1/{name}"][:] for name in ("Vinc", "Iinc", "Vtotal", "Itotal")}
        device["frequency"] = output["tls/tl1/frequency"][:]
        device["S11"] = output["tls/tl1/S11"][:]
        device["Zin"] = output["tls/tl1/Zin"][:]
        device["valid"] = output["tls/tl1/valid_Zin"][:].astype(bool)
        device["Ez"] = output["rxs/rx1/Ez"][:]

    assert np.max(np.abs(cpu["Vtotal"])) > 1e-3
    for name in ("Vinc", "Iinc", "Vtotal", "Itotal", "Ez"):
        scale = max(float(np.max(np.abs(cpu[name]))), 1e-12)
        assert np.isfinite(device[name]).all()
        tolerance = 2e-5 if precision == "single" else 2e-12
        assert_allclose(device[name], cpu[name], rtol=tolerance, atol=tolerance * scale)

    assert_allclose(device["frequency"], cpu["frequency"], rtol=0, atol=0)
    valid = cpu["valid"] & device["valid"]
    assert valid.any()
    tolerance = 4e-5 if precision == "single" else 4e-12
    assert_allclose(device["S11"][valid], cpu["S11"][valid], rtol=tolerance, atol=tolerance)
    impedance_scale = max(float(np.max(np.abs(cpu["Zin"][valid]))), 1.0)
    # Near an open circuit, Zin = Z0(1 + S11)/(1 - S11) amplifies a small
    # single-precision S11 difference. Keep the tighter comparison on S11
    # above and allow the corresponding conditioning in this secondary value.
    impedance_tolerance = 2e-4 if precision == "single" else 4e-12
    assert_allclose(
        device["Zin"][valid],
        cpu["Zin"][valid],
        rtol=impedance_tolerance,
        atol=impedance_tolerance * impedance_scale,
    )
