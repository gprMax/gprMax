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

"""End-to-end CUDA/CPU parity for the corrected Hyun magnetic frill."""

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


def _scene(symmetry=False):
    dl = 1e-3
    domain = (0.02, 0.02, 0.02)
    feed = (0.0, 0.0, 0.0) if symmetry else (0.01, 0.01, 0.0)
    receiver = (0.004, 0.004, 0.005) if symmetry else (0.014, 0.01, 0.005)

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    if symmetry:
        scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
        scene.add(gprMax.SymmetryBoundary(face="y0", type="pmc"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(domain[0], domain[1], dl),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.ThinWire(
            p1=feed,
            p2=(feed[0], feed[1], 0.01),
            radius=0.1e-3,
        )
    )
    scene.add(
        gprMax.MagneticFrillSource(
            p1=feed,
            polarisation="z",
            zcoax=50,
            waveform_id="w",
            start=0,
            stop=8e-11,
        )
    )
    scene.add(gprMax.Rx(p1=receiver, id="rx"))
    return scene


def _read(path):
    with h5py.File(str(path) + ".h5", "r") as output:
        frill = output["frills/frill1"]
        result = {
            name: frill[name][:] for name in ("Vinc", "Vtotal", "Itot", "frequency", "S11", "Zin")
        }
        result["valid"] = frill["valid_Zin"][:].astype(bool)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            result[component] = output[f"rxs/rx1/{component}"][:]
        return result


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize(
    ("precision", "symmetry"),
    [("single", False), ("double", False), ("double", True)],
)
def test_cuda_magnetic_frill_matches_cpu(tmp_path, gpu_device, precision, symmetry):
    suffix = f"{precision}_{'symmetry' if symmetry else 'full'}"
    cpu_path = tmp_path / f"cpu_frill_{suffix}"
    cuda_path = tmp_path / f"cuda_frill_{suffix}"
    gprMax.run(
        scenes=[_scene(symmetry)],
        n=1,
        outputfile=cpu_path,
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[_scene(symmetry)],
        n=1,
        outputfile=cuda_path,
        hide_progress_bars=True,
        gpu=[gpu_device],
        gpu_precision=precision,
    )

    cpu = _read(cpu_path)
    cuda = _read(cuda_path)
    assert np.max(np.abs(cpu["Vtotal"])) > 1e-3
    tolerance = 3e-5 if precision == "single" else 3e-12
    for name in ("Vinc", "Vtotal", "Itot", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        scale = max(float(np.max(np.abs(cpu[name]))), 1e-12)
        assert np.isfinite(cuda[name]).all()
        assert_allclose(cuda[name], cpu[name], rtol=tolerance, atol=tolerance * scale)

    assert_allclose(cuda["frequency"], cpu["frequency"], rtol=0, atol=0)
    valid = cpu["valid"] & cuda["valid"]
    assert valid.any()
    spectral_tolerance = 6e-5 if precision == "single" else 6e-12
    assert_allclose(
        cuda["S11"][valid],
        cpu["S11"][valid],
        rtol=spectral_tolerance,
        atol=spectral_tolerance,
    )
    impedance_scale = max(float(np.max(np.abs(cpu["Zin"][valid]))), 1.0)
    assert_allclose(
        cuda["Zin"][valid],
        cpu["Zin"][valid],
        rtol=5 * spectral_tolerance,
        atol=5 * spectral_tolerance * impedance_scale,
    )

    # The waveform is zero after stop, but the passive terminal recurrence
    # remains active and continues to respond to returning antenna energy.
    assert np.all(cuda["Vinc"][50:] == 0)
    assert np.max(np.abs(cuda["Vtotal"][50:])) > 0
