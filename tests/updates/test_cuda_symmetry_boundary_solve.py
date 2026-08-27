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

"""End-to-end CUDA/CPU parity for a nondispersive PMC boundary."""

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


def _scene(faces, source_position, receiver_position):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    for face in faces:
        scene.add(gprMax.SymmetryBoundary(face=face, type="pmc"))
    scene.add(
        gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w")
    )
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z", p1=source_position, waveform_id="w"
        )
    )
    scene.add(gprMax.Rx(p1=receiver_position, id="rx"))
    return scene


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize(
    "case,faces,source_position,receiver_position",
    [
        ("face", ("x0",), (0.0, 0.01, 0.01), (0.006, 0.01, 0.01)),
        (
            "max_face",
            ("xmax",),
            (0.02, 0.01, 0.01),
            (0.014, 0.01, 0.01),
        ),
        (
            "edge",
            ("x0", "y0"),
            (0.0, 0.0, 0.01),
            (0.006, 0.006, 0.01),
        ),
    ],
)
def test_cuda_pmc_waveform_matches_cpu(
    tmp_path, gpu_device, case, faces, source_position, receiver_position
):
    cpu_path = tmp_path / f"cpu_pmc_{case}"
    cuda_path = tmp_path / f"cuda_pmc_{case}"
    gprMax.run(
        scenes=[_scene(faces, source_position, receiver_position)],
        n=1,
        outputfile=cpu_path,
        hide_progress_bars=True,
        cpu_precision="double",
    )
    gprMax.run(
        scenes=[_scene(faces, source_position, receiver_position)],
        n=1,
        outputfile=cuda_path,
        hide_progress_bars=True,
        gpu=[gpu_device],
        gpu_precision="double",
    )

    with h5py.File(str(cpu_path) + ".h5", "r") as output:
        cpu = output["rxs/rx1/Ez"][:]
    with h5py.File(str(cuda_path) + ".h5", "r") as output:
        cuda = output["rxs/rx1/Ez"][:]

    scale = np.max(np.abs(cpu))
    assert scale > 1e-3
    assert np.isfinite(cuda).all()
    assert_allclose(cuda, cpu, rtol=2e-6, atol=2e-8 * scale)
