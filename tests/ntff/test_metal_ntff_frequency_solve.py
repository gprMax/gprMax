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

"""Real CPU/Metal parity for frequency-domain NTFF collection."""

import logging

import h5py
import numpy as np
import pytest

import gprMax

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

try:
    import Metal

    HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS_METAL = False


def _scene():
    frequency = 10e9
    centre = (0.02, 0.02, 0.02)
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.002,) * 3))
    scene.add(gprMax.Domain(p1=(0.04,) * 3))
    scene.add(gprMax.PMLThickness(thickness=4))
    scene.add(gprMax.TimeWindow(iterations=140))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=frequency, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=centre, waveform_id="pulse"))
    scene.add(gprMax.NTFFSurface(p1=(0.012,) * 3, p2=(0.028,) * 3, id="surface", origin=centre))
    scene.add(
        gprMax.KSIRFrequencyTransform("surface", "spectrum", (frequency,), save_surface_dft=False)
    )
    scene.add(
        gprMax.KSIRFarField(
            theta=np.array((45.0, 90.0, 135.0)),
            phi=np.zeros(3),
            transform_id="spectrum",
            id="pattern",
            outputs=("Etheta",),
        )
    )
    return scene


@pytest.mark.skipif(not HAS_METAL, reason="No Apple Metal device/PyObjC available")
def test_metal_frequency_ntff_matches_cpu(tmp_path):
    common = dict(n=1, hide_progress_bars=True, log_level=logging.WARNING)
    cpu_path = tmp_path / "cpu"
    metal_path = tmp_path / "metal"

    gprMax.run(scenes=[_scene()], outputfile=cpu_path, cpu_precision="double", **common)
    gprMax.run(
        scenes=[_scene()], outputfile=metal_path, metal=True, gpu_precision="single", **common
    )

    dataset = "ntff/surface/frequency/spectrum/far_field/pattern/fields/Etheta"
    with h5py.File(str(cpu_path) + ".h5", "r") as output:
        cpu = output[dataset][:]
    with h5py.File(str(metal_path) + ".h5", "r") as output:
        group = output["ntff/surface/frequency/spectrum"]
        metal = output[dataset][:]
        assert group.attrs["collection_backend"] == "metal_device"

    scale = np.max(np.abs(cpu))
    assert scale > 0
    assert np.isfinite(metal).all()
    assert np.max(np.abs(metal - cpu)) / scale < 5e-4
    assert np.linalg.norm(metal - cpu) / np.linalg.norm(cpu) < 5e-4
