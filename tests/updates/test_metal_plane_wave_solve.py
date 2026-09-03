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

"""CPU/Metal parity for discrete plane-wave execution paths."""

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


def _standard_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.TimeWindow(time=3e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e10, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=(0.004, 0.004, 0.004),
            p2=(0.016, 0.016, 0.016),
            m_vec=(1, 0, 0),
            psi=90,
            waveform_id="pulse",
        )
    )
    scene.add(gprMax.Rx(p1=(0.010, 0.010, 0.010), outputs=["Ez"]))
    return scene


def _axial_dispersive_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.04, 0.02, float("inf"))))
    scene.add(gprMax.PMLThickness(thickness=(4, 4, 0, 4, 4, 0)))
    scene.add(gprMax.TimeWindow(time=5e-10))
    scene.add(gprMax.Material(er=2.5, se=0, mr=1, sm=0, id="debye"))
    scene.add(
        gprMax.AddDebyeDispersion(
            poles=1,
            er_delta=(3.0,),
            tau=(80e-12,),
            material_ids=["debye"],
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0.025, 0, float("inf")),
            p2=(0.04, 0.02, float("inf")),
            material_id="debye",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=3e9, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.005, 0.005, float("inf")),
            p2=(0.035, 0.015, float("inf")),
            axis="x",
            psi=90,
            waveform_id="pulse",
        )
    )
    scene.add(gprMax.Rx(p1=(0.018, 0.010, float("inf")), outputs=["Ez"]))
    return scene


@pytest.mark.skipif(not HAS_METAL, reason="No Apple Metal device/PyObjC available")
@pytest.mark.parametrize(
    "case,scene_factory",
    [("standard", _standard_scene), ("axial_dispersive", _axial_dispersive_scene)],
)
def test_metal_plane_wave_matches_cpu(tmp_path, case, scene_factory):
    cpu_path = tmp_path / f"cpu_{case}"
    metal_path = tmp_path / f"metal_{case}"
    common = dict(n=1, hide_progress_bars=True, log_level=logging.WARNING)

    gprMax.run(
        scenes=[scene_factory()],
        outputfile=cpu_path,
        cpu_precision="double",
        **common,
    )
    gprMax.run(
        scenes=[scene_factory()],
        outputfile=metal_path,
        metal=True,
        gpu_precision="single",
        **common,
    )

    with h5py.File(str(cpu_path) + ".h5", "r") as output:
        cpu = output["rxs/rx1/Ez"][:]
    with h5py.File(str(metal_path) + ".h5", "r") as output:
        metal = output["rxs/rx1/Ez"][:]

    scale = np.max(np.abs(cpu))
    assert scale > 1e-6
    assert np.isfinite(metal).all()
    assert np.max(np.abs(metal - cpu)) / scale < 2e-5
    assert np.linalg.norm(metal - cpu) / np.linalg.norm(cpu) < 2e-5
