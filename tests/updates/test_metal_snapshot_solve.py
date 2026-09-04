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

"""Real CPU/Metal parity for snapshot timing and field values."""

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
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.TimeWindow(iterations=100))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(
            p1=(0.010, 0.010, 0.010),
            polarisation="z",
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.Snapshot(
            p1=(0.004, 0.004, 0.004),
            p2=(0.016, 0.016, 0.016),
            dl=(0.002, 0.002, 0.002),
            filename="fields",
            iterations=80,
            fileext=".h5",
        )
    )
    return scene


@pytest.mark.skipif(not HAS_METAL, reason="No Apple Metal device/PyObjC available")
def test_metal_snapshot_matches_cpu_at_the_same_electric_timestep(tmp_path):
    cpu_path = tmp_path / "cpu"
    metal_path = tmp_path / "metal"
    common = dict(n=1, hide_progress_bars=True, log_level=logging.WARNING)

    gprMax.run(
        scenes=[_scene()],
        outputfile=cpu_path,
        cpu_precision="double",
        **common,
    )
    gprMax.run(
        scenes=[_scene()],
        outputfile=metal_path,
        metal=True,
        gpu_precision="single",
        **common,
    )

    with h5py.File(tmp_path / "cpu_snaps" / "fields.h5", "r") as cpu_file, h5py.File(
        tmp_path / "metal_snaps" / "fields.h5", "r"
    ) as metal_file:
        assert cpu_file.attrs["iteration"] == metal_file.attrs["iteration"] == 80
        assert cpu_file.attrs["time"] == pytest.approx(metal_file.attrs["time"])
        assert cpu_file.attrs["magnetic_time"] == pytest.approx(
            metal_file.attrs["magnetic_time"]
        )

        cpu = np.concatenate([cpu_file[name][...].ravel() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")])
        metal = np.concatenate(
            [metal_file[name][...].ravel() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
        )

    scale = np.max(np.abs(cpu))
    assert scale > 1e-6
    assert np.isfinite(metal).all()
    assert np.max(np.abs(metal - cpu)) / scale < 2e-4
    assert np.linalg.norm(metal - cpu) / np.linalg.norm(cpu) < 2e-4
