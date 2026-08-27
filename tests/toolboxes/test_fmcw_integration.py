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

"""End-to-end FMCW synthesis from target and background FDTD runs."""

import numpy as np
import pytest

import gprMax
from toolboxes.FMCW import (
    Chirp,
    process_channel,
    reconstruct_fast_time,
    synthesize_deramped_sweep,
)

pytestmark = pytest.mark.integration


def _scene(target):
    inf = float("inf")
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(0.005, 0.005, 0.005)))
    scene.add(gprMax.Domain(p1=(0.2, 0.2, inf)))
    scene.add(gprMax.TimeWindow(time=20e-9))
    scene.add(gprMax.OMPThreads(n=2))
    scene.add(gprMax.Waveform(wave_type="impulse", amp=1, freq=1, id="source"))
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z",
            p1=(0.08, 0.14, inf),
            waveform_id="source",
        )
    )
    scene.add(gprMax.Rx(p1=(0.12, 0.14, inf), id="response", outputs=["Ez"]))
    if target:
        scene.add(
            gprMax.Cylinder(
                p1=(0.10, 0.07, 0),
                p2=(0.10, 0.07, inf),
                r=0.02,
                material_id="pec",
            )
        )
    return scene


def test_fmcw_target_background_and_stretch_receiver_pipeline(tmp_path):
    files = {}
    for name, target in (("target", True), ("background", False)):
        output = tmp_path / name
        gprMax.run(
            scenes=[_scene(target)],
            n=1,
            outputfile=output,
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )
        files[name] = output.with_suffix(".h5")

    chirp = Chirp(100e6, 900e6, 1e-3, 64)
    channel = process_channel(
        files["target"],
        chirp,
        background_filename=files["background"],
    )
    fast = reconstruct_fast_time(channel, window="rectangular")
    deramped = synthesize_deramped_sweep(channel)
    stretch_fast_time = np.conj(np.fft.fft(deramped.complex_signal)) / chirp.samples

    assert np.max(np.abs(channel.response)) > 0
    np.testing.assert_allclose(
        stretch_fast_time,
        fast.complex_envelope,
        rtol=5e-13,
        atol=5e-13 * np.max(np.abs(fast.complex_envelope)),
    )
