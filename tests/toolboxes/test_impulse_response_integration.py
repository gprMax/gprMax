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

import numpy as np
import pytest

import gprMax
from toolboxes.ImpulseResponse import (
    load_source_sampling,
    sample_builtin_waveform,
    synthesise_output,
)
from toolboxes.SFCW.processing import load_receiver, load_source

pytestmark = pytest.mark.integration


def _scene(wave_type):
    inf = float("inf")
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(0.005, 0.005, 0.005)))
    scene.add(gprMax.Domain(p1=(0.2, 0.2, inf)))
    scene.add(gprMax.TimeWindow(time=30e-9))
    scene.add(gprMax.OMPThreads(n=2))
    scene.add(gprMax.Waveform(wave_type=wave_type, amp=1, freq=500e6, id="source"))
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z",
            p1=(0.1, 0.1, inf),
            waveform_id="source",
        )
    )
    scene.add(gprMax.Rx(p1=(0.12, 0.1, inf), id="response", outputs=["Ez"]))
    return scene


def _hard_voltage_scene(wave_type):
    inf = float("inf")
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(0.005, 0.005, 0.005)))
    scene.add(gprMax.Domain(p1=(0.2, 0.2, inf)))
    scene.add(gprMax.TimeWindow(time=20e-9))
    scene.add(gprMax.OMPThreads(n=2))
    scene.add(gprMax.Waveform(wave_type=wave_type, amp=1, freq=500e6, id="source"))
    scene.add(
        gprMax.VoltageSource(
            polarisation="z",
            p1=(0.1, 0.1, inf),
            resistance=0,
            waveform_id="source",
        )
    )
    scene.add(gprMax.Rx(p1=(0.12, 0.1, inf), id="response", outputs=["Ez"]))
    return scene


def test_toolbox_ricker_synthesis_reproduces_direct_fdtd_run(tmp_path):
    impulse_stem = tmp_path / "impulse"
    ricker_stem = tmp_path / "ricker"
    for waveform_type, output in (("impulse", impulse_stem), ("ricker", ricker_stem)):
        gprMax.run(
            scenes=[_scene(waveform_type)],
            n=1,
            outputfile=output,
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )

    impulse_file = impulse_stem.with_suffix(".h5")
    direct_file = ricker_stem.with_suffix(".h5")
    impulse_source = load_source_sampling(impulse_file)
    target = sample_builtin_waveform(impulse_source, "ricker", 1, 500e6, "ricker500")
    synthesis = synthesise_output(impulse_file, target)
    direct_source = load_source(direct_file)
    direct_receiver = load_receiver(direct_file)

    np.testing.assert_allclose(target.samples, direct_source.samples, rtol=2e-13, atol=2e-16)
    peak = np.max(np.abs(direct_receiver.samples))
    relative_error = np.max(np.abs(synthesis.receivers[0].samples - direct_receiver.samples)) / peak
    assert relative_error < 1e-11


def test_toolbox_preserves_hard_voltage_source_evaluation_timing(tmp_path):
    files = {}
    for waveform_type in ("impulse", "ricker"):
        output = tmp_path / f"hard_{waveform_type}"
        gprMax.run(
            scenes=[_hard_voltage_scene(waveform_type)],
            n=1,
            outputfile=output,
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )
        files[waveform_type] = output.with_suffix(".h5")

    impulse_source = load_source_sampling(files["impulse"])
    target = sample_builtin_waveform(impulse_source, "ricker", 1, 500e6, "ricker500")
    synthesis = synthesise_output(files["impulse"], target)
    direct_source = load_source(files["ricker"])
    direct_receiver = load_receiver(files["ricker"])

    assert impulse_source.signal.time_offset == pytest.approx(impulse_source.signal.dt)
    assert impulse_source.evaluation_time_offset == 0.0
    np.testing.assert_allclose(target.samples, direct_source.samples, rtol=2e-13, atol=2e-16)
    peak = np.max(np.abs(direct_receiver.samples))
    relative_error = np.max(np.abs(synthesis.receivers[0].samples - direct_receiver.samples)) / peak
    assert relative_error < 1e-11
