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

"""End-to-end impulse-response synthesis against a direct FDTD run."""

import numpy as np
import pytest

import gprMax
from toolboxes.SFCW.processing import engineering_dft, load_receiver, load_source

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


def test_impulse_synthesis_reproduces_direct_ricker_simulation(tmp_path):
    impulse_file = tmp_path / "impulse"
    ricker_file = tmp_path / "ricker"
    for waveform, output in (("impulse", impulse_file), ("ricker", ricker_file)):
        gprMax.run(
            scenes=[_scene(waveform)],
            n=1,
            outputfile=output,
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )

    impulse_source = load_source(impulse_file.with_suffix(".h5"))
    impulse_receiver = load_receiver(impulse_file.with_suffix(".h5"))
    ricker_source = load_source(ricker_file.with_suffix(".h5"))
    ricker_receiver = load_receiver(ricker_file.with_suffix(".h5"))

    # The impulse is applied at the first source-update time. Therefore, the
    # receiver trace divided by its amplitude is the causal discrete impulse
    # response of exactly the same FDTD system used for the Ricker run.
    nonzero = np.flatnonzero(impulse_source.samples)
    np.testing.assert_array_equal(nonzero, [0])
    impulse_response = impulse_receiver.samples / impulse_source.samples[0]
    synthesised_trace = np.convolve(
        impulse_response,
        ricker_source.samples,
        mode="full",
    )[: ricker_receiver.samples.size]

    # Compare identical finite observation windows. Forming the causal linear
    # convolution before cropping is important: multiplying separately
    # truncated spectra does not represent the same finite-record operation.
    peak = np.max(np.abs(ricker_receiver.samples))
    max_relative_error = np.max(np.abs(synthesised_trace - ricker_receiver.samples)) / peak
    assert max_relative_error < 1e-11

    frequencies = np.linspace(200e6, 800e6, 61)
    direct_spectrum = engineering_dft(
        ricker_receiver.samples,
        ricker_receiver.dt,
        frequencies,
        time_offset=ricker_receiver.time_offset,
    )
    synthesised_spectrum = engineering_dft(
        synthesised_trace,
        ricker_receiver.dt,
        frequencies,
        time_offset=ricker_receiver.time_offset,
    )
    spectral_relative_error = np.max(np.abs(synthesised_spectrum - direct_spectrum)) / np.max(
        np.abs(direct_spectrum)
    )
    assert spectral_relative_error < 1e-11
