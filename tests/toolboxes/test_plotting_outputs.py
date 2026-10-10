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

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gprMax.toolboxes.Plotting.plot_Ascan import fft_plot_range, mpl_plot
from gprMax.toolboxes.Plotting.plot_Bscan import gather_receiver_outputs
from gprMax.toolboxes.Plotting.plot_source_wave import check_timewindow
from gprMax.toolboxes.Plotting.plot_source_wave import mpl_plot as plot_source_wave
from gprMax.user_objects.cmds_singleuse import TimeWindow
from gprMax.waveforms import Waveform


def _plotted_source_wave(tmp_path, monkeypatch, wave_type, freq, dt, iterations, fft=False):
    monkeypatch.chdir(tmp_path)
    w = Waveform()
    w.type = wave_type
    w.amp = 1
    w.freq = freq
    plt.close("all")
    pyplot = plot_source_wave(w, (iterations - 1) * dt, dt, iterations, fft=fft, show=False)
    fig = pyplot.gcf()
    lines = [(ax.lines[-1].get_xdata(), ax.lines[-1].get_ydata()) for ax in fig.axes]
    pyplot.close(fig)
    return lines


def test_bscan_gather_does_not_duplicate_first_receiver(tmp_path):
    filename = tmp_path / "receivers.h5"
    with h5py.File(filename, "w") as output:
        output.attrs["nrx"] = 2
        output.attrs["dt"] = 1e-10
        output.create_dataset("rxs/rx1/Ez", data=[1, 2, 3])
        output.create_dataset("rxs/rx2/Ez", data=[4, 5, 6])

    gathered, dt = gather_receiver_outputs(filename, "Ez")

    np.testing.assert_array_equal(gathered, [[1, 4], [2, 5], [3, 6]])
    assert dt == 1e-10


@pytest.mark.parametrize("timewindow, dt", [("6e-9", 1.926e-12), ("3e-9", 1.1e-12)])
def test_plot_source_wave_time_window_matches_solver(timewindow, dt):
    model = SimpleNamespace(dt=dt)
    TimeWindow(time=float(timewindow)).build(model)

    _, iterations = check_timewindow(timewindow, dt)

    assert iterations == model.iterations
    assert (iterations - 1) * dt >= float(timewindow)


def test_plot_source_wave_impulse_has_single_nonzero_sample(tmp_path, monkeypatch):
    dt = 1e-12
    [(time, waveform)] = _plotted_source_wave(tmp_path, monkeypatch, "impulse", 1e9, dt, 234)

    assert np.count_nonzero(waveform) == 1
    assert waveform[0] == 1
    np.testing.assert_array_equal(time, np.arange(234, dtype=float) * dt)


@pytest.mark.parametrize("samples", [9, 10])
def test_plot_source_wave_fft_fallback_plots_all_nonnegative_bins(tmp_path, monkeypatch, samples):
    dt = 1e-12
    _, (freqs, _) = _plotted_source_wave(tmp_path, monkeypatch, "impulse", 0, dt, samples, fft=True)

    allfreqs = np.fft.fftfreq(samples, dt)
    np.testing.assert_array_equal(freqs, allfreqs[allfreqs >= 0])


@pytest.mark.parametrize("samples, stop", [(8, 4), (9, 5), (1, 1)])
def test_fft_plot_range_handles_zero_signal(samples, stop):
    freqs = np.fft.fftfreq(samples, 1e-10)
    power = np.full(samples, -np.inf)

    assert fft_plot_range(freqs, power) == np.s_[0:stop]


@pytest.mark.parametrize("samples", [9, 10])
def test_plot_ascan_fft_of_zero_trace_plots_all_nonnegative_bins(tmp_path, samples):
    dt = 1e-10
    filename = tmp_path / "zero.h5"
    with h5py.File(filename, "w") as output:
        output.attrs["nrx"] = 1
        output.attrs["dt"] = dt
        output.attrs["Title"] = "Test"
        rx = output.create_group("rxs/rx1")
        rx.attrs["Name"] = "rx1"
        rx.create_dataset("Ex", data=np.zeros(samples))

    plt.close("all")
    pyplot = mpl_plot(filename, ["Ex"], fft=True, show=False)
    freqs = pyplot.gcf().axes[1].lines[-1].get_xdata()
    pyplot.close("all")

    allfreqs = np.fft.fftfreq(samples, dt)
    np.testing.assert_array_equal(freqs, allfreqs[allfreqs >= 0])


@pytest.mark.parametrize("samples", [9, 10])
def test_plot_ascan_fft_of_impulse_plots_all_nonnegative_bins(tmp_path, samples):
    dt = 1e-10
    filename = tmp_path / "impulse.h5"
    with h5py.File(filename, "w") as output:
        output.attrs["nrx"] = 1
        output.attrs["dt"] = dt
        output.attrs["Title"] = "Test"
        rx = output.create_group("rxs/rx1")
        rx.attrs["Name"] = "rx1"
        data = np.zeros(samples)
        data[0] = 1.0
        rx.create_dataset("Ex", data=data)

    plt.close("all")
    pyplot = mpl_plot(filename, ["Ex"], fft=True, show=False)
    freqs = pyplot.gcf().axes[1].lines[-1].get_xdata()
    pyplot.close("all")

    allfreqs = np.fft.fftfreq(samples, dt)
    np.testing.assert_array_equal(freqs, allfreqs[allfreqs >= 0])


def test_plot_ascan_single_output_missing_reports_correct_receiver(tmp_path):
    filename = tmp_path / "receivers.h5"
    with h5py.File(filename, "w") as output:
        output.attrs["nrx"] = 2
        output.attrs["dt"] = 1e-10
        output.attrs["Title"] = "Test"
        rx1 = output.create_group("rxs/rx1")
        rx1.attrs["Name"] = "rx1"
        rx1.create_dataset("Ez", data=[1, 2, 3])
        rx2 = output.create_group("rxs/rx2")
        rx2.attrs["Name"] = "rx2"
        rx2.create_dataset("Ey", data=[4, 5, 6])

    with pytest.raises(ValueError, match=r"available output for receiver 2 is Ey"):
        mpl_plot(filename, ["Ez"], show=False)


def test_plot_bscan_gather_filename(tmp_path):
    root = Path(__file__).resolve().parents[2]
    filename = tmp_path / "receivers.h5"
    with h5py.File(filename, "w") as output:
        output.attrs["nrx"] = 2
        output.attrs["dt"] = 1e-10
        output.create_dataset("rxs/rx1/Ez", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
        output.create_dataset("rxs/rx2/Ez", data=np.array([[5.0, 6.0], [7.0, 8.0]]))

    env = dict(
        os.environ,
        PYTHONPATH=str(root),
        MPLCONFIGDIR=str(tmp_path / "mpl"),
        MPLBACKEND="Agg",
    )
    subprocess.run(
        [sys.executable, "-m", "gprMax.toolboxes.Plotting.plot_Bscan", str(filename), "Ez"],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    rx2_file = tmp_path / "receivers_rx2.png"
    assert rx2_file.exists()
    rx2_content = rx2_file.read_bytes()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gprMax.toolboxes.Plotting.plot_Bscan",
            str(filename),
            "Ez",
            "-gather",
        ],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    gathered_file = tmp_path / "receivers_gathered.png"
    assert gathered_file.exists()
    assert rx2_file.read_bytes() == rx2_content
