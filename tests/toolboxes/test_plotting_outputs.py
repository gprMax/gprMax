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

import h5py
import numpy as np
import pytest

from gprMax.toolboxes.Plotting.plot_Ascan import fft_plot_range, mpl_plot
from gprMax.toolboxes.Plotting.plot_Bscan import gather_receiver_outputs


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


def test_fft_plot_range_handles_zero_signal():
    freqs = np.fft.fftfreq(8, 1e-10)
    power = np.full(8, -np.inf)

    assert fft_plot_range(freqs, power) == np.s_[0:4]


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
