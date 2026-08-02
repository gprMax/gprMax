# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

import h5py
import numpy as np

from toolboxes.Plotting.plot_Ascan import fft_plot_range
from toolboxes.Plotting.plot_Bscan import gather_receiver_outputs


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
