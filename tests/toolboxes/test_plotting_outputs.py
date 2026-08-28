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
