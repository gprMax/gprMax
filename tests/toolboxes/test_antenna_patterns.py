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

import json

import h5py
import numpy as np

from gprMax.toolboxes.AntennaPatterns.initial_save import process_pattern
from gprMax.toolboxes.AntennaPatterns.plot_fields import load_pattern_data, plot_pattern


def test_pattern_processing_and_plotting_workflow(tmp_path):
    outputfile = tmp_path / "pattern.h5"
    configfile = tmp_path / "pattern_config.json"
    patternfile = tmp_path / "pattern.npz"
    plotfile = tmp_path / "pattern.pdf"
    positions = ((1, 0, 1), (1, 0, 0), (1, 0, -1))

    with h5py.File(outputfile, "w") as output:
        output.attrs["Iterations"] = 8
        output.attrs["dt"] = 1e-10
        for index, position in enumerate(positions, start=1):
            receiver = output.create_group(f"rxs/rx{index}")
            receiver.attrs["Name"] = f"pattern_{index:03d}"
            receiver.attrs["Position"] = position
            receiver.create_dataset("Ex", data=np.linspace(0, index, 8))
            receiver.create_dataset("Ey", data=np.linspace(0, 2 * index, 8))
            receiver.create_dataset("Ez", data=np.linspace(0, 3 * index, 8))

    configfile.write_text(
        json.dumps(
            {
                "pattern": "E",
                "radii": [1],
                "theta_degrees": [-45, 0, 45],
                "origin": [0, 0, 0],
                "receiver_prefix": "pattern_",
                "relative_permittivity": 4,
                "relative_permeability": 1,
                "centre_frequency": 1e9,
                "antenna_dimension": 0.1,
                "impedance_scaling": True,
            }
        ),
        encoding="utf-8",
    )

    process_pattern(outputfile, configfile, patternfile)
    data = load_pattern_data(patternfile)
    plot_pattern(data, plotfile)

    assert data["patterns"].shape == (1, 3)
    assert np.all(np.isfinite(data["patterns"]))
    assert np.all(data["patterns"] >= 0)
    assert plotfile.is_file()
    assert plotfile.stat().st_size > 0

    # The writer may renumber public groups; physical angle order is defined
    # by these zero-padded Names, not by rxN. An ordinary feed is not a bin.
    with h5py.File(outputfile, "r+") as output:
        output.move("rxs/rx1", "rxs/temporary")
        output.move("rxs/rx3", "rxs/rx1")
        output.move("rxs/temporary", "rxs/rx3")
        feed = output.create_group("rxs/rx4")
        feed.attrs["Name"] = "ordinary_feed"
        feed.attrs["Position"] = (0, 0, 0)
        feed["Ez"] = np.ones(8) * 1e6
    process_pattern(outputfile, configfile, patternfile)
    np.testing.assert_array_equal(load_pattern_data(patternfile)["patterns"], data["patterns"])
