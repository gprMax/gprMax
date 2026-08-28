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
import pytest

from toolboxes.Utilities.outputfiles_merge import get_output_data, merge_files


def _write_output(filename, offset=0, iterations=3):
    with h5py.File(filename, "w") as output:
        output.attrs["Title"] = "merge test"
        output.attrs["Iterations"] = iterations
        output.attrs["nrx"] = 1
        output.attrs["dt"] = 1e-10
        receiver = output.create_group("rxs/rx1")
        receiver.attrs["Name"] = "surface"
        receiver.attrs["Position"] = (1.0, 2.0, 3.0)
        field = receiver.create_dataset("Ez", data=np.arange(iterations) + offset)
        field.attrs["SampleInterval"] = 1e-10
        field.attrs["TimeSampleOffset"] = 0.0

        subgrid = output.create_group("subgrids/fine")
        subgrid.attrs["Iterations"] = 2 * iterations
        subgrid.attrs["nrx"] = 1
        subgrid.attrs["dt"] = 0.5e-10
        subgrid_receiver = subgrid.create_group("rxs/rx1")
        subgrid_receiver.attrs["Name"] = "fine surface"
        subgrid_receiver.create_dataset("Ex", data=np.arange(2 * iterations) + 10 * offset)


def test_merge_preserves_receiver_metadata_and_subgrid_outputs(tmp_path):
    files = [tmp_path / "model1.h5", tmp_path / "model2.h5"]
    _write_output(files[0])
    _write_output(files[1], offset=1)

    merged = merge_files(files)

    main, _ = get_output_data(merged, 1, "Ez")
    fine, _ = get_output_data(merged, 1, "Ex", "subgrids/fine")
    np.testing.assert_array_equal(main, [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(fine[:, 1] - fine[:, 0], np.full(6, 10))
    with h5py.File(merged, "r") as output:
        assert output["rxs/rx1"].attrs["Name"] == "surface"
        np.testing.assert_array_equal(output["rxs/rx1"].attrs["Position"], [1, 2, 3])
        assert output["rxs/rx1/Ez"].attrs["SampleInterval"] == 1e-10
        assert output["rxs/rx1/Ez"].attrs["TimeSampleOffset"] == 0.0


def test_merge_rejects_inconsistent_iterations(tmp_path):
    files = [tmp_path / "model1.h5", tmp_path / "model2.h5"]
    _write_output(files[0])
    _write_output(files[1], iterations=4)

    destination = tmp_path / "merged.h5"
    with pytest.raises(ValueError, match="Inconsistent Iterations"):
        merge_files(files, destination)

    assert not destination.exists()
