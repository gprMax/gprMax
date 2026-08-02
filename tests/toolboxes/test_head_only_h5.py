# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

import h5py
import numpy as np

from toolboxes.AustinManWoman.head_only_h5 import extract_head


def test_extract_head_is_import_safe_and_preserves_geometry_metadata(tmp_path):
    source = tmp_path / "body.h5"
    data = np.arange(2 * 3 * 16, dtype=np.int16).reshape(2, 3, 16)
    with h5py.File(source, "w") as output:
        output.attrs["dx_dy_dz"] = (0.001, 0.001, 0.002)
        output.create_dataset("data", data=data)

    head = extract_head(source)

    with h5py.File(head, "r") as output:
        np.testing.assert_array_equal(output["data"], data[:, :, 14:])
        np.testing.assert_allclose(output.attrs["dx_dy_dz"], [0.001, 0.001, 0.002])
