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
        output.attrs["MaterialDatabase"] = "body_materials"
        output.create_dataset("data", data=data, chunks=(1, 2, 4), compression="gzip")
        output.create_dataset("material_keys", data=np.asarray(["air", "skin"], dtype="S"))
        metadata = output.create_group("metadata")
        metadata.attrs["model"] = "AustinMan"

    head = extract_head(source)

    with h5py.File(head, "r") as output:
        np.testing.assert_array_equal(output["data"], data[:, :, 14:])
        np.testing.assert_allclose(output.attrs["dx_dy_dz"], [0.001, 0.001, 0.002])
        assert output.attrs["MaterialDatabase"] == "body_materials"
        assert output.attrs["HeadExtractionFirstPlane"] == 14
        np.testing.assert_array_equal(output.attrs["HeadExtractionOriginalDimensions"], data.shape)
        np.testing.assert_array_equal(output["material_keys"], np.asarray([b"air", b"skin"]))
        assert output["metadata"].attrs["model"] == "AustinMan"
        assert output["data"].compression == "gzip"


def test_extract_head_accepts_an_explicit_first_plane(tmp_path):
    source = tmp_path / "body.h5"
    data = np.arange(2 * 2 * 10, dtype=np.uint16).reshape(2, 2, 10)
    with h5py.File(source, "w") as output:
        output.attrs["dx_dy_dz"] = (0.002, 0.002, 0.002)
        output.create_dataset("data", data=data)

    head = extract_head(source, first_head_plane=6)

    with h5py.File(head, "r") as output:
        np.testing.assert_array_equal(output["data"], data[:, :, 6:])


def test_extract_head_normalises_legacy_austin_spacing(tmp_path):
    source = tmp_path / "AustinMan_v2.3.h5"
    data = np.arange(2 * 2 * 8, dtype=np.uint16).reshape(2, 2, 8)
    with h5py.File(source, "w") as output:
        output.attrs["dx, dy, dz"] = (0.008, 0.008, 0.008)
        output.create_dataset("data", data=data)

    head = extract_head(source)

    with h5py.File(head, "r") as output:
        np.testing.assert_allclose(output.attrs["dx_dy_dz"], [0.008, 0.008, 0.008])
        np.testing.assert_allclose(output.attrs["dx, dy, dz"], [0.008, 0.008, 0.008])
        np.testing.assert_array_equal(output["data"], data[:, :, 7:])


def test_extract_head_rejects_an_invalid_first_plane(tmp_path):
    source = tmp_path / "body.h5"
    with h5py.File(source, "w") as output:
        output.attrs["dx_dy_dz"] = (0.002, 0.002, 0.002)
        output.create_dataset("data", data=np.zeros((2, 2, 8), dtype=np.uint16))

    with np.testing.assert_raises_regex(ValueError, "between 0 and 7"):
        extract_head(source, first_head_plane=8)
