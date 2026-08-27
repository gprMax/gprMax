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

import csv

import h5py
import numpy as np
import pytest

from toolboxes.GeometryImport.common import (
    build_tag_volume,
    unique_normalised_tags,
    write_geometry_hdf5,
)
from toolboxes.GeometryImport.volume import (
    LabelVolume,
    _canonicalise_axis_aligned,
    _integer_labels,
    convert_label_volume,
    write_label_template,
)
from toolboxes.STEPtoVoxel.voxeliser import make_grid_from_bbox


def test_component_tags_are_compact_and_independent_of_components():
    components = np.asarray([[[0, 1, 2, -1]]], dtype=np.int16)

    tags, names = build_tag_volume(components, ("eye", "eye", None))

    assert names == ("untagged", "eye")
    assert tags.dtype == np.uint8
    np.testing.assert_array_equal(tags.ravel(), [1, 1, 0, 0])


def test_tag_dtype_expands_only_when_required():
    components = np.arange(256, dtype=np.int16).reshape(1, 16, 16)
    tags, names = build_tag_volume(components, tuple(f"part_{index}" for index in range(256)))

    assert len(names) == 257
    assert tags.dtype == np.uint16


def test_external_names_are_normalised_without_accidental_merging():
    assert unique_normalised_tags(("left eye", "left-eye", "left eye")) == (
        "left_eye",
        "left-eye",
        "left_eye_2",
    )


def test_zero_padding_grid_still_encloses_fractional_lower_bound():
    lower = np.asarray((0.0006, -0.0014, 0.002))
    upper = np.asarray((0.0031, 0.0022, 0.0045))

    grid = make_grid_from_bbox(
        lower,
        upper,
        dx=0.001,
        dy=0.001,
        dz=0.001,
        pad=0,
    )

    assert np.all(grid.origin_world <= lower)
    assert np.all(grid.origin_world + grid.nxyz * grid.dxyz_world >= upper)


def test_grid_lattice_snap_is_stable_at_roundoff_scale():
    lower = np.asarray((0.001 - 1e-15, 0.0, 0.0))
    upper = np.asarray((0.002, 0.001, 0.001))

    grid = make_grid_from_bbox(lower, upper, dx=0.001, pad=0)

    assert grid.origin_world[0] == pytest.approx(0.001)


def test_shared_writer_omits_tags_when_none_exist(tmp_path):
    path = tmp_path / "geometry.h5"
    write_geometry_hdf5(path, np.zeros((2, 1, 1), dtype=np.int16), (1e-3, 1e-3, 1e-3))

    with h5py.File(path) as geometry:
        assert "tag_data" not in geometry
        assert "tag_names" not in geometry
        assert "GeometryTagsSchemaVersion" not in geometry.attrs


@pytest.mark.parametrize(
    "data",
    (
        np.asarray([[[65535]]], dtype=np.uint32),
        np.asarray([[[1.5]]], dtype=float),
    ),
)
def test_shared_writer_rejects_indices_which_cannot_be_stored_safely(tmp_path, data):
    with pytest.raises(ValueError):
        write_geometry_hdf5(tmp_path / "geometry.h5", data, (1e-3, 1e-3, 1e-3))


@pytest.mark.parametrize(
    "spacing",
    ((0.0, 1e-3, 1e-3), (np.nan, 1e-3, 1e-3), (np.inf, 1e-3, 1e-3)),
)
def test_shared_writer_rejects_invalid_spacing(tmp_path, spacing):
    with pytest.raises(ValueError, match="three positive values"):
        write_geometry_hdf5(
            tmp_path / "geometry.h5",
            np.zeros((1, 1, 1), dtype=np.int16),
            spacing,
        )


def test_shared_writer_rejects_nonfinite_origin(tmp_path):
    with pytest.raises(ValueError, match="finite coordinates"):
        write_geometry_hdf5(
            tmp_path / "geometry.h5",
            np.zeros((1, 1, 1), dtype=np.int16),
            (1e-3, 1e-3, 1e-3),
            origin=(0.0, np.nan, 0.0),
        )


def test_axis_permutation_and_flip_preserve_label_positions():
    data = np.arange(24).reshape(2, 3, 4)
    # Source axis 0 points to -y, source 1 to +x, source 2 to +z.
    vectors = np.asarray(((0, -2, 0), (3, 0, 0), (0, 0, 4)), dtype=float)

    result, spacing, origin = _canonicalise_axis_aligned(data, vectors, (10, 20, 30), unit_factor=1e-3)

    expected = np.flip(np.transpose(data, (1, 0, 2)), axis=1)
    np.testing.assert_array_equal(result, expected)
    assert spacing == (0.003, 0.002, 0.004)
    assert origin == pytest.approx((0.01, 0.018, 0.03))


def test_label_reader_uses_smallest_safe_integer_storage():
    assert _integer_labels(np.asarray([[[0.0, 255.0]]])).dtype == np.uint8
    assert _integer_labels(np.asarray([[[-1, 128]]])).dtype == np.int16


def test_label_reader_rejects_complex_values():
    with pytest.raises(ValueError, match="complex"):
        _integer_labels(np.asarray([[[1 + 0j]]]))


def test_label_template_keeps_background_available_but_excluded(tmp_path):
    volume = LabelVolume(
        np.asarray([[[0, 1]]]),
        (1e-3, 1e-3, 1e-3),
        (0, 0, 0),
        {1: "grey matter"},
        "synthetic",
    )
    path = write_label_template(volume, tmp_path / "labels.csv")

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["label"] == "0"
    assert rows[0]["include"] == "n"
    assert rows[1]["geometry_tag"] == "grey_matter"


def test_label_conversion_writes_materials_and_semantic_tags(tmp_path, monkeypatch):
    volume = LabelVolume(
        np.asarray([[[0, 1, 2, 2]]]),
        (1e-3, 2e-3, 3e-3),
        (0.1, 0.2, 0.3),
        {1: "left eye", 2: "right eye"},
        "synthetic",
    )
    monkeypatch.setattr(
        "toolboxes.GeometryImport.volume.load_label_volume",
        lambda source, unit="auto": volume,
    )
    assignments = tmp_path / "labels.csv"
    assignments.write_text(
        "label,name,include,material_name,geometry_tag\n"
        "0,background,n,,\n"
        "1,left eye,y,vitreous,left_eye\n"
        "2,right eye,y,vitreous,right_eye\n",
        encoding="utf-8",
    )

    result = convert_label_volume("unused.nii", assignments, tmp_path / "output")

    # The reusable HDF5 geometry does not depend on the optional PyVista/VTK
    # visualisation stack used to create ``geometry_preview.vti``.
    if result.preview_file is not None:
        assert result.preview_file.is_file()
    with h5py.File(result.geometry_file) as geometry:
        assert [value.decode() for value in geometry["material_keys"][:]] == ["material_000_vitreous"]
        assert [value.decode() for value in geometry["tag_names"][:]] == [
            "untagged",
            "left_eye",
            "right_eye",
        ]
        np.testing.assert_array_equal(geometry["data"][:].ravel(), [-1, 0, 0, 0])
        np.testing.assert_array_equal(geometry["tag_data"][:].ravel(), [0, 1, 2, 2])
        assert tuple(geometry.attrs["origin_xyz"]) == pytest.approx((0.0995, 0.199, 0.2985))
