"""Geometry/reference checks, not substitutes for the manual MoM/FDTD runs."""

import json

import numpy as np
import pytest
from scipy.io import loadmat

import gprMax
from testing.other_codes.matlab_mom.antenna_vivaldi_fs import vivaldi_antenna_gprmax as model


@pytest.mark.parametrize(
    "point,expected",
    [
        ((0.0, 0.04), True),
        ((0.0, 0.0), False),
        ((-0.128, 0.0), False),
        ((-0.15, 0.04), True),
        ((0.16, 0.0), False),
        ((0.0, 0.07), False),
    ],
)
def test_exported_outline_preserves_sheet_slot_and_cavity(point, expected):
    assert model.conductor_contains(np.array([point]), model.read_geometry())[0] == expected


@pytest.mark.parametrize("mesh", model.MESHES)
def test_scene_uses_thin_plates_and_unshorted_aligned_gap(mesh):
    scene, metadata = model.build_scene(mesh)
    dl = np.array(model.MESHES[mesh])
    masks = model.edge_masks(model.read_geometry(), dl)
    feed = np.array(metadata["feed_node_m"])
    np.testing.assert_allclose(feed, (0.116, 0.112, 0.072), rtol=0, atol=1e-14)
    np.testing.assert_allclose(feed / dl, np.rint(feed / dl), rtol=0, atol=1e-10)
    i, j, _ = np.rint(feed / dl).astype(int)
    assert not masks[1][i, j]
    assert masks[1][i, j - 1] and masks[1][i, j + 1]
    assert all(isinstance(obj, gprMax.Plate) for obj in scene.geometry_objects)
    for obj in scene.geometry_objects:
        assert obj.kwargs["p1"][2] == obj.kwargs["p2"][2] == model.ORIGIN[2]
    # Reflection about the slot centre, which is halfway between y nodes.
    np.testing.assert_array_equal(masks[0][:, :450], masks[0][:, :450][:, ::-1])
    np.testing.assert_array_equal(masks[1][:, :449], masks[1][:, :449][:, ::-1])


def test_runs_do_not_join_separate_pec_banks():
    assert list(model.runs(np.array([0, 1, 1, 0, 1, 0], dtype=bool))) == [(1, 3), (4, 5)]
    assert list(model.runs(np.zeros(4, dtype=bool))) == []
    assert list(model.runs(np.ones(4, dtype=bool))) == [(0, 4)]


def test_reference_covers_requested_patterns_and_contains_complex_ports():
    reference = loadmat(model.RESULTS / "vivaldi_antenna_matlab.mat")
    frequency = reference["frequency_hz"].ravel()
    z, s = reference["input_impedance"].ravel(), reference["s11"].ravel()
    assert np.all(np.isfinite(z)) and np.all(z.real > 0)
    assert np.all(np.isfinite(s)) and np.all(abs(s) <= 1)
    np.testing.assert_allclose(s, (z - 50) / (z + 50), rtol=1e-13, atol=1e-14)
    assert all(np.any(np.isclose(frequency, f)) for f in model.PATTERN_FREQUENCIES)
    for name in ("directivity_xy_dbi", "directivity_xz_dbi"):
        assert reference[name].shape == (181, 3)
        np.testing.assert_allclose(reference[name][0], reference[name][-1], atol=1e-9)


def test_geometry_reader_rejects_wrong_units(tmp_path):
    geometry = model.read_geometry()
    geometry["coordinate_units"] = "mm"
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match="metres"):
        model.read_geometry(path)
