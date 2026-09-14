"""Scientific contracts for the editable rectangular-patch optimisation."""

from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import gprMax
from gprMax.toolboxes.Optimisation import Integer, PortSpectrum
from gprMax.toolboxes.Optimisation.examples import rectangular_patch as example


def spectrum():
    """A deeper dip away from the target must not determine this score."""
    frequency = np.arange(2.8e9, 3.401e9, 10e6)
    s11 = np.full(frequency.size, 0.6 + 0j)
    s11[10] = 0.001  # -60 dB at 2.9 GHz: not the requested objective.
    s11[30] = 0.1j  # -20 dB at the requested 3.1 GHz.
    valid = np.ones(frequency.size, dtype=bool)
    return PortSpectrum(
        frequency,
        s11,
        valid,
        np.full(frequency.size, 50 + 0j),
        valid.copy(),
        50,
        -80,
        Path("synthetic"),
        "/ports/feed",
        10e6,
    )


class Output:
    def __init__(self, data):
        self.data = data
        self.saved = {}

    def port(self, name):
        assert name == "frills/frill1"
        return self.data

    def save_npz(self, name, **arrays):
        self.saved[name] = arrays


def test_objective_uses_fixed_frequency_and_records_its_complex_value():
    output = Output(spectrum())
    assert example.evaluate(example.STARTING_DESIGN, output) == pytest.approx(-20)
    assert output.saved["s11.npz"]["s11_at_target"] == pytest.approx(0.1j)
    assert output.saved["s11.npz"]["requested_feed_offset_mm"] == 7
    assert output.saved["s11.npz"]["realised_feed_offset_mm"] == 7


def test_odd_length_records_requested_and_realised_centre_offset():
    parameters = dict(length_mm=31, width_mm=30, feed_offset_mm=4)
    output = Output(spectrum())
    example.evaluate(parameters, output)
    assert output.saved["s11.npz"]["requested_feed_offset_mm"] == 4
    assert output.saved["s11.npz"]["realised_feed_offset_mm"] == 4.5


def test_bad_matching_remains_a_valid_candidate():
    data = spectrum()
    data.s11[:] = 0.99
    assert example.evaluate(example.STARTING_DESIGN, Output(data)) == pytest.approx(
        20 * np.log10(0.99)
    )


@pytest.mark.parametrize("change", ["invalid", "duration", "decay", "nonfinite"])
def test_unusable_spectra_cannot_receive_a_good_score(change):
    data = spectrum()
    if change == "invalid":
        data.valid[30] = False
    elif change == "duration":
        data = replace(data, independent_frequency_resolution_hz=40e6)
    elif change == "decay":
        data = replace(data, tail_relative_db=-20)
    else:
        data.s11[30] = complex(np.nan, 0)
    with pytest.raises(ValueError):
        example.evaluate(example.STARTING_DESIGN, Output(data))


@pytest.mark.parametrize("length,width,offset", list(product((26, 27, 36), (12, 13, 44), (1, 12))))
def test_geometry_keeps_dimensions_and_full_height_frill_probe(length, width, offset):
    scene = example.build_model(dict(length_mm=length, width_mm=width, feed_offset_mm=offset))
    plates = [o for o in scene.geometry_objects if isinstance(o, gprMax.Plate)]
    ground, patch = [o.kwargs for o in plates]
    np.testing.assert_allclose(
        np.subtract(patch["p2"], patch["p1"])[:2], [length * 0.001, width * 0.001]
    )
    wire = next(o for o in scene.geometry_objects if isinstance(o, gprMax.ThinWire)).kwargs
    source = next(o for o in scene.grid_objects if isinstance(o, gprMax.MagneticFrillSource))
    assert not any(isinstance(o, gprMax.VoltageSource) for o in scene.grid_objects)
    assert wire["p1"] == source.point
    assert wire["p1"][2] == ground["p1"][2]
    assert wire["p2"][2] == patch["p1"][2]
    assert wire["p2"][2] - wire["p1"][2] == pytest.approx(0.003)
    centre_x = (patch["p1"][0] + patch["p2"][0]) / 2
    # Check the actual geometric centre, independently of the rounding helper.
    expected_offset = offset + (0.5 if length % 2 else 0)
    assert centre_x - source.point[0] == pytest.approx(expected_offset * 0.001)
    assert source.point[0] / 0.001 == pytest.approx(round(source.point[0] / 0.001))
    assert source.point[0] - patch["p1"][0] >= 0.001 - 1e-12
    assert ground["p1"][0] < patch["p1"][0] < source.point[0] < patch["p2"][0] < ground["p2"][0]


def test_mesh_refinement_preserves_patch_and_probe_radius(monkeypatch):
    parameters = dict(length_mm=31, width_mm=33, feed_offset_mm=6)
    coarse = example.build_model(parameters)
    monkeypatch.setattr(example, "CELL_SIZE_M", 0.0005)
    fine = example.build_model(parameters)
    for a, b in zip(coarse.geometry_objects, fine.geometry_objects):
        assert a.kwargs == b.kwargs
    for a, b in zip(coarse.grid_objects, fine.grid_objects):
        if isinstance(a, gprMax.MagneticFrillSource):
            assert a.point == b.point


@pytest.mark.parametrize("name,offset", [("antenna_I", 4), ("antenna_II", 12)])
def test_published_feed_offset_is_measured_from_patch_centre(name, offset):
    assert example.PUBLISHED_DESIGNS[name]["feed_offset_mm"] == offset
    scene = example.build_model(example.PUBLISHED_DESIGNS[name])
    patch = [o.kwargs for o in scene.geometry_objects if isinstance(o, gprMax.Plate)][-1]
    feed = next(o for o in scene.grid_objects if isinstance(o, gprMax.MagneticFrillSource))
    centre = (patch["p1"][0] + patch["p2"][0]) / 2
    assert centre - feed.point[0] == pytest.approx(offset * 0.001)


@pytest.mark.parametrize("length", [26, 30, 36])
def test_fixed_centre_offset_keeps_feed_position_when_even_length_changes(length):
    scene = example.build_model(dict(length_mm=length, width_mm=30, feed_offset_mm=4))
    source = next(o for o in scene.grid_objects if isinstance(o, gprMax.MagneticFrillSource))
    assert source.point[0] == pytest.approx(0.046)


def test_all_default_lengths_and_offsets_keep_one_mm_edge_clearance():
    for length, offset in product(range(26, 37), range(1, 13)):
        example.validate_model_settings(dict(length_mm=length, width_mm=12, feed_offset_mm=offset))
        assert (
            length / 2
            - example.realised_feed_offset_mm(dict(length_mm=length, feed_offset_mm=offset))
            >= 1
        )


def test_unsafe_edited_bounds_cannot_put_the_feed_on_the_patch_edge(monkeypatch):
    monkeypatch.setattr(
        example, "PARAMETERS", dict(example.PARAMETERS, length_mm=Integer(24, 36, "mm"))
    )
    with pytest.raises(ValueError, match="at least 1 mm"):
        example.build_model(dict(length_mm=24, width_mm=30, feed_offset_mm=12))


def test_old_edge_inset_key_requires_explicit_conversion():
    with pytest.raises(ValueError, match="from the patch centre"):
        example.build_model(dict(length_mm=30, width_mm=30, feed_inset_mm=8))


def test_aperture_validity_rejects_mesh_too_fine_for_assumed_coax(monkeypatch):
    # a=0.2 mm still satisfies the thin-wire condition on a 0.5 mm mesh,
    # but the inferred outer radius exceeds 0.5 mm for this filler.
    monkeypatch.setattr(example, "CELL_SIZE_M", 0.0005)
    monkeypatch.setattr(example, "COAX_FILLER_PERMITTIVITY", 2.2)
    with pytest.raises(ValueError, match="aperture"):
        example.build_model(example.STARTING_DESIGN)
