import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.sources as sources_module


@pytest.mark.integration
@pytest.mark.slow
def test_automatic_guard_below_cutoff_does_not_corrupt_through_s21(
    tmp_path,
    monkeypatch,
):
    """Exercise the reported rectangular-waveguide automatic-guard failure."""

    source = Path(__file__).with_name("eigenmode_auto_guard_failure.in")
    inputfile = tmp_path / source.name
    input_text = source.read_text(encoding="utf-8")
    inputfile.write_text(input_text, encoding="utf-8")
    outputfile = tmp_path / "auto_guard_failure"
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    gprMax.run(
        inputfile=inputfile,
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
    )

    with h5py.File(outputfile.with_suffix(".h5"), "r") as output:
        source_port = output["eigenmode_ports/port1"]
        assert source_port.attrs["RequestedAnchorPolicy"] == "auto"
        resolved_anchors = np.asarray(source_port.attrs["AnchorFrequencies"])
        candidate_anchors = np.asarray(source_port.attrs["CandidateAnchorFrequencies"])
        usable = source_port["anchor_mode_valid"][...][:, 0].astype(bool)
        reported_guard = np.isclose(candidate_anchors, 22.660393e9, rtol=1e-6)
        assert np.any(reported_guard)
        assert not np.any(usable[reported_guard])
        assert not np.any((candidate_anchors < 24.9827048e9) & usable)
        assert np.all(source_port["power_normalization_valid"][...])
        source_signature = {
            "resolved_policy": source_port.attrs["ResolvedAnchorPolicy"],
            "resolved_anchors": resolved_anchors.copy(),
            "candidate_anchors": candidate_anchors.copy(),
            "mode_policies": np.asarray(source_port.attrs["ModeAnchorPolicies"]).copy(),
            "valid": source_port["anchor_mode_valid"][...].copy(),
            "propagating": source_port["anchor_mode_propagating"][...].copy(),
        }

    passive_input = tmp_path / "auto_guard_with_passive_plane.in"
    passive_input.write_text(
        input_text.replace(
            "#eigenmode_excitation:",
            "#eigenmode_port: 3 0.006 0.001 0.001 0.006 0.007 0.005 - 1 auto n\n"
            "#eigenmode_excitation:",
            1,
        ),
        encoding="utf-8",
    )
    passive_output = tmp_path / "auto_guard_with_passive_plane"
    gprMax.run(
        inputfile=passive_input,
        n=1,
        outputfile=passive_output,
        hide_progress_bars=True,
    )
    with h5py.File(passive_output.with_suffix(".h5"), "r") as output:
        source_port = output["eigenmode_ports/port1"]
        assert source_port.attrs["ResolvedAnchorPolicy"] == source_signature["resolved_policy"]
        np.testing.assert_array_equal(
            source_port.attrs["AnchorFrequencies"],
            source_signature["resolved_anchors"],
        )
        np.testing.assert_array_equal(
            source_port.attrs["CandidateAnchorFrequencies"],
            source_signature["candidate_anchors"],
        )
        np.testing.assert_array_equal(
            source_port.attrs["ModeAnchorPolicies"],
            source_signature["mode_policies"],
        )
        np.testing.assert_array_equal(
            source_port["anchor_mode_valid"][...],
            source_signature["valid"],
        )
        np.testing.assert_array_equal(
            source_port["anchor_mode_propagating"][...],
            source_signature["propagating"],
        )

    csv_path = outputfile.with_name(outputfile.name + "_sparameters.csv")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        s21 = [
            row
            for row in csv.DictReader(stream)
            if row["source_port"] == "1"
            and row["source_mode"] == "1"
            and row["destination_port"] == "2"
            and row["destination_mode"] == "1"
        ]

    assert len(s21) == 29
    assert all(row["valid"] == "1" for row in s21)
    assert max(abs(float(row["S_magnitude_db"])) for row in s21) < 0.1
    assert any("non-propagating" in warning for warning in warnings)
