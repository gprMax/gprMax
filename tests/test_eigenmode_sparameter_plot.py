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
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from testing.regression.eigenmode_sources.bending_waveguide.plot_bend_comparison import (
    plot_comparison,
)
from testing.regression.eigenmode_sources.plot_sparameters import plot_tree
from testing.regression.eigenmode_sources.validate_sparameters import (
    validate_tree,
)
from testing.regression.eigenmode_sources.plot_snapshots import snapshot_paths


def _load_example_plotter():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "features"
        / "eigenmode_ports"
        / "example_1_straight_waveguide"
        / "plot_results.py"
    )
    spec = importlib.util.spec_from_file_location("eigenmode_example_plotter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_snapshot(path: Path, time_ns: float):
    with h5py.File(path, "w") as output:
        output.attrs["time"] = time_ns * 1e-9
        output.attrs["dx_dy_dz"] = (0.001, 0.001, 0.001)
        output["Ez"] = np.full((3, 2, 1), time_ns)


def _write_case(root, source_mode, primary_transmission_db=-1, case_name=None):
    case_name = case_name or f"mode{source_mode}"
    case_dir = root / case_name
    case_dir.mkdir()
    path = case_dir / f"{case_name}_sparameters.csv"
    fieldnames = (
        "frequency_hz",
        "source_port",
        "source_mode",
        "destination_port",
        "destination_mode",
        "S_real",
        "S_imag",
        "S_magnitude",
        "S_magnitude_db",
        "S_phase_deg",
        "coefficient_magnitude_squared",
        "valid",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frequency in (1e9, 2e9):
            for destination_port, destination_mode, magnitude_db in (
                (1, source_mode, -30),
                (
                    2,
                    1,
                    primary_transmission_db if source_mode == 1 else -40,
                ),
                (
                    2,
                    2,
                    -40 if source_mode == 1 else primary_transmission_db,
                ),
            ):
                writer.writerow(
                    {
                        "frequency_hz": frequency,
                        "source_port": 1,
                        "source_mode": source_mode,
                        "destination_port": destination_port,
                        "destination_mode": destination_mode,
                        "S_real": 1,
                        "S_imag": 0,
                        "S_magnitude": 10 ** (magnitude_db / 20),
                        "S_magnitude_db": magnitude_db,
                        "S_phase_deg": 0,
                        "coefficient_magnitude_squared": 10 ** (magnitude_db / 10),
                        "valid": 1,
                    }
                )


def test_modal_sparameter_plot_writes_one_combined_plot_per_csv(tmp_path):
    _write_case(tmp_path, 1)
    _write_case(tmp_path, 2)

    outputs = plot_tree(tmp_path)

    assert {path.name for path in outputs} == {
        "mode1_sparameters_plot.png",
        "mode2_sparameters_plot.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)


def test_example_snapshots_are_sorted_by_physical_time_and_can_be_capped(
    tmp_path,
):
    plotter = _load_example_plotter()
    stem = tmp_path / "guide"
    snapshot_dir = tmp_path / "guide_snaps"
    snapshot_dir.mkdir()
    _write_snapshot(snapshot_dir / "guide_1000ps.h5", 1.0)
    _write_snapshot(snapshot_dir / "guide_1600ps.h5", 1.6)
    _write_snapshot(snapshot_dir / "guide_400ps.h5", 0.4)

    snapshots = plotter.read_field_snapshots(stem, maximum_time_ns=1.0)

    assert [snapshot[0] for snapshot in snapshots] == pytest.approx([0.4, 1.0])


def test_regression_snapshot_plot_ignores_stale_generated_files(tmp_path):
    case_dir = tmp_path / "guide"
    snapshot_dir = case_dir / "guide_snaps"
    snapshot_dir.mkdir(parents=True)
    (case_dir / "guide.in").write_text(
        "#snapshot: 0 0 0 1 1 inf 1 1 1 1e-9 xy_center_current.h5\n",
        encoding="utf-8",
    )
    _write_snapshot(snapshot_dir / "xy_center_current.h5", 1.0)
    _write_snapshot(snapshot_dir / "xy_center_stale.h5", 2.0)

    paths = snapshot_paths(case_dir, "xy")

    assert [path.name for path in paths] == ["xy_center_current.h5"]


def test_straight_waveguide_sparameter_validator_accepts_expected_response(
    tmp_path,
):
    straight_root = tmp_path / "straight_waveguide"
    straight_root.mkdir()
    _write_case(straight_root, 1, primary_transmission_db=0.1)

    messages = validate_tree(tmp_path)

    assert len(messages) == 1
    assert "mean S21=0.100 dB" in messages[0]


def test_curved_bend_comparison_and_validator_expect_large_radius_improvement(
    tmp_path,
):
    bend_root = tmp_path / "bending_waveguide"
    for polarisation in ("2d_tm", "2d_te"):
        polarisation_root = bend_root / polarisation
        polarisation_root.mkdir(parents=True)
        for case_name, s21_db in (
            ("small_bend", -5),
            ("medium_bend", -2),
            ("large_bend", -0.5),
        ):
            _write_case(
                polarisation_root,
                source_mode=1,
                primary_transmission_db=s21_db,
                case_name=case_name,
            )

    messages = validate_tree(tmp_path)
    output = plot_comparison(bend_root)

    assert len(messages) == 2
    assert all("curved bends" in message for message in messages)
    assert output.name == "bend_radius_sparameters_comparison.png"
    assert output.stat().st_size > 0


def test_curved_bend_validator_rejects_small_radius_improvement(tmp_path):
    polarisation_root = tmp_path / "bending_waveguide" / "2d_tm"
    polarisation_root.mkdir(parents=True)
    for case_name, s21_db in (
        ("small_bend", -1),
        ("medium_bend", -0.75),
        ("large_bend", -0.5),
    ):
        _write_case(
            polarisation_root,
            source_mode=1,
            primary_transmission_db=s21_db,
            case_name=case_name,
        )

    with pytest.raises(AssertionError, match="at least 2 dB"):
        validate_tree(tmp_path)
