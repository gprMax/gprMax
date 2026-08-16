import csv

import pytest

from testing.regression.eigenmode_sources.grid_spacing.plot_grid_spacing import (
    GRID_CASES,
    plot_grid_spacing,
    summarize_grid_spacing,
)


def _write_s21(path, values):
    fieldnames = (
        "frequency_hz",
        "source_port",
        "source_mode",
        "destination_port",
        "destination_mode",
        "S_magnitude_db",
        "valid",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frequency, value in zip((45e9, 55e9, 65e9), values):
            writer.writerow(
                {
                    "frequency_hz": frequency,
                    "source_port": 1,
                    "source_mode": 1,
                    "destination_port": 2,
                    "destination_mode": 1,
                    "S_magnitude_db": value,
                    "valid": 1,
                }
            )


def test_grid_spacing_summary_and_plot_report_s21_fluctuation(tmp_path):
    root = tmp_path / "grid_spacing"
    case_root = root / "rectangular_waveguide"
    amplitudes = (0.4, 0.1, 0.025)
    for (case_name, unused_spacing), amplitude in zip(GRID_CASES, amplitudes):
        directory = case_root / case_name
        directory.mkdir(parents=True)
        _write_s21(directory / f"{case_name}_sparameters.csv", (-amplitude, 0.0, amplitude))

    summaries = summarize_grid_spacing(root)
    plot_path, metrics_path = plot_grid_spacing(root)

    assert [summary["max_abs_db"] for summary in summaries] == pytest.approx(amplitudes)
    assert [summary["half_peak_to_peak_db"] for summary in summaries] == pytest.approx(amplitudes)
    assert plot_path.name == "grid_spacing_s21_fluctuation.png"
    assert plot_path.stat().st_size > 0
    assert metrics_path.name == "grid_spacing_metrics.csv"
    assert metrics_path.stat().st_size > 0
