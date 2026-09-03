from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GRID_CASES = (
    ("dx_0p20mm", 0.20),
    ("dx_0p10mm", 0.10),
    ("dx_0p05mm", 0.05),
)


def _s21(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["destination_port"]) == 2
            and int(row["destination_mode"]) == 1
            and bool(int(row["power_wave_valid"]))
        ]
    rows.sort(key=lambda row: float(row["frequency_hz"]))
    if not rows:
        raise ValueError(f"No valid fundamental S21 rows in {path}")
    return (
        np.asarray([float(row["frequency_hz"]) for row in rows]),
        np.asarray([float(row["S_magnitude_db"]) for row in rows]),
    )


def summarize_grid_spacing(root: Path) -> list[dict[str, object]]:
    summaries = []
    case_root = root / "rectangular_waveguide"
    for case_name, spacing_mm in GRID_CASES:
        paths = sorted((case_root / case_name).glob("*_sparameters.csv"))
        if len(paths) != 1:
            raise FileNotFoundError(
                f"Expected one S-parameter CSV below {case_root / case_name}, found {len(paths)}"
            )
        frequency, s21_db = _s21(paths[0])
        minimum = float(np.min(s21_db))
        maximum = float(np.max(s21_db))
        mean = float(np.mean(s21_db))
        summaries.append(
            {
                "case": case_name,
                "spacing_mm": spacing_mm,
                "frequency_hz": frequency,
                "s21_db": s21_db,
                "point_count": int(s21_db.size),
                "minimum_db": minimum,
                "maximum_db": maximum,
                "mean_db": mean,
                "max_abs_db": float(np.max(np.abs(s21_db))),
                "half_peak_to_peak_db": 0.5 * (maximum - minimum),
                "rms_about_zero_db": float(np.sqrt(np.mean(np.square(s21_db)))),
                "rms_about_mean_db": float(np.sqrt(np.mean(np.square(s21_db - mean)))),
            }
        )
    return summaries


def _write_metrics(path: Path, summaries) -> None:
    fields = (
        "case",
        "spacing_mm",
        "point_count",
        "minimum_db",
        "maximum_db",
        "mean_db",
        "max_abs_db",
        "half_peak_to_peak_db",
        "rms_about_zero_db",
        "rms_about_mean_db",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})


def plot_grid_spacing(root: Path) -> tuple[Path, Path]:
    summaries = summarize_grid_spacing(root)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for summary in summaries:
        axes[0].plot(
            np.asarray(summary["frequency_hz"]) * 1e-9,
            summary["s21_db"],
            label=f"{summary['spacing_mm']:.2f} mm",
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(
        xlabel="Frequency (GHz)",
        ylabel="Fundamental S21 magnitude (dB)",
        title="Rectangular-waveguide S21",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Cubic spacing")

    spacing = np.asarray([summary["spacing_mm"] for summary in summaries])
    for field, label, marker in (
        ("max_abs_db", "max |S21|", "o"),
        ("half_peak_to_peak_db", "half peak-to-peak", "s"),
        ("rms_about_zero_db", "RMS about 0 dB", "^"),
    ):
        axes[1].loglog(
            spacing,
            [summary[field] for summary in summaries],
            marker=marker,
            label=label,
        )
    finest = int(np.argmin(spacing))
    reference = summaries[finest]["max_abs_db"] * np.square(spacing / spacing[finest])
    axes[1].loglog(spacing, reference, "--", color="0.4", label=r"$O(\Delta^2)$ guide")
    axes[1].set(
        xlabel="Cubic grid spacing (mm)",
        ylabel="S21 fluctuation metric (dB)",
        title="Grid-refinement trend",
    )
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    plot_path = root / "grid_spacing_s21_fluctuation.png"
    metrics_path = root / "grid_spacing_metrics.csv"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    _write_metrics(metrics_path, summaries)
    return plot_path, metrics_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot rectangular-waveguide S21 fluctuation versus cubic grid spacing."
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    plot_path, metrics_path = plot_grid_spacing(parser.parse_args().root.resolve())
    print(f"Wrote {plot_path}")
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
