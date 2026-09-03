from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BEND_CASES = (
    ("small_bend", "small (15 mm)"),
    ("medium_bend", "medium (30 mm)"),
    ("large_bend", "large (100 mm)"),
)


def _read_sparameter(
    path: Path,
    destination_port: int,
    destination_mode: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["destination_port"]) == destination_port and int(row["destination_mode"]) == destination_mode
        ]
    rows.sort(key=lambda row: float(row["frequency_hz"]))
    if not rows:
        raise ValueError(f"No port-{destination_port} mode-{destination_mode} rows in {path}")
    valid = np.asarray([bool(int(row["power_wave_valid"])) for row in rows])
    return (
        np.asarray([float(row["frequency_hz"]) for row in rows]),
        np.where(
            valid,
            np.asarray([float(row["S_magnitude_db"]) for row in rows]),
            np.nan,
        ),
    )


def plot_comparison(root: Path) -> Path:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )
    for row, polarisation in enumerate(("2d_tm", "2d_te")):
        for column, (destination_port, parameter) in enumerate(((1, "S11"), (2, "S21"))):
            axis = axes[row, column]
            for case, label in BEND_CASES:
                csv_path = root / polarisation / case / f"{case}_sparameters.csv"
                frequency, magnitude_db = _read_sparameter(
                    csv_path,
                    destination_port,
                )
                axis.plot(frequency * 1e-9, magnitude_db, label=label)
            axis.set(
                title=f"{polarisation[-2:].upper()} fundamental {parameter}",
                ylabel=f"{parameter} magnitude (dB)",
            )
            axis.grid(True, alpha=0.3)
            axis.legend(title="Centreline radius")
    for axis in axes[-1]:
        axis.set_xlabel("Frequency (GHz)")
    output = root / "bend_radius_sparameters_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fundamental S11 and S21 for the curved-bend radii.")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    output = plot_comparison(parser.parse_args().root.resolve())
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
