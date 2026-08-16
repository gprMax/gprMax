from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_s21(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["destination_port"]) == 2 and int(row["destination_mode"]) == 1
        ]
    rows.sort(key=lambda row: float(row["frequency_hz"]))
    if not rows:
        raise ValueError(f"No port-2 mode-1 S21 rows in {path}")
    valid = np.asarray([bool(int(row["valid"])) for row in rows])
    return (
        np.asarray([float(row["frequency_hz"]) for row in rows]),
        np.where(
            valid,
            np.asarray([float(row["S_magnitude_db"]) for row in rows]),
            np.nan,
        ),
    )


def plot_comparison(root: Path) -> Path:
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for case, label in (("nonlossy", "non-lossy"), ("lossy", "lossy")):
        frequency, magnitude_db = _read_s21(root / case / f"{case}_sparameters.csv")
        axis.plot(frequency * 1e-9, magnitude_db, label=label)
    axis.set(
        xlabel="Frequency (GHz)",
        ylabel="S21 magnitude (dB)",
        title="Lossy versus non-lossy eigenmode injection",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    output = root / "lossy_vs_nonlossy_s21.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare lossy and non-lossy modal S21.")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    output = plot_comparison(parser.parse_args().root.resolve())
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
