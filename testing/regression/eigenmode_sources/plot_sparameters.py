from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_sparameter_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No S-parameter rows found in {path}")

    source_ports = {int(row["source_port"]) for row in rows}
    source_modes = {int(row["source_mode"]) for row in rows}
    if len(source_ports) != 1 or len(source_modes) != 1:
        raise ValueError(f"Expected one source port and mode in {path}")
    source_port = source_ports.pop()
    source_mode = source_modes.pop()

    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["destination_port"]), int(row["destination_mode"]))].append(row)

    series = {}
    for key, selected in sorted(grouped.items()):
        selected.sort(key=lambda row: float(row["frequency_hz"]))
        valid = np.asarray([bool(int(row["valid"])) for row in selected])
        series[key] = {
            "frequency": np.asarray([float(row["frequency_hz"]) for row in selected]),
            "magnitude_db": np.where(
                valid,
                np.asarray([float(row["S_magnitude_db"]) for row in selected]),
                np.nan,
            ),
            "phase_deg": np.where(
                valid,
                np.asarray([float(row["S_phase_deg"]) for row in selected]),
                np.nan,
            ),
        }
    return source_port, source_mode, series


def plot_csv(path: Path) -> Path:
    source_port, source_mode, series = read_sparameter_csv(path)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for (destination_port, destination_mode), values in series.items():
        label = f"S{destination_port}{source_port}: " f"mode {source_mode} to mode {destination_mode}"
        frequency_ghz = values["frequency"] * 1e-9
        axes[0].plot(frequency_ghz, values["magnitude_db"], label=label)
        axes[1].plot(frequency_ghz, values["phase_deg"], label=label)
    axes[0].set_ylabel("Magnitude (dB)")
    axes[1].set_ylabel("Phase (degrees)")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[0].set_title(path.parent.name.replace("_", " "))
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small", ncols=2)
    output = path.with_name(path.stem + "_plot.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_tree(root: Path) -> list[Path]:
    outputs = []
    for path in sorted(root.resolve().rglob("*_sparameters.csv")):
        outputs.append(plot_csv(path))
    if not outputs:
        raise FileNotFoundError(f"No S-parameter CSV files found below {root}")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot every modal S-parameter CSV below a regression root.")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    for output in plot_tree(parser.parse_args().root):
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
