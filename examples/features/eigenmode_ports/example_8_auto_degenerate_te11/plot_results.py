"""Plot automatic TE11 reflection and the centre receiver fields."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLE_DIR = Path(__file__).resolve().parent
MODE_LABELS = {1: "global x", 2: "global y"}
PLOT_FLOOR_DB = -120.0


def read_reflection(stem):
    """Read valid reflected power waves, grouped by destination mode."""
    path = stem.with_name(stem.name + "_sparameters.csv")
    traces = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                int(row["source_port"]) == 1
                and int(row["destination_port"]) == 1
                and int(row["power_wave_valid"])
            ):
                mode = int(row["destination_mode"])
                traces[mode].append(
                    (float(row["frequency_hz"]) * 1e-9, float(row["S_magnitude_db"]))
                )
    if not traces:
        raise ValueError(f"No valid reflected power-wave samples found in {path}")
    return {mode: np.asarray(sorted(values)) for mode, values in traces.items()}


def plot_results(stem):
    """Create one reflection/receiver summary for an automatic TE11 run."""
    stem = Path(stem)
    reflection = read_reflection(stem)
    with h5py.File(stem.with_suffix(".h5")) as output:
        port = output["eigenmode_ports/port1"]
        launched_mode = int(port.attrs["ExcitationModes"][0])
        receiver = output["rxs/rx1"]
        ex = receiver["Ex"][...]
        ey = receiver["Ey"][...]
        time_ns = np.arange(ex.size) * float(output.attrs["dt"]) * 1e9
        tracking = output["eigenmode_ports/port1/mode_tracking"]
        detected = tracking.attrs.get("AutomaticDegenerateGroups", "")
        if isinstance(detected, bytes):
            detected = detected.decode()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for mode, values in sorted(reflection.items()):
        axes[0].plot(
            values[:, 0],
            np.maximum(values[:, 1], PLOT_FLOOR_DB),
            marker="o",
            markersize=3,
            label=f"S11 into mode {mode} ({MODE_LABELS.get(mode, 'tracked')})",
        )
    axes[0].set(
        title="Reflected tracked modes",
        xlabel="Frequency (GHz)",
        ylabel=f"Magnitude (dB; floor {PLOT_FLOOR_DB:g} dB)",
        ylim=(PLOT_FLOOR_DB, 5),
    )
    axes[1].plot(time_ns, ex, label="Ex")
    axes[1].plot(time_ns, ey, label="Ey", linestyle="--")
    axes[1].set(
        title="Guide-centre receiver",
        xlabel="Time (ns)",
        ylabel="Electric field (V/m)",
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.suptitle(
        f"Automatic circular TE11: launched mode {launched_mode} "
        f"({MODE_LABELS[launched_mode]}); detected group {detected or 'unreported'}"
    )
    path = stem.with_name(stem.name + "_results.png")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--input", type=Path, help="simulation output path without .h5")
    args = parser.parse_args()
    stem = args.input or EXAMPLE_DIR / f"auto_degenerate_te11_mode{args.mode}"
    print(plot_results(stem))


if __name__ == "__main__":
    main()
