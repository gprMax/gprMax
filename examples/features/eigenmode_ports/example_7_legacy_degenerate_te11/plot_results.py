"""Plot labelled TE11 transmission/reflection and the centre electric field."""

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
LABELS = {1: "vertical (y)", 2: "horizontal (x)"}


def plot_results(stem):
    stem = Path(stem)
    traces = defaultdict(list)
    with stem.with_name(stem.name + "_sparameters.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            if int(row["power_wave_valid"]):
                key = (int(row["destination_port"]), int(row["destination_mode"]))
                traces[key].append((float(row["frequency_hz"]) / 1e9, float(row["S_magnitude_db"])))
    if not traces:
        raise ValueError("No valid propagating S-parameter samples; inspect the port output masks.")

    with h5py.File(stem.with_suffix(".h5")) as output:
        mode = int(output["eigenmode_ports/port1"].attrs["ExcitationModes"][0])
        receiver = output["rxs/rx1"]
        ex, ey = receiver["Ex"][...], receiver["Ey"][...]
        time = np.arange(len(ex)) * float(output.attrs["dt"]) * 1e9

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for (port, destination_mode), samples in sorted(traces.items()):
        data = np.asarray(sorted(samples))
        axes[0].plot(
            data[:, 0], np.maximum(data[:, 1], -120), label=f"S{port}1: {LABELS[destination_mode]}"
        )
    axes[0].set(
        xlabel="Frequency (GHz)",
        ylabel="Magnitude (dB; floor -120 dB)",
        title="Reflection (S11) and transmission (S21)",
        ylim=(-120, 5),
    )
    axes[1].plot(time, ey, label="Ey (vertical)")
    axes[1].plot(time, ex, label="Ex (horizontal)", linestyle="--")
    axes[1].set(xlabel="Time (ns)", ylabel="Electric field (V/m)", title="Guide-centre receiver")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle(f"Circular TE11: launched mode {mode}, {LABELS[mode]}")
    path = stem.with_name(stem.name + "_results.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--input", type=Path, help="simulation output path without .h5")
    args = parser.parse_args()
    stem = args.input or EXAMPLE_DIR / f"circular_te11_mode{args.mode}"
    print(plot_results(stem))


if __name__ == "__main__":
    main()
