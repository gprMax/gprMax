"""Plot reflection and polarization through the anisotropic-mode crossing."""

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


def crossing_frequency(frequencies, neff):
    """Linearly locate the crossing of two tracked phase-index branches."""
    difference = np.real(neff[:, 0] - neff[:, 1])
    exact = np.flatnonzero(difference == 0)
    if exact.size:
        return float(frequencies[exact[0]])
    changes = np.flatnonzero(difference[:-1] * difference[1:] < 0)
    if not changes.size:
        return float(frequencies[np.argmin(np.abs(difference))])
    index = int(changes[0])
    fraction = difference[index] / (difference[index] - difference[index + 1])
    return float(frequencies[index] + fraction * (frequencies[index + 1] - frequencies[index]))


def plot_results(stem):
    """Create one modal-reflection and polarization summary."""
    stem = Path(stem)
    reflection = read_reflection(stem)
    with h5py.File(stem.with_suffix(".h5")) as output:
        port = output["eigenmode_ports/port1"]
        launched_mode = int(port.attrs["ExcitationModes"][0])
        receiver = output["rxs/rx1"]
        ex = np.asarray(receiver["Ex"][...])
        ey = np.asarray(receiver["Ey"][...])
        time_ns = np.arange(ex.size) * float(output.attrs["dt"]) * 1e9
        crossing_ghz = crossing_frequency(
            np.asarray(port.attrs["AnchorFrequencies"]) * 1e-9,
            np.asarray(port["anchor_complex_neff"]),
        )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for mode, values in sorted(reflection.items()):
        axes[0].plot(
            values[:, 0],
            np.maximum(values[:, 1], PLOT_FLOOR_DB),
            marker="o",
            markersize=3,
            label=f"S11 into tracked mode {mode}",
        )
    axes[0].axvline(
        crossing_ghz,
        color="0.35",
        linestyle=":",
        label=f"tracked crossing ({crossing_ghz:.2f} GHz)",
    )
    axes[0].set(
        title="Reflection through the crossing band",
        xlabel="Frequency (GHz)",
        ylabel=f"Magnitude (dB; floor {PLOT_FLOOR_DB:g} dB)",
        ylim=(PLOT_FLOOR_DB, 5),
    )

    scale = max(float(np.max(np.abs(ex))), float(np.max(np.abs(ey))), np.finfo(float).tiny)
    axes[1].plot(time_ns, ex / scale, label="Ex")
    axes[1].plot(time_ns, ey / scale, label="Ey", linestyle="--")
    axes[1].set(
        title="Guide-centre polarization",
        xlabel="Time (ns)",
        ylabel="Normalized electric field",
        ylim=(-1.05, 1.05),
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.suptitle(f"Automatic mode crossing: launched tracked mode {launched_mode}")
    path = stem.with_name(stem.name + "_results.png")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--input", type=Path, help="simulation output path without .h5")
    args = parser.parse_args()
    stem = args.input or EXAMPLE_DIR / f"auto_mode_crossing_mode{args.mode}"
    print(plot_results(stem))


if __name__ == "__main__":
    main()
