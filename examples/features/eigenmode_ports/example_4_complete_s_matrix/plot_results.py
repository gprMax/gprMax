"""Plot magnitude and phase for each dominant-mode microstrip S parameter."""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "complete_s_matrix_study.h5"
OUTPUT = ROOT / "complete_s_matrix.png"


def magnitude_db(values, floor=-80):
    with np.errstate(divide="ignore"):
        result = 20 * np.log10(np.abs(values))
    return np.maximum(result, floor)


def main():
    with h5py.File(INPUT, "r") as result:
        frequency = result["frequency"][...] * 1e-9
        ports = result["channel_ports"][...].astype(int)
        modes = result["channel_modes"][...].astype(int)
        matrix = result["S"][...]
        valid_name = (
            "power_wave_valid_S" if "power_wave_valid_S" in result else "valid_S"
        )
        valid = result[valid_name][...].astype(bool)

    if set(zip(ports, modes)) != {(1, 1), (2, 1)}:
        raise ValueError("Expected dominant mode 1 at ports 1 and 2")
    port1 = int(np.flatnonzero((ports == 1) & (modes == 1))[0])
    port2 = int(np.flatnonzero((ports == 2) & (modes == 1))[0])
    s11 = matrix[:, port1, port1]
    s21 = matrix[:, port2, port1]
    s12 = matrix[:, port1, port2]
    s22 = matrix[:, port2, port2]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    entries = (
        (r"$S_{11}$", s11, valid[:, port1, port1]),
        (r"$S_{21}$", s21, valid[:, port2, port1]),
        (r"$S_{12}$", s12, valid[:, port1, port2]),
        (r"$S_{22}$", s22, valid[:, port2, port2]),
    )
    for magnitude_axis, (label, values, mask) in zip(axes.flat, entries):
        phase_axis = magnitude_axis.twinx()
        selected_frequency = frequency[mask]
        selected_values = values[mask]
        magnitude_line = magnitude_axis.plot(
            selected_frequency,
            magnitude_db(selected_values),
            color="tab:blue",
            label="Magnitude",
        )[0]
        phase_line = phase_axis.plot(
            selected_frequency,
            np.degrees(np.unwrap(np.angle(selected_values))),
            color="tab:orange",
            linestyle="--",
            label="Unwrapped phase",
        )[0]
        magnitude_axis.set_title(label)
        magnitude_axis.set_xlabel("Frequency (GHz)")
        magnitude_axis.set_ylabel("Magnitude (dB; floor -80 dB)", color="tab:blue")
        phase_axis.set_ylabel("Phase (degrees)", color="tab:orange")
        magnitude_axis.tick_params(axis="y", labelcolor="tab:blue")
        phase_axis.tick_params(axis="y", labelcolor="tab:orange")
        magnitude_axis.grid(True, alpha=0.3)
        magnitude_axis.legend(
            [magnitude_line, phase_line],
            [magnitude_line.get_label(), phase_line.get_label()],
            loc="best",
        )

    figure.suptitle("Complete dominant-mode S matrix of a gapped microstrip")

    figure.savefig(OUTPUT, dpi=180)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
