"""Plot active reflection and beam squint for Example 5."""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
STEM = ROOT / "phased_array"
ACTIVE_CSV = ROOT / "phased_array_active_sparameters.csv"
FAR_FIELD_GROUP = "ntff/array_surface/frequency/array_band/far_field/array_pattern"
OUTPUT = ROOT / "phased_array_active_s_and_beam_squint.png"
LIGHT_SPEED = 299_792_458.0
ELEMENT_SPACING = 18e-3
PROGRESSIVE_PHASE_DEG = -108.0


def read_active_s():
    traces = {}
    with ACTIVE_CSV.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not bool(int(row["power_wave_valid"])):
                continue
            port = int(row["port"])
            traces.setdefault(port, []).append(
                (float(row["frequency_hz"]) * 1e-9,
                 float(row["active_S_magnitude_db"]))
            )
    if len(traces) != 4:
        raise ValueError(f"Expected four driven-port active-S traces in {ACTIVE_CSV}")
    return {port: np.asarray(sorted(rows)) for port, rows in traces.items()}


def read_xy_patterns():
    with h5py.File(STEM.with_suffix(".h5"), "r") as output:
        group = output[FAR_FIELD_GROUP]
        frequency = group.parent.parent["frequencies"][...] * 1e-9
        theta = group["theta"][...]
        phi = group["phi"][...]
        directivity = group["fields/directivity_dbi"][...]
    theta_axis = np.unique(theta)
    phi_axis = np.unique(phi)
    theta_broadside = int(np.argmin(np.abs(theta_axis - 90)))
    signed_phi = (phi_axis + 180) % 360 - 180
    order = np.argsort(signed_phi)
    patterns = directivity.reshape(frequency.size, theta_axis.size, phi_axis.size)
    return frequency, signed_phi[order], patterns[:, theta_broadside, :][:, order]


def predicted_array_factor_angles(frequencies_ghz):
    frequency = np.asarray(frequencies_ghz) * 1e9
    phase_step = np.deg2rad(PROGRESSIVE_PHASE_DEG)
    argument = -phase_step * LIGHT_SPEED / (
        2 * np.pi * frequency * ELEMENT_SPACING
    )
    return np.degrees(np.arcsin(np.clip(argument, -1, 1)))


def main():
    active_s = read_active_s()
    frequencies, angles, patterns = read_xy_patterns()
    predicted_angles = predicted_array_factor_angles(frequencies)
    figure, (s_axis, pattern_axis) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )

    for port, trace in sorted(active_s.items()):
        s_axis.plot(trace[:, 0], trace[:, 1], marker="o", label=f"Port {port}")
    s_axis.set_title(r"Driven-state active reflection $\Gamma_{active,i}=b_i/a_i$")
    s_axis.set_xlabel("Frequency (GHz)")
    s_axis.set_ylabel("Active S (dB)")
    s_axis.grid(True, alpha=0.3)
    s_axis.legend()

    peak_angles = []
    forward = np.abs(angles) <= 90
    for frequency, pattern, predicted_angle in zip(
        frequencies, patterns, predicted_angles
    ):
        normalized = pattern - np.nanmax(pattern[forward])
        peak_angle = angles[forward][np.nanargmax(pattern[forward])]
        peak_angles.append(peak_angle)
        pattern_axis.plot(
            angles[forward], normalized[forward],
            label=(
                f"{frequency:g} GHz "
                f"(FDTD {peak_angle:g} deg, array factor {predicted_angle:.1f} deg)"
            ),
        )
    pattern_axis.set_title("Frequency-dependent xy-plane beam direction")
    pattern_axis.set_xlabel("Angle from +x toward +y (degrees)")
    pattern_axis.set_ylabel("Normalized directivity (dB)")
    pattern_axis.set_xlim(-90, 90)
    pattern_axis.set_ylim(-35, 1)
    pattern_axis.grid(True, alpha=0.3)
    pattern_axis.legend(fontsize="small")
    figure.suptitle(
        "Four-element phase-steered array: active S-parameters and beam squint"
    )
    figure.savefig(OUTPUT, dpi=180)
    print(f"Wrote {OUTPUT}")
    print("FDTD peak angles by frequency:", dict(zip(frequencies, peak_angles)))
    print(
        "Ideal array-factor angles by frequency:",
        dict(zip(frequencies, predicted_angles)),
    )


if __name__ == "__main__":
    main()
