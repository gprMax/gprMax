"""Plot generalized and physical TE10 S parameters across cutoff."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "near_cutoff_sparameters.csv"
OUTPUT = ROOT / "near_cutoff_sparameters.png"
C = 299_792_458.0
WIDTH = 0.006
CUTOFF = C / (2 * WIDTH)


def series(destination_port):
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if int(row["source_port"]) == 1
            and int(row["source_mode"]) == 1
            and int(row["destination_port"]) == destination_port
            and int(row["destination_mode"]) == 1
        ]
    rows.sort(key=lambda row: float(row["frequency_hz"]))
    return {
        "frequency": np.asarray([float(row["frequency_hz"]) for row in rows]),
        "value": np.asarray([
            float(row["S_real"]) + 1j * float(row["S_imag"]) for row in rows
        ]),
        "coefficient_valid": np.asarray([
            bool(int(row.get("coefficient_valid", row["generalized_valid"])))
            for row in rows
        ]),
        "power_wave_valid": np.asarray([
            bool(int(row.get("power_wave_valid", row["valid"]))) for row in rows
        ]),
    }


def analytical_s21(frequency):
    k0 = 2 * np.pi * frequency / C
    kc = np.pi / WIDTH
    beta = np.empty(frequency.shape, dtype=complex)
    propagating = frequency > CUTOFF
    beta[propagating] = np.sqrt(k0[propagating] ** 2 - kc ** 2)
    beta[~propagating] = -1j * np.sqrt(kc ** 2 - k0[~propagating] ** 2)
    return np.exp(-1j * beta * 0.012)


def main():
    s11 = series(1)
    s21 = series(2)
    frequency_ghz = s21["frequency"] * 1e-9
    theory = analytical_s21(s21["frequency"])
    with np.errstate(divide="ignore"):
        s11_db = 20 * np.log10(np.abs(s11["value"]))
        s21_db = 20 * np.log10(np.abs(s21["value"]))
        theory_db = 20 * np.log10(np.abs(theory))

    figure, (magnitude_axis, phase_axis) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, constrained_layout=True
    )
    for axis in (magnitude_axis, phase_axis):
        axis.axvspan(frequency_ghz[0], CUTOFF * 1e-9, color="0.9")
        axis.axvline(CUTOFF * 1e-9, color="0.3", linestyle="--")
        axis.grid(True, alpha=0.3)
    for label, values, valid in (
        ("S11 generalized coefficient", s11_db, s11["coefficient_valid"]),
        ("S21 generalized coefficient", s21_db, s21["coefficient_valid"]),
    ):
        magnitude_axis.plot(frequency_ghz[valid], values[valid], label=label)
    magnitude_axis.plot(frequency_ghz, theory_db, "k:", label="Analytical TE10 S21")
    magnitude_axis.set_ylabel("Magnitude (dB)")
    magnitude_axis.legend()
    magnitude_axis.set_title(
        "Below cutoff, a generalized coefficient exists but no real-power wave does"
    )

    valid = s21["coefficient_valid"]
    phase_axis.plot(
        frequency_ghz[valid],
        np.rad2deg(np.unwrap(np.angle(s21["value"][valid]))),
        label="gprMax S21",
    )
    phase_axis.plot(
        frequency_ghz,
        np.rad2deg(np.unwrap(np.angle(theory))),
        "k:", label="Analytical TE10",
    )
    phase_axis.set_xlabel("Frequency (GHz)")
    phase_axis.set_ylabel("Unwrapped phase (degrees)")
    phase_axis.legend()
    figure.savefig(OUTPUT, dpi=180)
    print(f"Wrote {OUTPUT}")
    print(
        "Below cutoff:", int(np.count_nonzero(s21["coefficient_valid"] &
                                               ~s21["power_wave_valid"])),
        "coefficient-valid bins are not power-wave-valid."
    )


if __name__ == "__main__":
    main()
