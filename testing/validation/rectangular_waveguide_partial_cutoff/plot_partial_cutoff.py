from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
GUIDE_WIDTH_M = 0.006
TE10_CUTOFF_HZ = SPEED_OF_LIGHT_M_PER_S / (2.0 * GUIDE_WIDTH_M)
PLOT_FLOOR_DB = -50.0
SOURCE_REFERENCE_M = 0.004
OUTPUT_REFERENCE_M = 0.016
INTEGRATED_ANALYTIC_ATOL_DB = 0.45
INTEGRATED_ANALYTIC_ATOL_DEG = 3.0


def read_series(
    path: Path,
    destination_port: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["source_port"]) == 1
            and int(row["source_mode"]) == 1
            and int(row["destination_port"]) == destination_port
            and int(row["destination_mode"]) == 1
        ]
    if not rows:
        raise ValueError(f"Missing S{destination_port}1 data in {path}")
    rows.sort(key=lambda row: float(row["frequency_hz"]))
    if "generalized_valid" in rows[0]:
        generalized_valid = np.asarray(
            [bool(int(row["generalized_valid"])) for row in rows]
        )
        power_valid = np.asarray([bool(int(row["valid"])) for row in rows])
        explicit_power_valid = np.asarray(
            [bool(int(row["power_wave_valid"])) for row in rows]
        )
        np.testing.assert_array_equal(power_valid, explicit_power_valid)
    else:
        # Compatibility with outputs produced by the unmerged cutoff branch,
        # where ``valid`` still carried the generalized mask.
        generalized_valid = np.asarray(
            [bool(int(row["valid"])) for row in rows]
        )
        power_valid = np.asarray(
            [bool(int(row.get("power_wave_valid", row["valid"]))) for row in rows]
        )
    return (
        np.asarray([float(row["frequency_hz"]) for row in rows]),
        np.asarray(
            [float(row["S_real"]) + 1j * float(row["S_imag"]) for row in rows]
        ),
        np.asarray([float(row["S_magnitude_db"]) for row in rows]),
        generalized_valid,
        power_valid,
    )


def _te10_beta(frequencies: np.ndarray) -> np.ndarray:
    """Return the passive TE10 beta for exp(+jwt-j beta x)."""

    frequencies = np.asarray(frequencies, dtype=np.float64)
    ratio_squared = np.square(TE10_CUTOFF_HZ / frequencies)
    neff = np.empty(frequencies.shape, dtype=np.complex128)
    propagating = frequencies > TE10_CUTOFF_HZ
    neff[propagating] = np.sqrt(1.0 - ratio_squared[propagating])
    neff[~propagating] = -1j * np.sqrt(ratio_squared[~propagating] - 1.0)
    return 2.0 * np.pi * frequencies * neff / SPEED_OF_LIGHT_M_PER_S


def _magnitude_db(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20.0 * np.log10(np.abs(values))


def plot_case(root: Path) -> Path:
    paths = sorted(root.resolve().glob("*_sparameters.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one S-parameter CSV below {root}, found {len(paths)}.")
    path = paths[0]
    s11_frequency, _s11, s11_db, s11_generalized_valid, s11_power_valid = read_series(
        path, 1
    )
    s21_frequency, s21, s21_db, s21_generalized_valid, s21_power_valid = read_series(
        path, 2
    )
    np.testing.assert_array_equal(s11_frequency, s21_frequency)

    frequency_ghz = s11_frequency * 1e-9
    cutoff_ghz = TE10_CUTOFF_HZ * 1e-9
    below_cutoff = s11_frequency < TE10_CUTOFF_HZ
    reference_span = OUTPUT_REFERENCE_M - SOURCE_REFERENCE_M
    beta = _te10_beta(s11_frequency)
    theoretical_s21 = np.exp(-1j * beta * reference_span)
    theoretical_s21_db = _magnitude_db(theoretical_s21)
    theoretical_s21_phase_deg = -np.rad2deg(np.real(beta) * reference_span)
    integrated_s21_phase_deg = np.rad2deg(np.unwrap(np.angle(s21)))
    expected_power_valid = ~below_cutoff
    if not np.all(s11_generalized_valid) or not np.all(s21_generalized_valid):
        raise ValueError("Every partial-cutoff generalized S coefficient must be valid.")
    np.testing.assert_array_equal(s11_power_valid, expected_power_valid)
    np.testing.assert_array_equal(s21_power_valid, expected_power_valid)
    if not np.all(np.isfinite(s11_db)) or not np.all(np.isfinite(s21_db)):
        raise ValueError("Every partial-cutoff generalized S coefficient must be finite.")
    magnitude_error_db = np.max(np.abs(s21_db - theoretical_s21_db))
    if magnitude_error_db >= INTEGRATED_ANALYTIC_ATOL_DB:
        raise ValueError(
            "gprMax S21 magnitude does not follow analytical TE10 transmission: "
            f"maximum error {magnitude_error_db:.3f} dB."
        )
    phase_error_deg = np.max(
        np.abs(np.rad2deg(np.angle(s21 / theoretical_s21)))
    )
    if phase_error_deg >= INTEGRATED_ANALYTIC_ATOL_DEG:
        raise ValueError(
            "gprMax S21 phase does not follow analytical TE10 transmission: "
            f"maximum circular error {phase_error_deg:.3f} degrees."
        )
    settled = s21_power_valid & (s21_frequency >= TE10_CUTOFF_HZ + 0.2e9)
    if not np.any(settled):
        raise ValueError("No settled propagating samples are available above cutoff.")
    if np.max(s11_db[settled]) >= -20.0:
        raise ValueError(
            "Settled propagating S11 must remain below -20 dB; "
            f"maximum is {np.max(s11_db[settled]):.3f} dB."
        )
    fig, (magnitude_axis, phase_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8.2),
        sharex=True,
        constrained_layout=True,
    )
    for axis in (magnitude_axis, phase_axis):
        axis.axvspan(
            frequency_ghz[0],
            cutoff_ghz,
            color="0.9",
            label="TE10 evanescent region" if axis is magnitude_axis else None,
            zorder=0,
        )
        axis.axvline(
            cutoff_ghz,
            color="0.3",
            linestyle="--",
            linewidth=1.2,
            label=(
                f"Analytical cutoff: {cutoff_ghz:.3f} GHz"
                if axis is magnitude_axis
                else None
            ),
        )

    for label, magnitude_db, valid, color in (
        ("gprMax S11", s11_db, s11_generalized_valid, "tab:blue"),
        ("gprMax S21", s21_db, s21_generalized_valid, "tab:orange"),
    ):
        plotted_db = np.maximum(magnitude_db[valid], PLOT_FLOOR_DB)
        magnitude_axis.plot(
            frequency_ghz[valid],
            plotted_db,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=color,
            label=label,
        )

    magnitude_axis.plot(
        frequency_ghz,
        np.maximum(theoretical_s21_db, PLOT_FLOOR_DB),
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=r"Analytical TE10 $S_{21}=e^{-j\beta L}$",
    )

    phase_axis.plot(
        frequency_ghz[s21_generalized_valid],
        integrated_s21_phase_deg[s21_generalized_valid],
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        color="tab:orange",
        label="gprMax S21",
    )
    phase_axis.plot(
        frequency_ghz,
        theoretical_s21_phase_deg,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=r"Analytical TE10 $\angle S_{21}=-\mathrm{Re}(\beta)L$",
    )

    magnitude_axis.set_ylim(PLOT_FLOOR_DB, 2.0)
    magnitude_axis.set_ylabel(f"Magnitude (dB; clipped at {PLOT_FLOOR_DB:g} dB)")
    magnitude_axis.set_title("TE10 transmission across cutoff: gprMax versus theory")
    magnitude_axis.grid(True, alpha=0.3)
    magnitude_axis.legend(loc="lower right", fontsize="small", ncol=2)
    magnitude_axis.text(
        0.01,
        0.02,
        "Below cutoff: branch-local E/H-balanced amplitudes\n"
        "(generalized_valid = 1, valid = power_wave_valid = 0).",
        transform=magnitude_axis.transAxes,
        fontsize="small",
        va="bottom",
    )
    phase_axis.set_xlim(frequency_ghz[0], frequency_ghz[-1])
    phase_axis.set_xlabel("Frequency (GHz)")
    phase_axis.set_ylabel("Unwrapped S21 phase (degrees)")
    phase_axis.grid(True, alpha=0.3)
    phase_axis.legend(loc="upper right", fontsize="small")

    for frequency, s11_value, s21_value, theory_value in zip(
        frequency_ghz[below_cutoff],
        s11_db[below_cutoff],
        s21_db[below_cutoff],
        theoretical_s21_db[below_cutoff],
    ):
        print(
            f"{frequency:.2f} GHz generalized amplitudes: "
            f"S11={s11_value:.3f} dB, gprMax S21={s21_value:.3f} dB, "
            f"analytical S21={theory_value:.3f} dB"
        )
    print(
        f"Maximum S21 theory error: {magnitude_error_db:.3f} dB magnitude, "
        f"{phase_error_deg:.3f} degrees phase"
    )

    output = root / "rectangular_waveguide_partial_cutoff_s11_s21.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot S11/S21 magnitude and S21 phase for the partial-cutoff "
            "rectangular-waveguide case."
        )
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    output = plot_case(parser.parse_args().root)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
