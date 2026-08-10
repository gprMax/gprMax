from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
GUIDE_WIDTH_M = 0.006
TE10_CUTOFF_HZ = SPEED_OF_LIGHT_M_PER_S / (2.0 * GUIDE_WIDTH_M)
PLOT_FLOOR_DB = -80.0
SOURCE_REFERENCE_M = 0.004
OUTPUT_REFERENCE_M = 0.016
SOURCE_SAMPLE_X_M = np.asarray((0.0042, 0.0062))
OUTPUT_SAMPLE_X_M = np.asarray((0.0138, 0.0158))


def read_series(path: Path, destination_port: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        np.asarray([float(row["frequency_hz"]) for row in rows]),
        np.asarray([float(row["S_magnitude_db"]) for row in rows]),
        np.asarray([bool(int(row["valid"])) for row in rows]),
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


def _directional_amplitudes(
    spectra: np.ndarray,
    sample_x: np.ndarray,
    reference_x: float,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate A exp(-j beta x) and B exp(+j beta x) at a reference plane."""

    forward = np.empty(beta.shape, dtype=np.complex128)
    backward = np.empty(beta.shape, dtype=np.complex128)
    offsets = sample_x - reference_x
    for index, propagation_constant in enumerate(beta):
        basis = np.column_stack(
            (
                np.exp(-1j * propagation_constant * offsets),
                np.exp(1j * propagation_constant * offsets),
            )
        )
        if not np.all(np.isfinite(basis)) or np.linalg.cond(basis) > 1e8:
            forward[index] = np.nan + 1j * np.nan
            backward[index] = np.nan + 1j * np.nan
            continue
        forward[index], backward[index] = np.linalg.solve(basis, spectra[index])
    return forward, backward


def generalized_amplitude_sparameters(
    output_path: Path,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover generalized TE10 amplitudes from two Ez samples at each port."""

    with h5py.File(output_path, "r") as output:
        dt = float(output.attrs["dt"])
        traces = np.column_stack(
            tuple(
                np.asarray(output[f"rxs/rx{index}/Ez"], dtype=np.float64) for index in range(1, 5)
            )
        )
    times = np.arange(traces.shape[0], dtype=np.float64) * dt
    phase = np.exp(-2j * np.pi * frequencies[:, np.newaxis] * times[np.newaxis, :])
    spectra = phase @ traces
    beta = _te10_beta(frequencies)
    source_forward, source_backward = _directional_amplitudes(
        spectra[:, :2],
        SOURCE_SAMPLE_X_M,
        SOURCE_REFERENCE_M,
        beta,
    )
    output_forward, _ = _directional_amplitudes(
        spectra[:, 2:],
        OUTPUT_SAMPLE_X_M,
        OUTPUT_REFERENCE_M,
        beta,
    )
    usable = (
        np.isfinite(source_forward)
        & np.isfinite(source_backward)
        & np.isfinite(output_forward)
        & (np.abs(source_forward) > np.finfo(np.float64).tiny)
    )
    s11 = np.full(frequencies.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    s21 = np.full_like(s11, np.nan + 1j * np.nan)
    s11[usable] = source_backward[usable] / source_forward[usable]
    s21[usable] = output_forward[usable] / source_forward[usable]
    return s11, s21


def _magnitude_db(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20.0 * np.log10(np.abs(values))


def plot_case(root: Path) -> Path:
    paths = sorted(root.resolve().glob("*_sparameters.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one S-parameter CSV below {root}, found {len(paths)}.")
    path = paths[0]
    s11_frequency, s11_db, s11_valid = read_series(path, 1)
    s21_frequency, s21_db, s21_valid = read_series(path, 2)
    np.testing.assert_array_equal(s11_frequency, s21_frequency)
    amplitude_s11, amplitude_s21 = generalized_amplitude_sparameters(
        path.with_name(path.name.removesuffix("_sparameters.csv") + ".h5"),
        s11_frequency,
    )

    frequency_ghz = s11_frequency * 1e-9
    cutoff_ghz = TE10_CUTOFF_HZ * 1e-9
    below_cutoff = s11_frequency < TE10_CUTOFF_HZ
    if not np.all(np.isfinite(amplitude_s11[below_cutoff])) or not np.all(
        np.isfinite(amplitude_s21[below_cutoff])
    ):
        raise ValueError("Below-cutoff generalized modal amplitudes are not finite.")
    amplitude_s11_db = _magnitude_db(amplitude_s11)
    amplitude_s21_db = _magnitude_db(amplitude_s21)
    attenuation = -np.imag(_te10_beta(s11_frequency[below_cutoff]))
    expected_s21_db = -20.0 / np.log(10.0) * attenuation * (OUTPUT_REFERENCE_M - SOURCE_REFERENCE_M)
    if np.max(np.abs(amplitude_s21_db[below_cutoff] - expected_s21_db)) >= 0.25:
        raise ValueError("Below-cutoff S21 does not follow analytical TE10 attenuation.")
    if np.max(amplitude_s11_db[below_cutoff]) >= -10.0:
        raise ValueError("Below-cutoff generalized S11 is unexpectedly large.")
    if not np.all(np.isfinite(amplitude_s21_db[s21_valid])):
        raise ValueError("Above-cutoff generalized S21 is not finite.")
    settled = s21_valid & (s21_frequency >= TE10_CUTOFF_HZ + 0.2e9)
    if not np.any(settled):
        raise ValueError("No settled propagating samples are available above cutoff.")
    maximum_s21_disagreement = np.max(np.abs(amplitude_s21_db[settled] - s21_db[settled]))
    if maximum_s21_disagreement >= 0.15:
        raise ValueError(
            "Generalized and power-wave S21 disagree by "
            f"{maximum_s21_disagreement:.3f} dB above cutoff."
        )
    fig, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    axis.axvspan(
        frequency_ghz[0],
        cutoff_ghz,
        color="0.9",
        label="TE10 cutoff region",
        zorder=0,
    )
    axis.axvline(
        cutoff_ghz,
        color="0.3",
        linestyle="--",
        linewidth=1.2,
        label=f"Analytical cutoff: {cutoff_ghz:.3f} GHz",
    )

    for label, magnitude_db, valid, color in (
        ("S11", s11_db, s11_valid, "tab:blue"),
        ("S21", s21_db, s21_valid, "tab:orange"),
    ):
        plotted_db = np.maximum(magnitude_db[valid], PLOT_FLOOR_DB)
        axis.plot(
            frequency_ghz[valid],
            plotted_db,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=color,
            label=label,
        )

    for label, values, color in (
        ("S11 evanescent amplitude", amplitude_s11, "tab:blue"),
        ("S21 evanescent amplitude", amplitude_s21, "tab:orange"),
    ):
        axis.plot(
            frequency_ghz[below_cutoff],
            np.maximum(_magnitude_db(values[below_cutoff]), PLOT_FLOOR_DB),
            marker="s",
            markersize=5,
            linestyle="--",
            linewidth=1.5,
            color=color,
            label=label,
        )

    axis.set_xlim(frequency_ghz[0], frequency_ghz[-1])
    axis.set_ylim(PLOT_FLOOR_DB, 2.0)
    axis.set_xlabel("Frequency (GHz)")
    axis.set_ylabel(f"Magnitude (dB; clipped at {PLOT_FLOOR_DB:g} dB)")
    axis.set_title("Straight rectangular waveguide crossing TE10 cutoff")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right", fontsize="small")
    axis.text(
        0.01,
        0.02,
        "Dashed cutoff-region traces are TE10 field-amplitude ratios, not real-power waves.",
        transform=axis.transAxes,
        fontsize="small",
        va="bottom",
    )

    for frequency, s11, s21 in zip(
        frequency_ghz[below_cutoff],
        amplitude_s11_db[below_cutoff],
        amplitude_s21_db[below_cutoff],
    ):
        print(f"{frequency:.2f} GHz generalized amplitudes: " f"S11={s11:.3f} dB, S21={s21:.3f} dB")

    output = root / "rectangular_waveguide_partial_cutoff_s11_s21.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot S11 and S21 for the partial-cutoff rectangular-waveguide case."
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    output = plot_case(parser.parse_args().root)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
