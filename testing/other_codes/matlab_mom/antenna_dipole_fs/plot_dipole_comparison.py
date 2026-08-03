"""Plot MATLAB-MoM, gprMax-KSIR, and analytical dipole comparisons."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOT_FLOOR_DB = -40.0
FFT_ZERO_PADDING = 8


def _read_csv(path):
    """Read a named CSV and always return one-dimensional columns."""

    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data)


def _complex_s11(data):
    """Reconstruct complex S11 from named real and imaginary columns."""

    return np.asarray(data["s11_real"] + 1j * data["s11_imag"])


def _complex_columns(data, real_name, imaginary_name):
    """Reconstruct complex values from two named CSV columns."""

    return np.asarray(data[real_name] + 1j * data[imaginary_name])


def _minimum_s11(frequency, magnitude_db):
    """Return a three-point parabolic estimate around the sampled minimum."""

    index = int(np.argmin(magnitude_db))
    result = {
        "frequency_hz": float(frequency[index]),
        "magnitude_db": float(magnitude_db[index]),
        "sampled_frequency_hz": float(frequency[index]),
        "sampled_magnitude_db": float(magnitude_db[index]),
    }
    if index == 0 or index == frequency.size - 1:
        return result

    spacing = np.mean(np.diff(frequency))
    local_frequency = (frequency[index - 1 : index + 2] - frequency[index]) / spacing
    coefficients = np.polyfit(local_frequency, magnitude_db[index - 1 : index + 2], 2)
    if coefficients[0] <= 0:
        return result
    vertex = -coefficients[1] / (2 * coefficients[0])
    if not -1 <= vertex <= 1:
        return result
    result["frequency_hz"] = float(frequency[index] + vertex * spacing)
    result["magnitude_db"] = float(np.polyval(coefficients, vertex))
    return result


def _threshold_bandwidth(frequency, magnitude_db, threshold_db=-10.0):
    """Linearly interpolate the first two crossings of an S11 threshold."""

    crossings = []
    offset = magnitude_db - threshold_db
    for index in range(offset.size - 1):
        if offset[index] * offset[index + 1] <= 0 and offset[index] != offset[index + 1]:
            fraction = -offset[index] / (offset[index + 1] - offset[index])
            crossings.append(
                frequency[index] + fraction * (frequency[index + 1] - frequency[index])
            )
    if len(crossings) < 2:
        return None
    return {
        "threshold_db": threshold_db,
        "lower_frequency_hz": float(crossings[0]),
        "upper_frequency_hz": float(crossings[1]),
        "bandwidth_hz": float(crossings[1] - crossings[0]),
    }


def _half_wave_pattern_db(theta_deg):
    """Return the normalised infinitesimal-radius half-wave dipole pattern."""

    theta = np.deg2rad(theta_deg)
    field = np.zeros(theta.shape)
    valid = np.abs(np.sin(theta)) > 1e-12
    field[valid] = np.abs(np.cos(0.5 * np.pi * np.cos(theta[valid])) / np.sin(theta[valid]))
    field /= np.max(field)
    return 20 * np.log10(np.maximum(field, np.finfo(float).tiny))


def _pattern_error(reference, candidate):
    """Calculate pattern errors away from analytical or numerical nulls."""

    valid = (reference > PLOT_FLOOR_DB) & (candidate > PLOT_FLOOR_DB)
    difference = candidate[valid] - reference[valid]
    return {
        "rms_difference_db": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference_db": float(np.max(np.abs(difference))),
    }


def main():
    matlab_pattern = _read_csv(RESULTS_DIR / "dipole_antenna_matlab_pattern.csv")
    gprmax_pattern = _read_csv(RESULTS_DIR / "dipole_antenna_gprmax_pattern.csv")
    if not np.array_equal(matlab_pattern["angle_deg"], gprmax_pattern["angle_deg"]):
        raise ValueError("MATLAB and gprMax pattern-angle grids do not match")

    angle = gprmax_pattern["angle_deg"]
    theta = np.abs(angle)
    analytical_xz_db = _half_wave_pattern_db(theta)
    analytical_xy_db = np.zeros(angle.shape)

    matlab_peak = max(
        np.max(matlab_pattern["xz_directivity_dbi"]),
        np.max(matlab_pattern["xy_directivity_dbi"]),
    )
    matlab_xz_db = matlab_pattern["xz_directivity_dbi"] - matlab_peak
    matlab_xy_db = matlab_pattern["xy_directivity_dbi"] - matlab_peak
    gprmax_xz_db = gprmax_pattern["xz_co_normalized_db"]
    gprmax_xy_db = gprmax_pattern["xy_co_normalized_db"]

    cuts = (
        (
            "Elevation plane (x-z)",
            analytical_xz_db,
            matlab_xz_db,
            gprmax_xz_db,
        ),
        (
            "Azimuth plane (x-y)",
            analytical_xy_db,
            matlab_xy_db,
            gprmax_xy_db,
        ),
    )
    angle_rad = np.deg2rad(angle)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), subplot_kw={"projection": "polar"})
    for axis, (title, analytical, matlab, gprmax) in zip(axes, cuts):
        axis.plot(
            angle_rad,
            np.maximum(analytical, PLOT_FLOOR_DB),
            color="#333333",
            linewidth=1.8,
            label="Analytical half-wave dipole",
        )
        axis.plot(
            angle_rad,
            np.maximum(matlab, PLOT_FLOOR_DB),
            color="#d95f02",
            linewidth=2,
            label="MATLAB MoM",
        )
        axis.plot(
            angle_rad,
            np.maximum(gprmax, PLOT_FLOOR_DB),
            color="#1b6ca8",
            linewidth=2,
            linestyle="--",
            label="gprMax FDTD + KSIR",
        )
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_rlim(PLOT_FLOOR_DB, 0)
        axis.set_rticks((-40, -30, -20, -10, 0))
        axis.set_rlabel_position(135)
        axis.set_title(title, pad=18)
        axis.grid(alpha=0.4)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, -0.20), ncol=3, frameon=False)
    fig.suptitle("151 mm centre-fed dipole at 0.95 GHz — normalised pattern")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    pattern_path = RESULTS_DIR / "dipole_pattern_comparison.png"
    fig.savefig(pattern_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    matlab_s11_data = _read_csv(RESULTS_DIR / "dipole_antenna_matlab_s11.csv")
    gprmax_s11_data = _read_csv(RESULTS_DIR / "dipole_antenna_gprmax_s11.csv")
    frequency = gprmax_s11_data["frequency_hz"]
    matlab_frequency = matlab_s11_data["frequency_hz"]
    if frequency.shape != matlab_frequency.shape or not np.allclose(
        frequency, matlab_frequency, rtol=1e-12, atol=1e-3
    ):
        raise ValueError("MATLAB and gprMax S11 frequency grids do not match")

    matlab_s11 = _complex_s11(matlab_s11_data)
    gprmax_s11 = _complex_s11(gprmax_s11_data)
    matlab_impedance = _complex_columns(
        matlab_s11_data,
        "input_impedance_real_ohm",
        "input_impedance_imag_ohm",
    )
    gprmax_impedance = _complex_columns(
        gprmax_s11_data,
        "input_impedance_real_ohm",
        "input_impedance_imag_ohm",
    )
    gprmax_impedance_s11 = _complex_columns(
        gprmax_s11_data,
        "impedance_s11_real",
        "impedance_s11_imag",
    )
    matlab_s11_db = 20 * np.log10(np.maximum(np.abs(matlab_s11), np.finfo(float).tiny))
    gprmax_s11_db = 20 * np.log10(np.maximum(np.abs(gprmax_s11), np.finfo(float).tiny))
    gprmax_impedance_s11_db = 20 * np.log10(
        np.maximum(np.abs(gprmax_impedance_s11), np.finfo(float).tiny)
    )
    matlab_phase = np.rad2deg(np.unwrap(np.angle(matlab_s11)))
    gprmax_phase = np.rad2deg(np.unwrap(np.angle(gprmax_s11)))
    gprmax_impedance_phase = np.rad2deg(np.unwrap(np.angle(gprmax_impedance_s11)))
    matlab_minimum = _minimum_s11(frequency, matlab_s11_db)
    gprmax_minimum = _minimum_s11(frequency, gprmax_s11_db)
    gprmax_impedance_minimum = _minimum_s11(frequency, gprmax_impedance_s11_db)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    axes[0].plot(
        frequency / 1e9,
        matlab_s11_db,
        color="#d95f02",
        linewidth=2.2,
        label=("MATLAB MoM " f"({matlab_minimum['frequency_hz'] / 1e9:.4f} GHz)"),
    )
    axes[0].plot(
        frequency / 1e9,
        gprmax_s11_db,
        color="#1b6ca8",
        linewidth=2.2,
        linestyle="--",
        label=("gprMax voltage waves " f"({gprmax_minimum['frequency_hz'] / 1e9:.4f} GHz)"),
    )
    axes[0].plot(
        frequency / 1e9,
        gprmax_impedance_s11_db,
        color="#2ca02c",
        linewidth=1.8,
        linestyle=":",
        label=(
            r"gprMax $V/I$ contour " f"({gprmax_impedance_minimum['frequency_hz'] / 1e9:.4f} GHz)"
        ),
    )
    axes[0].axhline(-10, color="0.45", linewidth=1, linestyle=":")
    axes[0].set_ylabel(r"$|S_{11}|$ (dB)")
    axes[0].set_ylim(-45, 0)
    axes[0].set_title(r"151 mm dipole — 73 $\Omega$ port comparison")
    axes[0].legend()
    axes[0].grid(alpha=0.35)

    axes[1].plot(frequency / 1e9, matlab_phase, color="#d95f02", linewidth=2.2)
    axes[1].plot(
        frequency / 1e9,
        gprmax_phase,
        color="#1b6ca8",
        linewidth=2.2,
        linestyle="--",
    )
    axes[1].plot(
        frequency / 1e9,
        gprmax_impedance_phase,
        color="#2ca02c",
        linewidth=1.8,
        linestyle=":",
    )
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"Unwrapped $S_{11}$ phase (degrees)")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    s11_path = RESULTS_DIR / "dipole_s11_comparison.png"
    fig.savefig(s11_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    axes[0].plot(
        frequency / 1e9,
        matlab_impedance.real,
        color="#d95f02",
        linewidth=2.2,
        label="MATLAB MoM",
    )
    axes[0].plot(
        frequency / 1e9,
        gprmax_impedance.real,
        color="#2ca02c",
        linewidth=2,
        linestyle="--",
        label=r"gprMax $V/I$ contour",
    )
    axes[0].axhline(73, color="0.45", linewidth=1, linestyle=":")
    axes[0].set_ylabel(r"Resistance, Re{$Z_\mathrm{in}$} ($\Omega$)")
    axes[0].set_title("151 mm dipole — input impedance")
    axes[0].legend()
    axes[0].grid(alpha=0.35)

    axes[1].plot(
        frequency / 1e9,
        matlab_impedance.imag,
        color="#d95f02",
        linewidth=2.2,
    )
    axes[1].plot(
        frequency / 1e9,
        gprmax_impedance.imag,
        color="#2ca02c",
        linewidth=2,
        linestyle="--",
    )
    axes[1].axhline(0, color="0.45", linewidth=1, linestyle=":")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"Reactance, Im{$Z_\mathrm{in}$} ($\Omega$)")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    impedance_path = RESULTS_DIR / "dipole_impedance_comparison.png"
    fig.savefig(impedance_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "pattern_frequency_hz": 0.95e9,
        "wire_outer_length_m": 151e-3,
        "matlab_wire_radius_m": 0.23e-3,
        "gprmax_equivalent_radius_cells": 0.23,
        "pattern": {
            "gprmax_vs_matlab_xz": _pattern_error(matlab_xz_db, gprmax_xz_db),
            "gprmax_vs_matlab_xy": _pattern_error(matlab_xy_db, gprmax_xy_db),
            "gprmax_vs_analytical_xz": _pattern_error(analytical_xz_db, gprmax_xz_db),
            "matlab_vs_analytical_xz": _pattern_error(analytical_xz_db, matlab_xz_db),
            "gprmax_xy_peak_to_peak_db": float(np.max(gprmax_xy_db) - np.min(gprmax_xy_db)),
            "matlab_xy_peak_to_peak_db": float(np.max(matlab_xy_db) - np.min(matlab_xy_db)),
        },
        "s11": {
            "reference_impedance_ohm": 73.0,
            "frequency_spacing_hz": float(np.mean(np.diff(frequency))),
            "fft_zero_padding_factor": FFT_ZERO_PADDING,
            "independent_frequency_resolution_hz": float(
                FFT_ZERO_PADDING * np.mean(np.diff(frequency))
            ),
            "gprmax_minimum": gprmax_minimum,
            "gprmax_impedance_minimum": gprmax_impedance_minimum,
            "matlab_minimum": matlab_minimum,
            "minimum_frequency_offset_hz": float(
                matlab_minimum["frequency_hz"] - gprmax_minimum["frequency_hz"]
            ),
            "gprmax_minus_10_db_bandwidth": _threshold_bandwidth(frequency, gprmax_s11_db),
            "gprmax_impedance_minus_10_db_bandwidth": _threshold_bandwidth(
                frequency, gprmax_impedance_s11_db
            ),
            "matlab_minus_10_db_bandwidth": _threshold_bandwidth(frequency, matlab_s11_db),
            "gprmax_voltage_wave_vs_impedance_s11_rms_complex": float(
                np.sqrt(np.mean(np.abs(gprmax_s11 - gprmax_impedance_s11) ** 2))
            ),
        },
        "impedance": {
            "gprmax_vs_matlab_resistance_rms_ohm": float(
                np.sqrt(np.mean((gprmax_impedance.real - matlab_impedance.real) ** 2))
            ),
            "gprmax_vs_matlab_reactance_rms_ohm": float(
                np.sqrt(np.mean((gprmax_impedance.imag - matlab_impedance.imag) ** 2))
            ),
        },
    }
    metrics_path = RESULTS_DIR / "dipole_comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved dipole pattern comparison to {pattern_path}")
    print(f"Saved dipole S11 comparison to {s11_path}")
    print(f"Saved dipole impedance comparison to {impedance_path}")
    print(f"Saved comparison metrics to {metrics_path}")


if __name__ == "__main__":
    main()
