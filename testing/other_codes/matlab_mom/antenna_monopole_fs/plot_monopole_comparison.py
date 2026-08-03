"""Plot MATLAB-MoM and gprMax-KSIR finite-ground monopole comparisons."""

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

    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def _complex_columns(data, real_name, imaginary_name):
    """Reconstruct complex values from two named CSV columns."""

    return np.asarray(data[real_name] + 1j * data[imaginary_name])


def _interpolate_complex_columns(data, real_name, imaginary_name, frequency):
    """Interpolate a complex spectrum onto a nearby independent FFT grid."""

    source_frequency = np.asarray(data["frequency_hz"])
    requested = np.asarray(frequency)
    if requested[0] < source_frequency[0] or requested[-1] > source_frequency[-1]:
        raise ValueError("Requested frequencies extend beyond the source spectrum")
    values = _complex_columns(data, real_name, imaginary_name)
    return np.interp(requested, source_frequency, values.real) + 1j * np.interp(
        requested, source_frequency, values.imag
    )


def _select_frequency_rows(data, requested_frequency):
    """Select exact gprMax rows on MATLAB's independent-frequency grid."""

    available = np.asarray(data["frequency_hz"])
    right = np.clip(np.searchsorted(available, requested_frequency), 0, available.size - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(available[left] - requested_frequency) < np.abs(
        available[right] - requested_frequency
    )
    indices = np.where(use_left, left, right)
    if not np.allclose(available[indices], requested_frequency, rtol=1e-12, atol=1e-3):
        raise ValueError("MATLAB frequencies are not a subset of the gprMax grid")
    return data[indices]


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
        crosses_threshold = offset[index] * offset[index + 1] <= 0
        if crosses_threshold and offset[index] != offset[index + 1]:
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


def _pattern_error(reference, candidate):
    """Calculate pattern errors away from numerical nulls."""

    valid = (reference > PLOT_FLOOR_DB) & (candidate > PLOT_FLOOR_DB)
    difference = candidate[valid] - reference[valid]
    return {
        "number_of_angles": int(np.count_nonzero(valid)),
        "rms_difference_db": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference_db": float(np.max(np.abs(difference))),
    }


def main():
    matlab_pattern = _read_csv(RESULTS_DIR / "monopole_antenna_matlab_pattern.csv")
    gprmax_pattern = _read_csv(RESULTS_DIR / "monopole_antenna_gprmax_pattern.csv")
    frill_pattern = _read_csv(RESULTS_DIR / "monopole_antenna_gprmax_frill_pattern.csv")
    if not np.array_equal(
        matlab_pattern["angle_deg"], gprmax_pattern["angle_deg"]
    ) or not np.array_equal(matlab_pattern["angle_deg"], frill_pattern["angle_deg"]):
        raise ValueError("MATLAB and gprMax pattern-angle grids do not match")

    angle = gprmax_pattern["angle_deg"]
    matlab_peak = max(
        np.max(matlab_pattern["xz_directivity_dbi"]),
        np.max(matlab_pattern["xy_directivity_dbi"]),
    )
    matlab_xz_db = matlab_pattern["xz_directivity_dbi"] - matlab_peak
    matlab_xy_db = matlab_pattern["xy_directivity_dbi"] - matlab_peak
    gprmax_xz_db = gprmax_pattern["xz_co_normalized_db"]
    gprmax_xy_db = gprmax_pattern["xy_co_normalized_db"]
    frill_xz_db = frill_pattern["xz_co_normalized_db"]
    frill_xy_db = frill_pattern["xy_co_normalized_db"]

    cuts = (
        ("Elevation plane (x-z)", matlab_xz_db, gprmax_xz_db, frill_xz_db),
        ("Ground-plane cut (x-y)", matlab_xy_db, gprmax_xy_db, frill_xy_db),
    )
    angle_rad = np.deg2rad(angle)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), subplot_kw={"projection": "polar"})
    for axis, (title, matlab, gprmax, frill) in zip(axes, cuts):
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
            label="gprMax voltage gap + KSIR",
        )
        axis.plot(
            angle_rad,
            np.maximum(frill, PLOT_FLOOR_DB),
            color="#2ca02c",
            linewidth=1.8,
            linestyle=":",
            label="gprMax magnetic frill + KSIR",
        )
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_rlim(PLOT_FLOOR_DB, 0)
        axis.set_rticks((-40, -30, -20, -10, 0))
        axis.set_rlabel_position(135)
        axis.set_title(title, pad=18)
        axis.grid(alpha=0.4)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, -0.22), ncol=2, frameon=False)
    fig.suptitle("79 mm monopole over 160 mm square PEC plate at 0.91 GHz")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    pattern_path = RESULTS_DIR / "monopole_pattern_comparison.png"
    fig.savefig(pattern_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    matlab_data = _read_csv(RESULTS_DIR / "monopole_antenna_matlab_s11.csv")
    gprmax_all = _read_csv(RESULTS_DIR / "monopole_antenna_gprmax_s11.csv")
    frill_all = _read_csv(RESULTS_DIR / "monopole_antenna_gprmax_frill_s11.csv")
    frequency = np.asarray(matlab_data["frequency_hz"])
    gprmax_data = _select_frequency_rows(gprmax_all, frequency)

    matlab_s11 = _complex_columns(matlab_data, "s11_real", "s11_imag")
    gprmax_voltage_s11 = _complex_columns(gprmax_data, "s11_real", "s11_imag")
    gprmax_impedance_s11 = _complex_columns(gprmax_data, "impedance_s11_real", "impedance_s11_imag")
    matlab_impedance = _complex_columns(
        matlab_data, "input_impedance_real_ohm", "input_impedance_imag_ohm"
    )
    gprmax_impedance = _complex_columns(
        gprmax_data, "input_impedance_real_ohm", "input_impedance_imag_ohm"
    )
    frill_s11 = _interpolate_complex_columns(frill_all, "s11_real", "s11_imag", frequency)
    frill_impedance = _interpolate_complex_columns(
        frill_all,
        "input_impedance_real_ohm",
        "input_impedance_imag_ohm",
        frequency,
    )

    matlab_s11_db = 20 * np.log10(np.maximum(np.abs(matlab_s11), np.finfo(float).tiny))
    voltage_s11_db = 20 * np.log10(np.maximum(np.abs(gprmax_voltage_s11), np.finfo(float).tiny))
    impedance_s11_db = 20 * np.log10(np.maximum(np.abs(gprmax_impedance_s11), np.finfo(float).tiny))
    frill_s11_db = 20 * np.log10(np.maximum(np.abs(frill_s11), np.finfo(float).tiny))
    minima = {
        "matlab": _minimum_s11(frequency, matlab_s11_db),
        "gprmax_voltage": _minimum_s11(frequency, voltage_s11_db),
        "gprmax_impedance": _minimum_s11(frequency, impedance_s11_db),
        "gprmax_frill": _minimum_s11(frequency, frill_s11_db),
    }
    resonance_band = (frequency >= 0.85e9) & (frequency <= 0.95e9)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    curves = (
        (matlab_s11, matlab_s11_db, "#d95f02", "-", "MATLAB MoM", minima["matlab"]),
        (
            gprmax_voltage_s11,
            voltage_s11_db,
            "#1b6ca8",
            "--",
            "gprMax voltage waves",
            minima["gprmax_voltage"],
        ),
        (
            gprmax_impedance_s11,
            impedance_s11_db,
            "#2ca02c",
            ":",
            r"gprMax $V/I$ contour",
            minima["gprmax_impedance"],
        ),
        (
            frill_s11,
            frill_s11_db,
            "#7b3294",
            "-.",
            "gprMax magnetic frill",
            minima["gprmax_frill"],
        ),
    )
    for values, values_db, colour, style, label, minimum in curves:
        axes[0].plot(
            frequency / 1e9,
            values_db,
            color=colour,
            linewidth=2,
            linestyle=style,
            label=f"{label} ({minimum['frequency_hz'] / 1e9:.4f} GHz)",
        )
        axes[1].plot(
            frequency / 1e9,
            np.rad2deg(np.unwrap(np.angle(values))),
            color=colour,
            linewidth=2,
            linestyle=style,
        )
    axes[0].axhline(-10, color="0.45", linewidth=1, linestyle=":")
    axes[0].set_ylabel(r"$|S_{11}|$ (dB)")
    axes[0].set_title(r"Finite-ground monopole — 36.5 $\Omega$ port")
    axes[0].legend()
    axes[0].grid(alpha=0.35)
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"Unwrapped $S_{11}$ phase (degrees)")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    s11_path = RESULTS_DIR / "monopole_s11_comparison.png"
    fig.savefig(s11_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    for values, colour, style, label in (
        (matlab_impedance, "#d95f02", "-", "MATLAB MoM"),
        (gprmax_impedance, "#2ca02c", "--", r"gprMax $V/I$ contour"),
        (frill_impedance, "#7b3294", "-.", "gprMax magnetic frill"),
    ):
        axes[0].plot(
            frequency / 1e9,
            values.real,
            color=colour,
            linewidth=2,
            linestyle=style,
            label=label,
        )
        axes[1].plot(
            frequency / 1e9,
            values.imag,
            color=colour,
            linewidth=2,
            linestyle=style,
        )
    axes[0].axhline(36.5, color="0.45", linewidth=1, linestyle=":")
    axes[0].set_ylabel(r"Resistance, Re{$Z_\mathrm{in}$} ($\Omega$)")
    axes[0].set_title("Finite-ground monopole — input impedance")
    axes[0].legend()
    axes[0].grid(alpha=0.35)
    axes[1].axhline(0, color="0.45", linewidth=1, linestyle=":")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"Reactance, Im{$Z_\mathrm{in}$} ($\Omega$)")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    impedance_path = RESULTS_DIR / "monopole_impedance_comparison.png"
    fig.savefig(impedance_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "pattern_frequency_hz": 0.91e9,
        "monopole_height_m": 79e-3,
        "ground_plane_m": [160e-3, 160e-3],
        "matlab_wire_radius_m": 0.23e-3,
        "pattern": {
            "gprmax_vs_matlab_xz": _pattern_error(matlab_xz_db, gprmax_xz_db),
            "gprmax_vs_matlab_xy": _pattern_error(matlab_xy_db, gprmax_xy_db),
            "frill_vs_matlab_xz": _pattern_error(matlab_xz_db, frill_xz_db),
            "frill_vs_matlab_xy": _pattern_error(matlab_xy_db, frill_xy_db),
            "frill_vs_voltage_gap_xz": _pattern_error(gprmax_xz_db, frill_xz_db),
            "frill_vs_voltage_gap_xy": _pattern_error(gprmax_xy_db, frill_xy_db),
        },
        "s11": {
            "reference_impedance_ohm": 36.5,
            "frequency_spacing_hz": float(np.median(np.diff(frequency))),
            "fft_zero_padding_factor": FFT_ZERO_PADDING,
            "independent_frequency_resolution_hz": float(np.median(np.diff(frequency))),
            "minima": minima,
            "gprmax_voltage_minus_10_db_bandwidth": _threshold_bandwidth(frequency, voltage_s11_db),
            "gprmax_impedance_minus_10_db_bandwidth": _threshold_bandwidth(
                frequency, impedance_s11_db
            ),
            "gprmax_frill_minus_10_db_bandwidth": _threshold_bandwidth(frequency, frill_s11_db),
            "matlab_minus_10_db_bandwidth": _threshold_bandwidth(frequency, matlab_s11_db),
            "gprmax_voltage_wave_vs_impedance_s11_rms_complex": float(
                np.sqrt(np.mean(np.abs(gprmax_voltage_s11 - gprmax_impedance_s11) ** 2))
            ),
        },
        "impedance": {
            "gprmax_vs_matlab_resistance_rms_ohm": float(
                np.sqrt(np.mean((gprmax_impedance.real - matlab_impedance.real) ** 2))
            ),
            "gprmax_vs_matlab_reactance_rms_ohm": float(
                np.sqrt(np.mean((gprmax_impedance.imag - matlab_impedance.imag) ** 2))
            ),
            "frill_vs_matlab_resistance_rms_ohm": float(
                np.sqrt(np.mean((frill_impedance.real - matlab_impedance.real) ** 2))
            ),
            "frill_vs_matlab_reactance_rms_ohm": float(
                np.sqrt(np.mean((frill_impedance.imag - matlab_impedance.imag) ** 2))
            ),
            "resonance_band_hz": [0.85e9, 0.95e9],
            "resonance_band_resistance_rms_ohm": float(
                np.sqrt(
                    np.mean(
                        (
                            gprmax_impedance.real[resonance_band]
                            - matlab_impedance.real[resonance_band]
                        )
                        ** 2
                    )
                )
            ),
            "resonance_band_reactance_rms_ohm": float(
                np.sqrt(
                    np.mean(
                        (
                            gprmax_impedance.imag[resonance_band]
                            - matlab_impedance.imag[resonance_band]
                        )
                        ** 2
                    )
                )
            ),
            "frill_resonance_band_resistance_rms_ohm": float(
                np.sqrt(
                    np.mean(
                        (
                            frill_impedance.real[resonance_band]
                            - matlab_impedance.real[resonance_band]
                        )
                        ** 2
                    )
                )
            ),
            "frill_resonance_band_reactance_rms_ohm": float(
                np.sqrt(
                    np.mean(
                        (
                            frill_impedance.imag[resonance_band]
                            - matlab_impedance.imag[resonance_band]
                        )
                        ** 2
                    )
                )
            ),
        },
    }
    metrics_path = RESULTS_DIR / "monopole_comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved monopole pattern comparison to {pattern_path}")
    print(f"Saved monopole S11 comparison to {s11_path}")
    print(f"Saved monopole impedance comparison to {impedance_path}")
    print(f"Saved comparison metrics to {metrics_path}")


if __name__ == "__main__":
    main()
