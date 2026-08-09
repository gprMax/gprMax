"""Compare MATLAB, voltage-gap, and magnetic-frill patch-antenna results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOT_FLOOR_DB = -40.0
REFERENCE_IMPEDANCE = 50.0


def _read_csv(name):
    """Read one named results CSV with one-dimensional structured columns."""

    return np.atleast_1d(
        np.genfromtxt(RESULTS_DIR / name, delimiter=",", names=True, encoding="utf-8")
    )


def _complex_s11(data):
    """Reconstruct complex S11 from the portable CSV columns."""

    return np.asarray(data["s11_real"] + 1j * data["s11_imag"])


def _interpolate_complex(source_frequency, values, frequency):
    """Interpolate a complex spectrum within its sampled frequency interval."""

    return np.interp(frequency, source_frequency, values.real) + 1j * np.interp(
        frequency, source_frequency, values.imag
    )


def _minimum_s11(frequency, magnitude_db):
    """Return a three-point parabolic estimate around a sampled minimum."""

    index = int(np.argmin(magnitude_db))
    result = {
        "frequency_hz": float(frequency[index]),
        "magnitude_db": float(magnitude_db[index]),
        "sampled_frequency_hz": float(frequency[index]),
        "sampled_magnitude_db": float(magnitude_db[index]),
    }
    if index == 0 or index == frequency.size - 1:
        return result
    spacing = float(np.mean(np.diff(frequency)))
    local_frequency = (frequency[index - 1 : index + 2] - frequency[index]) / spacing
    coefficients = np.polyfit(local_frequency, magnitude_db[index - 1 : index + 2], 2)
    if coefficients[0] <= 0:
        return result
    vertex = -coefficients[1] / (2 * coefficients[0])
    if -1 <= vertex <= 1:
        result["frequency_hz"] = float(frequency[index] + vertex * spacing)
        result["magnitude_db"] = float(np.polyval(coefficients, vertex))
    return result


def _threshold_bandwidth(frequency, magnitude_db, threshold_db=-10.0):
    """Interpolate the first pair of crossings of an S11 threshold."""

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


def _pattern_error(reference, candidate, angle):
    """Return upper-hemisphere error away from deep numerical nulls."""

    valid = (np.abs(angle) <= 90) & (reference > PLOT_FLOOR_DB) & (candidate > PLOT_FLOOR_DB)
    difference = candidate[valid] - reference[valid]
    return {
        "number_of_angles": int(np.count_nonzero(valid)),
        "rms_difference_db": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference_db": float(np.max(np.abs(difference))),
    }


def main():
    matlab_pattern = _read_csv("patch_antenna_matlab_pattern.csv")
    voltage_pattern = _read_csv("patch_antenna_gprmax_single_feed_pattern.csv")
    frill_pattern = _read_csv("patch_antenna_gprmax_frill_feed_pattern.csv")
    angle = np.asarray(matlab_pattern["angle_deg"])
    for name, data in (("voltage", voltage_pattern), ("frill", frill_pattern)):
        if not np.array_equal(angle, data["angle_deg"]):
            raise ValueError(f"The {name} pattern angle grid does not match MATLAB")

    matlab_peak = max(
        np.max(matlab_pattern["xz_directivity_dbi"]),
        np.max(matlab_pattern["yz_directivity_dbi"]),
    )
    patterns = {
        "matlab_xz": matlab_pattern["xz_directivity_dbi"] - matlab_peak,
        "matlab_yz": matlab_pattern["yz_directivity_dbi"] - matlab_peak,
        "voltage_xz": voltage_pattern["xz_co_normalized_db"],
        "voltage_yz": voltage_pattern["yz_co_normalized_db"],
        "frill_xz": frill_pattern["xz_co_normalized_db"],
        "frill_yz": frill_pattern["yz_co_normalized_db"],
    }

    angle_rad = np.deg2rad(angle)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), subplot_kw={"projection": "polar"})
    for axis, (title, suffix) in zip(axes, (("E-plane (x-z)", "xz"), ("H-plane (y-z)", "yz"))):
        for values, colour, style, label in (
            (patterns[f"matlab_{suffix}"], "#d95f02", "-", "MATLAB MoM delta gap"),
            (
                patterns[f"voltage_{suffix}"],
                "#1b6ca8",
                "--",
                "gprMax voltage gap",
            ),
            (
                patterns[f"frill_{suffix}"],
                "#7b3294",
                "-.",
                "gprMax magnetic frill",
            ),
        ):
            axis.plot(
                angle_rad,
                np.maximum(values, PLOT_FLOOR_DB),
                color=colour,
                linewidth=2,
                linestyle=style,
                label=label,
            )
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_rlim(PLOT_FLOOR_DB, 0)
        axis.set_rticks((-40, -30, -20, -10, 0))
        axis.set_rlabel_position(135)
        axis.set_title(title, pad=18)
        axis.grid(alpha=0.4)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, -0.22), ncol=2, frameon=False)
    fig.suptitle("Rectangular patch at 2.37 GHz — feed-model comparison")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    pattern_path = RESULTS_DIR / "patch_frill_pattern_comparison.png"
    fig.savefig(pattern_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    matlab_data = _read_csv("patch_antenna_matlab_s11.csv")
    voltage_data = _read_csv("patch_antenna_gprmax_single_feed_s11.csv")
    frill_data = _read_csv("patch_antenna_gprmax_frill_feed_s11.csv")
    matlab_frequency = np.asarray(matlab_data["frequency_hz"])
    voltage_frequency = np.asarray(voltage_data["frequency_hz"])
    frill_frequency = np.asarray(frill_data["frequency_hz"])
    common = (matlab_frequency >= max(voltage_frequency[0], frill_frequency[0])) & (
        matlab_frequency <= min(voltage_frequency[-1], frill_frequency[-1])
    )
    frequency = matlab_frequency[common]
    if frequency.size < 3:
        raise ValueError("The three port spectra have no useful common frequency range")

    matlab_s11 = _complex_s11(matlab_data)[common]
    voltage_s11 = _interpolate_complex(voltage_frequency, _complex_s11(voltage_data), frequency)
    frill_s11 = _interpolate_complex(frill_frequency, _complex_s11(frill_data), frequency)
    matlab_impedance = np.asarray(
        matlab_data["input_impedance_real_ohm"] + 1j * matlab_data["input_impedance_imag_ohm"]
    )[common]
    voltage_impedance = REFERENCE_IMPEDANCE * (1 + voltage_s11) / (1 - voltage_s11)
    frill_impedance = REFERENCE_IMPEDANCE * (1 + frill_s11) / (1 - frill_s11)

    native_curves = {}
    for name, data in (
        ("matlab", matlab_data),
        ("voltage", voltage_data),
        ("frill", frill_data),
    ):
        native_frequency = np.asarray(data["frequency_hz"])
        native_s11 = _complex_s11(data)
        native_db = 20 * np.log10(np.maximum(np.abs(native_s11), np.finfo(float).tiny))
        native_curves[name] = {
            "minimum": _minimum_s11(native_frequency, native_db),
            "bandwidth": _threshold_bandwidth(native_frequency, native_db),
        }

    s11_curves = (
        (matlab_s11, "#d95f02", "-", "MATLAB MoM", "matlab"),
        (voltage_s11, "#1b6ca8", "--", "gprMax voltage gap", "voltage"),
        (frill_s11, "#7b3294", "-.", "gprMax magnetic frill", "frill"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    for values, colour, style, label, key in s11_curves:
        magnitude_db = 20 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))
        minimum = native_curves[key]["minimum"]
        axes[0].plot(
            frequency / 1e9,
            magnitude_db,
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
    axes[0].set_title(r"Rectangular patch — 50 $\Omega$ port")
    axes[0].legend()
    axes[0].grid(alpha=0.35)
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"Unwrapped $S_{11}$ phase (degrees)")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    s11_path = RESULTS_DIR / "patch_frill_s11_comparison.png"
    fig.savefig(s11_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    for values, colour, style, label in (
        (matlab_impedance, "#d95f02", "-", "MATLAB MoM"),
        (voltage_impedance, "#1b6ca8", "--", "gprMax voltage gap"),
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
    axes[0].axhline(50, color="0.45", linewidth=1, linestyle=":")
    axes[0].set_ylabel(r"Resistance, Re{$Z_\mathrm{in}$} ($\Omega$)")
    axes[0].set_title("Rectangular patch — input impedance")
    axes[0].legend()
    axes[0].grid(alpha=0.35)
    axes[1].axhline(0, color="0.45", linewidth=1, linestyle=":")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"Reactance, Im{$Z_\mathrm{in}$} ($\Omega$)")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    impedance_path = RESULTS_DIR / "patch_frill_impedance_comparison.png"
    fig.savefig(impedance_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    resonance_band = (frequency >= 2.2e9) & (frequency <= 2.6e9)
    metrics = {
        "frequency_hz": 2.37e9,
        "reference_impedance_ohm": REFERENCE_IMPEDANCE,
        "frill_wire_radius_m": 0.23e-3,
        "matlab_square_via_side_m": 1e-3,
        "s11": native_curves,
        "pattern": {
            "voltage_vs_matlab_xz": _pattern_error(
                patterns["matlab_xz"], patterns["voltage_xz"], angle
            ),
            "voltage_vs_matlab_yz": _pattern_error(
                patterns["matlab_yz"], patterns["voltage_yz"], angle
            ),
            "frill_vs_matlab_xz": _pattern_error(
                patterns["matlab_xz"], patterns["frill_xz"], angle
            ),
            "frill_vs_matlab_yz": _pattern_error(
                patterns["matlab_yz"], patterns["frill_yz"], angle
            ),
            "frill_vs_voltage_xz": _pattern_error(
                patterns["voltage_xz"], patterns["frill_xz"], angle
            ),
            "frill_vs_voltage_yz": _pattern_error(
                patterns["voltage_yz"], patterns["frill_yz"], angle
            ),
        },
        "impedance_2p2_to_2p6_ghz": {
            "voltage_vs_matlab_resistance_rms_ohm": float(
                np.sqrt(
                    np.mean(
                        (
                            voltage_impedance.real[resonance_band]
                            - matlab_impedance.real[resonance_band]
                        )
                        ** 2
                    )
                )
            ),
            "voltage_vs_matlab_reactance_rms_ohm": float(
                np.sqrt(
                    np.mean(
                        (
                            voltage_impedance.imag[resonance_band]
                            - matlab_impedance.imag[resonance_band]
                        )
                        ** 2
                    )
                )
            ),
            "frill_vs_matlab_resistance_rms_ohm": float(
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
            "frill_vs_matlab_reactance_rms_ohm": float(
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
    metrics_path = RESULTS_DIR / "patch_frill_comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved magnetic-frill pattern comparison to {pattern_path}")
    print(f"Saved magnetic-frill S11 comparison to {s11_path}")
    print(f"Saved magnetic-frill impedance comparison to {impedance_path}")
    print(f"Saved magnetic-frill comparison metrics to {metrics_path}")


if __name__ == "__main__":
    main()
