"""Plot MATLAB-MoM and gprMax-KSIR two-element array comparisons."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOT_FLOOR_DB = -40.0


def _read_csv(path):
    """Read a named-column CSV as an array even when it has one row."""

    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def _complex_columns(data, real_name, imaginary_name):
    """Combine named real and imaginary CSV columns."""

    return np.asarray(data[real_name] + 1j * data[imaginary_name])


def _select_frequency_rows(data, requested_frequency):
    """Select exact shared-grid rows from dense gprMax FFT data."""

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


def _pattern_error(reference, candidate):
    """Return dB error only where both patterns exceed the plot floor."""

    valid = (reference > PLOT_FLOOR_DB) & (candidate > PLOT_FLOOR_DB)
    difference = candidate[valid] - reference[valid]
    return {
        "number_of_angles": int(np.count_nonzero(valid)),
        "rms_difference_db": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference_db": float(np.max(np.abs(difference))),
    }


def _minimum(frequency, magnitude_db):
    """Estimate a sampled curve's minimum with a local quadratic fit."""

    index = int(np.argmin(magnitude_db))
    result = {
        "frequency_hz": float(frequency[index]),
        "magnitude_db": float(magnitude_db[index]),
        "sampled_frequency_hz": float(frequency[index]),
        "sampled_magnitude_db": float(magnitude_db[index]),
    }
    if index == 0 or index == frequency.size - 1:
        return result
    local_frequency = frequency[index - 1 : index + 2]
    local_magnitude = magnitude_db[index - 1 : index + 2]
    centre = frequency[index]
    scale = np.mean(np.diff(local_frequency))
    coefficients = np.polyfit((local_frequency - centre) / scale, local_magnitude, 2)
    if coefficients[0] <= 0:
        return result
    vertex = -coefficients[1] / (2 * coefficients[0])
    if not -1 <= vertex <= 1:
        return result
    result["frequency_hz"] = float(centre + vertex * scale)
    result["magnitude_db"] = float(np.polyval(coefficients, vertex))
    return result


def _plot_patterns(matlab, gprmax):
    """Plot the array-axis and transverse principal-plane cuts."""

    if not np.array_equal(matlab["angle_deg"], gprmax["angle_deg"]):
        raise ValueError("MATLAB and gprMax pattern-angle grids do not match")
    angle = gprmax["angle_deg"]
    matlab_peak = max(
        np.max(matlab["xz_directivity_dbi"]),
        np.max(matlab["yz_directivity_dbi"]),
    )
    curves = (
        (
            "Array-axis plane (x-z)",
            matlab["xz_directivity_dbi"] - matlab_peak,
            gprmax["xz_co_normalized_db"],
        ),
        (
            "Transverse plane (y-z)",
            matlab["yz_directivity_dbi"] - matlab_peak,
            gprmax["yz_co_normalized_db"],
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), subplot_kw={"projection": "polar"})
    for axis, (title, matlab_db, gprmax_db) in zip(axes, curves):
        axis.plot(
            np.deg2rad(angle),
            np.maximum(matlab_db, PLOT_FLOOR_DB),
            color="#d95f02",
            linewidth=2,
            label="MATLAB MoM",
        )
        axis.plot(
            np.deg2rad(angle),
            np.maximum(gprmax_db, PLOT_FLOOR_DB),
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
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, -0.18), ncol=2, frameon=False)
    fig.suptitle("Two 75 mm dipoles, 80 mm spacing, equal phase at 1.9 GHz")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    output = RESULTS_DIR / "dipole_array_pattern_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    errors = {
        "gprmax_vs_matlab_xz": _pattern_error(curves[0][1], curves[0][2]),
        "gprmax_vs_matlab_yz": _pattern_error(curves[1][1], curves[1][2]),
    }
    return output, errors


def _plot_active_ports(matlab, gprmax_all):
    """Plot active reflection coefficient and active input impedance."""

    frequency = np.asarray(matlab["frequency_hz"])
    gprmax = _select_frequency_rows(gprmax_all, frequency)
    matlab_impedance = _complex_columns(
        matlab,
        "mean_active_impedance_real_ohm",
        "mean_active_impedance_imag_ohm",
    )
    matlab_gamma = _complex_columns(matlab, "active_gamma_real", "active_gamma_imag")

    gprmax_impedances = []
    gprmax_voltage_gamma = []
    gprmax_impedance_gamma = []
    for port in (1, 2):
        prefix = f"port_{port}_"
        gprmax_impedances.append(
            _complex_columns(
                gprmax,
                prefix + "active_impedance_real_ohm",
                prefix + "active_impedance_imag_ohm",
            )
        )
        gprmax_voltage_gamma.append(
            _complex_columns(
                gprmax,
                prefix + "voltage_gamma_real",
                prefix + "voltage_gamma_imag",
            )
        )
        gprmax_impedance_gamma.append(
            _complex_columns(
                gprmax,
                prefix + "impedance_gamma_real",
                prefix + "impedance_gamma_imag",
            )
        )
    gprmax_impedance = np.mean(gprmax_impedances, axis=0)
    gprmax_voltage_gamma_mean = np.mean(gprmax_voltage_gamma, axis=0)
    gprmax_impedance_gamma_mean = np.mean(gprmax_impedance_gamma, axis=0)

    curves = (
        (matlab_gamma, "#d95f02", "-", "MATLAB active $\\Gamma$"),
        (
            gprmax_voltage_gamma_mean,
            "#1b6ca8",
            "--",
            "gprMax voltage waves",
        ),
        (
            gprmax_impedance_gamma_mean,
            "#2ca02c",
            ":",
            r"gprMax active $V/I$",
        ),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    minima = {}
    for values, colour, style, label in curves:
        magnitude_db = 20 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))
        key = label.replace("$", "").replace("\\", "")
        minima[key] = _minimum(frequency, magnitude_db)
        axes[0].plot(
            frequency / 1e9,
            magnitude_db,
            color=colour,
            linewidth=2,
            linestyle=style,
            label=label,
        )
        axes[1].plot(
            frequency / 1e9,
            np.rad2deg(np.unwrap(np.angle(values))),
            color=colour,
            linewidth=2,
            linestyle=style,
        )
    axes[0].axhline(-10, color="0.45", linewidth=1, linestyle=":")
    axes[0].set_ylabel(r"$|\Gamma_\mathrm{active}|$ (dB)")
    axes[0].set_title(r"Equal-phase two-element array — 50 $\Omega$ ports")
    axes[0].legend()
    axes[0].grid(alpha=0.35)
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"Unwrapped active-$\Gamma$ phase (degrees)")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    gamma_output = RESULTS_DIR / "dipole_array_active_gamma_comparison.png"
    fig.savefig(gamma_output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    for values, colour, style, label in (
        (matlab_impedance, "#d95f02", "-", "MATLAB MoM"),
        (gprmax_impedance, "#2ca02c", "--", r"gprMax active $V/I$"),
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
    axes[0].set_ylabel(r"Re{$Z_\mathrm{active}$} ($\Omega$)")
    axes[0].set_title("Equal-phase array — mean active impedance per port")
    axes[0].legend()
    axes[0].grid(alpha=0.35)
    axes[1].axhline(0, color="0.45", linewidth=1, linestyle=":")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"Im{$Z_\mathrm{active}$} ($\Omega$)")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    impedance_output = RESULTS_DIR / "dipole_array_active_impedance_comparison.png"
    fig.savefig(impedance_output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_band = (frequency >= 1.65e9) & (frequency <= 2.15e9)
    metrics = {
        "frequency_samples": int(frequency.size),
        "independent_frequency_resolution_hz": float(np.median(np.diff(frequency))),
        "minima": minima,
        "gprmax_port_impedance_symmetry_rms_ohm": float(
            np.sqrt(np.mean(np.abs(gprmax_impedances[0] - gprmax_impedances[1]) ** 2))
        ),
        "matlab_port_impedance_symmetry_rms_ohm": float(
            np.sqrt(
                np.mean(
                    np.abs(
                        _complex_columns(
                            matlab,
                            "port_1_active_impedance_real_ohm",
                            "port_1_active_impedance_imag_ohm",
                        )
                        - _complex_columns(
                            matlab,
                            "port_2_active_impedance_real_ohm",
                            "port_2_active_impedance_imag_ohm",
                        )
                    )
                    ** 2
                )
            )
        ),
        "gprmax_voltage_wave_vs_impedance_gamma_rms_complex": float(
            np.sqrt(np.mean(np.abs(gprmax_voltage_gamma_mean - gprmax_impedance_gamma_mean) ** 2))
        ),
        "full_band_resistance_rms_ohm": float(
            np.sqrt(np.mean((gprmax_impedance.real - matlab_impedance.real) ** 2))
        ),
        "full_band_reactance_rms_ohm": float(
            np.sqrt(np.mean((gprmax_impedance.imag - matlab_impedance.imag) ** 2))
        ),
        "comparison_band_hz": [1.65e9, 2.15e9],
        "comparison_band_resistance_rms_ohm": float(
            np.sqrt(
                np.mean(
                    (
                        gprmax_impedance.real[comparison_band]
                        - matlab_impedance.real[comparison_band]
                    )
                    ** 2
                )
            )
        ),
        "comparison_band_reactance_rms_ohm": float(
            np.sqrt(
                np.mean(
                    (
                        gprmax_impedance.imag[comparison_band]
                        - matlab_impedance.imag[comparison_band]
                    )
                    ** 2
                )
            )
        ),
    }
    return gamma_output, impedance_output, metrics


def main():
    """Generate all comparison plots and a machine-readable metrics file."""

    matlab_pattern = _read_csv(RESULTS_DIR / "dipole_array_matlab_pattern.csv")
    gprmax_pattern = _read_csv(RESULTS_DIR / "dipole_array_gprmax_pattern.csv")
    pattern_output, pattern_metrics = _plot_patterns(matlab_pattern, gprmax_pattern)

    matlab_ports = _read_csv(RESULTS_DIR / "dipole_array_matlab_active.csv")
    gprmax_ports = _read_csv(RESULTS_DIR / "dipole_array_gprmax_active_ports.csv")
    gamma_output, impedance_output, port_metrics = _plot_active_ports(matlab_ports, gprmax_ports)
    metrics = {
        "pattern_frequency_hz": 1.9e9,
        "element_length_m": 75e-3,
        "element_spacing_m": 80e-3,
        "excitation": "equal amplitude and phase at both 50 ohm ports",
        "pattern": pattern_metrics,
        "active_ports": port_metrics,
    }
    metrics_output = RESULTS_DIR / "dipole_array_comparison_metrics.json"
    metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved array pattern comparison to {pattern_output}")
    print(f"Saved array active reflection comparison to {gamma_output}")
    print(f"Saved array active impedance comparison to {impedance_output}")
    print(f"Saved comparison metrics to {metrics_output}")


if __name__ == "__main__":
    main()
