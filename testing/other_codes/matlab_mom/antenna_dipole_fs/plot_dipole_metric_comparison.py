"""Compare cylindrical-dipole metrics from gprMax and MATLAB Antenna Toolbox."""

import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
GPRMAX_HDF5 = RESULTS_DIR / "dipole_antenna_gprmax.h5"
MATLAB_CSV = RESULTS_DIR / "dipole_antenna_matlab_3d_metrics.csv"
MATLAB_JSON = RESULTS_DIR / "dipole_antenna_matlab_3d_metrics.json"
OUTPUT_PNG = RESULTS_DIR / "dipole_antenna_metric_comparison.png"
OUTPUT_JSON = RESULTS_DIR / "dipole_antenna_metric_comparison.json"
GROUP = "ntff/dipole_surface/frequency/dipole_spectrum/far_field/metrics_3d"


def read_gprmax():
    with h5py.File(GPRMAX_HDF5, "r") as output:
        group = output[GROUP]
        result = {
            "theta": np.asarray(group["theta"], dtype=np.float64),
            "phi": np.asarray(group["phi"], dtype=np.float64),
            "radiation_intensity": np.asarray(
                group["fields/radiation_intensity"][0], dtype=np.float64
            ),
            "directivity_dbi": np.asarray(group["fields/directivity_dbi"][0], dtype=np.float64),
            "gain_dbi": np.asarray(group["fields/gain_dbi"][0], dtype=np.float64),
            "realized_gain_dbi": np.asarray(group["fields/realized_gain_dbi"][0], dtype=np.float64),
            "radiation_efficiency": float(group["fields/radiation_efficiency"][0]),
            "total_efficiency": float(group["fields/total_efficiency"][0]),
            "maximum_directivity": float(group["maximum_directivity"][0]),
            "radiated_power": float(group["radiated_power"][0]),
            "accepted_power": float(group["port_power/accepted_power"][0]),
            "incident_power": float(group["port_power/incident_power"][0]),
        }
    result["mismatch_efficiency"] = result["accepted_power"] / result["incident_power"]
    return result


def read_matlab():
    table = np.genfromtxt(MATLAB_CSV, delimiter=",", names=True)
    result = {
        "theta": np.asarray(table["theta_deg"], dtype=np.float64),
        "phi": np.asarray(table["phi_deg"], dtype=np.float64),
        "radiation_intensity": np.asarray(table["radiation_intensity_w_per_sr"], dtype=np.float64),
        "directivity_dbi": np.asarray(table["directivity_dbi"], dtype=np.float64),
        "gain_dbi": np.asarray(table["gain_dbi"], dtype=np.float64),
        "realized_gain_dbi": np.asarray(table["realized_gain_dbi"], dtype=np.float64),
    }
    result.update(json.loads(MATLAB_JSON.read_text(encoding="utf-8")))
    return result


def sphere_average(pattern, theta, phi):
    theta_axis = np.unique(theta)
    phi_axis = np.unique(phi)
    values = pattern.reshape(theta_axis.size, phi_axis.size)
    theta_rad = np.deg2rad(theta_axis)
    theta_integral = np.trapezoid(
        values * np.sin(theta_rad)[:, np.newaxis],
        theta_rad,
        axis=0,
    )
    return float((2 * np.pi / phi_axis.size) * np.sum(theta_integral) / (4 * np.pi))


def ideal_half_wave_directivity(theta, phi):
    theta_rad = np.deg2rad(theta)
    field = np.zeros(theta.shape, dtype=np.float64)
    interior = np.abs(np.sin(theta_rad)) > 1e-12
    field[interior] = np.cos(0.5 * np.pi * np.cos(theta_rad[interior])) / np.sin(
        theta_rad[interior]
    )
    intensity = field**2
    return intensity / sphere_average(intensity, theta, phi)


def signed_elevation(values, theta, phi):
    lookup = {
        (float(theta_value), float(phi_value)): value
        for theta_value, phi_value, value in zip(theta, phi, values)
    }
    angle = np.arange(-180.0, 182.0, 2.0)
    cut = np.asarray([lookup[(abs(value), 180.0 if value < 0 else 0.0)] for value in angle])
    return angle, cut


def azimuth_cut(values, theta, phi):
    selected = np.isclose(theta, 90.0)
    order = np.argsort(phi[selected])
    return phi[selected][order], values[selected][order]


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def compare(gprmax, matlab):
    if not np.allclose(gprmax["theta"], matlab["theta"], rtol=0, atol=1e-12):
        raise ValueError("The gprMax and MATLAB theta grids differ")
    if not np.allclose(gprmax["phi"], matlab["phi"], rtol=0, atol=1e-12):
        raise ValueError("The gprMax and MATLAB phi grids differ")

    theta, phi = gprmax["theta"], gprmax["phi"]
    ideal_directivity = ideal_half_wave_directivity(theta, phi)
    ideal_dbi = 10 * np.log10(np.maximum(ideal_directivity, np.finfo(float).tiny))
    floor = (
        max(
            np.max(gprmax["directivity_dbi"]),
            np.max(matlab["directivity_dbi"]),
        )
        - 30
    )
    valid = (gprmax["directivity_dbi"] >= floor) & (matlab["directivity_dbi"] >= floor)
    ideal_valid = valid & (ideal_dbi >= floor)
    gprmax_peak = float(10 * np.log10(gprmax["maximum_directivity"]))
    result = {
        "frequency_hz": float(matlab["frequency_hz"]),
        "angular_samples": int(theta.size),
        "gprmax": {
            "maximum_directivity_dbi": gprmax_peak,
            "maximum_gain_dbi": float(np.max(gprmax["gain_dbi"])),
            "maximum_realized_gain_dbi": float(np.max(gprmax["realized_gain_dbi"])),
            "radiation_efficiency": gprmax["radiation_efficiency"],
            "mismatch_efficiency": gprmax["mismatch_efficiency"],
            "total_efficiency": gprmax["total_efficiency"],
            "directivity_sphere_average": sphere_average(
                10 ** (gprmax["directivity_dbi"] / 10), theta, phi
            ),
            "gain_identity_max_error_db": float(
                np.max(
                    np.abs(
                        gprmax["gain_dbi"]
                        - gprmax["directivity_dbi"]
                        - 10 * np.log10(gprmax["radiation_efficiency"])
                    )
                )
            ),
            "realized_gain_identity_max_error_db": float(
                np.max(
                    np.abs(
                        gprmax["realized_gain_dbi"]
                        - gprmax["gain_dbi"]
                        - 10 * np.log10(gprmax["mismatch_efficiency"])
                    )
                )
            ),
            "radiation_intensity_identity_max_error_db": float(
                np.max(
                    np.abs(
                        10
                        * np.log10(
                            4 * np.pi * gprmax["radiation_intensity"] / gprmax["radiated_power"]
                        )
                        - gprmax["directivity_dbi"]
                    )
                )
            ),
        },
        "matlab": {
            "maximum_directivity_dbi": float(matlab["maximum_directivity_dbi"]),
            "maximum_gain_dbi": float(matlab["maximum_gain_dbi"]),
            "maximum_realized_gain_dbi": float(matlab["maximum_realized_gain_dbi"]),
            "radiation_efficiency": float(matlab["radiation_efficiency"]),
            "mismatch_efficiency": float(matlab["mismatch_efficiency"]),
            "total_efficiency": float(matlab["total_efficiency"]),
            "directivity_sphere_average": sphere_average(
                10 ** (matlab["directivity_dbi"] / 10), theta, phi
            ),
            "gain_identity_max_error_db": float(matlab["gain_identity_max_error_db"]),
            "realized_gain_identity_max_error_db": float(
                matlab["realized_gain_identity_max_error_db"]
            ),
            "radiation_intensity_identity_max_error_db": float(
                matlab["power_directivity_identity_max_error_db"]
            ),
        },
        "ideal_half_wave": {
            "maximum_directivity_dbi": float(np.max(ideal_dbi)),
        },
    }
    gprmax_normalized = gprmax["directivity_dbi"] - np.max(gprmax["directivity_dbi"])
    matlab_normalized = matlab["directivity_dbi"] - np.max(matlab["directivity_dbi"])
    ideal_normalized = ideal_dbi - np.max(ideal_dbi)
    result["cross_code"] = {
        "directivity_peak_difference_db": (
            result["gprmax"]["maximum_directivity_dbi"]
            - result["matlab"]["maximum_directivity_dbi"]
        ),
        "gain_peak_difference_db": (
            result["gprmax"]["maximum_gain_dbi"] - result["matlab"]["maximum_gain_dbi"]
        ),
        "realized_gain_peak_difference_db": (
            result["gprmax"]["maximum_realized_gain_dbi"]
            - result["matlab"]["maximum_realized_gain_dbi"]
        ),
        "directivity_rms_difference_db_above_floor": rms(
            (gprmax["directivity_dbi"] - matlab["directivity_dbi"])[valid]
        ),
        "normalized_pattern_rms_difference_db_above_floor": rms(
            (gprmax_normalized - matlab_normalized)[valid]
        ),
        "gprmax_vs_ideal_normalized_pattern_rms_db": rms(
            (gprmax_normalized - ideal_normalized)[ideal_valid]
        ),
        "matlab_vs_ideal_normalized_pattern_rms_db": rms(
            (matlab_normalized - ideal_normalized)[ideal_valid]
        ),
        "radiation_efficiency_difference": (
            result["gprmax"]["radiation_efficiency"] - result["matlab"]["radiation_efficiency"]
        ),
        "mismatch_efficiency_difference": (
            result["gprmax"]["mismatch_efficiency"] - result["matlab"]["mismatch_efficiency"]
        ),
        "total_efficiency_difference": (
            result["gprmax"]["total_efficiency"] - result["matlab"]["total_efficiency"]
        ),
    }
    return result, ideal_dbi


def plot_comparison(gprmax, matlab, metrics, ideal_dbi):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    angle, gprmax_elevation = signed_elevation(
        gprmax["directivity_dbi"], gprmax["theta"], gprmax["phi"]
    )
    _, matlab_elevation = signed_elevation(
        matlab["directivity_dbi"], matlab["theta"], matlab["phi"]
    )
    _, ideal_elevation = signed_elevation(ideal_dbi, gprmax["theta"], gprmax["phi"])
    axes[0, 0].plot(angle, gprmax_elevation, label="gprMax", linewidth=2)
    axes[0, 0].plot(angle, matlab_elevation, "--", label="MATLAB", linewidth=2)
    axes[0, 0].plot(angle, ideal_elevation, ":", label="Ideal half-wave", linewidth=2)
    axes[0, 0].set_title("Elevation-plane directivity")
    axes[0, 0].set_xlabel("Signed angle from +z (degrees)")
    axes[0, 0].set_ylabel("Directivity (dBi)")
    axes[0, 0].set_xlim(-180, 180)
    axes[0, 0].set_ylim(-30, 4)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    phi, gprmax_azimuth = azimuth_cut(gprmax["directivity_dbi"], gprmax["theta"], gprmax["phi"])
    _, matlab_azimuth = azimuth_cut(matlab["directivity_dbi"], matlab["theta"], matlab["phi"])
    axes[0, 1].plot(phi, gprmax_azimuth, label="gprMax", linewidth=2)
    axes[0, 1].plot(phi, matlab_azimuth, "--", label="MATLAB", linewidth=2)
    axes[0, 1].set_title("Azimuth-plane directivity")
    axes[0, 1].set_xlabel("Azimuth (degrees)")
    axes[0, 1].set_ylabel("Directivity (dBi)")
    axes[0, 1].set_xlim(0, 360)
    azimuth_centre = np.mean((gprmax_azimuth, matlab_azimuth))
    axes[0, 1].set_ylim(azimuth_centre - 0.05, azimuth_centre + 0.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    labels = ("Directivity", "Gain", "Realized gain")
    gprmax_peaks = [
        metrics["gprmax"]["maximum_directivity_dbi"],
        metrics["gprmax"]["maximum_gain_dbi"],
        metrics["gprmax"]["maximum_realized_gain_dbi"],
    ]
    matlab_peaks = [
        metrics["matlab"]["maximum_directivity_dbi"],
        metrics["matlab"]["maximum_gain_dbi"],
        metrics["matlab"]["maximum_realized_gain_dbi"],
    ]
    index = np.arange(3)
    width = 0.36
    axes[1, 0].bar(index - width / 2, gprmax_peaks, width, label="gprMax")
    axes[1, 0].bar(index + width / 2, matlab_peaks, width, label="MATLAB")
    axes[1, 0].set_xticks(index, labels)
    axes[1, 0].set_ylabel("Peak metric (dBi)")
    axes[1, 0].set_title("Absolute peak metrics")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    axes[1, 0].legend()

    efficiency_labels = ("Radiation", "Mismatch", "Total")
    gprmax_efficiency = 100 * np.asarray(
        [
            metrics["gprmax"]["radiation_efficiency"],
            metrics["gprmax"]["mismatch_efficiency"],
            metrics["gprmax"]["total_efficiency"],
        ]
    )
    matlab_efficiency = 100 * np.asarray(
        [
            metrics["matlab"]["radiation_efficiency"],
            metrics["matlab"]["mismatch_efficiency"],
            metrics["matlab"]["total_efficiency"],
        ]
    )
    axes[1, 1].bar(index - width / 2, gprmax_efficiency, width, label="gprMax")
    axes[1, 1].bar(index + width / 2, matlab_efficiency, width, label="MATLAB")
    axes[1, 1].set_xticks(index, efficiency_labels)
    axes[1, 1].set_ylabel("Efficiency (%)")
    axes[1, 1].set_title("73 Ohm power normalisation at 0.95 GHz")
    axes[1, 1].set_ylim(95, 100.2)
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    axes[1, 1].legend()

    figure.suptitle(
        "Cylindrical half-wave dipole: gprMax FDTD/KSIR vs MATLAB MoM",
        fontsize=15,
    )
    figure.savefig(OUTPUT_PNG, dpi=200)
    plt.close(figure)


def main():
    gprmax = read_gprmax()
    matlab = read_matlab()
    metrics, ideal_dbi = compare(gprmax, matlab)
    OUTPUT_JSON.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_comparison(gprmax, matlab, metrics, ideal_dbi)
    print(json.dumps(metrics, indent=2))
    print(f"Saved cylindrical-dipole metric comparison to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
