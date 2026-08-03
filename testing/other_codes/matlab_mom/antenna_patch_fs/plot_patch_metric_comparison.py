"""Compare full-sphere gprMax antenna metrics with MATLAB Antenna Toolbox."""

import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
GPRMAX_HDF5 = RESULTS_DIR / "patch_antenna_3d_gain_single_feed.h5"
MATLAB_CSV = RESULTS_DIR / "patch_antenna_matlab_3d_metrics.csv"
MATLAB_JSON = RESULTS_DIR / "patch_antenna_matlab_3d_metrics.json"
OUTPUT_PNG = RESULTS_DIR / "patch_antenna_metric_comparison.png"
OUTPUT_JSON = RESULTS_DIR / "patch_antenna_metric_comparison.json"
GROUP = "ntff/patch_surface/frequency/patch_spectrum/far_field/gain_3d"


def read_gprmax():
    """Read every compared metric from the persisted gprMax HDF5 file."""

    with h5py.File(GPRMAX_HDF5, "r") as output:
        group = output[GROUP]
        result = {
            "theta": np.asarray(group["theta"], dtype=np.float64),
            "phi": np.asarray(group["phi"], dtype=np.float64),
            "directivity_dbi": np.asarray(group["fields/directivity_dbi"][0], dtype=np.float64),
            "radiation_intensity": np.asarray(
                group["fields/radiation_intensity"][0], dtype=np.float64
            ),
            "gain_dbi": np.asarray(group["fields/gain_dbi"][0], dtype=np.float64),
            "realized_gain_dbi": np.asarray(group["fields/realized_gain_dbi"][0], dtype=np.float64),
            "radiation_efficiency": float(group["fields/radiation_efficiency"][0]),
            "total_efficiency": float(group["fields/total_efficiency"][0]),
            "maximum_directivity": float(group["maximum_directivity"][0]),
            "maximum_theta": float(group["maximum_directivity_theta"][0]),
            "maximum_phi": float(group["maximum_directivity_phi"][0]),
            "radiated_power": float(group["radiated_power"][0]),
            "accepted_power": float(group["port_power/accepted_power"][0]),
            "incident_power": float(group["port_power/incident_power"][0]),
        }
    result["mismatch_efficiency"] = result["accepted_power"] / result["incident_power"]
    return result


def read_matlab():
    """Read the Antenna Toolbox angular table and scalar metrics."""

    table = np.genfromtxt(MATLAB_CSV, delimiter=",", names=True)
    result = {
        "theta": np.asarray(table["theta_deg"], dtype=np.float64),
        "phi": np.asarray(table["phi_deg"], dtype=np.float64),
        "directivity_dbi": np.asarray(table["directivity_dbi"], dtype=np.float64),
        "gain_dbi": np.asarray(table["gain_dbi"], dtype=np.float64),
        "realized_gain_dbi": np.asarray(table["realized_gain_dbi"], dtype=np.float64),
        "radiation_intensity": np.asarray(table["radiation_intensity_w_per_sr"], dtype=np.float64),
    }
    result.update(json.loads(MATLAB_JSON.read_text(encoding="utf-8")))
    return result


def sphere_average(linear_pattern, theta, phi):
    """Integrate a regular periodic-phi grid and divide by 4 pi."""

    theta_axis = np.unique(theta)
    phi_axis = np.unique(phi)
    pattern = linear_pattern.reshape(theta_axis.size, phi_axis.size)
    theta_rad = np.deg2rad(theta_axis)
    theta_integral = np.trapezoid(
        pattern * np.sin(theta_rad)[:, np.newaxis],
        theta_rad,
        axis=0,
    )
    phi_step = 2 * np.pi / phi_axis.size
    return float(phi_step * np.sum(theta_integral) / (4 * np.pi))


def signed_cut(values, theta, phi, plane):
    """Return a conventional -180 to 180 degree principal-plane cut."""

    lookup = {
        (float(theta_value), float(phi_value)): value
        for theta_value, phi_value, value in zip(theta, phi, values)
    }
    angle = np.arange(-180.0, 182.0, 2.0)
    if plane == "xz":
        negative_phi, positive_phi = 180.0, 0.0
    else:
        negative_phi, positive_phi = 270.0, 90.0
    cut = np.asarray(
        [lookup[(abs(value), negative_phi if value < 0 else positive_phi)] for value in angle]
    )
    return angle, cut


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def compare(gprmax, matlab):
    """Calculate cross-code and within-code consistency diagnostics."""

    if not np.allclose(gprmax["theta"], matlab["theta"], rtol=0, atol=1e-12):
        raise ValueError("The gprMax and MATLAB theta grids differ")
    if not np.allclose(gprmax["phi"], matlab["phi"], rtol=0, atol=1e-12):
        raise ValueError("The gprMax and MATLAB phi grids differ")

    theta = gprmax["theta"]
    phi = gprmax["phi"]
    directivity_floor = (
        max(
            np.max(gprmax["directivity_dbi"]),
            np.max(matlab["directivity_dbi"]),
        )
        - 30
    )
    valid = (gprmax["directivity_dbi"] >= directivity_floor) & (
        matlab["directivity_dbi"] >= directivity_floor
    )
    upper_valid = valid & (theta <= 90)
    result = {
        "frequency_hz": float(matlab["frequency_hz"]),
        "angular_samples": int(theta.size),
        "comparison_floor_relative_to_peak_db": -30.0,
        "gprmax": {
            "maximum_directivity_dbi": float(10 * np.log10(gprmax["maximum_directivity"])),
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
    }
    directivity_difference = gprmax["directivity_dbi"] - matlab["directivity_dbi"]
    gain_difference = gprmax["gain_dbi"] - matlab["gain_dbi"]
    realized_difference = gprmax["realized_gain_dbi"] - matlab["realized_gain_dbi"]
    normalized_gprmax = gprmax["directivity_dbi"] - np.max(gprmax["directivity_dbi"])
    normalized_matlab = matlab["directivity_dbi"] - np.max(matlab["directivity_dbi"])
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
        "directivity_rms_difference_db_above_floor": rms(directivity_difference[valid]),
        "gain_rms_difference_db_above_floor": rms(gain_difference[valid]),
        "realized_gain_rms_difference_db_above_floor": rms(realized_difference[valid]),
        "normalized_pattern_rms_difference_db_above_floor": rms(
            (normalized_gprmax - normalized_matlab)[valid]
        ),
        "upper_hemisphere_directivity_rms_difference_db_above_floor": rms(
            directivity_difference[upper_valid]
        ),
        "upper_hemisphere_normalized_pattern_rms_difference_db_above_floor": rms(
            (normalized_gprmax - normalized_matlab)[upper_valid]
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
    return result


def plot_comparison(gprmax, matlab, metrics):
    """Create principal-cut, peak-metric, and efficiency comparisons."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, plane, title in (
        (axes[0, 0], "xz", "E-plane (x-z) directivity"),
        (axes[0, 1], "yz", "H-plane (y-z) directivity"),
    ):
        angle, gprmax_cut = signed_cut(
            gprmax["directivity_dbi"], gprmax["theta"], gprmax["phi"], plane
        )
        _, matlab_cut = signed_cut(matlab["directivity_dbi"], matlab["theta"], matlab["phi"], plane)
        axis.plot(angle, gprmax_cut, label="gprMax FDTD/KSIR", linewidth=2)
        axis.plot(angle, matlab_cut, "--", label="MATLAB MoM", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Signed angle from +z (degrees)")
        axis.set_ylabel("Directivity (dBi)")
        axis.set_xlim(-180, 180)
        axis.set_ylim(-25, 8)
        axis.grid(True, alpha=0.3)
        axis.legend()

    labels = ("Directivity", "Gain", "Realized gain")
    gprmax_peaks = (
        metrics["gprmax"]["maximum_directivity_dbi"],
        metrics["gprmax"]["maximum_gain_dbi"],
        metrics["gprmax"]["maximum_realized_gain_dbi"],
    )
    matlab_peaks = (
        metrics["matlab"]["maximum_directivity_dbi"],
        metrics["matlab"]["maximum_gain_dbi"],
        metrics["matlab"]["maximum_realized_gain_dbi"],
    )
    index = np.arange(len(labels))
    width = 0.36
    axes[1, 0].bar(index - width / 2, gprmax_peaks, width, label="gprMax")
    axes[1, 0].bar(index + width / 2, matlab_peaks, width, label="MATLAB")
    axes[1, 0].set_xticks(index, labels)
    axes[1, 0].set_ylabel("Peak metric (dBi)")
    axes[1, 0].set_title("Absolute peak antenna metrics")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    axes[1, 0].legend()

    efficiency_labels = ("Radiation", "Mismatch", "Total")
    gprmax_efficiency = 100 * np.asarray(
        (
            metrics["gprmax"]["radiation_efficiency"],
            metrics["gprmax"]["mismatch_efficiency"],
            metrics["gprmax"]["total_efficiency"],
        )
    )
    matlab_efficiency = 100 * np.asarray(
        (
            metrics["matlab"]["radiation_efficiency"],
            metrics["matlab"]["mismatch_efficiency"],
            metrics["matlab"]["total_efficiency"],
        )
    )
    axes[1, 1].bar(index - width / 2, gprmax_efficiency, width, label="gprMax")
    axes[1, 1].bar(index + width / 2, matlab_efficiency, width, label="MATLAB")
    axes[1, 1].set_xticks(index, efficiency_labels)
    axes[1, 1].set_ylabel("Efficiency (%)")
    axes[1, 1].set_title("Power normalisation at 2.37 GHz")
    axes[1, 1].set_ylim(0, 108)
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    axes[1, 1].legend()

    figure.suptitle(
        "Rectangular patch: gprMax FDTD/KSIR vs MATLAB Antenna Toolbox MoM",
        fontsize=15,
    )
    figure.savefig(OUTPUT_PNG, dpi=200)
    plt.close(figure)


def main():
    gprmax = read_gprmax()
    matlab = read_matlab()
    metrics = compare(gprmax, matlab)
    OUTPUT_JSON.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_comparison(gprmax, matlab, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metric comparison to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
