"""Compare gprMax and MATLAB three-turn helical-antenna metrics."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
MATLAB_CSV = RESULTS_DIR / "helix_antenna_matlab_metrics.csv"
MATLAB_JSON = RESULTS_DIR / "helix_antenna_matlab_metrics.json"
GROUP = "ntff/helix_surface/frequency/helix_spectrum/far_field/metrics_3d"


def read_gprmax(filename):
    """Read antenna metrics and complex fields from persisted HDF5."""

    with h5py.File(filename, "r") as output:
        group = output[GROUP]
        result = {
            "theta": np.asarray(group["theta"], dtype=np.float64),
            "phi": np.asarray(group["phi"], dtype=np.float64),
            "etheta": np.asarray(group["fields/Etheta"][0], dtype=np.complex128),
            "ephi": np.asarray(group["fields/Ephi"][0], dtype=np.complex128),
            "directivity_dbi": np.asarray(group["fields/directivity_dbi"][0], dtype=np.float64),
            "gain_dbi": np.asarray(group["fields/gain_dbi"][0], dtype=np.float64),
            "realized_gain_dbi": np.asarray(group["fields/realized_gain_dbi"][0], dtype=np.float64),
            "radiation_efficiency": float(group["fields/radiation_efficiency"][0]),
            "total_efficiency": float(group["fields/total_efficiency"][0]),
            "maximum_directivity": float(group["maximum_directivity"][0]),
            "accepted_power": float(group["port_power/accepted_power"][0]),
            "incident_power": float(group["port_power/incident_power"][0]),
            "terminal_voltage": complex(group["port_power/terminal_voltage_per_port"][0, 0]),
            "terminal_current": complex(group["port_power/terminal_current_per_port"][0, 0]),
        }
    result["mismatch_efficiency"] = result["accepted_power"] / result["incident_power"]
    result["input_impedance"] = result["terminal_voltage"] / result["terminal_current"]
    return result


def read_matlab():
    """Read the common-angle MATLAB Antenna Toolbox result."""

    table = np.genfromtxt(MATLAB_CSV, delimiter=",", names=True)
    result = {
        "theta": np.asarray(table["theta_deg"], dtype=np.float64),
        "phi": np.asarray(table["phi_deg"], dtype=np.float64),
        "directivity_dbi": np.asarray(table["directivity_dbi"], dtype=np.float64),
        "gain_dbi": np.asarray(table["gain_dbi"], dtype=np.float64),
        "realized_gain_dbi": np.asarray(table["realized_gain_dbi"], dtype=np.float64),
        "rhcp_directivity_dbi": np.asarray(table["rhcp_directivity_dbi"], dtype=np.float64),
        "lhcp_directivity_dbi": np.asarray(table["lhcp_directivity_dbi"], dtype=np.float64),
        "axial_ratio_db": np.asarray(table["axial_ratio_db"], dtype=np.float64),
    }
    result.update(json.loads(MATLAB_JSON.read_text(encoding="utf-8")))
    return result


def circular_components(gprmax, matlab):
    """Calculate circular components and map the phasor convention to MATLAB."""

    etheta = gprmax["etheta"]
    ephi = gprmax["ephi"]
    plus = (etheta + 1j * ephi) / np.sqrt(2)
    minus = (etheta - 1j * ephi) / np.sqrt(2)
    total_power = np.abs(etheta) ** 2 + np.abs(ephi) ** 2
    directivity = 10 ** (gprmax["directivity_dbi"] / 10)
    plus_power = directivity * np.abs(plus) ** 2 / total_power
    minus_power = directivity * np.abs(minus) ** 2 / total_power
    floor = np.finfo(float).tiny
    candidates = (
        10 * np.log10(np.maximum(plus_power, floor)),
        10 * np.log10(np.maximum(minus_power, floor)),
    )

    # e^(+jwt)/e^(-jwt) conventions exchange the RHCP/LHCP labels. Select the
    # mapping that minimizes the independently calculated MATLAB component
    # patterns, while preserving both computed gprMax fields unchanged.
    valid = (
        np.isfinite(matlab["rhcp_directivity_dbi"])
        & np.isfinite(matlab["lhcp_directivity_dbi"])
        & (gprmax["directivity_dbi"] > np.max(gprmax["directivity_dbi"]) - 25)
    )
    direct_cost = np.mean(
        (candidates[0][valid] - matlab["rhcp_directivity_dbi"][valid]) ** 2
        + (candidates[1][valid] - matlab["lhcp_directivity_dbi"][valid]) ** 2
    )
    swapped_cost = np.mean(
        (candidates[1][valid] - matlab["rhcp_directivity_dbi"][valid]) ** 2
        + (candidates[0][valid] - matlab["lhcp_directivity_dbi"][valid]) ** 2
    )
    if direct_cost <= swapped_cost:
        rhcp, lhcp = candidates
        mapping = "Etheta+jEphi maps to MATLAB RHCP"
    else:
        lhcp, rhcp = candidates
        mapping = "Etheta-jEphi maps to MATLAB RHCP"

    rhcp_field = np.sqrt(10 ** (rhcp / 10))
    lhcp_field = np.sqrt(10 ** (lhcp / 10))
    axial_ratio = 20 * np.log10(
        (rhcp_field + lhcp_field) / np.maximum(np.abs(rhcp_field - lhcp_field), np.finfo(float).eps)
    )
    return rhcp, lhcp, axial_ratio, mapping


def angular_cut(values, theta, phi, selected_phi, theta_stop=180):
    selected = np.isclose(phi, selected_phi) & (theta <= theta_stop)
    order = np.argsort(theta[selected])
    return theta[selected][order], values[selected][order]


def signed_axial_cut(values, theta, phi):
    positive_angle, positive = angular_cut(values, theta, phi, 0, 90)
    negative_angle, negative = angular_cut(values, theta, phi, 180, 90)
    return (
        np.concatenate((-negative_angle[:0:-1], positive_angle)),
        np.concatenate((negative[:0:-1], positive)),
    )


def first_crossing(theta, values, threshold):
    below = np.flatnonzero(values <= threshold)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    upper = int(below[0])
    lower = upper - 1
    fraction = (threshold - values[lower]) / (values[upper] - values[lower])
    return float(theta[lower] + fraction * (theta[upper] - theta[lower]))


def beamwidth(values, theta, phi):
    theta0, side0 = angular_cut(values, theta, phi, 0, 90)
    theta180, side180 = angular_cut(values, theta, phi, 180, 90)
    threshold = max(np.max(side0), np.max(side180)) - 3
    return first_crossing(theta0, side0, threshold) + first_crossing(theta180, side180, threshold)


def pole_average(values, theta, pole):
    return float(np.mean(values[np.isclose(theta, pole)]))


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def compare(gprmax, matlab):
    """Calculate scalar and pattern-difference metrics."""

    if not np.allclose(gprmax["theta"], matlab["theta"], atol=1e-12, rtol=0):
        raise ValueError("The gprMax and MATLAB theta grids differ")
    if not np.allclose(gprmax["phi"], matlab["phi"], atol=1e-12, rtol=0):
        raise ValueError("The gprMax and MATLAB phi grids differ")

    rhcp, lhcp, axial_ratio, mapping = circular_components(gprmax, matlab)
    theta = gprmax["theta"]
    valid = (
        np.isfinite(gprmax["directivity_dbi"])
        & np.isfinite(matlab["directivity_dbi"])
        & (gprmax["directivity_dbi"] > np.max(gprmax["directivity_dbi"]) - 25)
        & (matlab["directivity_dbi"] > np.max(matlab["directivity_dbi"]) - 25)
    )
    axis = np.isclose(theta, 0)
    back = np.isclose(theta, 180)
    result = {
        "frequency_hz": float(matlab["frequency_hz"]),
        "angular_samples": int(theta.size),
        "circular_component_mapping": mapping,
        "gprmax": {
            "maximum_directivity_dbi": float(10 * np.log10(gprmax["maximum_directivity"])),
            "maximum_gain_dbi": float(np.max(gprmax["gain_dbi"])),
            "maximum_realized_gain_dbi": float(np.max(gprmax["realized_gain_dbi"])),
            "axis_directivity_dbi": pole_average(gprmax["directivity_dbi"], theta, 0),
            "axis_realized_gain_dbi": pole_average(gprmax["realized_gain_dbi"], theta, 0),
            "axis_axial_ratio_db": float(np.mean(axial_ratio[axis])),
            "front_to_back_ratio_db": float(
                np.mean(gprmax["directivity_dbi"][axis]) - np.mean(gprmax["directivity_dbi"][back])
            ),
            "half_power_beamwidth_deg": beamwidth(gprmax["directivity_dbi"], theta, gprmax["phi"]),
            "radiation_efficiency": gprmax["radiation_efficiency"],
            "mismatch_efficiency": gprmax["mismatch_efficiency"],
            "total_efficiency": gprmax["total_efficiency"],
            "input_impedance_real_ohm": float(gprmax["input_impedance"].real),
            "input_impedance_imag_ohm": float(gprmax["input_impedance"].imag),
            "s11_magnitude_db": float(10 * np.log10(1 - gprmax["mismatch_efficiency"])),
        },
        "matlab": {
            "maximum_directivity_dbi": float(matlab["maximum_directivity_dbi"]),
            "maximum_gain_dbi": float(matlab["maximum_gain_dbi"]),
            "maximum_realized_gain_dbi": float(matlab["maximum_realized_gain_dbi"]),
            "axis_directivity_dbi": pole_average(matlab["directivity_dbi"], theta, 0),
            "axis_realized_gain_dbi": pole_average(matlab["realized_gain_dbi"], theta, 0),
            "axis_axial_ratio_db": float(matlab["axis_axial_ratio_native_db"]),
            "front_to_back_ratio_db": float(
                np.mean(matlab["directivity_dbi"][axis]) - np.mean(matlab["directivity_dbi"][back])
            ),
            "half_power_beamwidth_deg": beamwidth(matlab["directivity_dbi"], theta, matlab["phi"]),
            "radiation_efficiency": float(matlab["radiation_efficiency"]),
            "mismatch_efficiency": float(matlab["mismatch_efficiency"]),
            "total_efficiency": float(matlab["total_efficiency"]),
            "input_impedance_real_ohm": float(matlab["input_impedance_real_ohm"]),
            "input_impedance_imag_ohm": float(matlab["input_impedance_imag_ohm"]),
            "s11_magnitude_db": float(matlab["s11_magnitude_db"]),
        },
    }
    result["cross_code"] = {
        "maximum_directivity_difference_db": (
            result["gprmax"]["maximum_directivity_dbi"]
            - result["matlab"]["maximum_directivity_dbi"]
        ),
        "maximum_realized_gain_difference_db": (
            result["gprmax"]["maximum_realized_gain_dbi"]
            - result["matlab"]["maximum_realized_gain_dbi"]
        ),
        "directivity_rms_difference_db_above_minus_25_db": rms(
            (gprmax["directivity_dbi"] - matlab["directivity_dbi"])[valid]
        ),
        "axis_axial_ratio_difference_db": (
            result["gprmax"]["axis_axial_ratio_db"] - result["matlab"]["axis_axial_ratio_db"]
        ),
        "half_power_beamwidth_difference_deg": (
            result["gprmax"]["half_power_beamwidth_deg"]
            - result["matlab"]["half_power_beamwidth_deg"]
        ),
    }
    return result, rhcp, lhcp, axial_ratio


def plot_comparison(gprmax, matlab, metrics, rhcp, lhcp, axial_ratio, output_png):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    angle, gpr_cut = signed_axial_cut(gprmax["directivity_dbi"], gprmax["theta"], gprmax["phi"])
    _, matlab_cut = signed_axial_cut(matlab["directivity_dbi"], matlab["theta"], matlab["phi"])
    axes[0, 0].plot(angle, gpr_cut, linewidth=2, label="gprMax")
    axes[0, 0].plot(angle, matlab_cut, "--", linewidth=2, label="MATLAB")
    axes[0, 0].set(
        xlim=(-90, 90),
        ylim=(-25, 12),
        xlabel="Angle from +z (degrees)",
        ylabel="Directivity (dBi)",
        title="Axial-plane pattern",
    )
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    theta, gpr_rhcp = angular_cut(rhcp, gprmax["theta"], gprmax["phi"], 0, 90)
    _, gpr_lhcp = angular_cut(lhcp, gprmax["theta"], gprmax["phi"], 0, 90)
    _, matlab_rhcp = angular_cut(
        matlab["rhcp_directivity_dbi"], matlab["theta"], matlab["phi"], 0, 90
    )
    _, matlab_lhcp = angular_cut(
        matlab["lhcp_directivity_dbi"], matlab["theta"], matlab["phi"], 0, 90
    )
    axes[0, 1].plot(theta, gpr_rhcp, linewidth=2, label="gprMax RHCP")
    axes[0, 1].plot(theta, matlab_rhcp, "--", linewidth=2, label="MATLAB RHCP")
    axes[0, 1].plot(theta, gpr_lhcp, linewidth=1.5, label="gprMax LHCP")
    axes[0, 1].plot(theta, matlab_lhcp, "--", linewidth=1.5, label="MATLAB LHCP")
    axes[0, 1].set(
        xlim=(0, 90),
        ylim=(-30, 12),
        xlabel="Theta (degrees)",
        ylabel="Component directivity (dBi)",
        title="Circularly polarized components",
    )
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(ncol=2, fontsize=8)

    labels = ("Directivity", "Gain", "Realized gain")
    gpr_peaks = [
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
    axes[1, 0].bar(index - width / 2, gpr_peaks, width, label="gprMax")
    axes[1, 0].bar(index + width / 2, matlab_peaks, width, label="MATLAB")
    axes[1, 0].set_xticks(index, labels)
    axes[1, 0].set(ylabel="Peak metric (dBi)", title="Absolute antenna metrics")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    axes[1, 0].legend()

    theta, gpr_ar = angular_cut(axial_ratio, gprmax["theta"], gprmax["phi"], 0, 60)
    _, matlab_ar = angular_cut(matlab["axial_ratio_db"], matlab["theta"], matlab["phi"], 0, 60)
    axes[1, 1].plot(theta, gpr_ar, linewidth=2, label="gprMax")
    axes[1, 1].plot(theta, matlab_ar, "--", linewidth=2, label="MATLAB")
    axes[1, 1].axhline(3, color="black", linestyle=":", label="3 dB")
    axes[1, 1].set(
        xlim=(0, 60),
        ylim=(0, 15),
        xlabel="Theta (degrees)",
        ylabel="Axial ratio (dB)",
        title="Circular-polarization quality",
    )
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    figure.suptitle("Three-turn axial-mode helix at 2.2 GHz: gprMax vs MATLAB", fontsize=15)
    figure.savefig(output_png, dpi=200)
    plt.close(figure)


def plot_3d_gain(gprmax, output_png):
    theta_axis = np.unique(gprmax["theta"])
    phi_axis = np.unique(gprmax["phi"])
    gain = gprmax["realized_gain_dbi"].reshape(theta_axis.size, phi_axis.size)
    theta, phi = np.meshgrid(np.deg2rad(theta_axis), np.deg2rad(phi_axis), indexing="ij")
    peak = float(np.max(gain))
    clipped = np.maximum(gain, peak - 25)
    radius = 10 ** ((clipped - peak) / 20)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    normalizer = colors.Normalize(vmin=peak - 25, vmax=peak)
    colormap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        x,
        y,
        z,
        facecolors=colormap(normalizer(clipped)),
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    axis.set_box_aspect((1, 1, 1))
    axis.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title=f"gprMax realized gain at 2.2 GHz (peak {peak:.2f} dBi)",
    )
    axis.view_init(elev=25, azim=-55)
    scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=colormap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=axis, shrink=0.7, pad=0.08, label="Realized gain (dBi)")
    figure.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mode", choices=("resistive", "hard"), default="resistive")
    args = parser.parse_args()
    suffix = "_hard" if args.source_mode == "hard" else ""
    gprmax_hdf5 = RESULTS_DIR / f"helix_antenna_gprmax{suffix}.h5"
    output_png = RESULTS_DIR / f"helix_antenna_comparison{suffix}.png"
    output_3d_png = RESULTS_DIR / f"helix_antenna_gprmax{suffix}_3d_realized_gain.png"
    output_json = RESULTS_DIR / f"helix_antenna_comparison{suffix}_metrics.json"

    gprmax = read_gprmax(gprmax_hdf5)
    matlab = read_matlab()
    metrics, rhcp, lhcp, axial_ratio = compare(gprmax, matlab)
    metrics["gprmax_source_mode"] = args.source_mode
    output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_comparison(gprmax, matlab, metrics, rhcp, lhcp, axial_ratio, output_png)
    plot_3d_gain(gprmax, output_3d_png)
    print(json.dumps(metrics, indent=2))
    print(f"Saved helix comparison to {output_png}")
    print(f"Saved gprMax 3D realized-gain pattern to {output_3d_png}")


if __name__ == "__main__":
    main()
