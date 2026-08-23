"""Plot a principal-plane cut and 3-D realized gain for the wire dipole."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "antenna_wire_dipole_pattern"
PATTERN_FREQUENCY = 0.95e9
GROUP = "ntff/dipole_surface/frequency/dipole_pattern/far_field/full_sphere"
DYNAMIC_RANGE_DB = 30.0


def read_pattern(filename):
    with h5py.File(filename, "r") as output:
        group = output[GROUP]
        result = {
            "theta": np.asarray(group["theta"], dtype=np.float64),
            "phi": np.asarray(group["phi"], dtype=np.float64),
            "directivity_dbi": np.asarray(group["fields/directivity_dbi"][0], dtype=np.float64),
            "gain_dbi": np.asarray(group["fields/gain_dbi"][0], dtype=np.float64),
            "realized_gain_dbi": np.asarray(group["fields/realized_gain_dbi"][0], dtype=np.float64),
            "radiation_efficiency": float(group["fields/radiation_efficiency"][0]),
            "total_efficiency": float(group["fields/total_efficiency"][0]),
            "maximum_directivity_dbi": float(group["maximum_directivity_dbi"][0]),
            "accepted_power": float(group["port_power/accepted_power"][0]),
            "incident_power": float(group["port_power/incident_power"][0]),
        }
    return result


def signed_elevation(values, theta_axis, phi_axis):
    values = values.reshape(theta_axis.size, phi_axis.size)
    phi_zero = int(np.argmin(np.abs(phi_axis)))
    phi_opposite = int(np.argmin(np.abs(phi_axis - 180)))
    angle = np.concatenate((-theta_axis[:0:-1], theta_axis))
    cut = np.concatenate((values[:0:-1, phi_opposite], values[:, phi_zero]))
    return angle, cut


def plot_pattern(result, destination):
    theta_axis = np.unique(result["theta"])
    phi_axis = np.unique(result["phi"])
    shape = (theta_axis.size, phi_axis.size)
    realized = result["realized_gain_dbi"].reshape(shape)

    finite_realized = realized[np.isfinite(realized)]
    peak_realized = float(np.max(finite_realized))
    floor = peak_realized - DYNAMIC_RANGE_DB
    realized_for_plot = np.where(np.isfinite(realized), realized, floor)
    realized_for_plot = np.maximum(realized_for_plot, floor)

    phi_closed = np.append(phi_axis, 360.0)
    realized_closed = np.concatenate((realized_for_plot, realized_for_plot[:, :1]), axis=1)
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_closed, indexing="ij")
    radius = (realized_closed - floor) / DYNAMIC_RANGE_DB
    theta_rad = np.deg2rad(theta_grid)
    phi_rad = np.deg2rad(phi_grid)
    x = radius * np.sin(theta_rad) * np.cos(phi_rad)
    y = radius * np.sin(theta_rad) * np.sin(phi_rad)
    z = radius * np.cos(theta_rad)

    figure = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    cut_axis = figure.add_subplot(1, 2, 1)
    for key, label, style in (
        ("directivity_dbi", "Directivity", "-"),
        ("gain_dbi", "Gain", "--"),
        ("realized_gain_dbi", "Realized gain", ":"),
    ):
        angle, cut = signed_elevation(result[key], theta_axis, phi_axis)
        cut_axis.plot(angle, cut, style, linewidth=2, label=label)
    mismatch_efficiency = result["accepted_power"] / result["incident_power"]
    summary = (
        f"Peak directivity: {result['maximum_directivity_dbi']:.2f} dBi\n"
        f"Peak realized gain: {peak_realized:.2f} dBi\n"
        f"Radiation efficiency: {100 * result['radiation_efficiency']:.1f}%\n"
        f"Mismatch efficiency: {100 * mismatch_efficiency:.1f}%\n"
        f"Total efficiency: {100 * result['total_efficiency']:.1f}%"
    )
    cut_axis.text(
        0.03,
        0.04,
        summary,
        transform=cut_axis.transAxes,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    cut_axis.set_title("Elevation-plane antenna quantities")
    cut_axis.set_xlabel("Signed angle from +z (degrees)")
    cut_axis.set_ylabel("Level (dBi)")
    cut_axis.set_xlim(-180, 180)
    cut_axis.set_ylim(floor, result["maximum_directivity_dbi"] + 1)
    cut_axis.grid(True, alpha=0.3)
    cut_axis.legend(loc="upper right")

    surface_axis = figure.add_subplot(1, 2, 2, projection="3d")
    colour_norm = colors.Normalize(vmin=floor, vmax=peak_realized)
    colour_map = plt.get_cmap("viridis")
    face_colours = colour_map(colour_norm(realized_closed))
    surface_axis.plot_surface(
        x,
        y,
        z,
        facecolors=face_colours,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    scalar_map = plt.cm.ScalarMappable(norm=colour_norm, cmap=colour_map)
    scalar_map.set_array([])
    colour_bar = figure.colorbar(scalar_map, ax=surface_axis, shrink=0.72, pad=0.08)
    colour_bar.set_label("Realized gain (dBi)")
    surface_axis.set_xlabel("x")
    surface_axis.set_ylabel("y")
    surface_axis.set_zlabel("z (dipole axis)")
    surface_axis.set_box_aspect((1, 1, 1))
    surface_axis.set_xlim(-1, 1)
    surface_axis.set_ylim(-1, 1)
    surface_axis.set_zlim(-1, 1)
    surface_axis.view_init(elev=24, azim=-42)
    surface_axis.set_title(f"3-D realized gain ({DYNAMIC_RANGE_DB:g} dB radial range)")

    figure.suptitle(f"Wire dipole at {PATTERN_FREQUENCY / 1e9:g} GHz")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR / f"{OUTPUT_STEM}.h5",
        help="gprMax HDF5 output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / f"{OUTPUT_STEM}.png",
        help="destination PNG",
    )
    args = parser.parse_args()
    plot_pattern(read_pattern(args.input), args.output)


if __name__ == "__main__":
    main()
