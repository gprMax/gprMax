"""Plot S11 and far-field results for the eigenmode-fed horn antenna."""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = EXAMPLE_DIR / "horn_antenna"
SPARAMETER_PLOT_PATH = EXAMPLE_DIR / "horn_sparameters.png"
FAR_FIELD_3D_PLOT_PATH = EXAMPLE_DIR / "horn_farfield_3d.png"
PRINCIPAL_PLANES_PLOT_PATH = EXAMPLE_DIR / "horn_principal_planes.png"
FAR_FIELD_GROUP = "ntff/horn_surface/frequency/antenna_band/far_field/full_sphere"
DESIGN_FREQUENCY = 10e9
PLOT_FLOOR_DB = -60.0
FAR_FIELD_QUANTITIES = (
    ("directivity_dbi", "Directivity", "-"),
    ("gain_dbi", "Gain", "--"),
    ("realized_gain_dbi", "Realized gain", ":"),
)


def read_sparameters(stem: Path):
    """Return valid reflection rows grouped by destination mode."""

    path = stem.with_name(stem.name + "_sparameters.csv")
    traces = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not bool(int(row["valid"])):
                continue
            mode = int(row["destination_mode"])
            traces.setdefault(mode, []).append((float(row["frequency_hz"]) * 1e-9, float(row["S_magnitude_db"])))
    if not traces:
        raise ValueError(f"No valid S-parameter rows found in {path}")
    return {mode: np.asarray(sorted(values), dtype=np.float64) for mode, values in traces.items()}


def read_far_fields(path: Path):
    """Read the full frequency sweep from the documented far-field group."""

    with h5py.File(path, "r") as output:
        group = output[FAR_FIELD_GROUP]
        transform = group.parent.parent
        return {
            "frequencies": np.asarray(transform["frequencies"], dtype=np.float64),
            "theta": np.asarray(group["theta"], dtype=np.float64),
            "phi": np.asarray(group["phi"], dtype=np.float64),
            "fields": {
                quantity: np.asarray(
                    group[f"fields/{quantity}"],
                    dtype=np.float64,
                )
                for quantity, _, _ in FAR_FIELD_QUANTITIES
            },
        }


def angular_grid(values, theta, phi):
    """Return unique angular axes and a theta-by-phi field grid."""

    theta_axis = np.unique(theta)
    phi_axis = np.unique(phi)
    grid = np.asarray(values).reshape(theta_axis.size, phi_axis.size)
    return theta_axis, phi_axis, grid


def xz_plane(values, theta, phi):
    """Return the phi=0/180 cut through the xz (nominal E) plane."""

    theta_axis, phi_axis, grid = angular_grid(values, theta, phi)
    phi_zero = int(np.argmin(np.abs(phi_axis)))
    phi_opposite = int(np.argmin(np.abs(phi_axis - 180)))
    theta_samples = np.deg2rad(np.concatenate((theta_axis, theta_axis)))
    phi_samples = np.concatenate(
        (
            np.full(theta_axis.shape, phi_axis[phi_zero]),
            np.full(theta_axis.shape, phi_axis[phi_opposite]),
        )
    )
    phi_samples = np.deg2rad(phi_samples)
    signed_angle = np.rad2deg(
        np.arctan2(
            np.cos(theta_samples),
            np.sin(theta_samples) * np.cos(phi_samples),
        )
    )
    cut = np.concatenate((grid[:, phi_zero], grid[:, phi_opposite]))
    order = np.argsort(signed_angle)
    return signed_angle[order], cut[order]


def xy_plane(values, theta, phi):
    """Return the theta=90 cut through the xy (nominal H) plane."""

    theta_axis, phi_axis, grid = angular_grid(values, theta, phi)
    theta_broadside = int(np.argmin(np.abs(theta_axis - 90)))
    signed_phi = (phi_axis + 180) % 360 - 180
    order = np.argsort(signed_phi)
    return signed_phi[order], grid[theta_broadside, order]


def plot_sparameters(traces):
    """Write the antenna input-reflection figure."""

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for mode, data in sorted(traces.items()):
        axis.plot(
            data[:, 0],
            np.maximum(data[:, 1], PLOT_FLOOR_DB),
            marker="o",
            linewidth=2,
            label=f"S11, reflected mode {mode}",
        )
    axis.axvline(DESIGN_FREQUENCY * 1e-9, color="0.4", linestyle=":", label="Design frequency")
    axis.set_title("Antenna input reflection")
    axis.set_xlabel("Frequency (GHz)")
    axis.set_ylabel(f"Magnitude (dB; floor {PLOT_FLOOR_DB:g} dB)")
    axis.set_ylim(PLOT_FLOOR_DB, 5)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(SPARAMETER_PLOT_PATH, dpi=180)
    plt.close(figure)
    print(f"Wrote {SPARAMETER_PLOT_PATH}")


def plot_far_field_3d(far_field, frequency_index):
    """Write a normalized 3D directivity pattern at the selected frequency."""

    values = far_field["fields"]["directivity_dbi"][frequency_index]
    theta_axis, phi_axis, directivity = angular_grid(
        values,
        far_field["theta"],
        far_field["phi"],
    )
    theta_grid, phi_grid = np.meshgrid(
        np.deg2rad(theta_axis),
        np.deg2rad(phi_axis),
        indexing="ij",
    )
    peak = float(np.nanmax(directivity))
    radius = 10 ** ((directivity - peak) / 20)
    x = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = radius * np.cos(theta_grid)

    colour_min = max(float(np.nanmin(directivity)), peak - 30)
    normalizer = colors.Normalize(vmin=colour_min, vmax=peak)
    colormap = plt.get_cmap("viridis")

    figure = plt.figure(figsize=(8.2, 7.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        x,
        y,
        z,
        facecolors=colormap(normalizer(directivity)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    frequency_ghz = far_field["frequencies"][frequency_index] * 1e-9
    axis.set_title(f"Normalized 3D directivity pattern at {frequency_ghz:g} GHz\n" f"Peak directivity = {peak:.2f} dBi")
    colourbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalizer, cmap=colormap),
        ax=axis,
        shrink=0.72,
        pad=0.08,
    )
    colourbar.set_label("Directivity (dBi)")
    figure.savefig(FAR_FIELD_3D_PLOT_PATH, dpi=180)
    plt.close(figure)
    print(f"Wrote {FAR_FIELD_3D_PLOT_PATH}")


def plot_principal_planes(far_field, frequency_index):
    """Write directivity, gain, and realized-gain E- and H-plane cuts."""

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)
    plane_specs = (
        (axes[0], xz_plane, "E-plane: xz cut"),
        (axes[1], xy_plane, "H-plane: xy cut"),
    )
    for axis, cut_function, title in plane_specs:
        for quantity, label, style in FAR_FIELD_QUANTITIES:
            angle, values = cut_function(
                far_field["fields"][quantity][frequency_index],
                far_field["theta"],
                far_field["phi"],
            )
            axis.plot(angle, values, style, linewidth=2, label=label)
        axis.set_title(title)
        axis.set_xlabel("Signed angle from +x (degrees)")
        axis.set_ylabel("Level (dBi)")
        axis.set_xlim(-180, 180)
        axis.grid(True, alpha=0.3)
        axis.legend()

    frequency_ghz = far_field["frequencies"][frequency_index] * 1e-9
    figure.suptitle(f"Principal-plane patterns at {frequency_ghz:g} GHz")
    figure.savefig(PRINCIPAL_PLANES_PLOT_PATH, dpi=180)
    plt.close(figure)
    print(f"Wrote {PRINCIPAL_PLANES_PLOT_PATH}")


def main():
    traces = read_sparameters(OUTPUT_STEM)
    far_field = read_far_fields(OUTPUT_STEM.with_suffix(".h5"))
    design_index = int(np.argmin(np.abs(far_field["frequencies"] - DESIGN_FREQUENCY)))

    plot_sparameters(traces)
    plot_far_field_3d(far_field, design_index)
    plot_principal_planes(far_field, design_index)


if __name__ == "__main__":
    main()
