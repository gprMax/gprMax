"""Calculate and plot the full-sphere gain of the rectangular patch example."""

import argparse
import csv
import json
import logging
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import patch_antenna_gprmax as patch

import gprMax

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

ANGULAR_STEP = 2.0
GAIN_DYNAMIC_RANGE_DB = 30.0
OUTPUT_STEM = "patch_antenna_3d_gain_single_feed"
FAR_FIELD_ID = "gain_3d"


def build_gain_scene():
    """Add one physical port and a full-sphere gain request to the patch."""

    patch._configure_mesh("standard")
    scene, _ = patch.build_scene(feed_mode="single", mesh_mode="standard")
    feed_port_id, _ = patch._feed_edges("single")[0]
    scene.add(gprMax.KSIRAntennaPorts("patch_spectrum", (feed_port_id,)))
    scene.add(
        gprMax.KSIRFarFieldArray(
            theta_start=0,
            theta_stop=180,
            theta_step=ANGULAR_STEP,
            phi_start=0,
            phi_stop=360 - ANGULAR_STEP,
            phi_step=ANGULAR_STEP,
            transform_id="patch_spectrum",
            id=FAR_FIELD_ID,
            outputs=(
                "radiation_intensity",
                "gain",
                "gain_dbi",
                "realized_gain",
                "realized_gain_dbi",
                "directivity",
                "directivity_dbi",
                "radiation_efficiency",
                "total_efficiency",
            ),
        )
    )
    return scene


def read_gain(h5_path):
    """Read and validate the persisted full-sphere antenna metrics."""

    group_path = "ntff/patch_surface/frequency/patch_spectrum/far_field/" + FAR_FIELD_ID
    with h5py.File(h5_path, "r") as output:
        group = output[group_path]
        theta = np.asarray(group["theta"], dtype=np.float64)
        phi = np.asarray(group["phi"], dtype=np.float64)
        gain = np.asarray(group["fields/gain"][0], dtype=np.float64)
        gain_dbi = np.asarray(group["fields/gain_dbi"][0], dtype=np.float64)
        radiation_intensity = np.asarray(group["fields/radiation_intensity"][0], dtype=np.float64)
        directivity = np.asarray(group["fields/directivity"][0], dtype=np.float64)
        directivity_dbi = np.asarray(group["fields/directivity_dbi"][0], dtype=np.float64)
        realized_gain = np.asarray(group["fields/realized_gain"][0], dtype=np.float64)
        realized_gain_dbi = np.asarray(group["fields/realized_gain_dbi"][0], dtype=np.float64)
        radiation_efficiency = float(group["fields/radiation_efficiency"][0])
        total_efficiency = float(group["fields/total_efficiency"][0])
        radiated_power = float(group["radiated_power"][0])
        maximum_directivity = float(group["maximum_directivity"][0])
        accepted_power = float(group["port_power/accepted_power"][0])
        incident_power = float(group["port_power/incident_power"][0])

    expected_size = int(180 / ANGULAR_STEP + 1) * int(360 / ANGULAR_STEP)
    arrays = (
        theta,
        phi,
        gain,
        gain_dbi,
        radiation_intensity,
        realized_gain,
        realized_gain_dbi,
        directivity,
        directivity_dbi,
    )
    if any(values.shape != (expected_size,) for values in arrays):
        raise ValueError("The persisted 3-D gain arrays have unexpected dimensions")
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("The persisted 3-D gain arrays contain non-finite values")
    if not np.isclose(
        radiation_efficiency,
        radiated_power / accepted_power,
        rtol=5e-6,
    ):
        raise ValueError("The HDF5 radiation-efficiency power balance is inconsistent")
    return {
        "theta": theta,
        "phi": phi,
        "gain": gain,
        "gain_dbi": gain_dbi,
        "radiation_intensity": radiation_intensity,
        "realized_gain": realized_gain,
        "realized_gain_dbi": realized_gain_dbi,
        "directivity": directivity,
        "directivity_dbi": directivity_dbi,
        "radiation_efficiency": radiation_efficiency,
        "total_efficiency": total_efficiency,
        "radiated_power": radiated_power,
        "maximum_directivity": maximum_directivity,
        "accepted_power": accepted_power,
        "incident_power": incident_power,
    }


def write_csv(result, destination):
    """Write the full angular grid to a portable table."""

    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "theta_deg",
                "phi_deg",
                "gain_linear",
                "gain_dbi",
                "radiation_intensity_w_s2",
                "realized_gain_linear",
                "realized_gain_dbi",
                "directivity_linear",
                "directivity_dbi",
            )
        )
        writer.writerows(
            zip(
                result["theta"],
                result["phi"],
                result["gain"],
                result["gain_dbi"],
                result["radiation_intensity"],
                result["realized_gain"],
                result["realized_gain_dbi"],
                result["directivity"],
                result["directivity_dbi"],
            )
        )


def plot_gain(result, destination):
    """Plot a dB-scaled radial surface coloured by absolute gain."""

    theta_axis = np.arange(0, 180 + ANGULAR_STEP, ANGULAR_STEP)
    phi_axis = np.arange(0, 360, ANGULAR_STEP)
    shape = (theta_axis.size, phi_axis.size)
    gain_dbi = result["gain_dbi"].reshape(shape)

    # Close the periodic seam for rendering only. The persisted angular grid
    # correctly excludes the duplicate phi=360 degree samples.
    phi_closed = np.append(phi_axis, 360.0)
    gain_closed = np.concatenate((gain_dbi, gain_dbi[:, :1]), axis=1)
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_closed, indexing="ij")
    peak_gain_dbi = float(np.max(gain_closed))
    floor_dbi = peak_gain_dbi - GAIN_DYNAMIC_RANGE_DB
    radial_gain = np.clip(gain_closed - floor_dbi, 0, None) / GAIN_DYNAMIC_RANGE_DB
    theta_rad = np.deg2rad(theta_grid)
    phi_rad = np.deg2rad(phi_grid)
    x = radial_gain * np.sin(theta_rad) * np.cos(phi_rad)
    y = radial_gain * np.sin(theta_rad) * np.sin(phi_rad)
    z = radial_gain * np.cos(theta_rad)

    colour_norm = colors.Normalize(vmin=floor_dbi, vmax=peak_gain_dbi)
    colour_map = plt.get_cmap("viridis")
    face_colours = colour_map(colour_norm(np.maximum(gain_closed, floor_dbi)))

    figure = plt.figure(figsize=(9.2, 8.0), constrained_layout=True)
    axes = figure.add_subplot(111, projection="3d")
    axes.plot_surface(
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
    colour_bar = figure.colorbar(scalar_map, ax=axes, shrink=0.70, pad=0.08)
    colour_bar.set_label("Gain (dBi)")

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z (patch broadside)")
    axes.set_box_aspect((1, 1, 1))
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    axes.set_zlim(-1, 1)
    axes.view_init(elev=24, azim=-42)
    axes.set_title(
        "gprMax rectangular patch: 3-D gain at 2.37 GHz\n"
        f"peak {peak_gain_dbi:.2f} dBi; radial scale clipped at "
        f"{GAIN_DYNAMIC_RANGE_DB:.0f} dB"
    )
    figure.savefig(destination, dpi=220)
    plt.close(figure)
    return peak_gain_dbi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="read an existing HDF5 result without running gprMax",
    )
    args = parser.parse_args()

    patch.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_base = patch.RESULTS_DIR / OUTPUT_STEM
    if not args.postprocess_only:
        scene = build_gain_scene()
        options = {}
        if args.gpu is not None:
            options.update(gpu=[args.gpu], gpu_precision="single")
        gprMax.run(
            scenes=[scene],
            outputfile=output_base,
            hide_progress_bars=False,
            log_level=logging.INFO,
            **options,
        )

    h5_path = output_base.with_suffix(".h5")
    result = read_gain(h5_path)
    csv_path = output_base.with_suffix(".csv")
    png_path = output_base.with_suffix(".png")
    metrics_path = output_base.with_name(output_base.name + "_metrics.json")
    write_csv(result, csv_path)
    peak_gain_dbi = plot_gain(result, png_path)
    metrics = {
        "frequency_hz": patch.FREQUENCY,
        "feed": "single 50 Ohm voltage-source gap",
        "angular_step_deg": ANGULAR_STEP,
        "peak_gain_dbi_on_requested_grid": peak_gain_dbi,
        "maximum_directivity": result["maximum_directivity"],
        "maximum_directivity_dbi": 10 * np.log10(result["maximum_directivity"]),
        "radiation_efficiency": result["radiation_efficiency"],
        "mismatch_efficiency": result["accepted_power"] / result["incident_power"],
        "total_efficiency": result["total_efficiency"],
        "peak_realized_gain_dbi_on_requested_grid": float(np.max(result["realized_gain_dbi"])),
        "radiated_spectral_power": result["radiated_power"],
        "accepted_spectral_power": result["accepted_power"],
        "incident_spectral_power": result["incident_power"],
        "source_hdf5": h5_path.name,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved 3-D gain plot to {png_path}")
    print(f"Saved angular gain data to {csv_path}")


if __name__ == "__main__":
    main()
