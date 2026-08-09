"""Plot S-parameters and time-separated fields for the straight waveguide."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = EXAMPLE_DIR / "straight_waveguide"
SPARAMETER_PLOT_PATH = EXAMPLE_DIR / "straight_waveguide_sparameters.png"
FIELD_PLOT_PATH = EXAMPLE_DIR / "straight_waveguide_field_propagation.png"
PLOT_FLOOR_DB = -100.0


def read_sparameters(stem: Path):
    """Return valid CSV rows grouped by source/destination port and mode."""

    path = stem.with_name(stem.name + "_sparameters.csv")
    traces = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not bool(int(row["valid"])):
                continue
            key = (
                int(row["source_port"]),
                int(row["source_mode"]),
                int(row["destination_port"]),
                int(row["destination_mode"]),
            )
            traces[key].append((float(row["frequency_hz"]) * 1e-9, float(row["S_magnitude_db"])))
    if not traces:
        raise ValueError(f"No valid S-parameter rows found in {path}")
    return {key: np.asarray(sorted(values), dtype=np.float64) for key, values in traces.items()}


def plot_sparameters(axis, traces):
    """Plot all valid source/destination port and mode combinations."""

    for (
        source_port,
        source_mode,
        destination_port,
        destination_mode,
    ), data in sorted(traces.items()):
        axis.plot(
            data[:, 0],
            np.maximum(data[:, 1], PLOT_FLOOR_DB),
            marker="o",
            markersize=4,
            linewidth=2,
            label=(f"S{destination_port}{source_port}, " f"mode {destination_mode}<-{source_mode}"),
        )
    axis.set_title("Eigenmode S-parameters")
    axis.set_xlabel("Frequency (GHz)")
    axis.set_ylabel(f"Magnitude (dB; floor {PLOT_FLOOR_DB:g} dB)")
    axis.set_ylim(PLOT_FLOOR_DB, 5)
    axis.grid(True, alpha=0.3)
    axis.legend()


def read_field_snapshots(
    stem: Path,
    field: str = "Ez",
    maximum_time_ns: float | None = None,
):
    """Read adjacent 2D snapshots in physical-time order, optionally capped."""

    snapshot_dir = stem.with_name(stem.name + "_snaps")
    paths = list(snapshot_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No snapshots found in {snapshot_dir}; run the model before plotting")

    snapshots = []
    for path in paths:
        with h5py.File(path, "r") as output:
            if field not in output:
                raise KeyError(f"Snapshot {path} does not contain field {field!r}")
            values = np.squeeze(output[field][...])
            if values.ndim != 2:
                raise ValueError(f"Snapshot {path} field {field!r} is not two-dimensional")
            spacing = np.asarray(output.attrs["dx_dy_dz"], dtype=np.float64)
            extent = (
                0.0,
                values.shape[0] * spacing[0] * 1e3,
                0.0,
                values.shape[1] * spacing[1] * 1e3,
            )
            snapshots.append((float(output.attrs["time"]) * 1e9, values.T, extent))
    snapshots.sort(key=lambda snapshot: snapshot[0])
    if maximum_time_ns is not None:
        snapshots = [snapshot for snapshot in snapshots if snapshot[0] <= maximum_time_ns + 0.01]
    if not snapshots:
        raise ValueError(f"No snapshots at or before {maximum_time_ns:g} ns in {snapshot_dir}")
    return snapshots


def plot_field_snapshots(
    stem: Path,
    plot_path: Path,
    title: str,
    field: str = "Ez",
    maximum_time_ns: float | None = None,
):
    """Write one common-scale panel for each requested snapshot time."""

    snapshots = read_field_snapshots(stem, field, maximum_time_ns)
    limit = max(float(np.max(np.abs(values))) for _, values, _ in snapshots)
    if not np.isfinite(limit) or limit == 0:
        raise ValueError(f"Snapshots contain no finite non-zero {field} values")

    columns = len(snapshots) if len(snapshots) <= 4 else min(4, (len(snapshots) + 1) // 2)
    rows = int(np.ceil(len(snapshots) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    active_axes = []
    for axis, (time_ns, values, extent) in zip(axes.flat, snapshots):
        active_axes.append(axis)
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(f"t = {time_ns:.2f} ns")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    for axis in axes.flat[len(snapshots) :]:
        axis.set_visible(False)
    figure.colorbar(image, ax=active_axes, label=f"{field} (V/m)")
    figure.suptitle(title)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {plot_path}")


def main():
    traces = read_sparameters(OUTPUT_STEM)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    plot_sparameters(axis, traces)
    figure.suptitle("Straight 2D dielectric-slab waveguide")
    figure.savefig(SPARAMETER_PLOT_PATH, dpi=180)
    plt.close(figure)
    print(f"Wrote {SPARAMETER_PLOT_PATH}")
    plot_field_snapshots(
        OUTPUT_STEM,
        FIELD_PLOT_PATH,
        "Straight dielectric waveguide: Ez propagation",
    )


if __name__ == "__main__":
    main()
