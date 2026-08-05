import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLANE_AXES = {
    "xy": (("x", 0), ("y", 1)),
    "xz": (("x", 0), ("z", 2)),
    "yz": (("y", 1), ("z", 2)),
}


def declared_snapshot_names(root):
    """Return snapshot files requested by the current case input."""

    names = set()
    for input_path in sorted(root.glob("*.in")):
        for line in input_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("#snapshot:"):
                continue
            name = stripped.split()[-1]
            if name.endswith(".h5"):
                names.add(name)
    return names


def snapshot_paths(root, plane):
    """Return current declared snapshots, excluding stale generated files."""

    snap_dir = root / f"{root.name}_snaps"
    paths = sorted(snap_dir.glob(f"{plane}_center_*.h5"))
    declared = declared_snapshot_names(root)
    if declared:
        paths = [path for path in paths if path.name in declared]
    return paths


def read_eabs(path, plane):
    with h5py.File(path, "r") as handle:
        fields = [
            np.asarray(handle[name], dtype=np.float64)
            for name in ("Ex", "Ey", "Ez")
            if name in handle
        ]
        if not fields:
            raise ValueError(f"No electric field components found in {path}")
        eabs = np.sqrt(sum(field**2 for field in fields))
        time = float(handle.attrs["time"])
        spacing = tuple(float(v) for v in handle.attrs["dx_dy_dz"])
    plane_axes = tuple(axis for _, axis in PLANE_AXES[plane])
    slice_axis = next(axis for axis in range(3) if axis not in plane_axes)
    if eabs.ndim == 3:
        # Ordinary 3D centre-plane snapshots are one sample thick on this
        # axis. A TE 2D model is two internal cells thick, so select its
        # interior-side cell instead of passing a 3D array to imshow.
        eabs = np.take(eabs, eabs.shape[slice_axis] // 2, axis=slice_axis)
    return np.squeeze(eabs), time, spacing


def plot_family(root, plane, decibels=False):
    snap_dir = root / f"{root.name}_snaps"
    paths = snapshot_paths(root, plane)
    if not paths:
        return None

    data = [read_eabs(path, plane) for path in paths]
    absolute_max = max(np.nanmax(field) for field, _, _ in data)
    if not np.isfinite(absolute_max) or absolute_max <= 0:
        raise ValueError(f"Cannot plot {snap_dir}: the electric-field magnitude is zero.")
    if decibels:
        floor = 10 ** (-50 / 20)
        data = [
            (
                20 * np.log10(np.maximum(field / absolute_max, floor)),
                time,
                spacing,
            )
            for field, time, spacing in data
        ]
        vmin, vmax = -50, 0
        colorbar_label = "|E| / global max (dB)"
    else:
        vmin, vmax = 0, absolute_max
        colorbar_label = "|E|"
    ncols = 2
    nrows = int(np.ceil(len(data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    (label0, spacing0), (label1, spacing1) = PLANE_AXES[plane]

    for ax, (field, time, spacing), path in zip(axes, data, paths):
        extent = [
            0,
            spacing[spacing0] * (field.shape[0] - 1),
            0,
            spacing[spacing1] * (field.shape[1] - 1),
        ]
        image = ax.imshow(
            field.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{path.stem}, t={time * 1e9:.2f} ns")
        ax.set_xlabel(f"{label0} (m)")
        ax.set_ylabel(f"{label1} (m)")
        fig.colorbar(image, ax=ax, label=colorbar_label)

    for ax in axes[len(data) :]:
        ax.axis("off")

    suffix = "_dB" if decibels else ""
    output = root / f"{plane}_center_snapshots_Eabs{suffix}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_snapshots.py <case_dir> [<case_dir> ...]")

    for arg in sys.argv[1:]:
        root = Path(arg).resolve()
        for plane in PLANE_AXES:
            for decibels in (False, True):
                output = plot_family(root, plane, decibels=decibels)
                if output is not None:
                    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
