from pathlib import Path
import sys

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


def read_eabs(path):
    with h5py.File(path, "r") as handle:
        fields = [np.asarray(handle[name], dtype=np.float64) for name in ("Ex", "Ey", "Ez") if name in handle]
        if not fields:
            raise ValueError(f"No electric field components found in {path}")
        eabs = np.sqrt(sum(field**2 for field in fields))
        time = float(handle.attrs["time"])
        spacing = tuple(float(v) for v in handle.attrs["dx_dy_dz"])
    return np.squeeze(eabs), time, spacing


def plot_family(root, plane):
    snap_dir = root / f"{root.name}_snaps"
    paths = sorted(snap_dir.glob(f"{plane}_center_*.h5"))
    if not paths:
        return None

    data = [read_eabs(path) for path in paths]
    vmax = max(np.nanmax(field) for field, _, _ in data)
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
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(f"{path.stem}, t={time * 1e9:.2f} ns")
        ax.set_xlabel(f"{label0} (m)")
        ax.set_ylabel(f"{label1} (m)")
        fig.colorbar(image, ax=ax, label="|E|")

    for ax in axes[len(data) :]:
        ax.axis("off")

    output = root / f"{plane}_center_snapshots_Eabs.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_eigenmode_direction_snapshots.py <case_dir> [<case_dir> ...]")

    for arg in sys.argv[1:]:
        root = Path(arg).resolve()
        for plane in PLANE_AXES:
            output = plot_family(root, plane)
            if output is not None:
                print(f"Wrote {output}")


if __name__ == "__main__":
    main()
