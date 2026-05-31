from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_snapshot(path: Path, field: str) -> tuple[np.ndarray, float, tuple[float, float, float]]:
    with h5py.File(path, "r") as h5:
        if field not in h5:
            available = ", ".join(h5.keys())
            raise KeyError(f"{field!r} is not in {path.name}. Available fields: {available}")

        data = np.asarray(h5[field])
        time = float(h5.attrs["time"])
        dx, dy, dz = (float(v) for v in h5.attrs["dx_dy_dz"])

    return data, time, (dx, dy, dz)


def middle_z_slice(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D snapshot array, got shape {data.shape}")
    return data[:, :, data.shape[2] // 2].T


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot gprMax HDF5 field snapshots.")
    parser.add_argument(
        "snapshot_dir",
        nargs="?",
        default=base_dir / "simple_test_snaps",
        type=Path,
        help="Directory containing snapshot_*.h5 files.",
    )
    parser.add_argument("--field", default="Ez", help="Field component to plot, e.g. Ex, Ey, Ez, Hx, Hy, Hz.")
    parser.add_argument("--output", default=base_dir / "simple_test_Ez_snapshots.png", type=Path)
    args = parser.parse_args()

    snapshot_paths = sorted(args.snapshot_dir.glob("snapshot_*.h5"))
    if not snapshot_paths:
        raise FileNotFoundError(f"No snapshot_*.h5 files found in {args.snapshot_dir}")

    loaded = [load_snapshot(path, args.field) for path in snapshot_paths]
    vmax = max(float(np.max(np.abs(data))) for data, _, _ in loaded) or 1.0

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for ax, path, (data, time, (dx, dy, _)) in zip(axes.flat, snapshot_paths, loaded):
        image = middle_z_slice(data)
        extent = (0.0, data.shape[0] * dx, 0.0, data.shape[1] * dy)
        im = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(f"{path.stem}: {time * 1e9:.2f} ns")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    fig.colorbar(im, ax=axes, shrink=0.9, label=args.field)
    fig.suptitle(f"{args.field} snapshots, middle z slice")
    fig.savefig(args.output, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
