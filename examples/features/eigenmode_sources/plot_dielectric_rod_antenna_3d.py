"""Plot S11 and multi-frequency far fields for the dielectric-rod antenna."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_dielectric_slab_2d_tm import plot_sparameters, read_sparameters


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = EXAMPLE_DIR / "dielectric_rod_antenna_3d"
PLOT_PATH = EXAMPLE_DIR / "dielectric_rod_antenna_3d_results.png"
FAR_FIELD_GROUP = "ntff/radiation_surface/frequency/antenna_band/far_field/full_sphere"
DESIGN_FREQUENCY = 7e9
FAR_FIELD_QUANTITIES = (
    ("directivity_dbi", "Directivity", "-"),
    ("gain_dbi", "Gain", "--"),
    ("realized_gain_dbi", "Realized gain", ":"),
)


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
                quantity: np.asarray(group[f"fields/{quantity}"], dtype=np.float64)
                for quantity, _, _ in FAR_FIELD_QUANTITIES
            },
        }


def principal_plane(values, theta, phi):
    """Return the phi=0/180 cut on a signed angle axis."""

    theta_axis = np.unique(theta)
    phi_axis = np.unique(phi)
    grid = np.asarray(values).reshape(theta_axis.size, phi_axis.size)
    phi_zero = int(np.argmin(np.abs(phi_axis)))
    phi_opposite = int(np.argmin(np.abs(phi_axis - 180)))
    angle = np.concatenate((-theta_axis[:0:-1], theta_axis))
    cut = np.concatenate((grid[:0:-1, phi_opposite], grid[:, phi_zero]))
    return angle, cut


def main():
    traces = read_sparameters(OUTPUT_STEM)
    far_field = read_far_fields(OUTPUT_STEM.with_suffix(".h5"))
    frequencies_ghz = far_field["frequencies"] * 1e-9
    design_index = int(np.argmin(np.abs(far_field["frequencies"] - DESIGN_FREQUENCY)))

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 4.8),
        constrained_layout=True,
    )
    plot_sparameters(axes[0], traces)

    for quantity, label, style in FAR_FIELD_QUANTITIES:
        peak = np.nanmax(far_field["fields"][quantity], axis=1)
        axes[1].plot(
            frequencies_ghz,
            peak,
            style,
            marker="o",
            linewidth=2,
            label=label,
        )
    axes[1].set_title("Peak far-field level")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("Peak level (dBi)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for quantity, label, style in FAR_FIELD_QUANTITIES:
        angle, values = principal_plane(
            far_field["fields"][quantity][design_index],
            far_field["theta"],
            far_field["phi"],
        )
        axes[2].plot(angle, values, style, linewidth=2, label=label)
    axes[2].set_title(f"Principal plane at {frequencies_ghz[design_index]:g} GHz")
    axes[2].set_xlabel("Signed angle from +z (degrees)")
    axes[2].set_ylabel("Level (dBi)")
    axes[2].set_xlim(-180, 180)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.suptitle("Eigenmode-fed tapered dielectric-rod antenna")
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)
    print(f"Wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
