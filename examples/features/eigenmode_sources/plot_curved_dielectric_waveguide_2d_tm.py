"""Plot modal S-parameters and time-separated fields for the curved guide."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_dielectric_slab_2d_tm import (
    plot_field_snapshots,
    plot_sparameters,
    read_sparameters,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = EXAMPLE_DIR / "curved_dielectric_waveguide_2d_tm"
SPARAMETER_PLOT_PATH = EXAMPLE_DIR / "curved_dielectric_waveguide_2d_tm_sparameters.png"
FIELD_PLOT_PATH = EXAMPLE_DIR / "curved_dielectric_waveguide_2d_tm_field_propagation.png"


def main():
    traces = read_sparameters(OUTPUT_STEM)
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    plot_sparameters(axis, traces)
    axis.set_title("Curved dielectric waveguide: reflection and mode conversion")
    figure.savefig(SPARAMETER_PLOT_PATH, dpi=180)
    plt.close(figure)
    print(f"Wrote {SPARAMETER_PLOT_PATH}")
    plot_field_snapshots(
        OUTPUT_STEM,
        FIELD_PLOT_PATH,
        "Curved dielectric waveguide: Ez propagation",
        maximum_time_ns=2.5,
    )


if __name__ == "__main__":
    main()
