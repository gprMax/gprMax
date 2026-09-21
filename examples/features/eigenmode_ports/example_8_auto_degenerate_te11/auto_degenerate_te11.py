"""Example 8 - opt-in automatic tracking of the degenerate circular TE11 pair.

Automatic mode tracking is under development. Inspect the geometry-only
dispersion and field plots before using the tracked profiles in a simulation.
"""

import argparse
from pathlib import Path

import gprMax

EXAMPLE_DIR = Path(__file__).resolve().parent


def build_scene(mode=1):
    """Build a circular guide without a manually declared degenerate group."""
    if mode not in (1, 2):
        raise ValueError("Choose automatically oriented TE11 mode 1 or mode 2.")

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Example 8 - automatically detected circular TE11 pair"))
    scene.add(gprMax.Domain(p1=(0.016, 0.016, 0.072)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=3e-9))
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.PMLThickness(thickness=(0, 0, 8, 0, 0, 8)))

    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.016, 0.016, 0.072), material_id="pec"))
    scene.add(
        gprMax.Cylinder(
            p1=(0.008, 0.008, 0),
            p2=(0.008, 0.008, 0.072),
            r=0.006,
            material_id="free_space",
            averaging="n",
        )
    )
    scene.add(gprMax.EigenmodeBand(id="te11", fmin=20e9, fmax=24e9, points=41))
    for port, z, direction in ((1, 0.024, "+"), (2, 0.064, "-")):
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(0.001, 0.001, z),
                p2=(0.015, 0.015, z),
                direction=direction,
                modes=(1, 2),
                anchors="auto",
                tracking="auto",
                verification="fast",
                plot_fields=True,
                # Both degenerate and mode_polarizations are intentionally
                # omitted. Each port detects the pair and defaults to the
                # same global x/y axes.
            )
        )
    scene.add(
        gprMax.VirtualWaveguide(
            port=1,
            length_cells=24,
            pml_cells=8,
            source_clearance_cells=4,
        )
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=mode, waveform="auto"))
    scene.add(gprMax.Rx(p1=(0.008, 0.008, 0.040), id="centre"))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=int,
        choices=(1, 2),
        default=1,
        help="1: first transverse axis (x); 2: second transverse axis (y)",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="solve and plot the tracked dispersion and modal fields",
    )
    parser.add_argument("--output", type=Path, help="output path without .h5")
    args = parser.parse_args()
    output = args.output or EXAMPLE_DIR / f"auto_degenerate_te11_mode{args.mode}"
    gprMax.run(
        scenes=[build_scene(mode=args.mode)],
        outputfile=output,
        geometry_only=args.geometry_only,
        cpu_precision="double",
    )


if __name__ == "__main__":
    main()
