"""Example 7 - physically labelled circular-waveguide TE11 polarizations."""

import argparse
from pathlib import Path

import gprMax

EXAMPLE_DIR = Path(__file__).resolve().parent


def build_scene(mode=1):
    """Launch mode 1 (global y) or mode 2 (global -x) into a circular guide."""
    if mode not in (1, 2):
        raise ValueError("Choose TE11 mode 1 (vertical/y) or mode 2 (horizontal/-x).")
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Example 7 - physically aligned circular TE11 modes"))
    scene.add(gprMax.Domain(p1=(0.016, 0.016, 0.072)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=3e-9))
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.PMLThickness(thickness=(0, 0, 8, 0, 0, 8)))

    # A 6 mm radius air-filled bore, extruded uniformly through the z PMLs.
    # Equal transverse spacings and a centred circle preserve the TE11 pair.
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.016, 0.016, 0.072), material_id="pec"))
    scene.add(
        gprMax.Cylinder(
            p1=(0.008, 0.008, 0),
            p2=(0.008, 0.008, 0.072),
            r=0.006,
            material_id="free_space",
            # Preserve tangential PEC samples shared with the surrounding wall.
            averaging="y",
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
                plot_fields=True,
                # These two arguments give both ports the same physical basis
                # at EVERY frequency anchor, despite raw eigensolver rotations.
                degenerate=(1, 2),
                mode_polarizations="y",
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
    # Change only mode=1 to mode=2 to launch the other linear polarization.
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
        help="1: vertical/global y; 2: horizontal/global -x",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="solve and plot both labelled modes without time stepping",
    )
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument("--output", type=Path, help="output path without .h5")
    args = parser.parse_args()
    output = args.output or EXAMPLE_DIR / f"circular_te11_mode{args.mode}"
    options = {"geometry_only": args.geometry_only, "cpu_precision": "double"}
    if args.gpu is not None:
        options.update(gpu=[args.gpu], gpu_precision="single")
    gprMax.run(scenes=[build_scene(mode=args.mode)], outputfile=output, **options)


if __name__ == "__main__":
    main()
