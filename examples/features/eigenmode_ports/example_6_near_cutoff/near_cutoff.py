"""Example 6 - rectangular-waveguide S parameters across TE10 cutoff."""

import argparse
from pathlib import Path

import numpy as np

import gprMax

OUTPUT = Path(__file__).resolve().with_suffix("")


def build_scene():
    """Create the geometry, modal ports, and requested outputs."""
    scene = gprMax.Scene()
    scene.add(
        gprMax.Title(name="Example 6 - rectangular-waveguide S parameters across TE10 cutoff")
    )

    scene.add(gprMax.Domain(p1=(0.028, 0.008, 0.006)))
    scene.add(gprMax.Discretisation(p1=(0.0002, 0.0002, 0.0002)))
    scene.add(gprMax.TimeWindow(time=4e-9))
    scene.add(gprMax.PMLThickness(thickness=(12, 0, 0, 12, 0, 0)))

    # The air-filled guide is 6 mm wide and 4 mm high, so its analytical TE10
    # cutoff is c/(2a) = 24.9827048 GHz. Seven of the 100 requested DFT points
    # are below cutoff and the first propagating point is 24.9924242 GHz.
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.028, 0.001, 0.006), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0.007, 0), p2=(0.028, 0.008, 0.006), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.028, 0.008, 0.001), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0, 0.005), p2=(0.028, 0.008, 0.006), material_id="pec"))

    scene.add(gprMax.EigenmodeBand(id="cutoff_band", fmin=24.25e9, fmax=34.75e9, points=100))

    # Near cutoff, beta and modal impedance change rapidly. Put reference
    # anchors at every below-cutoff DFT point, the first nine propagating points,
    # and then about every 0.5 GHz. The extra 15, 20, 40 and 45 GHz candidates
    # cover the automatic waveform's transition spectrum without interpolating
    # the tracked evanescent branch through cutoff.
    dft_frequencies = np.linspace(24.25e9, 34.75e9, 100)
    anchors = tuple(
        np.concatenate(
            (
                [15e9, 20e9],
                dft_frequencies[:16],
                np.arange(26.25e9, 34.75e9 + 0.25e9, 0.5e9),
                [40e9, 45e9],
            )
        )
    )
    for port, x, direction in ((1, 0.004, "+"), (2, 0.016, "-")):
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(x, 0.001, 0.001),
                p2=(x, 0.007, 0.005),
                direction=direction,
                modes=(1,),
                anchors=anchors,
            )
        )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto"))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-only", action="store_true", help="solve and plot modes without time stepping"
    )
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="output path without .h5")
    args = parser.parse_args()
    options = {"geometry_only": args.geometry_only}
    if args.gpu is not None:
        options.update(gpu=[args.gpu], gpu_precision="single")
    gprMax.run(scenes=[build_scene()], outputfile=args.output, **options)


if __name__ == "__main__":
    main()
