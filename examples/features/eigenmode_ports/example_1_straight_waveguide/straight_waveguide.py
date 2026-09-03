"""Example 1 - straight 2D dielectric waveguide eigenmode ports."""

import argparse
from pathlib import Path

import gprMax

INF = float("inf")
OUTPUT = Path(__file__).resolve().with_suffix("")


def build_scene():
    """Create the geometry, modal ports, and requested outputs."""
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Example 1 - straight 2D dielectric waveguide eigenmode ports"))

    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.24, 0.08, INF)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=5e-9))
    scene.add(gprMax.PMLThickness(thickness=(5, 5, 0, 5, 5, 0)))

    scene.add(gprMax.Material(er=9, se=0, mr=1, sm=0, id="slab_core"))

    scene.add(gprMax.Box(p1=(0, 0.03, 0), p2=(0.24, 0.05, INF), material_id="slab_core"))

    # The source aperture contains 25 mm of free space on either side of the
    # 20 mm core so that its evanescent tails decay before reaching the aperture
    # and the transverse PML.
    # Excite mode 1 while monitoring the two physical guided modes at both ports.
    # Requesting more eigenpairs can expose artificial PEC-boundary box modes;
    # inspect every modal profile before adding it to the monitored mode list.
    scene.add(gprMax.EigenmodeBand(id="eigenmode_band", fmin=4e9, fmax=6e9, points=21))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.02, 0.005, 0),
            p2=(0.02, 0.075, INF),
            direction="+",
            modes=(1, 2),
            anchors="auto",
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=2,
            p1=(0.235, 0.005, 0),
            p2=(0.235, 0.075, INF),
            direction="-",
            modes=(1, 2),
            anchors="auto",
        )
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto"))
    scene.add(gprMax.Rx(p1=(0.08, 0.04, INF)))
    scene.add(gprMax.Rx(p1=(0.14, 0.04, INF)))
    scene.add(gprMax.Rx(p1=(0.20, 0.04, INF)))

    # Save the transient field at selected physical times.
    for time_ps in (350, 600, 850, 1100, 1350, 1600, 1900, 2200, 2500, 2800, 3400, 4200):
        scene.add(
            gprMax.Snapshot(
                p1=(0, 0, 0),
                p2=(0.24, 0.08, INF),
                dl=(0.001, 0.001, 0.001),
                time=time_ps * 1e-12,
                filename=f"straight_waveguide_{time_ps}ps",
                fileext=".h5",
            )
        )
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
