"""Example 2 - curved 2D dielectric waveguide eigenmode ports."""

import argparse
from pathlib import Path

import gprMax

INF = float("inf")
OUTPUT = Path(__file__).resolve().with_suffix("")


def build_scene():
    """Create the geometry, modal ports, and requested outputs."""
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Example 2 - curved 2D dielectric waveguide eigenmode ports"))

    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.195, 0.165, INF)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=6e-9))
    scene.add(gprMax.PMLThickness(thickness=(5, 5, 0, 5, 5, 0)))

    scene.add(gprMax.Material(er=9, se=0, mr=1, sm=0, id="bend_core"))

    # A 20 mm dielectric slab follows a 90 degree annular bend with a 15 mm
    # centreline radius. The cylindrical-sector pair creates the curved core.
    scene.add(gprMax.Box(p1=(0, 0.035, 0), p2=(0.10, 0.055, INF), material_id="bend_core"))
    scene.add(
        gprMax.CylindricalSector(
            normal="z",
            ctr1=0.10,
            ctr2=0.060,
            extent1=0,
            extent2=INF,
            r=0.025,
            start=270,
            end=90,
            material_id="bend_core",
            averaging="y",
        )
    )
    scene.add(
        gprMax.CylindricalSector(
            normal="z",
            ctr1=0.10,
            ctr2=0.060,
            extent1=0,
            extent2=INF,
            r=0.005,
            start=270,
            end=90,
            material_id="free_space",
            averaging="y",
        )
    )
    scene.add(gprMax.Box(p1=(0.105, 0.060, 0), p2=(0.125, 0.165, INF), material_id="bend_core"))

    # Excite mode 1 and monitor the two physical guided modes at both ports.
    # Port 2 is normal to y because the bend rotates propagation through 90
    # degrees.
    scene.add(gprMax.EigenmodeBand(id="eigenmode_band", fmin=4e9, fmax=8e9, points=81))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.02, 0.005, 0),
            p2=(0.02, 0.085, INF),
            direction="+",
            modes=(1, 2),
            anchors="auto",
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=2,
            p1=(0.075, 0.160, 0),
            p2=(0.155, 0.160, INF),
            direction="-",
            modes=(1, 2),
            anchors="auto",
        )
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto"))

    # Save the transient field at selected physical times.
    for time_ps in (400, 700, 1000, 1300, 1600, 1900, 2200, 2500):
        scene.add(
            gprMax.Snapshot(
                p1=(0, 0, 0),
                p2=(0.195, 0.165, INF),
                dl=(0.001, 0.001, 0.001),
                time=time_ps * 1e-12,
                filename=f"curved_waveguide_{time_ps}ps",
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
