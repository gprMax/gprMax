"""Example 4 - complete dominant-mode microstrip S matrix."""

import argparse
from pathlib import Path

import gprMax

OUTPUT = Path(__file__).resolve().with_suffix("")


def build_scene():
    """Create the geometry, modal ports, and requested outputs."""
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Example 4 - complete dominant-mode microstrip S matrix"))

    scene.add(gprMax.Domain(p1=(0.080, 0.030, 0.025)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=2e-9))
    scene.add(gprMax.PMLThickness(thickness=(8, 0, 0, 8, 0, 0)))

    # A lossless microstrip with a centred 2 mm series gap connects two internal
    # reference planes. The 2 mm substrate has relative permittivity 4, the strip
    # is 4 mm wide, and air surrounds its fringing fields.
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="substrate"))
    scene.add(gprMax.Box(p1=(0, 0.004, 0.006), p2=(0.080, 0.026, 0.007), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0.004, 0.007), p2=(0.080, 0.026, 0.009), material_id="substrate"))
    scene.add(gprMax.Box(p1=(0, 0.013, 0.009), p2=(0.039, 0.017, 0.010), material_id="pec"))
    scene.add(gprMax.Box(p1=(0.041, 0.013, 0.009), p2=(0.080, 0.017, 0.010), material_id="pec"))

    # Only the dominant quasi-TEM mode is retained at each port. Identical
    # cross-sections and inward-pointing directions give compatible reciprocal
    # references. Each uniform feed continues through its x-directed PML to
    # provide a matched termination; the gap remains between the reference planes.
    scene.add(gprMax.EigenmodeBand(id="matrix_band", fmin=4e9, fmax=8e9, points=101))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.008, 0.004, 0.006),
            p2=(0.008, 0.026, 0.018),
            direction="+",
            modes=(1,),
            anchors="auto",
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=2,
            p1=(0.072, 0.004, 0.006),
            p2=(0.072, 0.026, 0.018),
            direction="-",
            modes=(1,),
            anchors="auto",
        )
    )

    # The study changes this one reusable source to each declared channel. The
    # modal anchor solutions and geometry are built once, then two FDTD cases
    # assemble the complete 2 x 2 dominant-mode S matrix. For this reciprocal
    # structure, S21 and S12 should agree within numerical error.
    excitation = gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto")
    scene.add(excitation)
    study = gprMax.EigenmodeStudy(
        [
            gprMax.StudyCase(
                f"drive_port{port}_mode1", [gprMax.ObjectState(excitation, port=port, mode=1)]
            )
            for port in (1, 2)
        ]
    )

    scene.add(
        gprMax.GeometryView(
            p1=(0, 0.002, 0.004),
            p2=(0.080, 0.028, 0.020),
            dl=(0.001, 0.001, 0.001),
            filename="microstrip_gap",
            output_type="n",
        )
    )
    return scene, study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-only", action="store_true", help="solve and plot modes without time stepping"
    )
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="output path without .h5")
    parser.add_argument(
        "--restart", type=int, default=1, help="first study case to run (one-based)"
    )
    args = parser.parse_args()
    options = {"geometry_only": args.geometry_only}
    if args.gpu is not None:
        options.update(gpu=[args.gpu], gpu_precision="single")
    scene, study = build_scene()
    # Inspect the first source without starting the reusable time-domain study.
    if not args.geometry_only:
        options.update(study=study, i=args.restart)
    gprMax.run(scenes=[scene], outputfile=args.output, **options)


if __name__ == "__main__":
    main()
