"""Example 3 - rectangular-waveguide-fed pyramidal horn antenna."""

import argparse
from pathlib import Path

import gprMax

OUTPUT = Path(__file__).resolve().with_suffix("")


def build_scene():
    """Create the geometry, modal ports, and requested outputs."""
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Example 3 - rectangular-waveguide-fed pyramidal horn antenna"))

    scene.add(gprMax.Domain(p1=(0.130, 0.090, 0.070)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=4e-9))
    scene.add(gprMax.PMLThickness(thickness=8))

    # Coordinates below use integer millimetres to keep every step on the mesh.
    def box(p1, p2):
        scene.add(
            gprMax.Box(
                p1=tuple(value * 1e-3 for value in p1),
                p2=tuple(value * 1e-3 for value in p2),
                material_id="pec",
            )
        )

    def plate(p1, p2):
        scene.add(
            gprMax.Plate(
                p1=tuple(value * 1e-3 for value in p1),
                p2=tuple(value * 1e-3 for value in p2),
                material_id="pec",
            )
        )

    def hollow_section(x0, x1, y0, y1, z0, z1):
        # The bounds describe the air aperture; each wall is 1 mm thick.
        box((x0, y0 - 1, z0 - 1), (x1, y0, z1 + 1))
        box((x0, y1, z0 - 1), (x1, y1 + 1, z1 + 1))
        box((x0, y0 - 1, z0 - 1), (x1, y1 + 1, z0))
        box((x0, y0 - 1, z1), (x1, y1 + 1, z1 + 1))

    # The feed starts at the internal port; its continuation is virtual.
    hollow_section(12, 35, 33, 57, 29, 41)
    for section in range(9):
        x0 = 35 + 5 * section
        y0, y1 = 33 - 2 * section, 57 + 2 * section
        z0, z1 = 29 - 2 * section, 41 + 2 * section
        hollow_section(x0, x0 + 5, y0, y1, z0, z1)
        if section < 8:
            # Four annular plates close the expansion to the next section.
            x = x0 + 5
            plate((x, y0 - 2, z0 - 2), (x, y0, z1 + 2))
            plate((x, y1, z0 - 2), (x, y1 + 2, z1 + 2))
            plate((x, y0, z0 - 2), (x, y1, z0))
            plate((x, y0, z1), (x, y1, z1 + 2))

    # Launch the fundamental TE10-like mode of the rectangular feed.
    scene.add(
        gprMax.EigenmodeBand(
            id="eigenmode_band",
            fmin=8e9,
            fmax=12e9,
            points=101,
            frequencies=(8e9, 8.5e9, 9e9, 9.5e9, 10e9, 10.5e9, 11e9, 11.5e9, 12e9),
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.012, 0.033, 0.029),
            p2=(0.012, 0.057, 0.041),
            direction="+",
            modes=(1,),
            anchors="auto",
        )
    )
    scene.add(
        gprMax.VirtualWaveguide(port=1, length_cells=30, pml_cells=12, source_clearance_cells=6)
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto"))

    # The virtual feed permits a closed six-face surface in homogeneous air.
    scene.add(
        gprMax.NTFFSurface(p1=(0.010, 0.011, 0.011), p2=(0.118, 0.079, 0.059), id="horn_surface")
    )
    scene.add(
        gprMax.NTFFFrequencyTransform(
            surface_id="horn_surface",
            id="antenna_band",
            frequencies=(8e9, 8.5e9, 9e9, 9.5e9, 10e9, 10.5e9, 11e9, 11.5e9, 12e9),
            window="rectangular",
        )
    )
    scene.add(gprMax.NTFFAntennaPorts(transform_id="antenna_band", port_ids=("port1",)))
    scene.add(
        gprMax.NTFFFarFieldArray(
            theta_start=0,
            theta_stop=180,
            theta_step=5,
            phi_start=0,
            phi_stop=355,
            phi_step=5,
            transform_id="antenna_band",
            id="full_sphere",
            outputs=(
                "Etheta",
                "Ephi",
                "radiation_intensity",
                "directivity_dbi",
                "gain_dbi",
                "realized_gain_dbi",
                "radiation_efficiency",
                "total_efficiency",
            ),
        )
    )

    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, 0),
            p2=(0.130, 0.090, 0.070),
            dl=(0.001, 0.001, 0.001),
            filename="horn_antenna",
            output_type="n",
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
