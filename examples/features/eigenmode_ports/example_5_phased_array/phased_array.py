"""Example 5 - four-element phase-steered waveguide array."""

import argparse
from pathlib import Path

import gprMax

OUTPUT = Path(__file__).resolve().with_suffix("")


def build_scene():
    """Create the geometry, modal ports, and requested outputs."""
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Example 5 - four-element phase-steered waveguide array"))

    scene.add(gprMax.Domain(p1=(0.100, 0.104, 0.060)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=4e-9))
    scene.add(gprMax.PMLThickness(thickness=8))

    # Four 10 mm by 24 mm apertures, spaced 18 mm along y, radiate in +x.
    for centre_mm in (25, 43, 61, 79):
        y0, y1 = (centre_mm - 5) * 1e-3, (centre_mm + 5) * 1e-3
        for p1, p2 in (
            ((0.018, y0 - 0.001, 0.017), (0.045, y0, 0.043)),
            ((0.018, y1, 0.017), (0.045, y1 + 0.001, 0.043)),
            ((0.018, y0 - 0.001, 0.017), (0.045, y1 + 0.001, 0.018)),
            ((0.018, y0 - 0.001, 0.042), (0.045, y1 + 0.001, 0.043)),
        ):
            scene.add(gprMax.Box(p1=p1, p2=p2, material_id="pec"))

    # The ten uniform S-parameter bins do not land on 9, 10, or 11 GHz, so add
    # those exact frequencies for the five-bin NTFF request below.
    scene.add(
        gprMax.EigenmodeBand(
            id="array_band", fmin=8e9, fmax=12e9, points=10, frequencies=(9e9, 10e9, 11e9)
        )
    )
    for port, centre_mm in enumerate((25, 43, 61, 79), start=1):
        y0, y1 = (centre_mm - 5) * 1e-3, (centre_mm + 5) * 1e-3
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(0.018, y0, 0.018),
                p2=(0.018, y1, 0.042),
                direction="+",
                modes=(1,),
                anchors=(10e9,),
            )
        )
        scene.add(
            gprMax.VirtualWaveguide(
                port=port, length_cells=30, pml_cells=12, source_clearance_cells=6
            )
        )
        # Constant phase progression gives beam squint across the band.
        scene.add(
            gprMax.EigenmodeExcitation(
                port=port,
                mode=1,
                waveform="auto",
                amplitude=1,
                phase_deg=-108 * (port - 1),
                delay_s=0,
            )
        )

    # The virtual feeds leave enough air for a closed six-face NTFF surface.
    scene.add(
        gprMax.NTFFSurface(p1=(0.010, 0.010, 0.010), p2=(0.088, 0.094, 0.050), id="array_surface")
    )
    scene.add(
        gprMax.NTFFFrequencyTransform(
            surface_id="array_surface",
            id="array_band",
            frequencies=(8e9, 9e9, 10e9, 11e9, 12e9),
            window="rectangular",
        )
    )
    scene.add(
        gprMax.NTFFAntennaPorts(
            transform_id="array_band", port_ids=("port1", "port2", "port3", "port4")
        )
    )
    # Store only the dense theta=90 degree xy-plane cut used by the plot. Gain,
    # directivity, and efficiency still use gprMax's internal full-sphere
    # quadrature; that temporary normalization grid is not written to HDF5.
    scene.add(
        gprMax.NTFFFarFieldArray(
            theta_start=90,
            theta_stop=90,
            theta_step=1,
            phi_start=0,
            phi_stop=359,
            phi_step=1,
            transform_id="array_band",
            id="array_pattern",
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
            p2=(0.100, 0.104, 0.060),
            dl=(0.001, 0.001, 0.001),
            filename="phased_array",
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
