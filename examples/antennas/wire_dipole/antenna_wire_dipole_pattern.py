"""Full-sphere directivity and gain of a centre-fed wire dipole."""

import argparse
from pathlib import Path

import gprMax

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "antenna_wire_dipole_pattern"
PATTERN_FREQUENCY = 0.95e9
ANGULAR_STEP = 5.0


def build_scene():
    """Build the dipole, terminal port, and equivalent-current NTFF output."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Wire dipole full-sphere gain pattern"))
    scene.add(gprMax.Domain(p1=(0.080, 0.080, 0.220)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=40e-9))
    scene.add(gprMax.PMLThickness(thickness=10))
    scene.add(
        gprMax.Waveform(
            wave_type="gaussian",
            amp=1,
            freq=1e9,
            id="dipole_pulse",
        )
    )

    # A 150 mm z-directed PEC edge with a one-cell feed gap at its centre.
    scene.add(
        gprMax.Edge(
            p1=(0.040, 0.040, 0.035),
            p2=(0.040, 0.040, 0.185),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Edge(
            p1=(0.040, 0.040, 0.110),
            p2=(0.040, 0.040, 0.111),
            material_id="free_space",
        )
    )
    feed = (0.040, 0.040, 0.110)
    scene.add(
        gprMax.VoltageSource(
            p1=feed,
            polarisation="z",
            resistance=50,
            waveform_id="dipole_pulse",
        )
    )
    scene.add(gprMax.RxPort(p1=feed, id="feed"))

    scene.add(
        gprMax.NTFFSurface(
            p1=(0.018, 0.018, 0.020),
            p2=(0.062, 0.062, 0.200),
            id="dipole_surface",
        )
    )
    scene.add(
        gprMax.NTFFFrequencyTransform(
            surface_id="dipole_surface",
            id="dipole_pattern",
            frequencies=(PATTERN_FREQUENCY,),
            window="rectangular",
            save_surface_dft=False,
        )
    )
    scene.add(
        gprMax.NTFFAntennaPorts(
            transform_id="dipole_pattern",
            port_ids=("feed",),
        )
    )
    scene.add(
        gprMax.NTFFFarFieldArray(
            theta_start=0,
            theta_stop=180,
            theta_step=ANGULAR_STEP,
            phi_start=0,
            phi_stop=360 - ANGULAR_STEP,
            phi_step=ANGULAR_STEP,
            transform_id="dipole_pattern",
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
            p1=(0.035, 0.035, 0.030),
            p2=(0.045, 0.045, 0.190),
            dl=(0.001, 0.001, 0.001),
            filename=OUTPUT_STEM,
            output_type="f",
        )
    )
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / OUTPUT_STEM,
        help="output path without the .h5 suffix",
    )
    args = parser.parse_args()

    options = {}
    if args.gpu is not None:
        options.update(gpu=[args.gpu], gpu_precision="single")
    gprMax.run(scenes=[build_scene()], outputfile=args.output, **options)


if __name__ == "__main__":
    main()
