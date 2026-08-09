"""Monostatic backscatter RCS of a PEC sphere over a frequency sweep."""

import argparse
from pathlib import Path

import numpy as np

import gprMax

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "pec_sphere_rcs"

DL = 0.5e-3
DOMAIN_SIZE = 0.160
TIME_WINDOW = 12e-9
CENTRE = (0.080, 0.080, 0.080)
RADIUS = 0.016
PULSE_FREQUENCY = 4.5e9
FREQUENCIES = np.arange(0.75e9, 9.0e9 + 0.125e9, 0.25e9)


def build_scene():
    """Build a closed-surface equivalent-current RCS model."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="PEC-sphere monostatic backscatter RCS"))
    scene.add(gprMax.Domain(p1=(DOMAIN_SIZE,) * 3))
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(
        gprMax.Waveform(
            wave_type="ricker",
            amp=1,
            freq=PULSE_FREQUENCY,
            id="incident_pulse",
        )
    )
    scene.add(gprMax.Sphere(p1=CENTRE, r=RADIUS, material_id="pec"))

    # +x propagation with E parallel to +z. The TFSF box separates the
    # incident field from the scattered field used by the NTFF transform.
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=(0.040, 0.040, 0.040),
            p2=(0.120, 0.120, 0.120),
            m_vec=(1, 0, 0),
            psi=90,
            waveform_id="incident_pulse",
        )
    )

    scene.add(
        gprMax.NTFFSurface(
            p1=(0.028, 0.028, 0.028),
            p2=(0.132, 0.132, 0.132),
            id="rcs_surface",
            origin=CENTRE,
        )
    )
    scene.add(
        gprMax.NTFFFrequencyTransform(
            surface_id="rcs_surface",
            id="rcs_spectrum",
            frequencies=FREQUENCIES,
            window="rectangular",
            save_surface_dft=False,
            plane_wave_index=0,
        )
    )

    scene.add(
        gprMax.NTFFFarField(
            theta=(90.0,),
            phi=(180.0,),
            transform_id="rcs_spectrum",
            id="backscatter",
            outputs=("Etheta", "Ephi", "rcs"),
        )
    )
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, 0),
            p2=(DOMAIN_SIZE,) * 3,
            dl=(DL,) * 3,
            filename=OUTPUT_STEM,
            output_type="n",
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
