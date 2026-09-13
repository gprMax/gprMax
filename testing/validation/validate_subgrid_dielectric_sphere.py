"""CPU/CUDA HSG dielectric-sphere RCS checked against the independent Mie series.

Example: python -m testing.validation.validate_subgrid_dielectric_sphere --gpu 0
Use separate output directories for CPU/CUDA and interpolation-order comparisons.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

import gprMax
from testing.validation.mie_dielectric import dielectric_sphere_bistatic_rcs
from testing.validation.validate_dielectric_sphere_rcs import FREQUENCIES, RADIUS, RELATIVE_PERMITTIVITY


def build_scene(order=1, threads=4):
    scene = gprMax.Scene()
    centre = (0.081,) * 3
    for obj in (
        gprMax.Domain(p1=(0.162,) * 3),
        gprMax.Discretisation(p1=(0.0015,) * 3),
        gprMax.TimeWindow(time=12e-9),
        gprMax.OMPThreads(threads),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=4.5e9, id="pulse"),
    ):
        scene.add(obj)
    fine = gprMax.SubGridHSG(p1=(0.054,) * 3, p2=(0.108,) * 3, ratio=3, interpolation=order, id="sphere_grid")
    scene.add(fine)
    fine.add(gprMax.Material(er=RELATIVE_PERMITTIVITY, se=0, mr=1, sm=0, id="dielectric"))
    fine.add(gprMax.Sphere(p1=centre, r=RADIUS, material_id="dielectric"))
    fine.add(gprMax.Rx(p1=(0.081, 0.081, 0.081), id="inside_sphere"))
    scene.add(gprMax.Rx(p1=(0.138, 0.081, 0.081), id="scattered_field"))
    scene.add(gprMax.DiscretePlaneWaveAxial(p1=(0.030,) * 3, p2=(0.132,) * 3, axis="x", psi=90, waveform_id="pulse"))
    scene.add(gprMax.NTFFSurface(p1=(0.018,) * 3, p2=(0.144,) * 3, id="surface", origin=centre))
    scene.add(
        gprMax.KSIRFrequencyTransform("surface", "spectrum", FREQUENCIES, save_surface_dft=False, plane_wave_index=0)
    )
    scene.add(
        gprMax.KSIRFarField(
            theta=(90,), phi=(180,), transform_id="spectrum", id="backscatter", outputs=("Etheta", "Ephi", "rcs")
        )
    )
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--order", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "sphere"
    options = {} if args.gpu is None else {"gpu": [args.gpu]}
    start = perf_counter()
    gprMax.run(
        scenes=[build_scene(args.order, args.threads)],
        outputfile=target,
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        gpu_precision="double",
        hide_progress_bars=True,
        log_level=30,
        **options,
    )
    runtime = perf_counter() - start
    with h5py.File(target.with_suffix(".h5")) as f:
        group = f["ntff/surface/frequency/spectrum"]
        frequencies = group["frequencies"][...]
        rcs = group["far_field/backscatter/fields/rcs"][:, 0]
    mie = np.array(
        [dielectric_sphere_bistatic_rcs(freq, RADIUS, RELATIVE_PERMITTIVITY, (np.pi,))[0] for freq in frequencies]
    )
    errors = 10 * np.log10(rcs / mie)
    summary = dict(
        backend="cpu" if args.gpu is None else f"cuda:{args.gpu}",
        order=args.order,
        precision="double",
        coarse_dl=0.0015,
        fine_dl=0.0005,
        seconds=runtime,
        frequency_hz=frequencies.tolist(),
        rcs=rcs.tolist(),
        mie_rcs=mie.tolist(),
        error_db=errors.tolist(),
        rms_error_db=float(np.sqrt(np.mean(errors**2))),
        maximum_error_db=float(np.max(np.abs(errors))),
    )
    summary["passed"] = summary["rms_error_db"] <= 0.75 and summary["maximum_error_db"] <= 1.25
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit("HSG dielectric-sphere Mie comparison failed the production model's tolerances")


if __name__ == "__main__":
    main()
