"""Validate dielectric-sphere backscatter RCS against the Mie series.

The production path consists of a discrete plane wave, a closed
equivalent-current NTFF surface, frequency-domain accumulation, and HDF5
output. The sphere is a lossless non-magnetic dielectric with ``epsilon_r=4``.

Examples::

    python -m testing.validation.validate_dielectric_sphere_rcs --gpu 0
    python -m testing.validation.validate_dielectric_sphere_rcs --reuse
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib
import numpy as np
from scipy.constants import c

import gprMax
from testing.validation.mie_dielectric import dielectric_sphere_bistatic_rcs

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DL = 0.5e-3
DOMAIN_SIZE = 0.160
TIME_WINDOW = 12e-9
CENTRE = (0.080, 0.080, 0.080)
RADIUS = 0.016
RELATIVE_PERMITTIVITY = 4.0
PULSE_FREQUENCY = 4.5e9
FREQUENCIES = np.arange(0.75e9, 9.0e9 + 0.125e9, 0.25e9)
GROUP = "ntff/rcs_surface/frequency/rcs_spectrum/far_field/backscatter"

ACCEPTANCE_LIMITS_DB = {
    "rms_error_db": 0.75,
    "maximum_absolute_error_db": 1.25,
}


def build_scene(threads=4):
    """Build the production TFSF/equivalent-current RCS model."""

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.Domain(p1=(DOMAIN_SIZE,) * 3))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(
        gprMax.Material(
            er=RELATIVE_PERMITTIVITY,
            se=0,
            mr=1,
            sm=0,
            id="dielectric_sphere",
        )
    )
    scene.add(
        gprMax.Waveform(
            wave_type="ricker",
            amp=1,
            freq=PULSE_FREQUENCY,
            id="incident_pulse",
        )
    )
    scene.add(
        gprMax.Sphere(
            p1=CENTRE,
            r=RADIUS,
            material_id="dielectric_sphere",
        )
    )
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
    return scene


def _run(outputfile, threads, precision, gpu):
    solver_options = (
        {"cpu_precision": precision} if gpu is None else {"gpu": [gpu], "gpu_precision": precision}
    )
    start = perf_counter()
    gprMax.run(
        scenes=[build_scene(threads)],
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **solver_options,
    )
    return perf_counter() - start


def _read_output(path):
    with h5py.File(path, "r") as output:
        transform = output["ntff/rcs_surface/frequency/rcs_spectrum"]
        group = output[GROUP]
        frequencies = np.asarray(transform["frequencies"], dtype=np.float64)
        rcs = np.asarray(group["fields/rcs"][:, 0], dtype=np.float64)
        collection_backend = transform.attrs["collection_backend"]
    if isinstance(collection_backend, bytes):
        collection_backend = collection_backend.decode()
    return frequencies, rcs, str(collection_backend)


def analyse(frequencies, simulated):
    analytical = np.asarray(
        [
            dielectric_sphere_bistatic_rcs(
                frequency,
                RADIUS,
                RELATIVE_PERMITTIVITY,
                (np.pi,),
                polarisation="perpendicular",
            )[0]
            for frequency in frequencies
        ]
    )
    size_parameter = 2 * np.pi * frequencies * RADIUS / c
    area = np.pi * RADIUS**2
    error_db = 10 * np.log10(simulated / analytical)
    return {
        "frequencies": frequencies,
        "size_parameter": size_parameter,
        "exterior_wavelength_cells": c / (frequencies * DL),
        "interior_wavelength_cells": c / (frequencies * DL * np.sqrt(RELATIVE_PERMITTIVITY)),
        "simulated_rcs": simulated,
        "analytical_rcs": analytical,
        "simulated_normalized": simulated / area,
        "analytical_normalized": analytical / area,
        "error_db": error_db,
        "rms_error_db": float(np.sqrt(np.mean(error_db**2))),
        "maximum_absolute_error_db": float(np.max(np.abs(error_db))),
    }


def _save_csv(path, result):
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            (
                "frequency_hz",
                "size_parameter_ka",
                "exterior_wavelength_cells",
                "interior_wavelength_cells",
                "gprmax_rcs_m2",
                "mie_rcs_m2",
                "gprmax_normalized_rcs",
                "mie_normalized_rcs",
                "error_db",
            )
        )
        writer.writerows(
            zip(
                result["frequencies"],
                result["size_parameter"],
                result["exterior_wavelength_cells"],
                result["interior_wavelength_cells"],
                result["simulated_rcs"],
                result["analytical_rcs"],
                result["simulated_normalized"],
                result["analytical_normalized"],
                result["error_db"],
            )
        )


def _plot(path, result):
    dense_frequency = np.linspace(result["frequencies"][0], result["frequencies"][-1], 1200)
    dense_mie = np.asarray(
        [
            dielectric_sphere_bistatic_rcs(
                frequency,
                RADIUS,
                RELATIVE_PERMITTIVITY,
                (np.pi,),
            )[0]
            for frequency in dense_frequency
        ]
    )
    dense_ka = 2 * np.pi * dense_frequency * RADIUS / c
    area = np.pi * RADIUS**2
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2.4, 1)},
    )
    axes[0].semilogy(
        dense_ka,
        dense_mie / area,
        color="tab:orange",
        linewidth=2,
        label=r"Dielectric Mie series ($\epsilon_r=4$)",
    )
    axes[0].semilogy(
        result["size_parameter"],
        result["simulated_normalized"],
        "o",
        color="tab:blue",
        markersize=4,
        label="gprMax FDTD",
    )
    axes[1].plot(result["size_parameter"], result["error_db"], "o-", color="tab:red")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set(
        ylabel=r"Normalised backscatter RCS, $\sigma/(\pi a^2)$",
        title=(
            f"Dielectric sphere monostatic RCS: radius {RADIUS * 1e3:g} mm, "
            rf"$\epsilon_r={RELATIVE_PERMITTIVITY:g}$"
        ),
    )
    axes[1].set(
        xlabel=r"Exterior electrical size, $ka=2\pi f a/c$",
        ylabel="gprMax - Mie [dB]",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
    axes[0].legend()

    def ka_to_ghz(value):
        return value * c / (2 * np.pi * RADIUS) / 1e9

    def ghz_to_ka(value):
        return value * 1e9 * 2 * np.pi * RADIUS / c

    frequency_axis = axes[0].secondary_xaxis("top", functions=(ka_to_ghz, ghz_to_ka))
    frequency_axis.set_xlabel("Frequency [GHz]")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _acceptance_summary(result):
    checks = {
        metric: {
            "value": result[metric],
            "maximum": maximum,
            "passed": result[metric] <= maximum,
        }
        for metric, maximum in ACCEPTANCE_LIMITS_DB.items()
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def _write_report(path, summary):
    lines = [
        "# Dielectric-sphere RCS validation",
        "",
        (
            "A z-polarised discrete plane wave propagating along +x illuminates a "
            f"{RADIUS * 1e3:g} mm-radius, lossless dielectric sphere with relative "
            f"permittivity {RELATIVE_PERMITTIVITY:g}. A closed equivalent-current "
            "NTFF surface returns monostatic backscatter at theta=90 deg, phi=180 "
            "deg. The comparison uses the exact homogeneous-sphere Mie series "
            "evaluated independently of gprMax."
        ),
        "",
        f"Overall validation status: **{'PASS' if summary['acceptance']['passed'] else 'FAIL'}**.",
        "",
        f"- Grid spacing: {DL * 1e3:g} mm ({RADIUS / DL:g} cells per radius)",
        f"- Frequency range: {FREQUENCIES[0] / 1e9:g}--{FREQUENCIES[-1] / 1e9:g} GHz",
        f"- RMS RCS error: {summary['rms_error_db']:.4g} dB",
        "- Maximum absolute RCS error: " f"{summary['maximum_absolute_error_db']:.4g} dB",
        "",
        (
            "The error includes the voxelised representation of the curved material "
            "interface as well as FDTD and transformation errors. Narrow dielectric "
            "resonances are especially sensitive to small geometry and phase shifts."
        ),
        "",
        "## Outputs",
        "",
        "- [Backscatter comparison](dielectric_sphere_backscatter_rcs.png)",
        "- `dielectric_sphere_backscatter_rcs.csv`",
        "- `summary.json`",
        "",
    ]
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def run_validation(
    output_dir,
    cache_dir,
    threads=4,
    precision="single",
    gpu=None,
    reuse=False,
):
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputfile = cache_dir / "dielectric_sphere_rcs"
    h5_path = outputfile.with_suffix(".h5")
    summary_path = output_dir / "summary.json"
    previous_runtime = 0.0
    if reuse and summary_path.exists():
        previous_runtime = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "runtime_seconds", 0.0
        )
    runtime = 0.0
    if not (reuse and h5_path.exists()):
        runtime = _run(outputfile, threads, precision, gpu)
    frequencies, simulated, collection_backend = _read_output(h5_path)
    result = analyse(frequencies, simulated)
    _save_csv(output_dir / "dielectric_sphere_backscatter_rcs.csv", result)
    _plot(output_dir / "dielectric_sphere_backscatter_rcs.png", result)
    summary = {
        "configuration": {
            "dl_metres": DL,
            "radius_metres": RADIUS,
            "relative_permittivity": RELATIVE_PERMITTIVITY,
            "time_window_seconds": TIME_WINDOW,
            "backend": "cpu" if gpu is None else f"cuda:{gpu}",
            "collection_backend": collection_backend,
            "precision": precision,
            "threads": threads,
        },
        "evaluated_frequency_range_hz": [
            float(result["frequencies"][0]),
            float(result["frequencies"][-1]),
        ],
        "frequency_samples": int(result["frequencies"].size),
        "size_parameter_range": [
            float(result["size_parameter"][0]),
            float(result["size_parameter"][-1]),
        ],
        "exterior_wavelength_cells_range": [
            float(np.min(result["exterior_wavelength_cells"])),
            float(np.max(result["exterior_wavelength_cells"])),
        ],
        "interior_wavelength_cells_range": [
            float(np.min(result["interior_wavelength_cells"])),
            float(np.max(result["interior_wavelength_cells"])),
        ],
        "rms_error_db": result["rms_error_db"],
        "maximum_absolute_error_db": result["maximum_absolute_error_db"],
        "runtime_seconds": runtime if runtime else previous_runtime,
        "acceptance": _acceptance_summary(result),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("dielectric_sphere_rcs_results"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(Path(__file__).with_name("dielectric_sphere_rcs_results") / "_cache"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precision", choices=("single", "double"), default="single")
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    summary = run_validation(
        args.output_dir,
        args.cache_dir,
        threads=args.threads,
        precision=args.precision,
        gpu=args.gpu,
        reuse=args.reuse,
    )
    print(json.dumps(summary, indent=2))
    if not summary["acceptance"]["passed"]:
        raise SystemExit("Dielectric-sphere RCS analytical validation failed")


if __name__ == "__main__":
    main()
