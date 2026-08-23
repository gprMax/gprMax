"""Reproduce Hartley's Debye-soil sphere averaging comparison against Mie theory.

The experiment follows Section 4.4 of Hartley (2020): a 200-cell cubical
domain contains a 60-cell-radius sphere made from one of three two-pole
Puerto Rico clay/loam models. A 300 MHz Gaussian plane wave illuminates the
sphere. Runs with and without Hartley's arithmetic Debye interface average
are compared with the exact homogeneous-sphere Mie series.

The FDTD solve uses the production TFSF plane wave and conventional
frequency-domain equivalent-current far-field transform. Its plane-wave
reference provides RCS normalisation directly. Device-resident surface DFTs
allow the full thesis frequency sweep to run efficiently on CUDA.
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
from scipy.constants import epsilon_0

import gprMax
from testing.validation.mie_dielectric import dielectric_sphere_bistatic_rcs

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DOMAIN_CELLS = 200
SPHERE_RADIUS_CELLS = 60
CENTRE_CELL = 100
TFSF_LOWER_CELL = 35
TFSF_UPPER_CELL = 165
NTFF_LOWER_CELL = 33
NTFF_UPPER_CELL = 167
PML_CELLS = 10
PULSE_FREQUENCY = 300e6
DEFAULT_TIME_WINDOW = 300e-9
FREQUENCIES = np.arange(25e6, 525e6 + 2.5e6, 5e6)
BACKSCATTER_THETA = 90.0
BACKSCATTER_PHI = 180.0

# Table 4.1 and Table 4.2 of Hartley (2020). Conductivity values in the
# thesis table are mS/m and relaxation times are ns.
SOILS = {
    "a": {
        "label": "Puerto Rico type A clay/loam (2.5% moisture)",
        "moisture_percent": 2.5,
        "dl": 14e-3,
        "er_inf": 3.2,
        "conductivity": 0.397e-3,
        "delta_er": (0.75, 0.30),
        "tau": (2.71e-9, 0.108e-9),
        "thesis_minimum_iterations": 6880,
    },
    "b": {
        "label": "Puerto Rico type B clay/loam (5% moisture)",
        "moisture_percent": 5.0,
        "dl": 12e-3,
        "er_inf": 4.15,
        "conductivity": 1.11e-3,
        "delta_er": (1.80, 0.60),
        "tau": (3.79e-9, 0.151e-9),
        "thesis_minimum_iterations": 8450,
    },
    "c": {
        "label": "Puerto Rico type C clay/loam (10% moisture)",
        "moisture_percent": 10.0,
        "dl": 9e-3,
        "er_inf": 6.0,
        "conductivity": 2.0e-3,
        "delta_er": (2.75, 0.75),
        "tau": (3.98e-9, 0.251e-9),
        "thesis_minimum_iterations": 10200,
    },
}


def _point(cells, dl):
    return tuple(float(value * dl) for value in cells)


def _waveform():
    return gprMax.Waveform(
        wave_type="gaussian",
        amp=1,
        freq=PULSE_FREQUENCY,
        id="thesis_gaussian",
    )


def build_sphere_scene(soil_name, averaging, threads, time_window):
    """Build one full-size thesis sphere scene."""

    soil = SOILS[soil_name]
    dl = soil["dl"]
    domain = _point((DOMAIN_CELLS,) * 3, dl)
    centre = _point((CENTRE_CELL,) * 3, dl)
    material_id = f"puerto_rico_soil_{soil_name}"

    scene = gprMax.Scene()
    scene.add(gprMax.DispersiveAveraging(enabled=averaging))
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.TimeWindow(time=time_window))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(
        gprMax.Material(
            er=soil["er_inf"],
            se=soil["conductivity"],
            mr=1,
            sm=0,
            id=material_id,
        )
    )
    scene.add(
        gprMax.AddDebyeDispersion(
            poles=2,
            er_delta=soil["delta_er"],
            tau=soil["tau"],
            material_ids=(material_id,),
        )
    )
    scene.add(_waveform())
    scene.add(
        gprMax.Sphere(
            p1=centre,
            r=SPHERE_RADIUS_CELLS * dl,
            material_id=material_id,
        )
    )
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=_point((TFSF_LOWER_CELL,) * 3, dl),
            p2=_point((TFSF_UPPER_CELL,) * 3, dl),
            m_vec=(1, 0, 0),
            psi=90,
            waveform_id="thesis_gaussian",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            p1=_point((NTFF_LOWER_CELL,) * 3, dl),
            p2=_point((NTFF_UPPER_CELL,) * 3, dl),
            id="thesis_surface",
            origin=centre,
        )
    )
    scene.add(
        gprMax.NTFFFrequencyTransform(
            surface_id="thesis_surface",
            id="thesis_spectrum",
            frequencies=FREQUENCIES,
            window="rectangular",
            save_surface_dft=False,
            plane_wave_index=0,
        )
    )
    scene.add(
        gprMax.NTFFFarField(
            theta=(BACKSCATTER_THETA,),
            phi=(BACKSCATTER_PHI,),
            transform_id="thesis_spectrum",
            id="backscatter",
            outputs=("Etheta", "Ephi", "rcs"),
        )
    )
    return scene


def _run(scene, outputfile, precision, gpu):
    solver_options = (
        {"cpu_precision": precision} if gpu is None else {"gpu": [gpu], "gpu_precision": precision}
    )
    start = perf_counter()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **solver_options,
    )
    return perf_counter() - start


def _read_far_field(path):
    with h5py.File(path, "r") as output:
        transform = output["ntff/thesis_surface/frequency/thesis_spectrum"]
        group = transform["far_field/backscatter"]
        frequencies = np.asarray(transform["frequencies"], dtype=np.float64)
        rcs = np.asarray(group["fields/rcs"][:, 0], dtype=np.float64)
        backend = transform.attrs["collection_backend"]
        iterations = int(output.attrs["Iterations"])
        dt = float(output.attrs["dt"])
    if isinstance(backend, bytes):
        backend = backend.decode()
    return frequencies, rcs, str(backend), iterations, dt


def complex_permittivity(soil_name, frequencies):
    """Return the continuous two-pole Debye response in engineering convention."""

    soil = SOILS[soil_name]
    omega = 2 * np.pi * np.asarray(frequencies, dtype=np.float64)
    er = np.full(omega.shape, soil["er_inf"], dtype=np.complex128)
    er += soil["conductivity"] / (1j * omega * epsilon_0)
    for delta_er, tau in zip(soil["delta_er"], soil["tau"]):
        er += delta_er / (1 + 1j * omega * tau)
    return er


def analytical_rcs(soil_name, frequencies):
    soil = SOILS[soil_name]
    radius = SPHERE_RADIUS_CELLS * soil["dl"]
    er = complex_permittivity(soil_name, frequencies)
    return np.asarray(
        [
            dielectric_sphere_bistatic_rcs(
                frequency,
                radius,
                permittivity,
                (np.pi,),
                polarisation="perpendicular",
            )[0]
            for frequency, permittivity in zip(frequencies, er)
        ],
        dtype=np.float64,
    )


def analyse(soil_name, far_fields):
    frequencies = next(iter(far_fields.values()))[0]
    analytical = analytical_rcs(soil_name, frequencies)
    area = np.pi * (SPHERE_RADIUS_CELLS * SOILS[soil_name]["dl"]) ** 2
    result = {
        "frequencies": frequencies,
        "analytical_rcs": analytical,
        "analytical_normalized": analytical / area,
        "modes": {},
    }
    for mode, data in far_fields.items():
        mode_frequencies, simulated = data[:2]
        if not np.array_equal(mode_frequencies, frequencies):
            raise ValueError("averaged and rough runs use different frequencies")
        error_db = 10 * np.log10(simulated / analytical)
        relative_error = 100 * (simulated - analytical) / analytical
        result["modes"][mode] = {
            "simulated_rcs": simulated,
            "simulated_normalized": simulated / area,
            "error_db": error_db,
            "relative_error_percent": relative_error,
            "rms_error_db": float(np.sqrt(np.mean(error_db**2))),
            "maximum_absolute_error_db": float(np.max(np.abs(error_db))),
            "median_absolute_error_db": float(np.median(np.abs(error_db))),
            "relative_l2_error_percent": float(
                100 * np.linalg.norm(simulated - analytical) / np.linalg.norm(analytical)
            ),
            "mean_absolute_relative_error_percent": float(np.mean(np.abs(relative_error))),
            "rayleigh_mean_absolute_relative_error_percent": float(
                np.mean(np.abs(relative_error[frequencies <= 100e6]))
            ),
            "resonant_mean_absolute_relative_error_percent": float(
                np.mean(np.abs(relative_error[frequencies > 100e6]))
            ),
        }
    return result


def _save_csv(path, result):
    modes = tuple(result["modes"])
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = ["frequency_hz", "mie_rcs_m2", "mie_normalized_rcs"]
        for mode in modes:
            header.extend(
                (
                    f"{mode}_rcs_m2",
                    f"{mode}_normalized_rcs",
                    f"{mode}_error_db",
                    f"{mode}_relative_error_percent",
                )
            )
        writer.writerow(header)
        for index, frequency in enumerate(result["frequencies"]):
            row = [
                frequency,
                result["analytical_rcs"][index],
                result["analytical_normalized"][index],
            ]
            for mode in modes:
                values = result["modes"][mode]
                row.extend(
                    (
                        values["simulated_rcs"][index],
                        values["simulated_normalized"][index],
                        values["error_db"][index],
                        values["relative_error_percent"][index],
                    )
                )
            writer.writerow(row)


def _plot(path, soil_name, result):
    colours = {"averaged": "tab:blue", "rough": "tab:orange"}
    labels = {"averaged": "FDTD Debye averaged", "rough": "FDTD staircased"}
    frequency_mhz = result["frequencies"] / 1e6
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2.4, 1)},
    )
    axes[0].semilogy(
        frequency_mhz,
        result["analytical_normalized"],
        color="black",
        linewidth=2,
        label="Analytical Mie series",
    )
    for mode, values in result["modes"].items():
        axes[0].semilogy(
            frequency_mhz,
            values["simulated_normalized"],
            color=colours[mode],
            linewidth=1.3,
            label=labels[mode],
        )
        axes[1].plot(
            frequency_mhz,
            np.abs(values["error_db"]),
            color=colours[mode],
            label=labels[mode],
        )
    axes[0].set(
        ylabel=r"Normalised backscatter, $\sigma/(\pi a^2)$",
        title=SOILS[soil_name]["label"],
    )
    axes[1].set(
        xlabel="Frequency [MHz]",
        ylabel="Absolute RCS error [dB]",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_validation(
    output_dir,
    cache_dir,
    soil_names,
    modes,
    *,
    threads=4,
    precision="single",
    time_window=DEFAULT_TIME_WINDOW,
    gpu=None,
    reuse=False,
):
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for soil_name in soil_names:
        soil_output = output_dir / f"soil_{soil_name}"
        soil_cache = cache_dir / f"soil_{soil_name}"
        soil_output.mkdir(parents=True, exist_ok=True)
        soil_cache.mkdir(parents=True, exist_ok=True)
        runtimes = {}

        far_fields = {}
        diagnostics = {}
        for mode in modes:
            averaging = mode == "averaged"
            h5_path = soil_cache / f"sphere_{mode}.h5"
            if not (reuse and h5_path.exists()):
                runtimes[mode] = _run(
                    build_sphere_scene(
                        soil_name,
                        averaging,
                        threads,
                        time_window,
                    ),
                    h5_path.with_suffix(""),
                    precision,
                    gpu,
                )
            data = _read_far_field(h5_path)
            far_fields[mode] = data
            diagnostics[mode] = {
                "collection_backend": data[2],
                "iterations": data[3],
                "dt_seconds": data[4],
            }

        result = analyse(soil_name, far_fields)
        _save_csv(soil_output / "backscatter.csv", result)
        _plot(soil_output / "backscatter.png", soil_name, result)
        comparison = {}
        if {"averaged", "rough"}.issubset(result["modes"]):
            for metric in (
                "rms_error_db",
                "median_absolute_error_db",
                "relative_l2_error_percent",
                "mean_absolute_relative_error_percent",
                "rayleigh_mean_absolute_relative_error_percent",
                "resonant_mean_absolute_relative_error_percent",
            ):
                averaged = result["modes"]["averaged"][metric]
                rough = result["modes"]["rough"][metric]
                comparison[f"{metric}_reduction_factor"] = float(rough / averaged)
        summary = {
            "soil": soil_name,
            "configuration": {
                **SOILS[soil_name],
                "domain_cells": DOMAIN_CELLS,
                "sphere_radius_cells": SPHERE_RADIUS_CELLS,
                "time_window_seconds": time_window,
                "frequency_range_hz": [
                    float(FREQUENCIES[0]),
                    float(FREQUENCIES[-1]),
                ],
                "frequency_samples": int(FREQUENCIES.size),
                "precision": precision,
                "threads": threads,
                "backend": "cpu" if gpu is None else f"cuda:{gpu}",
            },
            "runtime_seconds": runtimes,
            "diagnostics": diagnostics,
            "metrics": {
                mode: {key: value for key, value in values.items() if np.isscalar(value)}
                for mode, values in result["modes"].items()
            },
            "averaging_improvement": comparison,
        }
        (soil_output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries[soil_name] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("debye_sphere_averaging_results"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).with_name("debye_sphere_averaging_results") / "_cache",
    )
    parser.add_argument(
        "--soil",
        choices=("a", "b", "c", "all"),
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=("averaged", "rough", "both"),
        default="both",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precision", choices=("single", "double"), default="single")
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument("--time-window", type=float, default=DEFAULT_TIME_WINDOW)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    soils = tuple(SOILS) if args.soil == "all" else (args.soil,)
    modes = ("averaged", "rough") if args.mode == "both" else (args.mode,)
    summary = run_validation(
        args.output_dir,
        args.cache_dir,
        soils,
        modes,
        threads=args.threads,
        precision=args.precision,
        time_window=args.time_window,
        gpu=args.gpu,
        reuse=args.reuse,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
