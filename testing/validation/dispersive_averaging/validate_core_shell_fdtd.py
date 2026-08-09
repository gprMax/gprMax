"""Validate generalized dispersive-interface averaging with a core-shell sphere.

The physical model consists of a Debye core and a Lorentz shell in free
space.  Only Yee edges cut by either sharp interface receive an arithmetic
compound response.  Backscatter from the conventional equivalent-current
NTFF is compared with the exact dispersive Aden--Kerker series.

Run, for example, with::

    python -m testing.validation.dispersive_averaging.validate_core_shell_fdtd \
        --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib
import numpy as np

import gprMax

from .coated_sphere_mie import coated_sphere_backscatter_rcs
from .pole_models import debye_term, lorentz_term, make_material

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PULSE_FREQUENCY = 1.0e9


@dataclass(frozen=True)
class SphereConfiguration:
    """Spatial and spectral sampling for one physically identical sphere."""

    name: str
    dl: float
    domain_cells: int
    centre_cell: int
    core_radius_cells: int
    outer_radius_cells: int
    tfsf_lower_cell: int
    tfsf_upper_cell: int
    ntff_lower_cell: int
    ntff_upper_cell: int
    pml_cells: int
    time_window: float
    frequency_spacing: float
    analytical_plot_samples: int

    @property
    def frequencies(self):
        return np.arange(0.15e9, 2.30e9 + self.frequency_spacing / 2, self.frequency_spacing)

    @property
    def analytical_plot_frequencies(self):
        return np.linspace(0.15e9, 2.30e9, self.analytical_plot_samples)

    @property
    def core_radius(self):
        return self.core_radius_cells * self.dl

    @property
    def outer_radius(self):
        return self.outer_radius_cells * self.dl


CONFIGURATIONS = {
    "baseline": SphereConfiguration(
        name="baseline",
        dl=4e-3,
        domain_cells=160,
        centre_cell=80,
        core_radius_cells=15,
        outer_radius_cells=25,
        tfsf_lower_cell=32,
        tfsf_upper_cell=128,
        ntff_lower_cell=30,
        ntff_upper_cell=130,
        pml_cells=10,
        time_window=60e-9,
        frequency_spacing=25e6,
        analytical_plot_samples=2001,
    ),
    "refined": SphereConfiguration(
        name="refined",
        dl=2.5e-3,
        domain_cells=256,
        centre_cell=128,
        core_radius_cells=24,
        outer_radius_cells=40,
        tfsf_lower_cell=52,
        tfsf_upper_cell=204,
        ntff_lower_cell=48,
        ntff_upper_cell=208,
        pml_cells=16,
        time_window=100e-9,
        frequency_spacing=12.5e6,
        analytical_plot_samples=4001,
    ),
}

CORE = {
    "er_inf": 3.0,
    "conductivity": 5e-3,
    "delta_er": 5.0,
    "tau": 0.25e-9,
}
SHELL = {
    "er_inf": 2.5,
    "delta_er": 2.0,
    "resonance_frequency": 1.2e9,
    "damping": 0.08 * 2 * np.pi * 1.2e9,
}


def _point(cells, dl):
    return tuple(float(value * dl) for value in cells)


def analytical_materials():
    """Return material models used by the exact coated-sphere solution."""

    core = make_material(
        "Debye core",
        CORE["er_inf"],
        (debye_term(CORE["delta_er"], CORE["tau"]),),
        conductivity=CORE["conductivity"],
    )
    shell = make_material(
        "Lorentz shell",
        SHELL["er_inf"],
        (
            lorentz_term(
                SHELL["delta_er"],
                SHELL["resonance_frequency"],
                SHELL["damping"],
            ),
        ),
    )
    return core, shell


def build_scene(configuration: SphereConfiguration, averaging: bool, threads: int):
    """Build one Debye-core/Lorentz-shell scattering scene."""

    centre = _point((configuration.centre_cell,) * 3, configuration.dl)
    scene = gprMax.Scene()
    scene.add(gprMax.DispersiveAveraging(enabled=averaging))
    scene.add(gprMax.Discretisation(p1=(configuration.dl,) * 3))
    scene.add(gprMax.Domain(p1=_point((configuration.domain_cells,) * 3, configuration.dl)))
    scene.add(gprMax.TimeWindow(time=configuration.time_window))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.PMLThickness(thickness=configuration.pml_cells))
    scene.add(
        gprMax.Material(
            er=CORE["er_inf"],
            se=CORE["conductivity"],
            mr=1,
            sm=0,
            id="debye_core",
        )
    )
    scene.add(
        gprMax.AddDebyeDispersion(
            poles=1,
            er_delta=(CORE["delta_er"],),
            tau=(CORE["tau"],),
            material_ids=("debye_core",),
        )
    )
    scene.add(gprMax.Material(er=SHELL["er_inf"], se=0, mr=1, sm=0, id="lorentz_shell"))
    scene.add(
        gprMax.AddLorentzDispersion(
            poles=1,
            er_delta=(SHELL["delta_er"],),
            omega=(SHELL["resonance_frequency"],),
            delta=(SHELL["damping"],),
            material_ids=("lorentz_shell",),
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=PULSE_FREQUENCY, id="pulse"))
    # Build the shell first and overwrite its centre with the distinct core.
    scene.add(
        gprMax.Sphere(
            p1=centre,
            r=configuration.outer_radius,
            material_id="lorentz_shell",
        )
    )
    scene.add(
        gprMax.Sphere(
            p1=centre,
            r=configuration.core_radius,
            material_id="debye_core",
        )
    )
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=_point((configuration.tfsf_lower_cell,) * 3, configuration.dl),
            p2=_point((configuration.tfsf_upper_cell,) * 3, configuration.dl),
            m_vec=(1, 0, 0),
            psi=90,
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            p1=_point((configuration.ntff_lower_cell,) * 3, configuration.dl),
            p2=_point((configuration.ntff_upper_cell,) * 3, configuration.dl),
            id="surface",
            origin=centre,
        )
    )
    scene.add(
        gprMax.NTFFFrequencyTransform(
            surface_id="surface",
            id="spectrum",
            frequencies=configuration.frequencies,
            window="rectangular",
            save_surface_dft=False,
            plane_wave_index=0,
        )
    )
    scene.add(
        gprMax.NTFFFarField(
            theta=(90.0,),
            phi=(180.0,),
            transform_id="spectrum",
            id="backscatter",
            outputs=("rcs",),
        )
    )
    return scene


def _run(configuration, mode, cache_dir, *, gpu, precision, threads, reuse):
    cache_name = f"core_shell_{mode}"
    if configuration.name != "baseline":
        cache_name = f"core_shell_{configuration.name}_{mode}"
    path = cache_dir / f"{cache_name}.h5"
    runtime = 0.0
    if not (reuse and path.exists()):
        options = (
            {"cpu_precision": precision}
            if gpu is None
            else {"gpu": [gpu], "gpu_precision": precision}
        )
        start = perf_counter()
        gprMax.run(
            scenes=[build_scene(configuration, mode == "averaged", threads)],
            n=1,
            outputfile=path.with_suffix(""),
            hide_progress_bars=True,
            log_level=logging.WARNING,
            **options,
        )
        runtime = perf_counter() - start
    with h5py.File(path, "r") as output:
        transform = output["ntff/surface/frequency/spectrum"]
        frequency = np.asarray(transform["frequencies"], dtype=float)
        rcs = np.asarray(transform["far_field/backscatter/fields/rcs"][:, 0], dtype=float)
        backend = transform.attrs["collection_backend"]
        if isinstance(backend, bytes):
            backend = backend.decode()
        iterations = int(output.attrs["Iterations"])
        dt = float(output.attrs["dt"])
    return frequency, rcs, runtime, str(backend), iterations, dt


def _analyse(configuration, data):
    frequencies = data["averaged"][0]
    core, shell = analytical_materials()
    analytical_at_samples = coated_sphere_backscatter_rcs(
        frequencies,
        core_radius=configuration.core_radius,
        outer_radius=configuration.outer_radius,
        core=core,
        shell=shell,
    )
    analytical_for_plot = coated_sphere_backscatter_rcs(
        configuration.analytical_plot_frequencies,
        core_radius=configuration.core_radius,
        outer_radius=configuration.outer_radius,
        core=core,
        shell=shell,
    )
    area = np.pi * configuration.outer_radius**2
    result = {
        "frequency": frequencies,
        "analytical": analytical_at_samples,
        "analytical_plot_frequency": configuration.analytical_plot_frequencies,
        "analytical_plot": analytical_for_plot,
        "area": area,
        "modes": {},
    }
    for mode, values in data.items():
        simulated = values[1]
        error_db = 10 * np.log10(simulated / analytical_at_samples)
        result["modes"][mode] = {
            "rcs": simulated,
            "error_db": error_db,
            "rms_error_db": float(np.sqrt(np.mean(error_db**2))),
            "median_absolute_error_db": float(np.median(np.abs(error_db))),
            "maximum_absolute_error_db": float(np.max(np.abs(error_db))),
            "relative_l2_error_percent": float(
                100
                * np.linalg.norm(simulated - analytical_at_samples)
                / np.linalg.norm(analytical_at_samples)
            ),
        }
    return result


def _write_results(configuration, result, data, output_dir):
    with (output_dir / "core_shell_backscatter.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frequency_hz",
                "analytical_rcs_m2",
                "averaged_rcs_m2",
                "staircased_rcs_m2",
                "averaged_error_db",
                "staircased_error_db",
            )
        )
        for index, frequency in enumerate(result["frequency"]):
            writer.writerow(
                (
                    frequency,
                    result["analytical"][index],
                    result["modes"]["averaged"]["rcs"][index],
                    result["modes"]["staircased"]["rcs"][index],
                    result["modes"]["averaged"]["error_db"][index],
                    result["modes"]["staircased"]["error_db"][index],
                )
            )

    with (output_dir / "core_shell_analytical_dense.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frequency_hz", "analytical_rcs_m2"))
        writer.writerows(zip(result["analytical_plot_frequency"], result["analytical_plot"]))

    summary_path = output_dir / "core_shell_metrics.json"
    previous_modes = {}
    if summary_path.exists():
        previous_modes = json.loads(summary_path.read_text()).get("modes", {})
    summary = {
        "model": {
            "configuration": configuration.name,
            "dl_m": configuration.dl,
            "domain_cells": configuration.domain_cells,
            "core_radius_m": configuration.core_radius,
            "outer_radius_m": configuration.outer_radius,
            "time_window_s": configuration.time_window,
            "frequency_sampling": {
                "fdtd_samples": int(result["frequency"].size),
                "fdtd_spacing_hz": float(np.diff(result["frequency"])[0]),
                "analytical_plot_samples": int(result["analytical_plot_frequency"].size),
                "analytical_plot_spacing_hz": float(
                    np.diff(result["analytical_plot_frequency"])[0]
                ),
            },
            "core": CORE,
            "shell": SHELL,
        },
        "modes": {},
    }
    for mode, values in data.items():
        summary["modes"][mode] = {
            **{key: value for key, value in result["modes"][mode].items() if np.isscalar(value)},
            "runtime_seconds": values[2]
            or previous_modes.get(mode, {}).get("runtime_seconds", 0.0),
            "collection_backend": values[3],
            "iterations": values[4],
            "dt_seconds": values[5],
        }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    frequency_ghz = result["frequency"] / 1e9
    analytical_frequency_ghz = result["analytical_plot_frequency"] / 1e9
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2.2, 1)},
    )
    axes[0].semilogy(
        analytical_frequency_ghz,
        result["analytical_plot"] / result["area"],
        color="black",
        linewidth=2,
        label="Aden--Kerker analytical",
    )
    colours = {"averaged": "black", "staircased": "black"}
    markers = {"averaged": "o", "staircased": "s"}
    labels = {"averaged": "General pole-residue average", "staircased": "Staircased"}
    for mode, values in result["modes"].items():
        axes[0].semilogy(
            frequency_ghz,
            values["rcs"] / result["area"],
            color=colours[mode],
            linestyle="none",
            marker=markers[mode],
            markerfacecolor="none",
            markersize=4,
            markeredgewidth=0.9,
            label=labels[mode],
        )
    bar_width = 0.36 * float(np.median(np.diff(frequency_ghz)))
    bar_offsets = {"averaged": -bar_width / 2, "staircased": bar_width / 2}
    facecolours = {"averaged": "white", "staircased": "0.65"}
    hatches = {"averaged": "///", "staircased": "..."}
    for mode, values in result["modes"].items():
        axes[1].bar(
            frequency_ghz + bar_offsets[mode],
            np.abs(values["error_db"]),
            width=bar_width,
            facecolor=facecolours[mode],
            edgecolor="black",
            hatch=hatches[mode],
            linewidth=0.45,
            label=labels[mode],
        )
    axes[0].set_ylabel(r"Normalised backscatter, $\sigma_b/(\pi a^2)$")
    axes[0].set_title(f"Debye-core/Lorentz-shell sphere ({configuration.dl * 1e3:g}-mm grid)")
    axes[1].set(
        xlabel="Frequency [GHz]",
        ylabel="Pointwise absolute error [dB]",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.savefig(output_dir / "core_shell_backscatter.png", dpi=180)
    if configuration.name != "baseline":
        figure.savefig(
            output_dir / f"core_shell_backscatter_{configuration.name}.png",
            dpi=180,
        )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--precision", choices=("single", "double"), default="single")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--configuration",
        choices=tuple(CONFIGURATIONS),
        default="baseline",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("averaged", "staircased"),
        default=("averaged", "staircased"),
        help="Run one mode to populate its cache, or both to also write comparisons.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    configuration = CONFIGURATIONS[args.configuration]
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "results" / "core_shell_fdtd"
        if configuration.name != "baseline":
            args.output_dir /= configuration.name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {
        mode: _run(
            configuration,
            mode,
            cache_dir,
            gpu=args.gpu,
            precision=args.precision,
            threads=args.threads,
            reuse=args.reuse,
        )
        for mode in args.modes
    }
    if set(data) == {"averaged", "staircased"}:
        result = _analyse(configuration, data)
        _write_results(configuration, result, data, args.output_dir)
        print(args.output_dir / "core_shell_metrics.json")
    else:
        print(cache_dir)


if __name__ == "__main__":
    main()
