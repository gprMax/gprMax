"""Validate volume-integrated SAR in a lossy sphere against Mie theory.

A unit-amplitude plane wave illuminates a homogeneous lossy dielectric sphere.
The FDTD absorbed-power-density output is integrated over every tagged sphere
cell and compared with the exact Mie absorption cross-section multiplied by
the incident plane-wave power density.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib
import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import gprMax
from testing.validation.mie_dielectric import dielectric_sphere_absorption_cross_section

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

FREQUENCY = 1.0e9
DL = 0.75e-3
DOMAIN = (0.108, 0.108, 0.108)
CENTRE = np.asarray((0.054, 0.054, 0.054))
RADIUS = 0.018
RELATIVE_PERMITTIVITY = 4.0
CONDUCTIVITY = 0.30
DENSITY = 1000.0
RELATIVE_POWER_LIMIT = 0.08


def build_scene(dl=DL):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=6e-9))
    scene.add(gprMax.PMLThickness(thickness=8))
    scene.add(gprMax.OMPThreads(n=8))
    scene.add(
        gprMax.Material(
            er=RELATIVE_PERMITTIVITY,
            se=CONDUCTIVITY,
            mr=1,
            sm=0,
            id="sphere_tissue",
        )
    )
    scene.add(gprMax.MaterialDensity(density=DENSITY, material_ids=("sphere_tissue",)))
    scene.add(
        gprMax.Sphere(
            p1=tuple(CENTRE),
            r=RADIUS,
            material_id="sphere_tissue",
            averaging="y",
            tag="lossy_sphere",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=FREQUENCY, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.012, 0.012, 0.012),
            p2=(0.096, 0.096, 0.096),
            axis="x",
            psi=90,
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.SAR(
            frequencies=(FREQUENCY,),
            waveform_id="pulse",
            tags="lossy_sphere",
            id="sphere",
            target_amplitude=1.0,
            averaging_masses=(0.001, 0.01),
        )
    )
    return scene


def analytical_absorbed_power():
    omega = 2 * np.pi * FREQUENCY
    epsilon_r = RELATIVE_PERMITTIVITY + CONDUCTIVITY / (1j * omega * epsilon_0)
    cross_section = dielectric_sphere_absorption_cross_section(FREQUENCY, RADIUS, epsilon_r)
    incident_power_density = 1 / (2 * np.sqrt(mu_0 / epsilon_0))
    return cross_section, incident_power_density, cross_section * incident_power_density


def run(
    output_dir: Path,
    *,
    dl=DL,
    backend="cpu",
    precision="double",
):
    if backend not in ("cpu", "cuda", "opencl"):
        raise ValueError("backend must be 'cpu', 'cuda', or 'opencl'")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "lossy_sphere_sar"
    scene = build_scene(dl)
    solver_options = {}
    if backend == "cuda":
        solver_options["gpu"] = [0]
    elif backend == "opencl":
        solver_options["opencl"] = [0]
    start = perf_counter()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output_base,
        hide_progress_bars=True,
        cpu_precision=precision,
        gpu_precision=precision,
        log_level=logging.WARNING,
        **solver_options,
    )
    runtime = perf_counter() - start

    with h5py.File(output_base.with_suffix(".h5"), "r") as output:
        group = output["sar/sphere"]
        cells = np.asarray(group["cell_indices"])
        absorbed_density = np.asarray(group["absorbed_power_density"])[0]
        one_gram_peak = float(group["spatial_average/1g/peak_sar"][0])
        ten_gram_peak = float(group["spatial_average/10g/peak_sar"][0])
    numerical_power = float(np.sum(absorbed_density) * dl**3)
    cross_section, incident_density, analytical_power = analytical_absorbed_power()
    relative_error = abs(numerical_power / analytical_power - 1)
    voxel_mass = cells.shape[0] * DENSITY * dl**3

    metrics = {
        "frequency_hz": FREQUENCY,
        "dl_m": dl,
        "radius_m": RADIUS,
        "radius_cells": RADIUS / dl,
        "backend": backend,
        "precision": precision,
        "relative_permittivity": RELATIVE_PERMITTIVITY,
        "conductivity_S_per_m": CONDUCTIVITY,
        "density_kg_per_m3": DENSITY,
        "tagged_cells": int(cells.shape[0]),
        "voxelised_mass_kg": voxel_mass,
        "mie_absorption_cross_section_m2": cross_section,
        "incident_power_density_W_per_m2": incident_density,
        "analytical_absorbed_power_W": analytical_power,
        "fdtd_absorbed_power_W": numerical_power,
        "relative_absorbed_power_error": relative_error,
        "peak_1g_sar_W_per_kg": one_gram_peak,
        "peak_10g_sar_W_per_kg": ten_gram_peak,
        "runtime_seconds": runtime,
    }
    (output_dir / "lossy_sphere_sar_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    centre_index = int(round(CENTRE[2] / dl - 0.5))
    selected = cells[:, 2] == centre_index
    plane_cells = cells[selected]
    plane_sar = absorbed_density[selected] / DENSITY
    size = int(round(DOMAIN[0] / dl))
    image = np.full((size, size), np.nan)
    image[plane_cells[:, 0], plane_cells[:, 1]] = plane_sar
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    plot = axes[0].imshow(
        image.T,
        origin="lower",
        extent=(0, DOMAIN[0] * 1e3, 0, DOMAIN[1] * 1e3),
        cmap="magma",
    )
    figure.colorbar(plot, ax=axes[0], label="Local SAR [W/kg]")
    axes[0].set(xlabel="x [mm]", ylabel="y [mm]", title="Central sphere slice")
    axes[1].bar(
        ("Mie", "gprMax"),
        (analytical_power * 1e6, numerical_power * 1e6),
        color=("white", "0.5"),
        edgecolor="black",
        hatch=("", "//"),
    )
    axes[1].set(ylabel="Absorbed power [uW]", title="Volume-integrated absorption")
    figure.tight_layout()
    figure.savefig(output_dir / "lossy_sphere_sar.png", dpi=180)
    plt.close(figure)
    if relative_error > RELATIVE_POWER_LIMIT:
        raise AssertionError(
            f"Sphere absorbed-power error {relative_error:.3%} exceeds "
            f"{RELATIVE_POWER_LIMIT:.1%}"
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("sar_lossy_sphere_results"))
    parser.add_argument("--dl", type=float, default=DL)
    parser.add_argument("--backend", choices=("cpu", "cuda", "opencl"), default="cpu")
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output_dir,
                dl=args.dl,
                backend=args.backend,
                precision=args.precision,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
