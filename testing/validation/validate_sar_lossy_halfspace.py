"""Validate tagged-cell SAR against a lossy half-space solution.

A normally incident unit-amplitude plane wave illuminates a homogeneous lossy
dielectric. The gprMax cell-centred SAR along the central propagation line is
compared with the Fresnel transmission coefficient and exponential attenuation
of the continuous half-space solution.

Run with::

    python -m testing.validation.validate_sar_lossy_halfspace
"""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib
import numpy as np
from scipy.constants import epsilon_0, mu_0

import gprMax

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

FREQUENCY = 1.0e9
SWEEP_FREQUENCIES = np.linspace(0.5e9, 7.0e9, 14)
SWEEP_WAVEFORM_FREQUENCY = 3.0e9
DL = 2.0e-3
DOMAIN = (0.12, 0.08, 0.08)
INTERFACE_X = 0.060
RELATIVE_PERMITTIVITY = 4.0
CONDUCTIVITY = 0.10
DENSITY = 1000.0
RELATIVE_L2_LIMIT = 0.05
MAXIMUM_POINTWISE_RELATIVE_LIMIT = 0.06
SWEEP_RELATIVE_L2_LIMIT = 0.065
SWEEP_MAXIMUM_POINTWISE_RELATIVE_LIMIT = 0.08


def build_scene(
    include_sar=True,
    interface_mode="averaged",
    frequencies=(FREQUENCY,),
    waveform_frequency=FREQUENCY,
):
    """Return the complete production-path validation model."""

    if interface_mode not in ("averaged", "noavg_free_space_last"):
        raise ValueError(f"Unknown interface mode {interface_mode!r}")

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=6e-9))
    scene.add(gprMax.PMLThickness(thickness=8))
    scene.add(gprMax.OMPThreads(n=4))
    scene.add(
        gprMax.Material(
            er=RELATIVE_PERMITTIVITY,
            se=CONDUCTIVITY,
            mr=1,
            sm=0,
            id="tissue",
        )
    )
    scene.add(gprMax.MaterialDensity(density=DENSITY, material_ids=("tissue",)))
    scene.add(
        gprMax.Box(
            p1=(INTERFACE_X, 0, 0),
            p2=DOMAIN,
            material_id="tissue",
            # Averaging places the numerical interface at the declared face.
            # The construction-order experiment instead assigns complete
            # edge coefficients and then overwrites the shared edge below.
            averaging="y" if interface_mode == "averaged" else "n",
            tag="tissue_halfspace",
        )
    )
    if interface_mode == "noavg_free_space_last":
        scene.add(
            gprMax.Box(
                p1=(0, 0, 0),
                p2=(INTERFACE_X, DOMAIN[1], DOMAIN[2]),
                material_id="free_space",
                averaging="n",
            )
        )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=waveform_frequency, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.018, 0.020, 0.020),
            p2=(0.102, 0.060, 0.060),
            axis="x",
            psi=90,
            waveform_id="pulse",
        )
    )
    if include_sar:
        scene.add(
            gprMax.SAR(
                frequencies=tuple(float(value) for value in frequencies),
                waveform_id="pulse",
                tags="tissue_halfspace",
                id="halfspace",
                target_amplitude=1.0,
                spectrum_limit=10,
                averaging_masses=(),
            )
        )
    return scene


def analytical_sar(depth, *, frequency=FREQUENCY, yee_collocated=True):
    """Normal-incidence SAR for a unit incident peak phasor.

    By default the analytical phasor is collocated exactly as the production
    SAR output: the two distinct Ez values on the cell's x faces are averaged
    before forming ``|E|^2``. Setting ``yee_collocated=False`` returns the
    continuous point value at the cell centre instead.
    """

    omega = 2 * np.pi * frequency
    epsilon_r = RELATIVE_PERMITTIVITY + CONDUCTIVITY / (1j * omega * epsilon_0)
    eta_0 = np.sqrt(mu_0 / epsilon_0)
    eta_2 = np.sqrt(mu_0 / (epsilon_0 * epsilon_r))
    transmission = 2 * eta_2 / (eta_0 + eta_2)
    propagation = 1j * omega * np.sqrt(mu_0 * epsilon_0 * epsilon_r)
    depth = np.asarray(depth)
    if yee_collocated:
        electric = (
            0.5
            * transmission
            * (
                np.exp(-propagation * (depth - 0.5 * DL))
                + np.exp(-propagation * (depth + 0.5 * DL))
            )
        )
    else:
        electric = transmission * np.exp(-propagation * depth)
    return CONDUCTIVITY * np.abs(electric) ** 2 / (2 * DENSITY)


def analytical_cell_average_sar(depth, *, frequency=FREQUENCY):
    """Exact volume-average SAR in a cell for the 1-D exponential field."""

    omega = 2 * np.pi * frequency
    epsilon_r = RELATIVE_PERMITTIVITY + CONDUCTIVITY / (1j * omega * epsilon_0)
    propagation = 1j * omega * np.sqrt(mu_0 * epsilon_0 * epsilon_r)
    attenuation = float(np.real(propagation))
    centre = analytical_sar(depth, frequency=frequency, yee_collocated=False)
    argument = attenuation * DL
    factor = 1.0 if argument == 0 else np.sinh(argument) / argument
    return centre * factor


def _solver_options(backend, precision):
    if backend == "cpu":
        return {"cpu_precision": precision}
    if backend == "cuda":
        return {"gpu": [0], "gpu_precision": precision}
    if backend == "opencl":
        return {"opencl": [0], "gpu_precision": precision}
    raise ValueError("backend must be 'cpu', 'cuda', or 'opencl'")


def run(
    output_dir: Path,
    benchmark=False,
    interface_mode="averaged",
    backend="cpu",
    precision="double",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "lossy_halfspace_sar"
    scene = build_scene(interface_mode=interface_mode)
    start = perf_counter()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output_base,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **_solver_options(backend, precision),
    )
    sar_runtime = perf_counter() - start

    with h5py.File(output_base.with_suffix(".h5"), "r") as output:
        group = output["sar/halfspace"]
        cells = np.asarray(group["cell_indices"])
        sar = np.asarray(group["sar"])[0]

    centre_y = int(round(0.040 / DL))
    centre_z = int(round(0.040 / DL))
    selected = (cells[:, 1] == centre_y) & (cells[:, 2] == centre_z)
    x = (cells[selected, 0] + 0.5) * DL
    numerical = sar[selected]
    order = np.argsort(x)
    x = x[order]
    numerical = numerical[order]
    # Omit the interface cell and cells at/behind the TFSF termination/PML.
    retained = (x >= INTERFACE_X + 2 * DL) & (x <= 0.096)
    x = x[retained]
    numerical = numerical[retained]
    effective_interface = INTERFACE_X + (
        0.5 * DL if interface_mode == "noavg_free_space_last" else 0.0
    )
    depth = x - effective_interface
    analytical = analytical_cell_average_sar(depth)
    analytical_collocated = analytical_sar(depth)
    analytical_point = analytical_sar(depth, yee_collocated=False)
    relative_l2 = float(np.linalg.norm(numerical - analytical) / np.linalg.norm(analytical))
    maximum_relative = float(np.max(np.abs(numerical - analytical) / analytical))
    point_relative_l2 = float(
        np.linalg.norm(numerical - analytical_point) / np.linalg.norm(analytical_point)
    )
    point_maximum_relative = float(np.max(np.abs(numerical - analytical_point) / analytical_point))
    collocated_relative_l2 = float(
        np.linalg.norm(numerical - analytical_collocated) / np.linalg.norm(analytical_collocated)
    )
    collocated_maximum_relative = float(
        np.max(np.abs(numerical - analytical_collocated) / analytical_collocated)
    )

    metrics = {
        "frequency_hz": FREQUENCY,
        "dl_m": DL,
        "minimum_cells_per_wavelength": 10,
        "interface_mode": interface_mode,
        "backend": backend,
        "precision": precision,
        "declared_interface_m": INTERFACE_X,
        "effective_interface_m": effective_interface,
        "relative_l2_error": relative_l2,
        "maximum_pointwise_relative_error": maximum_relative,
        "yee_collocated_relative_l2_error": collocated_relative_l2,
        "yee_collocated_maximum_pointwise_relative_error": collocated_maximum_relative,
        "continuous_centre_relative_l2_error": point_relative_l2,
        "continuous_centre_maximum_pointwise_relative_error": point_maximum_relative,
        "comparison_points": int(x.size),
        "sar_monitor_memory_bytes": int(scene.output_objects[-1]._monitor.nbytes),
        "selected_tagged_cells": int(scene.output_objects[-1]._monitor.cells.shape[0]),
        "sar_runtime_seconds": sar_runtime,
    }
    if benchmark:
        baseline = build_scene(include_sar=False, interface_mode=interface_mode)
        start = perf_counter()
        gprMax.run(
            scenes=[baseline],
            n=1,
            outputfile=output_dir / "lossy_halfspace_baseline",
            hide_progress_bars=True,
            log_level=logging.WARNING,
            **_solver_options(backend, precision),
        )
        baseline_runtime = perf_counter() - start
        metrics["baseline_runtime_seconds"] = baseline_runtime
        metrics["sar_runtime_ratio"] = sar_runtime / baseline_runtime
    (output_dir / "lossy_halfspace_sar_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot((x - INTERFACE_X) * 1e3, analytical, "k-", label="Analytical")
    axis.plot(
        (x - INTERFACE_X) * 1e3,
        numerical,
        "ko",
        markerfacecolor="white",
        label="gprMax",
    )
    axis.set_xlabel("Depth into lossy half-space [mm]")
    axis.set_ylabel("Local SAR [W/kg]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lossy_halfspace_sar.png", dpi=180)
    plt.close(fig)
    if relative_l2 > RELATIVE_L2_LIMIT:
        raise AssertionError(
            f"SAR relative L2 error {relative_l2:.4g} exceeds {RELATIVE_L2_LIMIT:.4g}"
        )
    if maximum_relative > MAXIMUM_POINTWISE_RELATIVE_LIMIT:
        raise AssertionError(
            "SAR maximum pointwise relative error "
            f"{maximum_relative:.4g} exceeds {MAXIMUM_POINTWISE_RELATIVE_LIMIT:.4g}"
        )
    return metrics


def run_sweep(
    output_dir: Path,
    interface_mode="averaged",
    backend="cpu",
    precision="double",
):
    """Validate one broadband run at several requested SAR frequencies."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "lossy_halfspace_sar_sweep"
    scene = build_scene(
        interface_mode=interface_mode,
        frequencies=SWEEP_FREQUENCIES,
        waveform_frequency=SWEEP_WAVEFORM_FREQUENCY,
    )
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output_base,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **_solver_options(backend, precision),
    )
    with h5py.File(output_base.with_suffix(".h5"), "r") as output:
        group = output["sar/halfspace"]
        cells = np.asarray(group["cell_indices"])
        numerical_all = np.asarray(group["sar"])
        source_valid = np.asarray(group["source_valid"], dtype=bool)
        mesh_valid = np.asarray(group["mesh_valid"], dtype=bool)
        cells_per_wavelength = np.asarray(group["cells_per_wavelength"])

    centre_y = int(round(0.040 / DL))
    centre_z = int(round(0.040 / DL))
    selected = (cells[:, 1] == centre_y) & (cells[:, 2] == centre_z)
    x = (cells[selected, 0] + 0.5) * DL
    order = np.argsort(x)
    x = x[order]
    retained = (x >= INTERFACE_X + 2 * DL) & (x <= 0.096)
    x = x[retained]
    numerical_all = numerical_all[:, selected][:, order][:, retained]
    effective_interface = INTERFACE_X + (
        0.5 * DL if interface_mode == "noavg_free_space_last" else 0.0
    )
    depth = x - effective_interface

    rows = []
    for index, frequency in enumerate(SWEEP_FREQUENCIES):
        analytical = analytical_cell_average_sar(depth, frequency=frequency)
        analytical_collocated = analytical_sar(depth, frequency=frequency)
        analytical_point = analytical_sar(depth, frequency=frequency, yee_collocated=False)
        numerical = numerical_all[index]
        rows.append(
            {
                "frequency_hz": float(frequency),
                "cells_per_wavelength": float(cells_per_wavelength[index]),
                "source_valid": bool(source_valid[index]),
                "mesh_valid": bool(mesh_valid[index]),
                "relative_l2_error": float(
                    np.linalg.norm(numerical - analytical) / np.linalg.norm(analytical)
                ),
                "maximum_pointwise_relative_error": float(
                    np.max(np.abs(numerical - analytical) / analytical)
                ),
                "yee_collocated_relative_l2_error": float(
                    np.linalg.norm(numerical - analytical_collocated)
                    / np.linalg.norm(analytical_collocated)
                ),
                "yee_collocated_maximum_pointwise_relative_error": float(
                    np.max(np.abs(numerical - analytical_collocated) / analytical_collocated)
                ),
                "continuous_centre_relative_l2_error": float(
                    np.linalg.norm(numerical - analytical_point) / np.linalg.norm(analytical_point)
                ),
                "continuous_centre_maximum_pointwise_relative_error": float(
                    np.max(np.abs(numerical - analytical_point) / analytical_point)
                ),
            }
        )
    metrics = {
        "interface_mode": interface_mode,
        "backend": backend,
        "precision": precision,
        "declared_interface_m": INTERFACE_X,
        "effective_interface_m": effective_interface,
        "dl_m": DL,
        "comparison_points_per_frequency": int(x.size),
        "frequencies": rows,
        "maximum_relative_l2_error": max(row["relative_l2_error"] for row in rows),
        "maximum_pointwise_relative_error": max(
            row["maximum_pointwise_relative_error"] for row in rows
        ),
        "maximum_allowed_relative_l2_error": SWEEP_RELATIVE_L2_LIMIT,
        "maximum_allowed_pointwise_relative_error": (SWEEP_MAXIMUM_POINTWISE_RELATIVE_LIMIT),
    }
    (output_dir / "lossy_halfspace_sar_sweep_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    frequency_ghz = SWEEP_FREQUENCIES / 1e9
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        frequency_ghz,
        100 * np.asarray([row["relative_l2_error"] for row in rows]),
        "ko-",
        markerfacecolor="white",
        label=r"$L_2$ error",
    )
    axis.plot(
        frequency_ghz,
        100 * np.asarray([row["maximum_pointwise_relative_error"] for row in rows]),
        "ks--",
        markerfacecolor="white",
        label="Maximum point error",
    )
    axis.set_xlabel("Frequency [GHz]")
    axis.set_ylabel("Cell-average SAR error [%]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lossy_halfspace_sar_sweep_error.png", dpi=180)
    plt.close(fig)
    if not np.all(source_valid & mesh_valid):
        raise AssertionError("SAR sweep contains an invalid source or mesh frequency")
    if metrics["maximum_relative_l2_error"] > SWEEP_RELATIVE_L2_LIMIT:
        raise AssertionError(
            "SAR sweep relative L2 error "
            f"{metrics['maximum_relative_l2_error']:.4g} exceeds "
            f"{SWEEP_RELATIVE_L2_LIMIT:.4g}"
        )
    if metrics["maximum_pointwise_relative_error"] > SWEEP_MAXIMUM_POINTWISE_RELATIVE_LIMIT:
        raise AssertionError(
            "SAR sweep maximum pointwise relative error "
            f"{metrics['maximum_pointwise_relative_error']:.4g} exceeds "
            f"{SWEEP_MAXIMUM_POINTWISE_RELATIVE_LIMIT:.4g}"
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("sar_halfspace_results"))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "cuda", "opencl"), default="cpu")
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument(
        "--interface-mode",
        choices=("averaged", "noavg_free_space_last"),
        default="averaged",
    )
    args = parser.parse_args()
    if args.sweep:
        metrics = run_sweep(
            args.output_dir,
            interface_mode=args.interface_mode,
            backend=args.backend,
            precision=args.precision,
        )
    else:
        metrics = run(
            args.output_dir,
            benchmark=args.benchmark,
            interface_mode=args.interface_mode,
            backend=args.backend,
            precision=args.precision,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
