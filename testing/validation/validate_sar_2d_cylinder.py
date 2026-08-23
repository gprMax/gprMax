"""Validate 2-D TM/TE SAR against the exact lossy-cylinder Mie series.

The primary case uses the homogeneous muscle-cylinder parameters reported by
Gasmelseed [1]: radius 6 cm, relative permittivity 48.9, conductivity
4.61 S/m, 5.5 GHz illumination, and 0.4 mm cells. That paper presents a
graphical TMz comparison but does not report numerical cylinder-error
estimates. The metrics in this driver are therefore new comparisons against
the independent cylindrical series. Fat and skin cases and the TEz solution
are explicitly additional tests, not results reported in [1].

For a fair comparison, the analytical fields are evaluated at the actual Yee
edges and complex-collocated exactly as the production SAR output does before
forming ``sigma * |E|^2 / (2 * rho)``.  The continuous-cylinder absorbed power
per unit invariant length is also evaluated by quadrature.

References
----------
[1] A. Gasmelseed, "Snell-corrected electric parameter scaling for 2-D FDTD
    dispersion reduction in layered tissues," Biomedical Optics Express,
    17(4), 1928-1935, 2026. doi:10.1364/BOE.590271.
[2] J. R. Wait, "Scattering of a plane wave from a circular dielectric
    cylinder at oblique incidence," Canadian Journal of Physics, 33(5),
    189-195, 1955. doi:10.1139/P55-024.

Run from the repository root, for example::

    python -m testing.validation.validate_sar_2d_cylinder --backend cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import c, epsilon_0, mu_0
from scipy.special import h2vp, hankel2, jv, jvp

import gprMax

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

INF = float("inf")
FREQUENCY = 5.5e9
SOURCE_FREQUENCY = 3.5e9
DL = 0.4e-3
TIME_WINDOW = 8e-9
DOMAIN = (0.18, 0.18, INF)
CENTRE = np.asarray((0.09, 0.09), dtype=np.float64)
RADIUS = 0.06
RELATIVE_PERMITTIVITY = 48.9
CONDUCTIVITY = 4.61
# Density only scales SAR; it does not change the electromagnetic solution.
# A representative muscle value is used consistently for numerical and Mie
# results because the cited cylinder paragraph specifies epsilon_r and sigma,
# but not a density value.
DENSITY = 1090.0
PML_CELLS = 12
TFSF_LOWER = (0.012, 0.012, INF)
TFSF_UPPER = (0.168, 0.168, INF)
RELATIVE_FLOOR = 0.05
INTERIOR_CELLS = 2
LOCAL_RELATIVE_L2_LIMIT = 0.05
LOCAL_MAXIMUM_RELATIVE_LIMIT = 0.20
ABSORBED_POWER_RELATIVE_LIMIT = 0.02
MIE_RESIDUAL_LIMIT = 1e-12


@dataclass(frozen=True)
class CylinderMaterial:
    """Homogeneous cylinder properties used by the exact/FDTD comparison."""

    name: str
    relative_permittivity: float
    conductivity: float
    density: float

    @property
    def tag(self):
        return f"{self.name}_cylinder"


MATERIAL_CASES = {
    "fat": CylinderMaterial("fat", 4.983, 0.274, 911.0),
    "skin": CylinderMaterial("skin", 35.36, 3.463, 1109.0),
    "muscle": CylinderMaterial("muscle", RELATIVE_PERMITTIVITY, CONDUCTIVITY, DENSITY),
}
DEFAULT_MATERIAL = MATERIAL_CASES["muscle"]


def _complex_relative_permittivity(frequency=FREQUENCY, material=DEFAULT_MATERIAL):
    """Return epsilon_r for the exp(+j omega t) convention used by gprMax."""

    omega = 2 * np.pi * frequency
    return material.relative_permittivity + material.conductivity / (1j * omega * epsilon_0)


def _series_coefficients(mode, frequency=FREQUENCY, material=DEFAULT_MATERIAL):
    """Return internal cylindrical-wave coefficients for TMz or TEz."""

    mode = mode.upper()
    if mode not in ("TM", "TE"):
        raise ValueError("mode must be 'TM' or 'TE'")
    k0 = 2 * np.pi * frequency / c
    epsilon_r = _complex_relative_permittivity(frequency, material)
    refractive_index = np.sqrt(epsilon_r)
    size = k0 * RADIUS
    internal_size = refractive_index * size
    maximum_argument = max(abs(size), abs(internal_size))
    maximum_order = int(np.ceil(maximum_argument + 4.05 * maximum_argument ** (1 / 3) + 12))
    orders = np.arange(-maximum_order, maximum_order + 1, dtype=np.int32)

    outside = jv(orders, size)
    outside_derivative = jvp(orders, size)
    outgoing = hankel2(orders, size)
    outgoing_derivative = h2vp(orders, size)
    inside = jv(orders, internal_size)
    inside_derivative = jvp(orders, internal_size)
    derivative_factor = refractive_index if mode == "TM" else 1 / refractive_index
    numerator = outside * outgoing_derivative - outside_derivative * outgoing
    denominator = inside * outgoing_derivative - derivative_factor * inside_derivative * outgoing
    internal = numerator / denominator
    incident = np.power(-1j, orders)

    # An independent algebraic residual catches coefficient/convention errors
    # before any FDTD comparison is attempted.
    scattered = (internal * inside - outside) / outgoing
    scalar_residual = outside + scattered * outgoing - internal * inside
    derivative_residual = (
        outside_derivative + scattered * outgoing_derivative - derivative_factor * internal * inside_derivative
    )
    scale = np.maximum(
        np.abs(outside) + np.abs(scattered * outgoing) + np.abs(internal * inside),
        np.finfo(np.float64).tiny,
    )
    derivative_scale = np.maximum(
        np.abs(outside_derivative)
        + np.abs(scattered * outgoing_derivative)
        + np.abs(derivative_factor * internal * inside_derivative),
        np.finfo(np.float64).tiny,
    )
    residual = max(
        float(np.max(np.abs(scalar_residual) / scale)),
        float(np.max(np.abs(derivative_residual) / derivative_scale)),
    )
    return orders, incident * internal, refractive_index * k0, residual


def mie_electric_field(
    mode,
    x,
    y,
    *,
    frequency=FREQUENCY,
    material=DEFAULT_MATERIAL,
    chunk_size=4096,
):
    """Return exact internal Cartesian E phasors for unit incident E."""

    mode = mode.upper()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    shape = np.broadcast_shapes(x.shape, y.shape)
    x = np.broadcast_to(x, shape).ravel()
    y = np.broadcast_to(y, shape).ravel()
    orders, coefficients, k1, residual = _series_coefficients(mode, frequency, material)
    omega = 2 * np.pi * frequency
    epsilon = epsilon_0 * _complex_relative_permittivity(frequency, material)
    impedance_0 = np.sqrt(mu_0 / epsilon_0)

    ex = np.zeros(x.size, dtype=np.complex128)
    ey = np.zeros(x.size, dtype=np.complex128)
    ez = np.zeros(x.size, dtype=np.complex128)
    for start in range(0, x.size, chunk_size):
        stop = min(start + chunk_size, x.size)
        local_x = x[start:stop] - CENTRE[0]
        local_y = y[start:stop] - CENTRE[1]
        radius = np.hypot(local_x, local_y)
        angle = np.arctan2(local_y, local_x)
        phase = np.exp(1j * orders[:, np.newaxis] * angle[np.newaxis, :])
        argument = k1 * radius
        bessel = jv(orders[:, np.newaxis], argument[np.newaxis, :])
        terms = coefficients[:, np.newaxis] * bessel * phase

        if mode == "TM":
            ez[start:stop] = np.sum(terms, axis=0)
            continue

        derivative = jvp(orders[:, np.newaxis], argument[np.newaxis, :])
        hz_terms = terms / impedance_0
        radial_derivative = np.sum(
            coefficients[:, np.newaxis] * k1 * derivative * phase / impedance_0,
            axis=0,
        )
        azimuthal_derivative = np.sum(
            1j * orders[:, np.newaxis] * hz_terms,
            axis=0,
        )
        # Staggered TE electric samples never lie exactly on the cylinder
        # axis for this grid placement. Keep the guard explicit for reuse.
        if np.any(radius == 0):
            raise ValueError("TE electric field was requested exactly on the cylinder axis")
        electric_r = azimuthal_derivative / (1j * omega * epsilon * radius)
        electric_phi = -radial_derivative / (1j * omega * epsilon)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        ex[start:stop] = electric_r * cosine - electric_phi * sine
        ey[start:stop] = electric_r * sine + electric_phi * cosine

    return (
        ex.reshape(shape),
        ey.reshape(shape),
        ez.reshape(shape),
        residual,
    )


def mie_collocated_sar(mode, cells, dl=DL, *, frequency=FREQUENCY, material=DEFAULT_MATERIAL):
    """Evaluate Mie fields at gprMax's actual Yee edges and collocate them."""

    cells = np.asarray(cells, dtype=np.int64)
    i = cells[:, 0].astype(np.float64)
    j = cells[:, 1].astype(np.float64)
    mode = mode.upper()
    residual = 0.0
    if mode == "TM":
        ez_values = []
        for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            _, _, ez, current = mie_electric_field(
                mode,
                (i + ox) * dl,
                (j + oy) * dl,
                frequency=frequency,
                material=material,
            )
            ez_values.append(ez)
            residual = max(residual, current)
        ez_cell = np.mean(ez_values, axis=0)
        electric_squared = np.abs(ez_cell) ** 2
    elif mode == "TE":
        ex_values = []
        for oy in (0, 1):
            ex, _, _, current = mie_electric_field(
                mode,
                (i + 0.5) * dl,
                (j + oy) * dl,
                frequency=frequency,
                material=material,
            )
            ex_values.append(ex)
            residual = max(residual, current)
        ey_values = []
        for ox in (0, 1):
            _, ey, _, current = mie_electric_field(
                mode,
                (i + ox) * dl,
                (j + 0.5) * dl,
                frequency=frequency,
                material=material,
            )
            ey_values.append(ey)
            residual = max(residual, current)
        ex_cell = np.mean(ex_values, axis=0)
        ey_cell = np.mean(ey_values, axis=0)
        electric_squared = np.abs(ex_cell) ** 2 + np.abs(ey_cell) ** 2
    else:
        raise ValueError("mode must be 'TM' or 'TE'")
    return material.conductivity * electric_squared / (2 * material.density), residual


def mie_absorbed_power_per_length(
    mode,
    *,
    frequency=FREQUENCY,
    material=DEFAULT_MATERIAL,
    quadrature_order=600,
):
    """Integrate the exact continuous SAR power density over the cylinder."""

    orders, coefficients, k1, residual = _series_coefficients(mode, frequency, material)
    nodes, weights = leggauss(quadrature_order)
    radius = 0.5 * RADIUS * (nodes + 1)
    radial_weights = 0.5 * RADIUS * weights
    argument = k1 * radius
    bessel = jv(orders[:, np.newaxis], argument[np.newaxis, :])
    if mode.upper() == "TM":
        angular_electric_squared = (
            2
            * np.pi
            * np.sum(
                np.abs(coefficients[:, np.newaxis] * bessel) ** 2,
                axis=0,
            )
        )
    else:
        derivative = jvp(orders[:, np.newaxis], argument[np.newaxis, :])
        omega = 2 * np.pi * frequency
        epsilon = epsilon_0 * _complex_relative_permittivity(frequency, material)
        impedance_0 = np.sqrt(mu_0 / epsilon_0)
        common = coefficients[:, np.newaxis] / impedance_0
        radial = orders[:, np.newaxis] * common * bessel / radius[np.newaxis, :]
        azimuthal = common * k1 * derivative
        angular_electric_squared = (
            2 * np.pi * np.sum(np.abs(radial) ** 2 + np.abs(azimuthal) ** 2, axis=0) / abs(omega * epsilon) ** 2
        )
    electric_integral = np.sum(radial_weights * radius * angular_electric_squared)
    return float(0.5 * material.conductivity * electric_integral), residual


def build_scene(mode, *, material=DEFAULT_MATERIAL, dl=DL, time_window=TIME_WINDOW, nthreads=8):
    """Build a homogeneous lossy-cylinder case for one polarisation."""

    mode = mode.upper()
    if mode not in ("TM", "TE"):
        raise ValueError("mode must be 'TM' or 'TE'")
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=time_window))
    scene.add(gprMax.PMLThickness(thickness=(PML_CELLS, PML_CELLS, 0, PML_CELLS, PML_CELLS, 0)))
    scene.add(gprMax.OMPThreads(n=nthreads))
    scene.add(
        gprMax.Material(
            er=material.relative_permittivity,
            se=material.conductivity,
            mr=1,
            sm=0,
            id=material.name,
        )
    )
    scene.add(gprMax.MaterialDensity(density=material.density, material_ids=material.name))
    scene.add(
        gprMax.Cylinder(
            p1=(CENTRE[0], CENTRE[1], INF),
            p2=(CENTRE[0], CENTRE[1], INF),
            r=RADIUS,
            material_id=material.name,
            averaging="y",
            tag=material.tag,
        )
    )
    # The completed pulse avoids the position-dependent finite-record bias of
    # a truncated continuous sine. Its 5.5 GHz bin remains strong, while its
    # significant upper spectrum remains compatible with the published grid.
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=SOURCE_FREQUENCY, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=TFSF_LOWER,
            p2=TFSF_UPPER,
            axis="x",
            psi=90 if mode == "TM" else 0,
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.SAR(
            frequencies=(FREQUENCY,),
            waveform_id="pulse",
            tags=material.tag,
            id=f"cylinder_{mode.lower()}",
            target_amplitude=1.0,
            spectrum_limit=10,
            averaging_masses=(),
        )
    )
    scene.add(
        gprMax.Radiometry(
            frequencies=(FREQUENCY,),
            waveform_id="pulse",
            tags=material.tag,
            id=f"cylinder_{mode.lower()}_cross_section",
            normalisation="incident_flux",
            target_flux=1.0,
            spectrum_limit=10,
        )
    )
    return scene


def _solver_options(backend, precision):
    if backend == "cpu":
        return {"cpu_precision": precision}
    if backend == "cuda":
        return {"gpu": [0], "gpu_precision": precision}
    if backend == "opencl":
        return {"opencl": [0], "gpu_precision": precision}
    raise ValueError("backend must be 'cpu', 'cuda', or 'opencl'")


def _comparison_metrics(cells, numerical, analytical, dl):
    centre_coordinates = (cells[:, :2] + 0.5) * dl
    radii = np.linalg.norm(centre_coordinates - CENTRE, axis=1)
    floor = analytical >= RELATIVE_FLOOR * np.max(analytical)
    interior = radii <= RADIUS - INTERIOR_CELLS * dl

    def calculate(selection):
        difference = numerical[selection] - analytical[selection]
        relative_l2 = np.linalg.norm(difference) / np.linalg.norm(analytical[selection])
        maximum_relative = np.max(np.abs(difference) / analytical[selection])
        return {
            "cell_count": int(np.count_nonzero(selection)),
            "relative_l2_error": float(relative_l2),
            "maximum_pointwise_relative_error": float(maximum_relative),
        }

    return {
        "above_5_percent_peak": calculate(floor),
        "interior_above_5_percent_peak": calculate(floor & interior),
    }


def run_case(
    output_dir,
    mode,
    *,
    material=DEFAULT_MATERIAL,
    dl=DL,
    time_window=TIME_WINDOW,
    backend="cpu",
    precision="double",
):
    """Run one mode and write its numerical/analytical comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    mode = mode.upper()
    output_base = output_dir / f"sar_2d_cylinder_{material.name}_{mode.lower()}_{backend}"
    start = perf_counter()
    gprMax.run(
        scenes=[build_scene(mode, material=material, dl=dl, time_window=time_window)],
        n=1,
        outputfile=output_base,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **_solver_options(backend, precision),
    )
    runtime = perf_counter() - start

    with h5py.File(output_base.with_suffix(".h5"), "r") as output:
        group = output[f"sar/cylinder_{mode.lower()}"]
        cells = np.asarray(group["cell_indices"])
        numerical = np.asarray(group["sar"])[0]
        numerical_power = float(group[f"tags/{material.tag}/absorbed_power_per_length"][0])
        valid = bool(group["valid"][0])
        source_valid = bool(group["source_valid"][0])
        mesh_valid = bool(group["mesh_valid"][0])
        collection_backend = str(group.attrs["CollectionBackend"])
        radiometry = output[f"radiometry/cylinder_{mode.lower()}_cross_section"]
        numerical_cross_section = float(
            radiometry[f"tags/{material.tag}/normalised_absorption"][0]
        )

    analytical, coefficient_residual = mie_collocated_sar(mode, cells, dl, material=material)
    analytical_power, power_coefficient_residual = mie_absorbed_power_per_length(mode, material=material)
    incident_flux = 1 / (2 * np.sqrt(mu_0 / epsilon_0))
    analytical_cross_section = analytical_power / incident_flux
    comparison = _comparison_metrics(cells, numerical, analytical, dl)
    voxelised_area = cells.shape[0] * dl**2
    exact_area = np.pi * RADIUS**2
    checks = {
        "source_and_mesh_valid": bool(source_valid and mesh_valid and valid),
        "mie_boundary_residual": bool(max(coefficient_residual, power_coefficient_residual) <= MIE_RESIDUAL_LIMIT),
        "interior_local_relative_l2": bool(
            comparison["interior_above_5_percent_peak"]["relative_l2_error"] <= LOCAL_RELATIVE_L2_LIMIT
        ),
        "interior_local_maximum_relative": bool(
            comparison["interior_above_5_percent_peak"]["maximum_pointwise_relative_error"]
            <= LOCAL_MAXIMUM_RELATIVE_LIMIT
        ),
        "absorbed_power_per_length": bool(abs(numerical_power / analytical_power - 1) <= ABSORBED_POWER_RELATIVE_LIMIT),
        "radiometry_cross_section_per_length": bool(
            abs(numerical_cross_section / analytical_cross_section - 1)
            <= ABSORBED_POWER_RELATIVE_LIMIT
        ),
    }
    metrics = {
        "literature_case": {
            "citation": "Gasmelseed, Biomedical Optics Express 17(4), 2026",
            "doi": "10.1364/BOE.590271",
            "reported_polarisation": "TMz",
            "te_status": "independent exact-series extension, not reported in the cited case",
            "case_status": (
                "reported homogeneous-cylinder material"
                if material.name == "muscle"
                else "new homogeneous-cylinder extension using reported tissue properties"
            ),
        },
        "mode": mode,
        "backend": backend,
        "collection_backend": collection_backend,
        "precision": precision,
        "frequency_hz": FREQUENCY,
        "source_centre_frequency_hz": SOURCE_FREQUENCY,
        "dl_m": dl,
        "time_window_s": time_window,
        "radius_m": RADIUS,
        "radius_cells": RADIUS / dl,
        "material": material.name,
        "relative_permittivity": material.relative_permittivity,
        "conductivity_S_per_m": material.conductivity,
        "density_kg_per_m3": material.density,
        "density_note": (
            "representative tissue density; identical scaling applied to both "
            "solutions and therefore not part of the electromagnetic validation"
        ),
        "tagged_cells": int(cells.shape[0]),
        "voxelised_area_m2": voxelised_area,
        "continuous_area_m2": exact_area,
        "relative_area_error": float(voxelised_area / exact_area - 1),
        "source_valid": source_valid,
        "mesh_valid": mesh_valid,
        "valid": valid,
        "mie_boundary_condition_relative_residual": max(coefficient_residual, power_coefficient_residual),
        "local_sar": comparison,
        "boundary_interpretation": (
            "The primary local check excludes a two-cell boundary band because the "
            "continuous circular interface and the staircased/smoothed Yee interface "
            "are not identical geometries. All-cell metrics remain reported."
        ),
        "analytical_absorbed_power_per_length_W_per_m": analytical_power,
        "fdtd_absorbed_power_per_length_W_per_m": numerical_power,
        "relative_absorbed_power_error": float(abs(numerical_power / analytical_power - 1)),
        "analytical_absorption_cross_section_per_length_m": analytical_cross_section,
        "fdtd_radiometry_absorption_cross_section_per_length_m": numerical_cross_section,
        "relative_radiometry_cross_section_error": float(
            abs(numerical_cross_section / analytical_cross_section - 1)
        ),
        "runtime_seconds": runtime,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    (output_dir / f"sar_2d_cylinder_{material.name}_{mode.lower()}_{backend}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    np.savez_compressed(
        output_dir / f"sar_2d_cylinder_{material.name}_{mode.lower()}_{backend}.npz",
        cells=cells,
        numerical_sar=numerical,
        analytical_collocated_sar=analytical,
    )
    return metrics, cells, numerical, analytical


def plot_results(output_dir, results, material=DEFAULT_MATERIAL, dl=DL):
    """Plot nearest-centreline SAR and error for each completed mode."""

    figure, axes = plt.subplots(len(results), 2, figsize=(10, 4.1 * len(results)), squeeze=False)
    for row, (mode, (_, cells, numerical, analytical)) in enumerate(results.items()):
        coordinates = (cells[:, :2] + 0.5) * dl
        y_distance = np.abs(coordinates[:, 1] - CENTRE[1])
        line = y_distance == np.min(y_distance)
        x = coordinates[line, 0]
        numerical_line = numerical[line]
        analytical_line = analytical[line]
        order = np.argsort(x)
        x = x[order]
        numerical_line = numerical_line[order]
        analytical_line = analytical_line[order]
        axes[row, 0].plot(
            (x - CENTRE[0]) * 1e3,
            analytical_line,
            "k-",
            linewidth=1.4,
            label="cylindrical Mie series",
        )
        axes[row, 0].plot(
            (x - CENTRE[0]) * 1e3,
            numerical_line,
            linestyle="none",
            marker="o" if mode == "TM" else "s",
            markersize=3.2,
            markerfacecolor="none",
            markeredgecolor="black",
            markevery=max(1, x.size // 55),
            label=f"gprMax 2-D {mode}",
        )
        axes[row, 0].set(
            xlabel="Position along nearest centreline [mm]",
            ylabel="Local SAR [W/kg]",
            title=f"{material.name}: {mode}z, 5.5 GHz",
        )
        axes[row, 0].grid(True, color="0.85", linewidth=0.6)
        axes[row, 0].legend()

        relative = 100 * np.abs(numerical_line - analytical_line) / analytical_line
        retained = analytical_line >= RELATIVE_FLOOR * np.max(analytical_line)
        axes[row, 1].plot(
            (x[retained] - CENTRE[0]) * 1e3,
            relative[retained],
            "k-",
            linewidth=1.1,
        )
        axes[row, 1].set(
            xlabel="Position along nearest centreline [mm]",
            ylabel="Absolute relative error [%]",
            title="Cells above 5% of analytical peak",
        )
        axes[row, 1].grid(True, color="0.85", linewidth=0.6)
    figure.tight_layout()
    figure.savefig(output_dir / "sar_2d_cylinder_mie_comparison.png", dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("testing/validation/sar_2d_cylinder_results"),
    )
    parser.add_argument("--backend", choices=("cpu", "cuda", "opencl"), default="cpu")
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--modes", nargs="+", choices=("TM", "TE"), default=("TM", "TE"))
    parser.add_argument(
        "--material",
        choices=tuple(MATERIAL_CASES),
        default="muscle",
        help="homogeneous cylinder material case",
    )
    parser.add_argument("--dl", type=float, default=DL)
    parser.add_argument("--time-window", type=float, default=TIME_WINDOW)
    args = parser.parse_args()
    material = MATERIAL_CASES[args.material]

    results = {}
    for mode in args.modes:
        results[mode] = run_case(
            args.output_dir,
            mode,
            material=material,
            dl=args.dl,
            time_window=args.time_window,
            backend=args.backend,
            precision=args.precision,
        )
    plot_results(args.output_dir, results, material, args.dl)
    summary = {mode: result[0] for mode, result in results.items()}
    (args.output_dir / f"sar_2d_cylinder_{material.name}_{args.backend}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    failed = [mode for mode, metrics in summary.items() if not metrics["pass"]]
    if failed:
        raise AssertionError(f"2-D cylinder SAR validation failed for mode(s): {failed}")


if __name__ == "__main__":
    main()
