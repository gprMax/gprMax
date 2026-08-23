"""Run the offline, band-limited dispersive-interface pole-reduction study."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gprmax-matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from .coated_sphere_mie import coated_sphere_backscatter_rcs
from .layered_media import PlanarMedium, normal_incidence_reflection
from .pole_models import (
    DispersiveModel,
    arithmetic_mix,
    debye_term,
    drude_term,
    lorentz_term,
    make_material,
)
from .reduction import ReductionResult, ReductionTemplate, fit_projected_model, fit_reduced_model

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _materials():
    soil_a = make_material(
        "Puerto Rico type A",
        3.2,
        (debye_term(0.75, 2.71e-9), debye_term(0.30, 0.108e-9)),
        conductivity=0.397e-3,
    )
    soil_c = make_material(
        "Puerto Rico type C",
        6.0,
        (debye_term(2.75, 3.98e-9), debye_term(0.75, 0.251e-9)),
        conductivity=2.0e-3,
    )
    hybrid_a = make_material(
        "hybrid A",
        2.5,
        (
            debye_term(4.0, 0.5e-9),
            lorentz_term(2.0, 1.2e9, 0.08 * 2 * np.pi * 1.2e9),
        ),
    )
    hybrid_b = make_material(
        "hybrid B",
        5.0,
        (
            debye_term(1.5, 0.08e-9),
            lorentz_term(3.0, 2.8e9, 0.12 * 2 * np.pi * 2.8e9),
        ),
    )
    metal_a = make_material(
        "Drude-Lorentz A",
        3.0,
        (
            drude_term(2.8e9, 0.25 * 2 * np.pi * 1e9),
            lorentz_term(2.0, 4.0e9, 0.08 * 2 * np.pi * 4.0e9),
        ),
    )
    metal_b = make_material(
        "Drude-Lorentz B",
        3.3,
        (
            drude_term(3.0e9, 0.30 * 2 * np.pi * 1e9),
            lorentz_term(2.3, 4.2e9, 0.09 * 2 * np.pi * 4.2e9),
        ),
    )
    return soil_a, soil_c, hybrid_a, hybrid_b, metal_a, metal_b


def _serialise_model(model: DispersiveModel) -> dict:
    return {
        "name": model.name,
        "epsilon_inf": model.epsilon_inf,
        "conductivity": model.conductivity,
        "inclusive_order": model.inclusive_order,
        "rational_order": model.rational_order,
        "poles": [
            {
                "kind": pole.kind,
                "source": pole.source,
                "w_real": float(pole.w.real),
                "w_imag": float(pole.w.imag),
                "q_real": float(pole.q.real),
                "q_imag": float(pole.q.imag),
            }
            for pole in model.poles
        ],
    }


def _serialise_fit(result: ReductionResult) -> dict:
    return {
        "method": result.method,
        "template": result.template.label(),
        "metrics": result.metrics,
        "cost": result.cost,
        "evaluations": result.evaluations,
        "success": result.success,
        "message": result.message,
        "model": _serialise_model(result.model),
    }


def _run_fits(target, frequencies, templates, *, fixed_conductivity=0.0):
    return [
        fit_reduced_model(
            target,
            frequencies,
            template,
            fixed_conductivity=fixed_conductivity,
            restarts=5,
            max_evaluations=12_000,
        )
        for template in templates
    ]


def _run_projected_fits(target, frequencies, templates, *, fixed_conductivity=0.0):
    return [
        fit_projected_model(
            target,
            frequencies,
            template,
            fixed_conductivity=fixed_conductivity,
            maximum_iterations=180,
            population_size=10,
        )
        for template in templates
    ]


def _plot_response(path, title, target, fits, frequencies):
    target_values = target.relative_permittivity(frequencies)
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    axes[0].semilogx(
        frequencies, target_values.real, color="black", linewidth=2, label="exact pole union"
    )
    axes[1].semilogx(
        frequencies, -target_values.imag, color="black", linewidth=2, label="exact pole union"
    )
    styles = (("--", "o"), (":", "s"), ("-.", "^"), ((0, (5, 1)), "x"))
    for index, result in enumerate(fits):
        values = result.model.relative_permittivity(frequencies)
        label = result.template.label()
        linestyle, marker = styles[index % len(styles)]
        plot_options = {
            "color": "black",
            "linestyle": linestyle,
            "marker": marker,
            "markerfacecolor": "none",
            "markevery": 18,
            "markersize": 4,
            "label": label,
        }
        axes[0].semilogx(frequencies, values.real, **plot_options)
        axes[1].semilogx(frequencies, -values.imag, **plot_options)
        relative = np.abs(values - target_values) / np.maximum(
            np.abs(target_values), np.finfo(float).eps
        )
        axes[2].loglog(frequencies, relative, **plot_options)
    axes[0].set_ylabel(r"$\Re\{\epsilon_r\}$")
    axes[1].set_ylabel(r"$-\Im\{\epsilon_r\}$")
    axes[2].set(xlabel="Frequency [Hz]", ylabel="Relative complex error")
    axes[0].set_title(title)
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_analytical_references(path, soil_a, soil_c):
    frequencies = np.geomspace(20e6, 2.5e9, 260)
    free_space = make_material("free space", 1.0)
    reflection = normal_incidence_reflection(
        frequencies,
        PlanarMedium(free_space),
        (PlanarMedium(soil_a, thickness=0.08),),
        PlanarMedium(soil_c),
    )
    rcs = coated_sphere_backscatter_rcs(
        frequencies,
        core_radius=0.035,
        outer_radius=0.060,
        core=soil_c,
        shell=soil_a,
    )
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].semilogx(frequencies, np.abs(reflection), label=r"$|\Gamma|$")
    phase = np.unwrap(np.angle(reflection)) * 180 / np.pi
    phase_axis = axes[0].twinx()
    phase_axis.semilogx(frequencies, phase, color="tab:orange", label="phase")
    axes[0].set(ylabel=r"$|\Gamma|$", title="Exact dispersive planar-layer reference")
    phase_axis.set_ylabel("Unwrapped phase [deg]", color="tab:orange")
    axes[1].loglog(frequencies, rcs / (np.pi * 0.060**2), color="tab:green")
    axes[1].set(
        xlabel="Frequency [Hz]",
        ylabel=r"$\sigma_b/(\pi a^2)$",
        title="Exact dispersive core-shell Mie reference",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return frequencies, reflection, rcs


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    soil_a, soil_c, hybrid_a, hybrid_b, metal_a, metal_b = _materials()

    studies = {}
    soil_frequency = np.geomspace(10e6, 3e9, 220)
    soil_target = arithmetic_mix((soil_a, soil_c), (0.5, 0.5), name="two-soil exact mixture")
    soil_templates = (
        ReductionTemplate(debye=1),
        ReductionTemplate(debye=2),
        ReductionTemplate(debye=3),
    )
    soil_baseline_fits = _run_fits(
        soil_target,
        soil_frequency,
        soil_templates,
        fixed_conductivity=soil_target.conductivity,
    )
    soil_fits = _run_projected_fits(
        soil_target,
        soil_frequency,
        soil_templates,
        fixed_conductivity=soil_target.conductivity,
    )
    _plot_response(
        RESULTS / "soil_debye_reduction.png",
        "Related two-pole Debye soils",
        soil_target,
        soil_baseline_fits,
        soil_frequency,
    )
    studies["related_debye_soils"] = {
        "frequency_range_hz": [float(soil_frequency[0]), float(soil_frequency[-1])],
        "target": _serialise_model(soil_target),
        "fits": [_serialise_fit(result) for result in soil_baseline_fits],
        "projected_fits": [_serialise_fit(result) for result in soil_fits],
    }

    hybrid_frequency = np.geomspace(50e6, 5e9, 240)
    hybrid_target = arithmetic_mix(
        (hybrid_a, hybrid_b), (0.5, 0.5), name="two-hybrid exact mixture"
    )
    hybrid_templates = (
        ReductionTemplate(debye=1, lorentz=1),
        ReductionTemplate(debye=1, lorentz=2),
        ReductionTemplate(debye=2, lorentz=2),
    )
    hybrid_baseline_fits = _run_fits(
        hybrid_target,
        hybrid_frequency,
        hybrid_templates,
    )
    hybrid_fits = _run_projected_fits(
        hybrid_target,
        hybrid_frequency,
        hybrid_templates,
    )
    _plot_response(
        RESULTS / "debye_lorentz_reduction.png",
        "Two dissimilar Debye-Lorentz materials",
        hybrid_target,
        hybrid_baseline_fits,
        hybrid_frequency,
    )
    studies["debye_lorentz"] = {
        "frequency_range_hz": [float(hybrid_frequency[0]), float(hybrid_frequency[-1])],
        "target": _serialise_model(hybrid_target),
        "fits": [_serialise_fit(result) for result in hybrid_baseline_fits],
        "projected_fits": [_serialise_fit(result) for result in hybrid_fits],
    }

    metal_frequency = np.geomspace(100e6, 8e9, 240)
    metal_target = arithmetic_mix((metal_a, metal_b), (0.5, 0.5), name="two-metal exact mixture")
    metal_templates = (
        ReductionTemplate(lorentz=1, drude=1),
        ReductionTemplate(lorentz=2, drude=1),
        ReductionTemplate(lorentz=2, drude=2),
    )
    metal_baseline_fits = _run_fits(
        metal_target,
        metal_frequency,
        metal_templates,
    )
    metal_fits = _run_projected_fits(
        metal_target,
        metal_frequency,
        metal_templates,
    )
    _plot_response(
        RESULTS / "drude_lorentz_reduction.png",
        "Two related Drude-Lorentz materials",
        metal_target,
        metal_baseline_fits,
        metal_frequency,
    )
    studies["drude_lorentz"] = {
        "frequency_range_hz": [float(metal_frequency[0]), float(metal_frequency[-1])],
        "target": _serialise_model(metal_target),
        "fits": [_serialise_fit(result) for result in metal_baseline_fits],
        "projected_fits": [_serialise_fit(result) for result in metal_fits],
    }

    reference_frequency, reflection, rcs = _plot_analytical_references(
        RESULTS / "analytical_layered_references.png", soil_a, soil_c
    )
    np.savetxt(
        RESULTS / "analytical_layered_references.csv",
        np.column_stack((reference_frequency, reflection.real, reflection.imag, rcs)),
        delimiter=",",
        header="frequency_hz,reflection_real,reflection_imag,core_shell_backscatter_rcs_m2",
        comments="",
    )
    summary = {
        "convention": "engineering exp(+j omega t)",
        "interface_fractions": [0.5, 0.5],
        "studies": studies,
    }
    (RESULTS / "pole_reduction_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(RESULTS / "pole_reduction_metrics.json")


if __name__ == "__main__":
    main()
