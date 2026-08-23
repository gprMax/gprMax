"""Constrained reduced-order fitting for inclusive dispersive mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from scipy.constants import epsilon_0
from scipy.optimize import differential_evolution, least_squares, nnls

from .pole_models import (
    DispersiveModel,
    debye_term,
    drude_term,
    lorentz_term,
    make_material,
    permittivity_error,
)


@dataclass(frozen=True)
class ReductionTemplate:
    """Numbers of physical terms allowed in a reduced material."""

    debye: int = 0
    lorentz: int = 0
    drude: int = 0

    @property
    def terms(self) -> int:
        return self.debye + self.lorentz + self.drude

    def label(self) -> str:
        return f"{self.debye}D+{self.lorentz}L+{self.drude}R"


@dataclass(frozen=True)
class ReductionResult:
    """Result and diagnostics from one constrained fit."""

    model: DispersiveModel
    template: ReductionTemplate
    metrics: dict[str, float]
    cost: float
    evaluations: int
    success: bool
    message: str
    method: str = "joint_least_squares"


def _logit(value: float) -> float:
    return float(np.log(value / (1.0 - value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = np.exp(-value)
        return float(1.0 / (1.0 + exponential))
    exponential = np.exp(value)
    return float(exponential / (1.0 + exponential))


def _initial_parameters(
    target: DispersiveModel,
    frequencies: np.ndarray,
    template: ReductionTemplate,
) -> np.ndarray:
    """Generate deterministic, physically admissible starting parameters."""

    low, high = float(frequencies[0]), float(frequencies[-1])
    sample = target.relative_permittivity(frequencies)
    strength = max(0.1, float(np.ptp(sample.real)), float(abs(sample.real[0] - target.epsilon_inf)))
    parameters: list[float] = []

    source_debye = sorted(
        (pole for pole in target.poles if pole.kind == "debye"),
        key=lambda pole: -1.0 / pole.q.real,
    )
    if template.debye:
        if source_debye:
            groups = np.array_split(np.asarray(source_debye, dtype=object), template.debye)
            for group in groups:
                if len(group):
                    deltas = np.asarray([-pole.w.real / pole.q.real for pole in group])
                    taus = np.asarray([-1.0 / pole.q.real for pole in group])
                    delta = max(float(np.sum(deltas)), 1e-12)
                    tau = float(np.exp(np.sum(deltas * np.log(taus)) / np.sum(deltas)))
                else:
                    delta = strength / template.debye
                    tau = 1.0 / (2 * np.pi * np.sqrt(low * high))
                parameters.extend((np.log(delta), np.log(tau)))
        else:
            relaxation_frequencies = np.geomspace(low, high, template.debye + 2)[1:-1]
            for relaxation_frequency in relaxation_frequencies:
                parameters.extend(
                    (
                        np.log(strength / max(template.debye, 1)),
                        np.log(1.0 / (2 * np.pi * relaxation_frequency)),
                    )
                )

    source_lorentz = sorted(
        (pole for pole in target.poles if pole.kind == "lorentz"),
        key=lambda pole: abs(pole.q),
    )
    if template.lorentz:
        if source_lorentz:
            groups = np.array_split(np.asarray(source_lorentz, dtype=object), template.lorentz)
            for group in groups:
                if len(group):
                    omega_0 = np.asarray([abs(pole.q) for pole in group])
                    damping = np.asarray([-pole.q.real for pole in group])
                    delta = np.asarray(
                        [-pole.w.imag * pole.q.imag / abs(pole.q) ** 2 for pole in group]
                    )
                    delta_sum = max(float(np.sum(delta)), 1e-12)
                    resonance = float(
                        np.exp(np.sum(delta * np.log(omega_0 / (2 * np.pi))) / np.sum(delta))
                    )
                    damping_ratio = float(np.sum(delta * damping / omega_0) / np.sum(delta))
                else:
                    delta_sum = strength / template.lorentz
                    resonance = np.sqrt(low * high)
                    damping_ratio = 0.08
                damping_ratio = float(np.clip(damping_ratio, 1.1e-5, 0.9799))
                scaled_ratio = (damping_ratio - 1e-5) / 0.97999
                parameters.extend((np.log(delta_sum), np.log(resonance), _logit(scaled_ratio)))
        else:
            resonances = np.geomspace(low * 1.2, high / 1.2, template.lorentz)
            for resonance in np.atleast_1d(resonances):
                parameters.extend(
                    (
                        np.log(strength / max(template.lorentz, 1)),
                        np.log(resonance),
                        _logit(0.08),
                    )
                )

    source_drude = sorted(
        (pole for pole in target.poles if pole.kind == "drude"),
        key=lambda pole: -pole.q.real,
    )
    if template.drude:
        if source_drude:
            groups = np.array_split(np.asarray(source_drude, dtype=object), template.drude)
            for group in groups:
                if len(group):
                    gamma = np.asarray([-pole.q.real for pole in group])
                    omega_p_squared = np.asarray(
                        [-pole.w.real * value for pole, value in zip(group, gamma)]
                    )
                    total_omega_p_squared = max(float(np.sum(omega_p_squared)), 1e-12)
                    collision_frequency = float(
                        np.exp(np.sum(omega_p_squared * np.log(gamma)) / np.sum(omega_p_squared))
                    )
                    plasma_frequency = np.sqrt(total_omega_p_squared) / (2 * np.pi)
                else:
                    plasma_frequency = np.sqrt(low * high)
                    collision_frequency = 2 * np.pi * low
                parameters.extend((np.log(plasma_frequency), np.log(collision_frequency)))
        else:
            plasma_frequencies = np.geomspace(np.sqrt(low * high), high, template.drude)
            collision_frequencies = np.geomspace(
                2 * np.pi * low, 2 * np.pi * np.sqrt(low * high), template.drude
            )
            for plasma_frequency, collision_frequency in zip(
                np.atleast_1d(plasma_frequencies), np.atleast_1d(collision_frequencies)
            ):
                parameters.extend((np.log(plasma_frequency), np.log(collision_frequency)))

    return np.asarray(parameters, dtype=float)


def _model_from_parameters(
    parameters: np.ndarray,
    template: ReductionTemplate,
    epsilon_inf: float,
    fixed_conductivity: float,
    name: str,
) -> DispersiveModel:
    terms: list[DispersiveModel] = []
    cursor = 0
    for index in range(template.debye):
        delta_epsilon, tau = np.exp(parameters[cursor : cursor + 2])
        cursor += 2
        terms.append(debye_term(delta_epsilon, tau, source=f"reduced D{index + 1}"))
    for index in range(template.lorentz):
        delta_epsilon = np.exp(parameters[cursor])
        resonance_frequency = np.exp(parameters[cursor + 1])
        damping_ratio = 1e-5 + 0.97999 * _sigmoid(parameters[cursor + 2])
        cursor += 3
        damping = damping_ratio * 2 * np.pi * resonance_frequency
        terms.append(
            lorentz_term(
                delta_epsilon,
                resonance_frequency,
                damping,
                source=f"reduced L{index + 1}",
            )
        )
    for index in range(template.drude):
        plasma_frequency, collision_frequency = np.exp(parameters[cursor : cursor + 2])
        cursor += 2
        terms.append(
            drude_term(
                plasma_frequency,
                collision_frequency,
                source=f"reduced R{index + 1}",
            )
        )
    return make_material(name, epsilon_inf, terms, fixed_conductivity)


def fit_reduced_model(
    target: DispersiveModel,
    frequencies: Sequence[float] | np.ndarray,
    template: ReductionTemplate,
    *,
    fixed_conductivity: float = 0.0,
    restarts: int = 8,
    seed: int = 481516,
    max_evaluations: int = 20_000,
) -> ReductionResult:
    """Fit a stable passive physical model over a chosen frequency band.

    Positive exponential parameterisations enforce positive Debye strengths,
    relaxation times, oscillator strengths, resonance frequencies, plasma
    frequencies, and damping.  Lorentz terms are constrained to remain
    underdamped.  This is a research fitter, not yet a replacement for a
    dedicated vector-fitting and passivity-enforcement package.
    """

    frequencies = np.asarray(frequencies, dtype=float)
    if template.terms < 1:
        raise ValueError("At least one reduced dispersive term is required")
    if frequencies.ndim != 1 or len(frequencies) < 8 or np.any(np.diff(frequencies) <= 0):
        raise ValueError("Frequencies must be a strictly increasing one-dimensional grid")

    target_values = target.relative_permittivity(frequencies)
    dynamic_scale = np.maximum(np.abs(target_values - target.epsilon_inf), 1.0)
    base = _initial_parameters(target, frequencies, template)
    random = np.random.default_rng(seed)
    best = None

    def residual(parameters: np.ndarray) -> np.ndarray:
        try:
            model = _model_from_parameters(
                parameters,
                template,
                target.epsilon_inf,
                fixed_conductivity,
                "candidate",
            )
            difference = (model.relative_permittivity(frequencies) - target_values) / dynamic_scale
            values = np.concatenate((difference.real, difference.imag))
            if np.all(np.isfinite(values)):
                return values
        except (FloatingPointError, OverflowError, ValueError):
            pass
        return np.full(2 * len(frequencies), 1e12)

    for restart in range(max(1, restarts)):
        initial = base if restart == 0 else base + random.normal(0.0, 0.8, base.shape)
        fitted = least_squares(
            residual,
            initial,
            method="trf",
            bounds=(-40.0, 40.0),
            max_nfev=max_evaluations,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        if best is None or fitted.cost < best.cost:
            best = fitted

    assert best is not None
    model = _model_from_parameters(
        best.x,
        template,
        target.epsilon_inf,
        fixed_conductivity,
        f"reduced {template.label()}",
    )
    return ReductionResult(
        model=model,
        template=template,
        metrics=permittivity_error(target, model, frequencies),
        cost=float(best.cost),
        evaluations=int(best.nfev),
        success=bool(best.success),
        message=str(best.message),
        method="joint_least_squares",
    )


def _initial_shape_parameters(
    target: DispersiveModel,
    frequencies: np.ndarray,
    template: ReductionTemplate,
) -> np.ndarray:
    """Return the nonlinear subset of the existing physical initial guess."""

    full = _initial_parameters(target, frequencies, template)
    shape: list[float] = []
    cursor = 0
    for _ in range(template.debye):
        shape.append(float(full[cursor + 1]))  # log(tau)
        cursor += 2
    for _ in range(template.lorentz):
        shape.extend((float(full[cursor + 1]), float(full[cursor + 2])))
        cursor += 3
    for _ in range(template.drude):
        shape.append(float(full[cursor + 1]))  # log(collision frequency)
        cursor += 2
    return np.asarray(shape, dtype=float)


def _shape_bounds(
    frequencies: np.ndarray,
    template: ReductionTemplate,
    *,
    margin_decades: float,
) -> list[tuple[float, float]]:
    """Construct broad, finite bounds for physical pole locations."""

    low, high = float(frequencies[0]), float(frequencies[-1])
    margin = 10.0**margin_decades
    bounds: list[tuple[float, float]] = []
    for _ in range(template.debye):
        bounds.append(
            (
                float(np.log(1.0 / (2 * np.pi * high * margin))),
                float(np.log(margin / (2 * np.pi * low))),
            )
        )
    for _ in range(template.lorentz):
        bounds.extend(
            (
                (float(np.log(low / margin)), float(np.log(high * margin))),
                (-12.0, 12.0),
            )
        )
    for _ in range(template.drude):
        bounds.append(
            (
                float(np.log(2 * np.pi * low / margin)),
                float(np.log(2 * np.pi * high * margin)),
            )
        )
    return bounds


def _projected_model(
    shape_parameters: np.ndarray,
    target_values: np.ndarray,
    frequencies: np.ndarray,
    dynamic_scale: np.ndarray,
    template: ReductionTemplate,
    epsilon_inf: float,
    fixed_conductivity: float,
    reference_frequency: float,
    name: str,
) -> tuple[DispersiveModel, np.ndarray, np.ndarray]:
    """Solve non-negative physical strengths for fixed pole locations.

    This is a constrained variable-projection step. Debye and Lorentz
    oscillator strengths are linear. For a Drude term, the pole residue and
    its exactly coupled conductivity contribution are both linear in
    ``omega_p**2``; using a reference plasma frequency keeps that column well
    scaled.
    """

    unit_terms: list[DispersiveModel] = []
    decoded: list[tuple[str, tuple[float, ...]]] = []
    cursor = 0
    for _ in range(template.debye):
        tau = float(np.exp(shape_parameters[cursor]))
        cursor += 1
        unit_terms.append(debye_term(1.0, tau))
        decoded.append(("debye", (tau,)))
    for _ in range(template.lorentz):
        resonance_frequency = float(np.exp(shape_parameters[cursor]))
        damping_ratio = 1e-5 + 0.97999 * _sigmoid(float(shape_parameters[cursor + 1]))
        cursor += 2
        damping = damping_ratio * 2 * np.pi * resonance_frequency
        unit_terms.append(lorentz_term(1.0, resonance_frequency, damping))
        decoded.append(("lorentz", (resonance_frequency, damping)))
    for _ in range(template.drude):
        collision_frequency = float(np.exp(shape_parameters[cursor]))
        cursor += 1
        unit_terms.append(drude_term(reference_frequency, collision_frequency))
        decoded.append(("drude", (collision_frequency,)))

    angular_frequency = 2 * np.pi * frequencies
    base = np.full(frequencies.shape, complex(epsilon_inf), dtype=complex)
    if fixed_conductivity:
        base += fixed_conductivity / (1j * angular_frequency * epsilon_0)
    target_dynamic = target_values - base
    columns = np.column_stack([term.relative_permittivity(frequencies) for term in unit_terms])
    weighted_columns = columns / dynamic_scale[:, None]
    weighted_target = target_dynamic / dynamic_scale
    matrix = np.vstack((weighted_columns.real, weighted_columns.imag))
    right_hand_side = np.concatenate((weighted_target.real, weighted_target.imag))
    strengths, _ = nnls(matrix, right_hand_side)

    terms: list[DispersiveModel] = []
    family_index = {"debye": 0, "lorentz": 0, "drude": 0}
    for (kind, values), strength in zip(decoded, strengths):
        family_index[kind] += 1
        if kind == "debye":
            terms.append(
                debye_term(
                    float(strength),
                    values[0],
                    source=f"reduced D{family_index[kind]}",
                )
            )
        elif kind == "lorentz":
            terms.append(
                lorentz_term(
                    float(strength),
                    values[0],
                    values[1],
                    source=f"reduced L{family_index[kind]}",
                )
            )
        else:
            plasma_frequency = reference_frequency * np.sqrt(max(float(strength), 0.0))
            # A zero-strength term is physically absent. Retain a numerically
            # negligible positive frequency so the requested template remains
            # serialisable and its effective lower order is visible in metrics.
            plasma_frequency = max(plasma_frequency, np.finfo(float).tiny)
            terms.append(
                drude_term(
                    plasma_frequency,
                    values[0],
                    source=f"reduced R{family_index[kind]}",
                )
            )

    model = make_material(name, epsilon_inf, terms, fixed_conductivity)
    difference = (model.relative_permittivity(frequencies) - target_values) / dynamic_scale
    residual = np.concatenate((difference.real, difference.imag))
    return model, residual, strengths


def fit_projected_model(
    target: DispersiveModel,
    frequencies: Sequence[float] | np.ndarray,
    template: ReductionTemplate,
    *,
    fixed_conductivity: float = 0.0,
    seed: int = 481516,
    maximum_iterations: int = 240,
    population_size: int = 12,
    margin_decades: float = 2.0,
    polish: bool = True,
) -> ReductionResult:
    """Fit pole locations globally and project physical strengths linearly.

    The construction follows the hybrid pole-location/least-squares strategy
    used for Debye expansions by Kelley, Destan, and Luebbers (2007), but uses
    weighted non-negative least squares on the joint complex response. It also
    preserves the exact high-frequency permittivity and requested independent
    conductivity and extends the projection to Lorentz and Drude terms.
    """

    frequencies = np.asarray(frequencies, dtype=float)
    if template.terms < 1:
        raise ValueError("At least one reduced dispersive term is required")
    if frequencies.ndim != 1 or len(frequencies) < 8 or np.any(np.diff(frequencies) <= 0):
        raise ValueError("Frequencies must be a strictly increasing one-dimensional grid")
    if maximum_iterations < 1 or population_size < 1:
        raise ValueError("Global-search iteration and population sizes must be positive")
    if margin_decades < 0:
        raise ValueError("The pole-search margin cannot be negative")

    target_values = target.relative_permittivity(frequencies)
    dynamic_scale = np.maximum(np.abs(target_values - target.epsilon_inf), 1.0)
    reference_frequency = float(np.sqrt(frequencies[0] * frequencies[-1]))
    initial = _initial_shape_parameters(target, frequencies, template)
    bounds = _shape_bounds(frequencies, template, margin_decades=margin_decades)

    def objective(parameters: np.ndarray) -> float:
        try:
            _, residual, _ = _projected_model(
                parameters,
                target_values,
                frequencies,
                dynamic_scale,
                template,
                target.epsilon_inf,
                fixed_conductivity,
                reference_frequency,
                "candidate",
            )
            if np.all(np.isfinite(residual)):
                return float(np.dot(residual, residual))
        except (FloatingPointError, OverflowError, RuntimeError, ValueError):
            pass
        return 1e24

    initial_cost = objective(initial)
    if initial_cost <= 1e-24:
        model, _, _ = _projected_model(
            initial,
            target_values,
            frequencies,
            dynamic_scale,
            template,
            target.epsilon_inf,
            fixed_conductivity,
            reference_frequency,
            f"projected {template.label()}",
        )
        return ReductionResult(
            model=model,
            template=template,
            metrics=permittivity_error(target, model, frequencies),
            cost=0.5 * initial_cost,
            evaluations=1,
            success=True,
            message="The grouped physical initial model is already exact.",
            method="global_variable_projection",
        )

    # Include the deterministic grouped-pole estimate in the initial global
    # population and use seeded uniform samples for the remaining members.
    population_members = max(population_size * len(bounds), 5)
    random = np.random.default_rng(seed)
    lower = np.asarray([bound[0] for bound in bounds])
    upper = np.asarray([bound[1] for bound in bounds])
    population = random.uniform(lower, upper, size=(population_members, len(bounds)))
    population[0] = np.clip(initial, lower, upper)
    fitted = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maximum_iterations,
        popsize=population_size,
        polish=polish,
        init=population,
        updating="immediate",
        workers=1,
        tol=1e-10,
        atol=1e-12,
    )
    model, _, _ = _projected_model(
        fitted.x,
        target_values,
        frequencies,
        dynamic_scale,
        template,
        target.epsilon_inf,
        fixed_conductivity,
        reference_frequency,
        f"projected {template.label()}",
    )
    return ReductionResult(
        model=model,
        template=template,
        metrics=permittivity_error(target, model, frequencies),
        cost=0.5 * float(fitted.fun),
        evaluations=int(fitted.nfev),
        success=bool(fitted.success),
        message=str(fitted.message),
        method="global_variable_projection",
    )


def search_reduced_models(
    target: DispersiveModel,
    frequencies: Sequence[float] | np.ndarray,
    *,
    maximum_terms: int,
    allowed_families: Sequence[str] = ("debye", "lorentz", "drude"),
    maximum_relative_error: float = 0.01,
    fixed_conductivity: float = 0.0,
) -> list[ReductionResult]:
    """Search physical templates and return results ordered by complexity/error."""

    allowed = set(allowed_families)
    templates = []
    for debye, lorentz, drude in product(
        range(maximum_terms + 1), range(maximum_terms + 1), range(maximum_terms + 1)
    ):
        template = ReductionTemplate(debye, lorentz, drude)
        if not 1 <= template.terms <= maximum_terms:
            continue
        if (
            (debye and "debye" not in allowed)
            or (lorentz and "lorentz" not in allowed)
            or (drude and "drude" not in allowed)
        ):
            continue
        templates.append(template)

    results = [
        fit_reduced_model(
            target,
            frequencies,
            template,
            fixed_conductivity=fixed_conductivity,
        )
        for template in templates
    ]
    results.sort(key=lambda result: (result.template.terms, result.metrics["maximum_relative"]))
    accepted = [
        result for result in results if result.metrics["maximum_relative"] <= maximum_relative_error
    ]
    return accepted or sorted(results, key=lambda result: result.metrics["maximum_relative"])
