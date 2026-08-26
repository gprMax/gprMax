# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.

"""Passive surface-impedance fits for common bulk metals.

The presets use the local good-conductor model

``Z(s) = sqrt(mu_0 * s / sigma)``

and approximate it by a Foster sum with non-negative coefficients.  The
result is positive real for the complete frequency axis, not only inside the
advertised fit band, and maps directly to the real state-space realization
used by :mod:`gprMax.impedance_surfaces`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.constants import epsilon_0
from scipy.optimize import lsq_linear, minimize

MATULA_1979_DOI = "https://doi.org/10.1063/1.555614"
DESAI_ALUMINIUM_DOI = "https://doi.org/10.1063/1.555725"
DESAI_REFRACTORY_METALS_DOI = "https://doi.org/10.1063/1.555723"
DEFAULT_METAL_FIT_ORDER = "auto"
DEFAULT_METAL_FIT_TOLERANCE = 2.0e-3
MIN_METAL_FIT_ORDER = 1
MAX_METAL_FIT_ORDER = 64
MAX_METAL_FIT_FREQUENCY_HZ = 300.0e9
MIN_GOOD_CONDUCTOR_RATIO = 100.0
AUTO_METAL_FIT_ORDERS = tuple(range(MIN_METAL_FIT_ORDER, MAX_METAL_FIT_ORDER + 1))

# The good-conductor target is scale invariant after normalising frequency by
# fmin and impedance by |Z(fmin)|. Trying a small deterministic family of
# Foster-grid extensions gives each requested *runtime* order a useful pole
# placement without turning the order search itself into a dictionary-size
# search. The selected fit is cached in this normalised form below.
_NORMALISED_GRID_EXTENSIONS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
_MAX_REFINEMENT_STARTS = 2
_HIGH_ORDER_REFINEMENT_CUTOFF = 2.5e-2
_MAX_REFINED_RUNTIME_ORDER = 16


@dataclass(frozen=True)
class MetalSurfacePreset:
    """Reference-temperature bulk-metal data used by the SIBC fit."""

    key: str
    name: str
    resistivity_ohm_m: float
    reference_temperature_k: float
    source: str = MATULA_1979_DOI

    @property
    def conductivity_s_per_m(self) -> float:
        return 1.0 / self.resistivity_ohm_m


@dataclass(frozen=True)
class FosterSurfaceImpedanceFit:
    """One fitted real state-space realization and its diagnostics."""

    preset: MetalSurfacePreset | None
    conductivity_s_per_m: float
    fmin_hz: float
    fmax_hz: float
    requested_order: str | int
    tolerance: float
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: float
    max_relative_error: float
    rms_relative_error: float
    attempts: tuple["FosterFitOrderDiagnostic", ...]

    @property
    def order(self) -> int:
        return int(self.A.shape[0])

    @property
    def selected_pole_count(self) -> int:
        """Actual number of independent Foster states used at runtime."""

        return self.order

    @property
    def meets_tolerance(self) -> bool:
        return self.max_relative_error <= self.tolerance


@dataclass(frozen=True)
class FosterFitOrderDiagnostic:
    """Accuracy for one requested and realised runtime pole count."""

    pole_count: int
    max_relative_error: float
    rms_relative_error: float

    @property
    def selected_pole_count(self) -> int:
        return self.pole_count


@dataclass(frozen=True)
class _CandidateFosterFit:
    """Internal runtime-order realization before automatic selection."""

    diagnostic: FosterFitOrderDiagnostic
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: float


@dataclass(frozen=True)
class _NormalisedFosterFit:
    """Scale-free Foster fit for one bandwidth ratio and runtime order."""

    relaxation: np.ndarray
    branch_resistance: np.ndarray
    direct_resistance: float
    max_relative_error: float
    rms_relative_error: float
    all_branches_active: bool


# Recommended uncorrected bulk-pure-metal resistivities at 293 K. Copper,
# gold, palladium, and silver are from Matula. Aluminium is from Desai et al.;
# molybdenum, tungsten, and zinc are from Desai et al.'s refractory-metal
# reference-data compilation. Values are intentionally stored as resistivity
# (the measured reference quantity), then inverted only when constructing the
# SIBC.
METAL_SURFACE_PRESETS = {
    "aluminium": MetalSurfacePreset("aluminium", "Aluminium", 2.650e-8, 293.0, DESAI_ALUMINIUM_DOI),
    "copper": MetalSurfacePreset("copper", "Copper", 1.676e-8, 293.0),
    "gold": MetalSurfacePreset("gold", "Gold", 2.192e-8, 293.0),
    "molybdenum": MetalSurfacePreset(
        "molybdenum", "Molybdenum", 5.34e-8, 293.0, DESAI_REFRACTORY_METALS_DOI
    ),
    "palladium": MetalSurfacePreset("palladium", "Palladium", 10.54e-8, 293.0),
    "silver": MetalSurfacePreset("silver", "Silver", 1.586e-8, 293.0),
    "tungsten": MetalSurfacePreset(
        "tungsten", "Tungsten", 5.28e-8, 293.0, DESAI_REFRACTORY_METALS_DOI
    ),
    "zinc": MetalSurfacePreset("zinc", "Zinc", 5.964e-8, 293.0, DESAI_REFRACTORY_METALS_DOI),
}

_ALIASES = {
    "ag": "silver",
    "al": "aluminium",
    "aluminum": "aluminium",
    "au": "gold",
    "cu": "copper",
    "mo": "molybdenum",
    "pd": "palladium",
    "w": "tungsten",
    "zn": "zinc",
}


def get_metal_surface_preset(name: str) -> MetalSurfacePreset:
    """Return a named common-metal preset, accepting element-symbol aliases."""

    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return METAL_SURFACE_PRESETS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(METAL_SURFACE_PRESETS))
        raise ValueError(
            f"unknown surface-impedance metal preset {name!r}; choose {choices}"
        ) from exc


def good_conductor_surface_impedance(
    frequencies_hz,
    conductivity_s_per_m: float,
    relative_permeability: float = 1.0,
):
    """Return the passive ``e^(+jwt)`` good-conductor surface impedance."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    conductivity = float(conductivity_s_per_m)
    mur = float(relative_permeability)
    if np.any(frequencies < 0) or not np.all(np.isfinite(frequencies)):
        raise ValueError("surface-impedance frequencies must be finite and non-negative")
    if not np.isfinite(conductivity) or conductivity <= 0:
        raise ValueError("metal conductivity must be finite and positive")
    if not np.isfinite(mur) or mur <= 0:
        raise ValueError("metal relative permeability must be finite and positive")
    mu0 = 4e-7 * np.pi
    omega = 2 * np.pi * frequencies
    return (1 + 1j) * np.sqrt(omega * mu0 * mur / (2 * conductivity))


def _validate_fit_request(
    fmin_hz: float,
    fmax_hz: float,
    order: str | int,
    tolerance: float,
) -> tuple[float, float, str | int, float]:
    """Validate and normalize a good-conductor fit request."""

    fmin = float(fmin_hz)
    fmax = float(fmax_hz)
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin <= 0 or fmax <= fmin:
        raise ValueError("metal surface fit band must satisfy 0 < fmin < fmax < infinity")

    if isinstance(order, str):
        requested_order = order.strip().lower()
        if requested_order != "auto":
            raise ValueError("metal surface fit order must be 'auto' or an integer")
    elif isinstance(order, (int, np.integer)) and not isinstance(order, (bool, np.bool_)):
        requested_order = int(order)
        if not MIN_METAL_FIT_ORDER <= requested_order <= MAX_METAL_FIT_ORDER:
            raise ValueError(
                f"metal surface fit order must be between {MIN_METAL_FIT_ORDER} "
                f"and {MAX_METAL_FIT_ORDER}"
            )
    else:
        raise ValueError("metal surface fit order must be 'auto' or an integer")

    fit_tolerance = float(tolerance)
    if not np.isfinite(fit_tolerance) or fit_tolerance <= 0:
        raise ValueError("metal surface fit tolerance must be finite and positive")
    return fmin, fmax, requested_order, fit_tolerance


def _validate_good_conductor_scope(
    conductivity_s_per_m: float,
    fmax_hz: float,
) -> None:
    """Reject bands outside the first microwave good-conductor model."""

    if fmax_hz > MAX_METAL_FIT_FREQUENCY_HZ:
        raise ValueError(
            "metal surface fits currently require a microwave band with "
            f"fmax <= {MAX_METAL_FIT_FREQUENCY_HZ:g} Hz; optical and terahertz "
            "metal dispersion is not supported"
        )
    ratio = conductivity_s_per_m / (2 * np.pi * fmax_hz * epsilon_0)
    if ratio < MIN_GOOD_CONDUCTOR_RATIO:
        raise ValueError(
            "metal surface fit violates the good-conductor requirement at fmax: "
            f"sigma/(omega*epsilon_0)={ratio:g}, but at least "
            f"{MIN_GOOD_CONDUCTOR_RATIO:g} is required"
        )


def _solve_nonnegative_foster_coefficients(
    frequencies: np.ndarray,
    target: np.ndarray,
    relaxation: np.ndarray,
) -> np.ndarray:
    """Solve the passive linear part of one scale-free Foster fit."""

    s = 1j * frequencies
    basis = np.column_stack([np.ones(frequencies.size), *(s / (s + pole) for pole in relaxation)])
    relative_weight = 1 / np.abs(target)
    matrix = np.vstack(
        (
            basis.real * relative_weight[:, np.newaxis],
            basis.imag * relative_weight[:, np.newaxis],
        )
    )
    rhs = np.concatenate((target.real * relative_weight, target.imag * relative_weight))
    column_norms = np.linalg.norm(matrix, axis=0)
    if np.any(column_norms == 0) or not np.all(np.isfinite(column_norms)):
        raise RuntimeError("metal surface fit produced a singular Foster basis")
    solution = lsq_linear(
        matrix / column_norms,
        rhs,
        bounds=(0, np.inf),
        method="bvls",
        tol=1e-12,
        max_iter=max(1000, 20 * relaxation.size),
    )
    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise RuntimeError(
            f"metal surface fit failed at runtime order {relaxation.size}: " f"{solution.message}"
        )
    return np.maximum(solution.x / column_norms, 0.0)


def _normalised_fit_error(
    frequencies: np.ndarray,
    target: np.ndarray,
    relaxation: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[float, float]:
    """Return maximum and RMS complex relative errors."""

    s = 1j * frequencies
    fitted = np.full(frequencies.shape, coefficients[0], dtype=np.complex128)
    for resistance, pole in zip(coefficients[1:], relaxation):
        fitted += resistance * s / (s + pole)
    relative_error = np.abs(fitted / target - 1)
    return (
        float(relative_error.max()),
        float(np.sqrt(np.mean(relative_error**2))),
    )


@lru_cache(maxsize=512)
def _fit_normalised_good_conductor_order(
    bandwidth_ratio: float,
    order: int,
) -> _NormalisedFosterFit:
    """Locally optimise ``order`` Foster branches in dimensionless units."""

    ratio = float(bandwidth_ratio)
    runtime_order = int(order)
    sample_count = max(513, 64 * runtime_order + 1)
    frequencies = np.geomspace(1.0, ratio, sample_count)
    target = (1 + 1j) * np.sqrt(frequencies)
    validation_count = max(4097, 256 * runtime_order + 1)
    validation_frequencies = np.geomspace(1.0, ratio, validation_count)
    validation_target = (1 + 1j) * np.sqrt(validation_frequencies)

    candidates = []

    def solve_candidate(relaxation):
        relaxation = np.sort(np.asarray(relaxation, dtype=np.float64))
        coefficients = _solve_nonnegative_foster_coefficients(
            frequencies,
            target,
            relaxation,
        )
        maximum, rms = _normalised_fit_error(
            validation_frequencies,
            validation_target,
            relaxation,
            coefficients,
        )
        return maximum, rms, relaxation, coefficients

    def evaluate(relaxation):
        candidates.append(solve_candidate(relaxation))

    for extension in _NORMALISED_GRID_EXTENSIONS:
        evaluate(
            np.geomspace(
                1.0 / extension,
                ratio * extension,
                runtime_order,
            )
        )

    # An overcomplete passive grid is also a cheap deterministic pole-location
    # generator. If BVLS activates exactly the requested number of branches,
    # refit those locations as an actual ``runtime_order`` realization. This
    # recovers useful non-uniform grids (especially for one-octave bands)
    # without ever carrying the unused dictionary entries into FDTD state.
    for dictionary_order in range(
        runtime_order + 1,
        min(MAX_METAL_FIT_ORDER, runtime_order + 6) + 1,
    ):
        for extension in (10.0, 100.0):
            dictionary_relaxation = np.geomspace(
                1.0 / extension,
                ratio * extension,
                dictionary_order,
            )
            dictionary_coefficients = _solve_nonnegative_foster_coefficients(
                frequencies,
                target,
                dictionary_relaxation,
            )
            threshold = 128 * np.finfo(np.float64).eps * float(dictionary_coefficients.max())
            active = dictionary_coefficients[1:] > threshold
            if np.count_nonzero(active) == runtime_order:
                evaluate(dictionary_relaxation[active])

    # Variable projection: pole locations are the only nonlinear variables;
    # every objective evaluation solves the residues and direct term by
    # non-negative BVLS. Bounded Powell search is deterministic, handles the
    # mildly nonsmooth maximum-error objective, and avoids a costly global
    # optimiser. Fixed/overcomplete grids above provide distinct robust starts.
    starts = sorted(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    log_lower = np.log(1.0e-4)
    log_upper = np.log(ratio * 1.0e4)
    used_starts = []
    for start in starts:
        # This policy depends only on bandwidth ratio and runtime order, never
        # on the user's requested tolerance. Low-order searches are cheap and
        # benefit most from pole movement. At higher order, refine only close
        # fixed-grid candidates; very high explicit orders are already far
        # below useful fitting errors and retain their deterministic grid fit.
        if runtime_order > _MAX_REFINED_RUNTIME_ORDER or (
            runtime_order > 3 and start[0] > _HIGH_ORDER_REFINEMENT_CUTOFF
        ):
            break
        start_logs = np.log(start[2])
        if any(np.allclose(start_logs, used, rtol=0, atol=1e-10) for used in used_starts):
            continue
        used_starts.append(start_logs)
        best_refined = [start]

        def objective(log_relaxation):
            candidate = solve_candidate(np.exp(np.sort(log_relaxation)))
            if (candidate[0], candidate[1]) < (
                best_refined[0][0],
                best_refined[0][1],
            ):
                best_refined[0] = candidate
            return candidate[0]

        minimize(
            objective,
            start_logs,
            method="Powell",
            bounds=[(log_lower, log_upper)] * runtime_order,
            options={
                "maxiter": max(40, 8 * runtime_order),
                "maxfev": (
                    max(300, 100 * runtime_order)
                    if runtime_order <= 3
                    else max(160, 24 * runtime_order)
                ),
                "xtol": 1e-6,
                "ftol": 1e-10,
            },
        )
        candidates.append(best_refined[0])
        maximum_starts = (
            _MAX_REFINEMENT_STARTS
            if runtime_order <= 3 or starts[0][0] > DEFAULT_METAL_FIT_TOLERANCE
            else 1
        )
        if len(used_starts) >= maximum_starts:
            break

    def all_branches_active(candidate):
        coefficients = candidate[3]
        threshold = 128 * np.finfo(np.float64).eps * float(coefficients.max())
        return bool(np.all(coefficients[1:] > threshold))

    # Prefer a genuine realization of the requested runtime count. Explicit
    # high orders may be deliberately over-specified, so retain the best
    # degenerate candidate only when no full-order candidate exists; auto
    # rejects that fallback below rather than counting zero-residue states.
    full_order_candidates = [
        candidate for candidate in candidates if all_branches_active(candidate)
    ]
    maximum, rms, relaxation, coefficients = min(
        full_order_candidates or candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    # Pole locations saw the validation grid through the outer objective, so
    # certify the final candidate once more on a separate, substantially
    # denser grid before it participates in automatic order selection.
    certification_count = max(16385, 1024 * runtime_order + 1)
    certification_frequencies = np.geomspace(1.0, ratio, certification_count)
    certification_target = (1 + 1j) * np.sqrt(certification_frequencies)
    maximum, rms = _normalised_fit_error(
        certification_frequencies,
        certification_target,
        relaxation,
        coefficients,
    )
    immutable_arrays = []
    for value in (relaxation, coefficients[1:]):
        contiguous = np.ascontiguousarray(value, dtype=np.float64)
        immutable_arrays.append(np.frombuffer(contiguous.tobytes(), dtype=np.float64))
    return _NormalisedFosterFit(
        relaxation=immutable_arrays[0],
        branch_resistance=immutable_arrays[1],
        direct_resistance=float(coefficients[0]),
        max_relative_error=maximum,
        rms_relative_error=rms,
        all_branches_active=all_branches_active((maximum, rms, relaxation, coefficients)),
    )


def _fit_good_conductor_candidate(
    conductivity_s_per_m: float,
    fmin_hz: float,
    fmax_hz: float,
    pole_count: int,
    *,
    require_all_poles: bool = False,
) -> _CandidateFosterFit:
    """Fit one requested runtime pole count and validate it independently."""

    normalised = _fit_normalised_good_conductor_order(
        fmax_hz / fmin_hz,
        pole_count,
    )
    omega_min = 2 * np.pi * fmin_hz
    impedance_scale = np.sqrt(omega_min * (4e-7 * np.pi) / (2 * conductivity_s_per_m))
    branch_relaxation = omega_min * normalised.relaxation
    branch_resistance = impedance_scale * normalised.branch_resistance
    direct = float(
        impedance_scale * (normalised.direct_resistance + normalised.branch_resistance.sum())
    )
    A = np.diag(-branch_relaxation)
    coupling = np.sqrt(branch_resistance * branch_relaxation)
    B = coupling
    C = -coupling

    immutable_arrays = []
    for value in (A, B, C):
        contiguous = np.ascontiguousarray(value, dtype=np.float64)
        immutable_arrays.append(
            np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)
        )
    A, B, C = immutable_arrays
    maximum_error = normalised.max_relative_error
    rms_error = normalised.rms_relative_error
    if require_all_poles and not normalised.all_branches_active:
        maximum_error = np.inf
        rms_error = np.inf
    diagnostic = FosterFitOrderDiagnostic(
        pole_count=int(A.shape[0]),
        max_relative_error=maximum_error,
        rms_relative_error=rms_error,
    )
    return _CandidateFosterFit(
        diagnostic=diagnostic,
        A=A,
        B=B,
        C=C,
        D=direct,
    )


def _fit_good_conductor(
    conductivity_s_per_m: float,
    fmin_hz: float,
    fmax_hz: float,
    order: str | int,
    tolerance: float,
    *,
    preset: MetalSurfacePreset | None,
) -> FosterSurfaceImpedanceFit:
    """Select the first certified pole count from a sequential local search."""

    conductivity = float(conductivity_s_per_m)
    if not np.isfinite(conductivity) or conductivity <= 0:
        raise ValueError("metal conductivity must be finite and positive")
    fmin, fmax, requested_order, fit_tolerance = _validate_fit_request(
        fmin_hz, fmax_hz, order, tolerance
    )
    _validate_good_conductor_scope(conductivity, fmax)
    attempted_pole_counts = (
        AUTO_METAL_FIT_ORDERS if requested_order == "auto" else (requested_order,)
    )
    attempts = []
    selected = None
    for pole_count in attempted_pole_counts:
        candidate = _fit_good_conductor_candidate(
            conductivity,
            fmin,
            fmax,
            pole_count,
            require_all_poles=requested_order == "auto",
        )
        attempts.append(candidate.diagnostic)
        selected = candidate
        if requested_order != "auto" or candidate.diagnostic.max_relative_error <= fit_tolerance:
            break

    if selected is None:  # pragma: no cover - validated order range guarantees work
        raise RuntimeError("metal surface fit did not evaluate any pole counts")
    if requested_order == "auto" and selected.diagnostic.max_relative_error > fit_tolerance:
        best = min(attempts, key=lambda attempt: attempt.max_relative_error)
        raise ValueError(
            f"automatic metal surface fit did not reach tolerance {fit_tolerance:g} "
            f"through {selected.diagnostic.pole_count} poles; the best attempted "
            f"count was {best.pole_count}, with maximum relative error "
            f"{best.max_relative_error:g}"
        )

    return FosterSurfaceImpedanceFit(
        preset=preset,
        conductivity_s_per_m=conductivity,
        fmin_hz=fmin,
        fmax_hz=fmax,
        requested_order=requested_order,
        tolerance=fit_tolerance,
        A=selected.A,
        B=selected.B,
        C=selected.C,
        D=selected.D,
        max_relative_error=selected.diagnostic.max_relative_error,
        rms_relative_error=selected.diagnostic.rms_relative_error,
        attempts=tuple(attempts),
    )


@lru_cache(maxsize=128)
def fit_metal_surface_impedance(
    name: str,
    fmin_hz: float,
    fmax_hz: float,
    order: str | int = DEFAULT_METAL_FIT_ORDER,
    tolerance: float = DEFAULT_METAL_FIT_TOLERANCE,
) -> FosterSurfaceImpedanceFit:
    """Fit a named common metal over a mandatory frequency band."""

    preset = get_metal_surface_preset(name)
    return _fit_good_conductor(
        preset.conductivity_s_per_m,
        fmin_hz,
        fmax_hz,
        order,
        tolerance,
        preset=preset,
    )


@lru_cache(maxsize=128)
def fit_conductivity_surface_impedance(
    conductivity_s_per_m: float,
    fmin_hz: float,
    fmax_hz: float,
    order: str | int = DEFAULT_METAL_FIT_ORDER,
    tolerance: float = DEFAULT_METAL_FIT_TOLERANCE,
) -> FosterSurfaceImpedanceFit:
    """Fit a passive good-conductor target over a mandatory frequency band."""

    return _fit_good_conductor(
        conductivity_s_per_m,
        fmin_hz,
        fmax_hz,
        order,
        tolerance,
        preset=None,
    )
