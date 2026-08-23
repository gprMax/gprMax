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
from scipy.optimize import nnls


MATULA_1979_DOI = "https://doi.org/10.1063/1.555614"
DEFAULT_METAL_FMIN_HZ = 1.0e6
DEFAULT_METAL_FMAX_HZ = 1.0e11
DEFAULT_METAL_FIT_ORDER = 16


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

    preset: MetalSurfacePreset
    fmin_hz: float
    fmax_hz: float
    candidate_order: int
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: float
    max_relative_error: float
    rms_relative_error: float

    @property
    def order(self) -> int:
        return int(self.A.shape[0])


# Recommended bulk-pure-metal resistivities at 293 K from Matula's tables 2,
# 5, and 11. Values are intentionally stored as resistivity (the measured
# reference quantity), then inverted only when constructing the SIBC.
METAL_SURFACE_PRESETS = {
    "copper": MetalSurfacePreset("copper", "Copper", 1.676e-8, 293.0),
    "gold": MetalSurfacePreset("gold", "Gold", 2.192e-8, 293.0),
    "silver": MetalSurfacePreset("silver", "Silver", 1.586e-8, 293.0),
}

_ALIASES = {
    "ag": "silver",
    "au": "gold",
    "cu": "copper",
}


def get_metal_surface_preset(name: str) -> MetalSurfacePreset:
    """Return a named common-metal preset, accepting element-symbol aliases."""

    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return METAL_SURFACE_PRESETS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(METAL_SURFACE_PRESETS))
        raise ValueError(f"unknown surface-impedance metal preset {name!r}; choose {choices}") from exc


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


@lru_cache(maxsize=64)
def fit_metal_surface_impedance(
    name: str,
    fmin_hz: float = DEFAULT_METAL_FMIN_HZ,
    fmax_hz: float = DEFAULT_METAL_FMAX_HZ,
    order: int = DEFAULT_METAL_FIT_ORDER,
) -> FosterSurfaceImpedanceFit:
    """Fit a passive Foster model over a logarithmic frequency band.

    ``order`` is the number of candidate relaxation poles. Near-zero fitted
    branches are removed, so the returned minimal practical order can be
    slightly smaller.
    """

    preset = get_metal_surface_preset(name)
    fmin = float(fmin_hz)
    fmax = float(fmax_hz)
    candidate_order = int(order)
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin <= 0 or fmax <= fmin:
        raise ValueError("metal preset fit band must satisfy 0 < fmin < fmax < infinity")
    if candidate_order < 4 or candidate_order > 64:
        raise ValueError("metal preset fit order must be between 4 and 64")

    sample_count = max(513, 64 * candidate_order + 1)
    frequencies = np.geomspace(fmin, fmax, sample_count)
    omega = 2 * np.pi * frequencies
    target = good_conductor_surface_impedance(
        frequencies, preset.conductivity_s_per_m
    )

    # R0 + sum Rm*s/(s+am), with every coefficient constrained non-negative.
    # Extending the relaxation grid beyond the fit band controls endpoint
    # error while preserving a passive asymptote on both sides.
    relaxation = 2 * np.pi * np.geomspace(fmin / 100, fmax * 100, candidate_order)
    basis = np.column_stack(
        [np.ones(sample_count), *(1j * omega / (1j * omega + pole) for pole in relaxation)]
    )
    relative_weight = 1 / np.abs(target)
    matrix = np.vstack(
        (
            basis.real * relative_weight[:, np.newaxis],
            basis.imag * relative_weight[:, np.newaxis],
        )
    )
    rhs = np.concatenate((target.real * relative_weight, target.imag * relative_weight))
    coefficients, _ = nnls(matrix, rhs)

    threshold = 128 * np.finfo(np.float64).eps * max(1.0, float(coefficients.max()))
    active = coefficients[1:] > threshold
    branch_resistance = coefficients[1:][active]
    branch_relaxation = relaxation[active]
    direct = float(coefficients[0] + branch_resistance.sum())
    A = np.diag(-branch_relaxation)
    # Symmetric square-root scaling balances each first-order realization:
    # C_m B_m = -R_m a_m while |B_m|=|C_m|.
    coupling = np.sqrt(branch_resistance * branch_relaxation)
    B = coupling
    C = -coupling

    fitted = basis @ coefficients
    relative_error = np.abs(fitted / target - 1)
    # Cached fits are process-global. Back arrays with immutable ``bytes`` so
    # neither direct writes nor mutation through NumPy's ``.base`` chain can
    # silently corrupt every later use of a metal preset.
    immutable_arrays = []
    for value in (A, B, C):
        contiguous = np.ascontiguousarray(value, dtype=np.float64)
        immutable_arrays.append(
            np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)
        )
    A, B, C = immutable_arrays

    return FosterSurfaceImpedanceFit(
        preset=preset,
        fmin_hz=fmin,
        fmax_hz=fmax,
        candidate_order=candidate_order,
        A=A,
        B=B,
        C=C,
        D=direct,
        max_relative_error=float(relative_error.max()),
        rms_relative_error=float(np.sqrt(np.mean(relative_error**2))),
    )
