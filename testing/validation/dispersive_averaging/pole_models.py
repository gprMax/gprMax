"""Inclusive pole-residue models for dispersive-interface experiments.

This module is deliberately independent of the production gprMax material
classes.  It expresses Debye, Lorentz, and Drude terms using the inclusive
susceptibility kernel of Giannakis and Giannopoulos (2014), so exact
arithmetic mixtures and reduced-order approximations can be studied before
any solver changes are proposed.

The engineering convention ``exp(+j omega t)`` is used throughout.  Passive
materials therefore have a negative imaginary relative permittivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.constants import epsilon_0


@dataclass(frozen=True)
class InclusivePole:
    """One inclusive susceptibility term ``Re(w exp(q t)) u(t)``.

    A real ``w`` and ``q`` represents a Debye-like exponential.  A complex
    term implicitly includes its complex conjugate and therefore represents
    one underdamped Lorentz pole pair.
    """

    w: complex
    q: complex
    kind: str = "general"
    source: str = ""

    def response(self, angular_frequency: np.ndarray) -> np.ndarray:
        """Return this pole's relative susceptibility."""

        s = 1j * angular_frequency
        if np.isclose(self.w.imag, 0.0) and np.isclose(self.q.imag, 0.0):
            return self.w.real / (s - self.q.real)
        return 0.5 * (self.w / (s - self.q) + np.conjugate(self.w) / (s - np.conjugate(self.q)))


@dataclass(frozen=True)
class DispersiveModel:
    """A linear isotropic material in inclusive pole-residue form."""

    epsilon_inf: float = 1.0
    conductivity: float = 0.0
    poles: tuple[InclusivePole, ...] = field(default_factory=tuple)
    name: str = "material"

    def relative_permittivity(self, frequencies: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate complex relative permittivity using engineering signs."""

        frequencies = np.asarray(frequencies, dtype=float)
        if np.any(frequencies <= 0):
            raise ValueError("Frequencies must be strictly positive")
        angular_frequency = 2 * np.pi * frequencies
        epsilon_r = np.full(frequencies.shape, complex(self.epsilon_inf), dtype=complex)
        if self.conductivity:
            epsilon_r += self.conductivity / (1j * angular_frequency * epsilon_0)
        for pole in self.poles:
            epsilon_r += pole.response(angular_frequency)
        return epsilon_r

    @property
    def inclusive_order(self) -> int:
        """Number of recursive inclusive terms used by an FDTD update."""

        return len(self.poles)

    @property
    def rational_order(self) -> int:
        """Number of first-order rational poles, including conjugates."""

        return sum(1 if np.isclose(pole.q.imag, 0.0) else 2 for pole in self.poles)

    def with_terms(self, *terms: "DispersiveModel", name: str | None = None) -> "DispersiveModel":
        """Add dispersive terms while retaining this model's base properties."""

        poles = list(self.poles)
        conductivity = self.conductivity
        for term in terms:
            poles.extend(term.poles)
            conductivity += term.conductivity
        return DispersiveModel(
            epsilon_inf=self.epsilon_inf,
            conductivity=conductivity,
            poles=tuple(poles),
            name=name or self.name,
        )


def debye_term(delta_epsilon: float, tau: float, *, source: str = "") -> DispersiveModel:
    """Construct one passive Debye term."""

    if delta_epsilon < 0 or tau <= 0:
        raise ValueError("A passive Debye term requires delta_epsilon >= 0 and tau > 0")
    pole = InclusivePole(delta_epsilon / tau, -1.0 / tau, "debye", source)
    return DispersiveModel(epsilon_inf=0.0, poles=(pole,), name=source or "Debye term")


def lorentz_term(
    delta_epsilon: float,
    resonance_frequency: float,
    damping: float,
    *,
    source: str = "",
) -> DispersiveModel:
    """Construct one passive underdamped Lorentz term.

    Args:
        delta_epsilon: Static strength of the oscillator.
        resonance_frequency: Undamped resonance frequency in Hz.
        damping: Damping coefficient in rad/s, matching gprMax's ``alpha``.
    """

    omega_0 = 2 * np.pi * resonance_frequency
    if delta_epsilon < 0 or resonance_frequency <= 0 or damping <= 0:
        raise ValueError("A passive Lorentz term requires positive physical parameters")
    if damping >= omega_0:
        raise ValueError("The inclusive one-pair representation requires an underdamped term")
    beta = np.sqrt(omega_0**2 - damping**2)
    pole = InclusivePole(
        -1j * delta_epsilon * omega_0**2 / beta,
        -damping + 1j * beta,
        "lorentz",
        source,
    )
    return DispersiveModel(epsilon_inf=0.0, poles=(pole,), name=source or "Lorentz term")


def drude_term(
    plasma_frequency: float,
    collision_frequency: float,
    *,
    source: str = "",
) -> DispersiveModel:
    """Construct one passive Drude term in gprMax's inclusive form.

    The zero-frequency pole is represented by an equivalent conductivity and
    the remaining decaying exponential by one real inclusive pole.
    Frequencies are supplied in Hz for the plasma frequency and rad/s for the
    collision frequency, matching the present gprMax material parameters.
    """

    omega_p = 2 * np.pi * plasma_frequency
    gamma = collision_frequency
    if plasma_frequency <= 0 or gamma <= 0:
        raise ValueError("A passive Drude term requires positive frequencies")
    pole = InclusivePole(-(omega_p**2) / gamma, -gamma, "drude", source)
    conductivity = epsilon_0 * omega_p**2 / gamma
    return DispersiveModel(
        epsilon_inf=0.0,
        conductivity=conductivity,
        poles=(pole,),
        name=source or "Drude term",
    )


def make_material(
    name: str,
    epsilon_inf: float,
    terms: Iterable[DispersiveModel] = (),
    conductivity: float = 0.0,
) -> DispersiveModel:
    """Create a material containing any combination of dispersion families."""

    poles: list[InclusivePole] = []
    total_conductivity = conductivity
    for term in terms:
        poles.extend(term.poles)
        total_conductivity += term.conductivity
    return DispersiveModel(epsilon_inf, total_conductivity, tuple(poles), name)


def arithmetic_mix(
    materials: Sequence[DispersiveModel],
    fractions: Sequence[float],
    *,
    name: str = "arithmetic interface mixture",
    merge_tolerance: float = 1e-12,
) -> DispersiveModel:
    """Form the exact arithmetic material response at a Yee edge.

    Each inclusive residue is multiplied by its cell fraction; pole locations
    are unchanged.  Equal pole locations are merged exactly (within the
    requested numerical tolerance).  No pole-order reduction is performed.
    """

    if len(materials) != len(fractions) or not materials:
        raise ValueError("Materials and fractions must be non-empty and have equal length")
    fractions_array = np.asarray(fractions, dtype=float)
    if np.any(fractions_array < 0) or not np.isclose(np.sum(fractions_array), 1.0):
        raise ValueError("Material fractions must be non-negative and sum to one")

    epsilon_inf = float(
        sum(f * material.epsilon_inf for f, material in zip(fractions_array, materials))
    )
    conductivity = float(
        sum(f * material.conductivity for f, material in zip(fractions_array, materials))
    )
    merged: list[InclusivePole] = []
    for fraction, material in zip(fractions_array, materials):
        for pole in material.poles:
            scaled = InclusivePole(
                fraction * pole.w, pole.q, pole.kind, pole.source or material.name
            )
            for index, existing in enumerate(merged):
                scale = max(1.0, abs(existing.q), abs(scaled.q))
                if abs(existing.q - scaled.q) <= merge_tolerance * scale:
                    merged[index] = InclusivePole(
                        existing.w + scaled.w,
                        existing.q,
                        existing.kind if existing.kind == scaled.kind else "general",
                        "+".join(filter(None, (existing.source, scaled.source))),
                    )
                    break
            else:
                merged.append(scaled)

    merged.sort(key=lambda pole: (float(pole.q.real), float(pole.q.imag), pole.kind))
    return DispersiveModel(epsilon_inf, conductivity, tuple(merged), name)


def permittivity_error(
    exact: DispersiveModel,
    approximation: DispersiveModel,
    frequencies: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return compact complex-permittivity error metrics."""

    exact_values = exact.relative_permittivity(frequencies)
    approximate_values = approximation.relative_permittivity(frequencies)
    difference = approximate_values - exact_values
    scale = np.maximum(np.abs(exact_values), np.finfo(float).eps)
    relative = np.abs(difference) / scale
    return {
        "rms_absolute": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "maximum_absolute": float(np.max(np.abs(difference))),
        "rms_relative": float(np.sqrt(np.mean(relative**2))),
        "maximum_relative": float(np.max(relative)),
    }
