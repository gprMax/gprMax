# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Frequency-domain reduction of the surface-impedance ADE.

This module deliberately contains no geometry or eigensolver policy.  It is
the small numerical seam between a packed FDTD surface model and a future
impedance-aware mode operator: FDFD must use the transfer function of the
actual discrete recurrence, including the Yee midpoint factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class AlgorithmicSurfaceResponse:
    """One-frequency response of a trapezoidal surface-impedance ADE.

    ``impedance`` is ``Z_alg`` in the discrete boundary equation

    ``cos(theta / 2) * E = Z_alg * K``.

    Consequently ``admittance`` is the load seen by the harmonic Ampere row,
    ``K / E = cos(theta / 2) / Z_alg``.
    """

    theta: float
    physical_angular_frequency: float
    discrete_angular_frequency: float
    midpoint_cosine: float
    impedance: complex
    admittance: complex

    @property
    def angular_frequency(self) -> float:
        """Backward-compatible alias for the discrete derivative frequency."""
        return self.discrete_angular_frequency


@dataclass(frozen=True)
class BoundaryMagneticTerm:
    """One ordinary-H line contribution to an integral boundary Ampere row.

    Axes and indices use the mode solver's local ``(u, v, w)`` basis.  The
    signed ``line_weight`` is the physical integration length after clipping
    the electric dual face.  Longitudinal phase terms are not included: for a
    propagation-invariant wall their row-normalised coefficient is the usual
    implicit beta coupling already present in the P/Q formulation.
    """

    axis: int
    index: tuple[int, int]
    line_weight: float


@dataclass(frozen=True)
class BoundaryAmpereRow:
    """One impedance-aware electric row in a modal cross-section."""

    electric_axis: int
    electric_index: tuple[int, int]
    retained_dual_area: float
    relative_permittivity: complex
    magnetic_terms: tuple[BoundaryMagneticTerm, ...]


@dataclass(frozen=True)
class FDFDSurfaceBoundary:
    """Topology and clipped Ampere rows supplied to the 2-D mode solver.

    The six retained masks are independent by design.  An impedance volume
    retains tangential boundary E and interface-normal H, so expressing it as
    a PEC mask would remove valid magnetic degrees of freedom.
    """

    electric_retained: tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]
    magnetic_retained: tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]
    rows: tuple[BoundaryAmpereRow, ...]

    @classmethod
    def create(
        cls,
        *,
        electric_retained: Sequence[npt.ArrayLike],
        magnetic_retained: Sequence[npt.ArrayLike],
        rows: Sequence[BoundaryAmpereRow],
    ) -> "FDFDSurfaceBoundary":
        if len(electric_retained) != 3 or len(magnetic_retained) != 3:
            raise ValueError("surface-boundary retained masks must contain three components")
        return cls(
            tuple(electric_retained),
            tuple(magnetic_retained),
            tuple(rows),
        )


def _validated_phase(frequency_hz: float, dt: float) -> tuple[float, float, float, float]:
    frequency_hz = float(frequency_hz)
    dt = float(dt)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("surface-ADE FDFD frequency must be finite and positive")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("surface-ADE FDFD time step must be finite and positive")
    theta = 2 * np.pi * frequency_hz * dt
    if theta >= np.pi:
        raise ValueError("surface-ADE FDFD frequency must lie below temporal Nyquist")
    midpoint_cosine = float(np.cos(0.5 * theta))
    physical_angular_frequency = float(2 * np.pi * frequency_hz)
    discrete_angular_frequency = float(2 * np.sin(0.5 * theta) / dt)
    return (
        float(theta),
        physical_angular_frequency,
        discrete_angular_frequency,
        midpoint_cosine,
    )


def evaluate_surface_ade(
    *,
    frequency_hz: float,
    dt: float,
    F: npt.ArrayLike,
    G: npt.ArrayLike,
    L: npt.ArrayLike,
    Z0: float,
) -> AlgorithmicSurfaceResponse:
    """Evaluate the exact harmonic transfer of the runtime ADE recurrence."""

    (
        theta,
        physical_angular_frequency,
        discrete_angular_frequency,
        midpoint_cosine,
    ) = _validated_phase(frequency_hz, dt)
    F = np.asarray(F, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64).reshape(-1)
    L = np.asarray(L, dtype=np.float64).reshape(-1)
    Z0 = float(Z0)
    order = G.size
    if F.shape != (order, order) or L.size != order:
        raise ValueError("surface-ADE F, G, and L dimensions are inconsistent")
    if not (
        np.all(np.isfinite(F))
        and np.all(np.isfinite(G))
        and np.all(np.isfinite(L))
        and np.isfinite(Z0)
    ):
        raise ValueError("surface-ADE coefficients must be finite")

    impedance = complex(Z0)
    if order:
        z = np.exp(1j * theta)
        impedance += complex(L @ np.linalg.solve(z * np.eye(order) - F, G))
    tolerance = 256 * np.finfo(np.float64).eps * max(1.0, abs(Z0), abs(impedance))
    if abs(impedance) <= tolerance:
        raise ValueError("surface-ADE algorithmic impedance is zero at the requested frequency")
    admittance = midpoint_cosine / impedance
    return AlgorithmicSurfaceResponse(
        theta=theta,
        physical_angular_frequency=physical_angular_frequency,
        discrete_angular_frequency=discrete_angular_frequency,
        midpoint_cosine=midpoint_cosine,
        impedance=impedance,
        admittance=admittance,
    )


def boundary_edge_relative_permittivity(
    *,
    response: AlgorithmicSurfaceResponse,
    epsilon0: float,
    retained_dual_area: float,
    electric_mass: float,
    conductive_mass: float,
    port_lengths: npt.ArrayLike,
    port_admittances: npt.ArrayLike | None = None,
    normalization_angular_frequency: float | None = None,
) -> complex:
    """Return the exact-time-discrete electric coefficient for a boundary row.

    The integral harmonic Ampere row is

    ``C_H H = (j*Omega*m_eps + c*m_sigma + sum(length*K/E)) E``.

    The returned coefficient obeys

    ``j*omega_norm*epsilon0*Aret*eps_eff =``
    ``j*Omega*m_eps + cos(theta/2)*m_sigma + sum(length*K/E)``.

    ``omega_norm`` must match the P/Q solver's curl normalization. It defaults
    to the physical angular frequency for compatibility; a solver using the
    Yee time derivative supplies ``response.discrete_angular_frequency``.
    ``Omega`` is the exact discrete time-derivative frequency. For one model
    on every attached face, omit
    ``port_admittances`` and the admittance in ``response`` is used for all
    ports.
    """

    epsilon0 = float(epsilon0)
    retained_dual_area = float(retained_dual_area)
    electric_mass = float(electric_mass)
    conductive_mass = float(conductive_mass)
    normalization_angular_frequency = float(
        response.physical_angular_frequency
        if normalization_angular_frequency is None
        else normalization_angular_frequency
    )
    if not np.isfinite(normalization_angular_frequency) or normalization_angular_frequency <= 0:
        raise ValueError("surface boundary normalization angular frequency must be finite and positive")
    lengths = np.asarray(port_lengths, dtype=np.float64).reshape(-1)
    if port_admittances is None:
        admittances = np.full(lengths.shape, response.admittance, dtype=np.complex128)
    else:
        admittances = np.asarray(port_admittances, dtype=np.complex128).reshape(-1)
    if admittances.size != lengths.size:
        raise ValueError("surface port lengths and admittances must have identical lengths")
    if (
        not np.isfinite(epsilon0)
        or epsilon0 <= 0
        or not np.isfinite(retained_dual_area)
        or retained_dual_area <= 0
        or not np.isfinite(electric_mass)
        or electric_mass <= 0
        or not np.isfinite(conductive_mass)
        or conductive_mass < 0
        or not np.all(np.isfinite(lengths))
        or np.any(lengths <= 0)
        or not np.all(np.isfinite(admittances))
    ):
        raise ValueError("surface boundary metric and constitutive data must be passive and finite")

    load = response.midpoint_cosine * conductive_mass + np.dot(lengths, admittances)
    # The ADE numerator is exact for the runtime time discretisation. Map the
    # complete discrete load onto the same angular-frequency convention as
    # the solver's standard and clipped spatial curls.
    denominator = 1j * normalization_angular_frequency * epsilon0 * retained_dual_area
    discrete_mass = 1j * response.discrete_angular_frequency * electric_mass
    return complex((discrete_mass + load) / denominator)
