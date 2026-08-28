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

"""Antenna-pattern quadrature and dimensionless radiation metrics."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SphericalQuadrature:
    """Gauss--Legendre/punctured-periodic quadrature over a unit sphere."""

    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    weights: npt.NDArray[np.floating]
    theta_order: int
    phi_order: int
    enclosure_radius: float


def spherical_quadrature(
    enclosure_radius: float,
    maximum_wavenumber: float,
    dtype: npt.DTypeLike,
    *,
    include_equator: bool = True,
) -> SphericalQuadrature:
    """Build an angular grid capable of resolving a bounded radiator.

    A source enclosed by radius ``a`` has an angular field bandwidth of
    approximately ``ka``. Radiation intensity is quadratic in field, so the
    conservative order below resolves approximately ``2ka`` plus a fixed
    margin. Gauss--Legendre weights integrate in ``mu = cos(theta)`` directly;
    periodic phi samples deliberately omit the duplicate 360-degree endpoint.
    """

    if not np.isfinite(enclosure_radius) or enclosure_radius <= 0:
        raise ValueError("enclosure_radius must be finite and positive")
    if not np.isfinite(maximum_wavenumber) or maximum_wavenumber < 0:
        raise ValueError("maximum_wavenumber must be finite and non-negative")

    real_dtype = np.dtype(dtype)
    if real_dtype.kind != "f":
        raise ValueError("spherical quadrature requires a real floating dtype")
    angular_bandlimit = max(
        4,
        int(np.ceil(2 * maximum_wavenumber * enclosure_radius)) + 6,
    )
    theta_order = max(12, angular_bandlimit + 1)
    # An odd Gauss--Legendre order includes mu=0 (theta=90 degrees). This
    # avoids a systematic underestimate of the common broadside maximum while
    # retaining the exact Gauss--Legendre integration rule.
    if include_equator:
        if theta_order % 2 == 0:
            theta_order += 1
    elif theta_order % 2 != 0:
        theta_order += 1
    phi_order = max(24, 2 * angular_bandlimit + 1)

    mu64, mu_weights64 = np.polynomial.legendre.leggauss(theta_order)
    # leggauss returns ascending mu, hence descending theta. Reverse it so
    # public metadata follows the conventional north-to-south ordering.
    mu64 = mu64[::-1]
    mu_weights64 = mu_weights64[::-1]
    theta_axis = np.arccos(mu64)
    phi_axis = 2 * np.pi * np.arange(phi_order, dtype=np.float64) / phi_order
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis, indexing="ij")
    weights = mu_weights64[:, np.newaxis] * (2 * np.pi / phi_order)
    weights = np.broadcast_to(weights, theta_grid.shape)

    theta = np.ascontiguousarray(np.rad2deg(theta_grid).ravel(), dtype=real_dtype)
    phi = np.ascontiguousarray(np.rad2deg(phi_grid).ravel(), dtype=real_dtype)
    quadrature_weights = np.ascontiguousarray(weights.ravel(), dtype=real_dtype)
    for values in (theta, phi, quadrature_weights):
        values.setflags(write=False)
    return SphericalQuadrature(
        theta=theta,
        phi=phi,
        weights=quadrature_weights,
        theta_order=theta_order,
        phi_order=phi_order,
        enclosure_radius=float(enclosure_radius),
    )


def radiation_intensity(
    electric_cartesian: npt.ArrayLike,
    theta: npt.ArrayLike,
    phi: npt.ArrayLike,
    impedance: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """Return range-normalized radiation intensity from Cartesian fields."""

    from gprMax.ntff.evaluator import project_cartesian_to_spherical

    electric = np.asarray(electric_cartesian)
    if electric.ndim != 3 or electric.shape[2] != 3 or electric.dtype.kind != "c":
        raise ValueError("electric_cartesian must have shape (nf, nd, 3) and be complex")
    impedance_values = np.asarray(impedance, dtype=np.empty((), dtype=electric.dtype).real.dtype)
    if np.any(~np.isfinite(impedance_values)) or np.any(impedance_values <= 0):
        raise ValueError("impedance must contain finite positive values")
    if impedance_values.ndim == 0:
        denominator = impedance_values
    elif impedance_values.shape == electric.shape[:2]:
        denominator = impedance_values
    else:
        raise ValueError("impedance must be scalar or have shape (nf, nd)")
    spherical = project_cartesian_to_spherical(
        electric,
        theta,
        phi,
        degrees=True,
    )
    tangential_squared = np.abs(spherical[:, :, 1]) ** 2 + np.abs(spherical[:, :, 2]) ** 2
    real_dtype = np.empty((), dtype=electric.dtype).real.dtype
    return np.asarray(0.5 * tangential_squared / denominator, dtype=real_dtype)


def directivity_from_intensity(
    intensity: npt.ArrayLike,
    radiated_power: npt.ArrayLike,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Calculate linear directivity and dBi with explicit invalid masks."""

    values = np.asarray(intensity)
    power = np.asarray(radiated_power, dtype=values.dtype)
    if values.ndim != 2 or power.shape != (values.shape[0],):
        raise ValueError("intensity and radiated_power shapes are inconsistent")
    directivity = np.full(values.shape, np.nan, dtype=values.dtype)
    finite_power = np.isfinite(power) & (power > 0)
    directivity[finite_power] = 4 * np.pi * values[finite_power] / power[finite_power, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        directivity_dbi = 10 * np.log10(directivity)
    return directivity, np.asarray(directivity_dbi, dtype=values.dtype)
