"""Aden-Kerker analytical scattering by a concentric core-shell sphere."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.constants import c
from scipy.special import spherical_jn, spherical_yn

from .pole_models import DispersiveModel


def _maximum_order(*size_parameters: complex) -> int:
    largest = max(abs(value) for value in size_parameters)
    return max(1, int(np.ceil(largest + 4 * np.cbrt(largest) + 2)))


def _psi(order: np.ndarray, argument: complex) -> tuple[np.ndarray, np.ndarray]:
    value = spherical_jn(order, argument)
    derivative = spherical_jn(order, argument, derivative=True)
    return argument * value, value + argument * derivative


def _chi(order: np.ndarray, argument: complex) -> tuple[np.ndarray, np.ndarray]:
    value = spherical_yn(order, argument)
    derivative = spherical_yn(order, argument, derivative=True)
    return -argument * value, -(value + argument * derivative)


def _xi_outgoing(order: np.ndarray, argument: complex) -> tuple[np.ndarray, np.ndarray]:
    """Engineering-convention outgoing Riccati-Hankel function."""

    psi, psi_derivative = _psi(order, argument)
    chi, chi_derivative = _chi(order, argument)
    return psi + 1j * chi, psi_derivative + 1j * chi_derivative


def _passive_index(relative_permittivity: complex) -> complex:
    index = np.sqrt(complex(relative_permittivity))
    if index.imag > 0 or (np.isclose(index.imag, 0.0) and index.real < 0):
        index = -index
    return index


def coated_sphere_coefficients(
    exterior_size_parameter: float,
    core_size_parameter: float,
    core_relative_permittivity: complex,
    shell_relative_permittivity: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Return electric and magnetic coefficients for a non-magnetic sphere.

    ``exterior_size_parameter`` and ``core_size_parameter`` are the outer and
    core radii multiplied by the exterior-medium wavenumber.  The exterior is
    free space.  Complex permittivities use the engineering ``exp(+j omega
    t)`` convention.
    """

    x = float(exterior_size_parameter)
    y = float(core_size_parameter)
    if not 0 < y < x:
        raise ValueError(
            "The core size parameter must lie strictly between zero and the outer value"
        )
    m_core = _passive_index(core_relative_permittivity)
    m_shell = _passive_index(shell_relative_permittivity)
    orders = np.arange(1, _maximum_order(x, m_shell * x, m_core * y) + 1)

    psi_x, psi_x_derivative = _psi(orders, x)
    xi_x, xi_x_derivative = _xi_outgoing(orders, x)
    psi_core, psi_core_derivative = _psi(orders, m_core * y)
    psi_shell_core, psi_shell_core_derivative = _psi(orders, m_shell * y)
    chi_shell_core, chi_shell_core_derivative = _chi(orders, m_shell * y)

    electric_internal = (
        m_shell * psi_shell_core * psi_core_derivative
        - m_core * psi_shell_core_derivative * psi_core
    ) / (
        m_shell * chi_shell_core * psi_core_derivative
        - m_core * chi_shell_core_derivative * psi_core
    )
    magnetic_internal = (
        m_shell * psi_shell_core_derivative * psi_core
        - m_core * psi_shell_core * psi_core_derivative
    ) / (
        m_shell * chi_shell_core_derivative * psi_core
        - m_core * chi_shell_core * psi_core_derivative
    )

    psi_shell_outer, psi_shell_outer_derivative = _psi(orders, m_shell * x)
    chi_shell_outer, chi_shell_outer_derivative = _chi(orders, m_shell * x)
    electric_radial = psi_shell_outer - electric_internal * chi_shell_outer
    electric_radial_derivative = (
        psi_shell_outer_derivative - electric_internal * chi_shell_outer_derivative
    )
    magnetic_radial = psi_shell_outer - magnetic_internal * chi_shell_outer
    magnetic_radial_derivative = (
        psi_shell_outer_derivative - magnetic_internal * chi_shell_outer_derivative
    )

    electric = (
        psi_x * electric_radial_derivative - m_shell * psi_x_derivative * electric_radial
    ) / (xi_x * electric_radial_derivative - m_shell * xi_x_derivative * electric_radial)
    magnetic = (
        m_shell * psi_x * magnetic_radial_derivative - psi_x_derivative * magnetic_radial
    ) / (m_shell * xi_x * magnetic_radial_derivative - xi_x_derivative * magnetic_radial)
    return np.asarray(electric, complex), np.asarray(magnetic, complex)


def coated_sphere_backscatter_rcs(
    frequencies: Sequence[float] | np.ndarray,
    core_radius: float,
    outer_radius: float,
    core: DispersiveModel,
    shell: DispersiveModel,
) -> np.ndarray:
    """Return monostatic RCS of a dispersive concentric sphere in square metres."""

    frequencies = np.asarray(frequencies, dtype=float)
    if np.any(frequencies <= 0) or not 0 < core_radius < outer_radius:
        raise ValueError("Frequencies and sphere radii must be positive and properly ordered")
    core_epsilon = core.relative_permittivity(frequencies)
    shell_epsilon = shell.relative_permittivity(frequencies)
    result = np.empty_like(frequencies)
    for index, frequency in enumerate(frequencies):
        wavenumber = 2 * np.pi * frequency / c
        electric, magnetic = coated_sphere_coefficients(
            wavenumber * outer_radius,
            wavenumber * core_radius,
            core_epsilon[index],
            shell_epsilon[index],
        )
        orders = np.arange(1, len(electric) + 1)
        backscatter_sum = np.sum((2 * orders + 1) * (-1) ** orders * (electric - magnetic))
        result[index] = np.pi * abs(backscatter_sum) ** 2 / wavenumber**2
    return result
