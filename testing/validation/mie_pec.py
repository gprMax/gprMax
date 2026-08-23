"""Analytical far-field scattering by a perfectly conducting sphere."""

import numpy as np
import numpy.typing as npt
from scipy.constants import c
from scipy.special import spherical_jn, spherical_yn


def _maximum_order(size_parameter: float) -> int:
    """Return the usual converged Mie-series truncation order."""

    return max(1, int(np.ceil(size_parameter + 4 * np.cbrt(size_parameter) + 2)))


def pec_mie_coefficients(
    size_parameter: float,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Return electric and magnetic PEC Mie coefficients for orders 1..N."""

    if not np.isfinite(size_parameter) or size_parameter <= 0:
        raise ValueError("size_parameter must be finite and greater than zero")
    orders = np.arange(1, _maximum_order(size_parameter) + 1)
    jn = spherical_jn(orders, size_parameter)
    yn = spherical_yn(orders, size_parameter)
    jn_derivative = spherical_jn(orders, size_parameter, derivative=True)
    yn_derivative = spherical_yn(orders, size_parameter, derivative=True)
    psi = size_parameter * jn
    psi_derivative = jn + size_parameter * jn_derivative
    xi = size_parameter * (jn + 1j * yn)
    xi_derivative = jn + 1j * yn + size_parameter * (jn_derivative + 1j * yn_derivative)
    electric = -psi_derivative / xi_derivative
    magnetic = -psi / xi
    return electric.astype(np.complex128), magnetic.astype(np.complex128)


def pec_mie_amplitudes(
    size_parameter: float, scattering_angles: npt.ArrayLike
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Return perpendicular and parallel dimensionless amplitudes ``S1,S2``."""

    angles = np.atleast_1d(np.asarray(scattering_angles, dtype=np.float64))
    if angles.ndim != 1 or not np.all(np.isfinite(angles)):
        raise ValueError("scattering_angles must be a finite one-dimensional array")
    electric, magnetic = pec_mie_coefficients(size_parameter)
    cosine = np.cos(angles)
    perpendicular = np.zeros(angles.shape, dtype=np.complex128)
    parallel = np.zeros_like(perpendicular)
    pi_previous = np.zeros_like(cosine)
    pi_current = np.ones_like(cosine)

    for order, (electric_n, magnetic_n) in enumerate(zip(electric, magnetic), start=1):
        if order == 1:
            pi_n = pi_current
        else:
            pi_n = ((2 * order - 1) * cosine * pi_current - order * pi_previous) / (order - 1)
            pi_previous, pi_current = pi_current, pi_n
        tau_n = order * cosine * pi_n - (order + 1) * pi_previous
        factor = (2 * order + 1) / (order * (order + 1))
        perpendicular += factor * (electric_n * pi_n + magnetic_n * tau_n)
        parallel += factor * (electric_n * tau_n + magnetic_n * pi_n)

    return perpendicular, parallel


def pec_sphere_bistatic_rcs(
    frequency: float,
    radius: float,
    scattering_angles: npt.ArrayLike,
    *,
    polarisation: str = "perpendicular",
) -> npt.NDArray[np.float64]:
    """Return PEC-sphere bistatic RCS in square metres."""

    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be finite and greater than zero")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and greater than zero")
    wavenumber = 2 * np.pi * frequency / c
    perpendicular, parallel = pec_mie_amplitudes(wavenumber * radius, scattering_angles)
    if polarisation == "perpendicular":
        amplitude = perpendicular
    elif polarisation == "parallel":
        amplitude = parallel
    else:
        raise ValueError("polarisation must be 'perpendicular' or 'parallel'")
    return np.asarray(4 * np.pi * np.abs(amplitude) ** 2 / wavenumber**2)
