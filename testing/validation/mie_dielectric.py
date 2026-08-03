"""Analytical scattering by a homogeneous non-magnetic dielectric sphere."""

import numpy as np
import numpy.typing as npt
from scipy.constants import c
from scipy.special import spherical_jn, spherical_yn


def _maximum_order(size_parameter: float) -> int:
    """Return a converged Mie-series truncation order."""

    return max(1, int(np.ceil(size_parameter + 4 * np.cbrt(size_parameter) + 2)))


def _riccati_bessel(
    orders: npt.NDArray[np.int64], argument: complex
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Return the Riccati-Bessel function and its argument derivative."""

    bessel = spherical_jn(orders, argument)
    derivative = spherical_jn(orders, argument, derivative=True)
    return (
        np.asarray(argument * bessel, dtype=np.complex128),
        np.asarray(bessel + argument * derivative, dtype=np.complex128),
    )


def dielectric_mie_coefficients(
    size_parameter: float,
    relative_permittivity: complex,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Return electric and magnetic Mie coefficients for orders 1..N.

    The sphere is homogeneous, isotropic, and non-magnetic, and the exterior
    medium is free space. Complex relative permittivity is accepted for a
    passive lossy dielectric using the engineering ``exp(j*omega*t)``
    convention, for which its imaginary part is non-positive.
    """

    if not np.isfinite(size_parameter) or size_parameter <= 0:
        raise ValueError("size_parameter must be finite and greater than zero")
    relative_permittivity = complex(relative_permittivity)
    if not np.isfinite(relative_permittivity):
        raise ValueError("relative_permittivity must be finite")
    if relative_permittivity.real <= 0:
        raise ValueError("relative_permittivity must have a positive real part")
    if relative_permittivity.imag > 0:
        raise ValueError(
            "relative_permittivity must have a non-positive imaginary part "
            "for the exp(j*omega*t) convention"
        )

    refractive_index = np.sqrt(relative_permittivity)
    orders = np.arange(1, _maximum_order(size_parameter) + 1)
    psi, psi_derivative = _riccati_bessel(orders, size_parameter)
    psi_internal, psi_internal_derivative = _riccati_bessel(
        orders, refractive_index * size_parameter
    )

    bessel = spherical_jn(orders, size_parameter)
    neumann = spherical_yn(orders, size_parameter)
    bessel_derivative = spherical_jn(orders, size_parameter, derivative=True)
    neumann_derivative = spherical_yn(orders, size_parameter, derivative=True)
    # Engineering exp(j*omega*t) uses an outgoing spherical Hankel function
    # of the second kind.
    xi = size_parameter * (bessel - 1j * neumann)
    xi_derivative = (
        bessel - 1j * neumann + size_parameter * (bessel_derivative - 1j * neumann_derivative)
    )

    electric = (
        refractive_index * psi_internal * psi_derivative - psi * psi_internal_derivative
    ) / (refractive_index * psi_internal * xi_derivative - xi * psi_internal_derivative)
    magnetic = (
        psi_internal * psi_derivative - refractive_index * psi * psi_internal_derivative
    ) / (psi_internal * xi_derivative - refractive_index * xi * psi_internal_derivative)
    return electric.astype(np.complex128), magnetic.astype(np.complex128)


def dielectric_mie_amplitudes(
    size_parameter: float,
    relative_permittivity: complex,
    scattering_angles: npt.ArrayLike,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Return perpendicular and parallel dimensionless amplitudes ``S1,S2``."""

    angles = np.atleast_1d(np.asarray(scattering_angles, dtype=np.float64))
    if angles.ndim != 1 or not np.all(np.isfinite(angles)):
        raise ValueError("scattering_angles must be a finite one-dimensional array")
    electric, magnetic = dielectric_mie_coefficients(size_parameter, relative_permittivity)
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


def dielectric_sphere_bistatic_rcs(
    frequency: float,
    radius: float,
    relative_permittivity: complex,
    scattering_angles: npt.ArrayLike,
    *,
    polarisation: str = "perpendicular",
) -> npt.NDArray[np.float64]:
    """Return dielectric-sphere bistatic RCS in square metres."""

    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be finite and greater than zero")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and greater than zero")
    wavenumber = 2 * np.pi * frequency / c
    perpendicular, parallel = dielectric_mie_amplitudes(
        wavenumber * radius,
        relative_permittivity,
        scattering_angles,
    )
    if polarisation == "perpendicular":
        amplitude = perpendicular
    elif polarisation == "parallel":
        amplitude = parallel
    else:
        raise ValueError("polarisation must be 'perpendicular' or 'parallel'")
    return np.asarray(4 * np.pi * np.abs(amplitude) ** 2 / wavenumber**2)
