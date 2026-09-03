"""Yee difference symbols shared by the scalar and vector modal solvers."""

import numpy as np


def positive_finite(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return value


def discrete_angular_frequency(frequency, fdtd_dt=None):
    """Return the leapfrog time symbol, or physical omega without a time step.

    For exp(+j*omega*t), the staggered derivative is j*Omega, where
    Omega = 2*sin(omega*fdtd_dt/2)/fdtd_dt. Use sinc for the small-step limit.
    Material pole models and waveform phases still use the physical frequency.
    """
    frequency = positive_finite(frequency, "frequency")
    omega = 2 * np.pi * frequency
    if fdtd_dt is None:
        return omega
    fdtd_dt = positive_finite(fdtd_dt, "fdtd_dt")
    if frequency * fdtd_dt >= 0.5:
        raise ValueError("FDFD frequency must lie below temporal Nyquist.")
    return omega * np.sinc(frequency * fdtd_dt)


def phase_propagation_constant(wavenumber, propagation_spacing=None):
    """Invert K = 2*sin(beta*dw/2)/dw on the passive first-zone branch.

    The eigenproblem solves for the longitudinal difference symbol K. Its
    phase beta belongs in exp(-j*beta*w), but must not replace K in Maxwell's
    field reconstruction. Real symbols beyond the spatial band edge continue
    into the decaying grid stop band, including at an exactly zero loss.
    """
    wavenumber = np.asarray(wavenumber, dtype=np.complex128)
    if propagation_spacing is None:
        return wavenumber.copy()
    spacing = positive_finite(propagation_spacing, "propagation_spacing")
    beta = 2 * np.arcsin(0.5 * spacing * wavenumber) / spacing
    return np.where(wavenumber.imag == 0, beta.real - 1j * np.abs(beta.imag), beta)


def spatially_resolved(wavenumber, propagation_spacing):
    """Exclude lossless grid stop-band modes from forward source power."""
    wavenumber = np.asarray(wavenumber, dtype=np.complex128)
    if propagation_spacing is None:
        return np.ones(wavenumber.shape, dtype=bool)
    symbol = 0.5 * propagation_spacing * wavenumber
    lossless = np.abs(symbol.imag) <= 1e-12 * np.maximum(1.0, np.abs(symbol))
    return ~(lossless & (np.abs(symbol.real) >= 1.0))
