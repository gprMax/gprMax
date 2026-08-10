"""Analytical normal-incidence reflection from dispersive planar stacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.constants import c, epsilon_0, mu_0

from .pole_models import DispersiveModel


@dataclass(frozen=True)
class PlanarMedium:
    """One homogeneous isotropic medium in a planar stack."""

    material: DispersiveModel
    relative_permeability: complex = 1.0
    thickness: float | None = None


def _passive_refractive_index(epsilon_r: np.ndarray, mu_r: complex) -> np.ndarray:
    """Choose the passive branch for the ``exp(+j omega t)`` convention."""

    refractive_index = np.sqrt(epsilon_r * mu_r + 0j)
    flip = (refractive_index.imag > 0) | (
        np.isclose(refractive_index.imag, 0.0) & (refractive_index.real < 0)
    )
    return np.where(flip, -refractive_index, refractive_index)


def _passive_impedance(epsilon_r: np.ndarray, mu_r: complex) -> np.ndarray:
    impedance = np.sqrt(mu_0 * mu_r / (epsilon_0 * epsilon_r) + 0j)
    flip = impedance.real < 0
    return np.where(flip, -impedance, impedance)


def normal_incidence_reflection(
    frequencies: Sequence[float] | np.ndarray,
    incident: PlanarMedium,
    finite_layers: Sequence[PlanarMedium],
    substrate: PlanarMedium,
    *,
    reference_distance: float = 0.0,
) -> np.ndarray:
    """Return the exact complex electric-field reflection coefficient.

    The incident medium and substrate are semi-infinite.  Every entry in
    ``finite_layers`` must have a positive thickness.  A recursive Fresnel
    form is used instead of multiplying transfer matrices, avoiding overflow
    in lossy stacks.
    """

    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or np.any(frequencies <= 0):
        raise ValueError("Frequencies must be a positive one-dimensional array")
    if reference_distance < 0:
        raise ValueError("Reference distance cannot be negative")
    for layer in finite_layers:
        if layer.thickness is None or layer.thickness <= 0:
            raise ValueError("Each finite layer requires a positive thickness")

    media = (incident, *finite_layers, substrate)
    epsilon = [medium.material.relative_permittivity(frequencies) for medium in media]
    impedance = [
        _passive_impedance(epsilon_r, medium.relative_permeability)
        for epsilon_r, medium in zip(epsilon, media)
    ]
    wavenumber = [
        2
        * np.pi
        * frequencies
        / c
        * _passive_refractive_index(epsilon_r, medium.relative_permeability)
        for epsilon_r, medium in zip(epsilon, media)
    ]

    interface_reflection = [
        (impedance[index + 1] - impedance[index]) / (impedance[index + 1] + impedance[index])
        for index in range(len(media) - 1)
    ]
    reflection = interface_reflection[-1]
    for index in range(len(finite_layers) - 1, -1, -1):
        layer_index = index + 1
        round_trip = np.exp(-2j * wavenumber[layer_index] * finite_layers[index].thickness)
        local = interface_reflection[index]
        reflection = (local + reflection * round_trip) / (1 + local * reflection * round_trip)

    if reference_distance:
        reflection *= np.exp(-2j * wavenumber[0] * reference_distance)
    return reflection


def single_interface_fresnel(
    frequencies: Sequence[float] | np.ndarray,
    incident: PlanarMedium,
    substrate: PlanarMedium,
) -> np.ndarray:
    """Convenience form used to verify the multilayer recursion."""

    frequencies = np.asarray(frequencies, dtype=float)
    epsilon_incident = incident.material.relative_permittivity(frequencies)
    epsilon_substrate = substrate.material.relative_permittivity(frequencies)
    impedance_incident = _passive_impedance(epsilon_incident, incident.relative_permeability)
    impedance_substrate = _passive_impedance(epsilon_substrate, substrate.relative_permeability)
    return (impedance_substrate - impedance_incident) / (impedance_substrate + impedance_incident)
