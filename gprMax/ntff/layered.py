# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Planar-layered equivalent-current near-to-far-field transformation.

The implementation follows the transmission-line Green-function formulation
of Capoglu, Taflove, and Backman, IEEE TAP 60(4), 1878--1885 (2012).  It uses
the engineering ``exp(+j omega t)`` convention and the same Love currents as
the homogeneous equivalent-current transform: ``J = n x H`` and
``M = -n x E``.

Only the frequency-domain propagation kernel lives here.  Surface sampling
and Yee-grid collocation remain shared with :mod:`gprMax.ntff.equivalent_currents`.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy.constants import epsilon_0, mu_0

from .equivalent_currents import EquivalentCurrentPhasors, collocate_love_currents

try:
    from gprMax.cython.ntff import (
        evaluate_layered_equivalent_current_far_zone as _evaluate_layered_cython,
    )
except ImportError:  # The pure-Python path remains usable before an extension rebuild.
    _evaluate_layered_cython = None


AXIS_BASES = {
    # Rows are the local (u, v, n) unit vectors in global Cartesian axes.
    # Every basis is right handed: u x v = n.
    "x": np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=float),
    "y": np.asarray(((0, 0, 1), (1, 0, 0), (0, 1, 0)), dtype=float),
    "z": np.eye(3),
}


@dataclass(frozen=True)
class LayeredMedium:
    """Resolved frequency-dependent properties of one planar stack.

    Materials and interfaces are ordered from the positive-axis exterior to
    the negative-axis exterior.  For ``N`` materials there are ``N - 1``
    strictly descending interface coordinates.
    """

    axis: str
    interfaces: npt.NDArray[np.floating]
    material_ids: tuple[str, ...]
    relative_permittivity: npt.NDArray[np.complexfloating]
    relative_permeability: npt.NDArray[np.complexfloating]

    def validate(self, frequencies: npt.ArrayLike) -> None:
        values = np.asarray(frequencies)
        interfaces = np.asarray(self.interfaces)
        eps = np.asarray(self.relative_permittivity)
        mu = np.asarray(self.relative_permeability)
        if self.axis not in AXIS_BASES:
            raise ValueError("layered-medium axis must be 'x', 'y', or 'z'")
        if len(self.material_ids) < 1:
            raise ValueError("a layered medium requires at least one material")
        if interfaces.shape != (len(self.material_ids) - 1,):
            raise ValueError("layered-medium interfaces and materials are inconsistent")
        if interfaces.size and (
            not np.all(np.isfinite(interfaces)) or not np.all(np.diff(interfaces) < 0)
        ):
            raise ValueError("layered-medium interfaces must be finite and strictly descending")
        expected = (values.size, len(self.material_ids))
        if eps.shape != expected or mu.shape != expected:
            raise ValueError(f"layered constitutive arrays must both have shape {expected}")
        if not np.all(np.isfinite(eps)) or not np.all(np.isfinite(mu)):
            raise ValueError("layered constitutive properties must be finite")
        if np.any(np.real(eps) <= 0) or np.any(np.real(mu) <= 0):
            raise ValueError(
                "layered materials require positive real permittivity and permeability"
            )
        # A conventional far field exists in the two semi-infinite exterior
        # regions only when they are lossless.  Finite internal layers may be
        # lossy and dispersive.
        for exterior in (0, -1):
            if not np.allclose(np.imag(eps[:, exterior]), 0, rtol=0, atol=1e-12) or not np.allclose(
                np.imag(mu[:, exterior]), 0, rtol=0, atol=1e-12
            ):
                raise ValueError("the two observation half-spaces must be lossless")


@dataclass(frozen=True)
class LayeredFarField:
    """Range-normalized layered far fields and direction-dependent media."""

    electric: npt.NDArray[np.complexfloating]
    magnetic: npt.NDArray[np.complexfloating]
    impedance: npt.NDArray[np.floating]
    wavenumber: npt.NDArray[np.floating]
    observation_material_index: npt.NDArray[np.int8]


def observation_properties(
    frequencies: npt.ArrayLike,
    directions: npt.ArrayLike,
    medium: LayeredMedium,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return exterior impedance and wavenumber for every frequency/direction."""

    frequency_values = np.asarray(frequencies, dtype=float)
    direction_values = np.asarray(directions, dtype=float)
    medium.validate(frequency_values)
    basis = AXIS_BASES[medium.axis]
    local_normal = direction_values @ basis[2]
    exterior = np.where(local_normal > 0, 0, -1)
    impedance = np.empty((frequency_values.size, direction_values.shape[0]), dtype=float)
    wavenumber = np.empty_like(impedance)
    for direction_number, material_index in enumerate(exterior):
        eps = np.real(medium.relative_permittivity[:, material_index])
        mu = np.real(medium.relative_permeability[:, material_index])
        impedance[:, direction_number] = np.sqrt(mu_0 * mu / (epsilon_0 * eps))
        wavenumber[:, direction_number] = (
            2 * np.pi * frequency_values * np.sqrt(epsilon_0 * eps * mu_0 * mu)
        )
    return impedance, wavenumber


def _outgoing_sqrt(values: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
    """Square root with the passive ``exp(+j omega t)`` outgoing branch."""

    result = np.sqrt(np.asarray(values, dtype=np.complex128))
    result = np.where(np.imag(result) > 0, -result, result)
    nearly_real = np.abs(np.imag(result)) <= 64 * np.finfo(float).eps
    result = np.where(nearly_real & (np.real(result) < 0), -result, result)
    return result


def _layer_thicknesses(interfaces: npt.ArrayLike, nlayers: int) -> npt.NDArray[np.floating]:
    thickness = np.zeros(nlayers, dtype=np.asarray(interfaces).dtype)
    if nlayers > 2:
        thickness[1:-1] = np.asarray(interfaces)[:-1] - np.asarray(interfaces)[1:]
    return thickness


def _safe_ratio(numerator: complex, denominator: complex, label: str) -> complex:
    scale = max(abs(numerator), abs(denominator), 1.0)
    if abs(denominator) <= 128 * np.finfo(float).eps * scale:
        raise FloatingPointError(f"singular planar-layered recursion in {label}")
    return numerator / denominator


def _voltage_coefficients(
    beta: npt.NDArray[np.complexfloating],
    line_impedance: npt.NDArray[np.complexfloating],
    thickness: npt.NDArray[np.floating],
    *,
    upper_observation: bool,
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    """Return upward/downward TL voltage-wave coefficients in every layer.

    This is an independent transcription of Eqs. (18)--(30) of Capoglu et
    al.  The lower-half-space launch sign is selected by the physical
    homogeneous-medium identity-dyadic limit, which also fixes the spherical
    basis orientation below the stack.
    """

    nlayers = line_impedance.size
    plus = np.zeros(nlayers, dtype=np.complex128)
    minus = np.zeros(nlayers, dtype=np.complex128)
    if nlayers == 1:
        if upper_observation:
            minus[0] = line_impedance[0]
        else:
            plus[0] = line_impedance[0]
        return plus, minus

    interface_impedance = np.empty(nlayers - 1, dtype=np.complex128)
    if upper_observation:
        interface_impedance[-1] = -line_impedance[-1]
        for layer in range(nlayers - 2, 0, -1):
            tangent = np.tan(beta[layer] * thickness[layer])
            load = interface_impedance[layer]
            eta = line_impedance[layer]
            interface_impedance[layer - 1] = eta * _safe_ratio(
                load - 1j * eta * tangent,
                eta - 1j * load * tangent,
                "upper input impedance",
            )

        minus[0] = line_impedance[0]
        for layer in range(1, nlayers):
            load = interface_impedance[layer - 1]
            propagation = np.exp(-1j * beta[layer - 1] * thickness[layer - 1])
            minus[layer] = (
                minus[layer - 1]
                * _safe_ratio(
                    load - line_impedance[layer],
                    load - line_impedance[layer - 1],
                    "upper transmitted voltage",
                )
                * propagation
            )
        for layer in range(nlayers - 1):
            load = interface_impedance[layer]
            plus[layer] = (
                minus[layer]
                * _safe_ratio(
                    load + line_impedance[layer],
                    load - line_impedance[layer],
                    "upper reflected voltage",
                )
                * np.exp(-1j * beta[layer] * thickness[layer])
            )
    else:
        interface_impedance[0] = line_impedance[0]
        for layer in range(1, nlayers - 1):
            tangent = np.tan(beta[layer] * thickness[layer])
            load = interface_impedance[layer - 1]
            eta = line_impedance[layer]
            interface_impedance[layer] = eta * _safe_ratio(
                load + 1j * eta * tangent,
                eta + 1j * load * tangent,
                "lower input impedance",
            )

        plus[-1] = line_impedance[-1]
        for layer in range(nlayers - 1, 0, -1):
            load = interface_impedance[layer - 1]
            propagation = np.exp(-1j * beta[layer] * thickness[layer])
            plus[layer - 1] = (
                plus[layer]
                * _safe_ratio(
                    load + line_impedance[layer - 1],
                    load + line_impedance[layer],
                    "lower transmitted voltage",
                )
                * propagation
            )
        for layer in range(1, nlayers):
            load = interface_impedance[layer - 1]
            minus[layer] = (
                plus[layer]
                * _safe_ratio(
                    load - line_impedance[layer],
                    load + line_impedance[layer],
                    "lower reflected voltage",
                )
                * np.exp(-1j * beta[layer] * thickness[layer])
            )
    return plus, minus


def _responses_at_positions(
    positions: npt.NDArray[np.floating],
    layer_index: npt.NDArray[np.integer],
    interfaces: npt.NDArray[np.floating],
    beta: npt.NDArray[np.complexfloating],
    line_impedance: npt.NDArray[np.complexfloating],
    plus: npt.NDArray[np.complexfloating],
    minus: npt.NDArray[np.complexfloating],
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    """Evaluate the TL current- and voltage-source responses at patch depths."""

    nlayers = line_impedance.size
    vi = np.empty(positions.size, dtype=np.complex128)
    vv = np.empty(positions.size, dtype=np.complex128)
    for layer in range(nlayers):
        selected = layer_index == layer
        if not np.any(selected):
            continue
        if nlayers == 1:
            # With no physical interface, use the transform origin as the
            # common fictitious reference plane. ``positions`` are already
            # relative to that origin.
            top = bottom = 0.0
        else:
            top = interfaces[0] if layer == 0 else interfaces[layer - 1]
            bottom = interfaces[-1] if layer == nlayers - 1 else interfaces[layer]
        # Exterior fictitious interfaces coincide with their only physical
        # boundary, removing arbitrary exterior phase references.
        z = positions[selected]
        phase_plus = np.exp(-1j * beta[layer] * (z - bottom))
        phase_minus = np.exp(1j * beta[layer] * (z - top))
        vi[selected] = plus[layer] * phase_plus + minus[layer] * phase_minus
        vv[selected] = (-plus[layer] * phase_plus + minus[layer] * phase_minus) / line_impedance[
            layer
        ]
    return vi, vv


def _local_layer_indices(
    axial_positions: npt.NDArray[np.floating], interfaces: npt.NDArray[np.floating]
) -> npt.NDArray[np.int64]:
    # An exact interface point belongs to the layer on its positive-axis side.
    return np.searchsorted(-interfaces, -axial_positions, side="left").astype(np.int64)


def _cython_layered_currents(
    currents: EquivalentCurrentPhasors,
    frequency_values: npt.NDArray[np.floating],
    local_directions: npt.NDArray[np.floating],
    relative_positions: npt.NDArray[np.floating],
    local_j: npt.NDArray[np.complexfloating],
    local_m: npt.NDArray[np.complexfloating],
    interfaces: npt.NDArray[np.floating],
    layer_index: npt.NDArray[np.int64],
    thickness: npt.NDArray[np.floating],
    medium: LayeredMedium,
    grazing_tolerance: float,
    nthreads: int,
) -> tuple[
    npt.NDArray[np.complexfloating],
    npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.int8],
]:
    """Prepare the small spectral stack and execute the OpenMP surface sum."""

    complex_dtype = currents.electric_current.dtype
    # Frequency-domain surface accumulators can be complex128 even when the
    # FDTD geometry was built in single precision. Cython's fused real and
    # complex memoryviews must use a matching precision, so derive the real
    # working type from the retained phasors rather than from the positions.
    real_dtype = np.dtype(np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64)
    nf = frequency_values.size
    nd = local_directions.shape[0]
    nl = len(medium.material_ids)
    shape = (nf, nd, nl)
    beta = np.empty(shape, dtype=complex_dtype)
    epsilon_ratio = np.empty(shape, dtype=complex_dtype)
    permeability_ratio = np.empty(shape, dtype=complex_dtype)
    eta_e = np.empty(shape, dtype=complex_dtype)
    eta_h = np.empty(shape, dtype=complex_dtype)
    plus_e = np.empty(shape, dtype=complex_dtype)
    minus_e = np.empty(shape, dtype=complex_dtype)
    plus_h = np.empty(shape, dtype=complex_dtype)
    minus_h = np.empty(shape, dtype=complex_dtype)
    observation_impedance = np.empty((nf, nd), dtype=real_dtype)
    observation_wavenumber = np.empty((nf, nd), dtype=real_dtype)
    electric_factor = np.empty((nf, nd), dtype=real_dtype)
    magnetic_factor = np.empty((nf, nd), dtype=real_dtype)
    observation_index = np.where(local_directions[:, 2] > 0, 0, -1).astype(np.int8)

    for direction_number, direction in enumerate(local_directions):
        cos_theta = float(direction[2])
        if abs(cos_theta) <= grazing_tolerance:
            raise ValueError(
                "layered NTFF is singular at exact grazing incidence; omit theta=90 degrees"
            )
        upper_observation = cos_theta > 0
        exterior = 0 if upper_observation else -1
        sin_theta = float(np.hypot(direction[0], direction[1]))
        for frequency_number, frequency in enumerate(frequency_values):
            eps_absolute = np.asarray(medium.relative_permittivity[frequency_number])
            mu_absolute = np.asarray(medium.relative_permeability[frequency_number])
            eps_observation = float(np.real(eps_absolute[exterior]))
            mu_observation = float(np.real(mu_absolute[exterior]))
            angular_frequency = 2 * np.pi * float(frequency)
            k_observation = angular_frequency * np.sqrt(
                epsilon_0 * eps_observation * mu_0 * mu_observation
            )
            impedance = np.sqrt(mu_0 * mu_observation / (epsilon_0 * eps_observation))
            eps = eps_absolute / eps_observation
            mu = mu_absolute / mu_observation
            q = _outgoing_sqrt(eps * mu - sin_theta**2)
            beta_values = k_observation * q
            eta_e_values = q / eps
            eta_h_values = mu / q
            pe, me = _voltage_coefficients(
                beta_values, eta_e_values, thickness, upper_observation=upper_observation
            )
            ph, mh = _voltage_coefficients(
                beta_values, eta_h_values, thickness, upper_observation=upper_observation
            )
            if interfaces.size:
                if upper_observation:
                    exterior_phase = np.exp(1j * beta_values[0] * interfaces[0])
                else:
                    exterior_phase = np.exp(-1j * beta_values[-1] * interfaces[-1])
                pe *= exterior_phase
                me *= exterior_phase
                ph *= exterior_phase
                mh *= exterior_phase
            beta[frequency_number, direction_number] = beta_values
            epsilon_ratio[frequency_number, direction_number] = eps
            permeability_ratio[frequency_number, direction_number] = mu
            eta_e[frequency_number, direction_number] = eta_e_values
            eta_h[frequency_number, direction_number] = eta_h_values
            plus_e[frequency_number, direction_number] = pe
            minus_e[frequency_number, direction_number] = me
            plus_h[frequency_number, direction_number] = ph
            minus_h[frequency_number, direction_number] = mh
            observation_impedance[frequency_number, direction_number] = impedance
            observation_wavenumber[frequency_number, direction_number] = k_observation
            electric_factor[frequency_number, direction_number] = (
                angular_frequency * mu_0 * mu_observation / (4 * np.pi)
            )
            magnetic_factor[frequency_number, direction_number] = (
                angular_frequency * impedance * epsilon_0 * eps_observation / (4 * np.pi)
            )

    layer_top = np.zeros(nl, dtype=real_dtype)
    layer_bottom = np.zeros(nl, dtype=real_dtype)
    if interfaces.size:
        layer_top[0] = interfaces[0]
        layer_top[1:] = interfaces
        layer_bottom[:-1] = interfaces
        layer_bottom[-1] = interfaces[-1]

    electric_local = np.zeros((nf, nd, 3), dtype=complex_dtype)
    magnetic_local = np.zeros_like(electric_local)
    _evaluate_layered_cython(
        max(1, int(nthreads)),
        np.ascontiguousarray(relative_positions, dtype=real_dtype),
        np.ascontiguousarray(layer_index),
        layer_top,
        layer_bottom,
        np.ascontiguousarray(currents.area_weights, dtype=real_dtype),
        np.ascontiguousarray(local_directions, dtype=real_dtype),
        observation_wavenumber,
        observation_impedance,
        electric_factor,
        magnetic_factor,
        beta,
        epsilon_ratio,
        permeability_ratio,
        eta_e,
        eta_h,
        plus_e,
        minus_e,
        plus_h,
        minus_h,
        np.ascontiguousarray(local_j, dtype=complex_dtype),
        np.ascontiguousarray(local_m, dtype=complex_dtype),
        electric_local,
        magnetic_local,
    )
    return (
        electric_local,
        magnetic_local,
        observation_impedance,
        observation_wavenumber,
        observation_index,
    )


def evaluate_layered_currents(
    currents: EquivalentCurrentPhasors,
    frequencies: npt.ArrayLike,
    directions: npt.ArrayLike,
    medium: LayeredMedium,
    *,
    origin: npt.ArrayLike = (0.0, 0.0, 0.0),
    grazing_tolerance: float = 1e-8,
    nthreads: int = 1,
    direction_block_size: int = 256,
) -> LayeredFarField:
    """Evaluate ``r exp(+j k_obs r) E,H`` for a planar layered medium."""

    complex_dtype = np.asarray(currents.electric_current).dtype
    real_dtype = np.dtype(np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64)
    frequency_values = np.asarray(frequencies, dtype=real_dtype)
    direction_values = np.asarray(directions, dtype=real_dtype)
    origin_values = np.asarray(origin, dtype=real_dtype)
    if (
        frequency_values.ndim != 1
        or frequency_values.size == 0
        or not np.all(np.isfinite(frequency_values))
        or np.any(frequency_values <= 0)
    ):
        raise ValueError("layered NTFF frequencies must be finite and strictly positive")
    if direction_values.ndim != 2 or direction_values.shape[1] != 3:
        raise ValueError("directions must have shape (ndirections, 3)")
    if not np.allclose(np.linalg.norm(direction_values, axis=1), 1, rtol=1e-6, atol=1e-7):
        raise ValueError("directions must contain unit vectors")
    if origin_values.shape != (3,) or not np.all(np.isfinite(origin_values)):
        raise ValueError("origin must contain three finite values")
    if not np.isfinite(grazing_tolerance) or grazing_tolerance <= 0:
        raise ValueError("grazing_tolerance must be finite and positive")
    if not isinstance(direction_block_size, (int, np.integer)) or direction_block_size <= 0:
        raise ValueError("direction_block_size must be a positive integer")
    medium.validate(frequency_values)

    # The TE/TM coefficients scale as frequencies x directions x layers.
    # Bound their temporary memory without changing the retained result size.
    if direction_values.shape[0] > direction_block_size:
        electric = None
        magnetic = None
        impedance = None
        wavenumber = None
        observation_index = np.empty(direction_values.shape[0], dtype=np.int8)
        for start in range(0, direction_values.shape[0], direction_block_size):
            stop = min(start + direction_block_size, direction_values.shape[0])
            block = evaluate_layered_currents(
                currents,
                frequency_values,
                direction_values[start:stop],
                medium,
                origin=origin_values,
                grazing_tolerance=grazing_tolerance,
                nthreads=nthreads,
                direction_block_size=direction_block_size,
            )
            if electric is None:
                electric = np.empty(
                    (frequency_values.size, direction_values.shape[0], 3),
                    dtype=block.electric.dtype,
                )
                magnetic = np.empty_like(electric)
                impedance = np.empty(
                    (frequency_values.size, direction_values.shape[0]),
                    dtype=block.impedance.dtype,
                )
                wavenumber = np.empty_like(impedance)
            electric[:, start:stop] = block.electric
            magnetic[:, start:stop] = block.magnetic
            impedance[:, start:stop] = block.impedance
            wavenumber[:, start:stop] = block.wavenumber
            observation_index[start:stop] = block.observation_material_index
        assert electric is not None
        assert magnetic is not None
        assert impedance is not None
        assert wavenumber is not None
        for array in (electric, magnetic, impedance, wavenumber, observation_index):
            array.setflags(write=False)
        return LayeredFarField(
            electric=electric,
            magnetic=magnetic,
            impedance=impedance,
            wavenumber=wavenumber,
            observation_material_index=observation_index,
        )

    basis = np.asarray(AXIS_BASES[medium.axis], dtype=real_dtype)
    relative_positions = (np.asarray(currents.positions) - origin_values) @ basis.T
    local_directions = direction_values @ basis.T
    local_j = np.einsum("fpi,ji->fpj", currents.electric_current, basis)
    local_m = np.einsum("fpi,ji->fpj", currents.magnetic_current, basis)
    interfaces = (
        np.asarray(medium.interfaces, dtype=real_dtype) - origin_values["xyz".index(medium.axis)]
    )
    layer_index = _local_layer_indices(relative_positions[:, 2], interfaces)
    thickness = _layer_thicknesses(interfaces, len(medium.material_ids))

    if _evaluate_layered_cython is not None:
        (
            electric_local,
            magnetic_local,
            observation_impedance,
            observation_wavenumber,
            observation_index,
        ) = _cython_layered_currents(
            currents,
            frequency_values,
            local_directions,
            relative_positions,
            local_j,
            local_m,
            interfaces,
            layer_index,
            thickness,
            medium,
            grazing_tolerance,
            nthreads,
        )
        electric = np.einsum("fdi,ij->fdj", electric_local, basis)
        magnetic = np.einsum("fdi,ij->fdj", magnetic_local, basis)
        for array in (electric, magnetic, observation_impedance, observation_wavenumber):
            array.setflags(write=False)
        observation_index.setflags(write=False)
        return LayeredFarField(
            electric=electric,
            magnetic=magnetic,
            impedance=observation_impedance,
            wavenumber=observation_wavenumber,
            observation_material_index=observation_index,
        )

    nf = frequency_values.size
    nd = direction_values.shape[0]
    electric_local = np.zeros((nf, nd, 3), dtype=currents.electric_current.dtype)
    magnetic_local = np.zeros_like(electric_local)
    observation_impedance = np.empty((nf, nd), dtype=real_dtype)
    observation_wavenumber = np.empty((nf, nd), dtype=real_dtype)
    observation_index = np.where(local_directions[:, 2] > 0, 0, -1).astype(np.int8)

    for direction_number, direction in enumerate(local_directions):
        cos_theta = float(direction[2])
        if abs(cos_theta) <= grazing_tolerance:
            raise ValueError(
                "layered NTFF is singular at exact grazing incidence; omit theta=90 degrees"
            )
        upper_observation = cos_theta > 0
        exterior = 0 if upper_observation else -1
        sin_theta = float(np.hypot(direction[0], direction[1]))
        if sin_theta > grazing_tolerance:
            cos_phi = float(direction[0] / sin_theta)
            sin_phi = float(direction[1] / sin_theta)
        else:
            cos_phi, sin_phi = 1.0, 0.0
        theta_hat = np.asarray(
            (cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta),
            dtype=real_dtype,
        )
        phi_hat = np.asarray((-sin_phi, cos_phi, 0.0), dtype=real_dtype)
        dyadic_sign = 1.0 if upper_observation else -1.0

        for frequency_number, frequency in enumerate(frequency_values):
            eps_absolute = np.asarray(medium.relative_permittivity[frequency_number])
            mu_absolute = np.asarray(medium.relative_permeability[frequency_number])
            eps_observation = float(np.real(eps_absolute[exterior]))
            mu_observation = float(np.real(mu_absolute[exterior]))
            angular_frequency = 2 * np.pi * float(frequency)
            k_observation = angular_frequency * np.sqrt(
                epsilon_0 * eps_observation * mu_0 * mu_observation
            )
            eta_observation = np.sqrt(mu_0 * mu_observation / (epsilon_0 * eps_observation))
            observation_impedance[frequency_number, direction_number] = eta_observation
            observation_wavenumber[frequency_number, direction_number] = k_observation

            eps = eps_absolute / eps_observation
            mu = mu_absolute / mu_observation
            q = _outgoing_sqrt(eps * mu - sin_theta**2)
            beta = k_observation * q
            eta_e = q / eps  # TM (electric) normalized line impedance
            eta_h = mu / q  # TE (magnetic) normalized line impedance
            plus_e, minus_e = _voltage_coefficients(
                beta, eta_e, thickness, upper_observation=upper_observation
            )
            plus_h, minus_h = _voltage_coefficients(
                beta, eta_h, thickness, upper_observation=upper_observation
            )
            # The recursions reference their exterior travelling wave to the
            # nearest physical interface. Restore the absolute phase relative
            # to the user-selected NTFF origin. This is essential: inserting
            # a fictitious interface between identical materials must not
            # change the far field.
            if interfaces.size:
                if upper_observation:
                    exterior_phase = np.exp(1j * beta[0] * interfaces[0])
                else:
                    exterior_phase = np.exp(-1j * beta[-1] * interfaces[-1])
                plus_e *= exterior_phase
                minus_e *= exterior_phase
                plus_h *= exterior_phase
                minus_h *= exterior_phase
            vi_e, vv_e = _responses_at_positions(
                relative_positions[:, 2],
                layer_index,
                interfaces,
                beta,
                eta_e,
                plus_e,
                minus_e,
            )
            vi_h, vv_h = _responses_at_positions(
                relative_positions[:, 2],
                layer_index,
                interfaces,
                beta,
                eta_h,
                plus_h,
                minus_h,
            )
            lateral_phase = np.exp(
                1j
                * k_observation
                * (
                    direction[0] * relative_positions[:, 0]
                    + direction[1] * relative_positions[:, 1]
                )
            )
            weight = np.asarray(currents.area_weights) * lateral_phase
            j = local_j[frequency_number]
            m = local_m[frequency_number]
            eps_patch = eps[layer_index]
            mu_patch = mu[layer_index]

            j_radial = cos_phi * j[:, 0] + sin_phi * j[:, 1]
            j_phi = -sin_phi * j[:, 0] + cos_phi * j[:, 1]
            m_radial = cos_phi * m[:, 0] + sin_phi * m[:, 1]
            m_phi = -sin_phi * m[:, 0] + cos_phi * m[:, 1]

            aj_theta = dyadic_sign * np.sum(
                weight * (vi_e * j_radial - vv_e * (sin_theta / eps_patch) * j[:, 2])
            )
            aj_phi = dyadic_sign * np.sum(weight * cos_theta * vi_h * j_phi)
            fm_theta = dyadic_sign * np.sum(
                weight * cos_theta * (vv_h * m_radial - vi_h * (sin_theta / mu_patch) * m[:, 2])
            )
            fm_phi = dyadic_sign * np.sum(weight * vv_e * m_phi)

            factor = -1j * angular_frequency / (4 * np.pi)
            e_theta = factor * (
                mu_0 * mu_observation * aj_theta
                + eta_observation * epsilon_0 * eps_observation * fm_phi
            )
            e_phi = factor * (
                mu_0 * mu_observation * aj_phi
                - eta_observation * epsilon_0 * eps_observation * fm_theta
            )
            electric_local[frequency_number, direction_number] = (
                e_theta * theta_hat + e_phi * phi_hat
            )
            magnetic_local[frequency_number, direction_number] = (
                np.cross(direction, electric_local[frequency_number, direction_number])
                / eta_observation
            )

    electric = np.einsum("fdi,ij->fdj", electric_local, basis)
    magnetic = np.einsum("fdi,ij->fdj", magnetic_local, basis)
    for array in (electric, magnetic, observation_impedance, observation_wavenumber):
        array.setflags(write=False)
    observation_index.setflags(write=False)
    return LayeredFarField(
        electric=electric,
        magnetic=magnetic,
        impedance=observation_impedance,
        wavenumber=observation_wavenumber,
        observation_material_index=observation_index,
    )


def evaluate_layered_equivalent_current_far_zone(
    surface_data,
    frequencies: npt.ArrayLike,
    directions: npt.ArrayLike,
    medium: LayeredMedium,
    *,
    origin: npt.ArrayLike,
    nthreads: int = 1,
) -> LayeredFarField:
    """Collocate Love currents and evaluate the planar-layered far field."""

    return evaluate_layered_currents(
        collocate_love_currents(surface_data),
        frequencies,
        directions,
        medium,
        origin=origin,
        nthreads=nthreads,
    )


def material_constitutive_arrays(
    materials: Sequence[object], frequencies: npt.ArrayLike
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    """Resolve gprMax material objects into complex relative properties."""

    frequency_values = np.asarray(frequencies, dtype=float)
    eps = np.empty((frequency_values.size, len(materials)), dtype=np.complex128)
    mu = np.empty_like(eps)
    for frequency_number, frequency in enumerate(frequency_values):
        if frequency <= 0:
            raise ValueError("layered material properties require positive frequencies")
        angular_frequency = 2 * np.pi * frequency
        for material_number, material in enumerate(materials):
            relative_permittivity = material.calculate_er(float(frequency))
            # ``Material.calculate_er`` predates frequency-domain conductive
            # outputs and returns only its real dielectric constant. The
            # dispersive subclass already includes conductivity, so add it
            # only for the non-dispersive base representation.
            if not hasattr(material, "inclusive_w"):
                relative_permittivity += material.se / (1j * angular_frequency * epsilon_0)
            eps[frequency_number, material_number] = relative_permittivity
            mu[frequency_number, material_number] = material.mr + material.sm / (
                1j * angular_frequency * mu_0
            )
    return eps, mu
