# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
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
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Pure NumPy reference evaluator for frequency-domain KSIR data."""

import numpy as np
import numpy.typing as npt
from scipy.constants import c

from .surfaces import KSIRComponentSurface

try:
    from gprMax.cython.ntff import evaluate_far_zone_patches as _evaluate_far_zone_patches_cython
except ImportError:  # pragma: no cover - source-tree fallback before compilation
    _evaluate_far_zone_patches_cython = None


def _floating_input_dtype(*values) -> np.dtype:
    dtype = np.result_type(*(np.asarray(value).dtype for value in values))
    return dtype if dtype.kind == "f" else np.dtype(float)


def spherical_directions(
    theta: npt.ArrayLike, phi: npt.ArrayLike, *, degrees: bool = False
) -> npt.NDArray[np.floating]:
    """Convert paired spherical angles to Cartesian unit vectors.

    ``theta`` is the polar angle from +z and ``phi`` is the azimuth from +x
    toward +y. Inputs are broadcast together and the result has shape
    ``broadcast(theta, phi).shape + (3,)``.
    """

    dtype = _floating_input_dtype(theta, phi)
    theta_values, phi_values = np.broadcast_arrays(
        np.asarray(theta, dtype=dtype), np.asarray(phi, dtype=dtype)
    )
    if not np.all(np.isfinite(theta_values)) or not np.all(np.isfinite(phi_values)):
        raise ValueError("theta and phi must contain only finite values")
    if degrees:
        theta_values = np.deg2rad(theta_values)
        phi_values = np.deg2rad(phi_values)

    sin_theta = np.sin(theta_values)
    return np.stack(
        (
            sin_theta * np.cos(phi_values),
            sin_theta * np.sin(phi_values),
            np.cos(theta_values),
        ),
        axis=-1,
    )


def spherical_observation_points(
    origin: npt.ArrayLike,
    radius: npt.ArrayLike,
    theta: npt.ArrayLike,
    phi: npt.ArrayLike,
    *,
    degrees: bool = False,
) -> npt.NDArray[np.floating]:
    """Create Cartesian KSIR observation points from spherical coordinates.

    ``theta`` is measured from +z and ``phi`` from +x toward +y. Radius and
    angles use NumPy broadcasting; the broadcast result is flattened to the
    ``(npoints, 3)`` array accepted by the exact-point KSIR evaluator. For an
    angular product grid, pass (for example) ``theta[:, None]`` and
    ``phi[None, :]``.

    Args:
        origin: Cartesian sphere origin ``(x, y, z)`` in metres.
        radius: positive radius value or broadcastable array in metres.
        theta: polar angle value or broadcastable array.
        phi: azimuthal angle value or broadcastable array.
        degrees: interpret angles in degrees rather than radians.

    Returns:
        Contiguous Cartesian point array with shape ``(npoints, 3)``.
    """

    dtype = _floating_input_dtype(origin, radius, theta, phi)
    centre = np.asarray(origin, dtype=dtype)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError("origin must contain exactly three finite values")

    radii, theta_values, phi_values = np.broadcast_arrays(
        np.asarray(radius, dtype=dtype),
        np.asarray(theta, dtype=dtype),
        np.asarray(phi, dtype=dtype),
    )
    if radii.size == 0:
        raise ValueError("spherical observation coordinates must not be empty")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("radius must contain only finite, positive values")

    directions = spherical_directions(
        theta_values, phi_values, degrees=degrees
    )
    points = centre + radii[..., np.newaxis] * directions
    return np.ascontiguousarray(points.reshape(-1, 3), dtype=dtype)


def _phasors(
    name: str, values: npt.ArrayLike, nfrequencies: int, npatches: int
) -> npt.NDArray[np.complexfloating]:
    array = np.asarray(values)
    expected_shape = (nfrequencies, npatches)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if array.dtype.kind != "c":
        raise ValueError(f"{name} must use a complex dtype")
    return array


def evaluate_far_zone_patches(
    patch_positions: npt.ArrayLike,
    patch_normals: npt.ArrayLike,
    area_weights: npt.ArrayLike,
    frequencies: npt.ArrayLike,
    directions: npt.ArrayLike,
    surface_field: npt.ArrayLike,
    normal_derivative: npt.ArrayLike,
    *,
    wave_speed: float = c,
    origin: npt.ArrayLike = (0.0, 0.0, 0.0),
    direction_block_size: int = 256,
    patch_block_size: int = 8192,
    nthreads: int = 1,
) -> npt.NDArray[np.complexfloating]:
    """Evaluate KSIR from explicit patch geometry and phasors.

    A Cython/OpenMP implementation is used when the extension is available.
    The blocked NumPy implementation remains the source-tree fallback and
    executable reference. ``direction_block_size`` and ``patch_block_size``
    control only that fallback.
    """

    raw_field = np.asarray(surface_field)
    raw_derivative = np.asarray(normal_derivative)
    complex_dtype = np.result_type(raw_field.dtype, raw_derivative.dtype)
    if complex_dtype.kind != "c":
        raise ValueError("surface phasors must use a complex dtype")
    real_dtype = np.empty((), dtype=complex_dtype).real.dtype
    freqs = np.asarray(frequencies, dtype=real_dtype)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(freqs)) or np.any(freqs < 0):
        raise ValueError("frequencies must contain finite, non-negative values")
    if not np.isfinite(wave_speed) or wave_speed <= 0:
        raise ValueError("wave_speed must be finite and greater than zero")
    reference_origin = np.asarray(origin, dtype=real_dtype)
    if reference_origin.shape != (3,) or not np.all(np.isfinite(reference_origin)):
        raise ValueError("origin must contain exactly three finite values")
    if (
        not isinstance(direction_block_size, (int, np.integer))
        or direction_block_size <= 0
    ):
        raise ValueError("direction_block_size must be a positive integer")
    if (
        not isinstance(patch_block_size, (int, np.integer))
        or patch_block_size <= 0
    ):
        raise ValueError("patch_block_size must be a positive integer")
    if not isinstance(nthreads, (int, np.integer)) or nthreads <= 0:
        raise ValueError("nthreads must be a positive integer")

    direction_vectors = np.asarray(directions, dtype=real_dtype)
    if direction_vectors.ndim != 2 or direction_vectors.shape[1] != 3:
        raise ValueError("directions must have shape (ndirections, 3)")
    if direction_vectors.shape[0] == 0 or not np.all(np.isfinite(direction_vectors)):
        raise ValueError("directions must contain finite unit vectors")
    direction_norms = np.linalg.norm(direction_vectors, axis=1)
    unit_tolerance = max(1e-12, 64 * np.finfo(real_dtype).eps)
    if not np.allclose(
        direction_norms, 1.0, rtol=unit_tolerance, atol=unit_tolerance
    ):
        raise ValueError("directions must be unit vectors")

    positions = np.asarray(patch_positions, dtype=real_dtype)
    normals = np.asarray(patch_normals, dtype=real_dtype)
    areas = np.asarray(area_weights, dtype=real_dtype)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("patch_positions must have shape (npatches, 3)")
    if normals.shape != positions.shape:
        raise ValueError("patch_normals must match patch_positions")
    if areas.shape != (positions.shape[0],):
        raise ValueError("area_weights must have one value per patch")
    if (
        not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(normals))
        or not np.all(np.isfinite(areas))
        or np.any(areas <= 0)
    ):
        raise ValueError("patch geometry must be finite with positive areas")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1.0):
        raise ValueError("patch_normals must be unit vectors")

    npatches = positions.shape[0]
    field = _phasors("surface_field", raw_field, freqs.size, npatches).astype(
        complex_dtype, copy=False
    )
    derivative = _phasors(
        "normal_derivative", raw_derivative, freqs.size, npatches
    ).astype(complex_dtype, copy=False)
    positions = np.ascontiguousarray(positions - reference_origin, dtype=real_dtype)
    normals = np.ascontiguousarray(normals, dtype=real_dtype)
    areas = np.ascontiguousarray(areas, dtype=real_dtype)
    direction_vectors = np.ascontiguousarray(direction_vectors, dtype=real_dtype)
    field = np.ascontiguousarray(field, dtype=complex_dtype)
    derivative = np.ascontiguousarray(derivative, dtype=complex_dtype)
    wavenumbers = np.ascontiguousarray(2 * np.pi * freqs / wave_speed, dtype=real_dtype)
    result = np.zeros(
        (freqs.size, direction_vectors.shape[0]), dtype=complex_dtype
    )

    if _evaluate_far_zone_patches_cython is not None:
        _evaluate_far_zone_patches_cython(
            int(nthreads),
            positions,
            normals,
            areas,
            wavenumbers,
            direction_vectors,
            field,
            derivative,
            result,
        )
        return result

    for direction_start in range(0, direction_vectors.shape[0], direction_block_size):
        direction_stop = min(
            direction_start + direction_block_size, direction_vectors.shape[0]
        )
        direction_block = direction_vectors[direction_start:direction_stop]
        block_result = result[:, direction_start:direction_stop]
        for patch_start in range(0, npatches, patch_block_size):
            patch_stop = min(patch_start + patch_block_size, npatches)
            direction_dot_position = (
                direction_block @ positions[patch_start:patch_stop].T
            )
            normal_dot_direction = (
                direction_block @ normals[patch_start:patch_stop].T
            )
            phase = np.exp(
                1j
                * wavenumbers[:, np.newaxis, np.newaxis]
                * direction_dot_position[np.newaxis, :, :]
            ).astype(complex_dtype, copy=False)
            integrand = -derivative[:, np.newaxis, patch_start:patch_stop] + (
                1j
                * wavenumbers[:, np.newaxis, np.newaxis]
                * normal_dot_direction[np.newaxis, :, :]
                * field[:, np.newaxis, patch_start:patch_stop]
            )
            block_result += np.sum(
                integrand
                * phase
                * areas[np.newaxis, np.newaxis, patch_start:patch_stop],
                axis=2,
            )
    return result / np.asarray(4 * np.pi, dtype=real_dtype)


def evaluate_exact_points_patches(
    patch_positions: npt.ArrayLike,
    patch_normals: npt.ArrayLike,
    area_weights: npt.ArrayLike,
    frequencies: npt.ArrayLike,
    points: npt.ArrayLike,
    surface_field: npt.ArrayLike,
    normal_derivative: npt.ArrayLike,
    *,
    wave_speed: float = c,
    point_block_size: int = 64,
    patch_block_size: int = 4096,
) -> npt.NDArray[np.complexfloating]:
    """Evaluate the exact finite-distance scalar KSIR surface integral.

    The engineering phasor convention ``exp(+j omega t)`` is used, so the
    outgoing Green function is ``exp(-j k R) / (4 pi R)``. Unlike
    :func:`evaluate_far_zone_patches`, this function retains every ``1/R``
    and ``1/R**2`` term and therefore returns the physical field at each
    requested point rather than a range-normalized far-field amplitude.
    """

    raw_field = np.asarray(surface_field)
    raw_derivative = np.asarray(normal_derivative)
    complex_dtype = np.result_type(raw_field.dtype, raw_derivative.dtype)
    if complex_dtype.kind != "c":
        raise ValueError("surface phasors must use a complex dtype")
    real_dtype = np.empty((), dtype=complex_dtype).real.dtype

    freqs = np.asarray(frequencies, dtype=real_dtype)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(freqs)) or np.any(freqs < 0):
        raise ValueError("frequencies must contain finite, non-negative values")
    if not np.isfinite(wave_speed) or wave_speed <= 0:
        raise ValueError("wave_speed must be finite and greater than zero")
    if not isinstance(point_block_size, (int, np.integer)) or point_block_size <= 0:
        raise ValueError("point_block_size must be a positive integer")
    if not isinstance(patch_block_size, (int, np.integer)) or patch_block_size <= 0:
        raise ValueError("patch_block_size must be a positive integer")

    observation_points = np.asarray(points, dtype=real_dtype)
    if observation_points.ndim == 1:
        observation_points = observation_points[np.newaxis, :]
    if (
        observation_points.ndim != 2
        or observation_points.shape[0] == 0
        or observation_points.shape[1] != 3
        or not np.all(np.isfinite(observation_points))
    ):
        raise ValueError("points must have shape (npoints, 3) and be finite")

    positions = np.asarray(patch_positions, dtype=real_dtype)
    normals = np.asarray(patch_normals, dtype=real_dtype)
    areas = np.asarray(area_weights, dtype=real_dtype)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("patch_positions must have shape (npatches, 3)")
    if normals.shape != positions.shape:
        raise ValueError("patch_normals must match patch_positions")
    if areas.shape != (positions.shape[0],):
        raise ValueError("area_weights must have one value per patch")
    if (
        not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(normals))
        or not np.all(np.isfinite(areas))
        or np.any(areas <= 0)
    ):
        raise ValueError("patch geometry must be finite with positive areas")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1.0):
        raise ValueError("patch_normals must be unit vectors")

    npatches = positions.shape[0]
    field = _phasors("surface_field", raw_field, freqs.size, npatches).astype(
        complex_dtype, copy=False
    )
    derivative = _phasors(
        "normal_derivative", raw_derivative, freqs.size, npatches
    ).astype(complex_dtype, copy=False)
    wavenumbers = np.asarray(2 * np.pi * freqs / wave_speed, dtype=real_dtype)
    result = np.zeros(
        (freqs.size, observation_points.shape[0]), dtype=complex_dtype
    )
    four_pi = np.asarray(4 * np.pi, dtype=real_dtype)

    for point_start in range(0, observation_points.shape[0], point_block_size):
        point_stop = min(
            point_start + point_block_size, observation_points.shape[0]
        )
        point_block = observation_points[point_start:point_stop]
        block_result = result[:, point_start:point_stop]
        for patch_start in range(0, npatches, patch_block_size):
            patch_stop = min(patch_start + patch_block_size, npatches)
            patch_positions_block = positions[patch_start:patch_stop]
            displacement = (
                point_block[:, np.newaxis, :] - patch_positions_block[np.newaxis, :, :]
            )
            radius = np.linalg.norm(displacement, axis=2)
            if np.any(radius == 0):
                raise ValueError("observation points must not coincide with surface patches")
            radial_direction = displacement / radius[:, :, np.newaxis]
            normal_dot_radial = np.sum(
                normals[np.newaxis, patch_start:patch_stop, :] * radial_direction,
                axis=2,
            )
            phase = np.exp(
                -1j
                * wavenumbers[:, np.newaxis, np.newaxis]
                * radius[np.newaxis, :, :]
            ).astype(complex_dtype, copy=False)
            inverse_radius = 1 / radius
            field_factor = normal_dot_radial[np.newaxis, :, :] * (
                inverse_radius[np.newaxis, :, :] ** 2
                + 1j
                * wavenumbers[:, np.newaxis, np.newaxis]
                * inverse_radius[np.newaxis, :, :]
            )
            integrand = (
                -derivative[:, np.newaxis, patch_start:patch_stop]
                * inverse_radius[np.newaxis, :, :]
                + field_factor * field[:, np.newaxis, patch_start:patch_stop]
            )
            block_result += np.sum(
                phase
                * integrand
                * areas[np.newaxis, np.newaxis, patch_start:patch_stop],
                axis=2,
            ) / four_pi
    return result


def evaluate_far_zone(
    surface: KSIRComponentSurface,
    frequencies: npt.ArrayLike,
    directions: npt.ArrayLike,
    surface_field: npt.ArrayLike,
    normal_derivative: npt.ArrayLike,
    *,
    wave_speed: float = c,
    origin: npt.ArrayLike = (0.0, 0.0, 0.0),
    direction_block_size: int = 256,
    patch_block_size: int = 8192,
    nthreads: int = 1,
) -> npt.NDArray[np.complexfloating]:
    """Evaluate the range-normalized scalar far-zone KSIR integral.

    The implementation follows the engineering convention
    ``exp(+j omega t)`` with outgoing radial dependence ``exp(-j k r)``.
    Therefore the returned value is ``r exp(+j k r) psi_far``.

    Args:
        surface: Closed Yee-aligned surface for one field component.
        frequencies: Non-negative frequencies in Hz.
        directions: Cartesian unit observation vectors with shape ``(nd, 3)``.
        surface_field: Collocated component phasors, shape ``(nf, npatches)``.
        normal_derivative: Outward-normal derivative phasors with the same shape.
        wave_speed: Homogeneous-background propagation speed in m/s.
        origin: Phase-reference origin in metres.
        direction_block_size: Maximum directions in one temporary block.
        patch_block_size: Maximum patches in one temporary block.
        nthreads: OpenMP threads used by the compiled evaluator.

    Returns:
        Range-normalized component phasors with shape ``(nf, nd)``.
    """

    return evaluate_far_zone_patches(
        surface.patch_positions,
        surface.normals,
        surface.area_weights,
        frequencies,
        directions,
        surface_field,
        normal_derivative,
        wave_speed=wave_speed,
        origin=origin,
        direction_block_size=direction_block_size,
        patch_block_size=patch_block_size,
        nthreads=nthreads,
    )


def spherical_basis(
    theta: npt.ArrayLike, phi: npt.ArrayLike, *, degrees: bool = False
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Return radial, polar, and azimuthal unit vectors for paired angles."""

    dtype = _floating_input_dtype(theta, phi)
    theta_values, phi_values = np.broadcast_arrays(
        np.asarray(theta, dtype=dtype), np.asarray(phi, dtype=dtype)
    )
    if not np.all(np.isfinite(theta_values)) or not np.all(np.isfinite(phi_values)):
        raise ValueError("theta and phi must contain only finite values")
    if degrees:
        theta_values = np.deg2rad(theta_values)
        phi_values = np.deg2rad(phi_values)

    sin_theta = np.sin(theta_values)
    cos_theta = np.cos(theta_values)
    sin_phi = np.sin(phi_values)
    cos_phi = np.cos(phi_values)
    radial = np.stack(
        (sin_theta * cos_phi, sin_theta * sin_phi, cos_theta), axis=-1
    )
    polar = np.stack(
        (cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta), axis=-1
    )
    azimuthal = np.stack((-sin_phi, cos_phi, np.zeros_like(phi_values)), axis=-1)
    return radial, polar, azimuthal


def project_cartesian_to_spherical(
    cartesian: npt.ArrayLike,
    theta: npt.ArrayLike,
    phi: npt.ArrayLike,
    *,
    degrees: bool = False,
) -> npt.NDArray[np.complexfloating]:
    """Project Cartesian vector phasors onto the spherical basis.

    The final two dimensions of the input must be direction and Cartesian
    component. The returned component order is radial, polar, azimuthal.
    """

    values = np.asarray(cartesian)
    if values.dtype.kind != "c":
        raise ValueError("cartesian must use a complex dtype")
    real_dtype = values.real.dtype
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("cartesian must have a final dimension of length three")
    radial, polar, azimuthal = spherical_basis(theta, phi, degrees=degrees)
    radial = radial.astype(real_dtype, copy=False)
    polar = polar.astype(real_dtype, copy=False)
    azimuthal = azimuthal.astype(real_dtype, copy=False)
    if radial.ndim != 2 or radial.shape != (values.shape[-2], 3):
        raise ValueError("theta and phi must identify one pair per direction")
    return np.stack(
        (
            np.sum(values * radial, axis=-1),
            np.sum(values * polar, axis=-1),
            np.sum(values * azimuthal, axis=-1),
        ),
        axis=-1,
    )
