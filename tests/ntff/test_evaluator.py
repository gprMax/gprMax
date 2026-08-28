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

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import c

from gprMax.ntff.evaluator import (
    evaluate_exact_points_patches,
    evaluate_far_zone,
    evaluate_far_zone_patches,
    project_cartesian_to_spherical,
    spherical_directions,
    spherical_observation_points,
)
from gprMax.ntff.surfaces import build_component_surface


def _outgoing_point_source_phasors(surface, frequency, source_position):
    positions = surface.patch_positions
    normals = surface.normals
    displacement = positions - source_position
    radius = np.linalg.norm(displacement, axis=1)
    radial_direction = displacement / radius[:, np.newaxis]
    wavenumber = 2 * np.pi * frequency / c
    field = np.exp(-1j * wavenumber * radius) / (4 * np.pi * radius)
    radial_derivative = (-1j * wavenumber - 1 / radius) * field
    normal_derivative = radial_derivative * np.sum(normals * radial_direction, axis=1)
    return field[np.newaxis, :], normal_derivative[np.newaxis, :]


def _point_source_error(spacing, lower, upper, directions, frequency):
    shape = tuple(value + 6 for value in upper)
    surface = build_component_surface("Ex", lower, upper, (spacing,) * 3, shape)
    source_position = 0.5 * (np.asarray(lower) + np.asarray(upper)) * spacing
    field, derivative = _outgoing_point_source_phasors(
        surface, frequency, source_position
    )
    actual = evaluate_far_zone(surface, [frequency], directions, field, derivative)[0]
    wavenumber = 2 * np.pi * frequency / c
    expected = np.exp(1j * wavenumber * (directions @ source_position)) / (4 * np.pi)
    return np.max(np.abs(actual - expected)), actual, expected


def test_spherical_directions_uses_theta_from_z_and_phi_from_x():
    directions = spherical_directions(
        [0, 90, 90, 90, 180], [0, 0, 90, 180, 0], degrees=True
    )

    assert_allclose(
        directions,
        np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, -1],
            ]
        ),
        atol=1e-15,
    )


def test_spherical_observation_points_broadcasts_to_cartesian_point_array():
    points = spherical_observation_points(
        (1.0, 2.0, 3.0),
        2.0,
        np.asarray([0.0, 90.0])[:, np.newaxis],
        np.asarray([0.0, 90.0])[np.newaxis, :],
        degrees=True,
    )

    assert points.shape == (4, 3)
    assert_allclose(
        points,
        [
            [1.0, 2.0, 5.0],
            [1.0, 2.0, 5.0],
            [3.0, 2.0, 3.0],
            [1.0, 4.0, 3.0],
        ],
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "origin,radius",
    [((0, 0), 1), ((0, 0, 0), 0), ((0, 0, 0), np.inf)],
)
def test_spherical_observation_points_rejects_invalid_geometry(origin, radius):
    with pytest.raises(ValueError):
        spherical_observation_points(origin, radius, 0, 0)


def test_outgoing_point_source_far_zone_converges_under_surface_refinement():
    directions = spherical_directions(
        [0, 30, 60, 90, 120, 150, 180],
        [0, 17, 83, 141, 211, 289, 0],
        degrees=True,
    )
    frequency = 300e6

    coarse_error, _, _ = _point_source_error(
        0.05, (4, 4, 4), (16, 16, 16), directions, frequency
    )
    fine_error, actual, expected = _point_source_error(
        0.025, (8, 8, 8), (32, 32, 32), directions, frequency
    )

    assert fine_error < 0.35 * coarse_error
    assert_allclose(actual, expected, rtol=3e-3, atol=3e-4)


def test_exact_point_evaluator_reproduces_outgoing_green_function():
    spacing = 0.025
    lower = (8, 8, 8)
    upper = (32, 32, 32)
    surface = build_component_surface(
        "Ex", lower, upper, (spacing,) * 3, (38, 38, 38)
    )
    source_position = 0.5 * (np.asarray(lower) + np.asarray(upper)) * spacing
    frequency = 300e6
    field, derivative = _outgoing_point_source_phasors(
        surface, frequency, source_position
    )
    points = np.asarray(((1.1, 0.5, 0.5), (0.5, 1.2, 0.65)))

    actual = evaluate_exact_points_patches(
        surface.patch_positions,
        surface.normals,
        surface.area_weights,
        [frequency],
        points,
        field,
        derivative,
        point_block_size=1,
        patch_block_size=79,
    )[0]
    radius = np.linalg.norm(points - source_position, axis=1)
    wavenumber = 2 * np.pi * frequency / c
    expected = np.exp(-1j * wavenumber * radius) / (4 * np.pi * radius)

    assert_allclose(actual, expected, rtol=5e-3, atol=5e-4)


def test_evaluator_supports_multiple_frequencies_and_directions():
    surface = build_component_surface(
        "Ez", (2, 2, 2), (5, 5, 5), (0.1, 0.1, 0.1), (9, 9, 9)
    )
    frequencies = np.array((0.0, 1e8))
    directions = spherical_directions([0, 90], [0, 0], degrees=True)
    field = np.zeros((2, surface.npatches), dtype=np.complex128)
    derivative = np.zeros_like(field)

    result = evaluate_far_zone(surface, frequencies, directions, field, derivative)

    assert result.shape == (2, 2)
    assert_allclose(result, 0.0)


def test_explicit_patch_evaluator_matches_surface_wrapper_and_precision():
    surface = build_component_surface(
        "Ey",
        (2, 2, 2),
        (5, 5, 5),
        (0.1, 0.1, 0.1),
        (9, 9, 9),
        real_dtype=np.float32,
    )
    frequencies = np.asarray([1e8, 2e8], dtype=np.float32)
    directions = spherical_directions(
        np.asarray([25, 90], dtype=np.float32),
        np.asarray([15, 120], dtype=np.float32),
        degrees=True,
    )
    patch_index = np.arange(surface.npatches, dtype=np.float32)
    field = np.asarray(
        (1 + 0.01 * patch_index)[np.newaxis, :] * np.asarray([[1], [2j]]),
        dtype=np.complex64,
    )
    derivative = np.asarray((0.2 - 0.3j) * field, dtype=np.complex64)

    wrapped = evaluate_far_zone(
        surface, frequencies, directions, field, derivative, patch_block_size=17
    )
    explicit = evaluate_far_zone_patches(
        surface.patch_positions,
        surface.normals,
        surface.area_weights,
        frequencies,
        directions,
        field,
        derivative,
        patch_block_size=17,
    )

    assert wrapped.dtype == np.complex64
    assert explicit.dtype == np.complex64
    assert_allclose(explicit, wrapped, rtol=2e-6, atol=2e-6)


def test_blocked_evaluator_and_reference_origin_preserve_expected_phase():
    spacing = 0.025
    lower = (8, 8, 8)
    upper = (32, 32, 32)
    frequency = 300e6
    directions = spherical_directions(
        [25, 70, 115, 160], [10, 95, 205, 315], degrees=True
    )
    surface = build_component_surface(
        "Ex", lower, upper, (spacing,) * 3, (38, 38, 38)
    )
    source_position = 0.5 * (np.asarray(lower) + np.asarray(upper)) * spacing
    field, derivative = _outgoing_point_source_phasors(
        surface, frequency, source_position
    )

    actual = evaluate_far_zone(
        surface,
        [frequency],
        directions,
        field,
        derivative,
        origin=source_position,
        direction_block_size=2,
        patch_block_size=37,
    )[0]

    assert_allclose(actual, 1 / (4 * np.pi), rtol=4e-3, atol=4e-4)


def test_cartesian_projection_uses_radial_polar_azimuthal_order():
    theta = np.array([0.0, 90.0, 90.0])
    phi = np.array([0.0, 0.0, 90.0])
    cartesian = np.array(
        [[[0, 0, 2], [0, 0, -3], [-4, 0, 0]]], dtype=np.complex128
    )

    spherical = project_cartesian_to_spherical(
        cartesian, theta, phi, degrees=True
    )

    assert_allclose(spherical[0, 0], [2, 0, 0], atol=1e-15)
    assert_allclose(spherical[0, 1], [0, 3, 0], atol=1e-15)
    assert_allclose(spherical[0, 2], [0, 0, 4], atol=1e-15)


@pytest.mark.parametrize(
    "frequencies,directions,field_shape,derivative_shape,match",
    [
        ([-1.0], [[1, 0, 0]], (1, None), (1, None), "non-negative"),
        ([1.0], [[2, 0, 0]], (1, None), (1, None), "unit vectors"),
        ([1.0], [[1, 0]], (1, None), (1, None), "shape"),
        ([1.0], [[1, 0, 0]], (2, None), (1, None), "surface_field"),
        ([1.0], [[1, 0, 0]], (1, None), (2, None), "normal_derivative"),
    ],
)
def test_evaluator_rejects_invalid_inputs(
    frequencies, directions, field_shape, derivative_shape, match
):
    surface = build_component_surface(
        "Ex", (2, 2, 2), (4, 4, 4), (0.1, 0.1, 0.1), (8, 8, 8)
    )
    field = np.zeros(
        tuple(surface.npatches if value is None else value for value in field_shape),
        dtype=np.complex128,
    )
    derivative = np.zeros(
        tuple(surface.npatches if value is None else value for value in derivative_shape),
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match=match):
        evaluate_far_zone(surface, frequencies, directions, field, derivative)
