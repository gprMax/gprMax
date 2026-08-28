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

"""Closed-form tests for antenna-pattern quadrature and directivity."""

import numpy as np
from numpy.testing import assert_allclose

from gprMax.ntff.antenna import directivity_from_intensity, spherical_quadrature
from gprMax.ntff.interface import (
    ExteriorRadiationMetrics,
    KSIRRadiationMetrics,
    _refine_radiation_maximum,
)


def test_spherical_quadrature_integrates_isotropic_pattern():
    quadrature = spherical_quadrature(0.1, 20.0, np.float64)
    intensity = np.ones((2, quadrature.weights.size))
    radiated_power = np.sum(
        intensity * quadrature.weights[np.newaxis, :],
        axis=1,
    )
    directivity, directivity_dbi = directivity_from_intensity(
        intensity,
        radiated_power,
    )

    assert_allclose(np.sum(quadrature.weights), 4 * np.pi, rtol=2e-14)
    assert_allclose(radiated_power, 4 * np.pi, rtol=2e-14)
    assert_allclose(directivity, 1.0, rtol=2e-14)
    assert_allclose(directivity_dbi, 0.0, atol=1e-13)
    assert quadrature.theta_order % 2 == 1
    assert np.any(np.isclose(quadrature.theta, 90.0))


def test_hertzian_dipole_pattern_has_one_point_five_directivity():
    quadrature = spherical_quadrature(0.01, 1.0, np.float64)
    theta = np.deg2rad(quadrature.theta)
    intensity = np.sin(theta)[np.newaxis, :] ** 2
    radiated_power = np.sum(
        intensity * quadrature.weights[np.newaxis, :],
        axis=1,
    )
    maximum_intensity = np.asarray([[1.0]])
    maximum, maximum_dbi = directivity_from_intensity(
        maximum_intensity,
        radiated_power,
    )

    assert_allclose(radiated_power, 8 * np.pi / 3, rtol=2e-14)
    assert_allclose(maximum[0, 0], 1.5, rtol=2e-14)
    assert_allclose(maximum_dbi[0, 0], 10 * np.log10(1.5), rtol=2e-14)


def test_directivity_marks_zero_power_invalid_and_true_null_as_minus_infinity():
    intensity = np.asarray([[0.0, 1.0], [1.0, 2.0]])
    directivity, directivity_dbi = directivity_from_intensity(
        intensity,
        np.asarray([4 * np.pi, 0.0]),
    )

    assert directivity[0, 0] == 0
    assert np.isneginf(directivity_dbi[0, 0])
    assert np.isnan(directivity[1]).all()
    assert np.isnan(directivity_dbi[1]).all()


def test_requested_directions_refine_but_never_reduce_stored_maximum():
    metrics = KSIRRadiationMetrics(
        radiated_power=np.asarray((2 * np.pi, 2 * np.pi)),
        maximum_directivity=np.asarray((1.5, 1.5)),
        maximum_directivity_dbi=10 * np.log10(np.asarray((1.5, 1.5))),
        maximum_theta=np.asarray((90.0, 90.0)),
        maximum_phi=np.asarray((0.0, 0.0)),
        theta_order=13,
        phi_order=25,
        enclosure_radius=0.1,
    )
    intensity = np.asarray(((0.5, 1.0), (0.5, 0.7)))
    refined = _refine_radiation_maximum(
        metrics,
        intensity,
        np.asarray((82.0, 87.0)),
        np.asarray((20.0, 30.0)),
    )

    assert_allclose(refined.maximum_directivity, (2.0, 1.5))
    assert_allclose(refined.maximum_directivity_dbi, 10 * np.log10((2.0, 1.5)))
    assert_allclose(refined.maximum_theta, (87.0, 90.0))
    assert_allclose(refined.maximum_phi, (30.0, 0.0))


def test_requested_directions_refine_each_layered_exterior_independently():
    exterior = ExteriorRadiationMetrics(
        axis="z",
        material_ids=("air", "soil"),
        relative_permittivity=np.ones((2, 1), dtype=np.complex128),
        relative_permeability=np.ones((2, 1), dtype=np.complex128),
        impedance=np.ones((2, 1)),
        radiated_power=np.asarray(((2 * np.pi,), (2 * np.pi,))),
        radiated_fraction=np.asarray(((0.5,), (0.5,))),
        maximum_intensity=np.asarray(((0.5,), (0.5,))),
        maximum_directivity=np.asarray(((0.5,), (0.5,))),
        maximum_directivity_dbi=10 * np.log10(np.asarray(((0.5,), (0.5,)))),
        maximum_theta=np.asarray(((45.0,), (135.0,))),
        maximum_phi=np.asarray(((0.0,), (0.0,))),
    )
    metrics = KSIRRadiationMetrics(
        radiated_power=np.asarray((4 * np.pi,)),
        maximum_directivity=np.asarray((0.5,)),
        maximum_directivity_dbi=10 * np.log10(np.asarray((0.5,))),
        maximum_theta=np.asarray((45.0,)),
        maximum_phi=np.asarray((0.0,)),
        theta_order=12,
        phi_order=24,
        enclosure_radius=0.1,
        exterior=exterior,
    )

    refined = _refine_radiation_maximum(
        metrics,
        np.asarray(((1.0, 2.0),)),
        np.asarray((30.0, 150.0)),
        np.asarray((10.0, 20.0)),
    )

    assert_allclose(refined.exterior.maximum_intensity[:, 0], (1.0, 2.0))
    assert_allclose(refined.exterior.maximum_directivity[:, 0], (1.0, 2.0))
    assert_allclose(refined.exterior.maximum_theta[:, 0], (30.0, 150.0))
    assert_allclose(refined.exterior.maximum_phi[:, 0], (10.0, 20.0))
