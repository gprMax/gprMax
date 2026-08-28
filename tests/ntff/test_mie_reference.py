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

"""Checks for the independent analytical PEC-sphere Mie reference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import c

from testing.validation.mie_dielectric import (
    dielectric_mie_coefficients,
    dielectric_sphere_absorption_cross_section,
    dielectric_sphere_bistatic_rcs,
)
from testing.validation.mie_pec import (
    pec_mie_amplitudes,
    pec_mie_coefficients,
    pec_sphere_bistatic_rcs,
)


def test_forward_mie_amplitudes_are_polarisation_independent():
    perpendicular, parallel = pec_mie_amplitudes(1.3, [0.0])

    assert_allclose(perpendicular, parallel, rtol=1e-14)


def test_backscatter_rcs_matches_coefficient_identity():
    frequency = 5e9
    radius = 0.0075
    wavenumber = 2 * np.pi * frequency / c
    size_parameter = wavenumber * radius
    electric, magnetic = pec_mie_coefficients(size_parameter)
    orders = np.arange(1, electric.size + 1)
    backscatter_efficiency = (
        np.abs(np.sum((2 * orders + 1) * (-1) ** orders * (electric - magnetic))) ** 2
        / size_parameter**2
    )

    actual = pec_sphere_bistatic_rcs(frequency, radius, [np.pi], polarisation="perpendicular")[0]

    assert_allclose(actual / (np.pi * radius**2), backscatter_efficiency, rtol=2e-14)


def test_mie_rcs_is_finite_nonnegative_and_has_requested_shape():
    angles = np.linspace(0, np.pi, 181)
    result = pec_sphere_bistatic_rcs(5e9, 0.0075, angles)

    assert result.shape == angles.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    assert np.max(result) > 0


def test_small_pec_sphere_backscatter_approaches_rayleigh_limit():
    radius = 1e-3
    size_parameter = 0.01
    frequency = size_parameter * c / (2 * np.pi * radius)
    actual = pec_sphere_bistatic_rcs(frequency, radius, [np.pi])[0]
    wavenumber = size_parameter / radius
    expected = 9 * np.pi * wavenumber**4 * radius**6

    assert_allclose(actual, expected, rtol=2e-4)


def test_dielectric_sphere_disappears_when_permittivity_matches_free_space():
    result = dielectric_sphere_bistatic_rcs(3e9, 0.01, 1.0, [0.0, np.pi])

    assert_allclose(result, 0, atol=1e-30)


def test_dielectric_backscatter_rcs_matches_coefficient_identity():
    frequency = 3e9
    radius = 0.012
    relative_permittivity = 4.0
    size_parameter = 2 * np.pi * frequency * radius / c
    electric, magnetic = dielectric_mie_coefficients(size_parameter, relative_permittivity)
    orders = np.arange(1, electric.size + 1)
    backscatter_efficiency = (
        np.abs(np.sum((2 * orders + 1) * (-1) ** orders * (electric - magnetic))) ** 2
        / size_parameter**2
    )

    actual = dielectric_sphere_bistatic_rcs(
        frequency,
        radius,
        relative_permittivity,
        [np.pi],
    )[0]

    assert_allclose(actual / (np.pi * radius**2), backscatter_efficiency, rtol=2e-14)


def test_small_dielectric_sphere_backscatter_approaches_rayleigh_limit():
    radius = 1e-3
    relative_permittivity = 4.0
    size_parameter = 0.01
    frequency = size_parameter * c / (2 * np.pi * radius)
    actual = dielectric_sphere_bistatic_rcs(
        frequency,
        radius,
        relative_permittivity,
        [np.pi],
    )[0]
    wavenumber = size_parameter / radius
    contrast = (relative_permittivity - 1) / (relative_permittivity + 2)
    expected = 4 * np.pi * wavenumber**4 * radius**6 * abs(contrast) ** 2

    assert_allclose(actual, expected, rtol=2e-4)


def test_lossless_dielectric_sphere_has_zero_absorption():
    actual = dielectric_sphere_absorption_cross_section(1e9, 0.01, 4.0)

    assert actual == pytest.approx(0.0, abs=2e-18)


def test_passive_lossy_dielectric_sphere_has_positive_absorption():
    actual = dielectric_sphere_absorption_cross_section(1e9, 0.01, 4.0 - 0.5j)

    assert np.isfinite(actual)
    assert actual > 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"frequency": 0, "radius": 1, "scattering_angles": [0]}, "frequency"),
        ({"frequency": 1, "radius": 0, "scattering_angles": [0]}, "radius"),
        (
            {
                "frequency": 1,
                "radius": 1,
                "scattering_angles": [0],
                "polarisation": "unknown",
            },
            "polarisation",
        ),
    ],
)
def test_mie_reference_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pec_sphere_bistatic_rcs(**kwargs)


@pytest.mark.parametrize(
    "permittivity,match",
    [
        (0, "positive real part"),
        (4 + 0.1j, "non-positive imaginary part"),
        (np.nan, "finite"),
    ],
)
def test_dielectric_mie_reference_rejects_invalid_permittivity(permittivity, match):
    with pytest.raises(ValueError, match=match):
        dielectric_sphere_bistatic_rcs(1, 1, permittivity, [0])
