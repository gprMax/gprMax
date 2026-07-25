"""Checks for the independent analytical PEC-sphere Mie reference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import c

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
    backscatter_efficiency = np.abs(
        np.sum((2 * orders + 1) * (-1) ** orders * (electric - magnetic))
    ) ** 2 / size_parameter**2

    actual = pec_sphere_bistatic_rcs(
        frequency, radius, [np.pi], polarisation="perpendicular"
    )[0]

    assert_allclose(
        actual / (np.pi * radius**2), backscatter_efficiency, rtol=2e-14
    )


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
