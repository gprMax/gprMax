import numpy as np

from testing.validation.planar_layered_ntff import validate_grounded_dipoles as dipoles
from testing.validation.planar_layered_ntff import validate_grounded_slab_reflection as slab


def _case(name):
    return next(case for case in dipoles.CASES if case.name == name)


def test_bare_pec_reflection_coefficients_are_minus_one():
    theta = np.deg2rad((0, 20, 45, 80))
    tm, te = dipoles._grounded_reflection(theta, 2e9, coated=False)
    np.testing.assert_array_equal(tm, -np.ones(theta.size))
    np.testing.assert_array_equal(te, -np.ones(theta.size))


def test_bare_pec_dipole_oracle_recovers_image_theory_power():
    theta = np.deg2rad(np.asarray((10, 30, 55, 80), dtype=float))
    phi = np.zeros_like(theta)
    frequency = 2e9
    wavenumber = 2 * np.pi * frequency / dipoles.c

    electric_normal = _case("electric_normal_bare")
    etheta, ephi = dipoles.analytical_fields(electric_normal, theta, phi, frequency)
    height = dipoles._physical_source_position(electric_normal)[2] - dipoles.GROUND
    expected = 4 * np.sin(theta) ** 2 * np.cos(wavenumber * height * np.cos(theta)) ** 2
    np.testing.assert_allclose(np.abs(etheta) ** 2, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(ephi, 0, atol=2e-15)

    magnetic_tangential = _case("magnetic_tangential_bare")
    etheta, ephi = dipoles.analytical_fields(magnetic_tangential, theta, phi, frequency)
    height = dipoles._physical_source_position(magnetic_tangential)[2] - dipoles.GROUND
    expected = 4 * np.cos(theta) ** 2 * np.cos(wavenumber * height * np.cos(theta)) ** 2
    np.testing.assert_allclose(np.abs(ephi) ** 2, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(etheta, 0, atol=2e-15)


def test_lossless_grounded_slab_reflection_has_unit_magnitude():
    frequencies = np.linspace(slab.FREQUENCY_MIN, slab.FREQUENCY_MAX, 101)
    reflection = slab._analytical_reflection(frequencies)
    np.testing.assert_allclose(np.abs(reflection), 1, rtol=2e-15, atol=2e-15)
