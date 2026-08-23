import numpy as np
from scipy.constants import epsilon_0

from testing.validation.dispersive_averaging.coated_sphere_mie import coated_sphere_coefficients
from testing.validation.dispersive_averaging.layered_media import (
    PlanarMedium,
    normal_incidence_reflection,
    single_interface_fresnel,
)
from testing.validation.dispersive_averaging.pole_models import (
    arithmetic_mix,
    debye_term,
    drude_term,
    lorentz_term,
    make_material,
)
from testing.validation.dispersive_averaging.reduction import (
    ReductionTemplate,
    fit_projected_model,
    fit_reduced_model,
)
from testing.validation.mie_dielectric import dielectric_mie_coefficients


def test_inclusive_terms_match_physical_formulas():
    frequencies = np.geomspace(10e6, 5e9, 80)
    angular_frequency = 2 * np.pi * frequencies

    debye = make_material("Debye", 3.0, (debye_term(4.0, 0.4e-9),))
    expected_debye = 3.0 + 4.0 / (1 + 1j * angular_frequency * 0.4e-9)
    np.testing.assert_allclose(debye.relative_permittivity(frequencies), expected_debye)

    resonance = 1.5e9
    damping = 0.07 * 2 * np.pi * resonance
    lorentz = make_material("Lorentz", 2.0, (lorentz_term(3.0, resonance, damping),))
    omega_0 = 2 * np.pi * resonance
    expected_lorentz = 2.0 + 3.0 * omega_0**2 / (
        omega_0**2 + 2j * angular_frequency * damping - angular_frequency**2
    )
    np.testing.assert_allclose(lorentz.relative_permittivity(frequencies), expected_lorentz)

    plasma = 2.5e9
    collision = 0.25 * 2 * np.pi * 1e9
    drude = make_material("Drude", 1.0, (drude_term(plasma, collision),))
    omega_p = 2 * np.pi * plasma
    expected_drude = 1.0 - omega_p**2 / (
        angular_frequency**2 - 1j * angular_frequency * collision
    )
    np.testing.assert_allclose(drude.relative_permittivity(frequencies), expected_drude)


def test_arithmetic_mix_is_exact_frequency_by_frequency():
    frequencies = np.geomspace(20e6, 4e9, 100)
    first = make_material("first", 2.0, (debye_term(3.0, 0.2e-9),))
    second = make_material(
        "second",
        5.0,
        (lorentz_term(2.0, 1.2e9, 0.1 * 2 * np.pi * 1.2e9),),
        conductivity=0.02,
    )
    mixed = arithmetic_mix((first, second), (0.25, 0.75))
    expected = 0.25 * first.relative_permittivity(
        frequencies
    ) + 0.75 * second.relative_permittivity(frequencies)
    np.testing.assert_allclose(mixed.relative_permittivity(frequencies), expected)


def test_planar_recursion_reduces_to_fresnel_interface():
    frequencies = np.geomspace(10e6, 3e9, 50)
    air = PlanarMedium(make_material("air", 1.0))
    soil = PlanarMedium(make_material("soil", 4.0, (debye_term(5.0, 0.5e-9),)))
    recursive = normal_incidence_reflection(frequencies, air, (), soil)
    direct = single_interface_fresnel(frequencies, air, soil)
    np.testing.assert_allclose(recursive, direct)


def test_planar_recursion_matches_closed_form_dielectric_slab():
    frequencies = np.geomspace(50e6, 4e9, 80)
    air = PlanarMedium(make_material("air", 1.0))
    dielectric = PlanarMedium(make_material("dielectric", 4.0), thickness=0.015)

    recursive = normal_incidence_reflection(frequencies, air, (dielectric,), air)
    interface = -1 / 3
    exit_interface = 1 / 3
    wavenumber = 2 * np.pi * frequencies / 299792458.0 * 2
    round_trip = np.exp(-2j * wavenumber * dielectric.thickness)
    closed_form = (interface + exit_interface * round_trip) / (
        1 + interface * exit_interface * round_trip
    )

    np.testing.assert_allclose(recursive, closed_form)


def test_refined_sphere_configuration_preserves_physical_geometry():
    from testing.validation.dispersive_averaging.validate_core_shell_fdtd import CONFIGURATIONS

    baseline = CONFIGURATIONS["baseline"]
    refined = CONFIGURATIONS["refined"]
    assert baseline.domain_cells * baseline.dl == refined.domain_cells * refined.dl
    assert baseline.core_radius == refined.core_radius
    assert baseline.outer_radius == refined.outer_radius
    assert baseline.pml_cells * baseline.dl == refined.pml_cells * refined.dl
    assert baseline.frequencies.size == 87
    assert refined.frequencies.size == 173


def test_coated_sphere_equal_material_limit_matches_homogeneous_mie():
    for size_parameter in (0.2, 1.0, 3.0, 8.0):
        coated_a, coated_b = coated_sphere_coefficients(
            size_parameter,
            0.6 * size_parameter,
            4.0 - 0.2j,
            4.0 - 0.2j,
        )
        reference_a, reference_b = dielectric_mie_coefficients(size_parameter, 4.0 - 0.2j)
        terms = min(len(coated_a), len(reference_a))
        np.testing.assert_allclose(coated_a[:terms], reference_a[:terms], atol=1e-13)
        np.testing.assert_allclose(coated_b[:terms], reference_b[:terms], atol=1e-13)


def test_related_four_pole_debye_mix_reduces_to_two_terms_over_band():
    frequencies = np.geomspace(10e6, 3e9, 160)
    first = make_material(
        "soil A",
        3.2,
        (debye_term(0.75, 2.71e-9), debye_term(0.30, 0.108e-9)),
        conductivity=0.397e-3,
    )
    second = make_material(
        "soil C",
        6.0,
        (debye_term(2.75, 3.98e-9), debye_term(0.75, 0.251e-9)),
        conductivity=2.0e-3,
    )
    exact = arithmetic_mix((first, second), (0.5, 0.5))
    reduced = fit_reduced_model(
        exact,
        frequencies,
        ReductionTemplate(debye=2),
        fixed_conductivity=exact.conductivity,
        restarts=3,
        max_evaluations=5000,
    )
    assert reduced.metrics["maximum_relative"] < 0.005
    assert reduced.model.inclusive_order == 2


def test_projected_debye_reduction_preserves_fixed_terms_and_positive_strengths():
    frequencies = np.geomspace(10e6, 3e9, 120)
    first = make_material(
        "soil A",
        3.2,
        (debye_term(0.75, 2.71e-9), debye_term(0.30, 0.108e-9)),
        conductivity=0.397e-3,
    )
    second = make_material(
        "soil C",
        6.0,
        (debye_term(2.75, 3.98e-9), debye_term(0.75, 0.251e-9)),
        conductivity=2.0e-3,
    )
    exact = arithmetic_mix((first, second), (0.5, 0.5))
    reduced = fit_projected_model(
        exact,
        frequencies,
        ReductionTemplate(debye=2),
        fixed_conductivity=exact.conductivity,
        seed=41,
        maximum_iterations=60,
        population_size=6,
    )

    assert reduced.method == "global_variable_projection"
    assert reduced.model.epsilon_inf == exact.epsilon_inf
    assert reduced.model.conductivity == exact.conductivity
    assert all(pole.w.real > 0 for pole in reduced.model.poles)
    assert all(pole.q.real < 0 for pole in reduced.model.poles)
    assert reduced.metrics["maximum_relative"] < 0.005


def test_projected_mixed_model_preserves_coupled_drude_conductivity():
    frequencies = np.geomspace(100e6, 8e9, 100)
    first = make_material(
        "metal A",
        3.0,
        (
            drude_term(2.8e9, 0.25 * 2 * np.pi * 1e9),
            lorentz_term(2.0, 4.0e9, 0.08 * 2 * np.pi * 4.0e9),
        ),
    )
    second = make_material(
        "metal B",
        3.3,
        (
            drude_term(3.0e9, 0.30 * 2 * np.pi * 1e9),
            lorentz_term(2.3, 4.2e9, 0.09 * 2 * np.pi * 4.2e9),
        ),
    )
    exact = arithmetic_mix((first, second), (0.5, 0.5))
    reduced = fit_projected_model(
        exact,
        frequencies,
        ReductionTemplate(lorentz=2, drude=2),
        maximum_iterations=5,
        population_size=5,
    )

    assert reduced.evaluations == 1
    assert reduced.model.epsilon_inf == exact.epsilon_inf
    np.testing.assert_allclose(reduced.model.conductivity, exact.conductivity, rtol=1e-13)
    assert reduced.metrics["maximum_relative"] < 1e-12
