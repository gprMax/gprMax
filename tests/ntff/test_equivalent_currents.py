"""Analytical checks for the conventional equivalent-current NTFF."""

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import c, epsilon_0, mu_0

import gprMax.ntff.equivalent_currents as equivalent_currents
from gprMax.ntff.equivalent_currents import (
    collocate_love_currents,
    evaluate_equivalent_current_far_zone,
)
from gprMax.ntff.surfaces import COMPONENT_OFFSETS, COMPONENTS, build_component_surface


def _dipole_fields(points, frequency, source, current):
    omega = 2 * np.pi * frequency
    wavenumber = omega / c
    displacement = points - source
    radius = np.linalg.norm(displacement, axis=1)
    radial = displacement / radius[:, np.newaxis]
    dipole_moment = current / (1j * omega)
    projection = radial @ dipole_moment
    transverse = dipole_moment - radial * projection[:, np.newaxis]
    quasistatic = 3 * radial * projection[:, np.newaxis] - dipole_moment
    exponential = np.exp(-1j * wavenumber * radius)
    electric = (
        exponential[:, np.newaxis]
        / (4 * np.pi * epsilon_0)
        * (
            (wavenumber**2 / radius)[:, np.newaxis] * transverse
            + (1 / radius**3 + 1j * wavenumber / radius**2)[:, np.newaxis] * quasistatic
        )
    )
    green = exponential / (4 * np.pi * radius)
    magnetic = (
        (-1j * wavenumber - 1 / radius)[:, np.newaxis]
        * green[:, np.newaxis]
        * np.cross(radial, current)
    )
    return electric, magnetic


def _analytic_surface_data(lower, upper, spacing, field_shape, frequency, source, current):
    result = {}
    for component in COMPONENTS:
        surface = build_component_surface(
            component, lower, upper, spacing, field_shape, real_dtype=np.float64
        )
        axis = "xyz".index(component[1].lower())
        offset = np.asarray(COMPONENT_OFFSETS[component])
        field_parts = []
        derivative_parts = []
        for face in surface.faces:
            inside_position = (face.inside_indices + offset) * spacing
            outside_position = (face.outside_indices + offset) * spacing
            inside_fields = _dipole_fields(inside_position, frequency, source, current)
            outside_fields = _dipole_fields(outside_position, frequency, source, current)
            family = 0 if component.startswith("E") else 1
            inside = inside_fields[family][:, axis]
            outside = outside_fields[family][:, axis]
            field_parts.append(0.5 * (inside + outside))
            derivative_parts.append((outside - inside) / face.normal_spacing)
        result[component] = SimpleNamespace(
            surface=surface,
            field=np.concatenate(field_parts)[np.newaxis, :],
            normal_derivative=np.concatenate(derivative_parts)[np.newaxis, :],
        )
    return result


def test_arithmetic_collocation_forms_constant_love_currents():
    surfaces = {}
    lower = (2, 2, 2)
    upper = (6, 7, 5)
    spacing = np.asarray((0.02, 0.03, 0.04))
    shape = (10, 11, 9)
    electric = np.asarray((2 + 3j, -1 + 0.5j, 4 - 2j))
    magnetic = np.asarray((0.2 - 0.1j, 0.7 + 0.3j, -0.4 + 0.9j))
    for component in COMPONENTS:
        surface = build_component_surface(component, lower, upper, spacing, shape)
        value = electric if component.startswith("E") else magnetic
        axis = "xyz".index(component[1].lower())
        surfaces[component] = SimpleNamespace(
            surface=surface,
            field=np.full((1, surface.npatches), value[axis], dtype=np.complex128),
            normal_derivative=np.zeros((1, surface.npatches), dtype=np.complex128),
        )

    currents = collocate_love_currents(surfaces)

    assert_allclose(currents.electric_current[0], np.cross(currents.normals, magnetic))
    assert_allclose(currents.magnetic_current[0], -np.cross(currents.normals, electric))


def test_analytic_hertzian_dipole_absolute_far_field():
    frequency = 1e9
    wavelength = c / frequency
    spacing = np.asarray((wavelength / 30,) * 3)
    lower = (10, 10, 10)
    upper = (30, 30, 30)
    shape = (41, 41, 41)
    source = np.asarray((20, 20, 20)) * spacing
    current = np.asarray((0, 0, 1), dtype=np.complex128)
    origin = source.copy()
    surface_data = _analytic_surface_data(lower, upper, spacing, shape, frequency, source, current)
    theta = np.deg2rad(np.linspace(5, 175, 35))
    directions = np.column_stack((np.sin(theta), np.zeros_like(theta), np.cos(theta)))

    actual = evaluate_equivalent_current_far_zone(
        surface_data,
        (frequency,),
        directions,
        origin=origin,
        wave_speed=c,
        impedance=np.sqrt(mu_0 / epsilon_0),
        nthreads=2,
    )[0]
    transverse_current = current - directions * (directions @ current)[:, np.newaxis]
    expected = (
        -1j
        * (2 * np.pi * frequency / c)
        * np.sqrt(mu_0 / epsilon_0)
        / (4 * np.pi)
        * transverse_current
    )

    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error < 0.002


def test_cython_evaluator_matches_numpy_reference(monkeypatch):
    frequency = 750e6
    spacing = np.asarray((0.012, 0.010, 0.009))
    lower = (5, 6, 7)
    upper = (12, 14, 16)
    shape = (20, 22, 24)
    source = np.asarray((8.5, 10, 11.5)) * spacing
    current = np.asarray((0.3 - 0.1j, -0.2 + 0.4j, 1), dtype=np.complex128)
    surface_data = _analytic_surface_data(lower, upper, spacing, shape, frequency, source, current)
    directions = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)), dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    kwargs = dict(
        origin=source,
        wave_speed=c,
        impedance=np.sqrt(mu_0 / epsilon_0),
        nthreads=2,
    )

    cython_result = evaluate_equivalent_current_far_zone(
        surface_data, (frequency,), directions, **kwargs
    )
    monkeypatch.setattr(equivalent_currents, "_evaluate_equivalent_current_cython", None)
    numpy_result = evaluate_equivalent_current_far_zone(
        surface_data, (frequency,), directions, **kwargs
    )

    assert_allclose(cython_result, numpy_result, rtol=2e-13, atol=2e-13)
