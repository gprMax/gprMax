"""Check the circular references and the validation's voxel SIBC contours."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import jv, jvp

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import (
    AlgorithmicSurfaceResponse, boundary_edge_relative_permittivity)
from testing.validation.eigenmode_tracking.profiles.cylindrical_modal_profiles import (
    analytical_fields, bessel_profile, reference_modes, voxel_surface_boundary)
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
from testing.validation.eigenmode_tracking.profiles.waveguide_modal_profiles import vector, matching


def response():
    return AlgorithmicSurfaceResponse(theta=.1, physical_angular_frequency=1e11,
        discrete_angular_frequency=1e11, midpoint_cosine=1., impedance=.1+.1j,
        admittance=5-5j)


def test_voxel_sibc_contours_reduce_to_verified_rectangle():
    path = Path(__file__).parent/'fdfd_eigenmode_solver/test_surface_impedance_fdfd_operator.py'
    spec = importlib.util.spec_from_file_location('rectangle_boundary_fixture', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    with _configured_solver_constants():
        actual = voxel_surface_boundary(np.ones((5, 4), bool), .001, response())
        expected = fixture._rectangular_impedance_boundary(5, 4, .001, .001, response())
    for a, b in zip(actual.electric_retained+actual.magnetic_retained,
                    expected.electric_retained+expected.magnetic_retained):
        np.testing.assert_array_equal(a, b)
    rows = {(r.electric_axis, r.electric_index): r for r in expected.rows}
    assert len(actual.rows) == len(rows)
    for row in actual.rows:
        ref = rows[(row.electric_axis, row.electric_index)]
        assert row.retained_dual_area == pytest.approx(ref.retained_dual_area)
        assert row.relative_permittivity == pytest.approx(ref.relative_permittivity)
        assert {(t.axis, t.index): t.line_weight for t in row.magnetic_terms} == pytest.approx(
            {(t.axis, t.index): t.line_weight for t in ref.magnetic_terms})


def test_concave_voxel_corner_retains_three_quadrants_and_two_ports():
    air = np.array([[True, True], [True, False]])
    with _configured_solver_constants():
        boundary = voxel_surface_boundary(air, .001, response())
        e0 = config.sim_config.em_consts['e0']
        expected = boundary_edge_relative_permittivity(response=response(), epsilon0=e0,
            retained_dual_area=3e-6/4, electric_mass=e0*3e-6/4,
            conductive_mass=0., port_lengths=(.0005, .0005))
    corner = next(r for r in boundary.rows if r.electric_axis == 2 and r.electric_index == (1, 1))
    assert corner.retained_dual_area == pytest.approx(3e-6/4)
    assert corner.relative_permittivity == pytest.approx(expected)
    for row in boundary.rows:
        assert boundary.electric_retained[row.electric_axis][row.electric_index]
        for term in row.magnetic_terms:
            assert boundary.magnetic_retained[term.axis][term.index]


@pytest.mark.parametrize('mode', reference_modes())
def test_bessel_references_satisfy_smooth_pec_wall(mode):
    root, family, m, _, _ = mode
    assert abs(jvp(m, root) if family == 'TE' else jv(m, root)) < 1e-12
    e, h = analytical_fields(mode, 32, 40e9, 299792458., 376.730313)
    assert all(np.all(np.isfinite(a)) for a in e+h)


@pytest.mark.parametrize('angular', ['cos', 'sin'])
def test_bessel_gradient_including_axis(angular):
    root = reference_modes()[0][0]
    x = np.array([0., .001, -.002]); y = np.array([0., -.003, .001])
    _, dx, dy, _ = bessel_profile(1, root, angular, x, y)
    step = 1e-8
    for direction, derivative in [(0, dx), (1, dy)]:
        plus = bessel_profile(1, root, angular, x+(step if direction == 0 else 0), y+(step if direction == 1 else 0))[0]
        minus = bessel_profile(1, root, angular, x-(step if direction == 0 else 0), y-(step if direction == 1 else 0))[0]
        np.testing.assert_allclose(derivative, (plus-minus)/(2*step), atol=1e-7)


def test_angular_partners_have_distinct_full_vector_profiles():
    modes = reference_modes()[:2]
    fields = np.column_stack([vector(*analytical_fields(m, 64, 40e9, 299792458., 376.730313), 376.730313) for m in modes])
    np.testing.assert_allclose(matching(fields, fields), np.eye(2), atol=1e-12)
