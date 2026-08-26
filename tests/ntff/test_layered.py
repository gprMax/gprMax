import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

import gprMax.ntff.layered as layered_module
from gprMax.ntff.equivalent_currents import EquivalentCurrentPhasors, _evaluate_numpy
from gprMax.ntff.layered import (
    LayeredMedium,
    _responses_at_positions,
    _safe_ratio,
    _voltage_coefficients,
    evaluate_layered_currents,
)


def test_layered_recursion_rejects_numerically_singular_denominator():
    with pytest.raises(FloatingPointError, match="singular planar-layered recursion"):
        _safe_ratio(1.0 + 0j, 1e-16 + 0j, "test interface")


def test_layered_recursion_preserves_large_but_finite_physical_response():
    assert _safe_ratio(1.0 + 0j, 1e-10 + 0j, "test interface") == pytest.approx(1e10)


def _currents(seed=7, nfrequencies=2, npatches=11):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-0.17, 0.19, (npatches, 3))
    areas = rng.uniform(1e-4, 7e-4, npatches)
    electric = rng.normal(size=(nfrequencies, npatches, 3)) + 1j * rng.normal(
        size=(nfrequencies, npatches, 3)
    )
    magnetic = rng.normal(size=(nfrequencies, npatches, 3)) + 1j * rng.normal(
        size=(nfrequencies, npatches, 3)
    )
    return EquivalentCurrentPhasors(
        positions=positions,
        normals=np.zeros_like(positions),
        area_weights=areas,
        electric_current=electric,
        magnetic_current=magnetic,
    )


def _directions(seed=13, count=17):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(count, 3))
    values /= np.linalg.norm(values, axis=1)[:, np.newaxis]
    return values


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_layered_homogeneous_limit_matches_existing_equivalent_current(axis):
    frequencies = np.asarray((0.35e9, 1.1e9))
    currents = _currents(nfrequencies=frequencies.size)
    directions = _directions()
    medium = LayeredMedium(
        axis=axis,
        interfaces=np.empty(0),
        material_ids=("free_space",),
        relative_permittivity=np.ones((frequencies.size, 1), dtype=np.complex128),
        relative_permeability=np.ones((frequencies.size, 1), dtype=np.complex128),
    )

    actual = evaluate_layered_currents(currents, frequencies, directions, medium)
    expected = _evaluate_numpy(
        currents,
        2 * np.pi * frequencies / c,
        directions,
        np.sqrt(mu_0 / epsilon_0),
    )

    # The existing evaluator uses gprMax's configured vacuum constants while
    # this independent oracle uses SciPy's current CODATA constants.
    np.testing.assert_allclose(actual.electric, expected, rtol=2e-11, atol=4e-12)
    np.testing.assert_allclose(
        actual.magnetic,
        np.cross(directions[np.newaxis, :, :], expected) / np.sqrt(mu_0 / epsilon_0),
        rtol=2e-11,
        atol=4e-12,
    )


def test_identical_layers_do_not_create_an_interface():
    frequencies = np.asarray((0.6e9, 1.4e9))
    currents = _currents(nfrequencies=frequencies.size)
    directions = _directions(count=19)
    homogeneous = LayeredMedium(
        axis="z",
        interfaces=np.empty(0),
        material_ids=("m",),
        relative_permittivity=np.full((frequencies.size, 1), 2.7 + 0j),
        relative_permeability=np.full((frequencies.size, 1), 1.2 + 0j),
    )
    split = LayeredMedium(
        axis="z",
        interfaces=np.asarray((0.08, -0.04)),
        material_ids=("m0", "m1", "m2"),
        relative_permittivity=np.full((frequencies.size, 3), 2.7 + 0j),
        relative_permeability=np.full((frequencies.size, 3), 1.2 + 0j),
    )

    expected = evaluate_layered_currents(currents, frequencies, directions, homogeneous)
    actual = evaluate_layered_currents(currents, frequencies, directions, split)
    np.testing.assert_allclose(actual.electric, expected.electric, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.magnetic, expected.magnetic, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(
    ("real_dtype", "complex_dtype", "rtol", "atol"),
    ((np.float32, np.complex64, 2e-5, 2e-6), (np.float64, np.complex128, 2e-13, 2e-13)),
)
def test_cython_kernel_matches_independent_numpy_path(
    monkeypatch, real_dtype, complex_dtype, rtol, atol
):
    if layered_module._evaluate_layered_cython is None:
        pytest.skip("layered NTFF Cython extension has not been rebuilt")
    frequencies = np.asarray((0.55e9, 1.25e9), dtype=real_dtype)
    original = _currents(nfrequencies=frequencies.size, npatches=23)
    currents = EquivalentCurrentPhasors(
        positions=original.positions.astype(real_dtype),
        normals=original.normals.astype(real_dtype),
        area_weights=original.area_weights.astype(real_dtype),
        electric_current=original.electric_current.astype(complex_dtype),
        magnetic_current=original.magnetic_current.astype(complex_dtype),
    )
    directions = _directions(count=21).astype(real_dtype)
    medium = LayeredMedium(
        axis="y",
        interfaces=np.asarray((0.06, -0.035), dtype=real_dtype),
        material_ids=("upper", "lossy_film", "lower"),
        relative_permittivity=np.asarray(
            ((1.0, 3.4 - 0.18j, 2.1), (1.0, 3.1 - 0.12j, 2.1)),
            dtype=complex_dtype,
        ),
        relative_permeability=np.asarray(
            ((1.0, 1.2 - 0.03j, 1.0), (1.0, 1.15 - 0.02j, 1.0)),
            dtype=complex_dtype,
        ),
    )

    accelerated = evaluate_layered_currents(currents, frequencies, directions, medium, nthreads=2)
    monkeypatch.setattr(layered_module, "_evaluate_layered_cython", None)
    reference = evaluate_layered_currents(currents, frequencies, directions, medium)

    np.testing.assert_allclose(accelerated.electric, reference.electric, rtol=rtol, atol=atol)
    np.testing.assert_allclose(accelerated.magnetic, reference.magnetic, rtol=rtol, atol=atol)
    np.testing.assert_allclose(accelerated.impedance, reference.impedance, rtol=rtol, atol=atol)
    np.testing.assert_allclose(accelerated.wavenumber, reference.wavenumber, rtol=rtol, atol=atol)


def test_cython_accepts_double_phasors_with_single_precision_geometry():
    """Runtime DFT phasors may be double while FDTD coordinates are single."""

    if layered_module._evaluate_layered_cython is None:
        pytest.skip("layered NTFF Cython extension has not been rebuilt")
    original = _currents(nfrequencies=1, npatches=7)
    currents = EquivalentCurrentPhasors(
        positions=original.positions.astype(np.float32),
        normals=original.normals.astype(np.float32),
        area_weights=original.area_weights.astype(np.float32),
        electric_current=original.electric_current.astype(np.complex128),
        magnetic_current=original.magnetic_current.astype(np.complex128),
    )
    medium = LayeredMedium(
        axis="z",
        interfaces=np.asarray((0.0,), dtype=np.float32),
        material_ids=("upper", "lower"),
        relative_permittivity=np.asarray(((1.0, 2.5),), dtype=np.complex128),
        relative_permeability=np.ones((1, 2), dtype=np.complex128),
    )

    result = evaluate_layered_currents(
        currents,
        np.asarray((1.25e9,), dtype=np.float32),
        np.asarray(((0.3, 0.4, np.sqrt(0.75)),), dtype=np.float32),
        medium,
    )

    assert result.electric.dtype == np.complex128
    assert result.magnetic.dtype == np.complex128
    assert result.impedance.dtype == np.float64
    assert result.wavenumber.dtype == np.float64
    assert np.all(np.isfinite(result.electric))
    assert np.all(np.isfinite(result.magnetic))


def test_direction_blocking_is_numerically_transparent():
    frequencies = np.asarray((0.7e9, 1.3e9))
    currents = _currents(nfrequencies=2, npatches=19)
    directions = _directions(count=37)
    medium = LayeredMedium(
        axis="z",
        interfaces=np.asarray((0.04, -0.03)),
        material_ids=("upper", "film", "lower"),
        relative_permittivity=np.asarray(((1, 2.7 - 0.1j, 1.8),) * 2),
        relative_permeability=np.asarray(((1, 1.1, 1),) * 2),
    )

    unblocked = evaluate_layered_currents(
        currents, frequencies, directions, medium, direction_block_size=1000
    )
    blocked = evaluate_layered_currents(
        currents, frequencies, directions, medium, direction_block_size=7
    )

    np.testing.assert_array_equal(blocked.electric, unblocked.electric)
    np.testing.assert_array_equal(blocked.magnetic, unblocked.magnetic)
    np.testing.assert_array_equal(blocked.impedance, unblocked.impedance)
    np.testing.assert_array_equal(blocked.wavenumber, unblocked.wavenumber)


@pytest.mark.parametrize("upper_observation", (True, False))
def test_single_interface_voltage_transmission_matches_fresnel_limit(upper_observation):
    beta = np.asarray((3.1 + 0j, 5.3 - 0.2j))
    eta = np.asarray((0.73 + 0j, 0.42 - 0.03j))
    plus, minus = _voltage_coefficients(
        beta,
        eta,
        np.zeros(2),
        upper_observation=upper_observation,
    )
    transmitted = minus[1] if upper_observation else plus[0]
    expected = 2 * eta[0] * eta[1] / (eta[0] + eta[1])
    np.testing.assert_allclose(transmitted, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("upper_observation", (True, False))
def test_three_layer_responses_match_capoglu_closed_form(upper_observation):
    """Exercise the multilayer normalization that a homogeneous test hides."""

    thickness_value = 0.02
    source_depth = -0.01
    interfaces = np.asarray((0.0, -thickness_value))
    beta = np.asarray((20.0 + 0j, 30.0 - 1j, 25.0 + 0j))
    paper_eta = np.asarray((0.8 + 0j, 0.55 - 0.03j, 0.7 + 0j))
    thickness = np.asarray((0.0, thickness_value, 0.0))
    plus, minus = _voltage_coefficients(
        beta,
        paper_eta,
        thickness,
        upper_observation=upper_observation,
    )
    if upper_observation:
        exterior_phase = np.exp(1j * beta[0] * interfaces[0])
    else:
        exterior_phase = np.exp(-1j * beta[-1] * interfaces[-1])
    plus *= exterior_phase
    minus *= exterior_phase
    raw_vi, vv = _responses_at_positions(
        np.asarray((source_depth,)),
        np.asarray((1,)),
        interfaces,
        beta,
        paper_eta,
        plus,
        minus,
    )
    vi_normalized = raw_vi

    # Eqs. (18)--(30) imply Gamma_mn=(eta_n-eta_m)/(eta_m+eta_n).
    # The numerator in the printed Eq. (38) has the opposite sign; using it
    # is inconsistent with both the recursions and the homogeneous limit.
    gamma10 = (paper_eta[0] - paper_eta[1]) / (paper_eta[1] + paper_eta[0])
    gamma12 = (paper_eta[2] - paper_eta[1]) / (paper_eta[1] + paper_eta[2])
    denominator = 1 - gamma10 * gamma12 * np.exp(-2j * beta[1] * thickness_value)
    if upper_observation:
        coefficient = (1 + gamma10) / denominator
        downward = np.exp(1j * beta[1] * source_depth)
        upward = gamma12 * np.exp(-1j * beta[1] * (2 * thickness_value + source_depth))
        expected_vv = coefficient * (downward - upward)
        expected_vi_normalized = paper_eta[1] * coefficient * (downward + upward)
    else:
        coefficient = -(
            (1 + gamma12) * np.exp(-1j * (beta[1] - beta[2]) * thickness_value) / denominator
        )
        upward = np.exp(-1j * beta[1] * source_depth)
        downward = gamma10 * np.exp(1j * beta[1] * source_depth)
        expected_vv = coefficient * (upward - downward)
        expected_vi_normalized = -paper_eta[1] * coefficient * (upward + downward)

    np.testing.assert_allclose(vv[0], expected_vv, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(vi_normalized[0], expected_vi_normalized, rtol=2e-14, atol=2e-14)


def test_lossy_observation_halfspace_is_rejected():
    medium = LayeredMedium(
        axis="z",
        interfaces=np.asarray((0.0,)),
        material_ids=("lossy", "free_space"),
        relative_permittivity=np.asarray(((2 - 0.1j, 1 + 0j),)),
        relative_permeability=np.ones((1, 2), dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="observation half-spaces must be lossless"):
        evaluate_layered_currents(
            _currents(nfrequencies=1),
            np.asarray((1e9,)),
            np.asarray(((0.0, 0.0, 1.0),)),
            medium,
        )


def test_exact_grazing_direction_is_rejected_explicitly():
    medium = LayeredMedium(
        axis="z",
        interfaces=np.empty(0),
        material_ids=("free_space",),
        relative_permittivity=np.ones((1, 1), dtype=np.complex128),
        relative_permeability=np.ones((1, 1), dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="exact grazing"):
        evaluate_layered_currents(
            _currents(nfrequencies=1),
            np.asarray((1e9,)),
            np.asarray(((1.0, 0.0, 0.0),)),
            medium,
        )


@pytest.mark.parametrize("frequency", (0.0, -1.0, np.nan, np.inf))
def test_invalid_frequency_is_rejected(frequency):
    medium = LayeredMedium(
        axis="z",
        interfaces=np.empty(0),
        material_ids=("free_space",),
        relative_permittivity=np.ones((1, 1), dtype=np.complex128),
        relative_permeability=np.ones((1, 1), dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="finite and strictly positive"):
        evaluate_layered_currents(
            _currents(nfrequencies=1),
            np.asarray((frequency,)),
            np.asarray(((0.0, 0.0, 1.0),)),
            medium,
        )
