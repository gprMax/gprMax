import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

import gprMax.ntff.layered_time as layered_time_module
from gprMax.ntff.equivalent_current_time import EquivalentCurrentTimeMonitor
from gprMax.ntff.layered import (
    LayeredMedium,
    LayeredTermination,
    _exterior_reference,
    _layer_thicknesses,
    _responses_at_positions,
    _voltage_coefficients,
)
from gprMax.ntff.layered_time import (
    LayeredEquivalentCurrentTimeMonitor,
    build_layered_impulse_responses,
)


def _frequency_responses(
    interfaces,
    eps_absolute,
    mu_absolute,
    direction,
    source_position,
    frequencies,
    termination=None,
):
    """Independent calls into the established frequency layered recursion."""

    interfaces = np.asarray(interfaces, dtype=float)
    eps_absolute = np.asarray(eps_absolute, dtype=float)
    mu_absolute = np.asarray(mu_absolute, dtype=float)
    direction = np.asarray(direction, dtype=float)
    source_layer = np.searchsorted(-interfaces, -source_position, side="left")
    upper = direction[2] > 0
    exterior = 0 if upper else -1
    eps = eps_absolute / eps_absolute[exterior]
    mu = mu_absolute / mu_absolute[exterior]
    q = np.sqrt(eps * mu - np.hypot(direction[0], direction[1]) ** 2)
    eta_e = q / eps
    eta_h = mu / q
    wave_speed = c / np.sqrt(eps_absolute[exterior] * mu_absolute[exterior])
    thickness = _layer_thicknesses(interfaces, eps.size, termination)
    result = {key: [] for key in ("vi_e", "vv_e", "vi_h", "vv_h")}
    for frequency in frequencies:
        beta = 2 * np.pi * frequency * q / wave_speed
        responses = []
        for impedance in (eta_e, eta_h):
            plus, minus = _voltage_coefficients(
                beta,
                impedance,
                thickness,
                upper_observation=upper,
                termination=termination,
            )
            if interfaces.size or termination is not None:
                reference = _exterior_reference(interfaces, termination, upper_observation=upper)
                phase = np.exp(1j * beta[0] * reference) if upper else np.exp(-1j * beta[-1] * reference)
                plus *= phase
                minus *= phase
            responses.extend(
                _responses_at_positions(
                    np.asarray([source_position]),
                    np.asarray([source_layer]),
                    interfaces,
                    beta,
                    impedance,
                    plus,
                    minus,
                    termination,
                )
            )
        for key, response in zip(result, responses):
            result[key].append(response[0])
    return {key: np.asarray(values) for key, values in result.items()}


@pytest.mark.parametrize(
    "interfaces,eps,mu,source_position,direction",
    [
        ((), (1.0,), (1.0,), -0.017, (0.3, 0.4, np.sqrt(0.75))),
        ((0.0,), (1.0, 4.0), (1.0, 1.0), -0.013, (0.3, 0.0, np.sqrt(0.91))),
        ((0.0,), (1.0, 4.0), (1.0, 1.0), 0.013, (0.3, 0.0, -np.sqrt(0.91))),
        (
            (0.012, -0.009),
            (1.0, 2.7, 5.2),
            (1.0, 1.4, 0.9),
            0.003,
            (0.22, -0.31, np.sqrt(1 - 0.22**2 - 0.31**2)),
        ),
        (
            (0.012, -0.009),
            (1.0, 2.7, 5.2),
            (1.0, 1.4, 0.9),
            0.003,
            (0.22, -0.31, -np.sqrt(1 - 0.22**2 - 0.31**2)),
        ),
    ],
)
def test_impulse_train_fourier_transform_matches_frequency_recursion(interfaces, eps, mu, source_position, direction):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    frequencies = np.asarray((0.2e9, 0.71e9, 1.3e9, 2.4e9))
    impulse = build_layered_impulse_responses(
        interfaces,
        eps,
        mu,
        direction,
        source_position,
        impulse_tolerance=1e-13,
    )
    expected = _frequency_responses(interfaces, eps, mu, direction, source_position, frequencies)
    for name in expected:
        actual = getattr(impulse, name).frequency_response(frequencies)
        np.testing.assert_allclose(actual, expected[name], rtol=2e-11, atol=2e-12)


def test_homogeneous_response_is_one_delayed_impulse():
    direction = np.asarray((0.0, 0.6, 0.8))
    result = build_layered_impulse_responses((), (3.0,), (1.5,), direction, -0.02)
    expected_delay = -direction[2] * -0.02 / (c / np.sqrt(4.5))
    expected_impedance = np.sqrt(mu_0 * 1.5 / (epsilon_0 * 3.0))
    for train in (result.vi_e, result.vv_e, result.vi_h, result.vv_h):
        assert train.delays.shape == (1,)
        assert train.delays[0] == pytest.approx(expected_delay)
    assert result.vi_e.amplitudes[0] == pytest.approx(direction[2])
    assert result.vv_e.amplitudes[0] == pytest.approx(1.0)
    assert result.vi_h.amplitudes[0] == pytest.approx(1 / direction[2])
    assert result.vv_h.amplitudes[0] == pytest.approx(1.0)
    assert result.observation_impedance == pytest.approx(expected_impedance)


def test_halfspace_impulse_matches_closed_form_transmission():
    """Half-space responses reproduce the closed transmitted/reflected impulses.

    This is an independent half-space oracle: it uses only the elementary
    lossless transmission-line coefficient and propagation delay rather than
    the established layered frequency recursion. It is Eqs. (53)--(55) of
    Capoglu's thesis after accounting for this implementation's factor-two
    Green-response convention and impedance normalisation.
    """

    source_position = -0.013
    direction = np.asarray((0.3, 0.0, np.sqrt(0.91)))
    eps = np.asarray((1.0, 4.0))
    mu = np.asarray((1.0, 1.0))
    result = build_layered_impulse_responses((0.0,), eps, mu, direction, source_position, impulse_tolerance=1e-14)

    q = np.sqrt(eps * mu - direction[0] ** 2)
    delay = abs(source_position) * q[1] / c
    for suffix, impedance in (("e", q / eps), ("h", mu / q)):
        transmission = 2 * impedance[0] / (impedance[1] + impedance[0])
        for prefix, source_scale in (("vi", impedance[1]), ("vv", 1.0)):
            train = getattr(result, f"{prefix}_{suffix}")
            np.testing.assert_allclose(train.delays, (delay,), rtol=2e-15, atol=1e-20)
            np.testing.assert_allclose(
                train.amplitudes,
                (source_scale * transmission,),
                rtol=2e-15,
                atol=2e-15,
            )

    source_position = abs(source_position)
    result = build_layered_impulse_responses((0.0,), eps, mu, direction, source_position, impulse_tolerance=1e-14)
    expected_delays = (-source_position * q[0] / c, source_position * q[0] / c)
    for suffix, impedance in (("e", q / eps), ("h", mu / q)):
        reflection = (impedance[1] - impedance[0]) / (impedance[1] + impedance[0])
        for prefix, source_scale, reflection_sign in (
            ("vi", impedance[0], 1.0),
            ("vv", 1.0, -1.0),
        ):
            train = getattr(result, f"{prefix}_{suffix}")
            np.testing.assert_allclose(train.delays, expected_delays, rtol=2e-15, atol=1e-20)
            np.testing.assert_allclose(
                train.amplitudes,
                (source_scale, reflection_sign * source_scale * reflection),
                rtol=2e-15,
                atol=2e-15,
            )


def test_finite_slab_impulses_match_closed_form_echo_series():
    """The upward response of a finite slab is a geometric echo train.

    This independently evaluates Eqs. (71)--(75) of Capoglu's thesis, with
    the same response normalisation described in the half-space test.
    """

    thickness = 0.011
    source_position = 0.007
    interfaces = (0.0, -thickness)
    eps = np.asarray((1.0, 3.6, 1.8))
    mu = np.asarray((1.0, 1.3, 0.9))
    direction = np.asarray((0.2, -0.1, np.sqrt(0.95)))
    result = build_layered_impulse_responses(
        interfaces,
        eps,
        mu,
        direction,
        source_position,
        impulse_tolerance=1e-14,
    )

    q = np.sqrt(eps * mu - np.hypot(direction[0], direction[1]) ** 2)
    slowness = q / c
    echo_number = np.arange(1, 5)
    expected_delays = np.concatenate(
        (
            (-slowness[0] * source_position,),
            (slowness[0] * source_position,),
            slowness[0] * source_position + 2 * echo_number * thickness * slowness[1],
        )
    )

    for suffix, impedance in (("e", q / eps), ("h", mu / q)):
        transmission_01 = 2 * impedance[1] / (impedance[0] + impedance[1])
        transmission_10 = 2 * impedance[0] / (impedance[1] + impedance[0])
        reflection_12 = (impedance[2] - impedance[1]) / (impedance[2] + impedance[1])
        reflection_10 = (impedance[0] - impedance[1]) / (impedance[0] + impedance[1])
        reflection_01 = -reflection_10
        round_trip = reflection_12 * reflection_10
        for prefix, source_scale, downward_sign in (
            ("vi", impedance[0], 1.0),
            ("vv", 1.0, -1.0),
        ):
            first_echo = downward_sign * source_scale * transmission_01 * reflection_12 * transmission_10
            expected_amplitudes = np.concatenate(
                (
                    (source_scale,),
                    (downward_sign * source_scale * reflection_01,),
                    first_echo * round_trip ** (echo_number - 1),
                )
            )
            train = getattr(result, f"{prefix}_{suffix}")
            np.testing.assert_allclose(train.delays[: expected_delays.size], expected_delays, rtol=2e-14, atol=1e-20)
            np.testing.assert_allclose(
                train.amplitudes[: expected_amplitudes.size],
                expected_amplitudes,
                rtol=2e-14,
                atol=2e-15,
            )


def test_grounded_slab_impulses_match_capoglu_equations_59_63_to_65():
    """Reproduce the grounded-slab Green impulse trains used in thesis Fig. 11."""

    thickness = 2e-3
    source_depth = 0.8e-3
    interfaces = np.asarray((0.0,))
    termination = LayeredTermination("pec", "negative", -thickness)
    eps = np.asarray((1.0, 2.5))
    mu = np.asarray((1.0, 1.0))
    direction = np.asarray((0.5, 0.5, np.sqrt(0.5)))
    result = build_layered_impulse_responses(
        interfaces,
        eps,
        mu,
        direction,
        -source_depth,
        impulse_tolerance=1e-14,
        termination=termination,
    )

    q = np.sqrt(eps * mu - np.hypot(direction[0], direction[1]) ** 2)
    slowness = q / c
    reflection_count = np.arange(5)
    for suffix, impedance in (("e", q / eps), ("h", mu / q)):
        transmission_10 = 2 * impedance[0] / (impedance[1] + impedance[0])
        reflection_10 = (impedance[0] - impedance[1]) / (impedance[0] + impedance[1])
        round_trip = -reflection_10
        amplitudes = transmission_10 * round_trip**reflection_count
        upward_delays = (source_depth + 2 * reflection_count * thickness) * slowness[1]
        ground_delays = (2 * thickness - source_depth + 2 * reflection_count * thickness) * slowness[1]
        delays = np.concatenate((upward_delays, ground_delays))
        order = np.argsort(delays)

        vv_expected = np.concatenate((amplitudes, amplitudes))[order]
        vi_expected = impedance[1] * np.concatenate((amplitudes, -amplitudes))[order]
        for name, expected in ((f"vv_{suffix}", vv_expected), (f"vi_{suffix}", vi_expected)):
            train = getattr(result, name)
            np.testing.assert_allclose(train.delays[: order.size], delays[order], rtol=2e-14, atol=1e-20)
            np.testing.assert_allclose(train.amplitudes[: order.size], expected, rtol=3e-14, atol=2e-15)


def test_grounded_slab_impulse_spectrum_matches_frequency_short_circuit():
    interfaces = np.asarray((0.0,))
    termination = LayeredTermination("pec", "negative", -0.002)
    eps = np.asarray((1.0, 2.5))
    mu = np.asarray((1.0, 1.0))
    direction = np.asarray((0.5, 0.5, np.sqrt(0.5)))
    source_position = -0.001
    frequencies = np.asarray((2e9, 11e9, 27e9, 45e9))
    impulse = build_layered_impulse_responses(
        interfaces,
        eps,
        mu,
        direction,
        source_position,
        impulse_tolerance=1e-13,
        termination=termination,
    )
    expected = _frequency_responses(
        interfaces,
        eps,
        mu,
        direction,
        source_position,
        frequencies,
        termination,
    )
    for name in expected:
        np.testing.assert_allclose(
            getattr(impulse, name).frequency_response(frequencies),
            expected[name],
            rtol=5e-11,
            atol=5e-12,
        )


def test_positive_axis_grounded_slab_matches_frequency_short_circuit():
    """Exercise the mirrored recursion with the PEC above the source."""

    interfaces = np.asarray((0.0,))
    termination = LayeredTermination("pec", "positive", 0.002)
    eps = np.asarray((2.5, 1.0))
    mu = np.asarray((1.0, 1.0))
    direction = np.asarray((0.5, 0.5, -np.sqrt(0.5)))
    source_position = 0.001
    frequencies = np.asarray((2e9, 11e9, 27e9, 45e9))
    impulse = build_layered_impulse_responses(
        interfaces,
        eps,
        mu,
        direction,
        source_position,
        impulse_tolerance=1e-13,
        termination=termination,
    )
    expected = _frequency_responses(
        interfaces,
        eps,
        mu,
        direction,
        source_position,
        frequencies,
        termination,
    )
    for name in expected:
        np.testing.assert_allclose(
            getattr(impulse, name).frequency_response(frequencies),
            expected[name],
            rtol=5e-11,
            atol=5e-12,
        )


def test_loss_dispersion_evanescence_and_grazing_are_rejected():
    with pytest.raises(ValueError, match="lossless"):
        build_layered_impulse_responses((), (2 - 0.1j,), (1,), (0, 0, 1), 0)
    with pytest.raises(ValueError, match="propagating"):
        build_layered_impulse_responses((0,), (4, 1), (1, 1), (0.9, 0, np.sqrt(0.19)), -0.01)
    with pytest.raises(ValueError, match="grazing"):
        build_layered_impulse_responses((), (1,), (1,), (1, 0, 0), 0)


def test_identical_fictitious_interfaces_do_not_change_response():
    direction = np.asarray((0.2, 0.3, np.sqrt(0.87)))
    reference = build_layered_impulse_responses((), (2,), (1.2,), direction, -0.014)
    split = build_layered_impulse_responses(
        (0.02, -0.005, -0.03),
        (2, 2, 2, 2),
        (1.2, 1.2, 1.2, 1.2),
        direction,
        -0.014,
    )
    frequencies = np.linspace(0, 3e9, 17)
    for name in ("vi_e", "vv_e", "vi_h", "vv_h"):
        np.testing.assert_allclose(
            getattr(split, name).frequency_response(frequencies),
            getattr(reference, name).frequency_response(frequencies),
            rtol=1e-13,
            atol=1e-13,
        )


def test_random_lossless_multilayers_match_frequency_recursion():
    """Exercise both observation exteriors, every source layer, and magnetic contrast."""

    rng = np.random.default_rng(20260826)
    frequencies = np.asarray((0.13e9, 0.47e9, 1.1e9, 2.7e9))
    for nlayers in range(2, 6):
        interfaces = np.sort(rng.uniform(-0.025, 0.025, nlayers - 1))[::-1]
        eps = rng.uniform(1.0, 4.0, nlayers)
        mu = rng.uniform(0.8, 2.2, nlayers)
        for upper_observation in (True, False):
            normal = np.sqrt(1 - 0.08**2 - 0.05**2)
            direction = np.asarray((0.08, -0.05, normal if upper_observation else -normal))
            for source_layer in range(nlayers):
                if source_layer == 0:
                    source_position = interfaces[0] + 0.007
                elif source_layer == nlayers - 1:
                    source_position = interfaces[-1] - 0.007
                else:
                    source_position = 0.5 * (interfaces[source_layer - 1] + interfaces[source_layer])
                impulse = build_layered_impulse_responses(
                    interfaces,
                    eps,
                    mu,
                    direction,
                    source_position,
                    source_layer=source_layer,
                    impulse_tolerance=1e-13,
                )
                expected = _frequency_responses(interfaces, eps, mu, direction, source_position, frequencies)
                for name in expected:
                    np.testing.assert_allclose(
                        getattr(impulse, name).frequency_response(frequencies),
                        expected[name],
                        rtol=5e-11,
                        atol=5e-12,
                    )


def test_path_truncation_converges_and_reports_discarded_state_amplitudes():
    interfaces = np.asarray((0.014, -0.011))
    eps = np.asarray((1.0, 8.0, 2.3))
    mu = np.asarray((1.0, 1.7, 0.9))
    direction = np.asarray((0.06, 0.04, np.sqrt(1 - 0.06**2 - 0.04**2)))
    frequencies = np.linspace(0.1e9, 3e9, 31)
    expected = _frequency_responses(interfaces, eps, mu, direction, 0.0, frequencies)
    loose = build_layered_impulse_responses(interfaces, eps, mu, direction, 0.0, impulse_tolerance=1e-3)
    tight = build_layered_impulse_responses(interfaces, eps, mu, direction, 0.0, impulse_tolerance=1e-11)
    for name in expected:
        loose_error = np.max(np.abs(getattr(loose, name).frequency_response(frequencies) - expected[name]))
        tight_error = np.max(np.abs(getattr(tight, name).frequency_response(frequencies) - expected[name]))
        assert tight_error < loose_error
        assert tight_error < 2e-10
        assert getattr(loose, name).discarded_path_amplitude_sum > 0


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_homogeneous_layered_monitor_matches_existing_time_transform(axis):
    kwargs = dict(
        name="homogeneous",
        lower=(2, 2, 2),
        upper=(5, 5, 5),
        spacing=(0.01, 0.01, 0.01),
        field_shape=(8, 8, 8),
        dt=1e-10,
        iterations=12,
        theta=(25, 65, 120),
        phi=(25, 55, 205),
        origin=(0.035, 0.035, 0.035),
        real_dtype=np.float64,
        nthreads=1,
    )
    conventional = EquivalentCurrentTimeMonitor(
        **kwargs,
        wave_speed=c,
        impedance=np.sqrt(mu_0 / epsilon_0),
    )
    medium = LayeredMedium(
        axis=axis,
        interfaces=np.empty(0),
        material_ids=("free_space",),
        relative_permittivity=np.ones((1, 1), dtype=complex),
        relative_permeability=np.ones((1, 1), dtype=complex),
    )
    layered = LayeredEquivalentCurrentTimeMonitor(**kwargs, medium=medium)
    fields = np.random.default_rng(9137).standard_normal((12, 6, 8, 8, 8))
    for iteration in range(12):
        conventional.observe_electric(iteration, *fields[iteration, :3])
        conventional.observe_magnetic(iteration, *fields[iteration, 3:])
        layered.observe_electric(iteration, *fields[iteration, :3])
        layered.observe_magnetic(iteration, *fields[iteration, 3:])
    conventional.finalise()
    layered.finalise()
    np.testing.assert_allclose(layered.result.times, conventional.result.times, rtol=0, atol=0)
    for component in ("Etheta", "Ephi"):
        np.testing.assert_allclose(
            layered.result.fields[component],
            conventional.result.fields[component],
            rtol=3e-13,
            atol=3e-13,
        )


@pytest.mark.parametrize("real_dtype", (np.float32, np.float64))
def test_cython_deposition_matches_numpy_reference(monkeypatch, real_dtype):
    if layered_time_module._deposit_layered_impulse_time is None:
        pytest.skip("layered NTFF Cython extension is not built")
    kwargs = dict(
        name="halfspace",
        lower=(2, 2, 2),
        upper=(5, 5, 5),
        spacing=(0.01, 0.01, 0.01),
        field_shape=(8, 8, 8),
        dt=1e-10,
        iterations=12,
        theta=(30, 150),
        phi=(20, 70),
        origin=(0.035, 0.035, 0.035),
        real_dtype=real_dtype,
        nthreads=2,
    )
    medium = LayeredMedium(
        axis="z",
        interfaces=np.asarray((0.035,)),
        material_ids=("upper", "lower"),
        relative_permittivity=np.asarray(((1.0, 2.4),), dtype=complex),
        relative_permeability=np.asarray(((1.0, 1.2),), dtype=complex),
    )
    fields = np.asarray(
        np.random.default_rng(2718).standard_normal((12, 6, 8, 8, 8)),
        dtype=real_dtype,
    )

    compiled = LayeredEquivalentCurrentTimeMonitor(**kwargs, medium=medium)
    # Axial response templates must be shared by transverse patches rather
    # than expanding to one impulse train per direction/patch row.
    assert compiled._response_csr[0][0].size < compiled._row_template.size + 1
    for iteration in range(12):
        compiled.observe_electric(iteration, *fields[iteration, :3])
        compiled.observe_magnetic(iteration, *fields[iteration, 3:])
    compiled.finalise()

    monkeypatch.setattr(layered_time_module, "_deposit_layered_impulse_time", None)
    reference = LayeredEquivalentCurrentTimeMonitor(**kwargs, medium=medium)
    for iteration in range(12):
        reference.observe_electric(iteration, *fields[iteration, :3])
        reference.observe_magnetic(iteration, *fields[iteration, 3:])
    reference.finalise()

    np.testing.assert_array_equal(compiled.result.times, reference.result.times)
    for component in ("Etheta", "Ephi"):
        tolerance = 2e-6 if real_dtype == np.float32 else 3e-13
        np.testing.assert_allclose(
            compiled.result.fields[component],
            reference.result.fields[component],
            rtol=tolerance,
            atol=tolerance,
        )
