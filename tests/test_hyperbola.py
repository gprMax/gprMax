"""Tests for hyperbola.py, anchored to examples/cylinder_Bscan_2D.in."""

import numpy as np
import pytest

from toolboxes.Marimo.hyperbola import (
    C_M_PER_NS,
    apex_source_x,
    apex_time,
    permittivity,
    permittivity_from_apex,
    ricker_delay,
    travel_time,
    velocity,
)

# Every constant below is read directly out of examples/cylinder_Bscan_2D.in:
#   #material: 6 0 1 0 half_space
#   #waveform: ricker 1 1.5e9 my_ricker
#   #hertzian_dipole: z 0.040 0.170 0 my_ricker
#   #rx: 0.080 0.170 0
#   #box: 0 0 0 0.240 0.170 0.002 half_space
#   #cylinder: 0.120 0.080 0 0.120 0.080 0.002 0.010 pec
EPS_R = 6.0
FREQ = 1.5e9
X_TARGET = 0.120
SURFACE_Y = 0.170
TARGET_Y = 0.080
DEPTH = SURFACE_Y - TARGET_Y      # 0.090 m to the cylinder centre
RADIUS = 0.010
OFFSET = 0.080 - 0.040            # 0.040 m, receiver trails the source
DELAY = np.sqrt(2.0) / FREQ * 1e9


class TestVelocity:
    def test_vacuum(self):
        assert velocity(1.0) == pytest.approx(C_M_PER_NS)

    def test_half_space(self):
        assert velocity(EPS_R) == pytest.approx(0.122389, abs=1e-6)

    def test_round_trip_with_permittivity(self):
        assert permittivity(velocity(6.0)) == pytest.approx(6.0)

    def test_non_positive_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            velocity(0.0)
        with pytest.raises(ValueError, match="must be positive"):
            permittivity(-1.0)


class TestRickerDelay:
    def test_matches_gprmax_definition(self):
        # gprMax waveforms.py: chi = sqrt(2) / freq for the ricker family
        assert ricker_delay(FREQ) == pytest.approx(0.942809, abs=1e-6)

    def test_is_a_large_fraction_of_the_example_window(self):
        # 3e-9 time_window in the .in file. Omitting the delay is not a
        # rounding error, it is a third of the record.
        assert ricker_delay(FREQ) / 3.005 > 0.30

    def test_scales_inversely_with_frequency(self):
        assert ricker_delay(3e9) == pytest.approx(ricker_delay(1.5e9) / 2)

    def test_non_positive_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            ricker_delay(0.0)


class TestTravelTime:
    def test_hyperbola_is_symmetric_about_the_apex(self):
        ax = apex_source_x(X_TARGET, OFFSET)
        left = travel_time(ax - 0.03, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY)
        right = travel_time(ax + 0.03, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY)
        assert left == pytest.approx(right)

    def test_apex_is_the_minimum(self):
        xs = np.linspace(0.0, 0.24, 481)
        t = travel_time(xs, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY)
        assert xs[int(np.argmin(t))] == pytest.approx(apex_source_x(X_TARGET, OFFSET), abs=0.001)

    def test_bistatic_apex_is_offset_from_the_target(self):
        # Reading the apex straight off the source axis misplaces the target
        # by half the antenna separation.
        assert apex_source_x(X_TARGET, OFFSET) == pytest.approx(0.100)
        assert apex_source_x(X_TARGET, 0.0) == pytest.approx(X_TARGET)

    def test_radius_correction_shortens_both_rays(self):
        point = travel_time(0.040, X_TARGET, DEPTH, EPS_R, OFFSET, 0.0, DELAY)
        cyl = travel_time(0.040, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY)
        assert cyl < point
        assert point - cyl == pytest.approx(2 * RADIUS / velocity(EPS_R))

    def test_delay_shifts_the_whole_curve_rigidly(self):
        xs = np.linspace(0.0, 0.2, 51)
        a = travel_time(xs, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, 0.0)
        b = travel_time(xs, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY)
        assert (b - a) == pytest.approx(np.full_like(xs, DELAY))

    def test_higher_permittivity_arrives_later(self):
        ax = apex_source_x(X_TARGET, OFFSET)
        times = [travel_time(ax, X_TARGET, DEPTH, e, OFFSET, RADIUS, DELAY) for e in (2, 4, 6, 9)]
        assert times == sorted(times)

    def test_accepts_scalar_and_array(self):
        one = travel_time(0.04, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY)
        many = travel_time(np.array([0.04, 0.06]), X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY)
        assert np.ndim(one) == 0
        assert many.shape == (2,)
        assert many[0] == pytest.approx(one)

    def test_bad_geometry_rejected(self):
        with pytest.raises(ValueError, match="depth must be positive"):
            travel_time(0.04, X_TARGET, 0.0, EPS_R)
        with pytest.raises(ValueError, match="cannot be negative"):
            travel_time(0.04, X_TARGET, DEPTH, EPS_R, radius=-0.01)
        with pytest.raises(ValueError, match="reaches the surface"):
            travel_time(0.04, X_TARGET, 0.05, EPS_R, radius=0.06)


class TestAgainstMeasuredRun:
    """Checked against a real gprMax 4.0.0b0 run of cylinder_Bscan_2D.in.

    Measured peak arrival of the cylinder reflection was 2.486 ns at source
    x=0.040 and 2.227 ns at source x=0.098. The analytic model is expected to
    be close but not exact: it predicts the geometric arrival of an impulse,
    while the measurement is the peak of a finite-bandwidth wavelet after
    propagation through a dispersive 2D FDTD grid.
    """

    MEASURED = {0.040: 2.486, 0.098: 2.227}

    def test_within_four_percent_of_measurement(self):
        for x_s, observed in self.MEASURED.items():
            predicted = travel_time(x_s, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY)
            assert abs(predicted - observed) / observed < 0.04

    def test_predicted_shift_matches_measured_shift(self):
        predicted = travel_time(0.098, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY) - travel_time(
            0.040, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, DELAY
        )
        assert predicted == pytest.approx(2.227 - 2.486, abs=0.03)

    def test_omitting_the_delay_is_catastrophic(self):
        # The failure mode this module exists to prevent.
        no_delay = travel_time(0.040, X_TARGET, DEPTH, EPS_R, OFFSET, RADIUS, 0.0)
        assert abs(no_delay - 2.486) > 0.8

    def test_apex_lands_inside_the_example_time_window(self):
        assert apex_time(DEPTH, EPS_R, OFFSET, RADIUS, DELAY) < 3.005


class TestPermittivityFromApex:
    def test_round_trips_the_forward_model(self):
        t = apex_time(DEPTH, EPS_R, OFFSET, RADIUS, DELAY)
        eps, v = permittivity_from_apex(t, DEPTH, OFFSET, RADIUS, DELAY)
        assert eps == pytest.approx(EPS_R)
        assert v == pytest.approx(velocity(EPS_R))

    def test_recovers_a_plausible_value_from_the_measured_apex(self):
        # The measured hyperbola turns over near 2.23 ns. With the true depth
        # the calculator should land near the modelled half_space value.
        eps, _ = permittivity_from_apex(2.23, DEPTH, OFFSET, RADIUS, DELAY)
        assert 4.5 < eps < 7.5

    def test_round_trips_across_a_range(self):
        for e in (1.5, 3.0, 6.0, 12.0, 20.0):
            t = apex_time(DEPTH, e, OFFSET, RADIUS, DELAY)
            assert permittivity_from_apex(t, DEPTH, OFFSET, RADIUS, DELAY)[0] == pytest.approx(e)

    def test_apex_before_the_source_delay_rejected(self):
        with pytest.raises(ValueError, match="before the source delay"):
            permittivity_from_apex(0.5, DEPTH, OFFSET, RADIUS, DELAY)

    def test_superluminal_result_rejected(self):
        # Too early for the path length: physically impossible, not eps_r < 1.
        with pytest.raises(ValueError, match="faster than light"):
            permittivity_from_apex(DELAY + 0.05, DEPTH, OFFSET, RADIUS, DELAY)

    def test_bad_depth_rejected(self):
        with pytest.raises(ValueError, match="depth must be positive"):
            permittivity_from_apex(2.2, 0.0, OFFSET, RADIUS, DELAY)
