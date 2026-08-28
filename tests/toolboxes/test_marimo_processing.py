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

"""Tests for processing.py gain functions and background removal."""

import numpy as np
import pytest

from toolboxes.Marimo.processing import (
    GAIN_KINDS,
    apply_gain,
    fft_spectrum,
    gain_curve,
    gain_label,
    remove_mean_trace,
    spectrum_view_limit,
    subtract_traces,
)

DT = 4.717308673499368e-12
ITERATIONS = 100
N_TRACES = 7  # deliberately not equal to ITERATIONS: a transposed matrix
# would raise instead of silently producing a plausible-looking result


def _time_ns(iterations=ITERATIONS):
    return np.arange(iterations) * DT * 1e9


def _trace(iterations=ITERATIONS, amplitude=1.0):
    return np.sin(np.linspace(0, 6.28, iterations)) * amplitude


def _matrix(iterations=ITERATIONS, n_traces=N_TRACES):
    return np.column_stack([_trace(iterations, amplitude=j + 1) for j in range(n_traces)])


class TestGainCurve:
    def test_none_is_identity(self):
        curve = gain_curve(_time_ns(), "none")
        assert np.all(curve == 1.0)

    def test_curve_length_matches_time_axis(self):
        for kind in GAIN_KINDS:
            curve = gain_curve(_time_ns(), kind, factor=2.0)
            assert curve.shape == (ITERATIONS,)

    def test_constant_is_flat(self):
        curve = gain_curve(_time_ns(), "constant", factor=3.0)
        assert np.all(curve == 3.0)

    def test_linear_starts_at_one_and_rises(self):
        t = _time_ns()
        curve = gain_curve(t, "linear", factor=2.0)
        assert curve[0] == pytest.approx(1.0)
        assert curve[-1] == pytest.approx(1.0 + 2.0 * t[-1])
        assert np.all(np.diff(curve) >= 0)

    def test_exponential_matches_closed_form(self):
        t = _time_ns()
        curve = gain_curve(t, "exponential", factor=1.5)
        assert curve == pytest.approx(np.exp(1.5 * t))

    def test_db_twenty_per_ns_is_ten_times_after_one_ns(self):
        t = np.array([0.0, 1.0, 2.0])
        curve = gain_curve(t, "db", factor=20.0)
        assert curve == pytest.approx([1.0, 10.0, 100.0])

    def test_start_ns_leaves_earlier_samples_untouched(self):
        t = np.linspace(0.0, 3.0, 61)
        curve = gain_curve(t, "exponential", factor=2.0, start_ns=1.0)
        assert np.all(curve[t < 1.0] == 1.0)
        assert curve[t == 1.0] == pytest.approx(1.0)
        assert np.all(curve[t > 1.0] > 1.0)

    def test_max_gain_clamps(self):
        curve = gain_curve(np.linspace(0.0, 3.0, 61), "exponential", factor=5.0, max_gain=50.0)
        assert curve.max() == pytest.approx(50.0)

    def test_curve_never_goes_negative(self):
        # A negative linear factor decays past zero without the clip, which
        # would flip trace polarity partway down and fake a reflection.
        curve = gain_curve(np.linspace(0.0, 3.0, 61), "linear", factor=-10.0)
        assert np.all(curve >= 0.0)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown gain kind"):
            gain_curve(_time_ns(), "agc")

    def test_two_dimensional_time_axis_rejected(self):
        with pytest.raises(ValueError, match="must be 1D"):
            gain_curve(np.zeros((10, 2)), "linear")


class TestApplyGain:
    def test_trace_scaled_by_curve(self):
        t = _time_ns()
        trace = _trace()
        gained, curve = apply_gain(trace, t, "exponential", factor=2.0)
        assert gained == pytest.approx(trace * curve)

    def test_matrix_keeps_shape(self):
        matrix = _matrix()
        gained, curve = apply_gain(matrix, _time_ns(), "linear", factor=1.0)
        assert gained.shape == (ITERATIONS, N_TRACES)
        assert curve.shape == (ITERATIONS,)

    def test_matrix_column_matches_single_trace(self):
        # Antonis's composability ask: gain applied to one A-scan and the
        # same gain broadcast across a B-scan must be the same operation.
        t = _time_ns()
        matrix = _matrix()
        gained_matrix, _ = apply_gain(matrix, t, "db", factor=6.0, start_ns=0.5)
        for j in range(N_TRACES):
            gained_trace, _ = apply_gain(matrix[:, j], t, "db", factor=6.0, start_ns=0.5)
            assert gained_matrix[:, j] == pytest.approx(gained_trace)

    def test_gain_applied_down_time_not_across_traces(self):
        # Pins the broadcast axis. stack_traces() returns (n_samples,
        # n_traces); multiplying by a bare (n_samples,) curve would broadcast
        # against the trace axis instead and silently scale the wrong way.
        t = np.arange(4, dtype=float)
        matrix = np.ones((4, 3))
        gained, _ = apply_gain(matrix, t, "linear", factor=1.0)
        assert gained[:, 0] == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert np.all(gained[0, :] == 1.0)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="time axis has"):
            apply_gain(_trace(iterations=50), _time_ns(100), "linear")

    def test_transposed_matrix_rejected(self):
        with pytest.raises(ValueError, match="n_samples, n_traces"):
            apply_gain(_matrix().T, _time_ns(), "linear")

    def test_three_dimensional_data_rejected(self):
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            apply_gain(np.zeros((ITERATIONS, 2, 2)), _time_ns(), "linear")

    def test_none_returns_data_unchanged(self):
        matrix = _matrix()
        gained, curve = apply_gain(matrix, _time_ns(), "none")
        assert gained == pytest.approx(matrix)
        assert np.all(curve == 1.0)

    def test_input_not_mutated(self):
        matrix = _matrix()
        original = matrix.copy()
        apply_gain(matrix, _time_ns(), "exponential", factor=3.0)
        assert matrix == pytest.approx(original)

    def test_integer_input_does_not_truncate(self):
        data = np.ones(4, dtype=int)
        gained, _ = apply_gain(data, np.arange(4, dtype=float), "linear", factor=0.5)
        assert gained == pytest.approx([1.0, 1.5, 2.0, 2.5])


class TestGainLabel:
    def test_none(self):
        assert gain_label("none") == "no gain"

    def test_constant(self):
        assert gain_label("constant", 2.5) == "constant gain x2.5"

    def test_carries_units_and_start(self):
        assert gain_label("db", 6.0, start_ns=1.2) == "dB gain 6 dB per ns from 1.2 ns"

    def test_carries_clamp(self):
        label = gain_label("exponential", 2.0, max_gain=50.0)
        assert label == "exponential gain 2 per ns, clamped at x50"

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown gain kind"):
            gain_label("agc")


class TestRemoveMeanTrace:
    def test_identical_traces_cancel_to_zero(self):
        matrix = np.column_stack([_trace()] * N_TRACES)
        result, mean_trace = remove_mean_trace(matrix)
        assert result == pytest.approx(np.zeros_like(matrix))
        assert mean_trace == pytest.approx(_trace())

    def test_stationary_component_removed_target_kept(self):
        direct = _trace()
        matrix = np.column_stack([direct.copy() for _ in range(N_TRACES)])
        target = np.zeros(ITERATIONS)
        target[60] = 5.0
        matrix[:, 3] += target
        result, _ = remove_mean_trace(matrix)
        assert result[60, 3] == pytest.approx(5.0 * (1 - 1 / N_TRACES))
        assert np.argmax(np.abs(result[:, 3])) == 60

    def test_shape_preserved(self):
        result, mean_trace = remove_mean_trace(_matrix())
        assert result.shape == (ITERATIONS, N_TRACES)
        assert mean_trace.shape == (ITERATIONS,)

    def test_one_dimensional_input_rejected(self):
        with pytest.raises(ValueError, match="must be 2D"):
            remove_mean_trace(_trace())

    def test_input_not_mutated(self):
        matrix = _matrix()
        original = matrix.copy()
        remove_mean_trace(matrix)
        assert matrix == pytest.approx(original)


class TestPowerAndSecGain:
    def test_power_floored_at_one(self):
        # Bare t**b is 0 at t=0 and <1 below 1 ns, which would blank the start
        # of a 3 ns gprMax window instead of amplifying anything.
        t = np.linspace(0.0, 3.0, 61)
        curve = gain_curve(t, "power", power=1.0)
        assert curve[0] == pytest.approx(1.0)
        assert np.all(curve >= 1.0)

    def test_power_exact_where_it_amplifies(self):
        t = np.linspace(0.0, 4.0, 81)
        curve = gain_curve(t, "power", power=2.0)
        above = t > 1.0
        assert curve[above] == pytest.approx(t[above] ** 2.0)

    def test_power_monotonic(self):
        curve = gain_curve(np.linspace(0.0, 5.0, 101), "power", power=1.5)
        assert np.all(np.diff(curve) >= 0)

    def test_sec_is_product_of_its_two_terms(self):
        # SEC is exp(a*t) * t**b. Pin the composition so the two halves can't
        # drift apart from the standalone kinds.
        t = np.linspace(0.0, 3.0, 61)
        expo = gain_curve(t, "exponential", factor=0.4)
        powr = gain_curve(t, "power", power=1.5)
        sec = gain_curve(t, "sec", factor=0.4, power=1.5)
        assert sec == pytest.approx(expo * powr)

    def test_sec_with_zero_exponent_matches_exponential(self):
        t = np.linspace(0.0, 3.0, 61)
        sec = gain_curve(t, "sec", factor=0.4, power=0.0)
        expo = gain_curve(t, "exponential", factor=0.4)
        assert sec == pytest.approx(expo)

    def test_sec_honours_start_and_clamp(self):
        t = np.linspace(0.0, 5.0, 101)
        curve = gain_curve(t, "sec", factor=1.0, power=2.0, start_ns=1.0, max_gain=20.0)
        assert np.all(curve[t < 1.0] == 1.0)
        assert curve.max() == pytest.approx(20.0)

    def test_power_applies_across_matrix(self):
        matrix = _matrix()
        t = _time_ns()
        gained, curve = apply_gain(matrix, t, "sec", factor=0.5, power=1.0)
        for j in range(N_TRACES):
            assert gained[:, j] == pytest.approx(matrix[:, j] * curve)

    def test_every_kind_has_ui_metadata(self):
        for kind, meta in GAIN_KINDS.items():
            assert set(meta) == {"label", "factor_unit", "uses_power"}
            gain_curve(_time_ns(), kind, factor=1.0, power=1.0)


class TestGainLabelExtra:
    def test_power(self):
        assert gain_label("power", power=1.5) == "power gain t^1.5"

    def test_sec(self):
        assert gain_label("sec", 0.5, 1.5) == "SEC gain a=0.5 per ns, b=1.5"


class TestMovingWindowBackgroundRemoval:
    def test_window_removes_local_background(self):
        matrix = np.column_stack([_trace()] * 9)
        result, background = remove_mean_trace(matrix, window=3)
        assert result == pytest.approx(np.zeros_like(matrix))
        assert background.shape == matrix.shape

    def test_window_preserves_a_dipping_event_global_mean_would_smear(self):
        # A drifting background is exactly the case the moving window exists
        # for: the global mean leaves residual, the window does not.
        n_traces = 21
        drift = np.linspace(0.0, 4.0, n_traces)
        matrix = np.zeros((ITERATIONS, n_traces))
        matrix[30, :] = drift
        global_result, _ = remove_mean_trace(matrix)
        window_result, _ = remove_mean_trace(matrix, window=3)
        assert np.abs(window_result[30]).max() < np.abs(global_result[30]).max()

    def test_window_wider_than_profile_matches_global_mean(self):
        matrix = _matrix()
        wide, _ = remove_mean_trace(matrix, window=N_TRACES * 10)
        glob, _ = remove_mean_trace(matrix)
        assert wide == pytest.approx(glob)

    def test_window_of_one_is_a_no_op_removing_everything(self):
        matrix = _matrix()
        result, background = remove_mean_trace(matrix, window=1)
        assert result == pytest.approx(np.zeros_like(matrix))
        assert background == pytest.approx(matrix)

    def test_window_below_one_rejected(self):
        with pytest.raises(ValueError, match="at least 1 trace"):
            remove_mean_trace(_matrix(), window=0)

    def test_edge_columns_use_full_width_window(self):
        # Clamped, not zero-padded: an edge trace must not be biased toward
        # zero by averaging over a half-empty window.
        matrix = np.tile(np.arange(5, dtype=float), (ITERATIONS, 1))
        _, background = remove_mean_trace(matrix, window=3)
        assert background[0, 0] == pytest.approx(1.0)
        assert background[0, -1] == pytest.approx(3.0)


class TestFFTSpectrum:
    """Exercises the wrapper around gprMax's own fft_power.

    Skipped rather than failed when gprMax isn't importable, so the rest of
    this suite still runs on a checkout without a built extension.
    """

    # 637 samples at this dt is the real cylinder_Ascan_2D window, 3.005 ns.
    # ITERATIONS (100) is too short to hold one period of a 1.5 GHz Ricker,
    # which would make any frequency assertion meaningless.
    N_FFT = 637

    @staticmethod
    def _ricker(n=637, dt=DT, f0=1.5e9):
        t = np.arange(n) * dt
        tau = np.pi * f0 * (t - 1.0 / f0)
        return (1.0 - 2.0 * tau**2) * np.exp(-(tau**2))

    def test_positive_frequencies_only(self):
        pytest.importorskip("gprMax.utilities.utilities")
        freqs, power, _ = fft_spectrum(self._ricker(), DT)
        assert np.all(freqs >= 0)
        assert freqs.size == (self.N_FFT + 1) // 2
        assert power.size == freqs.size

    def test_full_axis_includes_negative_frequencies(self):
        # Pins why positive_only exists: the raw fft_power axis is mirrored.
        pytest.importorskip("gprMax.utilities.utilities")
        freqs, _, _ = fft_spectrum(self._ricker(), DT, positive_only=False)
        assert freqs.size == self.N_FFT
        assert np.any(freqs < 0)

    def test_peak_frequency_matches_source(self):
        pytest.importorskip("gprMax.utilities.utilities")
        freqs, power, _ = fft_spectrum(self._ricker(f0=1.5e9), DT)
        # A 3 ns window gives 0.333 GHz bins, so the best possible answer is
        # the nearest bin to 1.5 GHz, not 1.5 GHz itself.
        bin_hz = float(freqs[1] - freqs[0])
        assert abs(freqs[int(np.argmax(power))] - 1.5e9) <= bin_hz

    def test_power_is_relative_to_own_peak(self):
        pytest.importorskip("gprMax.utilities.utilities")
        _, power, _ = fft_spectrum(self._ricker(), DT)
        assert power.max() == pytest.approx(0.0, abs=1e-9)

    def test_peak_db_recovers_amplitude_difference(self):
        # The whole reason peak_db is returned: fft_power alone makes a
        # 1000x-larger trace look identical to the original.
        pytest.importorskip("gprMax.utilities.utilities")
        quiet = self._ricker()
        loud = quiet * 1000.0
        _, p_quiet, peak_quiet = fft_spectrum(quiet, DT)
        _, p_loud, peak_loud = fft_spectrum(loud, DT)
        assert p_quiet == pytest.approx(p_loud)
        assert peak_loud - peak_quiet == pytest.approx(60.0, abs=0.01)

    def test_flat_trace_reports_none_instead_of_a_fake_spectrum(self):
        # Ex in a 2D TMz model has no finite normalised spectrum.
        pytest.importorskip("gprMax.utilities.utilities")
        freqs, power, peak_db = fft_spectrum(np.zeros(self.N_FFT), DT)
        assert peak_db is None
        assert freqs.size == (self.N_FFT + 1) // 2
        assert np.all(power == 0.0)

    def test_positive_half_keeps_all_nonnegative_bins_for_odd_length(self):
        pytest.importorskip("gprMax.utilities.utilities")
        freqs, _, _ = fft_spectrum(np.arange(9, dtype=float), DT)
        assert freqs.size == 5
        assert np.all(freqs >= 0.0)

    def test_two_dimensional_input_rejected(self):
        with pytest.raises(ValueError, match="must be 1D"):
            fft_spectrum(np.zeros((10, 2)), DT)

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            fft_spectrum(np.array([]), DT)


class TestSpectrumViewLimit:
    def test_limit_is_a_multiple_of_the_peak(self):
        freqs = np.linspace(0.0, 100e9, 500)
        power = np.full(500, -60.0)
        power[10] = 0.0  # peak at freqs[10]
        limit = spectrum_view_limit(freqs, power, multiple=4.0)
        assert limit == pytest.approx(4.0 * freqs[10])

    def test_clamped_to_available_maximum(self):
        # plot_Ascan.py's index-based freqmaxpower*4 overshoots here.
        freqs = np.linspace(0.0, 100e9, 500)
        power = np.full(500, -60.0)
        power[400] = 0.0
        limit = spectrum_view_limit(freqs, power, multiple=4.0)
        assert limit == pytest.approx(freqs.max())

    def test_dc_peak_falls_back_to_full_range(self):
        freqs = np.linspace(0.0, 100e9, 500)
        power = np.full(500, -60.0)
        power[0] = 0.0
        assert spectrum_view_limit(freqs, power) == pytest.approx(freqs.max())

    def test_empty_input_returns_zero(self):
        assert spectrum_view_limit(np.array([]), np.array([])) == 0.0


class TestSubtractTraces:
    def test_identical_runs_cancel(self):
        t = _trace()
        assert subtract_traces(t, t, DT, DT) == pytest.approx(np.zeros_like(t))

    def test_isolates_the_difference(self):
        # A target run is the target-free background run plus the target response.
        background = _trace()
        target_response = np.zeros(ITERATIONS)
        target_response[60] = 5.0
        result = subtract_traces(background + target_response, background, DT, DT)
        assert result == pytest.approx(target_response)

    def test_works_on_matrices(self):
        a, b = _matrix(), _matrix()
        result = subtract_traces(a, b, DT, DT)
        assert result.shape == (ITERATIONS, N_TRACES)
        assert result == pytest.approx(np.zeros_like(a))

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            subtract_traces(_trace(100), _trace(50), DT, DT)

    def test_equal_length_but_different_dt_rejected(self):
        # The failure this guard exists for: two runs with the same iteration
        # count but different time steps line up index by index while
        # representing different instants.
        with pytest.raises(ValueError, match="time steps differ"):
            subtract_traces(_trace(), _trace(), DT, DT * 1.5)

    def test_negligible_dt_difference_accepted(self):
        # Float round-trips through HDF5 should not trip the guard.
        subtract_traces(_trace(), _trace(), DT, DT * (1 + 1e-12))

    def test_missing_dt_skips_the_time_check(self):
        # Callers without dt still get the shape check.
        subtract_traces(_trace(), _trace())

    def test_inputs_not_mutated(self):
        a, b = _trace(), _trace(amplitude=2.0)
        a0, b0 = a.copy(), b.copy()
        subtract_traces(a, b, DT, DT)
        assert a == pytest.approx(a0)
        assert b == pytest.approx(b0)

    def test_integer_input_does_not_truncate(self):
        result = subtract_traces(np.array([3, 3]), np.array([1, 2]))
        assert result == pytest.approx([2.0, 1.0])
