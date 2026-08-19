"""Tests for processing.py gain functions and background removal."""

import numpy as np
import pytest

from toolboxes.Marimo.processing import (
    GAIN_KINDS,
    apply_gain,
    gain_curve,
    gain_label,
    remove_mean_trace,
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
        assert gain_label("db", 6.0, start_ns=1.2) == "db gain 6 dB per ns from 1.2 ns"

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
