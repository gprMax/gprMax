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

"""Tests for waveform input validation.

Waveforms can be created directly through the Python API, bypassing the
checks performed by the #waveform command. These tests ensure invalid
inputs raise clear errors during waveform evaluation.
"""

import numpy as np
import pytest

from gprMax.toolboxes.Plotting.plot_source_wave import check_timewindow

pytestmark = pytest.mark.unit

BUILTIN_TYPES = [
    "gaussian",
    "gauspulse",
    "gaussiandot",
    "gaussiandotnorm",
    "gaussiandotdot",
    "gaussiandotdotnorm",
    "gaussianprime",
    "gaussiandoubleprime",
    "ricker",
    "sine",
    "contsine",
    "impulse",
]


class TestBuiltinFrequency:
    @pytest.mark.parametrize("wave_type", BUILTIN_TYPES)
    @pytest.mark.parametrize("freq", [0, -1e9, np.inf, -np.inf, np.nan, None])
    def test_rejects_non_positive_or_non_finite_frequency(self, make_waveform, wave_type, freq):
        w = make_waveform(wave_type, freq=1e9)
        w.freq = freq
        with pytest.raises(ValueError, match="finite excitation frequency greater than zero"):
            w.calculate_value(1e-9, dt=1e-12)

    @pytest.mark.parametrize("wave_type", ["gaussian", "ricker", "gauspulse", "sine", "impulse"])
    def test_coefficients_reject_zero_frequency_directly(self, make_waveform, wave_type):
        w = make_waveform(wave_type, freq=0)
        with pytest.raises(ValueError, match="finite excitation frequency greater than zero"):
            w.calculate_coefficients()

    def test_impulse_frequency_is_required_despite_being_unused(self, make_waveform):
        """Matches the #waveform command and its documented "use 1" convention."""
        w = make_waveform("impulse", freq=0)
        with pytest.raises(ValueError, match="finite excitation frequency greater than zero"):
            w.calculate_value(0.0, dt=1e-12)


class TestUnknownType:
    def test_unknown_type_fails_before_sampling(self, make_waveform):
        w = make_waveform("notarealtype", freq=1e9)
        with pytest.raises(ValueError, match="Unknown waveform type"):
            w.calculate_value(1e-9, dt=1e-12)

    def test_unknown_type_coefficients_fail_fast(self, make_waveform):
        w = make_waveform("notarealtype", freq=1e9)
        with pytest.raises(ValueError, match="Unknown waveform type"):
            w.calculate_coefficients()

    def test_missing_type_fails_with_unknown_type(self, make_waveform):
        w = make_waveform("gaussian", freq=1e9)
        w.type = None
        with pytest.raises(ValueError, match="Unknown waveform type"):
            w.calculate_value(1e-9, dt=1e-12)


class TestAmplitude:
    @pytest.mark.parametrize("amp", [np.inf, -np.inf, np.nan, None])
    def test_builtin_rejects_non_finite_amplitude(self, make_waveform, amp):
        w = make_waveform("gaussian", freq=1e9, amp=1.0)
        w.amp = amp
        with pytest.raises(ValueError, match="finite amplitude"):
            w.calculate_value(1e-9, dt=1e-12)

    @pytest.mark.parametrize("amp", [np.inf, np.nan])
    def test_user_rejects_non_finite_amplitude(self, make_waveform, amp):
        w = make_waveform("user", amp=1.0)
        w.userfunc = lambda t: 1.0
        w.amp = amp
        with pytest.raises(ValueError, match="finite amplitude"):
            w.calculate_value(1e-9, dt=1e-12)

    def test_zero_amplitude_remains_valid(self, make_waveform):
        """Zero amplitude marks passive ports; it must not be rejected."""
        w = make_waveform("gaussian", freq=1e9, amp=0.0)
        assert w.calculate_value(1.0 / w.freq, dt=1e-12) == 0


class TestSampleTime:
    @pytest.mark.parametrize("time", [np.inf, -np.inf, np.nan])
    def test_rejects_non_finite_sample_time(self, make_waveform, time):
        w = make_waveform("gaussian", freq=1e9)
        with pytest.raises(ValueError, match="finite sample time"):
            w.calculate_value(time, dt=1e-12)

    @pytest.mark.parametrize("dt", [0, -1e-12, np.inf, np.nan, None])
    def test_rejects_non_positive_or_non_finite_timestep(self, make_waveform, dt):
        w = make_waveform("gaussian", freq=1e9)
        with pytest.raises(ValueError, match="finite timestep greater than zero"):
            w.calculate_value(1e-9, dt=dt)

    @pytest.mark.parametrize("dt", [0, -1e-12, np.inf, np.nan])
    def test_impulse_rejects_bad_timestep(self, make_waveform, dt):
        w = make_waveform("impulse", freq=1e9)
        with pytest.raises(ValueError, match="finite timestep greater than zero"):
            w.calculate_value(0.0, dt=dt)


class TestUserWaveform:
    def test_missing_userfunc_fails_fast(self, make_waveform):
        w = make_waveform("user")
        with pytest.raises(ValueError, match="callable"):
            w.calculate_value(1e-9, dt=1e-12)

    def test_non_callable_userfunc_fails_fast(self, make_waveform):
        w = make_waveform("user")
        w.userfunc = "not-callable"
        with pytest.raises(ValueError, match="callable"):
            w.calculate_value(1e-9, dt=1e-12)

    def test_userfunc_returning_non_numeric_reports_numeric_value(self, make_waveform):
        w = make_waveform("user")
        w.userfunc = lambda t: "bad"
        with pytest.raises(ValueError, match="numeric value"):
            w.calculate_value(1e-9, dt=1e-12)

    def test_userfunc_returning_none_reports_numeric_value(self, make_waveform):
        w = make_waveform("user")
        w.userfunc = lambda t: None
        with pytest.raises(ValueError, match="numeric value"):
            w.calculate_value(1e-9, dt=1e-12)

    def test_userfunc_returning_non_finite_still_rejected(self, make_waveform):
        w = make_waveform("user")
        w.userfunc = lambda t: np.inf
        with pytest.raises(ValueError, match="non-finite value"):
            w.calculate_value(1e-9, dt=1e-12)


class TestValidBehaviourPreserved:
    def test_gaussian_still_peaks_at_chi(self, make_waveform):
        w = make_waveform("gaussian", freq=1e9)
        assert w.calculate_value(1.0 / w.freq, dt=1e-12) == pytest.approx(1.0)

    def test_impulse_gating_unchanged(self, make_waveform):
        w = make_waveform("impulse", freq=1e9)
        assert w.calculate_value(0.0, dt=1e-12) == 1
        assert w.calculate_value(2e-12, dt=1e-12) == 0

    def test_user_scaling_unchanged(self, make_waveform):
        w = make_waveform("user", amp=3.0)
        w.userfunc = lambda t: 4.0
        assert w.calculate_value(0.5e-9, dt=1e-12) == pytest.approx(12.0)


class TestCheckTimewindow:
    @pytest.mark.parametrize("dt", [0, -1e-12, np.inf, -np.inf, np.nan])
    def test_rejects_non_positive_or_non_finite_dt(self, dt):
        with pytest.raises(ValueError, match="Time step must be finite and greater than zero"):
            check_timewindow("6e-9", dt)

    def test_valid_window_still_accepted(self):
        timewindow, iterations = check_timewindow("6e-9", 1.926e-12)
        assert iterations > 1
        assert timewindow > 0
