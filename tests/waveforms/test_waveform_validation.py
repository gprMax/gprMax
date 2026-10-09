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

"""Direct Waveform validation: rejected bad inputs, validate-once sampling."""

import numpy as np
import pytest

from gprMax.waveforms import Waveform

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

    def test_impulse_frequency_is_required_despite_being_unused(self, make_waveform):
        """Matches the #waveform command and its documented "use 1" convention."""
        w = make_waveform("impulse", freq=0)
        with pytest.raises(ValueError, match="finite excitation frequency greater than zero"):
            w.calculate_value(0.0, dt=1e-12)

    def test_direct_coefficients_reject_zero_frequency(self, make_waveform):
        w = make_waveform("gaussian", freq=0)
        with pytest.raises(ValueError, match="finite excitation frequency greater than zero"):
            w.calculate_coefficients()


class TestUnknownType:
    def test_unknown_type_fails_on_evaluation(self, make_waveform):
        w = make_waveform("notarealtype", freq=1e9)
        with pytest.raises(ValueError, match="Unknown waveform type"):
            w.calculate_value(1e-9, dt=1e-12)

    def test_missing_type_fails_on_evaluation(self, make_waveform):
        w = make_waveform("gaussian", freq=1e9)
        w.type = None
        with pytest.raises(ValueError, match="Unknown waveform type"):
            w.calculate_value(1e-9, dt=1e-12)


class TestAmplitude:
    @pytest.mark.parametrize("amp", [np.inf, -np.inf, np.nan, None])
    def test_rejects_non_finite_amplitude(self, make_waveform, amp):
        w = make_waveform("gaussian", freq=1e9, amp=1.0)
        w.amp = amp
        with pytest.raises(ValueError, match="finite amplitude"):
            w.calculate_value(1e-9, dt=1e-12)

    def test_zero_amplitude_remains_valid(self, make_waveform):
        """Zero amplitude marks passive ports; it must not be rejected."""
        w = make_waveform("gaussian", freq=1e9, amp=0.0)
        assert w.calculate_value(1.0 / w.freq, dt=1e-12) == 0


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

    def test_userfunc_output_checked_on_every_sample(self, make_waveform):
        """Late userfunc failures still raise after validation."""
        calls = []

        def sometimes_bad(t):
            calls.append(t)
            return np.inf if len(calls) > 2 else 1.0

        w = make_waveform("user")
        w.userfunc = sometimes_bad
        assert w.calculate_value(0.0, dt=1e-12) == pytest.approx(1.0)
        assert w.calculate_value(1e-12, dt=1e-12) == pytest.approx(1.0)
        with pytest.raises(ValueError, match="non-finite value"):
            w.calculate_value(2e-12, dt=1e-12)


class TestValidateOnce:
    def test_validate_refreshes_coefficients(self, make_waveform):
        w = make_waveform("gaussian", freq=1e9)
        w.validate()
        assert w.chi == pytest.approx(1e-9)
        assert w.calculate_value(1e-9, dt=1e-12) == pytest.approx(1.0)

    def test_coefficients_computed_once_per_bulk_run(self, make_waveform, monkeypatch):
        """Bulk sampling must not re-validate every sample."""
        calls = []
        original = Waveform.calculate_coefficients

        def counting(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(Waveform, "calculate_coefficients", counting)
        w = make_waveform("gaussian", freq=1e9)
        for iteration in range(200):
            w.calculate_value(iteration * 1e-12, dt=1e-12)
        assert len(calls) == 1

    def test_modified_parameters_revalidate_on_next_sample(self, make_waveform, monkeypatch):
        calls = []
        original = Waveform.calculate_coefficients

        def counting(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(Waveform, "calculate_coefficients", counting)
        w = make_waveform("gaussian", freq=1e9)
        w.calculate_value(0.0, dt=1e-12)
        assert len(calls) == 1
        w.freq = 2e9
        assert w.calculate_value(0.5e-9, dt=1e-12) == pytest.approx(1.0)
        assert len(calls) == 2

    def test_explicit_validate_before_bulk_loop_covers_all_samples(self, make_waveform, monkeypatch):
        """The documented bulk pattern: validate once, then sample freely."""
        calls = []
        original = Waveform.calculate_coefficients

        def counting(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(Waveform, "calculate_coefficients", counting)
        w = make_waveform("ricker", freq=1e9)
        w.validate()
        values = [w.calculate_value(n * 1e-12, dt=1e-12) for n in range(200)]
        assert len(calls) == 1
        assert all(np.isfinite(values))


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
