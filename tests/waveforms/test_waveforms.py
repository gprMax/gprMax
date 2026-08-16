import numpy as np
import pytest


class TestGaussian:
    def test_peaks_at_chi_with_unit_value(self, make_waveform):
        w = make_waveform("gaussian", freq=1e9)
        chi = 1.0 / w.freq
        assert w.calculate_value(chi, dt=1e-12) == pytest.approx(1.0)

    def test_symmetric_around_chi(self, make_waveform):
        w = make_waveform("gaussian", freq=1e9)
        chi = 1.0 / w.freq
        delta = 0.2 * chi
        left = w.calculate_value(chi - delta, dt=1e-12)
        right = w.calculate_value(chi + delta, dt=1e-12)
        assert left == pytest.approx(right)


class TestGaussianDot:
    def test_zero_at_chi(self, make_waveform):
        w = make_waveform("gaussiandot", freq=1e9)
        chi = 1.0 / w.freq
        assert w.calculate_value(chi, dt=1e-12) == pytest.approx(0.0, abs=1e-15)

    def test_antisymmetric_around_chi(self, make_waveform):
        w = make_waveform("gaussiandot", freq=1e9)
        chi = 1.0 / w.freq
        delta = 0.2 * chi
        left = w.calculate_value(chi - delta, dt=1e-12)
        right = w.calculate_value(chi + delta, dt=1e-12)
        assert left == pytest.approx(-right)


class TestGaussianDotNorm:
    def test_peaks_at_unit_magnitude(self, make_waveform):
        """Peak magnitude is normalised to 1 — definition of the 'norm' suffix."""
        w = make_waveform("gaussiandotnorm", freq=1e9)
        times = np.linspace(0, 2.0 / w.freq, 5000)
        values = np.array([w.calculate_value(t, dt=1e-12) for t in times])
        assert np.max(np.abs(values)) == pytest.approx(1.0)


class TestGaussianDotDotNorm:
    def test_trough_at_chi(self, make_waveform):
        """The Mexican hat's central trough sits at -1 by normalisation."""
        w = make_waveform("gaussiandotdotnorm", freq=1e9)
        chi = np.sqrt(2) / w.freq
        assert w.calculate_value(chi, dt=1e-12) == pytest.approx(-1.0)


class TestRicker:
    def test_is_negated_gaussiandotdotnorm(self, make_waveform):
        """Ricker is defined as the negative of gaussiandotdotnorm."""
        ricker = make_waveform("ricker", freq=1e9)
        norm = make_waveform("gaussiandotdotnorm", freq=1e9)
        for t in [0.5e-9, 1e-9, 1.5e-9, 2e-9]:
            assert ricker.calculate_value(t, dt=1e-12) == pytest.approx(
                -norm.calculate_value(t, dt=1e-12)
            )


class TestPrimeVsDot:
    """`*prime` and `*dot` variants: are they aliases or distinct waveforms?

    The source comment in waveforms.py:45-53 says they SHOULD differ
    (prime = derivative of base gaussian, dot = user-specified centre freq).
    The code partially matches that contract; see each test for details.
    """

    def test_gaussianprime_matches_gaussiandot_today(self, make_waveform):
        """Currently identical: both land in the same `elif` branch at
        waveforms.py:101 AND share gaussian-family coefficients at
        waveforms.py:70-78.

        This contradicts the source comment at waveforms.py:45-53, which says
        gaussianprime is the *derivative* of the base gaussian (so the centre
        freq should drift up) while gaussiandot has a user-specified centre
        freq. Likely a missed entry in calculate_coefficients — worth raising
        with mentors. This test pins the *current* behaviour; if the source
        is fixed, this test should flip to assert inequality.
        """
        a = make_waveform("gaussianprime", freq=1e9)
        b = make_waveform("gaussiandot", freq=1e9)
        for t in [0.3e-9, 0.7e-9, 1.0e-9, 1.5e-9, 2.0e-9]:
            assert a.calculate_value(t, dt=1e-12) == pytest.approx(b.calculate_value(t, dt=1e-12))

    def test_gaussiandoubleprime_differs_from_gaussiandotdot(self, make_waveform):
        """Distinct by design. Both share the formula at waveforms.py:110-114
        but use different coefficients:
            gaussiandoubleprime → chi=1/f,  zeta=2*pi^2*f^2  (gaussian family)
            gaussiandotdot      → chi=sqrt(2)/f, zeta=pi^2*f^2 (ricker family)
        This matches the source comment at waveforms.py:45-53.
        """
        a = make_waveform("gaussiandoubleprime", freq=1e9)
        b = make_waveform("gaussiandotdot", freq=1e9)
        va = a.calculate_value(1e-9, dt=1e-12)
        vb = b.calculate_value(1e-9, dt=1e-12)
        # Empirically they differ by ~20x. Anything > order-of-magnitude works.
        assert abs(va - vb) > max(abs(va), abs(vb)) * 0.5


class TestSine:
    def test_zero_at_origin(self, make_waveform):
        w = make_waveform("sine", freq=1e9)
        assert w.calculate_value(0.0, dt=1e-12) == pytest.approx(0.0, abs=1e-15)

    def test_unit_at_quarter_period(self, make_waveform):
        w = make_waveform("sine", freq=1e9)
        t = 1.0 / (4 * w.freq)
        assert w.calculate_value(t, dt=1e-12) == pytest.approx(1.0)

    def test_silenced_after_one_cycle(self, make_waveform):
        """Source truncates sine to silence once `time * freq > 1`."""
        w = make_waveform("sine", freq=1e9)
        t = 1.5 / w.freq
        assert w.calculate_value(t, dt=1e-12) == 0


class TestContSine:
    def test_zero_at_origin(self, make_waveform):
        w = make_waveform("contsine", freq=1e9)
        assert w.calculate_value(0.0, dt=1e-12) == pytest.approx(0.0, abs=1e-15)

    def test_ramp_saturates_to_pure_sine(self, make_waveform):
        """Beyond t = 4/freq the ramp clamps to 1, so contsine == sin(2πft)."""
        w = make_waveform("contsine", freq=1e9)
        t = 5.123 / w.freq
        expected = np.sin(2 * np.pi * w.freq * t)
        assert w.calculate_value(t, dt=1e-12) == pytest.approx(expected)


class TestImpulse:
    def test_unit_at_origin(self, make_waveform):
        w = make_waveform("impulse")
        assert w.calculate_value(0.0, dt=1e-12) == 1

    def test_unit_within_dt(self, make_waveform):
        w = make_waveform("impulse")
        dt = 1e-12
        assert w.calculate_value(dt / 2, dt=dt) == 1

    def test_zero_after_dt(self, make_waveform):
        w = make_waveform("impulse")
        dt = 1e-12
        assert w.calculate_value(2 * dt, dt=dt) == 0


class TestUser:
    def test_calls_userfunc_and_scales_by_amp(self, make_waveform):
        """`user` type delegates to self.userfunc(time) then multiplies by amp."""
        w = make_waveform("user", amp=3.0)
        w.userfunc = lambda t: 4.0
        assert w.calculate_value(0.5e-9, dt=1e-12) == pytest.approx(12.0)

    def test_passes_time_to_userfunc(self, make_waveform):
        captured = []
        w = make_waveform("user")

        def grab(t):
            captured.append(t)
            return 0.0

        w.userfunc = grab
        w.calculate_value(0.75e-9, dt=1e-12)
        assert captured == [0.75e-9]


class TestAmplitudeScaling:
    ALL_TYPES = [
        "gaussian",
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

    @pytest.mark.parametrize("wave_type", ALL_TYPES)
    def test_doubling_amp_doubles_output(self, make_waveform, wave_type):
        single = make_waveform(wave_type, freq=1e9, amp=1.0)
        double = make_waveform(wave_type, freq=1e9, amp=2.0)
        t = 0.5e-9
        assert double.calculate_value(t, dt=1e-12) == pytest.approx(
            2.0 * single.calculate_value(t, dt=1e-12)
        )

    @pytest.mark.parametrize("wave_type", ALL_TYPES)
    def test_zero_amp_zero_output(self, make_waveform, wave_type):
        """amp = 0 must zero every waveform type — linearity edge case."""
        w = make_waveform(wave_type, freq=1e9, amp=0.0)
        assert w.calculate_value(0.5e-9, dt=1e-12) == 0


class TestCausality:
    """Gaussian-family waveforms shift by `chi` so they start at ~0.

    Only normalised variants are checked — non-normalised types have
    large absolute peaks that make an absolute tolerance meaningless.
    """

    @pytest.mark.parametrize(
        "wave_type",
        ["gaussian", "gaussiandotnorm", "gaussiandotdotnorm", "ricker"],
    )
    def test_near_zero_at_origin(self, make_waveform, wave_type):
        w = make_waveform(wave_type, freq=1e9)
        assert abs(w.calculate_value(0.0, dt=1e-12)) < 1e-6


class TestErrors:
    def test_unknown_type_currently_raises_unbound_local(self, make_waveform):
        """Tripwire for the missing `else` branch in calculate_value.

        Today this falls through every if/elif so `ampvalue` is never
        assigned and the final `ampvalue *= self.amp` raises
        UnboundLocalError. A follow-up fix should make this a clean
        ValueError; update this test in the same PR.
        """
        w = make_waveform("notarealtype", freq=1e9)
        with pytest.raises(UnboundLocalError):
            w.calculate_value(1e-9, dt=1e-12)


pytestmark = pytest.mark.unit
