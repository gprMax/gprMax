# Unit Test for gprMax/waveforms.py
# Authors: Dang3Rsm

import pytest
import numpy as np
from gprMax.waveforms import Waveform

@pytest.mark.unit
class TestWaveforms:
    def test_waveform_base_init(self):
        """Verify base class hold state correctly."""
        w = Waveform()
        assert w.amp == 1
        assert w.type is None
        assert w.freq is None

    def test_gaussian_peak(self):
        """Verify Gaussian peak at 'amp' at t = chi."""
        w = Waveform()
        w.type = "gaussian"
        w.freq = 1e9
        w.amp = 2.5
        
        center_time = 1 / w.freq
        val = w.calculate_value(center_time, dt=1e-12)

        # exp(-zeta * 0^2) = 1. Value should be 1 * amp
        assert pytest.approx(val) == 2.5

    def test_gaussiandotnorm_peak(self):
        """Verify gaussiandotnorm reaches exactly 'amp' at peak."""
        w = Waveform()
        w.type, w.freq, w.amp = "gaussiandotnorm", 1e9, 1.0
        w.calculate_coefficients()
        
        # The peak of a first derivative of a Gaussian
        t_peak = w.chi - np.sqrt(1 / (2 * w.zeta))
        val = abs(w.calculate_value(t_peak, dt=1e-12))
        
        assert pytest.approx(val) == 1.0

    def test_ricker_math(self):
        """Verify Ricker pulse reaches peak amplitude."""
        w = Waveform()
        w.type = "ricker"
        w.freq = 1e9
        w.amp = 1.0
        
        center = np.sqrt(2) / w.freq
        val = w.calculate_value(center, dt=1e-12)
        
        # At center, (2*zeta*0 - 1) * exp(0) * normalise
        # Normalise is 1/(2*zeta). Result is -(-1 * 1 * 1/(2*zeta)) * 2*zeta = 1
        assert pytest.approx(val) == 1.0

    def test_sine_cutoff(self):
        """Sine pulse should terminate after one cycle."""
        w = Waveform()
        w.type, w.freq = "sine", 1e9
        cycle_time = 1.0 / w.freq
        
        val = w.calculate_value(cycle_time + 1e-12, dt=1e-12)
        assert val == 0.0

    def test_impulse_logic(self):
        """Test impulse pulse logic: value should be 1 at t=0 and t=dt, then 0 after."""
        w = Waveform()
        w.type = "impulse"
        dt = 1e-9
        assert w.calculate_value(0, dt) == 1
        assert w.calculate_value(dt*0.5, dt) == 1
        assert w.calculate_value(dt*1.5, dt) == 0

    def test_user_lambda(self):
        """Test user-defined waveform function."""
        w = Waveform()
        w.type = "user"
        w.userfunc = lambda t: t**3
        assert w.calculate_value(2, 1e-12) == 8