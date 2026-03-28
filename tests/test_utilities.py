# Unit Test for gprMax/utilities/utilities.py
# Authors: Dang3Rsm


import numpy as np
import pytest

from gprMax.utilities.utilities import (
    atoi,
    fft_power,
    get_terminal_width,
    logo,
    natural_keys,
    round32,
    round_value,
    timer,
)


@pytest.mark.unit
class TestUtilities:
    """Tests for general utility functions in gprMax."""

    def test_get_terminal_width(self):
        """Check get_terminal_width"""
        width = get_terminal_width()
        assert isinstance(width, int)

    def test_logo_content(self):
        """Check logo"""
        fake_version = "9.9.9"
        logo_str = logo(fake_version)
        assert fake_version in logo_str
        assert "gprMax" in logo_str
        assert "Copyright" in logo_str

    def test_timer(self):
        """Check time fn"""
        t1 = timer()
        t2 = timer()
        assert t2 >= t1

    @pytest.mark.parametrize(
        "value, decimals, expected",
        [
            (2.5, 0, 2),  # ROUND_HALF_DOWN
            (3.5, 0, 3),  # ROUND_HALF_DOWN
            (2.501, 0, 3),  # rounding up
            (2.555, 2, 2.55),  # round_float (FLOOR)
            (10.999, 1, 10.9),  # round_float (FLOOR)
            (-2.5, 0, -2),  # Negative value handling
        ],
    )
    def test_round_value(self, value, decimals, expected):
        assert round_value(value, decimals) == expected

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            (0, 0),
            (1, 32),
            (32, 32),
            (33, 64),
            (65.5, 96),
        ],
    )
    def test_round32(self, input_val, expected):
        """Verify round up to nearest multiple of 32"""
        assert round32(input_val) == expected

    def test_atoi(self):
        """Verify convert a string to integer."""
        assert atoi("42") == 42
        assert atoi("abc") == "abc"

    def test_natural_keys(self):
        """Verify human style sorting"""
        list_to_sort = ["model10.in", "model2.in", "model1.in"]
        list_to_sort.sort(key=natural_keys)
        assert list_to_sort == ["model1.in", "model2.in", "model10.in"]

    def test_fft_power_basic(self):
        """Basic check for FFT power calculation logic."""
        dt = 0.001
        t = np.arange(0, 1, dt)
        f = 50
        waveform = np.sin(2 * np.pi * f * t)

        freqs, power = fft_power(waveform, dt)

        # Output sizes match
        assert freqs.shape == waveform.shape
        assert power.shape == waveform.shape

        # No NaN or Inf
        assert np.all(np.isfinite(power))

        # Maximum power is normalized to 0 dB
        assert np.isclose(np.max(power), 0.0)

        # Peak frequency corresponds to the sine frequency
        peak_freq = freqs[np.argmax(power)]
        assert np.isclose(abs(peak_freq), f, atol=1)
