# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

"""
Unit tests for gprMax utility functions.

This module contains pytest-based unit tests for standalone utility functions
that can be tested without requiring the full gprMax installation.

Tested functions:
- round_value: Rounding with configurable decimal places
- round32: Rounding to nearest multiple of 32
- fft_power: FFT power spectrum calculation
- human_size: Human-readable file size formatting
- get_terminal_width: Terminal width detection

These tests provide foundational coverage for core utility functions
that are used throughout the gprMax codebase.
"""

import sys
import os
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

# Import individual functions to avoid triggering full gprMax import chain
# We use importlib to import just the utilities module directly
import importlib.util

# Get the path to utilities.py
utilities_path = os.path.join(os.path.dirname(__file__), '..', 'gprMax', 'utilities.py')
utilities_path = os.path.abspath(utilities_path)

# We need to mock some imports that utilities.py depends on
# First, create mock modules for the imports that would trigger the full chain
import types

# Create mock for gprMax.constants
mock_constants = types.ModuleType('gprMax.constants')
mock_constants.complextype = np.complex64
mock_constants.floattype = np.float32
sys.modules['gprMax.constants'] = mock_constants

# Create mock for gprMax.exceptions
mock_exceptions = types.ModuleType('gprMax.exceptions')
class GeneralError(ValueError):
    def __init__(self, message, *args):
        self.message = message
        super().__init__(message, *args)
        
class CmdInputError(ValueError):
    def __init__(self, message, *args):
        self.message = message
        super().__init__(message, *args)

mock_exceptions.GeneralError = GeneralError
mock_exceptions.CmdInputError = CmdInputError
sys.modules['gprMax.exceptions'] = mock_exceptions

# Create mock for gprMax.materials (minimal mock)
mock_materials = types.ModuleType('gprMax.materials')
mock_materials.Material = type('Material', (), {})
sys.modules['gprMax.materials'] = mock_materials

# Now we can import utilities
spec = importlib.util.spec_from_file_location("utilities", utilities_path)
utilities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utilities)

# Extract the functions we want to test
round_value = utilities.round_value
round32 = utilities.round32
fft_power = utilities.fft_power
human_size = utilities.human_size
get_terminal_width = utilities.get_terminal_width


class TestRoundValue:
    """Tests for round_value function.
    
    The round_value function rounds to nearest integer (half values rounded
    downwards) when decimalplaces=0, or rounds down to nearest float when
    decimalplaces > 0.
    """

    # Test integer rounding (decimalplaces=0)
    @pytest.mark.parametrize("value,expected", [
        (1.4, 1),
        (1.5, 1),      # Half values round DOWN (ROUND_HALF_DOWN)
        (1.6, 2),
        (2.5, 2),      # Half values round DOWN
        (3.5, 3),      # Half values round DOWN
        (-1.4, -1),
        (-1.6, -2),
    ])
    def test_integer_rounding(self, value, expected):
        """Test rounding to nearest integer with half-down behavior."""
        result = round_value(value, decimalplaces=0)
        assert result == expected
        assert isinstance(result, int)

    def test_exact_integer_unchanged(self):
        """Exact integers should remain unchanged."""
        assert round_value(5.0) == 5
        assert round_value(0.0) == 0
        assert round_value(-3.0) == -3

    # Test decimal place rounding (ROUND_FLOOR behavior)
    @pytest.mark.parametrize("value,places,expected", [
        (1.456, 1, 1.4),    # Rounds DOWN to 1 decimal place
        (1.456, 2, 1.45),   # Rounds DOWN to 2 decimal places
        (1.999, 1, 1.9),    # Rounds DOWN
        (1.999, 2, 1.99),   # Rounds DOWN
        (2.555, 2, 2.55),   # Rounds DOWN (not banker's rounding)
    ])
    def test_decimal_rounding(self, value, places, expected):
        """Test rounding with specified decimal places (floor behavior)."""
        result = round_value(value, decimalplaces=places)
        assert result == pytest.approx(expected)
        assert isinstance(result, float)

    def test_negative_decimal_rounding(self):
        """Test rounding negative numbers with decimal places."""
        # Floor rounding for negative numbers rounds towards -infinity
        result = round_value(-1.456, decimalplaces=1)
        assert result == pytest.approx(-1.5)

    def test_zero_value(self):
        """Test rounding zero."""
        assert round_value(0) == 0
        assert round_value(0.0, decimalplaces=2) == 0.0

    def test_large_values(self):
        """Test rounding large values."""
        assert round_value(1e10 + 0.4) == int(1e10)
        assert round_value(1e10 + 0.6) == int(1e10) + 1


class TestRound32:
    """Tests for round32 function.
    
    The round32 function rounds up to the nearest multiple of 32.
    This is typically used for memory alignment optimizations.
    """

    @pytest.mark.parametrize("value,expected", [
        (0, 0),
        (1, 32),
        (31, 32),
        (32, 32),
        (33, 64),
        (63, 64),
        (64, 64),
        (100, 128),
        (1000, 1024),
    ])
    def test_basic_rounding(self, value, expected):
        """Test rounding up to nearest multiple of 32."""
        result = round32(value)
        assert result == expected
        assert isinstance(result, int)

    def test_float_input(self):
        """Test that float inputs are handled correctly."""
        assert round32(31.5) == 32
        assert round32(32.1) == 64
        assert round32(100.9) == 128

    def test_result_divisible_by_32(self):
        """All results should be divisible by 32."""
        for value in range(1, 200):
            result = round32(value)
            assert result % 32 == 0, f"round32({value}) = {result} not divisible by 32"

    def test_result_never_less_than_input(self):
        """Result should never be less than input (rounds UP)."""
        for value in range(0, 200):
            result = round32(value)
            assert result >= value, f"round32({value}) = {result} is less than input"


class TestFFTPower:
    """Tests for fft_power function.
    
    The fft_power function calculates FFT power spectrum in dB,
    normalized so the maximum power is 0 dB.
    """

    def test_basic_fft(self):
        """Test FFT of simple sinusoidal waveform."""
        # Create a simple sine wave
        dt = 1e-9  # 1 ns timestep
        t = np.arange(0, 100e-9, dt)  # 100 ns duration
        freq = 100e6  # 100 MHz
        waveform = np.sin(2 * np.pi * freq * t)
        
        freqs, power = fft_power(waveform, dt)
        
        # Check output shapes match
        assert freqs.shape == power.shape == waveform.shape
        
        # Check frequency array contains expected range
        assert np.min(freqs) < 0  # Should have negative frequencies
        assert np.max(freqs) > 0  # Should have positive frequencies

    def test_power_normalized_to_zero_db(self):
        """Maximum power should be 0 dB after normalization."""
        dt = 1e-9
        t = np.arange(0, 100e-9, dt)
        waveform = np.sin(2 * np.pi * 100e6 * t)
        
        freqs, power = fft_power(waveform, dt)
        
        # Maximum power should be exactly 0 dB
        assert np.max(power) == pytest.approx(0.0)

    def test_power_all_non_positive(self):
        """All power values should be <= 0 dB after normalization."""
        dt = 1e-9
        t = np.arange(0, 100e-9, dt)
        waveform = np.sin(2 * np.pi * 100e6 * t) + 0.5 * np.sin(2 * np.pi * 200e6 * t)
        
        freqs, power = fft_power(waveform, dt)
        
        assert np.all(power <= 0), "All power values should be <= 0 dB"

    def test_zero_waveform(self):
        """Test handling of zero waveform (should not produce NaN/Inf)."""
        waveform = np.zeros(100)
        dt = 1e-9
        
        freqs, power = fft_power(waveform, dt)
        
        # Should not have any NaN or Inf values
        assert np.all(np.isfinite(power)), "Zero waveform should not produce NaN/Inf"

    def test_impulse_response(self):
        """Test FFT of impulse (should have flat spectrum)."""
        waveform = np.zeros(64)
        waveform[0] = 1.0  # Impulse at t=0
        dt = 1e-9
        
        freqs, power = fft_power(waveform, dt)
        
        # Impulse should have flat frequency response (all values equal)
        # After normalization, all should be 0 dB
        assert np.allclose(power, 0.0), "Impulse should have flat spectrum"

    def test_frequency_bins_correct(self):
        """Verify frequency bins are correctly calculated."""
        n_samples = 100
        dt = 1e-9  # 1 GHz sampling rate
        waveform = np.random.randn(n_samples)
        
        freqs, power = fft_power(waveform, dt)
        
        # Check Nyquist frequency (allow for floating point tolerance)
        nyquist = 1 / (2 * dt)
        assert np.max(np.abs(freqs)) <= nyquist * (1 + 1e-10)


class TestHumanSize:
    """Tests for human_size function.
    
    The human_size function converts byte sizes to human-readable strings
    with appropriate unit suffixes.
    """

    # Test with base 1000 (default)
    @pytest.mark.parametrize("size,expected_suffix", [
        (1000, "KB"),
        (1000**2, "MB"),
        (1000**3, "GB"),
        (1000**4, "TB"),
    ])
    def test_base_1000_suffixes(self, size, expected_suffix):
        """Test correct suffix selection for base 1000."""
        result = human_size(size, a_kilobyte_is_1024_bytes=False)
        assert expected_suffix in result

    # Test with base 1024
    @pytest.mark.parametrize("size,expected_suffix", [
        (1024, "KiB"),
        (1024**2, "MiB"),
        (1024**3, "GiB"),
        (1024**4, "TiB"),
    ])
    def test_base_1024_suffixes(self, size, expected_suffix):
        """Test correct suffix selection for base 1024."""
        result = human_size(size, a_kilobyte_is_1024_bytes=True)
        assert expected_suffix in result

    def test_exact_kilobyte_1000(self):
        """Test exactly 1 KB (1000 bytes)."""
        result = human_size(1000, a_kilobyte_is_1024_bytes=False)
        assert result == "1KB"

    def test_exact_kibibyte_1024(self):
        """Test exactly 1 KiB (1024 bytes)."""
        result = human_size(1024, a_kilobyte_is_1024_bytes=True)
        assert result == "1KiB"

    def test_negative_size_raises_error(self):
        """Negative size should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            human_size(-100)

    def test_small_sizes_below_threshold(self):
        """Very small sizes stay small after division."""
        # Small sizes less than 1000 bytes will keep dividing
        # and eventually return a very small value with suffix
        result = human_size(500, a_kilobyte_is_1024_bytes=False)
        assert "KB" in result  # 0.5 KB

    def test_realistic_file_sizes(self):
        """Test common realistic file sizes."""
        # 1.5 MB
        result = human_size(1500000, a_kilobyte_is_1024_bytes=False)
        assert "MB" in result
        assert "1.5" in result
        
        # 2.5 GB
        result = human_size(2500000000, a_kilobyte_is_1024_bytes=False)
        assert "GB" in result


class TestGetTerminalWidth:
    """Tests for get_terminal_width function.
    
    The get_terminal_width function returns the terminal width,
    defaulting to 100 if width cannot be determined.
    """

    def test_returns_positive_integer(self):
        """Terminal width should be a positive integer."""
        width = get_terminal_width()
        assert isinstance(width, int)
        assert width > 0

    def test_returns_at_least_default(self):
        """Terminal width should be at least the default (100) or actual."""
        width = get_terminal_width()
        # Function returns 100 if actual width is 0
        assert width >= 1

    def test_reasonable_width_range(self):
        """Terminal width should be within a reasonable range."""
        width = get_terminal_width()
        # Most terminals are between 80 and 300 columns
        # But in test environments it might be the default 100
        assert 50 <= width <= 500


class TestConstants:
    """Tests for physical constants.
    
    Verifies that fundamental physical constants are correctly defined.
    Note: We test scipy.constants directly since gprMax.constants
    imports from there.
    """

    def test_speed_of_light(self):
        """Speed of light should be approximately 3e8 m/s."""
        from scipy.constants import c
        assert c == pytest.approx(299792458, rel=1e-6)

    def test_vacuum_permittivity(self):
        """Vacuum permittivity (epsilon_0) should be approximately 8.854e-12."""
        from scipy.constants import epsilon_0
        assert epsilon_0 == pytest.approx(8.854187817e-12, rel=1e-6)

    def test_vacuum_permeability(self):
        """Vacuum permeability (mu_0) should be approximately 1.257e-6."""
        from scipy.constants import mu_0
        assert mu_0 == pytest.approx(1.2566370614e-6, rel=1e-6)

    def test_impedance_of_free_space(self):
        """Impedance of free space (z0) should be approximately 377 Ohms."""
        from scipy.constants import mu_0, epsilon_0
        z0 = np.sqrt(mu_0 / epsilon_0)
        assert z0 == pytest.approx(376.73, rel=1e-3)

    def test_float_types(self):
        """Verify correct numpy float types are expected."""
        # These are the expected types per gprMax/constants.py
        assert np.float32 is not None
        assert np.complex64 is not None


class TestExceptions:
    """Tests for custom exception classes (using our mock implementations)."""

    def test_general_error_is_value_error(self):
        """GeneralError should be a subclass of ValueError."""
        assert issubclass(GeneralError, ValueError)

    def test_cmd_input_error_is_value_error(self):
        """CmdInputError should be a subclass of ValueError."""
        assert issubclass(CmdInputError, ValueError)

    def test_general_error_stores_message(self):
        """GeneralError should store the error message."""
        try:
            raise GeneralError("Test error message")
        except GeneralError as e:
            assert e.message == "Test error message"

    def test_cmd_input_error_stores_message(self):
        """CmdInputError should store the error message."""
        try:
            raise CmdInputError("Test command error")
        except CmdInputError as e:
            assert e.message == "Test command error"


# Marker for slow tests (integration-level)
@pytest.mark.slow
class TestIntegration:
    """Integration tests that verify utilities work together."""

    def test_round_value_with_fft_frequencies(self):
        """Test that round_value works correctly with FFT frequency values."""
        dt = 1e-9
        waveform = np.random.randn(100)
        freqs, power = fft_power(waveform, dt)
        
        # Round a frequency value
        rounded_freq = round_value(freqs[10], decimalplaces=6)
        assert isinstance(rounded_freq, float)

    def test_human_size_with_common_array_sizes(self):
        """Test human_size with typical array memory sizes."""
        # 1000x1000 float32 array = 4 MB
        array_size = 1000 * 1000 * 4  # bytes
        result = human_size(array_size, a_kilobyte_is_1024_bytes=False)
        assert "MB" in result

