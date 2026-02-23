"""
Tests for mathematical helper functions in gprMax.utilities
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from gprMax.utilities import round_value, round32

@pytest.mark.parametrize("value, expected", [
    (1.1, 1),
    (1.5, 1),
    (1.6, 2),
    (-1.5, -1),
    (-1.6, -2),
    (0.0, 0),
    (2.5, 2),
    (3.5, 3),
])
def test_round_value_integer(value, expected):
    """Test rounding to nearest integer (decimalplaces=0)."""
    assert round_value(value) == expected


@pytest.mark.parametrize("value, decimalplaces, expected", [
    (1.239, 2, 1.23),
    (1.231, 2, 1.23),
    (-1.239, 2, -1.24),
    (-1.231, 2, -1.24),
    (1.12345, 4, 1.1234),
])
def test_round_value_float(value, decimalplaces, expected):
    """Test rounding down to nearest float with precision."""
    assert round_value(value, decimalplaces=decimalplaces) == expected


@pytest.mark.parametrize("value, expected", [
    (0, 0),
    (1, 32),
    (31, 32),
    (32, 32),
    (33, 64),
    (100, 128),
    (-1, 0),
    (-33, -32),
])
def test_round32(value, expected):
    """Test rounding up to nearest multiple of 32."""
    assert round32(value) == expected
