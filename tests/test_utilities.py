"""
Tests for helper functions in gprMax.utilities
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from gprMax.utilities import human_size


@pytest.mark.parametrize("value, expected", [
    (0, '0KB'),
    (500, '0.5KB'),
    (1000, '1KB'),
    (1234, '1.23KB'),  # Checking the 3g formatting
    (1500, '1.5KB'),
    (1000**2, '1MB'),
    (1500000, '1.5MB'),
    (1000**3, '1GB'),
])
def test_human_size_1000(value, expected):
    """Test standard base-1000 conversions."""
    assert human_size(value) == expected


@pytest.mark.parametrize("value, expected", [
    (0, '0KiB'),
    (512, '0.5KiB'),
    (1024, '1KiB'),
    (1234, '1.21KiB'),  # Checking the 3g formatting (1234 / 1024)
    (1536, '1.5KiB'),
    (1024**2, '1MiB'),
    (1572864, '1.5MiB'),  # 1.5 * 1024^2
    (1024**3, '1GiB'),
])
def test_human_size_1024(value, expected):
    """Test base-1024 (kibibyte) conversions."""
    assert human_size(value, a_kilobyte_is_1024_bytes=True) == expected


def test_human_size_negative():
    """Test behavior with negative numbers."""
    with pytest.raises(ValueError, match='Number must be non-negative.'):
        human_size(-1)


def test_human_size_too_large():
    """Test behavior when the number exceeds suffix lists."""
    # 1000**9 is beyond 'YB' which is 1000**8
    with pytest.raises(ValueError, match='Number is too large.'):
        human_size(1000**9)


