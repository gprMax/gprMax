import pytest
from gprMax.utilities.utilities import atoi, natural_keys

def test_atoi():
    """Test atoi function for string to int conversion."""
    assert atoi("123") == 123
    assert atoi("abc") == "abc"
    assert atoi("42test") == "42test"
    assert atoi("-5") == -5

def test_natural_keys():
    """Test natural_keys for human-friendly sorting."""
    test_strings = ["file2.txt", "file10.txt", "file1.txt", "file20.txt"]
    sorted_strings = sorted(test_strings, key=natural_keys)
    
    expected = ["file1.txt", "file2.txt", "file10.txt", "file20.txt"]
    assert sorted_strings == expected

def test_natural_keys_mixed():
    """Test natural_keys with mixed alphanumeric strings."""
    test_strings = ["item10", "item2", "item1", "item20a", "item3"]
    sorted_strings = sorted(test_strings, key=natural_keys)
    
    expected = ["item1", "item2", "item3", "item10", "item20a"]
    assert sorted_strings == expected
