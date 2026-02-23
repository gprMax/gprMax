"""
Tests for IO and string formatting helper functions in gprMax.utilities
"""

import sys
import os
import io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from gprMax.utilities import get_terminal_width, logo, open_path_file

def test_get_terminal_width(monkeypatch):
    """Test retrieving terminal width with fallback behavior."""
    # Case 1: width = 0, should default to 100
    monkeypatch.setattr("gprMax.utilities.get_terminal_size", lambda: (0, 0))
    assert get_terminal_width() == 100

    # Case 2: width = 120, should return 120
    monkeypatch.setattr("gprMax.utilities.get_terminal_size", lambda: (120, 50))
    assert get_terminal_width() == 120

def test_logo(capsys, monkeypatch):
    """Test logo printing to stdout."""
    # Mock terminal size to ensure consistent line wrapping behavior
    monkeypatch.setattr("gprMax.utilities.get_terminal_size", lambda: (120, 50))
    
    logo("1.0.0")
    captured = capsys.readouterr()
    
    assert "www.gprmax.com" in captured.out
    assert "v1.0.0" in captured.out

def test_open_path_file_path(tmp_path):
    """Test open_path_file context manager with a string path."""
    file = tmp_path / "test.txt"
    file.write_text("hello gprmax", encoding='utf-8')
    
    with open_path_file(str(file)) as f:
        assert f.read() == "hello gprmax"
        assert not f.closed
    # After context manager exits, file should be properly closed
    assert f.closed

def test_open_path_file_object():
    """Test open_path_file context manager with an existing file object."""
    fake_file = io.StringIO("virtual hello")
    
    with open_path_file(fake_file) as f:
        assert f.read() == "virtual hello"
        
    # Ensure it wasn't closed by the context manager since it was provided as an object
    assert not fake_file.closed
