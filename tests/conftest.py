"""
Pytest configuration and shared fixtures for the gprMax test suite.

This file is automatically loaded by pytest before any tests are collected.
It performs two tasks:

1. Forces matplotlib to use the non-interactive 'Agg' backend *before* any
   test module imports pyplot. Without this, tests crash in headless CI
   environments (Linux runners, Docker containers) with:
       "cannot connect to X server" or "_tkinter.TclError"
   Fixes #621.

2. Provides shared path fixtures so individual test modules do not need to
   hard-code paths to the test model directories.
"""
import os

import matplotlib
matplotlib.use('Agg')  # noqa: E402 — must be called before pyplot is imported

import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def models_basic_path():
    """Absolute path to the directory containing basic test models."""
    return os.path.join(os.path.dirname(__file__), 'models_basic')


@pytest.fixture
def models_advanced_path():
    """Absolute path to the directory containing advanced test models."""
    return os.path.join(os.path.dirname(__file__), 'models_advanced')
