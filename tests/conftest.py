# tests/conftest.py
#
# Shared pytest configuration and fixtures for the entire gprMax test suite.
#
# Why this file matters for CI:
#   • matplotlib.use('Agg') MUST be called before any other matplotlib import.
#     Without this, matplotlib tries to connect to a display server (X11/Quartz)
#     which does not exist on headless CI runners.  The result is a cryptic
#     crash: "_tkinter.TclError: no display name and no $DISPLAY environment
#     variable" or "cannot connect to X server".  Placing the backend selection
#     here in conftest.py guarantees it runs before any test module imports
#     matplotlib — regardless of test collection order.
#     Fixes GitHub Issue #621.
#
# References:
#   https://matplotlib.org/stable/users/explain/backends.html
#   https://github.com/gprMax/gprMax/issues/621
#

import os

import matplotlib
import pytest

# ── Headless matplotlib backend ───────────────────────────────────────────────
#
# 'Agg' (Anti-Grain Geometry) is a non-interactive, file-only backend.
# It produces identical pixel-accurate output to the interactive backends but
# requires no display server.  This is the standard choice for CI rendering.
#
# IMPORTANT: This call must come before any 'import matplotlib.pyplot' anywhere
# in the test session.  conftest.py is loaded first, so this is safe here.
#
matplotlib.use("Agg")


# ── Test markers (registered to avoid PytestUnknownMarkWarning) ───────────────
#
# Markers declared here are automatically recognised by pytest.
# The actual marker definitions in setup.cfg (under [tool:pytest] → markers)
# drive 'pytest --co -q' output, but pytest also requires them to be
# registered via pytest_configure OR listed in setup.cfg.  Both approaches
# are used here for maximum compatibility.
#

def pytest_configure(config):
    """Register custom gprMax test markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests that run full FDTD simulations (deselect with -m 'not slow')",
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a CUDA-capable GPU (deselect with -m 'not gpu')",
    )
    config.addinivalue_line(
        "markers",
        "mpi: marks tests that require an MPI runtime (deselect with -m 'not mpi')",
    )


# ── Shared path fixtures ──────────────────────────────────────────────────────
#
# These fixtures provide consistent, reliable paths to test resource directories.
# Using fixtures instead of hardcoded paths means:
#   • Tests pass regardless of the working directory from which pytest is invoked.
#   • Refactoring the directory layout only requires changing these fixtures.
#

@pytest.fixture(scope="session")
def tests_root_path():
    """Absolute path to the 'tests/' directory.

    Scope 'session' means this is computed once for the entire test run.
    """
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def repo_root_path(tests_root_path):
    """Absolute path to the repository root (one level above 'tests/')."""
    return os.path.dirname(tests_root_path)


@pytest.fixture(scope="session")
def models_basic_path(tests_root_path):
    """Path to the directory containing basic test .in model files.

    These are small models used for quick functional verification.
    They are tagged with @pytest.mark.slow when they invoke the FDTD solver.
    """
    return os.path.join(tests_root_path, "models_basic")


@pytest.fixture(scope="session")
def models_advanced_path(tests_root_path):
    """Path to the directory containing advanced test .in model files.

    These models exercise specific physical configurations (dispersive media,
    subgrids, GPU acceleration) and are always marked with @pytest.mark.slow.
    """
    return os.path.join(tests_root_path, "models_advanced")


# ── CI environment detection fixture ─────────────────────────────────────────

@pytest.fixture(scope="session")
def is_ci():
    """Returns True when running inside a CI environment (GitHub Actions, etc.).

    Usage in tests::

        def test_something(is_ci):
            if is_ci:
                pytest.skip("Requires interactive display; skipped in CI")
    """
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
