"""Tests for optional dependency metadata in the installed package."""

import re
from importlib.metadata import metadata, requires

import pytest

OPTIONAL_REQUIREMENTS = {
    "mpi": ("mpi4py",),
    "mpi-fractals": ("mpi4py", "mpi4py-fft"),
    "cuda": ("pycuda",),
    "opencl": ("pyopencl",),
    "metal": ("pyobjc-framework-metal",),
    "accelerators": ("pycuda", "pyopencl", "pyobjc-framework-metal"),
}

SUPPORTED_PYTHON = ">=3.11,<3.14"


def _normalised_requirements():
    return [requirement.lower() for requirement in (requires("gprMax") or [])]


def _has_extra(requirement, extra):
    return re.search(rf"extra\s*==\s*['\"]{re.escape(extra)}['\"]", requirement)


@pytest.mark.unit
def test_supported_python_range_is_declared():
    assert metadata("gprMax")["Requires-Python"] == SUPPORTED_PYTHON


@pytest.mark.unit
def test_optional_extras_are_declared():
    declared = set(metadata("gprMax").get_all("Provides-Extra") or [])

    assert set(OPTIONAL_REQUIREMENTS) <= declared


@pytest.mark.unit
def test_optional_packages_are_not_core_dependencies():
    requirements = _normalised_requirements()
    optional_packages = {package for packages in OPTIONAL_REQUIREMENTS.values() for package in packages}

    for package in optional_packages:
        matching = [item for item in requirements if item.startswith(f"{package};")]
        assert matching
        assert all("extra ==" in item for item in matching)


@pytest.mark.unit
def test_each_extra_selects_its_expected_packages():
    requirements = _normalised_requirements()

    for extra, packages in OPTIONAL_REQUIREMENTS.items():
        for package in packages:
            assert any(item.startswith(f"{package};") and _has_extra(item, extra) for item in requirements)


@pytest.mark.unit
def test_platform_markers_protect_cuda_and_metal():
    requirements = _normalised_requirements()

    cuda = [item for item in requirements if item.startswith("pycuda;")]
    metal = [item for item in requirements if item.startswith("pyobjc-framework-metal;")]

    assert cuda and all('sys_platform != "darwin"' in item for item in cuda)
    assert metal and all('sys_platform == "darwin"' in item for item in metal)
