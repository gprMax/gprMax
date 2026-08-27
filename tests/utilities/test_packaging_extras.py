# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

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
    "marimo": ("marimo", "plotly", "kaleido"),
}

SUPPORTED_PYTHON = ">=3.11,<3.14"


def _normalised_requirements():
    return [requirement.lower() for requirement in (requires("gprMax") or [])]


def _has_extra(requirement, extra):
    return re.search(rf"extra\s*==\s*['\"]{re.escape(extra)}['\"]", requirement)


def _requires_package(requirement, package):
    """Match a package name with or without an intervening version specifier."""

    return re.match(rf"{re.escape(package)}(?:\s*[<>=!~]|\s*;)", requirement)


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
    optional_packages = {
        package for packages in OPTIONAL_REQUIREMENTS.values() for package in packages
    }

    for package in optional_packages:
        matching = [item for item in requirements if _requires_package(item, package)]
        assert matching
        assert all("extra ==" in item for item in matching)


@pytest.mark.unit
def test_each_extra_selects_its_expected_packages():
    requirements = _normalised_requirements()

    for extra, packages in OPTIONAL_REQUIREMENTS.items():
        for package in packages:
            assert any(
                _requires_package(item, package) and _has_extra(item, extra)
                for item in requirements
            )


@pytest.mark.unit
def test_platform_markers_protect_cuda_and_metal():
    requirements = _normalised_requirements()

    cuda = [item for item in requirements if item.startswith("pycuda;")]
    metal = [item for item in requirements if item.startswith("pyobjc-framework-metal;")]

    assert cuda and all('sys_platform != "darwin"' in item for item in cuda)
    assert metal and all('sys_platform == "darwin"' in item for item in metal)
