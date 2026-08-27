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

"""Configuration and execution checks for the reusable-KSIR benchmark."""

import json
from types import SimpleNamespace

import pytest
from numpy.testing import assert_allclose

from testing.benchmarking.benchmark_ntff import (
    _frequencies,
    _monitor_bounds,
    _solver_options,
    run_benchmark,
)


def test_monitor_bounds_are_centred_and_clear_of_pml():
    lower, upper = _monitor_bounds(80, 24, 10, 0.002)

    assert_allclose(lower, (0.056,) * 3)
    assert_allclose(upper, (0.104,) * 3)


@pytest.mark.parametrize("surface_cells", [23, 58, 80])
def test_monitor_bounds_reject_invalid_benchmark_surfaces(surface_cells):
    with pytest.raises(ValueError):
        _monitor_bounds(80, surface_cells, 10, 0.002)


def test_benchmark_frequency_count_is_exact_and_centred():
    assert _frequencies(1, 4e9) == [4e9]
    frequencies = _frequencies(10, 4e9)

    assert len(frequencies) == 10
    assert frequencies[0] == pytest.approx(2.4e9)
    assert frequencies[-1] == pytest.approx(5.6e9)


def test_solver_options_use_gprmax_precision_and_backend_arguments():
    assert _solver_options("cpu", 3, "double") == {"cpu_precision": "double"}
    assert _solver_options("cuda", 3, "single") == {
        "gpu": [3],
        "gpu_precision": "single",
    }
    assert _solver_options("opencl", 2, "double")["opencl"] == [2]
    assert _solver_options("metal", 1, "single")["metal"] == [1]


def test_cpu_benchmark_executes_production_collection_path():
    args = SimpleNamespace(
        backend="cpu",
        device=0,
        cells=24,
        iterations=160,
        dl=0.002,
        pml_cells=3,
        threads=2,
        precision="double",
        centre_frequency=4e9,
        surface_cells=[8],
        frequency_counts=[1],
        repeats=1,
        warmups=0,
    )

    result = run_benchmark(args)

    assert result["configuration"]["backend"] == "cpu"
    assert result["baseline"]["median_seconds"] > 0
    assert len(result["cases"]) == 1
    assert result["cases"][0]["component_patches"] > 0
    assert result["cases"][0]["collection_backend"] == "cython_openmp"
    assert result["cases"][0]["median_seconds"] > 0
    json.dumps(result)
