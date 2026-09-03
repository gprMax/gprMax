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

"""Configuration and smoke tests for the standalone benchmark modules."""

import itertools
from pathlib import Path
from types import SimpleNamespace

from testing.benchmarking import bench_simple, benchmark_layered_ntff
from testing.benchmarking import benchmark_metal as metal_benchmark


def test_simple_benchmark_builds_the_requested_matrix_without_running():
    scenes = bench_simple.build_scenes(
        domains=(0.024, 0.030),
        threads=(1, 2),
        cell_size=0.001,
        time_window=1e-11,
    )

    assert len(scenes) == 4
    assert all(scene.single_use_objects for scene in scenes)
    assert all(scene.grid_objects for scene in scenes)


def test_metal_benchmark_uses_current_python_api_backend_options():
    assert metal_benchmark._solver_options("cpu", 3) == {
        "cpu_precision": "single"
    }
    assert metal_benchmark._solver_options("metal", 3) == {
        "metal": [3],
        "gpu_precision": "single",
    }


def test_metal_benchmark_runs_cpu_and_metal_for_each_case(monkeypatch):
    calls = []
    ticks = itertools.cycle((0.0, 1.0))
    monkeypatch.setattr(metal_benchmark, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        metal_benchmark.gprMax,
        "run",
        lambda **kwargs: calls.append(kwargs),
    )
    args = SimpleNamespace(
        sizes=(24,),
        cell_size=0.001,
        iterations=4,
        threads=1,
        device=0,
        repeats=2,
        warmups=1,
    )

    results = metal_benchmark.run_benchmark(args)

    assert len(calls) == 6
    assert sum("metal" in call for call in calls) == 3
    assert sum("cpu_precision" in call for call in calls) == 3
    assert results["cases"][0]["cpu"]["median_seconds"] == 1
    assert results["cases"][0]["metal"]["median_seconds"] == 1
    assert results["cases"][0]["speedup"] == 1


def test_metal_benchmark_plot_is_written(tmp_path):
    backend = {
        "run_seconds": [1.0],
        "median_seconds": 1.0,
        "performance_mcells_per_second": 2.0,
    }
    results = {
        "cases": [
            {
                "size": 24,
                "cpu": backend,
                "metal": {**backend, "performance_mcells_per_second": 4.0},
                "speedup": 2.0,
            }
        ]
    }
    output = tmp_path / "benchmark.png"

    metal_benchmark.create_performance_plot(results, output)

    assert output.stat().st_size > 0


def test_layered_ntff_benchmark_matches_numpy_oracle():
    args = SimpleNamespace(
        frequencies=2,
        directions=5,
        patches=4,
        threads=1,
        seed=22,
    )

    result = benchmark_layered_ntff.run_benchmark(args)

    assert result["maximum_electric_difference"] < 1e-12
    assert result["maximum_magnetic_difference"] < 1e-12


def test_python_source_headers_do_not_contain_legacy_author_bylines():
    root = Path(__file__).parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if any(part in {".git", "__pycache__", "build", "dist"} for part in path.parts):
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.lstrip("# ").lower().startswith("authors:"):
                offenders.append(f"{path.relative_to(root)}:{line_number}")

    assert offenders == []
