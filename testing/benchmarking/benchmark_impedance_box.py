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

"""Benchmark the CPU cost of local surface-impedance pole updates.

The benchmark includes an ordinary-grid baseline, a zero-pole resistive
surface, the automatically selected copper fit, and explicitly requested
metal orders. It reports both complete-solve overhead and an isolated sparse
kernel rate. Timings are descriptive rather than CI acceptance thresholds.

Example::

    python -m testing.benchmarking.benchmark_impedance_box --threads 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import gprMax

_SOLVE_SECONDS = []


@dataclass(frozen=True)
class SurfaceCase:
    """One material choice run on the common benchmark geometry."""

    key: str
    kind: str
    requested_order: str | int | None = None


def benchmark_cases(explicit_orders) -> tuple[SurfaceCase, ...]:
    """Return the baseline and surface cases in deterministic order."""

    orders = tuple(dict.fromkeys(int(value) for value in explicit_orders))
    if any(value <= 0 for value in orders):
        raise ValueError("explicit impedance benchmark orders must be positive")
    return (
        SurfaceCase("baseline", "none"),
        SurfaceCase("resistive", "resistance"),
        SurfaceCase("metal_auto", "metal", "auto"),
        *(SurfaceCase(f"metal_order_{value}", "metal", value) for value in orders),
    )


def build_scene(
    cells: int,
    iterations: int,
    threads: int,
    case: SurfaceCase,
    *,
    fit_fmin_hz: float,
    fit_fmax_hz: float,
    fit_tolerance: float,
):
    """Build one otherwise identical baseline or impedance-box scene."""

    dl = 0.001
    extent = cells * dl
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(extent, extent, extent)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(iterations=iterations))
    scene.add(gprMax.PMLThickness(thickness=5))
    scene.add(gprMax.OMPThreads(threads))
    if case.kind != "none":
        if case.kind == "resistance":
            surface = gprMax.SurfaceImpedance(id="benchmark_wall", resistance=50.0)
        elif case.kind == "metal":
            surface = gprMax.SurfaceImpedance(
                id="benchmark_wall",
                preset="copper",
                fit_frequency_range=(fit_fmin_hz, fit_fmax_hz),
                fit_order=case.requested_order,
                fit_tolerance=fit_tolerance,
            )
        else:
            raise ValueError(f"unknown impedance benchmark case kind {case.kind!r}")
        lower = 0.25 * extent
        upper = 0.75 * extent
        scene.add(surface)
        scene.add(
            gprMax.Box(
                p1=(lower,) * 3,
                p2=(upper,) * 3,
                material_id="benchmark_wall",
                averaging="n",
            )
        )
    return scene


def timed_run(path: Path, *, args, case: SurfaceCase, iterations: int | None = None):
    """Run one scene and return wall-clock and solver-only durations."""

    started = perf_counter()
    gprMax.run(
        scenes=[
            build_scene(
                args.cells,
                args.iterations if iterations is None else iterations,
                args.threads,
                case,
                fit_fmin_hz=args.fit_band[0],
                fit_fmax_hz=args.fit_band[1],
                fit_tolerance=args.fit_tolerance,
            )
        ],
        outputfile=path,
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )
    return perf_counter() - started, _SOLVE_SECONDS[-1]


def _array_bytes(owner, *names: str) -> int:
    """Return packed bytes for attributes present on a runtime system."""

    return sum(getattr(owner, name).nbytes for name in names if hasattr(owner, name))


def _surface_metadata(grid, system) -> dict:
    """Return order and storage metadata for this one-model benchmark."""

    if len(system.model_ids) != 1:
        raise RuntimeError("impedance-box benchmark expects exactly one used surface model")
    model = grid.surface_impedance_models[system.model_ids[0]]
    selected_poles = int(model.order)
    state_values = int(system.port_count * selected_poles)
    real_size = grid.Ex.dtype.itemsize
    state_array_bytes = _array_bytes(system, "state_y")
    return {
        "requested_order": getattr(model, "fit_requested_order", None),
        "selected_poles": selected_poles,
        "fit_tolerance": getattr(model, "fit_tolerance", None),
        "fit_max_relative_error": getattr(model, "fit_max_relative_error", None),
        "fit_rms_relative_error": getattr(model, "fit_rms_relative_error", None),
        "port_state_values": state_values,
        "port_state_bytes": state_values * real_size,
        "packed_state_array_bytes": state_array_bytes,
        "model_local_coefficient_bytes": _array_bytes(system, "model_f", "model_q", "model_Z0"),
        "precomputed_edge_port_bytes": _array_bytes(
            system,
            "edge_runtime",
            "port_g_over_Z0",
            "port_inv_Z0",
        ),
    }


def run_benchmark(args):
    """Execute all cases and return a JSON-serializable result dictionary."""

    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.solvers import Solver
    from gprMax.updates.cpu_updates import CPUUpdates

    cases = benchmark_cases(args.explicit_orders)
    _SOLVE_SECONDS.clear()
    built_grids = []
    original_grid_build = FDTDGrid.build
    original_solve = Solver.solve

    def capture_grid(self):
        result = original_grid_build(self)
        # Retain only the most recently built grid here. ``latest_grids``
        # below keeps one final grid per case; holding every warm-up/repeat
        # grid would make the benchmark itself consume unbounded memory.
        built_grids[:] = [self]
        return result

    def measure_solve(self, iterator):
        started = perf_counter()
        try:
            return original_solve(self, iterator)
        finally:
            _SOLVE_SECONDS.append(perf_counter() - started)

    class BenchmarkPatches:
        def __enter__(self):
            FDTDGrid.build = capture_grid
            Solver.solve = measure_solve

        def __exit__(self, *_):
            FDTDGrid.build = original_grid_build
            Solver.solve = original_solve

    total_seconds = {case.key: [] for case in cases}
    solve_seconds = {case.key: [] for case in cases}
    latest_grids = {}
    warmup_iterations = max(10, args.iterations // 10) if args.iterations > 10 else args.iterations

    with BenchmarkPatches(), tempfile.TemporaryDirectory(
        prefix="gprmax_impedance_benchmark_", dir=Path.cwd()
    ) as tmp:
        directory = Path(tmp)
        for case in cases:
            timed_run(
                directory / f"warmup_{case.key}",
                args=args,
                case=case,
                iterations=warmup_iterations,
            )

        for repeat in range(args.repeats):
            ordered_cases = cases if repeat % 2 == 0 else tuple(reversed(cases))
            for case in ordered_cases:
                elapsed, solve_elapsed = timed_run(
                    directory / f"{case.key}_{repeat}", args=args, case=case
                )
                total_seconds[case.key].append(elapsed)
                solve_seconds[case.key].append(solve_elapsed)
                latest_grids[case.key] = built_grids[-1]

        kernel_seconds = {}
        for case in cases[1:]:
            grid = latest_grids[case.key]
            system = grid.impedance_surfaces
            samples = []
            for _ in range(args.kernel_repeats):
                grid.reset_fields()
                grid.Hx.fill(0.1)
                grid.Hy.fill(-0.2)
                grid.Hz.fill(0.3)
                started = perf_counter()
                for _ in range(args.kernel_iterations):
                    system.update(grid)
                samples.append(perf_counter() - started)
            kernel_seconds[case.key] = samples

        updates = {key: CPUUpdates(grid) for key, grid in latest_grids.items()}

        def hot_loop(case_key, count):
            case_updates = updates[case_key]
            for _ in range(count):
                case_updates.update_magnetic()
                case_updates.update_electric_a()
                case_updates.update_impedance_surfaces()

        for case in cases:
            latest_grids[case.key].reset_fields()
            hot_loop(case.key, 5)

        hot_seconds = {case.key: [] for case in cases}
        for repeat in range(args.hot_repeats):
            ordered_cases = cases if repeat % 2 == 0 else tuple(reversed(cases))
            for case in ordered_cases:
                latest_grids[case.key].reset_fields()
                started = perf_counter()
                hot_loop(case.key, args.hot_iterations)
                hot_seconds[case.key].append(perf_counter() - started)

    baseline_total = statistics.median(total_seconds["baseline"])
    baseline_solve = statistics.median(solve_seconds["baseline"])
    baseline_hot = statistics.median(hot_seconds["baseline"])
    surface_results = []
    for case in cases[1:]:
        grid = latest_grids[case.key]
        system = grid.impedance_surfaces
        kernel_median = statistics.median(kernel_seconds[case.key])
        total_median = statistics.median(total_seconds[case.key])
        solve_median = statistics.median(solve_seconds[case.key])
        hot_median = statistics.median(hot_seconds[case.key])
        metadata = _surface_metadata(grid, system)
        edge_updates = args.kernel_iterations * system.edge_count
        port_updates = args.kernel_iterations * system.port_count
        pole_updates = args.kernel_iterations * metadata["port_state_values"]
        surface_results.append(
            {
                "case": case.key,
                "kind": case.kind,
                **metadata,
                "boundary_edges": system.edge_count,
                "surface_ports": system.port_count,
                "total_seconds": total_seconds[case.key],
                "total_median_seconds": total_median,
                "total_median_overhead_percent": 100 * (total_median / baseline_total - 1),
                "solve_seconds": solve_seconds[case.key],
                "solve_median_seconds": solve_median,
                "solve_median_overhead_percent": 100 * (solve_median / baseline_solve - 1),
                "sparse_kernel_seconds": kernel_seconds[case.key],
                "sparse_kernel_median_seconds": kernel_median,
                "sparse_edge_updates_per_second": edge_updates / kernel_median,
                "sparse_port_updates_per_second": port_updates / kernel_median,
                "sparse_pole_updates_per_second": pole_updates / kernel_median,
                "bulk_plus_surface_hot_seconds": hot_seconds[case.key],
                "bulk_plus_surface_hot_median_seconds": hot_median,
                "bulk_plus_surface_overhead_percent": 100 * (hot_median / baseline_hot - 1),
            }
        )

    return {
        "configuration": {
            "domain_cells": args.cells,
            "iterations": args.iterations,
            "threads": args.threads,
            "repeats": args.repeats,
            "kernel_iterations": args.kernel_iterations,
            "kernel_repeats": args.kernel_repeats,
            "hot_loop_iterations": args.hot_iterations,
            "hot_loop_repeats": args.hot_repeats,
            "metal_preset": "copper",
            "fit_frequency_range_hz": list(args.fit_band),
            "fit_tolerance": args.fit_tolerance,
            "explicit_orders": list(args.explicit_orders),
        },
        "baseline": {
            "total_seconds": total_seconds["baseline"],
            "total_median_seconds": baseline_total,
            "solve_seconds": solve_seconds["baseline"],
            "solve_median_seconds": baseline_solve,
            "bulk_hot_seconds": hot_seconds["baseline"],
            "bulk_hot_median_seconds": baseline_hot,
        },
        "surface_cases": surface_results,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="Benchmark local CPU surface-impedance pole updates."
    )
    parser.add_argument("--cells", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kernel-iterations", type=int, default=1000)
    parser.add_argument("--kernel-repeats", type=int, default=3)
    parser.add_argument("--hot-iterations", type=int, default=250)
    parser.add_argument("--hot-repeats", type=int, default=3)
    parser.add_argument("--fit-band", nargs=2, type=float, default=(8e9, 12e9))
    parser.add_argument("--fit-tolerance", type=float, default=2e-3)
    parser.add_argument("--explicit-orders", nargs="+", type=int, default=(4, 8, 16, 32))
    parser.add_argument("--output", type=Path)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if (
        args.cells < 24
        or args.iterations <= 0
        or args.threads <= 0
        or args.repeats <= 0
        or args.kernel_iterations <= 0
        or args.kernel_repeats <= 0
        or args.hot_iterations <= 0
        or args.hot_repeats <= 0
        or args.fit_tolerance <= 0
        or args.fit_band[0] <= 0
        or args.fit_band[1] <= args.fit_band[0]
        or any(value <= 0 for value in args.explicit_orders)
    ):
        parser.error(
            "cells must be >=24; the fit band must be increasing and positive; "
            "all iteration/thread/repeat/order/tolerance values must be positive"
        )
    results = run_benchmark(args)
    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
