"""Benchmark the CPU overhead of a closed surface-impedance box.

Example::

    python -m testing.benchmarking.benchmark_impedance_box --threads 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

import gprMax

_SOLVE_SECONDS = []


def build_scene(cells: int, iterations: int, threads: int, with_surface: bool):
    dl = 0.001
    extent = cells * dl
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(extent, extent, extent)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(iterations=iterations))
    scene.add(gprMax.PMLThickness(thickness=5))
    scene.add(gprMax.OMPThreads(threads))
    if with_surface:
        lower = 0.25 * extent
        upper = 0.75 * extent
        scene.add(gprMax.SurfaceImpedance(id="benchmark_wall", resistance=50.0))
        scene.add(gprMax.ImpedanceBox((lower,) * 3, (upper,) * 3, "benchmark_wall"))
    return scene


def timed_run(path: Path, *, cells, iterations, threads, with_surface):
    started = perf_counter()
    gprMax.run(
        scenes=[build_scene(cells, iterations, threads, with_surface)],
        outputfile=path,
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )
    return perf_counter() - started, _SOLVE_SECONDS[-1]


def run_benchmark(args):
    import gprMax.impedance_surfaces as implementation
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.solvers import Solver
    from gprMax.updates.cpu_updates import CPUUpdates

    _SOLVE_SECONDS.clear()
    captured = {}
    original_compile = implementation.compile_impedance_surfaces

    def capture(grid):
        system = original_compile(grid)
        captured["grid"] = grid
        captured["system"] = system
        return system

    built_grids = []
    original_grid_build = FDTDGrid.build

    def capture_grid(self):
        result = original_grid_build(self)
        built_grids.append(self)
        return result

    original_solve = Solver.solve

    def measure_solve(self, iterator):
        started = perf_counter()
        try:
            return original_solve(self, iterator)
        finally:
            _SOLVE_SECONDS.append(perf_counter() - started)

    class BenchmarkPatches:
        def __enter__(self):
            implementation.compile_impedance_surfaces = capture
            FDTDGrid.build = capture_grid
            Solver.solve = measure_solve

        def __exit__(self, *_):
            implementation.compile_impedance_surfaces = original_compile
            FDTDGrid.build = original_grid_build
            Solver.solve = original_solve

    with BenchmarkPatches(), tempfile.TemporaryDirectory(
        prefix="gprmax_impedance_benchmark_", dir=Path.cwd()
    ) as tmp:
        directory = Path(tmp)
        timed_run(
            directory / "warmup",
            cells=args.cells,
            iterations=max(10, args.iterations // 10),
            threads=args.threads,
            with_surface=False,
        )
        timed_run(
            directory / "warmup_impedance",
            cells=args.cells,
            iterations=max(10, args.iterations // 10),
            threads=args.threads,
            with_surface=True,
        )
        baseline = []
        impedance = []
        baseline_solve = []
        impedance_solve = []
        for index in range(args.repeats):
            order = (False, True) if index % 2 == 0 else (True, False)
            for with_surface in order:
                elapsed, solve_elapsed = timed_run(
                    directory / f"{'impedance' if with_surface else 'baseline'}_{index}",
                    cells=args.cells,
                    iterations=args.iterations,
                    threads=args.threads,
                    with_surface=with_surface,
                )
                (impedance if with_surface else baseline).append(elapsed)
                (impedance_solve if with_surface else baseline_solve).append(solve_elapsed)

        grid = captured["grid"]
        system = captured["system"]
        grid.Hx.fill(0.1)
        grid.Hy.fill(-0.2)
        grid.Hz.fill(0.3)
        started = perf_counter()
        for _ in range(args.kernel_iterations):
            system.update(grid)
        kernel_seconds = perf_counter() - started

        baseline_grid = next(
            item for item in reversed(built_grids) if item.impedance_surfaces is None
        )
        impedance_grid = next(
            item for item in reversed(built_grids) if item.impedance_surfaces is not None
        )
        baseline_grid.reset_fields()
        impedance_grid.reset_fields()
        baseline_updates = CPUUpdates(baseline_grid)
        impedance_updates = CPUUpdates(impedance_grid)

        def hot_loop(updates, count):
            for _ in range(count):
                updates.update_magnetic()
                updates.update_electric_a()
                updates.update_impedance_surfaces()

        hot_loop(baseline_updates, 5)
        hot_loop(impedance_updates, 5)

        def measure_hot(grid, updates):
            grid.reset_fields()
            started = perf_counter()
            hot_loop(updates, args.hot_iterations)
            return perf_counter() - started

        baseline_hot = []
        impedance_hot = []
        for repeat in range(args.hot_repeats):
            if repeat % 2 == 0:
                baseline_hot.append(measure_hot(baseline_grid, baseline_updates))
                impedance_hot.append(measure_hot(impedance_grid, impedance_updates))
            else:
                impedance_hot.append(measure_hot(impedance_grid, impedance_updates))
                baseline_hot.append(measure_hot(baseline_grid, baseline_updates))

    baseline_median = statistics.median(baseline)
    impedance_median = statistics.median(impedance)
    baseline_solve_median = statistics.median(baseline_solve)
    impedance_solve_median = statistics.median(impedance_solve)
    baseline_hot_median = statistics.median(baseline_hot)
    impedance_hot_median = statistics.median(impedance_hot)
    return {
        "domain_cells": args.cells,
        "iterations": args.iterations,
        "threads": args.threads,
        "repeats": args.repeats,
        "boundary_edges": system.edge_count,
        "surface_ports": system.port_count,
        "baseline_seconds": baseline,
        "impedance_seconds": impedance,
        "wall_median_overhead_percent": 100 * (impedance_median / baseline_median - 1),
        "baseline_solve_seconds": baseline_solve,
        "impedance_solve_seconds": impedance_solve,
        "solve_median_overhead_percent": 100
        * (impedance_solve_median / baseline_solve_median - 1),
        "sparse_kernel_seconds": kernel_seconds,
        "sparse_edge_updates_per_second": (
            args.kernel_iterations * system.edge_count / kernel_seconds
        ),
        "hot_loop_iterations": args.hot_iterations,
        "hot_loop_repeats": args.hot_repeats,
        "baseline_bulk_hot_seconds": baseline_hot,
        "impedance_bulk_hot_seconds": impedance_hot,
        "bulk_plus_surface_overhead_percent": 100
        * (impedance_hot_median / baseline_hot_median - 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kernel-iterations", type=int, default=1000)
    parser.add_argument("--hot-iterations", type=int, default=250)
    parser.add_argument("--hot-repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        args.cells < 24
        or args.iterations <= 0
        or args.threads <= 0
        or args.repeats <= 0
        or args.kernel_iterations <= 0
        or args.hot_iterations <= 0
        or args.hot_repeats <= 0
    ):
        parser.error("cells must be >=24 and all iteration/thread/repeat counts must be positive")
    print(json.dumps(run_benchmark(args), indent=2))


if __name__ == "__main__":
    main()
