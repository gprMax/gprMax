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

"""Benchmark Apple Metal and CPU throughput on identical gprMax scenes.

Run from the repository root, for example::

    python -m testing.benchmarking.benchmark_metal --sizes 50 100 150
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

import gprMax

DEFAULT_SIZES = (50, 75, 100, 125, 150, 175, 200, 250, 300, 400)


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_scene(size, *, cell_size, iterations, threads):
    """Build one two-material cube used by both benchmark backends."""

    extent = size * cell_size
    centre = (extent / 2,) * 3
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f"Metal benchmark {size}x{size}x{size}"))
    scene.add(gprMax.Domain(p1=(extent,) * 3))
    scene.add(gprMax.Discretisation(p1=(cell_size,) * 3))
    scene.add(gprMax.TimeWindow(iterations=iterations))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.Material(er=6, se=0, mr=1, sm=0, id="half_space"))
    scene.add(
        gprMax.Waveform(
            wave_type="ricker",
            amp=1,
            freq=1.5e9,
            id="my_ricker",
        )
    )
    scene.add(
        gprMax.HertzianDipole(
            p1=centre,
            polarisation="x",
            waveform_id="my_ricker",
        )
    )
    scene.add(gprMax.Rx(p1=centre, outputs=["Ex"]))
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(extent, extent / 2, extent),
            material_id="half_space",
        )
    )
    return scene


def _solver_options(backend, device):
    if backend == "metal":
        return {"metal": [device], "gpu_precision": "single"}
    return {"cpu_precision": "single"}


def _run_once(args, output_directory, size, backend, repeat):
    outputfile = output_directory / f"{backend}_{size}_{repeat}"
    started = perf_counter()
    gprMax.run(
        scenes=[
            build_scene(
                size,
                cell_size=args.cell_size,
                iterations=args.iterations,
                threads=args.threads,
            )
        ],
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **_solver_options(backend, args.device),
    )
    return perf_counter() - started


def _backend_result(size, iterations, times):
    median_seconds = statistics.median(times)
    return {
        "run_seconds": times,
        "median_seconds": median_seconds,
        "performance_mcells_per_second": (
            size**3 * iterations / (median_seconds * 1e6)
        ),
    }


def run_benchmark(args):
    """Execute the CPU/Metal matrix and return JSON-serialisable results."""

    cases = []
    with tempfile.TemporaryDirectory(prefix="gprmax_metal_benchmark_") as temporary:
        output_directory = Path(temporary)
        for size in args.sizes:
            for warmup in range(args.warmups):
                _run_once(args, output_directory, size, "cpu", f"warmup_{warmup}")
                _run_once(args, output_directory, size, "metal", f"warmup_{warmup}")

            times = {"cpu": [], "metal": []}
            for repeat in range(args.repeats):
                backends = ("cpu", "metal") if repeat % 2 == 0 else ("metal", "cpu")
                for backend in backends:
                    times[backend].append(
                        _run_once(args, output_directory, size, backend, repeat)
                    )

            cpu = _backend_result(size, args.iterations, times["cpu"])
            metal = _backend_result(size, args.iterations, times["metal"])
            cases.append(
                {
                    "size": size,
                    "cells": size**3,
                    "cpu": cpu,
                    "metal": metal,
                    "speedup": (
                        metal["performance_mcells_per_second"]
                        / cpu["performance_mcells_per_second"]
                    ),
                }
            )

    return {
        "configuration": {
            "sizes": list(args.sizes),
            "cell_size": args.cell_size,
            "iterations": args.iterations,
            "threads": args.threads,
            "metal_device": args.device,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "precision": "single",
        },
        "cases": cases,
    }


def create_performance_plot(results, path, *, show=False):
    """Write throughput and speedup plots for a benchmark result."""

    import matplotlib.pyplot as plt

    cases = results["cases"]
    sizes = [case["size"] for case in cases]
    cpu = [case["cpu"]["performance_mcells_per_second"] for case in cases]
    metal = [case["metal"]["performance_mcells_per_second"] for case in cases]
    speedups = [case["speedup"] for case in cases]

    figure, (throughput_axis, speedup_axis) = plt.subplots(1, 2, figsize=(15, 6))
    throughput_axis.plot(sizes, metal, "ro-", label="Apple Metal")
    throughput_axis.plot(sizes, cpu, "bo-", label="CPU (OpenMP)")
    throughput_axis.set_xlabel("Domain size [cells per side]")
    throughput_axis.set_ylabel("Performance [Mcells/s]")
    throughput_axis.set_title("gprMax throughput")
    throughput_axis.legend()
    throughput_axis.grid(True, alpha=0.3)

    speedup_axis.plot(sizes, speedups, "go-")
    speedup_axis.axhline(1.0, color="red", linestyle="--", alpha=0.7)
    speedup_axis.set_xlabel("Domain size [cells per side]")
    speedup_axis.set_ylabel("Metal/CPU speedup")
    speedup_axis.set_title("Apple Metal speedup")
    speedup_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=_positive_int, default=DEFAULT_SIZES)
    parser.add_argument("--cell-size", type=_positive_float, default=0.001)
    parser.add_argument("--iterations", type=_positive_int, default=1500)
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--warmups", type=_nonnegative_int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metal_benchmark_results.json"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("metal_benchmark_results.png"),
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = _parser().parse_args()
    results = run_benchmark(args)
    payload = json.dumps(results, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    if not args.no_plot:
        create_performance_plot(results, args.plot, show=args.show)
    print(payload)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
