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

"""Benchmark the incremental wall-clock cost of KSIR frequency collection.

The benchmark brackets monitored cases with unmonitored baseline runs. Each
case requests one far-field direction and does not save surface phasors, so
the measured increase is dominated by timestep collection rather than angular
post-processing or extra HDF5 datasets.

Run from the repository root, for example::

    python -m testing.benchmarking.benchmark_ntff --backend cpu --threads 8
    python -m testing.benchmarking.benchmark_ntff --backend cuda --device 0
"""

import argparse
import json
import logging
import os
import platform
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

import gprMax
import gprMax.config as config

ACCELERATOR_ARGUMENTS = {"cuda": "gpu", "opencl": "opencl", "metal": "metal"}


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _monitor_bounds(cells, surface_cells, pml_cells, dl):
    """Return centred bounds with at least two free cells before the PML."""

    if surface_cells % 2:
        raise ValueError("surface cell counts must be even")
    lower = (cells - surface_cells) // 2
    upper = lower + surface_cells
    if lower < pml_cells + 2 or upper > cells - pml_cells - 2:
        raise ValueError(
            f"surface size {surface_cells} does not leave two cells beyond "
            f"a {pml_cells}-cell PML in a {cells}-cell domain"
        )
    return (lower * dl,) * 3, (upper * dl,) * 3


def _frequencies(count, centre_frequency):
    if count == 1:
        return [centre_frequency]
    return np.linspace(
        0.6 * centre_frequency,
        1.4 * centre_frequency,
        count,
    ).tolist()


def _solver_options(backend, device, precision):
    if backend == "cpu":
        return {"cpu_precision": precision}
    return {
        ACCELERATOR_ARGUMENTS[backend]: [device],
        "gpu_precision": precision,
    }


def _scene(args, surface_cells=None, frequency_count=0):
    extent = args.cells * args.dl
    centre = (0.5 * extent,) * 3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(args.dl,) * 3))
    scene.add(gprMax.Domain(p1=(extent,) * 3))
    scene.add(gprMax.TimeWindow(iterations=args.iterations))
    scene.add(gprMax.PMLThickness(thickness=args.pml_cells))
    scene.add(gprMax.OMPThreads(n=args.threads))
    scene.add(
        gprMax.Waveform(
            wave_type="ricker",
            amp=1,
            freq=args.centre_frequency,
            id="pulse",
        )
    )
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z",
            p1=centre,
            waveform_id="pulse",
        )
    )

    transform = None
    if surface_cells is not None:
        lower, upper = _monitor_bounds(args.cells, surface_cells, args.pml_cells, args.dl)
        scene.add(
            gprMax.NTFFSurface(
                p1=lower,
                p2=upper,
                id="benchmark_surface",
                origin=centre,
            )
        )
        transform = gprMax.KSIRFrequencyTransform(
            surface_id="benchmark_surface",
            id="benchmark_spectrum",
            frequencies=_frequencies(frequency_count, args.centre_frequency),
            save_surface_dft=False,
        )
        far_field = gprMax.KSIRFarField(
            theta=(90.0,),
            phi=(0.0,),
            transform_id="benchmark_spectrum",
            id="benchmark_direction",
            outputs=("Etheta",),
        )
        scene.add(transform)
        scene.add(far_field)
    return scene, transform


def _run_once(args, output_directory, label, surface_cells=None, frequency_count=0):
    scene, transform = _scene(args, surface_cells, frequency_count)
    outputfile = output_directory / label
    start = perf_counter()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **_solver_options(args.backend, args.device, args.precision),
    )
    elapsed = perf_counter() - start

    component_patches = 0
    collection_backend = None
    if transform is not None:
        component_patches = sum(
            component.surface.npatches for component in transform.surface_data.values()
        )
        with h5py.File(outputfile.with_suffix(".h5"), "r") as output:
            group = output["ntff/benchmark_surface/frequency/benchmark_spectrum"]
            collection_backend = group.attrs["collection_backend"]
            if isinstance(collection_backend, bytes):
                collection_backend = collection_backend.decode()
    return elapsed, component_patches, collection_backend


def _median_runs(
    args,
    output_directory,
    label,
    surface_cells=None,
    frequency_count=0,
):
    times = []
    patches = 0
    collection_backend = None
    for repeat in range(args.repeats):
        elapsed, patches, collection_backend = _run_once(
            args,
            output_directory,
            f"{label}_{repeat}",
            surface_cells,
            frequency_count,
        )
        times.append(elapsed)
    return statistics.median(times), times, patches, collection_backend


def run_benchmark(args):
    """Run the requested matrix and return JSON-serialisable measurements."""

    for surface_cells in args.surface_cells:
        _monitor_bounds(args.cells, surface_cells, args.pml_cells, args.dl)

    with tempfile.TemporaryDirectory(prefix="gprmax_ntff_benchmark_") as temporary:
        output_directory = Path(temporary)
        for warmup in range(args.warmups):
            _run_once(args, output_directory, f"warmup_{warmup}")
            _run_once(
                args,
                output_directory,
                f"warmup_ntff_{warmup}",
                max(args.surface_cells),
                max(args.frequency_counts),
            )

        _, baseline_before, _, _ = _median_runs(args, output_directory, "baseline")
        cases = []
        for surface_cells in args.surface_cells:
            for frequency_count in args.frequency_counts:
                label = f"surface_{surface_cells}_frequencies_{frequency_count}"
                elapsed, times, patches, collection_backend = _median_runs(
                    args,
                    output_directory,
                    label,
                    surface_cells,
                    frequency_count,
                )
                cases.append(
                    {
                        "surface_cells": surface_cells,
                        "frequency_count": frequency_count,
                        "component_patches": patches,
                        "collection_backend": collection_backend,
                        "median_seconds": elapsed,
                        "run_seconds": times,
                    }
                )
        _, baseline_after, _, _ = _median_runs(args, output_directory, "baseline_after")
        baseline_times = baseline_before + baseline_after
        baseline = statistics.median(baseline_times)
        for case in cases:
            case["slowdown"] = case["median_seconds"] / baseline
            case["overhead_percent"] = 100 * (case["slowdown"] - 1)

    hostinfo = config.sim_config.hostinfo or {}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "cpu_id": hostinfo.get("cpuID"),
            "physical_cores": hostinfo.get("physicalcores"),
        },
        "software": {
            "gprmax": gprMax.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "configuration": {
            "backend": args.backend,
            "device": None if args.backend == "cpu" else args.device,
            "cells": args.cells,
            "iterations": args.iterations,
            "dl": args.dl,
            "pml_cells": args.pml_cells,
            "threads": args.threads,
            "precision": args.precision,
            "centre_frequency": args.centre_frequency,
            "components": ["Ex", "Ey", "Ez"],
            "surface_cells": args.surface_cells,
            "frequency_counts": args.frequency_counts,
            "repeats": args.repeats,
            "warmups": args.warmups,
        },
        "baseline": {
            "median_seconds": baseline,
            "run_seconds": baseline_times,
            "before_seconds": baseline_before,
            "after_seconds": baseline_after,
        },
        "cases": cases,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="Benchmark incremental reusable-KSIR collection overhead."
    )
    parser.add_argument("--backend", choices=("cpu", "cuda", "opencl", "metal"), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cells", type=_positive_int, default=80)
    parser.add_argument("--iterations", type=_positive_int, default=300)
    parser.add_argument("--dl", type=float, default=0.002)
    parser.add_argument("--pml-cells", type=_positive_int, default=10)
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--centre-frequency", type=float, default=4e9)
    parser.add_argument("--surface-cells", nargs="+", type=_positive_int, default=[24, 40, 56])
    parser.add_argument("--frequency-counts", nargs="+", type=_positive_int, default=[1, 5, 10])
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--warmups", type=int, choices=range(0, 10), default=1)
    parser.add_argument("--output", type=Path, default=Path("ntff_benchmark_results.json"))
    return parser


def main():
    args = _parser().parse_args()
    results = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
