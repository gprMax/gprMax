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

"""Benchmark the Cython planar-layered NTFF sum against its NumPy oracle.

Run from the repository root, for example::

    python -m testing.benchmarking.benchmark_layered_ntff --threads 8
"""

import argparse
import json
from time import perf_counter

import numpy as np

import gprMax.ntff.layered as layered
from gprMax.ntff.equivalent_currents import EquivalentCurrentPhasors


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def run_benchmark(args):
    """Return repeatable kernel timing and numerical-difference metrics."""

    if layered._evaluate_layered_cython is None:
        raise RuntimeError("the gprMax Cython extensions must be built")
    rng = np.random.default_rng(args.seed)
    frequencies = np.linspace(0.5e9, 3e9, args.frequencies)
    theta = np.linspace(2, 178, args.directions)
    theta = theta[np.abs(theta - 90) > 0.1]
    phi = np.deg2rad(37)
    directions = np.column_stack(
        (
            np.sin(np.deg2rad(theta)) * np.cos(phi),
            np.sin(np.deg2rad(theta)) * np.sin(phi),
            np.cos(np.deg2rad(theta)),
        )
    )
    shape = (args.frequencies, args.patches, 3)
    currents = EquivalentCurrentPhasors(
        positions=rng.uniform(-0.05, 0.05, (args.patches, 3)),
        normals=np.zeros((args.patches, 3)),
        area_weights=rng.uniform(1e-6, 4e-6, args.patches),
        electric_current=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        magnetic_current=rng.normal(size=shape) + 1j * rng.normal(size=shape),
    )
    medium = layered.LayeredMedium(
        axis="z",
        interfaces=np.asarray((0.02, -0.02)),
        material_ids=("upper", "film", "lower"),
        relative_permittivity=np.broadcast_to(
            np.asarray((1, 3.1 - 0.08j, 2.2)), (args.frequencies, 3)
        ).copy(),
        relative_permeability=np.broadcast_to(
            np.asarray((1, 1.15, 1)), (args.frequencies, 3)
        ).copy(),
    )

    start = perf_counter()
    accelerated = layered.evaluate_layered_currents(
        currents, frequencies, directions, medium, nthreads=args.threads
    )
    cython_seconds = perf_counter() - start

    cython_kernel = layered._evaluate_layered_cython
    try:
        layered._evaluate_layered_cython = None
        start = perf_counter()
        reference = layered.evaluate_layered_currents(currents, frequencies, directions, medium)
        numpy_seconds = perf_counter() - start
    finally:
        layered._evaluate_layered_cython = cython_kernel

    return {
        "frequencies": args.frequencies,
        "directions": int(directions.shape[0]),
        "patches": args.patches,
        "threads": args.threads,
        "cython_seconds": cython_seconds,
        "numpy_seconds": numpy_seconds,
        "speedup": numpy_seconds / cython_seconds,
        "maximum_electric_difference": float(
            np.max(np.abs(accelerated.electric - reference.electric))
        ),
        "maximum_magnetic_difference": float(
            np.max(np.abs(accelerated.magnetic - reference.magnetic))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequencies", type=_positive_int, default=16)
    parser.add_argument("--directions", type=_positive_int, default=360)
    parser.add_argument("--patches", type=_positive_int, default=1000)
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args), indent=2))


if __name__ == "__main__":
    main()
