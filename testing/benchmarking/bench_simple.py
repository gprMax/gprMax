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

"""Run a simple free-space CPU benchmark over domain and thread matrices."""

import argparse
import itertools
from pathlib import Path

import gprMax


DEFAULT_DOMAINS = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
DEFAULT_THREADS = (1, 2, 4, 8, 16, 32, 64, 128)


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_scenes(domains, threads, cell_size=0.001, time_window=3e-9):
    """Build the requested Cartesian product of benchmark scenes."""

    scenes = []
    title = Path(__file__).stem
    for extent, thread_count in itertools.product(domains, threads):
        scene = gprMax.Scene()
        scene.add(gprMax.Title(name=title))
        scene.add(gprMax.Domain(p1=(extent,) * 3))
        scene.add(gprMax.Discretisation(p1=(cell_size,) * 3))
        scene.add(gprMax.TimeWindow(time=time_window))
        scene.add(
            gprMax.Waveform(
                wave_type="gaussiandotnorm",
                amp=1,
                freq=900e6,
                id="MySource",
            )
        )
        scene.add(
            gprMax.HertzianDipole(
                p1=(extent / 2,) * 3,
                polarisation="x",
                waveform_id="MySource",
            )
        )
        scene.add(gprMax.OMPThreads(n=thread_count))
        scenes.append(scene)
    return scenes


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        type=_positive_float,
        default=DEFAULT_DOMAINS,
        help="cube side lengths in metres",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=_positive_int,
        default=DEFAULT_THREADS,
        help="OpenMP thread counts",
    )
    parser.add_argument("--cell-size", type=_positive_float, default=0.001)
    parser.add_argument("--time-window", type=_positive_float, default=3e-9)
    parser.add_argument("--output", type=Path, default=Path(__file__))
    return parser


def main():
    args = _parser().parse_args()
    scenes = build_scenes(
        args.domains,
        args.threads,
        cell_size=args.cell_size,
        time_window=args.time_window,
    )
    gprMax.run(
        scenes=scenes,
        n=len(scenes),
        geometry_only=False,
        outputfile=args.output,
    )


if __name__ == "__main__":
    main()
