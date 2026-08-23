# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Command-line interface for timing-aware SFCW post-processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .processing import (
    list_receivers,
    list_sources,
    process_output,
    reconstruct_time_response,
    write_sfcw_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m toolboxes.SFCW",
        description="Synthesize stepped-frequency responses from a gprMax output.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="list stored sources and receivers")
    inspect.add_argument("input", type=Path)

    process = commands.add_parser("process", help="calculate and save an SFCW response")
    process.add_argument("input", type=Path, help="gprMax HDF5 output file")
    process.add_argument(
        "--source-file",
        type=Path,
        help="file containing the source history (needed for a merged B-scan)",
    )
    process.add_argument("--source", help="HDF5 source group, e.g. /srcs/src1")
    process.add_argument("--receiver", help="HDF5 receiver group, e.g. /rxs/rx1")
    process.add_argument("--component", help="receiver component, e.g. Ez")
    process.add_argument("--f-start", type=float, required=True, help="first frequency [Hz]")
    process.add_argument("--f-stop", type=float, required=True, help="last frequency [Hz]")
    process.add_argument("--steps", type=int, required=True, help="number of frequency steps")
    process.add_argument("--method", choices=("direct", "homodyne"), default="direct")
    process.add_argument(
        "--window",
        choices=("rectangular", "gaussian", "hann", "hamming", "blackman"),
        default="gaussian",
    )
    process.add_argument("--gaussian-sigma", type=float, default=0.2)
    process.add_argument("--zero-pad", type=int, default=1, metavar="FACTOR")
    process.add_argument(
        "--time-shift",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="display delay applied before the periodic inverse FFT",
    )
    process.add_argument("--source-floor-db", type=float, default=-100.0)
    process.add_argument("--homodyne-cycles", type=int, default=8)
    process.add_argument(
        "--tail-taper",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="raised-cosine taper applied to the end of the receiver history",
    )
    process.add_argument("--output", type=Path, help="processed HDF5 output filename")
    process.add_argument("--plot", type=Path, help="optional PNG/PDF plot filename")
    process.add_argument("--show", action="store_true", help="display the plot interactively")
    return parser


def _validate_frequency_arguments(args) -> np.ndarray:
    if not np.isfinite(args.f_start) or not np.isfinite(args.f_stop):
        raise ValueError("frequency limits must be finite")
    if args.f_start <= 0 or args.f_stop <= args.f_start:
        raise ValueError("frequency limits must satisfy 0 < f-start < f-stop")
    if args.steps < 2:
        raise ValueError("steps must be at least two")
    return np.linspace(args.f_start, args.f_stop, args.steps, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        print("Sources:")
        sources = list_sources(args.input)
        print("\n".join(f"  {path}" for path in sources) if sources else "  none")
        print("Receivers:")
        receivers = list_receivers(args.input)
        if receivers:
            for path, components in receivers.items():
                print(f"  {path}: {', '.join(components)}")
        else:
            print("  none")
        return 0

    frequencies = _validate_frequency_arguments(args)
    result = process_output(
        args.input,
        frequencies,
        source_filename=args.source_file,
        source_path=args.source,
        receiver_path=args.receiver,
        component=args.component,
        method=args.method,
        source_floor_db=args.source_floor_db,
        homodyne_cycles=args.homodyne_cycles,
        tail_taper_fraction=args.tail_taper,
    )
    invalid = int(np.count_nonzero(~result.source_valid))
    if invalid:
        raise ValueError(
            f"the source spectrum is below the requested floor at {invalid} frequency steps"
        )
    time_response = reconstruct_time_response(
        result,
        window=args.window,
        gaussian_sigma=args.gaussian_sigma,
        zero_pad_factor=args.zero_pad,
        time_shift=args.time_shift,
    )
    output = args.output or args.input.with_name(f"{args.input.stem}_sfcw.h5")
    write_sfcw_output(output, result, time_response)
    print(f"Written SFCW output: {output}")
    print(f"Receiver tail level: {result.receiver_tail_relative_db:.1f} dB")
    if result.receiver_tail_relative_db > -60 and args.tail_taper == 0:
        print(
            "Warning: the receiver tail exceeds -60 dB. Extend the FDTD time window "
            "or use --tail-taper after checking that late physical arrivals are not removed."
        )
    if args.plot is not None or args.show:
        from .plotting import plot_sfcw_result

        plot_sfcw_result(result, time_response, output=args.plot, show=args.show)
        if args.plot is not None:
            print(f"Written SFCW plot: {args.plot}")
    return 0
