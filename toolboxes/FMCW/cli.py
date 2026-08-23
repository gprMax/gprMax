# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Command-line interface for FMCW synthesis from gprMax outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.constants import c

from toolboxes.SFCW.processing import list_receivers, list_sources

from .processing import (
    Chirp,
    load_instrument_response,
    load_receiver_delay_response,
    process_channel,
    process_incident_referenced_channel,
    reconstruct_fast_time,
    synthesize_deramped_sweep,
    write_fmcw_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m toolboxes.FMCW",
        description="Synthesize FMCW GPR products from a broadband gprMax output.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="list stored sources and receivers")
    inspect.add_argument("input", type=Path)

    process = commands.add_parser("process", help="calculate and save an FMCW response")
    process.add_argument("input", type=Path, help="target-model gprMax HDF5 output")
    process.add_argument("--background", type=Path, help="empty/reference-model HDF5 output")
    process.add_argument("--source-file", type=Path)
    process.add_argument("--background-source-file", type=Path)
    process.add_argument("--source", help="target HDF5 source group")
    process.add_argument("--receiver", help="target HDF5 receiver group")
    process.add_argument("--component", help="receiver component, e.g. Ez")
    process.add_argument("--background-source", help="background HDF5 source group")
    process.add_argument("--background-receiver", help="background HDF5 receiver group")
    process.add_argument("--background-component", help="background receiver component")
    process.add_argument(
        "--incident-reference",
        action="store_true",
        help="normalise total-minus-background by the background receiver instead of a stored source",
    )
    process.add_argument("--f-start", type=float, required=True, help="lower sweep edge [Hz]")
    process.add_argument("--f-stop", type=float, required=True, help="upper sweep edge [Hz]")
    process.add_argument("--samples", type=int, required=True, help="samples in one sweep")
    process.add_argument("--sweep-time", type=float, required=True, help="sweep duration [s]")
    process.add_argument("--direction", choices=("up", "down"), default="up")
    process.add_argument(
        "--window",
        choices=("rectangular", "gaussian", "hann", "hamming", "blackman"),
        default="blackman",
    )
    process.add_argument(
        "--receiver-delay-response",
        type=Path,
        help="CSV range-gate/IF response: delay_s plus gain, real/imag, or magnitude/phase_deg",
    )
    process.add_argument("--gaussian-sigma", type=float, default=0.2)
    process.add_argument(
        "--instrument-response",
        type=Path,
        help="CSV transfer function: frequency_hz plus real/imag or magnitude/phase_deg",
    )
    process.add_argument("--source-floor-db", type=float, default=-100.0)
    process.add_argument("--tail-taper", type=float, default=0.0, metavar="FRACTION")
    process.add_argument(
        "--relative-permittivity",
        type=float,
        metavar="ER",
        help="also store the two-way homogeneous-medium range axis",
    )
    process.add_argument(
        "--rvp",
        choices=("neglect", "include"),
        default="neglect",
        help="include or neglect residual video phase in the fast-time result",
    )
    process.add_argument(
        "--deramped",
        action="store_true",
        help="also store ideal complex stretch-receiver I/Q samples",
    )
    process.add_argument(
        "--deramped-rvp",
        choices=("neglect", "include"),
        default="neglect",
    )
    process.add_argument("--output", type=Path)
    process.add_argument("--plot", type=Path)
    process.add_argument("--show", action="store_true")
    return parser


def _inspect(filename: Path) -> int:
    print("Sources:")
    sources = list_sources(filename)
    print("\n".join(f"  {path}" for path in sources) if sources else "  none")
    print("Receivers:")
    receivers = list_receivers(filename)
    if receivers:
        for path, components in receivers.items():
            print(f"  {path}: {', '.join(components)}")
    else:
        print("  none")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.input)

    chirp = Chirp(
        f_start=args.f_start,
        f_stop=args.f_stop,
        sweep_time=args.sweep_time,
        samples=args.samples,
        direction=args.direction,
    )
    if args.incident_reference:
        if args.background is None:
            raise ValueError("--incident-reference requires --background")
        channel = process_incident_referenced_channel(
            args.input,
            args.background,
            chirp,
            receiver_path=args.receiver,
            component=args.component,
            incident_receiver_path=args.background_receiver,
            incident_component=args.background_component,
            source_floor_db=args.source_floor_db,
            tail_taper_fraction=args.tail_taper,
        )
    else:
        channel = process_channel(
            args.input,
            chirp,
            background_filename=args.background,
            source_filename=args.source_file,
            background_source_filename=args.background_source_file,
            source_path=args.source,
            receiver_path=args.receiver,
            component=args.component,
            background_source_path=args.background_source,
            background_receiver_path=args.background_receiver,
            background_component=args.background_component,
            source_floor_db=args.source_floor_db,
            tail_taper_fraction=args.tail_taper,
        )
    invalid = int(np.count_nonzero(~channel.source_valid))
    if invalid:
        raise ValueError(f"the source spectrum is below the requested floor at {invalid} FMCW samples")
    instrument = None
    if args.instrument_response is not None:
        instrument = load_instrument_response(args.instrument_response, chirp.frequency)
    delay_response = None
    if args.receiver_delay_response is not None:
        delay_response = load_receiver_delay_response(
            args.receiver_delay_response,
            chirp.delay,
        )
    velocity = None
    if args.relative_permittivity is not None:
        if not np.isfinite(args.relative_permittivity) or args.relative_permittivity <= 0:
            raise ValueError("relative-permittivity must be finite and positive")
        velocity = c / np.sqrt(args.relative_permittivity)
    fast = reconstruct_fast_time(
        channel,
        instrument_response=instrument,
        receiver_delay_response=delay_response,
        window=args.window,
        gaussian_sigma=args.gaussian_sigma,
        propagation_velocity=velocity,
        residual_video_phase=args.rvp,
    )
    deramped = None
    if args.deramped:
        deramped = synthesize_deramped_sweep(
            channel,
            instrument_response=instrument,
            residual_video_phase=args.deramped_rvp,
        )
    output = args.output or args.input.with_name(f"{args.input.stem}_fmcw.h5")
    write_fmcw_output(output, channel, fast, deramped)
    print(f"Written FMCW output: {output}")
    print(f"Delay resolution: {1 / chirp.bandwidth:.6g} s")
    print(f"Unambiguous periodic delay: {chirp.samples / chirp.bandwidth:.6g} s")
    print(f"Target receiver tail level: {channel.target.receiver_tail_relative_db:.1f} dB")
    if channel.target.receiver_tail_relative_db > -60 and args.tail_taper == 0:
        print(
            "Warning: the target receiver tail exceeds -60 dB. Extend the FDTD time "
            "window or use --tail-taper only after checking for physical late arrivals."
        )
    if args.plot is not None or args.show:
        from .plotting import plot_fmcw_result

        plot_fmcw_result(channel, fast, deramped, output=args.plot, show=args.show)
        if args.plot is not None:
            print(f"Written FMCW plot: {args.plot}")
    return 0
