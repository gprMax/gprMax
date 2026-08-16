# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Command-line interface for impulse-response waveform synthesis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .processing import (
    BUILTIN_WAVEFORM_TYPES,
    list_receivers,
    list_sources,
    load_csv_waveforms,
    load_source_sampling,
    sample_builtin_waveform,
    synthesise_output,
    write_synthesised_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m toolboxes.ImpulseResponse",
        description="Synthesize receiver outputs for many pulses from one impulse FDTD run.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="list stored scalar sources and receivers")
    inspect.add_argument("input", type=Path)

    process = commands.add_parser("process", help="synthesize and save receiver histories")
    process.add_argument("input", type=Path, help="impulse-excited gprMax HDF5 output")
    process.add_argument(
        "--source-file",
        type=Path,
        help="file containing the impulse source history (for a merged B-scan)",
    )
    process.add_argument("--source", help="HDF5 scalar source group, e.g. /srcs/src1")
    process.add_argument(
        "--waveform",
        nargs=4,
        action="append",
        metavar=("TYPE", "AMPLITUDE", "FREQUENCY", "ID"),
        help="built-in gprMax waveform; may be repeated",
    )
    process.add_argument(
        "--waveform-file",
        type=Path,
        action="append",
        help="CSV with a time column and one or more named waveform columns",
    )
    process.add_argument(
        "--receiver",
        action="append",
        metavar="PATH:COMPONENT",
        help="receiver component to synthesize; omitted means every stored receiver output",
    )
    process.add_argument("--start-time", type=float, default=0.0, help="source start time [s]")
    process.add_argument(
        "--stop-time",
        type=float,
        help="optional stop time for built-in waveforms [s]",
    )
    process.add_argument(
        "--tail-taper",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="raised-cosine taper applied to the impulse-response tail",
    )
    process.add_argument(
        "--max-frequency",
        type=float,
        help="highest frequency considered numerically valid [Hz]",
    )
    process.add_argument(
        "--output-dir",
        type=Path,
        help="directory for one receiver-compatible HDF5 file per waveform",
    )
    process.add_argument("--plot", type=Path, help="optional summary plot filename")
    process.add_argument("--show", action="store_true", help="display the summary plot")
    return parser


def _parse_receivers(values: list[str] | None):
    if values is None:
        return None
    result: list[tuple[str, str]] = []
    for value in values:
        try:
            path, component = value.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError("receiver selections must use PATH:COMPONENT") from exc
        if not path or not component:
            raise ValueError("receiver selections must use PATH:COMPONENT")
        result.append((path, component))
    return result


def _target_waveforms(args, source):
    result = []
    for specification in args.waveform or ():
        waveform_type, amplitude, frequency, waveform_id = specification
        if waveform_type.lower() not in BUILTIN_WAVEFORM_TYPES:
            raise ValueError(
                f"unknown waveform {waveform_type!r}; choose from {BUILTIN_WAVEFORM_TYPES}"
            )
        result.append(
            sample_builtin_waveform(
                source,
                waveform_type,
                float(amplitude),
                float(frequency),
                waveform_id,
                start_time=args.start_time,
                stop_time=args.stop_time,
            )
        )
    for filename in args.waveform_file or ():
        result.extend(load_csv_waveforms(filename, source, start_time=args.start_time))
    if not result:
        raise ValueError("supply at least one --waveform or --waveform-file")
    identifiers = [item.id for item in result]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        raise ValueError(f"waveform IDs must be unique; duplicates: {duplicates}")
    return result


def _inspect(filename: Path) -> None:
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        _inspect(args.input)
        return 0

    source_file = args.source_file or args.input
    source = load_source_sampling(source_file, args.source)
    waveforms = _target_waveforms(args, source)
    selections = _parse_receivers(args.receiver)
    output_dir = args.output_dir or args.input.with_name(f"{args.input.stem}_waveforms")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for waveform in waveforms:
        result = synthesise_output(
            args.input,
            waveform,
            source_filename=source_file,
            source_path=source.signal.path,
            receiver_selections=selections,
            tail_taper_fraction=args.tail_taper,
            valid_max_frequency=args.max_frequency,
        )
        output = write_synthesised_output(
            output_dir / f"{args.input.stem}_{waveform.id}.h5",
            result,
        )
        results.append(result)
        print(f"Written {waveform.id}: {output}")
        worst_tail = max(item.impulse_tail_relative_db for item in result.receivers)
        print(f"  worst impulse-response tail: {worst_tail:.1f} dB")
        if worst_tail > -60 and args.tail_taper == 0:
            print("  Warning: extend the FDTD time window or inspect the late response.")
        if result.energy_above_valid_band is not None:
            print(
                "  spectral energy above the stated valid band: "
                f"{100 * result.energy_above_valid_band:.4g}%"
            )

    if args.plot is not None or args.show:
        from .plotting import plot_synthesis_results

        plot_synthesis_results(results, output=args.plot, show=args.show)
        if args.plot is not None:
            print(f"Written summary plot: {args.plot}")
    return 0
