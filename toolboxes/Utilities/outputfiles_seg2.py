# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Export a series of gprMax A-scans in SEG-2 format.

SEG-2 was designed for shallow seismic and digital-radar data. Its
``SAMPLE_INTERVAL`` trace keyword is a floating-point value in seconds, so a
normal FDTD time step can be represented without the resampling needed by
legacy GPR adaptations of SEG-Y. Trace data are stored as little-endian IEEE
single-precision values (SEG-2 data format code 4).
"""

from __future__ import annotations

import argparse
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from toolboxes.Utilities.outputfiles_trace import (
    TraceRecord,
    collect_traces,
    discover_files,
    quantity_units,
)


FILE_DESCRIPTOR_ID = 0x3A55
TRACE_DESCRIPTOR_ID = 0x4422
SEG2_REVISION = 1
IEEE_FLOAT32_FORMAT_CODE = 4
MAX_TRACES = 16_383
MAX_UINT16 = 65_535


@dataclass(frozen=True)
class Seg2ExportSummary:
    """Metadata describing a completed SEG-2 export."""

    outputfile: Path
    trace_count: int
    sample_count: int
    sample_interval: float
    receiver: int
    component: str
    source_path: str
    time_sample_offset: float


def _clean_text(value: object) -> str:
    return " ".join(str(value).replace("\x00", " ").replace("\n", " ").split())


def _free_form(strings: Iterable[tuple[str, object]]) -> bytes:
    """Build a four-byte-aligned SEG-2 free-format string block."""

    result = bytearray()
    for keyword, value in strings:
        text = f"{keyword} {_clean_text(value)}".rstrip().encode("ascii", errors="replace")
        size = 2 + len(text) + 1
        if size > MAX_UINT16:
            raise ValueError(f"SEG-2 free-format string {keyword!r} is too long")
        result.extend(struct.pack("<H", size))
        result.extend(text)
        result.append(0)
    result.extend(struct.pack("<H", 0))
    result.extend(bytes((-len(result)) % 4))
    return bytes(result)


def _location(position: tuple[float, float, float]) -> str:
    return " ".join(f"{coordinate:.17g}" for coordinate in position)


def _trace_block(
    record: TraceRecord,
    trace_number: int,
    *,
    dt: float,
    time_offset: float,
    component: str,
) -> bytes:
    sample_count = int(record.samples.size)
    data = np.asarray(record.samples, dtype="<f4").tobytes(order="C")
    strings = _free_form(
        (
            ("CHANNEL_NUMBER", 1),
            ("DELAY", f"{time_offset:.17g}"),
            ("DESCALING_FACTOR", 1),
            ("GPRMAX_COMPONENT", component),
            ("GPRMAX_UNITS", quantity_units(component)),
            ("RAW_RECORD", trace_number),
            ("RECEIVER_LOCATION", _location(record.receiver_position)),
            ("SAMPLE_INTERVAL", f"{dt:.17g}"),
            ("SOURCE_LOCATION", _location(record.source_position)),
            ("TRACE_TYPE", "RADAR_DATA"),
        )
    )
    descriptor_size = 32 + len(strings)
    if descriptor_size > 65_532:
        raise ValueError("SEG-2 trace descriptor exceeds 65532 bytes")
    descriptor = bytearray(32)
    struct.pack_into("<H", descriptor, 0, TRACE_DESCRIPTOR_ID)
    struct.pack_into("<H", descriptor, 2, descriptor_size)
    struct.pack_into("<I", descriptor, 4, len(data))
    struct.pack_into("<I", descriptor, 8, sample_count)
    descriptor[12] = IEEE_FLOAT32_FORMAT_CODE
    return bytes(descriptor) + strings + data


def write_seg2(
    records: list[TraceRecord],
    outputfile: str | Path,
    *,
    dt: float,
    time_offset: float,
    component: str,
    receiver: int,
    source_path: str,
    title: str = "",
    overwrite: bool = False,
) -> Path:
    """Write validated traces as a little-endian SEG-2 file."""

    if not records:
        raise ValueError("No traces were supplied")
    if len(records) > MAX_TRACES:
        raise ValueError(f"SEG-2 supports at most {MAX_TRACES} traces per file")
    sample_count = int(records[0].samples.size)
    if sample_count < 1 or sample_count >= 2**32:
        raise ValueError("SEG-2 sample count must be in the range 1 to 2^32-1")
    if any(record.samples.size != sample_count for record in records):
        raise ValueError("All SEG-2 traces must have the same number of samples")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Sample interval must be finite and positive")
    if not math.isfinite(time_offset):
        raise ValueError("Sample-zero time offset must be finite")

    destination = Path(outputfile)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"SEG-2 output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    trace_blocks = [
        _trace_block(
            record,
            trace_number,
            dt=dt,
            time_offset=time_offset,
            component=component,
        )
        for trace_number, record in enumerate(records, start=1)
    ]
    file_strings = _free_form(
        (
            ("COMPANY", "GPRMAX"),
            ("GPRMAX_COMPONENT", component),
            ("GPRMAX_RECEIVER", receiver),
            ("GPRMAX_SOURCE_PATH", source_path),
            ("GPRMAX_UNITS", quantity_units(component)),
            ("INSTRUMENT", "GPRMAX SYNTHETIC"),
            ("TITLE", title or "UNTITLED"),
            ("TRACE_SORT", "AS_ACQUIRED"),
            ("UNITS", "METERS"),
            ("NOTE", "One selected gprMax time-domain quantity per original model run"),
        )
    )
    pointer_size = 4 * len(records)
    first_trace = 32 + pointer_size + len(file_strings)
    pointers: list[int] = []
    offset = first_trace
    for block in trace_blocks:
        if offset >= 2**32:
            raise ValueError("SEG-2 file exceeds the 32-bit trace-pointer range")
        pointers.append(offset)
        offset += len(block)
    if offset >= 2**32:
        raise ValueError("SEG-2 file exceeds the 32-bit trace-pointer range")

    descriptor = bytearray(32)
    struct.pack_into("<H", descriptor, 0, FILE_DESCRIPTOR_ID)
    struct.pack_into("<H", descriptor, 2, SEG2_REVISION)
    struct.pack_into("<H", descriptor, 4, pointer_size)
    struct.pack_into("<H", descriptor, 6, len(records))
    descriptor[8] = 1
    descriptor[9] = 0
    descriptor[11] = 1
    descriptor[12] = ord("\n")

    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(descriptor)
            stream.write(struct.pack(f"<{len(pointers)}I", *pointers))
            stream.write(file_strings)
            for block in trace_blocks:
                stream.write(block)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def export_seg2(
    outputfiles: Iterable[str | Path],
    seg2_outputfile: str | Path,
    rxnumber: int,
    rxcomponent: str,
    *,
    grid_path: str = "/",
    source_path: str | None = None,
    trace_group: str | None = None,
    overwrite: bool = False,
) -> Seg2ExportSummary:
    """Convert a series of original gprMax A-scan files into SEG-2."""

    records, dt, time_offset, resolved_source, title = collect_traces(
        outputfiles,
        rxnumber,
        rxcomponent,
        grid_path=grid_path,
        source_path=source_path,
        trace_group=trace_group,
    )
    destination = write_seg2(
        records,
        seg2_outputfile,
        dt=dt,
        time_offset=time_offset,
        component=rxcomponent,
        receiver=rxnumber,
        source_path=resolved_source,
        title=title,
        overwrite=overwrite,
    )
    return Seg2ExportSummary(
        outputfile=destination,
        trace_count=len(records),
        sample_count=records[0].samples.size,
        sample_interval=dt,
        receiver=rxnumber,
        component=rxcomponent,
        source_path=resolved_source,
        time_sample_offset=time_offset,
    )


def _default_output_name(basefilename: str, component: str) -> Path:
    base = Path(basefilename)
    if base.suffix.lower() == ".h5":
        base = base.with_suffix("")
    return base.parent / f"{base.name}_{component}.sg2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a naturally ordered gprMax A-scan series as SEG-2.",
        usage="python -m toolboxes.Utilities.outputfiles_seg2 basefilename component [options]",
    )
    parser.add_argument("basefilename", help="base name of the gprMax .h5 A-scan series")
    parser.add_argument("component", help="receiver component to export, e.g. Ez or Ey")
    parser.add_argument("-r", "--receiver", type=int, default=1, help="receiver number (default: 1)")
    parser.add_argument("-o", "--output-file", type=Path, default=None, help="destination .sg2 file")
    parser.add_argument("--grid", default="/", help="HDF5 grid path (default: /)")
    parser.add_argument("--source", default=None, help="source position path, e.g. srcs/src1")
    parser.add_argument("--trace-group", default=None, help="position-bearing trace group, e.g. tls/tl1")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)

    files = discover_files(args.basefilename)
    if not files:
        parser.error(f"No original A-scan files match {args.basefilename}*.h5")
    destination = args.output_file or _default_output_name(args.basefilename, args.component)
    try:
        summary = export_seg2(
            files,
            destination,
            args.receiver,
            args.component,
            grid_path=args.grid,
            source_path=args.source,
            trace_group=args.trace_group,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Written {summary.trace_count} traces x {summary.sample_count} samples "
        f"to {summary.outputfile} (dt={summary.sample_interval:.9g} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
