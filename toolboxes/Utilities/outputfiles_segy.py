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

"""Export a series of gprMax A-scans as a SEG-Y file.

SEG-Y's legacy sample-interval field is an integer number of microseconds,
which cannot represent a normal FDTD time step.  Revision 2 added an IEEE
double-precision extended sample interval to the binary header.  This module
writes that field in microseconds and leaves the legacy field zero unless the
interval is exactly representable there.  Trace samples are big-endian IEEE
single-precision values (SEG-Y format code 5).

Many GPR programs pre-date SEG-Y revision 2 and instead interpret the legacy
sample-interval field as integer picoseconds.  The optional ``gpr`` profile
writes that widespread, deliberately non-standard convention as SEG-Y
revision 1.  When necessary, traces are resampled to the nearest integer
picosecond so the stored time axis remains physically correct.
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


TEXTUAL_HEADER_BYTES = 3200
BINARY_HEADER_BYTES = 400
TRACE_HEADER_BYTES = 240
COORDINATE_SCALE = -10_000
COORDINATE_MULTIPLIER = 10_000.0
SEG_Y_MAJOR = 2
SEG_Y_MINOR = 1
IEEE_FLOAT32_FORMAT_CODE = 5
STANDARD_PROFILE = "standard"
GPR_PROFILE = "gpr"
SUPPORTED_PROFILES = (STANDARD_PROFILE, GPR_PROFILE)


@dataclass(frozen=True)
class SegyExportSummary:
    """Metadata describing a completed SEG-Y export."""

    outputfile: Path
    trace_count: int
    sample_count: int
    sample_interval: float
    receiver: int
    component: str
    source_path: str
    time_sample_offset: float
    profile: str


def _put(buffer: bytearray, offset: int, fmt: str, value) -> None:
    struct.pack_into(">" + fmt, buffer, offset, value)


def _legacy_unsigned(value: float, maximum: int = 65_535) -> int:
    rounded = round(value)
    if 1 <= rounded <= maximum and math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-12):
        return int(rounded)
    return 0


def _gpr_sampling(sample_count: int, dt: float) -> tuple[float, int, int]:
    """Return compatible interval, sample count, and stored picoseconds."""

    dt_ps = dt * 1e12
    stored_dt_ps = round(dt_ps)
    if not 1 <= stored_dt_ps <= 65_535:
        raise ValueError(
            "The GPR SEG-Y profile requires a sample interval representable "
            "as an integer from 1 to 65535 picoseconds"
        )
    compatible_dt = stored_dt_ps * 1e-12
    if sample_count == 1:
        compatible_count = 1
    else:
        duration = (sample_count - 1) * dt
        compatible_count = math.floor(duration / compatible_dt) + 1
    if not 1 <= compatible_count <= 65_535:
        raise ValueError("The GPR SEG-Y profile requires between 1 and 65535 samples per trace")
    return compatible_dt, compatible_count, int(stored_dt_ps)


def _gpr_resample(records: list[TraceRecord], dt: float) -> tuple[list[TraceRecord], float, int]:
    """Return traces sampled at an integer-picosecond interval.

    The legacy GPR adaptation of SEG-Y stores picoseconds in the uint16 field
    which the seismic standard defines as microseconds.  Resampling, rather
    than merely rounding that header value, prevents distortion of the time
    axis in GPR software.
    """

    old_count = records[0].samples.size
    compatible_dt, new_count, stored_dt_ps = _gpr_sampling(old_count, dt)
    if math.isclose(dt, compatible_dt, rel_tol=0.0, abs_tol=1e-24):
        return records, compatible_dt, int(stored_dt_ps)
    if old_count == 1:
        return records, compatible_dt, int(stored_dt_ps)
    old_time = np.arange(old_count, dtype=np.float64) * dt
    new_time = np.arange(new_count, dtype=np.float64) * compatible_dt
    converted = [
        TraceRecord(
            samples=np.interp(new_time, old_time, record.samples),
            source_position=record.source_position,
            receiver_position=record.receiver_position,
            filename=record.filename,
        )
        for record in records
    ]
    return converted, compatible_dt, int(stored_dt_ps)


def _scaled_coordinate(value: float) -> int:
    scaled = round(float(value) * COORDINATE_MULTIPLIER)
    if not -(2**31) <= scaled < 2**31:
        raise ValueError(f"Coordinate {value} m exceeds the SEG-Y int32 range at scalar {COORDINATE_SCALE}")
    return int(scaled)


def _text_line(number: int, text: str = "") -> str:
    clean = " ".join(str(text).replace("\n", " ").split())
    return f"C{number:02d} {clean}"[:80].ljust(80)


def _trace_identification_code(component: str) -> int:
    # SEG-Y rev 2.1 explicitly defines vertical electromagnetic components.
    # Horizontal Cartesian gprMax components are not necessarily aligned with
    # the acquisition line, so they remain generic time-domain traces.
    if component == "Ez":
        return 27
    if component == "Hz":
        return 33
    return 1


def _trace_value_unit(component: str) -> int:
    if component.startswith("I"):
        return 4  # amperes
    return -1  # other; described in the textual header


def _build_textual_header(
    *,
    title: str,
    component: str,
    receiver: int,
    source_path: str,
    records: list[TraceRecord],
    dt: float,
    time_offset: float,
    profile: str,
    original_dt: float,
) -> bytes:
    dt_us = dt * 1e6
    if profile == GPR_PROFILE:
        heading = "GPRMAX SYNTHETIC GPR DATA - GPR SEG-Y REVISION 1 PROFILE"
        data_use_line = "LEGACY GPR PROFILE; DATA USE CODE 1 (PRODUCTION); NO OTHER PROCESSING"
        interval_lines = [
            _text_line(6, f"SAMPLE INTERVAL: {dt:.17g} S ({dt * 1e12:.17g} PICOSECONDS)"),
            _text_line(7, "LEGACY INTERVAL FIELDS CONTAIN PICOSECONDS, NOT SEGY MICROSECONDS"),
            _text_line(16, f"ORIGINAL FDTD INTERVAL BEFORE RESAMPLING: {original_dt:.17g} S"),
        ]
    else:
        heading = "GPRMAX SYNTHETIC GPR DATA - SEG-Y REVISION 2.1"
        data_use_line = "SYNTHETIC FIELD DATA; DATA USE CODE 2 (TEST); NO PROCESSING APPLIED"
        interval_lines = [
            _text_line(6, f"EXACT SAMPLE INTERVAL: {dt:.17g} S ({dt_us:.17g} MICROSECONDS)"),
            _text_line(7, "EXACT INTERVAL IS IN BINARY HEADER BYTES 3273-3280 (IEEE FLOAT64)"),
            _text_line(16, "LEGACY INTEGER-MICROSECOND INTERVAL IS ZERO WHEN NOT EXACTLY REPRESENTABLE"),
        ]
    lines = [
        _text_line(1, heading),
        _text_line(2, f"MODEL TITLE: {title or 'UNTITLED'}"),
        _text_line(3, f"TRACE COUNT: {len(records)}  SAMPLES/TRACE: {records[0].samples.size}"),
        _text_line(4, f"RECEIVER: RX{receiver}  COMPONENT: {component}  UNITS: {quantity_units(component)}"),
        _text_line(5, f"SOURCE POSITION PATH: {source_path}"),
        *interval_lines[:2],
        _text_line(8, f"PHYSICAL TIME OF SAMPLE ZERO: {time_offset:.17g} S"),
        _text_line(9, "DATA: BIG-ENDIAN IEEE FLOAT32, SEG-Y FORMAT CODE 5; NO AMPLITUDE SCALING"),
        _text_line(10, "COORDINATES: GPRMAX X/Y METRES; Z STORED AS ELEVATION; SCALAR -10000"),
        _text_line(11, "SOURCE/RECEIVER COORDINATES ARE READ FROM EACH ORIGINAL A-SCAN FILE"),
        _text_line(12, "ONE SELECTED RECEIVER TRACE PER ORIGINAL GPRMAX MODEL RUN"),
        _text_line(13, f"FIRST INPUT: {records[0].filename}"),
        _text_line(14, f"LAST INPUT: {records[-1].filename}"),
        _text_line(15, data_use_line),
        interval_lines[2],
    ]
    lines.extend(_text_line(number) for number in range(17, 40))
    lines.append(_text_line(40, "END TEXTUAL HEADER"))
    encoded = "".join(lines).encode("ascii", errors="replace")
    if len(encoded) != TEXTUAL_HEADER_BYTES:
        raise AssertionError("SEG-Y textual header is not exactly 3200 bytes")
    return encoded


def _build_binary_header(
    trace_count: int,
    sample_count: int,
    dt: float,
    line_number: int,
    profile: str,
) -> bytes:
    if trace_count < 1:
        raise ValueError("SEG-Y requires at least one trace")
    if sample_count < 1 or sample_count >= 2**32:
        raise ValueError("SEG-Y sample count must be in the range 1 to 2^32-1")
    if profile == GPR_PROFILE and sample_count > 65_535:
        raise ValueError("The GPR SEG-Y revision 1 profile supports at most 65535 samples per trace")
    dt_us = dt * 1e6
    legacy_dt = int(round(dt * 1e12)) if profile == GPR_PROFILE else _legacy_unsigned(dt_us)
    header = bytearray(BINARY_HEADER_BYTES)
    _put(header, 0, "i", 1)  # job identification number
    _put(header, 4, "i", int(line_number))
    _put(header, 8, "i", 1)  # reel number
    _put(header, 12, "H", 1)  # one selected receiver trace per ensemble
    _put(header, 16, "H", legacy_dt)
    _put(header, 18, "H", legacy_dt)
    _put(header, 20, "H", sample_count if sample_count <= 65_535 else 0)
    _put(header, 22, "H", sample_count if sample_count <= 65_535 else 0)
    _put(header, 24, "H", IEEE_FLOAT32_FORMAT_CODE)
    _put(header, 26, "H", 1)  # ensemble fold
    _put(header, 28, "H", 1)  # as recorded
    _put(header, 54, "H", 1)  # metres
    if profile == STANDARD_PROFILE:
        _put(header, 60, "I", 1)  # extended traces per ensemble
        _put(header, 68, "I", sample_count)
        _put(header, 72, "d", dt_us)
        _put(header, 80, "d", dt_us)
        _put(header, 88, "I", sample_count)
        _put(header, 96, "I", 0x01020304)  # byte-order detection constant
        header[300] = SEG_Y_MAJOR
        header[301] = SEG_Y_MINOR
    else:
        header[300] = 1
        header[301] = 0
    _put(header, 302, "H", 1)  # fixed-length traces
    _put(header, 304, "h", 0)  # no extended textual headers
    if profile == STANDARD_PROFILE:
        _put(header, 306, "H", 0)  # no trace-header extensions
        _put(header, 308, "H", 17)  # land + 2-D survey
        _put(header, 312, "Q", trace_count)
    return bytes(header)


def _build_trace_header(
    record: TraceRecord,
    trace_number: int,
    sample_count: int,
    dt: float,
    component: str,
    profile: str,
) -> bytes:
    header = bytearray(TRACE_HEADER_BYTES)
    source = record.source_position
    receiver = record.receiver_position
    midpoint = tuple((a + b) / 2 for a, b in zip(source, receiver))
    separation = math.dist(source, receiver)
    _put(header, 0, "i", trace_number)
    _put(header, 4, "i", trace_number)
    _put(header, 8, "i", trace_number)  # original field record
    _put(header, 12, "i", 1)  # trace number within original record
    _put(header, 16, "i", trace_number)  # energy source point
    _put(header, 20, "i", trace_number)  # ensemble number
    _put(header, 24, "i", 1)  # trace number within ensemble
    _put(header, 28, "h", 1 if profile == GPR_PROFILE else _trace_identification_code(component))
    _put(header, 34, "h", 1 if profile == GPR_PROFILE else 2)
    _put(header, 36, "i", round(separation))
    _put(header, 40, "i", _scaled_coordinate(receiver[2]))
    _put(header, 44, "i", _scaled_coordinate(source[2]))
    _put(header, 68, "h", COORDINATE_SCALE)
    _put(header, 70, "h", COORDINATE_SCALE)
    _put(header, 72, "i", _scaled_coordinate(source[0]))
    _put(header, 76, "i", _scaled_coordinate(source[1]))
    _put(header, 80, "i", _scaled_coordinate(receiver[0]))
    _put(header, 84, "i", _scaled_coordinate(receiver[1]))
    _put(header, 88, "h", 1)  # coordinate units are metres
    _put(header, 114, "H", sample_count if sample_count <= 65_535 else 0)
    trace_dt = int(round(dt * 1e12)) if profile == GPR_PROFILE else _legacy_unsigned(dt * 1e6)
    _put(header, 116, "H", trace_dt)
    _put(header, 180, "i", _scaled_coordinate(midpoint[0]))
    _put(header, 184, "i", _scaled_coordinate(midpoint[1]))
    _put(header, 188, "i", 1)  # inline number
    _put(header, 192, "i", trace_number)  # crossline/trace index
    _put(header, 202, "h", _trace_value_unit(component))
    if profile == STANDARD_PROFILE:
        header[232:240] = b"SEG00000"
    return bytes(header)


def write_segy(
    records: list[TraceRecord],
    outputfile: str | Path,
    *,
    dt: float,
    time_offset: float,
    component: str,
    receiver: int,
    source_path: str,
    title: str = "",
    line_number: int = 1,
    overwrite: bool = False,
    profile: str = STANDARD_PROFILE,
) -> Path:
    """Write validated traces using a standard or GPR compatibility profile."""

    if not records:
        raise ValueError("No traces were supplied")
    sample_count = int(records[0].samples.size)
    if any(record.samples.size != sample_count for record in records):
        raise ValueError("All SEG-Y traces must have the same number of samples")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Sample interval must be finite and positive")
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unknown SEG-Y profile {profile!r}; choose from {', '.join(SUPPORTED_PROFILES)}")
    if not -(2**31) <= int(line_number) < 2**31:
        raise ValueError("Line number must fit in a signed 32-bit SEG-Y field")

    destination = Path(outputfile)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"SEG-Y output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    original_dt = dt
    output_records = records
    if profile == GPR_PROFILE:
        output_records, dt, _ = _gpr_resample(records, dt)
        sample_count = int(output_records[0].samples.size)

    textual = _build_textual_header(
        title=title,
        component=component,
        receiver=receiver,
        source_path=source_path,
        records=output_records,
        dt=dt,
        time_offset=time_offset,
        profile=profile,
        original_dt=original_dt,
    )
    binary = _build_binary_header(len(output_records), sample_count, dt, line_number, profile)

    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(textual)
            stream.write(binary)
            for trace_number, record in enumerate(output_records, start=1):
                stream.write(_build_trace_header(record, trace_number, sample_count, dt, component, profile))
                stream.write(np.asarray(record.samples, dtype=">f4").tobytes(order="C"))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def export_segy(
    outputfiles: Iterable[str | Path],
    segy_outputfile: str | Path,
    rxnumber: int,
    rxcomponent: str,
    *,
    grid_path: str = "/",
    source_path: str | None = None,
    trace_group: str | None = None,
    line_number: int = 1,
    overwrite: bool = False,
    profile: str = STANDARD_PROFILE,
) -> SegyExportSummary:
    """Convert a series of original gprMax A-scan files into SEG-Y."""

    records, dt, time_offset, resolved_source, title = collect_traces(
        outputfiles,
        rxnumber,
        rxcomponent,
        grid_path=grid_path,
        source_path=source_path,
        trace_group=trace_group,
    )
    destination = write_segy(
        records,
        segy_outputfile,
        dt=dt,
        time_offset=time_offset,
        component=rxcomponent,
        receiver=rxnumber,
        source_path=resolved_source,
        title=title,
        line_number=line_number,
        overwrite=overwrite,
        profile=profile,
    )
    output_sample_count = records[0].samples.size
    output_dt = dt
    if profile == GPR_PROFILE:
        output_dt, output_sample_count, _ = _gpr_sampling(output_sample_count, dt)
    return SegyExportSummary(
        outputfile=destination,
        trace_count=len(records),
        sample_count=output_sample_count,
        sample_interval=output_dt,
        receiver=rxnumber,
        component=rxcomponent,
        source_path=resolved_source,
        time_sample_offset=time_offset,
        profile=profile,
    )


def _default_output_name(basefilename: str, component: str) -> Path:
    base = Path(basefilename)
    if base.suffix.lower() == ".h5":
        base = base.with_suffix("")
    return base.parent / f"{base.name}_{component}.sgy"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a naturally ordered gprMax A-scan series as SEG-Y.",
        usage=("python -m toolboxes.Utilities.outputfiles_segy " "basefilename component [options]"),
    )
    parser.add_argument("basefilename", help="base name of the gprMax .h5 A-scan series")
    parser.add_argument("component", help="receiver component to export, e.g. Ez or Ey")
    parser.add_argument("-r", "--receiver", type=int, default=1, help="receiver number (default: 1)")
    parser.add_argument("-o", "--output-file", type=Path, default=None, help="destination .sgy file")
    parser.add_argument(
        "--grid",
        default="/",
        help="HDF5 grid path, e.g. / or subgrids/fine (default: /)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="source position path within the grid, e.g. srcs/src1 or tls/tl1",
    )
    parser.add_argument(
        "--trace-group",
        default=None,
        help=(
            "group containing a real time-domain quantity and Position metadata, "
            "e.g. tls/tl1 for Vtotal (default: rxs/rxN)"
        ),
    )
    parser.add_argument("--line-number", type=int, default=1, help="SEG-Y line number")
    parser.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILES,
        default=STANDARD_PROFILE,
        help=(
            "standard writes SEG-Y 2.1 with the exact interval; gpr writes the "
            "legacy revision-1 integer-picosecond convention used by GPR software"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)

    files = discover_files(args.basefilename)
    if not files:
        parser.error(f"No original A-scan files match {args.basefilename}*.h5")
    destination = args.output_file or _default_output_name(args.basefilename, args.component)
    try:
        summary = export_segy(
            files,
            destination,
            args.receiver,
            args.component,
            grid_path=args.grid,
            source_path=args.source,
            trace_group=args.trace_group,
            line_number=args.line_number,
            overwrite=args.overwrite,
            profile=args.profile,
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
