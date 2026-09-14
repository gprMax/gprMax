# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Load acquisition traces shared by the SEG-Y, SEG-2 and DT1 exporters.

Collection validates original A-scans and their common timing, without
resampling or amplitude scaling. Each format writer handles its own storage
conversion. A legacy merged array is rejected because this interface needs
the source/receiver positions from every original run.
"""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from gprMax.utilities.utilities import natural_keys
from gprMax.toolboxes.Utilities.receiver_identity import match_receiver, receiver_catalogue, select_receiver
from gprMax.toolboxes.Utilities.trace_time import read_time_history, receiver_time_offset

TIME_DOMAIN_QUANTITIES = {
    "Ex",
    "Ey",
    "Ez",
    "Hx",
    "Hy",
    "Hz",
    "Ix",
    "Iy",
    "Iz",
    "Vinc",
    "Vtotal",
    "Iinc",
    "Itotal",
    "Itot",
}


@dataclass(frozen=True)
class TraceRecord:
    """One real ``(nt,)`` trace with stored Cartesian positions in metres.

    Sample units depend on the selected component. The common sample
    interval and physical sample-zero time are returned by ``collect_traces``
    separately from these per-run values.
    """

    samples: np.ndarray
    source_position: tuple[float, float, float]
    receiver_position: tuple[float, float, float]
    filename: str


def _grid_group(output: h5py.File, grid_path: str = "/") -> h5py.Group:
    path = str(grid_path).strip("/")
    if not path:
        return output
    if path not in output:
        raise ValueError(f"Grid path {grid_path!r} is not present in {output.filename}")
    group = output[path]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Grid path {grid_path!r} is not an HDF5 group")
    return group


def _source_candidates(grid: h5py.Group) -> list[str]:
    candidates: list[str] = []
    for parent_name in ("srcs", "tls", "frills"):
        if parent_name not in grid:
            continue
        parent = grid[parent_name]
        for name in sorted(parent.keys(), key=natural_keys):
            item = parent[name]
            if isinstance(item, h5py.Group) and "Position" in item.attrs:
                candidates.append(f"{parent_name}/{name}")
    return candidates


def _resolve_source_path(grid: h5py.Group, requested: str | None) -> str:
    candidates = _source_candidates(grid)
    if requested is not None:
        requested = requested.strip("/")
        if requested not in candidates:
            choices = ", ".join(candidates) if candidates else "none"
            raise ValueError(
                f"Source {requested!r} is not available; sources with positions: {choices}"
            )
        return requested
    if not candidates:
        raise ValueError("No source with position metadata was found in the selected grid")
    if len(candidates) > 1:
        raise ValueError(
            "More than one source has position metadata; select one with source_path "
            f"({', '.join(candidates)})"
        )
    return candidates[0]


def _position(group: h5py.Group, description: str) -> tuple[float, float, float]:
    values = np.asarray(group.attrs.get("Position", ()), dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{description} does not have a finite three-dimensional Position")
    return tuple(float(value) for value in values)


def default_time_offset(component: str, dt: float) -> float:
    """Infer sample-zero time in seconds when dataset metadata is absent.

    E/voltage defaults to zero; H/current defaults to ``-dt/2``. Explicit
    dataset offsets override this fallback, including source-specific timing.
    """

    return receiver_time_offset(component, dt)


def quantity_units(component: str) -> str:
    """Return the native SI unit associated with a time-domain quantity."""

    if component.startswith("E"):
        return "V/m"
    if component.startswith("H"):
        return "A/m"
    if component.startswith("I"):
        return "A"
    if component.startswith("V"):
        return "V"
    return "gprMax native SI units"


def validate_float32_traces(records: Iterable[TraceRecord]) -> None:
    """Reject traces whose storage conversion would produce non-finite samples.

    Writers call this before opening an output file. The temporary cast checks
    the actual float32 rounding boundary without rescaling or changing the
    caller's arrays; endian-specific packing remains with each format writer.
    """

    for record in records:
        with np.errstate(over="ignore", invalid="ignore"):
            stored = np.asarray(record.samples, dtype=np.float32)
        if not np.all(np.isfinite(stored)):
            raise ValueError(
                f"Trace samples cannot be represented as finite float32 values: {record.filename}"
            )


def collect_traces(
    outputfiles: Iterable[str | Path],
    rxnumber: int,
    rxcomponent: str,
    *,
    grid_path: str = "/",
    source_path: str | None = None,
    trace_group: str | None = None,
) -> tuple[list[TraceRecord], float, float, str, str]:
    """Read original A-scans in the supplied order; this function does not sort.

    ``rxnumber`` is one-based. Source and trace-group paths are relative to
    ``grid_path``; a supplied trace group replaces the default ``rxs/rxN``.
    A public receiver path/number selects in the first file; later files are
    matched by identity. ``trace_group`` also accepts ``name:``/``study:``
    selectors. Ambiguous legacy multi-receiver identities are rejected.
    Only finite, real, one-dimensional histories with matching sample counts,
    intervals and time offsets are accepted. Stored positions are copied
    without another coordinate transform.
    Returns the trace records, sample interval in seconds, sample-zero time
    offset in seconds, resolved source path, and model title.
    """

    files = [Path(filename) for filename in outputfiles]
    if not files:
        raise ValueError("No gprMax A-scan files were supplied")
    missing = [str(filename) for filename in files if not filename.is_file()]
    if missing:
        raise FileNotFoundError("Output file(s) not found: " + ", ".join(missing))
    if rxnumber < 1:
        raise ValueError("Receiver number must be at least one")
    if rxcomponent not in TIME_DOMAIN_QUANTITIES:
        raise ValueError(
            f"{rxcomponent!r} is not a supported real time-domain quantity; "
            "frequency-domain outputs such as S-parameters and impedance cannot be exported"
        )

    records: list[TraceRecord] = []
    expected_dt: float | None = None
    expected_samples: int | None = None
    expected_offset: float | None = None
    resolved_source: str | None = None
    title = ""
    reference_receiver = None

    for filename in files:
        with h5py.File(filename, "r") as output:
            grid = _grid_group(output, grid_path)
            receiver_path = trace_group.strip("/") if trace_group is not None else f"rxs/rx{rxnumber}"
            # Public paths/selectors identify the first file's receiver;
            # source/terminal trace groups keep their independent namespaces.
            if receiver_path.startswith(("rxs/", "name:", "study:", "build:")):
                catalogue = receiver_catalogue(grid)
                selected = (
                    select_receiver(catalogue, receiver_path)
                    if reference_receiver is None
                    else match_receiver(reference_receiver, catalogue)
                )
                reference_receiver = reference_receiver or selected
                receiver_path = selected.path
            dataset_path = f"{receiver_path}/{rxcomponent}"
            if receiver_path not in grid:
                raise ValueError(f"Trace group {receiver_path!r} is not available in {filename}")
            if dataset_path not in grid:
                available = ", ".join(grid[receiver_path].keys())
                raise ValueError(
                    f"Component {rxcomponent!r} is not available for receiver {rxnumber} "
                    f"in {filename}; available: {available or 'none'}"
                )

            dataset = grid[dataset_path]
            if dataset.ndim != 1:
                raise ValueError(
                    f"{filename}:{dataset_path} has shape {dataset.shape}; export requires "
                    "original one-dimensional A-scan files, not a legacy merged file"
                )
            history = read_time_history(dataset)
            samples = np.asarray(history.samples, dtype=np.float64)
            dt, offset = history.dt, history.offset

            candidate = _resolve_source_path(grid, source_path)
            if resolved_source is None:
                resolved_source = candidate
            elif candidate != resolved_source:
                raise ValueError(f"Resolved source path changes in {filename}")

            if expected_dt is None:
                expected_dt = dt
                expected_samples = samples.size
                expected_offset = offset
                title_value = output.attrs.get("Title", "")
                title = (
                    title_value.decode(errors="replace")
                    if isinstance(title_value, bytes)
                    else str(title_value)
                )
            else:
                if not math.isclose(dt, expected_dt, rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError(
                        f"Sample interval in {filename} is {dt}, expected {expected_dt} seconds"
                    )
                if samples.size != expected_samples:
                    raise ValueError(
                        f"Trace in {filename} has {samples.size} samples, expected {expected_samples}"
                    )
                if not math.isclose(offset, expected_offset, rel_tol=1e-12, abs_tol=1e-30):
                    raise ValueError(
                        f"Sample-zero time offset in {filename} is {offset}, "
                        f"expected {expected_offset} seconds"
                    )

            records.append(
                TraceRecord(
                    samples=samples,
                    source_position=_position(grid[resolved_source], f"Source {resolved_source}"),
                    receiver_position=_position(grid[receiver_path], f"Receiver {rxnumber}"),
                    filename=filename.name,
                )
            )

    assert expected_dt is not None
    assert expected_offset is not None
    assert resolved_source is not None
    return records, expected_dt, expected_offset, resolved_source, title


def discover_files(basefilename: str) -> list[Path]:
    """Find a naturally ordered original gprMax A-scan series."""

    base = Path(basefilename)
    if base.is_file():
        return [base]
    matches = [
        Path(filename)
        for filename in glob.glob(basefilename + "*.h5")
        if "_merged" not in Path(filename).stem
    ]
    matches.sort(key=lambda path: natural_keys(str(path)))
    return matches
