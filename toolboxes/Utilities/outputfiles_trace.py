# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Shared trace collection for gprMax interchange-format exporters."""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from gprMax.utilities.utilities import natural_keys


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
    """One scalar time series and the acquisition geometry that produced it."""

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
            raise ValueError(f"Source {requested!r} is not available; sources with positions: {choices}")
        return requested
    if not candidates:
        raise ValueError("No source with position metadata was found in the selected grid")
    if len(candidates) > 1:
        raise ValueError(
            "More than one source has position metadata; select one with source_path " f"({', '.join(candidates)})"
        )
    return candidates[0]


def _position(group: h5py.Group, description: str) -> tuple[float, float, float]:
    values = np.asarray(group.attrs.get("Position", ()), dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{description} does not have a finite three-dimensional Position")
    return tuple(float(value) for value in values)


def default_time_offset(component: str, dt: float) -> float:
    """Return the physical time of sample zero for a Yee-grid quantity."""

    if component.startswith("E"):
        return 0.0
    if component.startswith(("H", "I")):
        return -0.5 * dt
    return 0.0


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


def collect_traces(
    outputfiles: Iterable[str | Path],
    rxnumber: int,
    rxcomponent: str,
    *,
    grid_path: str = "/",
    source_path: str | None = None,
    trace_group: str | None = None,
) -> tuple[list[TraceRecord], float, float, str, str]:
    """Read and validate a naturally ordered series of gprMax A-scan files.

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

    for filename in files:
        with h5py.File(filename, "r") as output:
            grid = _grid_group(output, grid_path)
            receiver_path = trace_group.strip("/") if trace_group is not None else f"rxs/rx{rxnumber}"
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
            raw_samples = np.asarray(dataset)
            if np.iscomplexobj(raw_samples):
                raise ValueError(f"Complex-valued trace data cannot be exported: {filename}")
            samples = np.asarray(raw_samples, dtype=np.float64)
            if not np.all(np.isfinite(samples)):
                raise ValueError(f"Trace data contain NaN or infinite values: {filename}")

            dt = float(dataset.attrs.get("SampleInterval", grid.attrs.get("dt", math.nan)))
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError(f"Invalid or missing sample interval in {filename}")
            offset = float(dataset.attrs.get("TimeSampleOffset", default_time_offset(rxcomponent, dt)))
            if not math.isfinite(offset):
                raise ValueError(f"Invalid sample-zero time offset in {filename}")

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
                title = title_value.decode(errors="replace") if isinstance(title_value, bytes) else str(title_value)
            else:
                if not math.isclose(dt, expected_dt, rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError(f"Sample interval in {filename} is {dt}, expected {expected_dt} seconds")
                if samples.size != expected_samples:
                    raise ValueError(f"Trace in {filename} has {samples.size} samples, expected {expected_samples}")
                if not math.isclose(offset, expected_offset, rel_tol=1e-12, abs_tol=1e-30):
                    raise ValueError(
                        f"Sample-zero time offset in {filename} is {offset}, " f"expected {expected_offset} seconds"
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
    matches = [Path(filename) for filename in glob.glob(basefilename + "*.h5") if "_merged" not in Path(filename).stem]
    matches.sort(key=lambda path: natural_keys(str(path)))
    return matches
