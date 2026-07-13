"""
h5_reader.py: gprMax HDF5 output file reader for the Marimo dashboard.

Reads one or more gprMax v4 .h5 output files into a structured dict.
No marimo dependency — pure h5py + numpy. Designed to be the foundation
layer that all dashboard cells import from.

Usage:
    from toolboxes.Marimo.h5_reader import load_files, load_file

    data = load_files([
        "examples/cylinder_Ascan_2D.h5",
        "examples/cylinder_Ascan_2D_freespace.h5",
    ])

    # Access a specific trace
    ez = data["cylinder_Ascan_2D.h5"]["receivers"]["rx1"]["components"]["Ez"]
    dt = data["cylinder_Ascan_2D.h5"]["meta"]["dt"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

# ComponentMap: {"Ez": np.ndarray, "Hx": np.ndarray, ...}
ComponentMap = dict[str, np.ndarray]

# ReceiverInfo: full data for one receiver point
ReceiverInfo = dict[str, Any]
# {
#   "name":       str              e.g. "Rx(70,85,0)"
#   "position":   list[float]      e.g. [0.14, 0.17, 0.0]
#   "components": ComponentMap
# }

# FileMeta: root-level attributes from the HDF5 file
FileMeta = dict[str, Any]
# {
#   "title":       str
#   "dt":          float           time step in seconds
#   "dx_dy_dz":    list[float]     spatial discretisation [m, m, m]
#   "iterations":  int
#   "nrx":         int             number of receivers
#   "nsrc":        int             number of sources
#   "nx_ny_nz":    list[int]       grid dimensions in cells
#   "gprmax_version": str
# }

# SourceInfo: metadata for one source point
SourceInfo = dict[str, Any]
# {
#   "type":     str   e.g. "HertzianDipole"
#   "position": list[float]
# }

# FileData: everything read from one .h5 file
FileData = dict[str, Any]
# {
#   "path":      str               absolute path to the file
#   "meta":      FileMeta
#   "receivers": dict[str, ReceiverInfo]   keyed by "rx1", "rx2", ...
#   "sources":   dict[str, SourceInfo]     keyed by "src1", "src2", ...
#   "time_ns":   np.ndarray                time axis in nanoseconds
# }


# Core reader


def load_file(path: str | Path) -> FileData:
    """Read a single gprMax v4 HDF5 output file.

    Args:
        path: Path to the .h5 file.

    Returns:
        FileData dict containing metadata, all receiver traces, and
        a precomputed nanosecond time axis.

    Raises:
        FileNotFoundError: if the path does not exist.
        KeyError: if the file is missing expected gprMax structure.
        OSError: if the file cannot be opened (not a valid HDF5 file).
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
        meta = _read_meta(f)
        receivers = _read_receivers(f, path)
        sources = _read_sources(f)

    # Build nanosecond time axis from root dt
    iterations = meta["iterations"]
    dt = meta["dt"]
    time_ns = np.arange(iterations) * dt * 1e9  # seconds → nanoseconds

    return {
        "path": str(path),
        "meta": meta,
        "receivers": receivers,
        "sources": sources,
        "time_ns": time_ns,
    }


def load_files(paths: list[str | Path]) -> dict[str, FileData]:
    """Read multiple gprMax HDF5 output files.

    Args:
        paths: List of paths to .h5 files.

    Returns:
        Dict keyed by filename (not full path) mapping to FileData.
        If two files share the same filename, the second is keyed by
        its full path to avoid collision.

    Raises:
        FileNotFoundError: if any path does not exist.
    """
    result: dict[str, FileData] = {}
    seen_names: set[str] = set()

    for p in paths:
        data = load_file(p)
        name = Path(p).name

        # Avoid key collision when two files share the same filename
        if name in seen_names:
            name = str(Path(p).resolve())
        seen_names.add(name)

        result[name] = data

    return result


# Time axis utility


def get_time_axis(file_data: FileData, unit: str = "ns") -> np.ndarray:
    """Return the time axis for a loaded file.

    Args:
        file_data: FileData dict from load_file().
        unit: "ns" (nanoseconds, default), "s" (seconds), or "iter" (iterations).

    Returns:
        1D numpy array.
    """
    iterations = file_data["meta"]["iterations"]
    dt = file_data["meta"]["dt"]

    if unit == "ns":
        return np.arange(iterations) * dt * 1e9
    elif unit == "s":
        return np.arange(iterations) * dt
    elif unit == "iter":
        return np.arange(iterations, dtype=float)
    else:
        raise ValueError(f"Unknown unit '{unit}'. Use 'ns', 's', or 'iter'.")


# Component and receiver utilities


def list_components(file_data: FileData, receiver: str = "rx1") -> list[str]:
    """Return available field component names for a given receiver.

    Args:
        file_data: FileData dict from load_file().
        receiver: Receiver key, e.g. "rx1".

    Returns:
        Sorted list of component names, e.g. ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"].
    """
    try:
        return sorted(file_data["receivers"][receiver]["components"].keys())
    except KeyError:
        return []


def list_receivers(file_data: FileData) -> list[str]:
    """Return receiver keys available in the file.

    Returns:
        List of receiver keys, e.g. ["rx1", "rx2"].
    """
    return list(file_data["receivers"].keys())


def get_trace(
    file_data: FileData,
    component: str,
    receiver: str = "rx1",
    polarity: int = 1,
) -> np.ndarray:
    """Return a single field component array.

    Args:
        file_data: FileData dict from load_file().
        component: Field component name, e.g. "Ez". Append "-" for negative
                   polarity, e.g. "Ez-".
        receiver: Receiver key, e.g. "rx1".
        polarity: +1 or -1. If component ends with "-", polarity is forced to -1.

    Returns:
        1D numpy array of field values.
    """
    # Handle polarity suffix convention from plot_Ascan.py
    if component.endswith("-"):
        component = component[:-1]
        polarity = -1

    try:
        arr = file_data["receivers"][receiver]["components"][component]
    except KeyError:
        available = list_components(file_data, receiver)
        raise KeyError(
            f"Component '{component}' not found in {receiver}. "
            f"Available: {available}"
        )

    return arr * polarity


def get_unit_label(component: str) -> str:
    """Return the SI unit label for a field component.

    Args:
        component: Component name, e.g. "Ez", "Hx", "Ix".

    Returns:
        Unit string for axis labelling.
    """
    component = component.rstrip("-")
    if component.startswith("E"):
        return "V/m"
    elif component.startswith("H"):
        return "A/m"
    elif component.startswith("I"):
        return "A"
    return ""


def build_label(filename: str, receiver: str, component: str) -> str:
    """Build a descriptive trace label for plot legends.

    Args:
        filename: Key from load_files() dict (usually the filename).
        receiver: Receiver key, e.g. "rx1".
        component: Component name, e.g. "Ez".

    Returns:
        Human-readable label, e.g. "cylinder_Ascan_2D · rx1 · Ez".
    """
    stem = Path(filename).stem
    return f"{stem} · {receiver} · {component}"


# Metadata formatting


def format_metadata_text(file_data: FileData) -> str:
    """Format file metadata as a human-readable markdown string.

    Intended for use in mo.md() metadata banners.

    Args:
        file_data: FileData dict from load_file().

    Returns:
        Markdown-formatted metadata string.
    """
    m = file_data["meta"]
    dx, dy, dz = m["dx_dy_dz"]
    nx, ny, nz = m["nx_ny_nz"]
    dt_ps = m["dt"] * 1e12  # seconds → picoseconds
    total_ns = m["iterations"] * m["dt"] * 1e9

    lines = [
        f"**{m['title']}**",
        f"gprMax {m['gprmax_version']} · "
        f"{m['iterations']} iterations · "
        f"{total_ns:.3f} ns total window",
        f"dt = {dt_ps:.4f} ps · "
        f"dx/dy/dz = {dx*1000:.1f}/{dy*1000:.1f}/{dz*1000:.1f} mm · "
        f"grid = {nx}×{ny}×{nz} cells",
        f"{m['nrx']} receiver(s) · {m['nsrc']} source(s)",
    ]

    # Add receiver positions
    for rx_key, rx_info in file_data["receivers"].items():
        pos = rx_info["position"]
        name = rx_info["name"]
        lines.append(
            f"  {rx_key}: {name} at "
            f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m"
        )

    return "\n\n".join(lines[:3]) + "\n\n" + "\n".join(lines[3:])


# Private helpers


def _read_meta(f: h5py.File) -> FileMeta:
    """Extract root-level metadata attributes."""
    attrs = dict(f.attrs)
    return {
        "title": str(attrs.get("Title", "Untitled")),
        "dt": float(attrs["dt"]),
        "dx_dy_dz": list(map(float, attrs["dx_dy_dz"])),
        "iterations": int(attrs["Iterations"]),
        "nrx": int(attrs.get("nrx", 0)),
        "nsrc": int(attrs.get("nsrc", 0)),
        "nx_ny_nz": list(map(int, attrs.get("nx_ny_nz", [0, 0, 0]))),
        "gprmax_version": str(attrs.get("gprMax", "unknown")),
    }


def _read_receivers(f: h5py.File, path: Path) -> dict[str, ReceiverInfo]:
    """Read all receiver data from the HDF5 file."""
    receivers: dict[str, ReceiverInfo] = {}

    if "rxs" not in f:
        return receivers

    for rx_key in f["rxs"].keys():
        rx_group = f["rxs"][rx_key]
        rx_attrs = dict(rx_group.attrs)

        components: ComponentMap = {}
        for comp_name in rx_group.keys():
            dataset = rx_group[comp_name]
            if isinstance(dataset, h5py.Dataset):
                components[comp_name] = dataset[:]

        position = list(map(float, rx_attrs.get("Position", [0.0, 0.0, 0.0])))

        receivers[rx_key] = {
            "name": str(rx_attrs.get("Name", rx_key)),
            "position": position,
            "components": components,
        }

    return receivers


def _read_sources(f: h5py.File) -> dict[str, SourceInfo]:
    """Read all source metadata from the HDF5 file."""
    sources: dict[str, SourceInfo] = {}

    if "srcs" not in f:
        return sources

    for src_key in f["srcs"].keys():
        src_attrs = dict(f["srcs"][src_key].attrs)
        sources[src_key] = {
            "type": str(src_attrs.get("Type", "unknown")),
            "position": list(
                map(float, src_attrs.get("Position", [0.0, 0.0, 0.0]))
            ),
        }

    return sources