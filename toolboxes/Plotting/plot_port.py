# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Plot authoritative terminal quantities stored by gprMax port outputs."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrequencyTrace:
    """One complex quantity on a port frequency axis."""

    name: str
    label: str
    values: npt.NDArray[np.complexfloating]
    valid: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class TimeTrace:
    """One stored terminal time history."""

    name: str
    label: str
    time: npt.NDArray[np.floating]
    values: npt.NDArray[np.floating]
    quantity: str


@dataclass(frozen=True)
class PortData:
    """Common plotting view of one gprMax terminal-output group.

    ``s_parameters`` is a tuple rather than a single S11 array so the plotter
    can accept additional Sij datasets without changing its public model.
    """

    path: str
    id: str
    source_type: str
    frequency: npt.NDArray[np.floating]
    s_parameters: tuple[FrequencyTrace, ...]
    impedances: tuple[FrequencyTrace, ...]
    admittances: tuple[FrequencyTrace, ...]
    time_traces: tuple[TimeTrace, ...]
    spectral_traces: tuple[FrequencyTrace, ...]
    validity_masks: dict[str, npt.NDArray[np.bool_]]
    diagnostics: dict[str, npt.NDArray[np.floating]]
    metadata: dict[str, object]

    @property
    def primary_s11(self) -> FrequencyTrace:
        return next(trace for trace in self.s_parameters if trace.name == "S11")


TIME_DATASETS = (
    ("Vgenerator", "Generator voltage", "voltage", ("time", "time_voltage")),
    ("Vinc", "Incident voltage", "voltage", ("time_voltage", "time")),
    ("Vtotal", "Total voltage", "voltage", ("time_voltage", "time")),
    ("Iinc", "Incident current", "current", ("time_current", "time")),
    ("Itotal", "Total current", "current", ("time_current", "time")),
    ("Itot", "Total current", "current", ("time_current", "time")),
    ("Iloop", "Ampere-loop current", "current", ("time_current", "time")),
    ("Inetwork", "Network current", "current", ("time_current", "time")),
)

SPECTRAL_DATASETS = (
    ("Vgenerator_spectrum", "Generator voltage", "voltage"),
    ("Vincident_spectrum", "Incident voltage", "voltage"),
    ("Vreflected_spectrum", "Reflected voltage", "voltage"),
    ("Vreflected_source_spectrum", "Reflected voltage (source plane)", "voltage"),
    ("Vreflected_current_spectrum", "Reflected voltage (current check)", "voltage"),
    ("Vtotal_spectrum", "Total voltage", "voltage"),
    ("Iincident_spectrum", "Incident current", "current"),
    ("Itotal_spectrum", "Total current", "current"),
    ("Iterminal_spectrum", "Terminal current", "current"),
    ("Iterminal_current_spectrum", "Terminal current (line check)", "current"),
    ("Iloop_spectrum", "Ampere-loop current", "current"),
    ("Inetwork_spectrum", "Network current", "current"),
)

VALIDITY_DATASETS = (
    "source_valid",
    "mesh_valid",
    "gap_correction_valid",
    "line_propagation_valid",
)

DIAGNOSTIC_DATASETS = (
    "incident_relative_dB",
    "cells_per_minimum_wavelength",
)

METADATA_ATTRIBUTES = (
    "ReferenceImpedance",
    "PortMode",
    "SpectrumLimitMode",
    "ValidFrequencyRange",
    "MeshFrequencyLimit",
    "MinimumWavelengthCells",
    "LimitingMaterial",
    "IncidentFloorDB",
    "TailRelativeLevelDB",
    "IndependentFrequencyResolution",
    "ZinPrimaryMethod",
    "CurrentCheckMethod",
    "phasor_time_sign",
    "forward_transform_sign",
)


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _normalise_path(path: str) -> str:
    return str(path).strip("/")


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip("/"))
    return text.strip("_") or "port"


def discover_port_outputs(filename: str | Path) -> tuple[str, ...]:
    """Return groups containing the authoritative terminal-output schema."""

    paths: list[str] = []
    with h5py.File(filename, "r") as output:

        def visitor(name, item):
            if isinstance(item, h5py.Group) and "frequency" in item and "S11" in item:
                paths.append(name)

        output.visititems(visitor)
    return tuple(sorted(paths))


# Compatibility name used by the retired plot_antenna_params module.
discover_terminal_outputs = discover_port_outputs


def select_port_paths(
    available: tuple[str, ...], selectors: list[str] | tuple[str, ...] | None, filename: str | Path
) -> tuple[str, ...]:
    """Resolve port IDs or complete HDF5 paths, preserving requested order."""

    if not available:
        raise ValueError(f"{filename} does not contain stored port output")
    if not selectors:
        return available

    selected: list[str] = []
    for selector in selectors:
        candidate = _normalise_path(selector)
        exact = [path for path in available if path == candidate]
        named = [path for path in available if path.rsplit("/", 1)[-1] == candidate]
        matches = exact or named
        if len(matches) > 1:
            raise ValueError(
                f"Port name {selector!r} is ambiguous; use one of: {', '.join(matches)}"
            )
        if not matches:
            raise ValueError(
                f"Port {selector!r} not found in {filename}; available outputs: "
                f"{', '.join(available)}"
            )
        if matches[0] not in selected:
            selected.append(matches[0])
    return tuple(selected)


def _require_vector(group, name, length, *, complex_values=False):
    if name not in group:
        return None
    dtype = np.complex128 if complex_values else np.float64
    # Read the native HDF5 type before converting. Some HDF5 builds do not
    # provide a direct integer-to-complex conversion path for Dataset.__array__.
    values = np.asarray(group[name][...]).astype(dtype, copy=False)
    if values.ndim != 1 or values.size != length:
        raise ValueError(
            f"{group.name}/{name} must be a one-dimensional array with {length} values"
        )
    return values


def _valid_mask(group, name, values, length):
    if name in group:
        mask = np.asarray(group[name], dtype=bool)
        if mask.ndim != 1 or mask.size != length:
            raise ValueError(f"{group.name}/{name} has the wrong shape")
    else:
        mask = np.ones(length, dtype=bool)
    return np.asarray(mask & np.isfinite(values), dtype=bool)


def _diagnostic_valid(group, dataset, values, length):
    explicit = f"valid_{dataset}"
    if explicit in group:
        return _valid_mask(group, explicit, values, length)
    mask = np.isfinite(values)
    for name in ("source_valid", "mesh_valid"):
        if name in group:
            candidate = np.asarray(group[name], dtype=bool)
            if candidate.shape == mask.shape:
                mask &= candidate
    return np.asarray(mask, dtype=bool)


def _source_type(path, group):
    if "SourceType" in group.attrs:
        return _text(group.attrs["SourceType"])
    if path.startswith("tls/") or "/tls/" in path:
        return "Transmission line"
    if path.startswith("frills/") or "/frills/" in path:
        return "Magnetic frill"
    return "Terminal port"


def _time_trace(output, group, name, label, quantity, time_names):
    if name not in group:
        return None
    values = np.asarray(group[name], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{group.name}/{name} must be a one-dimensional history")
    time = None
    for time_name in time_names:
        if time_name in group:
            candidate = np.asarray(group[time_name], dtype=np.float64)
            if candidate.shape == values.shape:
                time = candidate
                break
    if time is None:
        dt = output.attrs.get("dt")
        if dt is None:
            return None
        time = np.arange(values.size, dtype=np.float64) * float(dt)
    return TimeTrace(name, label, time, values, quantity)


def read_port_output(filename: str | Path, port: str | None = None) -> PortData:
    """Read one current gprMax terminal output without recalculating it."""

    available = discover_port_outputs(filename)
    if port is None:
        if len(available) != 1:
            raise ValueError(
                f"{filename} contains {len(available)} port outputs; select one from "
                f"{', '.join(available)}"
            )
        path = available[0]
    else:
        path = select_port_paths(available, (port,), filename)[0]

    with h5py.File(filename, "r") as output:
        group = output[path]
        frequency = np.asarray(group["frequency"], dtype=np.float64)
        if frequency.ndim != 1 or frequency.size == 0 or not np.all(np.isfinite(frequency)):
            raise ValueError(f"{group.name}/frequency must be a finite one-dimensional array")
        if np.any(np.diff(frequency) < 0):
            raise ValueError(f"{group.name}/frequency must be monotonically increasing")
        count = frequency.size

        s_parameters: list[FrequencyTrace] = []
        for name, label, valid_name in (
            ("S11", r"$S_{11}$", "valid_S11"),
            ("S11_source", r"$S_{11}$ (source plane)", None),
            ("S11_current", r"$S_{11}$ (current check)", "valid_S11_current"),
        ):
            values = _require_vector(group, name, count, complex_values=True)
            if values is not None:
                valid = (
                    _valid_mask(group, valid_name, values, count)
                    if valid_name is not None
                    else _diagnostic_valid(group, name, values, count)
                )
                s_parameters.append(FrequencyTrace(name, label, values, valid))

        impedances: list[FrequencyTrace] = []
        for name, label, valid_name in (
            ("Zin", r"$Z_\mathrm{in}$", "valid_Zin"),
            ("Zin_source", r"$Z_\mathrm{in}$ (source plane)", None),
            ("Zin_current", r"$Z_\mathrm{in}$ (current check)", "valid_Zin_current"),
        ):
            values = _require_vector(group, name, count, complex_values=True)
            if values is not None:
                valid = (
                    _valid_mask(group, valid_name, values, count)
                    if valid_name is not None
                    else _diagnostic_valid(group, name, values, count)
                )
                impedances.append(FrequencyTrace(name, label, values, valid))

        admittances: list[FrequencyTrace] = []
        values = _require_vector(group, "Yin", count, complex_values=True)
        if values is not None:
            admittances.append(
                FrequencyTrace(
                    "Yin",
                    r"$Y_\mathrm{in}$",
                    values,
                    _valid_mask(group, "valid_Yin", values, count),
                )
            )

        time_traces = tuple(
            trace
            for name, label, quantity, time_names in TIME_DATASETS
            if (trace := _time_trace(output, group, name, label, quantity, time_names)) is not None
        )
        spectral_traces: list[FrequencyTrace] = []
        for name, label, quantity in SPECTRAL_DATASETS:
            values = _require_vector(group, name, count, complex_values=True)
            if values is not None:
                spectral_traces.append(FrequencyTrace(name, label, values, np.isfinite(values)))

        validity_masks = {}
        for name in VALIDITY_DATASETS:
            if name in group:
                values = np.asarray(group[name], dtype=bool)
                if values.ndim != 1 or values.size != count:
                    raise ValueError(f"{group.name}/{name} has the wrong shape")
                validity_masks[name] = values

        diagnostics = {}
        for name in DIAGNOSTIC_DATASETS:
            values = _require_vector(group, name, count)
            if values is not None:
                diagnostics[name] = values

        metadata = {name: group.attrs[name] for name in METADATA_ATTRIBUTES if name in group.attrs}
        if not any(trace.name == "S11" for trace in s_parameters):
            raise ValueError(f"{group.name} does not contain primary S11 output")
        return PortData(
            path=path,
            id=path.rsplit("/", 1)[-1],
            source_type=_source_type(path, group),
            frequency=frequency,
            s_parameters=tuple(s_parameters),
            impedances=tuple(impedances),
            admittances=tuple(admittances),
            time_traces=time_traces,
            spectral_traces=tuple(spectral_traces),
            validity_masks=validity_masks,
            diagnostics=diagnostics,
            metadata=metadata,
        )


def read_port_params(filename, port_id=None):
    """Compatibility dictionary for callers of the retired reader."""

    port = read_port_output(filename, port_id)
    s11 = port.primary_s11
    zin = next((trace for trace in port.impedances if trace.name == "Zin"), None)
    yin = next((trace for trace in port.admittances if trace.name == "Yin"), None)
    return {
        "port_id": port.id,
        "port_path": port.path,
        "source_type": port.source_type,
        "frequency": port.frequency,
        "s11": s11.values,
        "zin": None if zin is None else zin.values,
        "yin": None if yin is None else yin.values,
        "valid_s11": s11.valid,
        "valid_zin": None if zin is None else zin.valid,
        "valid_yin": None if yin is None else yin.valid,
        "reference_impedance": port.metadata.get("ReferenceImpedance"),
        "time_traces": {
            quantity: [
                {
                    "name": trace.name,
                    "label": trace.label,
                    "time": trace.time,
                    "values": trace.values,
                }
                for trace in port.time_traces
                if trace.quantity == quantity
            ]
            for quantity in ("voltage", "current")
        },
        "spectral_traces": {
            quantity: [
                {
                    "name": trace.name,
                    "label": trace.label,
                    "values": trace.values,
                    "valid": trace.valid,
                }
                for trace in port.spectral_traces
                if (trace.name.startswith("I")) == (quantity == "current")
            ]
            for quantity in ("voltage", "current")
        },
    }


def _engineering_scale(maximum, scales):
    maximum = abs(float(maximum))
    for threshold, scale, label in scales:
        if maximum >= threshold:
            return scale, label
    return 1.0, scales[-1][2]


def _frequency_scale(frequency):
    maximum = np.max(np.abs(frequency), initial=0.0)
    return _engineering_scale(
        maximum,
        ((1e9, 1e9, "GHz"), (1e6, 1e6, "MHz"), (1e3, 1e3, "kHz"), (0, 1, "Hz")),
    )


def _time_scale(traces):
    maximum = max((np.max(np.abs(trace.time), initial=0.0) for trace in traces), default=0.0)
    return _engineering_scale(
        maximum,
        ((1, 1, "s"), (1e-3, 1e-3, "ms"), (1e-6, 1e-6, "us"), (1e-9, 1e-9, "ns"), (0, 1e-12, "ps")),
    )


def _frequency_selection(port, fmin, fmax):
    selected = port.frequency > 0
    if fmin is not None:
        selected &= port.frequency >= fmin
    if fmax is not None:
        selected &= port.frequency <= fmax
    if not np.any(selected):
        raise ValueError(f"no positive port frequencies lie in the selected range for {port.path}")
    return selected


def _plot_complex_trace(axis_real, axis_imag, x, trace, selected, show_invalid, *, db=False):
    valid = selected & trace.valid
    invalid = selected & ~trace.valid & np.isfinite(trace.values)
    if db:
        with np.errstate(divide="ignore", invalid="ignore"):
            first = 20 * np.log10(np.abs(trace.values))
        second = np.angle(trace.values, deg=True)
    else:
        first = trace.values.real
        second = trace.values.imag
    if show_invalid and np.any(invalid):
        axis_real.plot(x[invalid], first[invalid], color="0.75", linestyle=":")
        axis_imag.plot(x[invalid], second[invalid], color="0.75", linestyle=":")
    if np.any(valid):
        axis_real.plot(x[valid], first[valid], label=trace.label)
        axis_imag.plot(x[valid], second[valid], label=trace.label)


def plot_port_parameters(port, *, fmin=None, fmax=None, show_invalid=False):
    """Create a summary of stored S-parameters, impedance, and admittance."""

    selected = _frequency_selection(port, fmin, fmax)
    frequency_scale, frequency_unit = _frequency_scale(port.frequency[selected])
    x = port.frequency / frequency_scale
    has_impedance = bool(port.impedances)
    has_admittance = bool(port.admittances)
    rows = 1 + int(has_impedance) + int(has_admittance)
    figure, axes = plt.subplots(
        rows,
        2,
        figsize=(13, 3.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for trace in port.s_parameters:
        _plot_complex_trace(axes[0, 0], axes[0, 1], x, trace, selected, show_invalid, db=True)
    axes[0, 0].axhline(-10, color="0.5", linewidth=0.8, linestyle="--", label="-10 dB")
    axes[0, 0].set(ylabel="Magnitude [dB]", title="Reflection coefficient")
    axes[0, 1].set(ylabel="Phase [degrees]", title="Reflection-coefficient phase")

    row = 1
    if has_impedance:
        for trace in port.impedances:
            _plot_complex_trace(axes[row, 0], axes[row, 1], x, trace, selected, show_invalid)
        axes[row, 0].set(ylabel="Resistance [Ohms]", title="Input resistance")
        axes[row, 1].set(ylabel="Reactance [Ohms]", title="Input reactance")
        row += 1
    if has_admittance:
        for trace in port.admittances:
            _plot_complex_trace(axes[row, 0], axes[row, 1], x, trace, selected, show_invalid)
        axes[row, 0].set(ylabel="Conductance [S]", title="Input conductance")
        axes[row, 1].set(ylabel="Susceptance [S]", title="Input susceptance")

    for axis in axes.flat:
        axis.set_xlabel(f"Frequency [{frequency_unit}]")
        axis.grid(True, alpha=0.3)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    reference = port.metadata.get("ReferenceImpedance")
    title = f"{port.source_type}: {port.path}"
    if reference is not None:
        title += f", Z0 = {float(reference):g} Ohms"
    figure.suptitle(title)
    return figure


def plot_port_signals(port, *, fmin=None, fmax=None, tmin=None, tmax=None):
    """Plot all terminal histories and spectra which are present."""

    quantities = tuple(
        quantity
        for quantity in ("voltage", "current")
        if any(trace.quantity == quantity for trace in port.time_traces)
        or any(
            (trace.name.startswith("I")) == (quantity == "current")
            for trace in port.spectral_traces
        )
    )
    if not quantities:
        return None

    selected = _frequency_selection(port, fmin, fmax)
    frequency_scale, frequency_unit = _frequency_scale(port.frequency[selected])
    time_scale, time_unit = _time_scale(port.time_traces)
    figure, axes = plt.subplots(
        len(quantities),
        2,
        figsize=(13, 4 * len(quantities)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, quantity in enumerate(quantities):
        histories = [trace for trace in port.time_traces if trace.quantity == quantity]
        spectra = [
            trace
            for trace in port.spectral_traces
            if (trace.name.startswith("I")) == (quantity == "current")
        ]
        for trace in histories:
            selected_time = np.ones(trace.time.shape, dtype=bool)
            if tmin is not None:
                selected_time &= trace.time >= tmin
            if tmax is not None:
                selected_time &= trace.time <= tmax
            axes[row, 0].plot(
                trace.time[selected_time] / time_scale,
                trace.values[selected_time],
                label=trace.label,
            )
        for trace in spectra:
            axes[row, 1].plot(
                port.frequency[selected] / frequency_scale,
                np.abs(trace.values[selected]),
                label=trace.label,
            )
        axes[row, 0].set(
            xlabel=f"Time [{time_unit}]",
            ylabel="Voltage [V]" if quantity == "voltage" else "Current [A]",
            title=f"{quantity.capitalize()} histories",
        )
        axes[row, 1].set(
            xlabel=f"Frequency [{frequency_unit}]",
            ylabel="|V(f)| [V s]" if quantity == "voltage" else "|I(f)| [A s]",
            title=f"{quantity.capitalize()} spectra",
        )
        for axis in axes[row]:
            axis.grid(True, alpha=0.3)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend()
            else:
                axis.text(
                    0.5, 0.5, "Not stored", ha="center", va="center", transform=axis.transAxes
                )
    figure.suptitle(f"{port.source_type}: {port.path}")
    return figure


def plot_port_validity(port, *, fmin=None, fmax=None):
    """Plot stored validity reasons and numerical-band diagnostics."""

    if not port.validity_masks and not port.diagnostics:
        return None
    selected = _frequency_selection(port, fmin, fmax)
    scale, unit = _frequency_scale(port.frequency[selected])
    rows = int(bool(port.validity_masks)) + int(bool(port.diagnostics))
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=(12, 3.4 * rows),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
    )
    row = 0
    if port.validity_masks:
        axis = axes[row, 0]
        offsets = np.arange(len(port.validity_masks), dtype=float) * 1.25
        for offset, (name, mask) in zip(offsets, port.validity_masks.items(), strict=True):
            axis.step(
                port.frequency[selected] / scale,
                mask[selected].astype(float) + offset,
                where="mid",
            )
        axis.set(ylabel="Validity (1 = valid)")
        axis.set_yticks(offsets + 1, tuple(port.validity_masks))
        axis.set_ylim(-0.15, offsets[-1] + 1.15)
        axis.grid(True, axis="x", alpha=0.3)
        row += 1

    if port.diagnostics:
        axis = axes[row, 0]
        source = port.diagnostics.get("incident_relative_dB")
        mesh = port.diagnostics.get("cells_per_minimum_wavelength")
        if source is not None:
            axis.plot(
                port.frequency[selected] / scale,
                source[selected],
                label="Incident spectrum relative to peak",
            )
            floor = port.metadata.get("IncidentFloorDB")
            if floor is not None:
                axis.axhline(float(floor), color="0.5", linestyle="--", label="Source floor")
            axis.set_ylabel("Incident level [dB]")
        if mesh is not None:
            mesh_axis = axis.twinx() if source is not None else axis
            mesh_axis.plot(
                port.frequency[selected] / scale,
                mesh[selected],
                color="tab:orange",
                label="Cells per minimum wavelength",
            )
            minimum = port.metadata.get("MinimumWavelengthCells")
            if minimum is not None:
                mesh_axis.axhline(
                    float(minimum),
                    color="tab:orange",
                    linestyle="--",
                    label="Mesh criterion",
                )
            mesh_axis.set_ylabel("Cells per minimum wavelength")
            if mesh_axis is axis:
                axis.legend()
            else:
                handles, labels = axis.get_legend_handles_labels()
                other_handles, other_labels = mesh_axis.get_legend_handles_labels()
                axis.legend(handles + other_handles, labels + other_labels)
        else:
            axis.legend()
        axis.grid(True, axis="x", alpha=0.3)

    axes[-1, 0].set_xlabel(f"Frequency [{unit}]")
    figure.suptitle(f"Port validity diagnostics: {port.path}")
    return figure


def _report_port(port):
    trace = port.primary_s11
    valid = trace.valid & (port.frequency > 0) & np.isfinite(trace.values)
    if np.any(valid):
        with np.errstate(divide="ignore"):
            magnitude_db = 20 * np.log10(np.abs(trace.values))
        indices = np.flatnonzero(valid)
        minimum = indices[np.nanargmin(magnitude_db[valid])]
        message = (
            f"{port.path}: S11 minimum {magnitude_db[minimum]:.3g} dB at "
            f"{port.frequency[minimum]:.6g} Hz"
        )
        primary_zin = next((item for item in port.impedances if item.name == "Zin"), None)
        if primary_zin is not None and primary_zin.valid[minimum]:
            value = primary_zin.values[minimum]
            message += f", Zin {value.real:.3g}{value.imag:+.3g}j Ohms"
        logger.info(message)
    tail = port.metadata.get("TailRelativeLevelDB")
    if tail is not None and np.isfinite(tail) and float(tail) > -40:
        logger.warning(
            "%s: stored terminal tail is %.1f dB relative to its peak; spectral leakage may be significant",
            port.path,
            float(tail),
        )


def save_port_figures(
    filename: str | Path,
    port: PortData,
    *,
    output_dir: str | Path | None = None,
    image_format: str = "png",
    dpi: int = 180,
    parameters=None,
    signals=None,
    validity=None,
) -> tuple[Path, ...]:
    """Save supplied port figures using unique, reproducible filenames."""

    image_format = image_format.lower()
    if image_format not in {"png", "pdf", "svg"}:
        raise ValueError("image_format must be png, pdf, or svg")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    input_path = Path(filename)
    directory = Path(output_dir) if output_dir is not None else input_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{input_path.stem}_{_safe_filename(port.path)}"
    outputs = []
    for suffix, figure in (
        ("parameters", parameters),
        ("signals", signals),
        ("validity", validity),
    ):
        if figure is None:
            continue
        destination = directory / f"{stem}_{suffix}.{image_format}"
        figure.savefig(destination, dpi=dpi, format=image_format, bbox_inches="tight")
        outputs.append(destination)
        logger.info("Plot saved to: %s", destination)
    return tuple(outputs)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot S-parameters and terminal quantities stored by gprMax ports.",
    )
    parser.add_argument("outputfile", type=Path, help="gprMax HDF5 output file")
    parser.add_argument(
        "--port",
        action="append",
        help="port ID or complete HDF5 path; repeat to select several (default: all)",
    )
    parser.add_argument("--list-ports", action="store_true", help="list stored port paths")
    parser.add_argument("--fmin", type=float, help="minimum displayed frequency [Hz]")
    parser.add_argument("--fmax", type=float, help="maximum displayed frequency [Hz]")
    parser.add_argument("--tmin", type=float, help="minimum displayed history time [s]")
    parser.add_argument("--tmax", type=float, help="maximum displayed history time [s]")
    parser.add_argument(
        "--parameters-only", "--params-only", action="store_true", help="omit terminal signals"
    )
    parser.add_argument(
        "--validity",
        action="store_true",
        help="also plot stored source, mesh, gap, and propagation validity masks",
    )
    parser.add_argument(
        "--show-invalid",
        action="store_true",
        help="show finite invalid parameter values as grey dotted lines",
    )
    parser.add_argument("--save", "-save", action="store_true", help="save instead of display")
    parser.add_argument("--output-dir", type=Path, help="directory for saved plots")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv=None):
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    args = build_parser().parse_args(argv)
    available = discover_port_outputs(args.outputfile)
    if args.list_ports:
        print("\n".join(available) if available else "No port outputs found")
        return 0
    selected = select_port_paths(available, args.port, args.outputfile)
    figures = []
    for path in selected:
        port = read_port_output(args.outputfile, path)
        _report_port(port)
        parameters = plot_port_parameters(
            port,
            fmin=args.fmin,
            fmax=args.fmax,
            show_invalid=args.show_invalid,
        )
        signals = None
        if not args.parameters_only:
            signals = plot_port_signals(
                port,
                fmin=args.fmin,
                fmax=args.fmax,
                tmin=args.tmin,
                tmax=args.tmax,
            )
        validity = (
            plot_port_validity(port, fmin=args.fmin, fmax=args.fmax) if args.validity else None
        )
        figures.extend(figure for figure in (parameters, signals, validity) if figure is not None)
        if args.save or args.output_dir is not None:
            save_port_figures(
                args.outputfile,
                port,
                output_dir=args.output_dir,
                image_format=args.format,
                dpi=args.dpi,
                parameters=parameters,
                signals=signals,
                validity=validity,
            )

    if not args.save and args.output_dir is None:
        plt.show()
    for figure in figures:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
