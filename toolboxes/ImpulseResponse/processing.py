# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Synthesis of arbitrary source waveforms from one FDTD impulse response."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve

from toolboxes.SFCW.processing import (
    SampledSignal,
    apply_tail_taper,
    list_receivers,
    list_sources,
    load_receiver,
    load_source,
    tail_relative_db,
)

BUILTIN_WAVEFORM_TYPES = (
    "gaussian",
    "gaussiandot",
    "gaussiandotnorm",
    "gaussiandotdot",
    "gaussiandotdotnorm",
    "gaussianprime",
    "gaussiandoubleprime",
    "ricker",
    "sine",
    "contsine",
    "impulse",
)


@dataclass(frozen=True)
class SourceSampling:
    """Stored scalar source and the time used to evaluate its waveform."""

    signal: SampledSignal
    evaluation_time_offset: float
    update_lattice: str
    driving_quantity: str


@dataclass(frozen=True)
class TargetWaveform:
    """A target driving waveform sampled exactly on the source update lattice."""

    id: str
    samples: npt.NDArray[np.float64]
    dt: float
    source_time_offset: float
    evaluation_time_offset: float
    waveform_type: str
    amplitude: float | None = None
    frequency: float | None = None
    start_time: float = 0.0
    stop_time: float | None = None
    input_file: str = ""

    @property
    def source_times(self) -> npt.NDArray[np.float64]:
        """Physical times associated with the stored driving samples."""

        return self.source_time_offset + self.dt * np.arange(self.samples.size, dtype=np.float64)


@dataclass(frozen=True)
class SynthesisedReceiver:
    """One receiver component synthesised for a target waveform."""

    input: SampledSignal
    samples: npt.NDArray[np.float64]
    impulse_tail_relative_db: float


@dataclass(frozen=True)
class SynthesisResult:
    """All requested receiver components for one target waveform."""

    waveform: TargetWaveform
    impulse_source: SourceSampling
    receivers: tuple[SynthesisedReceiver, ...]
    impulse_index: int
    impulse_amplitude: float
    receiver_file: str
    source_file: str
    tail_taper_fraction: float
    valid_max_frequency: float | None
    energy_above_valid_band: float | None


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _normalise_path(path: str) -> str:
    return "/" + str(path).strip("/")


def _validate_id(value: str) -> str:
    identifier = str(value).strip()
    if not identifier or re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) is None:
        raise ValueError(
            "waveform IDs must contain only letters, digits, underscores, dots, or hyphens"
        )
    return identifier


def load_source_sampling(
    filename: str | Path,
    source_path: str | None = None,
) -> SourceSampling:
    """Load the impulse source and its waveform-evaluation time convention."""

    signal = load_source(filename, source_path)
    group_path = _normalise_path(signal.path)
    with h5py.File(filename, "r") as output:
        excitation = output[f"{group_path}/excitation"]
        driving_quantity = _text(excitation.attrs.get("DrivingQuantity", signal.quantity))
        update_lattice = _text(excitation.attrs.get("UpdateLattice", ""))
        if "WaveformEvaluationTimeOffset" in excitation.attrs:
            evaluation_offset = float(excitation.attrs["WaveformEvaluationTimeOffset"])
        elif driving_quantity == "imposed_gap_voltage":
            # A hard voltage source evaluates waveform sample n at n*dt but
            # imposes it on the electric field stored at (n+1)*dt.
            evaluation_offset = 0.0
        else:
            evaluation_offset = signal.time_offset
    return SourceSampling(
        signal=signal,
        evaluation_time_offset=evaluation_offset,
        update_lattice=update_lattice,
        driving_quantity=driving_quantity,
    )


def find_single_impulse(samples: npt.ArrayLike) -> tuple[int, float]:
    """Return the index and amplitude of a one-sample discrete impulse."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("the selected source must contain one finite time history")
    peak = float(np.max(np.abs(values), initial=0.0))
    if peak == 0:
        raise ValueError("the selected source excitation is identically zero")
    tolerance = 64 * np.finfo(np.float64).eps * peak
    significant = np.flatnonzero(np.abs(values) > tolerance)
    if significant.size != 1:
        raise ValueError(
            "waveform synthesis requires a one-sample impulse excitation; "
            f"the selected source has {significant.size} significant samples"
        )
    index = int(significant[0])
    return index, float(values[index])


def validate_single_active_source(filename: str | Path, selected_path: str) -> None:
    """Reject a reference run containing another active scalar source."""

    selected = _normalise_path(selected_path)
    active: list[str] = []
    for path in list_sources(filename):
        signal = load_source(filename, path)
        peak = float(np.max(np.abs(signal.samples), initial=0.0))
        if peak > 0:
            active.append(_normalise_path(path))
    others = [path for path in active if path != selected]
    if others:
        raise ValueError(
            "the impulse reference must contain one active scalar source; "
            f"additional active sources were found at {others}"
        )


def _builtin_values(waveform_type: str, amplitude: float, frequency: float, time, dt):
    """Vectorised equivalent of :class:`gprMax.waveforms.Waveform`."""

    key = str(waveform_type).lower()
    if key not in BUILTIN_WAVEFORM_TYPES:
        raise ValueError(f"unknown built-in waveform {waveform_type!r}")
    if not np.isfinite(amplitude):
        raise ValueError("waveform amplitude must be finite")
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("waveform frequency must be finite and positive")
    values = np.asarray(time, dtype=np.float64)

    if key in {
        "gaussian",
        "gaussiandot",
        "gaussiandotnorm",
        "gaussianprime",
        "gaussiandoubleprime",
    }:
        chi = 1 / frequency
        zeta = 2 * np.pi**2 * frequency**2
    elif key in {"gaussiandotdot", "gaussiandotdotnorm", "ricker"}:
        chi = np.sqrt(2) / frequency
        zeta = np.pi**2 * frequency**2

    if key == "gaussian":
        result = np.exp(-zeta * (values - chi) ** 2)
    elif key in {"gaussiandot", "gaussianprime"}:
        delay = values - chi
        result = -2 * zeta * delay * np.exp(-zeta * delay**2)
    elif key == "gaussiandotnorm":
        delay = values - chi
        normalise = np.sqrt(np.e / (2 * zeta))
        result = -2 * zeta * delay * np.exp(-zeta * delay**2) * normalise
    elif key in {"gaussiandotdot", "gaussiandoubleprime"}:
        delay = values - chi
        result = 2 * zeta * (2 * zeta * delay**2 - 1) * np.exp(-zeta * delay**2)
    elif key == "gaussiandotdotnorm":
        delay = values - chi
        result = (2 * zeta * (2 * zeta * delay**2 - 1) * np.exp(-zeta * delay**2)) / (2 * zeta)
    elif key == "ricker":
        delay = values - chi
        result = -(2 * zeta * (2 * zeta * delay**2 - 1) * np.exp(-zeta * delay**2)) / (2 * zeta)
    elif key == "sine":
        result = np.sin(2 * np.pi * frequency * values)
        result = np.where(values * frequency > 1, 0.0, result)
    elif key == "contsine":
        ramp = np.minimum(0.25 * values * frequency, 1.0)
        result = ramp * np.sin(2 * np.pi * frequency * values)
    else:
        result = np.where((values == 0) | (values < dt), 1.0, 0.0)
    return np.asarray(amplitude * result, dtype=np.float64)


def sample_builtin_waveform(
    source: SourceSampling,
    waveform_type: str,
    amplitude: float,
    frequency: float,
    waveform_id: str,
    *,
    start_time: float = 0.0,
    stop_time: float | None = None,
) -> TargetWaveform:
    """Sample one built-in gprMax waveform as the selected source would."""

    if not np.isfinite(start_time) or start_time < 0:
        raise ValueError("start_time must be finite and non-negative")
    if stop_time is not None and (not np.isfinite(stop_time) or stop_time < start_time):
        raise ValueError("stop_time must be finite and no earlier than start_time")
    signal = source.signal
    update_times = signal.dt * np.arange(signal.samples.size, dtype=np.float64)
    local_times = update_times - start_time + source.evaluation_time_offset
    active = update_times >= start_time
    if stop_time is not None:
        active &= update_times <= stop_time
    samples = np.zeros(signal.samples.size, dtype=np.float64)
    samples[active] = _builtin_values(
        waveform_type,
        float(amplitude),
        float(frequency),
        local_times[active],
        signal.dt,
    )
    return TargetWaveform(
        id=_validate_id(waveform_id),
        samples=samples,
        dt=signal.dt,
        source_time_offset=signal.time_offset,
        evaluation_time_offset=source.evaluation_time_offset,
        waveform_type=str(waveform_type).lower(),
        amplitude=float(amplitude),
        frequency=float(frequency),
        start_time=float(start_time),
        stop_time=None if stop_time is None else float(stop_time),
    )


def load_csv_waveforms(
    filename: str | Path,
    source: SourceSampling,
    *,
    start_time: float = 0.0,
) -> tuple[TargetWaveform, ...]:
    """Load named waveforms from a CSV time column and linearly resample them."""

    path = Path(filename)
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    names = table.dtype.names or ()
    if len(names) < 2 or names[0].lower() not in {"time", "t", "seconds"}:
        raise ValueError("waveform CSV must contain a first 'time' column and one or more signals")
    times = np.atleast_1d(np.asarray(table[names[0]], dtype=np.float64))
    if times.size < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("CSV waveform times must be finite and strictly increasing")
    if not np.isfinite(start_time) or start_time < 0:
        raise ValueError("start_time must be finite and non-negative")

    signal = source.signal
    update_times = signal.dt * np.arange(signal.samples.size, dtype=np.float64)
    local_times = update_times - start_time + source.evaluation_time_offset
    active = update_times >= start_time
    waveforms: list[TargetWaveform] = []
    for name in names[1:]:
        values = np.atleast_1d(np.asarray(table[name], dtype=np.float64))
        if values.shape != times.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"CSV waveform column {name!r} must contain finite values")
        samples = np.zeros(signal.samples.size, dtype=np.float64)
        samples[active] = np.interp(
            local_times[active],
            times,
            values,
            left=0.0,
            right=0.0,
        )
        waveforms.append(
            TargetWaveform(
                id=_validate_id(name),
                samples=samples,
                dt=signal.dt,
                source_time_offset=signal.time_offset,
                evaluation_time_offset=source.evaluation_time_offset,
                waveform_type="sampled_csv",
                start_time=float(start_time),
                input_file=str(path),
            )
        )
    return tuple(waveforms)


def waveform_energy_above(samples: npt.ArrayLike, dt: float, frequency: float) -> float:
    """Return the fraction of discrete spectral energy above a frequency."""

    values = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(frequency) or frequency <= 0 or frequency > 1 / (2 * dt):
        raise ValueError("valid maximum frequency must lie in (0, Nyquist]")
    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(values.size, d=dt)
    energy = np.abs(spectrum) ** 2
    total = float(np.sum(energy))
    if total == 0:
        return 0.0
    return float(np.sum(energy[frequencies > frequency]) / total)


def synthesise_receiver(
    impulse_source: SourceSampling,
    receiver: SampledSignal,
    waveform: TargetWaveform,
    *,
    tail_taper_fraction: float = 0.0,
) -> tuple[SynthesisedReceiver, int, float]:
    """Causally convolve one stored receiver impulse response with a waveform."""

    source = impulse_source.signal
    if not np.isclose(source.dt, receiver.dt, rtol=0, atol=32 * np.finfo(float).eps * source.dt):
        raise ValueError("source and receiver sample intervals are different")
    if not np.isclose(source.dt, waveform.dt, rtol=0, atol=32 * np.finfo(float).eps * source.dt):
        raise ValueError("source and target-waveform sample intervals are different")
    if waveform.samples.ndim != 1 or waveform.samples.size != source.samples.size:
        raise ValueError("target waveform must have the same sample count as the impulse source")
    if not np.isfinite(tail_taper_fraction) or not 0 <= tail_taper_fraction <= 1:
        raise ValueError("tail_taper_fraction must lie in [0, 1]")

    impulse_index, impulse_amplitude = find_single_impulse(source.samples)
    tail_db = tail_relative_db(receiver.samples)
    impulse_output = apply_tail_taper(receiver.samples, tail_taper_fraction)
    target_shape = (waveform.samples.size,) + (1,) * (impulse_output.ndim - 1)
    full = fftconvolve(
        impulse_output / impulse_amplitude,
        waveform.samples.reshape(target_shape),
        mode="full",
        axes=0,
    )
    stop = impulse_index + receiver.samples.shape[0]
    if stop > full.shape[0]:
        pad = [(0, stop - full.shape[0])] + [(0, 0)] * (full.ndim - 1)
        full = np.pad(full, pad)
    samples = np.asarray(full[impulse_index:stop], dtype=np.float64)
    return (
        SynthesisedReceiver(
            input=receiver,
            samples=samples,
            impulse_tail_relative_db=tail_db,
        ),
        impulse_index,
        impulse_amplitude,
    )


def _receiver_selections(
    filename: str | Path,
    selections: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None,
) -> tuple[tuple[str, str], ...]:
    available = list_receivers(filename)
    if selections is None:
        requested = tuple(
            (path, component) for path, components in available.items() for component in components
        )
        if not requested:
            raise ValueError(f"no receiver components are available in {filename}")
        return requested
    requested: list[tuple[str, str]] = []
    for path, component in selections:
        normalised = _normalise_path(path)
        if normalised not in available or component not in available[normalised]:
            raise ValueError(
                f"receiver selection {normalised}:{component} is unavailable; "
                f"available receivers are {available}"
            )
        requested.append((normalised, component))
    if not requested:
        raise ValueError("at least one receiver component must be selected")
    return tuple(requested)


def synthesise_output(
    filename: str | Path,
    waveform: TargetWaveform,
    *,
    source_filename: str | Path | None = None,
    source_path: str | None = None,
    receiver_selections: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
    tail_taper_fraction: float = 0.0,
    valid_max_frequency: float | None = None,
) -> SynthesisResult:
    """Synthesize selected receiver outputs for one target waveform."""

    receiver_file = Path(filename)
    source_file = Path(source_filename or filename)
    source = load_source_sampling(source_file, source_path)
    validate_single_active_source(source_file, source.signal.path)
    selections = _receiver_selections(receiver_file, receiver_selections)
    receivers: list[SynthesisedReceiver] = []
    impulse_index = 0
    impulse_amplitude = 0.0
    for path, component in selections:
        receiver = load_receiver(receiver_file, path, component)
        result, impulse_index, impulse_amplitude = synthesise_receiver(
            source,
            receiver,
            waveform,
            tail_taper_fraction=tail_taper_fraction,
        )
        receivers.append(result)
    energy = None
    if valid_max_frequency is not None:
        energy = waveform_energy_above(waveform.samples, waveform.dt, valid_max_frequency)
    return SynthesisResult(
        waveform=waveform,
        impulse_source=source,
        receivers=tuple(receivers),
        impulse_index=impulse_index,
        impulse_amplitude=impulse_amplitude,
        receiver_file=str(receiver_file),
        source_file=str(source_file),
        tail_taper_fraction=float(tail_taper_fraction),
        valid_max_frequency=None if valid_max_frequency is None else float(valid_max_frequency),
        energy_above_valid_band=energy,
    )


def _copy_attrs(source, destination) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _copy_group_attrs(filename: str | Path, path: str, destination) -> None:
    with h5py.File(filename, "r") as source:
        if path in source:
            _copy_attrs(source[path], destination)


def write_synthesised_output(filename: str | Path, result: SynthesisResult) -> Path:
    """Write one waveform result using receiver paths compatible with gprMax output."""

    path = Path(filename)
    resolved_output = path.resolve()
    if resolved_output in {
        Path(result.receiver_file).resolve(),
        Path(result.source_file).resolve(),
    }:
        raise ValueError("the synthesised output must not overwrite an input HDF5 file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(result.receiver_file, "r") as original, h5py.File(path, "w") as output:
        _copy_attrs(original, output)
        output.attrs["Format"] = "gprMax impulse-response waveform synthesis"
        output.attrs["InputImpulseFile"] = result.receiver_file
        output.attrs["InputSourceFile"] = result.source_file
        output.attrs["ImpulseSourcePath"] = result.impulse_source.signal.path
        output.attrs["ImpulseIndex"] = result.impulse_index
        output.attrs["ImpulseAmplitude"] = result.impulse_amplitude
        output.attrs["TailTaperFraction"] = result.tail_taper_fraction
        output.attrs["Iterations"] = result.receivers[0].samples.shape[0]
        output.attrs["nsrc"] = 1
        output.attrs["nrx"] = len({item.input.path.rsplit("/", 1)[0] for item in result.receivers})
        if result.valid_max_frequency is not None:
            output.attrs["ValidMaximumFrequency"] = result.valid_max_frequency
            output.attrs["WaveformEnergyAboveValidBand"] = result.energy_above_valid_band

        source_path = _normalise_path(result.impulse_source.signal.path)
        source_group = output.require_group(source_path)
        _copy_group_attrs(result.source_file, source_path, source_group)
        excitation_path = f"{source_path}/excitation"
        excitation = output.require_group(excitation_path)
        _copy_group_attrs(result.source_file, excitation_path, excitation)
        excitation.attrs["WaveformID"] = result.waveform.id
        excitation.attrs["WaveformType"] = result.waveform.waveform_type
        excitation.attrs["WaveformEvaluationTimeOffset"] = result.waveform.evaluation_time_offset
        excitation.attrs["SourceStartTime"] = result.waveform.start_time
        excitation.attrs["SourceStopTime"] = (
            result.waveform.samples.size * result.waveform.dt
            if result.waveform.stop_time is None
            else result.waveform.stop_time
        )
        if result.waveform.amplitude is not None:
            excitation.attrs["WaveformAmplitude"] = result.waveform.amplitude
        if result.waveform.frequency is not None:
            excitation.attrs["WaveformFrequency"] = result.waveform.frequency
        if result.waveform.input_file:
            excitation.attrs["WaveformInputFile"] = result.waveform.input_file
        samples = excitation.create_dataset(
            "samples",
            data=result.waveform.samples,
            compression="gzip",
            shuffle=True,
        )
        samples.attrs["SampleCount"] = result.waveform.samples.size

        reference = output.create_group("impulse_reference")
        reference.attrs["SourcePath"] = source_path
        reference.attrs["SourceTimeSampleOffset"] = result.impulse_source.signal.time_offset
        reference.attrs[
            "WaveformEvaluationTimeOffset"
        ] = result.impulse_source.evaluation_time_offset
        reference.create_dataset(
            "source_samples",
            data=result.impulse_source.signal.samples,
            compression="gzip",
            shuffle=True,
        )

        for receiver in result.receivers:
            dataset_path = _normalise_path(receiver.input.path)
            group_path = dataset_path.rsplit("/", 1)[0]
            group = output.require_group(group_path)
            _copy_group_attrs(result.receiver_file, group_path, group)
            dataset = group.create_dataset(
                dataset_path.rsplit("/", 1)[1],
                data=receiver.samples,
                compression="gzip",
                shuffle=True,
            )
            with h5py.File(result.receiver_file, "r") as receiver_source:
                _copy_attrs(receiver_source[dataset_path], dataset)
            dataset.attrs["SynthesisedFromImpulse"] = True
            dataset.attrs["ImpulseTailRelativeDB"] = receiver.impulse_tail_relative_db
    return path
