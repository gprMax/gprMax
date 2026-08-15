# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""FMCW synthesis from timing-aware gprMax broadband responses."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from toolboxes.SFCW.processing import (
    FrequencyResponse,
    apply_tail_taper,
    direct_frequency_response,
    engineering_dft,
    load_receiver,
    load_source,
    spectral_window,
    tail_relative_db,
)


@dataclass(frozen=True)
class Chirp:
    """Uniform linear-FMCW sweep sampled at the requested tone centres."""

    f_start: float
    f_stop: float
    sweep_time: float
    samples: int
    direction: str = "up"

    def __post_init__(self):
        if not np.isfinite(self.f_start) or self.f_start <= 0:
            raise ValueError("f_start must be finite and positive")
        if not np.isfinite(self.f_stop) or self.f_stop <= self.f_start:
            raise ValueError("f_stop must be finite and greater than f_start")
        if not np.isfinite(self.sweep_time) or self.sweep_time <= 0:
            raise ValueError("sweep_time must be finite and positive")
        if isinstance(self.samples, bool) or not isinstance(self.samples, Integral) or self.samples < 2:
            raise ValueError("samples must be an integer of at least two")
        if self.direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")

    @property
    def bandwidth(self) -> float:
        return float(self.f_stop - self.f_start)

    @property
    def slope(self) -> float:
        sign = 1 if self.direction == "up" else -1
        return sign * self.bandwidth / self.sweep_time

    @property
    def frequency_step(self) -> float:
        return self.bandwidth / self.samples

    @property
    def frequency(self) -> npt.NDArray[np.float64]:
        """Ascending endpoint-exclusive frequency grid used for processing."""

        return self.f_start + self.frequency_step * np.arange(self.samples, dtype=np.float64)

    @property
    def instantaneous_frequency(self) -> npt.NDArray[np.float64]:
        """Frequency order encountered by the physical sweep."""

        frequency = self.frequency
        return frequency if self.direction == "up" else frequency[::-1]

    @property
    def slow_time(self) -> npt.NDArray[np.float64]:
        return self.sweep_time * np.arange(self.samples, dtype=np.float64) / self.samples

    @property
    def delay(self) -> npt.NDArray[np.float64]:
        return np.arange(self.samples, dtype=np.float64) / self.bandwidth


@dataclass(frozen=True)
class ChannelResponse:
    """Source-normalised target response, optionally background subtracted."""

    chirp: Chirp
    response: npt.NDArray[np.complex128]
    target: FrequencyResponse
    background: FrequencyResponse | None
    source_valid: npt.NDArray[np.bool_]
    normalisation: str


@dataclass(frozen=True)
class FastTimeResponse:
    """FMCW delay-domain response reconstructed from one sweep."""

    delay: npt.NDArray[np.float64]
    complex_envelope: npt.NDArray[np.complex128]
    complex_bandpass: npt.NDArray[np.complex128]
    real_bandpass: npt.NDArray[np.float64]
    processed_spectrum: npt.NDArray[np.complex128]
    instrument_response: npt.NDArray[np.complex128]
    receiver_delay_response: npt.NDArray[np.complex128]
    weights: npt.NDArray[np.float64]
    window: str
    range: npt.NDArray[np.float64] | None
    propagation_velocity: float | None
    residual_video_phase: str


@dataclass(frozen=True)
class DerampedSweep:
    """Synthetic ideal stretch-receiver I/Q samples for one FMCW sweep."""

    slow_time: npt.NDArray[np.float64]
    instantaneous_frequency: npt.NDArray[np.float64]
    complex_signal: npt.NDArray[np.complex128]
    in_phase: npt.NDArray[np.float64]
    quadrature: npt.NDArray[np.float64]
    beat_frequency: npt.NDArray[np.float64]
    delay: npt.NDArray[np.float64]
    residual_video_phase: str


def process_channel(
    filename: str | Path,
    chirp: Chirp,
    *,
    background_filename: str | Path | None = None,
    source_filename: str | Path | None = None,
    background_source_filename: str | Path | None = None,
    source_path: str | None = None,
    receiver_path: str | None = None,
    component: str | None = None,
    background_source_path: str | None = None,
    background_receiver_path: str | None = None,
    background_component: str | None = None,
    source_floor_db: float = -100.0,
    tail_taper_fraction: float = 0.0,
) -> ChannelResponse:
    """Calculate an FMCW channel from one broadband target/reference pair.

    Target and background records are normalised by their own exact stored
    source histories before subtraction. This remains correct when nominally
    identical simulations have small source-sampling differences.
    """

    frequency = chirp.frequency
    target_source = load_source(source_filename or filename, source_path)
    target_receiver = load_receiver(filename, receiver_path, component)
    target = direct_frequency_response(
        target_source,
        target_receiver,
        frequency,
        source_floor_db=source_floor_db,
        tail_taper_fraction=tail_taper_fraction,
    )
    response = np.asarray(target.response, dtype=np.complex128)
    valid = np.asarray(target.source_valid, dtype=bool)
    background = None

    if background_filename is not None:
        background_source = load_source(
            background_source_filename or background_filename,
            background_source_path or source_path,
        )
        background_receiver = load_receiver(
            background_filename,
            background_receiver_path or receiver_path,
            background_component or component,
        )
        background = direct_frequency_response(
            background_source,
            background_receiver,
            frequency,
            source_floor_db=source_floor_db,
            tail_taper_fraction=tail_taper_fraction,
        )
        background_response = np.asarray(background.response, dtype=np.complex128)
        if response.ndim == 2 and background_response.ndim == 1:
            background_response = background_response[:, None]
        compatible = background_response.shape == response.shape or (
            response.ndim == 2 and background_response.ndim == 2 and background_response.shape == (response.shape[0], 1)
        )
        if not compatible:
            raise ValueError("target and background responses have incompatible trace dimensions")
        response = response - background_response
        valid = valid & background.source_valid

    return ChannelResponse(
        chirp=chirp,
        response=np.asarray(response, dtype=np.complex128),
        target=target,
        background=background,
        source_valid=np.asarray(valid, dtype=bool),
        normalisation="stored-source",
    )


def process_incident_referenced_channel(
    filename: str | Path,
    incident_filename: str | Path,
    chirp: Chirp,
    *,
    receiver_path: str | None = None,
    component: str | None = None,
    incident_receiver_path: str | None = None,
    incident_component: str | None = None,
    source_floor_db: float = -100.0,
    tail_taper_fraction: float = 0.0,
) -> ChannelResponse:
    """Normalise a total-field record by a measured incident reference.

    This route is intended for discrete plane waves and other excitations that
    do not expose a scalar stored-source history. The returned response is
    ``(total - incident) / incident``. One incident trace may normalise every
    trace of a merged target B-scan.
    """

    total = load_receiver(filename, receiver_path, component)
    incident = load_receiver(
        incident_filename,
        incident_receiver_path or receiver_path,
        incident_component or component,
    )
    tolerance = 32 * np.finfo(float).eps * total.dt
    if not np.isclose(total.dt, incident.dt, rtol=0, atol=tolerance):
        raise ValueError("total and incident sample intervals are different")
    if incident.samples.ndim != 1:
        raise ValueError("incident-reference normalisation requires one incident trace")
    frequency = chirp.frequency
    total_spectrum = engineering_dft(
        apply_tail_taper(total.samples, tail_taper_fraction),
        total.dt,
        frequency,
        time_offset=total.time_offset,
    )
    incident_spectrum = engineering_dft(
        apply_tail_taper(incident.samples, tail_taper_fraction),
        incident.dt,
        frequency,
        time_offset=incident.time_offset,
    )
    peak = float(np.max(np.abs(incident_spectrum), initial=0.0))
    if peak == 0:
        raise ValueError("the incident reference is identically zero")
    valid = np.abs(incident_spectrum) > peak * 10 ** (float(source_floor_db) / 20)
    divisor = _broadcast_frequency_vector(incident_spectrum, total_spectrum.ndim)
    valid_divisor = _broadcast_frequency_vector(valid, total_spectrum.ndim)
    response = np.full(total_spectrum.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    np.divide(
        total_spectrum - divisor,
        divisor,
        out=response,
        where=valid_divisor,
    )
    target_response = np.full(total_spectrum.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    np.divide(total_spectrum, divisor, out=target_response, where=valid_divisor)
    target = FrequencyResponse(
        frequency=frequency,
        response=target_response,
        source_spectrum=incident_spectrum,
        receiver_spectrum=total_spectrum,
        source_valid=np.asarray(valid, dtype=bool),
        source=incident,
        receiver=total,
        method="incident-reference",
        receiver_tail_relative_db=tail_relative_db(total.samples),
        tail_taper_fraction=float(tail_taper_fraction),
    )
    background = FrequencyResponse(
        frequency=frequency,
        response=np.ones(frequency.shape, dtype=np.complex128),
        source_spectrum=incident_spectrum,
        receiver_spectrum=incident_spectrum,
        source_valid=np.asarray(valid, dtype=bool),
        source=incident,
        receiver=incident,
        method="incident-reference",
        receiver_tail_relative_db=tail_relative_db(incident.samples),
        tail_taper_fraction=float(tail_taper_fraction),
    )
    return ChannelResponse(
        chirp=chirp,
        response=response,
        target=target,
        background=background,
        source_valid=np.asarray(valid, dtype=bool),
        normalisation="incident-reference",
    )


def interpolate_instrument_response(
    source_frequency: npt.ArrayLike,
    source_response: npt.ArrayLike,
    requested_frequency: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Interpolate a complex instrument transfer function in magnitude/phase."""

    return _interpolate_complex_response(
        source_frequency,
        source_response,
        requested_frequency,
        coordinate_name="instrument frequencies",
        coverage_name="requested FMCW band",
    )


def _interpolate_complex_response(
    source_coordinate,
    source_response,
    requested_coordinate,
    *,
    coordinate_name,
    coverage_name,
):
    """Interpolate finite complex data using magnitude and unwrapped phase."""

    frequency = np.asarray(source_coordinate, dtype=np.float64)
    response = np.asarray(source_response, dtype=np.complex128)
    requested = np.asarray(requested_coordinate, dtype=np.float64)
    if frequency.ndim != 1 or response.ndim != 1 or frequency.size != response.size:
        raise ValueError(f"{coordinate_name} and response must be equal-length 1D arrays")
    if frequency.size < 2 or np.any(np.diff(frequency) <= 0):
        raise ValueError(f"{coordinate_name} must be strictly increasing")
    if requested.ndim != 1 or requested.size == 0 or not np.all(np.isfinite(requested)):
        raise ValueError("requested coordinates must be a non-empty finite 1D array")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(response)):
        raise ValueError("source coordinates and response must be finite")
    tolerance = 64 * np.finfo(float).eps * frequency[-1]
    if requested.min() < frequency[0] - tolerance or requested.max() > frequency[-1] + tolerance:
        raise ValueError(f"response does not cover the {coverage_name}")
    magnitude = np.interp(requested, frequency, np.abs(response))
    phase = np.interp(requested, frequency, np.unwrap(np.angle(response)))
    return np.asarray(magnitude * np.exp(1j * phase), dtype=np.complex128)


def load_instrument_response(
    filename: str | Path,
    requested_frequency: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Load and interpolate a CSV instrument response.

    Supported headers are ``frequency_hz,real,imag`` and
    ``frequency_hz,magnitude,phase_deg``.
    """

    table = np.genfromtxt(filename, delimiter=",", names=True, dtype=np.float64)
    if table.dtype.names is None:
        raise ValueError("instrument CSV must contain a header row")
    names = {name.lower(): name for name in table.dtype.names}
    frequency_name = names.get("frequency_hz") or names.get("frequency")
    if frequency_name is None:
        raise ValueError("instrument CSV requires a frequency_hz column")
    if "real" in names and "imag" in names:
        response = table[names["real"]] + 1j * table[names["imag"]]
    elif "magnitude" in names and "phase_deg" in names:
        response = table[names["magnitude"]] * np.exp(1j * np.deg2rad(table[names["phase_deg"]]))
    else:
        raise ValueError("instrument CSV requires real/imag or magnitude/phase_deg columns")
    return interpolate_instrument_response(
        table[frequency_name],
        response,
        requested_frequency,
    )


def load_receiver_delay_response(
    filename: str | Path,
    requested_delay: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Load a delay/beat-dependent receiver response from CSV.

    Supported headers are ``delay_s,real,imag``,
    ``delay_s,magnitude,phase_deg``, and ``delay_s,gain``. This response can
    represent range gating or the IF/ADC response after converting beat
    frequency to propagation delay.
    """

    table = np.genfromtxt(filename, delimiter=",", names=True, dtype=np.float64)
    if table.dtype.names is None:
        raise ValueError("receiver-delay CSV must contain a header row")
    names = {name.lower(): name for name in table.dtype.names}
    delay_name = names.get("delay_s") or names.get("delay")
    if delay_name is None:
        raise ValueError("receiver-delay CSV requires a delay_s column")
    if "real" in names and "imag" in names:
        response = table[names["real"]] + 1j * table[names["imag"]]
    elif "magnitude" in names and "phase_deg" in names:
        response = table[names["magnitude"]] * np.exp(1j * np.deg2rad(table[names["phase_deg"]]))
    elif "gain" in names:
        response = np.asarray(table[names["gain"]], dtype=np.complex128)
    else:
        raise ValueError("receiver-delay CSV requires gain, real/imag, or magnitude/phase_deg columns")
    return _interpolate_complex_response(
        table[delay_name],
        response,
        requested_delay,
        coordinate_name="receiver delays",
        coverage_name="requested delay range",
    )


def _broadcast_frequency_vector(vector, target_ndim):
    return np.asarray(vector).reshape((-1,) + (1,) * (target_ndim - 1))


def reconstruct_fast_time(
    channel: ChannelResponse,
    *,
    instrument_response: npt.ArrayLike | None = None,
    receiver_delay_response: npt.ArrayLike | None = None,
    window: str = "blackman",
    gaussian_sigma: float = 0.2,
    normalise_window: bool = True,
    propagation_velocity: float | None = None,
    residual_video_phase: str = "neglect",
) -> FastTimeResponse:
    """Apply instrument/window corrections and reconstruct fast time.

    The default follows Eide et al. and neglects residual video phase (RVP).
    ``include`` applies the RVP that would be present after deramping a real
    linear chirp. This is useful for short, steep laboratory chirps; for
    ordinary subsurface FMCW sweeps it is normally indistinguishable.
    """

    if residual_video_phase not in {"neglect", "include"}:
        raise ValueError("residual_video_phase must be 'neglect' or 'include'")
    response = np.asarray(channel.response, dtype=np.complex128)
    if not np.all(channel.source_valid) or not np.all(np.isfinite(response)):
        raise ValueError("channel contains frequencies with an invalid source spectrum")
    count = channel.chirp.samples
    weights = spectral_window(
        window,
        count,
        gaussian_sigma=gaussian_sigma,
        normalise=normalise_window,
    )
    if instrument_response is None:
        instrument = np.ones(count, dtype=np.complex128)
    else:
        instrument = np.asarray(instrument_response, dtype=np.complex128)
        if instrument.shape != (count,) or not np.all(np.isfinite(instrument)):
            raise ValueError("instrument_response must be a finite vector with one value per sample")

    frequency_shape = (-1,) + (1,) * (response.ndim - 1)
    processed = response * instrument.reshape(frequency_shape) * weights.reshape(frequency_shape)
    envelope = np.asarray(np.fft.ifft(processed, axis=0), dtype=np.complex128)
    delay = channel.chirp.delay
    if residual_video_phase == "include":
        rvp = np.exp(1j * np.pi * channel.chirp.slope * delay**2)
        envelope = envelope * _broadcast_frequency_vector(rvp, envelope.ndim)

    if receiver_delay_response is None:
        delay_response = np.ones(count, dtype=np.complex128)
    else:
        delay_response = np.asarray(receiver_delay_response, dtype=np.complex128)
        if delay_response.shape != (count,) or not np.all(np.isfinite(delay_response)):
            raise ValueError("receiver_delay_response must be a finite vector with one value per sample")
        envelope = envelope * _broadcast_frequency_vector(delay_response, envelope.ndim)

    carrier = np.exp(2j * np.pi * channel.chirp.f_start * delay)
    complex_bandpass = envelope * _broadcast_frequency_vector(carrier, envelope.ndim)
    range_axis = None
    if propagation_velocity is not None:
        if not np.isfinite(propagation_velocity) or propagation_velocity <= 0:
            raise ValueError("propagation_velocity must be finite and positive")
        range_axis = 0.5 * float(propagation_velocity) * delay

    return FastTimeResponse(
        delay=delay,
        complex_envelope=np.asarray(envelope, dtype=np.complex128),
        complex_bandpass=np.asarray(complex_bandpass, dtype=np.complex128),
        real_bandpass=np.asarray(2 * complex_bandpass.real, dtype=np.float64),
        processed_spectrum=np.asarray(processed, dtype=np.complex128),
        instrument_response=instrument,
        receiver_delay_response=delay_response,
        weights=weights,
        window=str(window).lower(),
        range=range_axis,
        propagation_velocity=propagation_velocity,
        residual_video_phase=residual_video_phase,
    )


def synthesize_deramped_sweep(
    channel: ChannelResponse,
    *,
    instrument_response: npt.ArrayLike | None = None,
    residual_video_phase: str = "neglect",
) -> DerampedSweep:
    """Synthesize ideal complex stretch-receiver samples.

    The returned convention gives positive beat frequency for a delayed point
    response during an up-chirp. RVP can be included exactly on the discrete
    delay grid. It is omitted by default, as in the subsurface approximation
    used by Eide et al.
    """

    if residual_video_phase not in {"neglect", "include"}:
        raise ValueError("residual_video_phase must be 'neglect' or 'include'")
    response = np.asarray(channel.response, dtype=np.complex128)
    count = channel.chirp.samples
    if instrument_response is None:
        instrument = np.ones(count, dtype=np.complex128)
    else:
        instrument = np.asarray(instrument_response, dtype=np.complex128)
        if instrument.shape != (count,) or not np.all(np.isfinite(instrument)):
            raise ValueError("instrument_response must be a finite vector with one value per sample")
    corrected = response * _broadcast_frequency_vector(instrument, response.ndim)

    if residual_video_phase == "include":
        delay_response = np.fft.ifft(corrected, axis=0)
        rvp = np.exp(1j * np.pi * channel.chirp.slope * channel.chirp.delay**2)
        delay_response = delay_response * _broadcast_frequency_vector(rvp, response.ndim)
        ascending = np.conj(np.fft.fft(delay_response, axis=0))
    else:
        ascending = np.conj(corrected)
    signal = ascending if channel.chirp.direction == "up" else ascending[::-1, ...]

    beat = np.fft.fftfreq(count, d=channel.chirp.sweep_time / count)
    delay = beat / channel.chirp.slope
    return DerampedSweep(
        slow_time=channel.chirp.slow_time,
        instantaneous_frequency=channel.chirp.instantaneous_frequency,
        complex_signal=np.asarray(signal, dtype=np.complex128),
        in_phase=np.asarray(signal.real, dtype=np.float64),
        quadrature=np.asarray(signal.imag, dtype=np.float64),
        beat_frequency=np.asarray(beat, dtype=np.float64),
        delay=np.asarray(delay, dtype=np.float64),
        residual_video_phase=residual_video_phase,
    )


def write_fmcw_output(
    filename: str | Path,
    channel: ChannelResponse,
    fast_time: FastTimeResponse,
    deramped: DerampedSweep | None = None,
) -> Path:
    """Write processed FMCW products and provenance to HDF5."""

    path = Path(filename)
    chirp = channel.chirp
    with h5py.File(path, "w") as output:
        output.attrs["Format"] = "gprMax FMCW toolbox"
        output.attrs["EngineeringConvention"] = "Re{X exp(+j omega t)}"
        output.attrs["FStart"] = chirp.f_start
        output.attrs["FStop"] = chirp.f_stop
        output.attrs["Bandwidth"] = chirp.bandwidth
        output.attrs["SweepTime"] = chirp.sweep_time
        output.attrs["ChirpSlope"] = chirp.slope
        output.attrs["Direction"] = chirp.direction
        output.attrs["FrequencyEndpointIncluded"] = False
        output.attrs["BackgroundSubtracted"] = channel.background is not None
        output.attrs["Normalisation"] = channel.normalisation
        output.attrs["TargetFile"] = channel.target.receiver.filename
        output.attrs["TargetSourceFile"] = channel.target.source.filename
        output.attrs["SourcePath"] = channel.target.source.path
        output.attrs["ReceiverPath"] = channel.target.receiver.path
        if channel.background is not None:
            output.attrs["BackgroundFile"] = channel.background.receiver.filename
            output.attrs["BackgroundSourceFile"] = channel.background.source.filename

        frequency = output.create_dataset("frequency", data=chirp.frequency)
        frequency.attrs["Units"] = "Hz"
        output.create_dataset("channel_response", data=channel.response)
        output.create_dataset("source_valid", data=channel.source_valid.astype(np.uint8))
        target = output.create_group("target")
        target.create_dataset("response", data=channel.target.response)
        target.create_dataset("source_spectrum", data=channel.target.source_spectrum)
        target.create_dataset("receiver_spectrum", data=channel.target.receiver_spectrum)
        if channel.background is not None:
            background = output.create_group("background")
            background.create_dataset("response", data=channel.background.response)
            background.create_dataset("source_spectrum", data=channel.background.source_spectrum)
            background.create_dataset("receiver_spectrum", data=channel.background.receiver_spectrum)

        group = output.create_group("fast_time")
        group.attrs["Window"] = fast_time.window
        group.attrs["ResidualVideoPhase"] = fast_time.residual_video_phase
        delay = group.create_dataset("delay", data=fast_time.delay)
        delay.attrs["Units"] = "s"
        group.create_dataset("weights", data=fast_time.weights)
        group.create_dataset("instrument_response", data=fast_time.instrument_response)
        group.create_dataset("receiver_delay_response", data=fast_time.receiver_delay_response)
        group.create_dataset("processed_spectrum", data=fast_time.processed_spectrum)
        group.create_dataset("complex_envelope", data=fast_time.complex_envelope)
        group.create_dataset("complex_bandpass", data=fast_time.complex_bandpass)
        group.create_dataset("real_bandpass", data=fast_time.real_bandpass)
        if fast_time.range is not None:
            range_dataset = group.create_dataset("range", data=fast_time.range)
            range_dataset.attrs["Units"] = "m"
            range_dataset.attrs["PropagationVelocity"] = fast_time.propagation_velocity

        if deramped is not None:
            group = output.create_group("deramped_sweep")
            group.attrs["ResidualVideoPhase"] = deramped.residual_video_phase
            slow_time = group.create_dataset("slow_time", data=deramped.slow_time)
            slow_time.attrs["Units"] = "s"
            instantaneous = group.create_dataset("instantaneous_frequency", data=deramped.instantaneous_frequency)
            instantaneous.attrs["Units"] = "Hz"
            group.create_dataset("complex_signal", data=deramped.complex_signal)
            group.create_dataset("I", data=deramped.in_phase)
            group.create_dataset("Q", data=deramped.quadrature)
            beat = group.create_dataset("beat_frequency", data=deramped.beat_frequency)
            beat.attrs["Units"] = "Hz"
            mapped_delay = group.create_dataset("delay", data=deramped.delay)
            mapped_delay.attrs["Units"] = "s"
    return path
