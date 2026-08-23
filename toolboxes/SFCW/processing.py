# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Timing-aware stepped-frequency processing of gprMax impulse responses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from scipy.signal import czt, fftconvolve


@dataclass(frozen=True)
class SampledSignal:
    """A uniformly sampled signal with an explicit physical time origin.

    The first array dimension is time. Receiver samples may have a second
    dimension containing the traces of a merged B-scan.
    """

    path: str
    samples: npt.NDArray[np.floating]
    dt: float
    time_offset: float
    quantity: str = ""
    units: str = ""
    source_type: str = ""
    spatial_scale: float = 1.0
    filename: str = ""

    @property
    def times(self) -> npt.NDArray[np.float64]:
        return self.time_offset + self.dt * np.arange(self.samples.shape[0], dtype=np.float64)


@dataclass(frozen=True)
class FrequencyResponse:
    """Source-normalised complex response at requested SFCW frequencies."""

    frequency: npt.NDArray[np.float64]
    response: npt.NDArray[np.complex128]
    source_spectrum: npt.NDArray[np.complex128]
    receiver_spectrum: npt.NDArray[np.complex128]
    source_valid: npt.NDArray[np.bool_]
    source: SampledSignal
    receiver: SampledSignal
    method: str
    receiver_tail_relative_db: float
    tail_taper_fraction: float

    @property
    def i(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.response.real, dtype=np.float64)

    @property
    def q(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.response.imag, dtype=np.float64)


@dataclass(frozen=True)
class TimeResponse:
    """Windowed inverse stepped-frequency response."""

    time: npt.NDArray[np.float64]
    complex_envelope: npt.NDArray[np.complex128]
    complex_bandpass: npt.NDArray[np.complex128]
    real_bandpass: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    window: str
    zero_pad_factor: int
    time_shift: float


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalise_path(path: str) -> str:
    value = "/" + str(path).strip("/")
    return value if value != "/" else value


def list_sources(filename: str | Path) -> list[str]:
    """Return HDF5 groups containing a primary scalar excitation history."""

    paths: list[str] = []
    with h5py.File(filename, "r") as output:

        def visitor(name, item):
            if isinstance(item, h5py.Group) and "excitation" in item:
                excitation = item["excitation"]
                if isinstance(excitation, h5py.Group) and "samples" in excitation:
                    paths.append("/" + name.strip("/"))

        output.visititems(visitor)
    return sorted(paths)


def list_receivers(filename: str | Path) -> dict[str, tuple[str, ...]]:
    """Return receiver groups and their available field/current components."""

    result: dict[str, tuple[str, ...]] = {}
    allowed = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz"}
    with h5py.File(filename, "r") as output:

        def visitor(name, item):
            if not isinstance(item, h5py.Group) or "/rxs/rx" not in f"/{name}":
                return
            components = tuple(sorted(key for key in item.keys() if key in allowed))
            if components:
                result["/" + name.strip("/")] = components

        output.visititems(visitor)
    return dict(sorted(result.items()))


def _nearest_dt(item) -> float:
    current = item
    while current is not None:
        if "dt" in current.attrs:
            return float(current.attrs["dt"])
        if current.name == "/":
            break
        current = current.parent
    raise ValueError(f"no sample interval is available for {item.name}")


def load_source(filename: str | Path, source_path: str | None = None) -> SampledSignal:
    """Load one exact source excitation history from a gprMax output file."""

    available = list_sources(filename)
    if source_path is None:
        if len(available) != 1:
            raise ValueError(
                "source_path is required unless the output contains exactly one "
                f"scalar source excitation; available sources: {available}"
            )
        source_path = available[0]
    source_path = _normalise_path(source_path)
    with h5py.File(filename, "r") as output:
        if source_path not in output:
            raise ValueError(f"source group {source_path!r} is not present in {filename}")
        group = output[source_path]
        if "excitation/samples" not in group:
            raise ValueError(f"source group {source_path!r} has no scalar excitation samples")
        excitation = group["excitation"]
        samples = np.asarray(excitation["samples"], dtype=np.float64)
        dt = float(excitation.attrs.get("SampleInterval", _nearest_dt(group)))
        return SampledSignal(
            path=source_path,
            samples=samples,
            dt=dt,
            time_offset=float(excitation.attrs["TimeSampleOffset"]),
            quantity=_text(excitation.attrs.get("DrivingQuantity", "")),
            units=_text(excitation.attrs.get("Units", "")),
            source_type=_text(excitation.attrs.get("SourceType", group.attrs.get("Type", ""))),
            spatial_scale=float(excitation.attrs.get("SpatialScale", 1.0)),
            filename=str(Path(filename)),
        )


def load_receiver(
    filename: str | Path,
    receiver_path: str | None = None,
    component: str | None = None,
) -> SampledSignal:
    """Load one receiver history and its Yee-time convention."""

    available = list_receivers(filename)
    if receiver_path is None:
        if len(available) != 1:
            raise ValueError(
                "receiver_path is required unless the output contains exactly one receiver; "
                f"available receivers: {list(available)}"
            )
        receiver_path = next(iter(available))
    receiver_path = _normalise_path(receiver_path)
    if receiver_path not in available:
        raise ValueError(f"receiver group {receiver_path!r} is not present in {filename}")
    components = available[receiver_path]
    if component is None:
        if len(components) != 1:
            raise ValueError(
                "component is required unless the selected receiver contains exactly one output; "
                f"available components: {components}"
            )
        component = components[0]
    if component not in components:
        raise ValueError(
            f"receiver {receiver_path!r} has no {component!r} output; "
            f"available components: {components}"
        )

    with h5py.File(filename, "r") as output:
        dataset = output[f"{receiver_path}/{component}"]
        dt = float(dataset.attrs.get("SampleInterval", _nearest_dt(dataset.parent)))
        inferred_offset = 0.0 if component.startswith("E") else -0.5 * dt
        return SampledSignal(
            path=f"{receiver_path}/{component}",
            samples=np.asarray(dataset, dtype=np.float64),
            dt=dt,
            time_offset=float(dataset.attrs.get("TimeSampleOffset", inferred_offset)),
            quantity=component,
            units="V/m"
            if component.startswith("E")
            else "A/m"
            if component.startswith("H")
            else "A",
            filename=str(Path(filename)),
        )


def _validate_spectrum_inputs(samples, dt, frequencies, time_offset):
    values = np.asarray(samples, dtype=np.float64)
    requested = np.asarray(frequencies, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] == 0:
        raise ValueError("samples must be a non-empty time history or time-by-trace array")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(requested)) or np.any(requested < 0):
        raise ValueError("frequencies must be finite and non-negative")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(time_offset):
        raise ValueError("time_offset must be finite")
    nyquist = 1 / (2 * dt)
    tolerance = 64 * np.finfo(np.float64).eps * nyquist
    if np.any(requested > nyquist + tolerance):
        raise ValueError(f"requested frequencies exceed the FDTD Nyquist frequency {nyquist:g} Hz")
    return values, requested


def engineering_dft(
    samples: npt.ArrayLike,
    dt: float,
    frequencies: npt.ArrayLike,
    *,
    time_offset: float = 0.0,
    block_size: int = 128,
) -> npt.NDArray[np.complex128]:
    """Evaluate a sampled signal at arbitrary frequencies using ``exp(-jwt)``.

    Uniform stepped-frequency requests use a chirp-z transform. Arbitrary
    requests fall back to bounded-memory direct evaluation.
    """

    values, requested = _validate_spectrum_inputs(samples, dt, frequencies, time_offset)
    sample_count = values.shape[0]
    if requested.size == 1:
        phase = np.exp(-2j * np.pi * requested[0] * dt * np.arange(sample_count))
        raw = np.asarray((phase @ values)[None, ...], dtype=np.complex128)
    else:
        steps = np.diff(requested)
        uniform = np.allclose(
            steps,
            steps[0],
            rtol=256 * np.finfo(np.float64).eps,
            atol=256 * np.finfo(np.float64).eps * max(1.0, abs(steps[0])),
        )
        if uniform:
            a = np.exp(2j * np.pi * requested[0] * dt)
            w = np.exp(-2j * np.pi * steps[0] * dt)
            raw = np.asarray(czt(values, m=requested.size, w=w, a=a, axis=0), dtype=np.complex128)
        else:
            if not isinstance(block_size, int) or block_size <= 0:
                raise ValueError("block_size must be a positive integer")
            indices = np.arange(sample_count, dtype=np.float64)
            raw = np.empty((requested.size,) + values.shape[1:], dtype=np.complex128)
            for start in range(0, requested.size, block_size):
                stop = min(start + block_size, requested.size)
                phase = np.exp(-2j * np.pi * requested[start:stop, None] * dt * indices)
                raw[start:stop] = phase @ values
    origin_phase = np.exp(-2j * np.pi * requested * float(time_offset)).reshape(
        (-1,) + (1,) * (raw.ndim - 1)
    )
    return np.asarray(float(dt) * origin_phase * raw, dtype=np.complex128)


def direct_frequency_response(
    source: SampledSignal,
    receiver: SampledSignal,
    frequencies: npt.ArrayLike,
    *,
    source_floor_db: float = -100.0,
    tail_taper_fraction: float = 0.0,
) -> FrequencyResponse:
    """Calculate a timing-correct source-normalised receiver response."""

    if source.samples.ndim != 1:
        raise ValueError("the source excitation must be a one-dimensional time history")
    if not np.isclose(source.dt, receiver.dt, rtol=0, atol=32 * np.finfo(float).eps * source.dt):
        raise ValueError("source and receiver sample intervals are different")
    requested = np.asarray(frequencies, dtype=np.float64)
    tail_db = tail_relative_db(receiver.samples)
    receiver_samples = apply_tail_taper(receiver.samples, tail_taper_fraction)
    source_spectrum = engineering_dft(
        source.samples, source.dt, requested, time_offset=source.time_offset
    )
    receiver_spectrum = engineering_dft(
        receiver_samples, receiver.dt, requested, time_offset=receiver.time_offset
    )
    peak = float(np.max(np.abs(source_spectrum), initial=0.0))
    if peak == 0:
        raise ValueError("the selected source excitation is identically zero")
    threshold = peak * 10 ** (float(source_floor_db) / 20)
    valid = np.asarray(np.abs(source_spectrum) > threshold, dtype=bool)
    response = np.full(receiver_spectrum.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    source_divisor = source_spectrum.reshape((-1,) + (1,) * (receiver_spectrum.ndim - 1))
    valid_divisor = valid.reshape((-1,) + (1,) * (receiver_spectrum.ndim - 1))
    np.divide(receiver_spectrum, source_divisor, out=response, where=valid_divisor)
    return FrequencyResponse(
        frequency=requested,
        response=response,
        source_spectrum=source_spectrum,
        receiver_spectrum=receiver_spectrum,
        source_valid=valid,
        source=source,
        receiver=receiver,
        method="direct",
        receiver_tail_relative_db=tail_db,
        tail_taper_fraction=float(tail_taper_fraction),
    )


def _single_impulse(samples) -> tuple[int, float]:
    values = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(values), initial=0.0))
    if peak == 0:
        raise ValueError("the selected source excitation is identically zero")
    tolerance = 64 * np.finfo(np.float64).eps * peak
    nonzero = np.flatnonzero(np.abs(values) > tolerance)
    if nonzero.size != 1:
        raise ValueError(
            "homodyne processing requires a one-sample impulse excitation; "
            f"the selected source has {nonzero.size} significant samples"
        )
    index = int(nonzero[0])
    return index, float(values[index])


def homodyne_frequency_response(
    source: SampledSignal,
    receiver: SampledSignal,
    frequencies: npt.ArrayLike,
    *,
    cycles: int = 8,
    tail_taper_fraction: float = 0.0,
) -> FrequencyResponse:
    """Reproduce tone convolution and ideal quadrature homodyne detection.

    A least-squares DC extraction removes the small finite-record cross term
    between the sampled cosine and quadrature references. It is equivalent to
    an ideal low-pass homodyne detector over the selected steady-state record.
    """

    if not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("cycles must be a positive integer")
    if source.samples.ndim != 1:
        raise ValueError("the source excitation must be a one-dimensional time history")
    if not np.isclose(source.dt, receiver.dt, rtol=0, atol=32 * np.finfo(float).eps * source.dt):
        raise ValueError("source and receiver sample intervals are different")
    if receiver.samples.ndim != 1:
        raise ValueError("homodyne processing currently accepts one A-scan at a time")
    requested = np.asarray(frequencies, dtype=np.float64)
    _validate_spectrum_inputs(receiver.samples, receiver.dt, requested, receiver.time_offset)
    if np.any(requested <= 0):
        raise ValueError("homodyne processing requires strictly positive frequencies")

    impulse_index, impulse_amplitude = _single_impulse(source.samples)
    tail_db = tail_relative_db(receiver.samples)
    receiver_samples = apply_tail_taper(receiver.samples, tail_taper_fraction)
    kernel = np.asarray(receiver_samples[impulse_index:] / impulse_amplitude, dtype=np.float64)
    if kernel.size == 0:
        raise ValueError("receiver history ends before the source impulse")
    source_origin = source.time_offset + impulse_index * source.dt
    output_origin = receiver.time_offset + impulse_index * receiver.dt
    response = np.empty(requested.size, dtype=np.complex128)

    for index, frequency in enumerate(requested):
        measurement_samples = max(16, int(np.ceil(cycles / (frequency * source.dt))))
        tone_samples = kernel.size + measurement_samples
        source_times = source_origin + source.dt * np.arange(tone_samples, dtype=np.float64)
        tone = np.cos(2 * np.pi * frequency * source_times)
        received = fftconvolve(kernel, tone, mode="full")
        start = kernel.size - 1
        steady = np.asarray(received[start : start + measurement_samples], dtype=np.float64)
        output_times = output_origin + receiver.dt * np.arange(
            start, start + measurement_samples, dtype=np.float64
        )
        references = np.column_stack(
            (
                np.cos(2 * np.pi * frequency * output_times),
                -np.sin(2 * np.pi * frequency * output_times),
            )
        )
        in_phase, quadrature = np.linalg.lstsq(references, steady, rcond=None)[0]
        response[index] = in_phase + 1j * quadrature

    source_spectrum = engineering_dft(
        source.samples, source.dt, requested, time_offset=source.time_offset
    )
    receiver_spectrum = engineering_dft(
        receiver_samples, receiver.dt, requested, time_offset=receiver.time_offset
    )
    return FrequencyResponse(
        frequency=requested,
        response=response,
        source_spectrum=source_spectrum,
        receiver_spectrum=receiver_spectrum,
        source_valid=np.ones(requested.shape, dtype=bool),
        source=source,
        receiver=receiver,
        method="homodyne",
        receiver_tail_relative_db=tail_db,
        tail_taper_fraction=float(tail_taper_fraction),
    )


def tail_relative_db(samples: npt.ArrayLike, fraction: float = 0.05) -> float:
    """Return the peak magnitude in the record tail relative to the full peak."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] == 0:
        raise ValueError("samples must be a non-empty time history or time-by-trace array")
    if not np.isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise ValueError("fraction must lie in the interval (0, 1]")
    peak = float(np.max(np.abs(values), initial=0.0))
    if peak == 0:
        return float("-inf")
    count = min(values.shape[0], max(8, int(np.ceil(fraction * values.shape[0]))))
    tail = float(np.max(np.abs(values[-count:, ...]), initial=0.0))
    if tail == 0:
        return float("-inf")
    return float(20 * np.log10(tail / peak))


def apply_tail_taper(samples: npt.ArrayLike, fraction: float) -> npt.NDArray[np.float64]:
    """Apply a raised-cosine taper to the end of a finite response."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] == 0:
        raise ValueError("samples must be a non-empty time history or time-by-trace array")
    if not np.isfinite(fraction) or fraction < 0 or fraction > 1:
        raise ValueError("tail_taper_fraction must lie in the interval [0, 1]")
    result = values.copy()
    if fraction == 0:
        return result
    count = min(values.shape[0], max(2, int(np.ceil(fraction * values.shape[0]))))
    coordinate = np.linspace(0, np.pi, count, dtype=np.float64)
    taper = (0.5 * (1 + np.cos(coordinate))).reshape((-1,) + (1,) * (values.ndim - 1))
    result[-count:, ...] *= taper
    return result


def spectral_window(
    name: str,
    count: int,
    *,
    gaussian_sigma: float = 0.2,
    normalise: bool = True,
) -> npt.NDArray[np.float64]:
    """Return a stepped-frequency weighting window."""

    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    key = str(name).lower().replace("-", "")
    if key in {"rectangular", "rect", "boxcar", "none"}:
        weights = np.ones(count, dtype=np.float64)
    elif key in {"hann", "hanning"}:
        weights = np.hanning(count)
    elif key == "hamming":
        weights = np.hamming(count)
    elif key == "blackman":
        weights = np.blackman(count)
    elif key in {"gaussian", "gauss"}:
        if not np.isfinite(gaussian_sigma) or gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be finite and positive")
        coordinate = np.linspace(-0.5, 0.5, count, dtype=np.float64)
        weights = np.exp(-0.5 * (coordinate / gaussian_sigma) ** 2)
    else:
        raise ValueError("window must be rectangular, hann, hamming, blackman, or gaussian")
    if normalise:
        mean = float(np.mean(weights))
        if mean > 0:
            weights = weights / mean
    return np.asarray(weights, dtype=np.float64)


def reconstruct_time_response(
    frequency_response: FrequencyResponse,
    *,
    window: str = "gaussian",
    gaussian_sigma: float = 0.2,
    zero_pad_factor: int = 1,
    normalise_window: bool = True,
    time_shift: float = 0.0,
) -> TimeResponse:
    """Inverse-transform uniformly stepped complex data into a real response."""

    frequencies = np.asarray(frequency_response.frequency, dtype=np.float64)
    response = np.asarray(frequency_response.response, dtype=np.complex128)
    if frequencies.size < 2:
        raise ValueError("at least two stepped frequencies are required for reconstruction")
    steps = np.diff(frequencies)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-10, atol=0):
        raise ValueError("time reconstruction requires increasing, uniformly spaced frequencies")
    if not np.all(np.isfinite(response)):
        raise ValueError("frequency response contains invalid values")
    if not isinstance(zero_pad_factor, int) or zero_pad_factor <= 0:
        raise ValueError("zero_pad_factor must be a positive integer")
    if not np.isfinite(time_shift):
        raise ValueError("time_shift must be finite")

    weights = spectral_window(
        window,
        frequencies.size,
        gaussian_sigma=gaussian_sigma,
        normalise=normalise_window,
    )
    count = frequencies.size * zero_pad_factor
    padded = np.zeros((count,) + response.shape[1:], dtype=np.complex128)
    weight_shape = (-1,) + (1,) * (response.ndim - 1)
    shift_phase = np.exp(-2j * np.pi * frequencies * float(time_shift)).reshape(weight_shape)
    padded[: frequencies.size, ...] = response * weights.reshape(weight_shape) * shift_phase
    df = float(steps[0])
    time = np.arange(count, dtype=np.float64) / (count * df)
    # Follow the discrete inverse FFT used by an SFCW instrument. Scaling by
    # the zero-padding factor preserves amplitude when extra spectral zeros
    # are added solely to interpolate the displayed time samples.
    complex_envelope = np.asarray(
        np.fft.ifft(padded, axis=0) * zero_pad_factor,
        dtype=np.complex128,
    )
    carrier = np.exp(2j * np.pi * frequencies[0] * time).reshape((-1,) + (1,) * (response.ndim - 1))
    complex_bandpass = np.asarray(complex_envelope * carrier, dtype=np.complex128)
    real_bandpass = np.asarray(2 * complex_bandpass.real, dtype=np.float64)
    return TimeResponse(
        time=time,
        complex_envelope=complex_envelope,
        complex_bandpass=complex_bandpass,
        real_bandpass=real_bandpass,
        weights=weights,
        window=str(window).lower(),
        zero_pad_factor=zero_pad_factor,
        time_shift=float(time_shift),
    )


def process_output(
    filename: str | Path,
    frequencies: npt.ArrayLike,
    *,
    source_filename: str | Path | None = None,
    source_path: str | None = None,
    receiver_path: str | None = None,
    component: str | None = None,
    method: str = "direct",
    source_floor_db: float = -100.0,
    homodyne_cycles: int = 8,
    tail_taper_fraction: float = 0.0,
) -> FrequencyResponse:
    """Load a gprMax output and calculate one SFCW frequency response."""

    source = load_source(source_filename or filename, source_path)
    receiver = load_receiver(filename, receiver_path, component)
    key = str(method).lower()
    if key == "direct":
        return direct_frequency_response(
            source,
            receiver,
            frequencies,
            source_floor_db=source_floor_db,
            tail_taper_fraction=tail_taper_fraction,
        )
    if key == "homodyne":
        return homodyne_frequency_response(
            source,
            receiver,
            frequencies,
            cycles=homodyne_cycles,
            tail_taper_fraction=tail_taper_fraction,
        )
    raise ValueError("method must be 'direct' or 'homodyne'")


def write_sfcw_output(
    filename: str | Path,
    frequency_response: FrequencyResponse,
    time_response: TimeResponse | None = None,
) -> Path:
    """Write processed complex stepped-frequency data to an HDF5 file."""

    path = Path(filename)
    with h5py.File(path, "w") as output:
        output.attrs["Format"] = "gprMax SFCW toolbox"
        output.attrs["Method"] = frequency_response.method
        output.attrs["SourcePath"] = frequency_response.source.path
        output.attrs["ReceiverPath"] = frequency_response.receiver.path
        output.attrs["SourceFile"] = frequency_response.source.filename
        output.attrs["ReceiverFile"] = frequency_response.receiver.filename
        output.attrs["SourceTimeSampleOffset"] = frequency_response.source.time_offset
        output.attrs["ReceiverTimeSampleOffset"] = frequency_response.receiver.time_offset
        output.attrs["SourceQuantity"] = frequency_response.source.quantity
        output.attrs["SourceUnits"] = frequency_response.source.units
        output.attrs["SourceSpatialScale"] = frequency_response.source.spatial_scale
        output.attrs["ReceiverQuantity"] = frequency_response.receiver.quantity
        output.attrs["ReceiverUnits"] = frequency_response.receiver.units
        output.attrs["SampleInterval"] = frequency_response.source.dt
        output.attrs["EngineeringConvention"] = "Re{X exp(+j omega t)}"
        output.attrs["ReceiverTailRelativeDB"] = frequency_response.receiver_tail_relative_db
        output.attrs["TailTaperFraction"] = frequency_response.tail_taper_fraction
        frequency = output.create_dataset("frequency", data=frequency_response.frequency)
        frequency.attrs["Units"] = "Hz"
        response = output.create_dataset("response", data=frequency_response.response)
        response.attrs["Quantity"] = "source-normalised receiver response"
        in_phase = output.create_dataset("I", data=frequency_response.i)
        quadrature = output.create_dataset("Q", data=frequency_response.q)
        in_phase.attrs["MixerNormalisation"] = 2.0
        quadrature.attrs["MixerNormalisation"] = 2.0
        output.create_dataset("source_spectrum", data=frequency_response.source_spectrum)
        output.create_dataset("receiver_spectrum", data=frequency_response.receiver_spectrum)
        output.create_dataset(
            "source_valid", data=np.asarray(frequency_response.source_valid, dtype=np.uint8)
        )
        if time_response is not None:
            group = output.create_group("time_response")
            group.attrs["Window"] = time_response.window
            group.attrs["ZeroPadFactor"] = time_response.zero_pad_factor
            group.attrs["TimeShift"] = time_response.time_shift
            time = group.create_dataset("time", data=time_response.time)
            time.attrs["Units"] = "s"
            group.create_dataset("weights", data=time_response.weights)
            group.create_dataset("complex_envelope", data=time_response.complex_envelope)
            group.create_dataset("complex_bandpass", data=time_response.complex_bandpass)
            group.create_dataset("real_bandpass", data=time_response.real_bandpass)
    return path
