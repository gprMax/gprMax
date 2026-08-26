# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Direct time-domain Green responses for lossless planar layers.

The construction follows the transmission-line interpretation in Chapter 2
of I. R. Capoglu's 2007 PhD thesis.  For real, frequency-independent layer
properties, every TE/TM reflection and transmission coefficient is constant
and propagation contributes only a delay.  The four scalar Green responses
used by the layered equivalent-current transform can therefore be represented
as sparse trains of weighted Dirac impulses.

The small impulse-train engine is kept independent of the streaming monitor
so it can be verified directly against the established frequency-domain
layered kernel before it is coupled to the Yee-surface sampler below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from heapq import heappop, heappush
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
from scipy.constants import c, epsilon_0, mu_0

from .equivalent_current_time import EquivalentCurrentTimeMonitor, EquivalentCurrentTimeResult
from .layered import AXIS_BASES, LayeredMedium, LayeredTermination, _observation_indices
from .time_domain import TERMINAL_DECAY_THRESHOLD, TERMINAL_DECAY_WINDOW_SAMPLES

logger = logging.getLogger(__name__)

try:
    from gprMax.cython.ntff import deposit_layered_impulse_time as _deposit_layered_impulse_time
except ImportError:  # Source-tree use before extensions are rebuilt.
    _deposit_layered_impulse_time = None


@dataclass(frozen=True)
class ImpulseTrain:
    """A real causal or reduced-time impulse train.

    ``delays`` are measured relative to the layered NTFF origin.  They may be
    negative because the far-zone response is range normalised to that origin;
    physical surface-current causality is recovered after the lateral and Yee
    time offsets are included.
    """

    delays: npt.NDArray[np.floating]
    amplitudes: npt.NDArray[np.floating]
    discarded_path_amplitude_sum: float = 0.0

    def __post_init__(self) -> None:
        delays = np.ascontiguousarray(self.delays, dtype=float)
        amplitudes = np.ascontiguousarray(self.amplitudes, dtype=float)
        if delays.ndim != 1 or amplitudes.shape != delays.shape:
            raise ValueError("impulse-train delays and amplitudes must be matching vectors")
        if not np.all(np.isfinite(delays)) or not np.all(np.isfinite(amplitudes)):
            raise ValueError("impulse-train delays and amplitudes must be finite")
        if not np.isfinite(self.discarded_path_amplitude_sum) or self.discarded_path_amplitude_sum < 0:
            raise ValueError("discarded path amplitude sum must be finite and non-negative")
        delays.setflags(write=False)
        amplitudes.setflags(write=False)
        object.__setattr__(self, "delays", delays)
        object.__setattr__(self, "amplitudes", amplitudes)

    def frequency_response(self, frequencies: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        """Evaluate the train using gprMax's ``exp(+j omega t)`` convention."""

        values = np.asarray(frequencies, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("frequencies must be finite")
        angular_frequency = 2 * np.pi * values
        return np.exp(-1j * angular_frequency[..., np.newaxis] * self.delays) @ self.amplitudes


@dataclass(frozen=True)
class LayeredImpulseResponses:
    """The four TE/TM transmission-line responses at one source depth."""

    vi_e: ImpulseTrain
    vv_e: ImpulseTrain
    vi_h: ImpulseTrain
    vv_h: ImpulseTrain
    observation_impedance: float
    observation_wave_speed: float
    observation_material_index: int


@dataclass(frozen=True)
class _PreparedImpulseTrain:
    integer_delay: npt.NDArray[np.int64]
    fraction: npt.NDArray[np.floating]
    amplitudes: npt.NDArray[np.floating]


def _readonly(values, dtype=None):
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _validate_lossless_stack(
    interfaces: npt.ArrayLike,
    relative_permittivity: npt.ArrayLike,
    relative_permeability: npt.ArrayLike,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    interfaces = np.asarray(interfaces, dtype=float)
    eps = np.asarray(relative_permittivity)
    mu = np.asarray(relative_permeability)
    if interfaces.ndim != 1:
        raise ValueError("layered interfaces must be a vector")
    if eps.ndim != 1 or mu.shape != eps.shape or eps.size != interfaces.size + 1:
        raise ValueError("layered interfaces and constitutive vectors are inconsistent")
    if interfaces.size and (not np.all(np.isfinite(interfaces)) or not np.all(np.diff(interfaces) < 0)):
        raise ValueError("layered interfaces must be finite and strictly descending")
    if np.any(np.abs(np.imag(eps)) > 1e-13) or np.any(np.abs(np.imag(mu)) > 1e-13):
        raise ValueError("direct time-domain layered NTFF requires lossless materials")
    eps = np.asarray(np.real(eps), dtype=float)
    mu = np.asarray(np.real(mu), dtype=float)
    if not np.all(np.isfinite(eps)) or not np.all(np.isfinite(mu)):
        raise ValueError("layered material properties must be finite")
    if np.any(eps <= 0) or np.any(mu <= 0):
        raise ValueError("layered material properties must be positive")
    return interfaces, eps, mu


def _coalesce_impulses(
    impulses: list[tuple[float, float]], amplitude_tolerance: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    if not impulses:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    impulses.sort(key=lambda item: item[0])
    delays = []
    amplitudes = []
    for delay, amplitude in impulses:
        if delays:
            scale = max(abs(delay), abs(delays[-1]), np.finfo(float).tiny)
            if abs(delay - delays[-1]) <= 128 * np.finfo(float).eps * scale:
                amplitudes[-1] += amplitude
                continue
        delays.append(delay)
        amplitudes.append(amplitude)
    delays_array = np.asarray(delays, dtype=float)
    amplitudes_array = np.asarray(amplitudes, dtype=float)
    keep = np.abs(amplitudes_array) > amplitude_tolerance
    return delays_array[keep], amplitudes_array[keep]


def _line_response_train(
    interfaces: npt.NDArray[np.floating],
    line_impedance: npt.NDArray[np.floating],
    axial_slowness: npt.NDArray[np.floating],
    source_layer: int,
    source_position: float,
    *,
    upper_observation: bool,
    response: str,
    relative_tolerance: float,
    max_impulses: int,
    maximum_delay: float | None,
    termination: LayeredTermination | None,
) -> ImpulseTrain:
    """Enumerate the multiple-reflection paths for one scalar TL response."""

    if response not in ("vi", "vv"):
        raise ValueError("response must be 'vi' or 'vv'")
    nlayers = line_impedance.size
    exterior = 0 if upper_observation else nlayers - 1
    outward_direction = -1 if upper_observation else 1
    initial_scale = float(line_impedance[source_layer]) if response == "vi" else 1.0
    amplitude_cutoff = relative_tolerance * max(abs(initial_scale), 1.0)

    # Multiple reflection histories can arrive at the same interface, in the
    # same direction, at mathematically identical times. Expanding each one as
    # a separate binary-tree node becomes exponentially expensive and also
    # misses their cancellation. Coalesce such events before scattering them.
    # The quantisation is only a few hundred floating-point ulps of the largest
    # one-way stack delay/coordinate and is many orders below an FDTD time or
    # spatial step.
    coordinate_scale = max(
        abs(float(source_position)),
        float(np.max(np.abs(interfaces), initial=0.0)),
        1e-12,
    )
    delay_scale = max(
        coordinate_scale * float(np.max(axial_slowness, initial=0.0)),
        1e-18,
    )
    position_quantum = 512 * np.finfo(float).eps * coordinate_scale
    delay_quantum = 512 * np.finfo(float).eps * delay_scale
    queue: list[tuple[int, int, tuple[int, int, int, int]]] = []
    pending: dict[tuple[int, int, int, int], list[float]] = {}
    serial = 0

    def push_event(delay, amplitude, position, layer, direction):
        nonlocal serial
        delay_tick = int(np.rint(delay / delay_quantum))
        position_tick = int(np.rint(position / position_quantum))
        key = (delay_tick, position_tick, int(layer), int(direction))
        if key in pending:
            pending[key][1] += amplitude
            return
        pending[key] = [float(delay), float(amplitude), float(position)]
        heappush(queue, (delay_tick, serial, key))
        serial += 1

    for direction in (-1, 1):
        amplitude = initial_scale
        if response == "vv" and direction == 1:
            amplitude = -amplitude
        push_event(0.0, amplitude, float(source_position), source_layer, direction)

    completed: list[tuple[float, float]] = []
    omitted = 0.0
    processed = 0
    while queue:
        _, _, key = heappop(queue)
        delay, amplitude, position = pending.pop(key)
        _, _, layer, direction = key
        processed += 1
        if processed > max_impulses:
            raise RuntimeError(
                "layered impulse train exceeded max_impulses; increase the limit, "
                "relax impulse_tolerance, or avoid observation close to grazing"
            )
        if abs(amplitude) <= amplitude_cutoff:
            omitted += abs(amplitude)
            continue

        if layer == exterior and direction == outward_direction:
            reference_delay = (
                -axial_slowness[layer] * position if upper_observation else axial_slowness[layer] * position
            )
            total_delay = delay + reference_delay
            if maximum_delay is None or total_delay <= maximum_delay:
                completed.append((total_delay, amplitude))
            continue

        if direction == -1:
            if layer == 0:
                if termination is None or termination.side != "positive":
                    # This is the non-observation exterior. The wave escapes
                    # and cannot return in an open planar stack.
                    continue
                boundary = float(termination.position)
                arrival_delay = delay + abs(position - boundary) * axial_slowness[layer]
                if maximum_delay is None or arrival_delay <= maximum_delay:
                    push_event(arrival_delay, -amplitude, boundary, layer, 1)
                continue
            boundary = float(interfaces[layer - 1])
            next_layer = layer - 1
        else:
            if layer == nlayers - 1:
                if termination is None or termination.side != "negative":
                    continue
                boundary = float(termination.position)
                arrival_delay = delay + abs(position - boundary) * axial_slowness[layer]
                if maximum_delay is None or arrival_delay <= maximum_delay:
                    push_event(arrival_delay, -amplitude, boundary, layer, -1)
                continue
            boundary = float(interfaces[layer])
            next_layer = layer + 1

        arrival_delay = delay + abs(position - boundary) * axial_slowness[layer]
        if maximum_delay is not None and arrival_delay > maximum_delay:
            continue
        z_here = float(line_impedance[layer])
        z_next = float(line_impedance[next_layer])
        denominator = z_here + z_next
        if abs(denominator) <= 128 * np.finfo(float).eps * max(z_here, z_next, 1.0):
            raise FloatingPointError("singular lossless layered-interface impedance")
        reflection = (z_next - z_here) / denominator
        transmission = 2 * z_next / denominator

        reflected = amplitude * reflection
        transmitted = amplitude * transmission
        if abs(reflected) > amplitude_cutoff:
            push_event(arrival_delay, reflected, boundary, layer, -direction)
        else:
            omitted += abs(reflected)
        if abs(transmitted) > amplitude_cutoff:
            push_event(arrival_delay, transmitted, boundary, next_layer, direction)
        else:
            omitted += abs(transmitted)

    delays, amplitudes = _coalesce_impulses(completed, amplitude_cutoff)
    return ImpulseTrain(delays, amplitudes, omitted)


def build_layered_impulse_responses(
    interfaces: npt.ArrayLike,
    relative_permittivity: npt.ArrayLike,
    relative_permeability: npt.ArrayLike,
    local_direction: npt.ArrayLike,
    source_position: float,
    *,
    source_layer: int | None = None,
    impulse_tolerance: float = 1e-10,
    max_impulses: int = 100_000,
    maximum_delay: float | None = None,
    grazing_tolerance: float = 1e-8,
    termination: LayeredTermination | None = None,
) -> LayeredImpulseResponses:
    """Build the four scalar lossless layered Green impulse responses.

    Args:
        interfaces: Strictly descending interface coordinates relative to the
            NTFF origin.
        relative_permittivity: Real, frequency-independent relative values.
        relative_permeability: Real, frequency-independent relative values.
        local_direction: Unit observation direction in the stack's ``u,v,n``
            basis.
        source_position: Source/surface-patch coordinate along ``n``.
        source_layer: Optional precomputed layer index.
        impulse_tolerance: Relative per-path truncation threshold.
        max_impulses: Safety bound on processed path states.
        maximum_delay: Optional reduced-time cutoff.
        grazing_tolerance: Minimum absolute observation normal cosine.
    """

    interfaces, eps_absolute, mu_absolute = _validate_lossless_stack(
        interfaces, relative_permittivity, relative_permeability
    )
    if termination is not None:
        termination.validate(interfaces)
    # Callers may retain immutable direction arrays as monitor metadata.  The
    # normalisation below must never try to mutate that public input.
    direction = np.array(local_direction, dtype=float, copy=True)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("local_direction must contain three finite values")
    direction_norm = float(np.linalg.norm(direction))
    if not np.isclose(direction_norm, 1, rtol=1e-6, atol=1e-7):
        raise ValueError("local_direction must be a unit vector")
    direction /= direction_norm
    if abs(float(direction[2])) <= grazing_tolerance:
        raise ValueError("direct time-domain layered NTFF is singular at grazing incidence")
    if not np.isfinite(source_position):
        raise ValueError("source_position must be finite")
    if not np.isfinite(impulse_tolerance) or not 0 < impulse_tolerance < 1:
        raise ValueError("impulse_tolerance must lie strictly between zero and one")
    if not isinstance(max_impulses, (int, np.integer)) or max_impulses < 2:
        raise ValueError("max_impulses must be an integer of at least two")
    if maximum_delay is not None and not np.isfinite(maximum_delay):
        raise ValueError("maximum_delay must be finite when supplied")

    if source_layer is None:
        source_layer = int(np.searchsorted(-interfaces, -source_position, side="left"))
    if not isinstance(source_layer, (int, np.integer)) or not 0 <= source_layer < eps_absolute.size:
        raise ValueError("source_layer is outside the layered stack")

    upper_observation = bool(direction[2] > 0)
    if termination is not None:
        if upper_observation and termination.side == "positive":
            raise ValueError("far-field observation cannot point through a positive-axis PEC")
        if not upper_observation and termination.side == "negative":
            raise ValueError("far-field observation cannot point through a negative-axis PEC")
        beyond_termination = (
            source_position > termination.position
            if termination.side == "positive"
            else source_position < termination.position
        )
        if beyond_termination:
            raise ValueError("source position cannot lie beyond the PEC termination")
    exterior = 0 if upper_observation else eps_absolute.size - 1
    eps_observation = float(eps_absolute[exterior])
    mu_observation = float(mu_absolute[exterior])
    wave_speed = c / np.sqrt(eps_observation * mu_observation)
    observation_impedance = np.sqrt(mu_0 * mu_observation / (epsilon_0 * eps_observation))
    eps = eps_absolute / eps_observation
    mu = mu_absolute / mu_observation
    sin_theta = float(np.hypot(direction[0], direction[1]))
    q_squared = eps * mu - sin_theta**2
    if np.any(q_squared <= 0):
        raise ValueError(
            "direct time-domain layered NTFF requires propagating TE/TM waves in every layer; "
            "use the frequency-domain transform for evanescent or total-internal-reflection cases"
        )
    q = np.sqrt(q_squared)
    axial_slowness = q / wave_speed
    eta_e = q / eps
    eta_h = mu / q

    kwargs = dict(
        source_layer=int(source_layer),
        source_position=float(source_position),
        upper_observation=upper_observation,
        relative_tolerance=float(impulse_tolerance),
        max_impulses=int(max_impulses),
        maximum_delay=maximum_delay,
        termination=termination,
    )
    return LayeredImpulseResponses(
        vi_e=_line_response_train(interfaces, eta_e, axial_slowness, response="vi", **kwargs),
        vv_e=_line_response_train(interfaces, eta_e, axial_slowness, response="vv", **kwargs),
        vi_h=_line_response_train(interfaces, eta_h, axial_slowness, response="vi", **kwargs),
        vv_h=_line_response_train(interfaces, eta_h, axial_slowness, response="vv", **kwargs),
        observation_impedance=float(observation_impedance),
        observation_wave_speed=float(wave_speed),
        observation_material_index=int(exterior),
    )


class LayeredEquivalentCurrentTimeMonitor(EquivalentCurrentTimeMonitor):
    """Stream Capoglu's direct layered-medium time-domain NFFFT.

    The parent class supplies the already validated equivalent-current surface
    geometry, Yee collocation stencils, MPI ownership, and field gathering.
    Its homogeneous delay buffers are replaced here by the verified layered
    TE/TM impulse trains. CPU and MPI collection use the Cython/OpenMP path;
    CUDA, OpenCL, and Metal use device-resident gather and deposition kernels.
    """

    def __init__(
        self,
        name,
        lower,
        upper,
        spacing,
        field_shape,
        dt,
        iterations,
        theta,
        phi,
        origin,
        medium: LayeredMedium,
        *,
        real_dtype,
        impulse_tolerance=1e-10,
        max_impulses=100_000,
        nthreads=1,
        device_backend=None,
        mpi_grid=None,
    ):
        if medium.relative_permittivity.shape[0] != 1:
            raise ValueError("direct layered time medium must contain one constitutive sample")
        eps_absolute = np.asarray(medium.relative_permittivity[0])
        mu_absolute = np.asarray(medium.relative_permeability[0])
        _validate_lossless_stack(medium.interfaces, eps_absolute, mu_absolute)
        medium.validate(np.asarray((1.0,)))

        # The parent allocation is temporary and is immediately replaced. It
        # keeps all existing surface-stencil code in one authoritative place;
        # a later performance-only refactor can split the sampler into a base
        # class without changing this monitor's numerical behaviour.
        open_exterior = -1 if medium.termination is not None and medium.termination.side == "positive" else 0
        exterior_speed = c / np.sqrt(float(np.real(eps_absolute[open_exterior] * mu_absolute[open_exterior])))
        exterior_impedance = np.sqrt(
            mu_0
            * float(np.real(mu_absolute[open_exterior]))
            / (epsilon_0 * float(np.real(eps_absolute[open_exterior])))
        )
        super().__init__(
            name,
            lower,
            upper,
            spacing,
            field_shape,
            dt,
            iterations,
            theta,
            phi,
            origin,
            real_dtype=real_dtype,
            wave_speed=exterior_speed,
            impedance=exterior_impedance,
            nthreads=nthreads,
            device_backend=device_backend,
            mpi_grid=mpi_grid,
        )
        self.medium = medium
        self.impulse_tolerance = float(impulse_tolerance)
        self.max_impulses = int(max_impulses)
        self.collection_backend = (
            f"{device_backend}_device_layered"
            if device_backend is not None
            else ("cython_openmp_layered" if _deposit_layered_impulse_time is not None else "numpy_layered_reference")
        )
        if self.mpi_comm is not None:
            self.collection_backend = f"mpi_{self.collection_backend}"

        basis = np.asarray(AXIS_BASES[medium.axis], dtype=self.real_dtype)
        self.local_basis = _readonly(basis, self.real_dtype)
        relative_positions = (self.positions - self.origin) @ basis.T
        local_directions = self.directions @ basis.T
        _observation_indices(local_directions[:, 2], medium)
        normal_origin = self.origin["xyz".index(medium.axis)]
        interfaces = np.asarray(medium.interfaces, dtype=self.real_dtype) - normal_origin
        termination = None
        if medium.termination is not None:
            termination = LayeredTermination(
                medium.termination.kind,
                medium.termination.side,
                medium.termination.position - normal_origin,
            )
            outside = (
                relative_positions[:, 2] > termination.position
                if termination.side == "positive"
                else relative_positions[:, 2] < termination.position
            )
            if np.any(outside):
                raise ValueError("equivalent-current surface cannot extend beyond the PEC termination")
        layer_index = np.searchsorted(-interfaces, -relative_positions[:, 2], side="left").astype(np.int64)
        self.local_positions = _readonly(relative_positions, self.real_dtype)
        self.local_directions = _readonly(local_directions, self.real_dtype)
        self.layer_index = _readonly(layer_index, np.int64)
        self.interfaces_relative = _readonly(interfaces, self.real_dtype)
        self.termination_relative = termination
        self.eps_absolute = _readonly(np.real(eps_absolute), self.real_dtype)
        self.mu_absolute = _readonly(np.real(mu_absolute), self.real_dtype)

        template_responses = []
        row_template = []
        row_integer_shift = []
        row_fractional_shift = []
        observation_impedance = np.empty(self.directions.shape[0], dtype=self.real_dtype)
        observation_wave_speed = np.empty_like(observation_impedance)
        observation_index = np.empty(self.directions.shape[0], dtype=np.int32)
        minimum_delay = np.inf
        maximum_delay = -np.inf
        impulse_counts = np.zeros((self.directions.shape[0], 4), dtype=np.int64)
        discarded_sums = np.zeros((self.directions.shape[0], 4), dtype=self.real_dtype)
        names = ("vi_e", "vv_e", "vi_h", "vv_h")

        for direction_number, direction in enumerate(self.local_directions):
            exterior = 0 if direction[2] > 0 else self.eps_absolute.size - 1
            direction_speed = c / np.sqrt(self.eps_absolute[exterior] * self.mu_absolute[exterior])
            lateral_delays = (
                -(direction[0] * self.local_positions[:, 0] + direction[1] * self.local_positions[:, 1])
                / direction_speed
            )
            lateral_coordinates = lateral_delays / self.dt
            # Rectangular surfaces contain many patches at the same axial
            # coordinate. Their four TL Green responses are identical; only
            # the inexpensive lateral delay differs. Avoid re-enumerating the
            # same reflection paths for every transverse patch.
            axial_response_cache = {}
            for patch in range(self.npatches):
                axial_position = float(self.local_positions[patch, 2])
                cache_key = (int(self.layer_index[patch]), axial_position)
                cached = axial_response_cache.get(cache_key)
                if cached is None:
                    item = build_layered_impulse_responses(
                        self.interfaces_relative,
                        self.eps_absolute,
                        self.mu_absolute,
                        direction,
                        axial_position,
                        source_layer=cache_key[0],
                        impulse_tolerance=self.impulse_tolerance,
                        max_impulses=self.max_impulses,
                        termination=self.termination_relative,
                    )
                    prepared = []
                    for response_name in names:
                        train = getattr(item, response_name)
                        coordinate = train.delays / self.dt
                        integer = np.floor(coordinate).astype(np.int64)
                        prepared.append(
                            _PreparedImpulseTrain(
                                _readonly(integer, np.int64),
                                _readonly(coordinate - integer, self.real_dtype),
                                _readonly(train.amplitudes, self.real_dtype),
                            )
                        )
                    template_index = len(template_responses)
                    template_responses.append(tuple(prepared))
                    cached = (item, template_index)
                    axial_response_cache[cache_key] = cached
                item, template_index = cached
                lateral_coordinate = lateral_coordinates[patch]
                lateral_integer = int(np.floor(lateral_coordinate))
                row_template.append(template_index)
                row_integer_shift.append(lateral_integer)
                row_fractional_shift.append(lateral_coordinate - lateral_integer)
                observation_impedance[direction_number] = item.observation_impedance
                observation_wave_speed[direction_number] = item.observation_wave_speed
                observation_index[direction_number] = item.observation_material_index
                for response_number, response_name in enumerate(names):
                    train = getattr(item, response_name)
                    delays = train.delays + lateral_delays[patch]
                    impulse_counts[direction_number, response_number] += train.delays.size
                    discarded_sums[direction_number, response_number] += train.discarded_path_amplitude_sum
                    if delays.size:
                        minimum_delay = min(minimum_delay, float(np.min(delays / self.dt)))
                        maximum_delay = max(maximum_delay, float(np.max(delays / self.dt)))

        if self.mpi_comm is not None:
            bounds = self.mpi_comm.allgather((minimum_delay, maximum_delay))
            minimum_delay = min(item[0] for item in bounds)
            maximum_delay = max(item[1] for item in bounds)
            global_counts = np.empty_like(impulse_counts)
            global_discarded = np.empty_like(discarded_sums)
            from mpi4py import MPI

            self.mpi_comm.Allreduce(impulse_counts, global_counts, op=MPI.SUM)
            self.mpi_comm.Allreduce(discarded_sums, global_discarded, op=MPI.SUM)
            impulse_counts = global_counts
            discarded_sums = global_discarded
        if not np.isfinite(minimum_delay) or not np.isfinite(maximum_delay):
            raise RuntimeError("layered equivalent-current surface has no impulse responses")

        def pack_response(response_number):
            offsets = [0]
            integers = []
            fractions = []
            amplitudes = []
            for responses in template_responses:
                train = responses[response_number]
                integers.append(train.integer_delay)
                fractions.append(train.fraction)
                amplitudes.append(train.amplitudes)
                offsets.append(offsets[-1] + train.integer_delay.size)
            return (
                _readonly(offsets, np.int64),
                _readonly(np.concatenate(integers) if integers else np.empty(0), np.int64),
                _readonly(np.concatenate(fractions) if fractions else np.empty(0), self.real_dtype),
                _readonly(np.concatenate(amplitudes) if amplitudes else np.empty(0), self.real_dtype),
            )

        self._response_csr = tuple(pack_response(index) for index in range(4))
        self._row_template = _readonly(row_template, np.int64)
        self._row_integer_shift = _readonly(row_integer_shift, np.int64)
        self._row_fractional_shift = _readonly(row_fractional_shift, self.real_dtype)
        self.impedance = _readonly(observation_impedance, self.real_dtype)
        self.wave_speed = _readonly(observation_wave_speed, self.real_dtype)
        self.observation_material_index = _readonly(observation_index, np.int32)
        self.impulse_counts = _readonly(impulse_counts, np.int64)
        self.discarded_path_amplitude_sums = _readonly(discarded_sums, self.real_dtype)

        cos_theta = self.local_directions[:, 2]
        sin_theta = np.hypot(self.local_directions[:, 0], self.local_directions[:, 1])
        cos_phi = np.ones_like(cos_theta)
        sin_phi = np.zeros_like(cos_theta)
        nonaxial = sin_theta > 1e-8
        cos_phi[nonaxial] = self.local_directions[nonaxial, 0] / sin_theta[nonaxial]
        sin_phi[nonaxial] = self.local_directions[nonaxial, 1] / sin_theta[nonaxial]
        dyadic_sign = np.where(cos_theta > 0, 1.0, -1.0)
        exterior_eps = self.eps_absolute[self.observation_material_index]
        exterior_mu = self.mu_absolute[self.observation_material_index]
        patch_eps = self.eps_absolute[self.layer_index]
        patch_mu = self.mu_absolute[self.layer_index]
        self._cos_theta = _readonly(cos_theta, self.real_dtype)
        self._sin_theta = _readonly(sin_theta, self.real_dtype)
        self._cos_phi = _readonly(cos_phi, self.real_dtype)
        self._sin_phi = _readonly(sin_phi, self.real_dtype)
        self._local_theta_basis = _readonly(
            np.column_stack(
                (
                    cos_theta * cos_phi,
                    cos_theta * sin_phi,
                    -sin_theta,
                )
            ),
            self.real_dtype,
        )
        self._local_phi_basis = _readonly(
            np.column_stack((-sin_phi, cos_phi, np.zeros_like(cos_phi))),
            self.real_dtype,
        )
        self._j_common = _readonly(mu_0 * exterior_mu * dyadic_sign, self.real_dtype)
        self._m_common = _readonly(
            self.impedance * epsilon_0 * exterior_eps * dyadic_sign,
            self.real_dtype,
        )
        self._inverse_eps_ratio = _readonly(exterior_eps[:, np.newaxis] / patch_eps[np.newaxis, :], self.real_dtype)
        self._inverse_mu_ratio = _readonly(exterior_mu[:, np.newaxis] / patch_mu[np.newaxis, :], self.real_dtype)

        self._time_origin_step = int(np.floor(minimum_delay)) - 2
        last_step = self.iterations - 1 + int(np.ceil(maximum_delay)) + 2
        self._raw_length = last_step - self._time_origin_step + 1
        self._theta_output = (
            np.zeros((self.directions.shape[0], self._raw_length), dtype=self.real_dtype)
            if device_backend is None
            else None
        )
        self._phi_output = None if self._theta_output is None else np.zeros_like(self._theta_output)
        self._complete_start_step = int(np.ceil(maximum_delay + 1))
        self._complete_stop_step = int(np.floor(minimum_delay + self.iterations - 1))
        if self._complete_stop_step < self._complete_start_step:
            raise ValueError("time window is too short to contain one complete layered retarded history")
        self._previous_electric = None
        self._previous_magnetic = None
        self._next_electric = 0
        self._next_magnetic = 0
        self._finalised = False
        self._result = None

    def validate_materials(self, material_ids, id_lookup):
        """Layered transform surfaces intentionally cross multiple materials."""

        self.surface_material_id = -1
        return self.surface_material_id

    def _deposit_train(self, output, sample_index, offset, train, value):
        if train.integer_delay.size == 0 or value == 0:
            return
        destination = sample_index + train.integer_delay - self._time_origin_step
        if destination[0] < 0 or destination[-1] + 1 >= self._raw_length:
            raise RuntimeError("layered equivalent-current output buffer is too short")
        coordinate_fraction = train.fraction + offset
        carry = np.floor(coordinate_fraction).astype(np.int64)
        fraction = coordinate_fraction - carry
        destination = destination + carry
        weighted = value * train.amplitudes
        np.add.at(output, destination, (1 - fraction) * weighted)
        np.add.at(output, destination + 1, fraction * weighted)

    def _deposit_response(self, output, sample_index, offset, response_number, values):
        values = np.ascontiguousarray(values, dtype=self.real_dtype)
        offsets, integer, fraction, amplitude = self._response_csr[response_number]
        if _deposit_layered_impulse_time is not None:
            _deposit_layered_impulse_time(
                self.nthreads,
                sample_index,
                int(offset == 0.5),
                values,
                self._row_template,
                self._row_integer_shift,
                self._row_fractional_shift,
                offsets,
                integer,
                fraction,
                amplitude,
                self._time_origin_step,
                output,
            )
            return
        for direction in range(values.shape[0]):
            for patch in range(values.shape[1]):
                row = direction * values.shape[1] + patch
                template = int(self._row_template[row])
                start, stop = int(offsets[template]), int(offsets[template + 1])
                train = _PreparedImpulseTrain(
                    integer[start:stop] + self._row_integer_shift[row],
                    fraction[start:stop],
                    amplitude[start:stop],
                )
                self._deposit_train(
                    output[direction],
                    sample_index,
                    offset + self._row_fractional_shift[row],
                    train,
                    values[direction, patch],
                )

    def _deposit_electric_current(self, sample_index, offset, current):
        """Deposit the derivative of ``J = n x H``."""

        local = np.asarray(current @ self.local_basis.T, dtype=self.real_dtype)
        radial = (
            self._cos_phi[:, np.newaxis] * local[np.newaxis, :, 0]
            + self._sin_phi[:, np.newaxis] * local[np.newaxis, :, 1]
        )
        phi_component = (
            -self._sin_phi[:, np.newaxis] * local[np.newaxis, :, 0]
            + self._cos_phi[:, np.newaxis] * local[np.newaxis, :, 1]
        )
        area_common = self.area_weights[np.newaxis, :] * self._j_common[:, np.newaxis]
        self._deposit_response(
            self._theta_output,
            sample_index,
            offset,
            0,
            area_common * radial,
        )
        self._deposit_response(
            self._theta_output,
            sample_index,
            offset,
            1,
            area_common * (-self._sin_theta[:, np.newaxis]) * self._inverse_eps_ratio * local[np.newaxis, :, 2],
        )
        self._deposit_response(
            self._phi_output,
            sample_index,
            offset,
            2,
            area_common * self._cos_theta[:, np.newaxis] * phi_component,
        )

    def _deposit_magnetic_current(self, sample_index, offset, current):
        """Deposit the derivative of ``M = -n x E``."""

        local = np.asarray(current @ self.local_basis.T, dtype=self.real_dtype)
        radial = (
            self._cos_phi[:, np.newaxis] * local[np.newaxis, :, 0]
            + self._sin_phi[:, np.newaxis] * local[np.newaxis, :, 1]
        )
        phi_component = (
            -self._sin_phi[:, np.newaxis] * local[np.newaxis, :, 0]
            + self._cos_phi[:, np.newaxis] * local[np.newaxis, :, 1]
        )
        area_common = self.area_weights[np.newaxis, :] * self._m_common[:, np.newaxis]
        self._deposit_response(
            self._theta_output,
            sample_index,
            offset,
            1,
            area_common * phi_component,
        )
        self._deposit_response(
            self._phi_output,
            sample_index,
            offset,
            3,
            area_common * (-self._cos_theta[:, np.newaxis]) * radial,
        )
        self._deposit_response(
            self._phi_output,
            sample_index,
            offset,
            2,
            area_common
            * self._cos_theta[:, np.newaxis]
            * self._sin_theta[:, np.newaxis]
            * self._inverse_mu_ratio
            * local[np.newaxis, :, 2],
        )

    def observe_electric(self, iteration, Ex, Ey, Ez):
        if self.device_backend is not None:
            raise RuntimeError("device layered-current monitors are observed by the backend")
        if iteration != self._next_electric:
            raise ValueError(f"expected electric iteration {self._next_electric}, got {iteration}")
        electric = self._gather_vector(("Ex", "Ey", "Ez"), (Ex, Ey, Ez))
        magnetic_current = -np.cross(self.normals, electric)
        if self._previous_electric is not None:
            derivative = (magnetic_current - self._previous_electric) / self.dt
            self._deposit_magnetic_current(iteration - 1, 0.5, derivative)
        self._previous_electric = magnetic_current
        self._next_electric += 1

    def observe_magnetic(self, iteration, Hx, Hy, Hz):
        if self.device_backend is not None:
            raise RuntimeError("device layered-current monitors are observed by the backend")
        if iteration != self._next_magnetic:
            raise ValueError(f"expected magnetic iteration {self._next_magnetic}, got {iteration}")
        magnetic = self._gather_vector(("Hx", "Hy", "Hz"), (Hx, Hy, Hz))
        electric_current = np.cross(self.normals, magnetic)
        if self._previous_magnetic is not None:
            derivative = (electric_current - self._previous_magnetic) / self.dt
            self._deposit_electric_current(iteration, 0.0, derivative)
        self._previous_magnetic = electric_current
        self._next_magnetic += 1

    def finalise(self):
        if self._finalised:
            return
        if self._next_electric != self.iterations or self._next_magnetic != self.iterations:
            raise RuntimeError("layered equivalent-current monitor missed one or more time steps")
        if self._theta_output is None or self._phi_output is None:
            raise RuntimeError("layered equivalent-current device output was not loaded")
        if self.mpi_comm is not None:
            from mpi4py import MPI

            coordinator = self.mpi_comm.Get_rank() == 0
            theta = np.empty_like(self._theta_output) if coordinator else None
            phi = np.empty_like(self._phi_output) if coordinator else None
            self.mpi_comm.Reduce(self._theta_output, theta, op=MPI.SUM, root=0)
            self.mpi_comm.Reduce(self._phi_output, phi, op=MPI.SUM, root=0)
            if not coordinator:
                self._finalised = True
                return
            self._theta_output = theta
            self._phi_output = phi
        start = self._complete_start_step - self._time_origin_step
        stop = self._complete_stop_step - self._time_origin_step + 1
        scale = -1 / (4 * np.pi)
        local_theta = scale * self._theta_output[:, start:stop]
        local_phi = scale * self._phi_output[:, start:stop]
        electric_local = (
            local_theta[:, :, np.newaxis] * self._local_theta_basis[:, np.newaxis, :]
            + local_phi[:, :, np.newaxis] * self._local_phi_basis[:, np.newaxis, :]
        )
        electric_global = np.einsum("dti,ij->dtj", electric_local, self.local_basis, optimize=True)
        electric_theta = _readonly(
            np.einsum("dti,di->dt", electric_global, self.theta_basis, optimize=True),
            self.real_dtype,
        )
        electric_phi = _readonly(
            np.einsum("dti,di->dt", electric_global, self.phi_basis, optimize=True),
            self.real_dtype,
        )
        times = _readonly(
            self.dt
            * np.arange(
                self._complete_start_step,
                self._complete_stop_step + 1,
                dtype=self.real_dtype,
            ),
            self.real_dtype,
        )
        ratios = np.zeros(self.directions.shape[0], dtype=self.real_dtype)
        width = min(TERMINAL_DECAY_WINDOW_SAMPLES, times.size)
        for values in (electric_theta, electric_phi):
            peaks = np.max(np.abs(values), axis=1)
            terminals = np.max(np.abs(values[:, -width:]), axis=1)
            nonzero = peaks > 0
            ratios[nonzero] = np.maximum(ratios[nonzero], terminals[nonzero] / peaks[nonzero])
        decay_ok = ratios <= TERMINAL_DECAY_THRESHOLD
        self._result = EquivalentCurrentTimeResult(
            name=self.name,
            times=times,
            theta=self.theta,
            phi=self.phi,
            directions=self.directions,
            fields=MappingProxyType({"Etheta": electric_theta, "Ephi": electric_phi}),
            terminal_field_ratios=_readonly(ratios, self.real_dtype),
            terminal_decay_ok=_readonly(decay_ok, bool),
            terminal_decay_threshold=TERMINAL_DECAY_THRESHOLD,
            terminal_decay_window_samples=TERMINAL_DECAY_WINDOW_SAMPLES,
            collection_backend=self.collection_backend,
        )
        if not np.all(decay_ok):
            worst = int(np.argmax(ratios))
            logger.warning(
                "Layered equivalent-current time monitor %r has not decayed below %.1e "
                "(direction %d ratio %.3e); increase the simulation time window.",
                self.name,
                TERMINAL_DECAY_THRESHOLD,
                worst,
                ratios[worst],
            )
        self._finalised = True
