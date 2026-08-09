# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
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
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Streaming frequency-domain KSIR monitor shared by all local solvers."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import numpy.typing as npt
from scipy.constants import c, epsilon_0, mu_0

try:
    from gprMax.cython.ntff import accumulate_surface_dft as _accumulate_surface_dft
except ImportError:  # pragma: no cover - permits source-tree use before compilation
    _accumulate_surface_dft = None

from .conventions import (
    FORWARD_TRANSFORM_KERNEL,
    OUTGOING_GREEN_RADIAL_FACTOR,
    PHASOR_TIME_DEPENDENCE,
)
from .closures import (
    ResolvedKSIRClosure,
    closure_from_metadata,
)
from .evaluator import (
    evaluate_far_zone_patches,
    project_cartesian_to_spherical,
    spherical_directions,
)
from .surfaces import (
    COMPONENT_OFFSETS,
    FACES,
    KSIRComponentSurface,
    build_component_surface,
)


ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")
MAGNETIC_COMPONENTS = ("Hx", "Hy", "Hz")
ALL_COMPONENTS = ELECTRIC_COMPONENTS + MAGNETIC_COMPONENTS
DFT_PHASE_REANCHOR_INTERVAL = 1024


def _readonly(array: npt.NDArray) -> npt.NDArray:
    array.setflags(write=False)
    return array


def _dft_phase_at_time(
    frequencies: npt.ArrayLike, time: float, dtype: npt.DTypeLike
) -> npt.NDArray[np.complexfloating]:
    """Return ``exp(-j omega t)`` without large-argument float32 drift.

    Frequencies and the returned oscillator retain the configured simulation
    types. Only the transcendental argument reduction uses float64, then the
    result is cast to the configured complex dtype.
    """

    phase_frequencies = np.asarray(frequencies, dtype=np.float64)
    return np.exp(-2j * np.pi * phase_frequencies * float(time)).astype(dtype)


def _frequencies(values: npt.ArrayLike, dtype) -> npt.NDArray[np.floating]:
    frequencies = np.asarray(values, dtype=dtype)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("frequencies must contain finite, non-negative values")
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError("frequencies must not contain duplicates")
    return _readonly(frequencies.copy())


def validate_nyquist_frequencies(frequencies: npt.ArrayLike, dt: float) -> float:
    """Reject requested DFT frequencies above the temporal Nyquist limit.

    The comparison deliberately uses the original values in double precision,
    before they are cast to the configured simulation precision. This avoids a
    just-above-Nyquist request being rounded down to the limit in a float32
    simulation.

    Returns:
        The Nyquist frequency in Hz.
    """

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and greater than zero")
    values = np.asarray(frequencies, dtype=np.float64)
    nyquist = 0.5 / float(dt)
    if np.any(np.isfinite(values) & (values > nyquist)):
        highest = float(np.max(values[np.isfinite(values)]))
        raise ValueError(
            "frequencies must not exceed the temporal Nyquist limit "
            f"1/(2*dt) = {nyquist:g} Hz; highest requested frequency is "
            f"{highest:g} Hz"
        )
    return nyquist


def _angles(name: str, values: npt.ArrayLike, dtype) -> npt.NDArray[np.floating]:
    angles = np.atleast_1d(np.asarray(values, dtype=dtype))
    if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if angles.size > 1 and np.any(np.diff(angles) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    if name == "theta" and (np.any(angles < 0) or np.any(angles > 180)):
        raise ValueError("theta must lie between 0 and 180 degrees")
    if name == "phi" and (np.any(angles < 0) or np.any(angles > 360)):
        raise ValueError("phi must lie between 0 and 360 degrees")
    return _readonly(angles.copy())


def _window(name: str, iterations: int, dtype) -> npt.NDArray[np.floating]:
    normalized = name.lower().replace("-", "_")
    if normalized in ("rectangular", "boxcar", "none"):
        values = np.ones(iterations, dtype=dtype)
    elif normalized in ("hann", "hanning"):
        values = np.hanning(iterations).astype(dtype)
    else:
        raise ValueError("window must be 'rectangular' or 'hann'")
    return _readonly(values)


def surface_compatibility_signature(
    surface: KSIRComponentSurface,
    frequencies: npt.ArrayLike,
    dt: float,
    iterations: int,
    sample_time_offset: float,
    window: str,
    background_er: float,
    background_mr: float,
    closure_signature: str,
    numerical_dtype,
) -> str:
    """Return a stable hash for saved-surface subtraction compatibility."""

    digest = sha256()
    for text in (
        "gprMax-KSIR-surface-v2",
        surface.component,
        PHASOR_TIME_DEPENDENCE,
        FORWARD_TRANSFORM_KERNEL,
        OUTGOING_GREEN_RADIAL_FACTOR,
        window,
        closure_signature,
        np.dtype(numerical_dtype).name,
    ):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    arrays = (
        np.asarray(surface.lower, dtype="<i8"),
        np.asarray(surface.upper, dtype="<i8"),
        np.asarray(surface.grid_spacing, dtype="<f8"),
        np.asarray(surface.field_shape, dtype="<i8"),
        np.asarray(surface.patch_positions, dtype="<f8"),
        np.asarray(surface.normals, dtype="<f8"),
        np.asarray(surface.area_weights, dtype="<f8"),
        np.asarray(frequencies, dtype="<f8"),
        np.asarray(
            (dt, sample_time_offset, background_er, background_mr), dtype="<f8"
        ),
        np.asarray((iterations,), dtype="<i8"),
    )
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class KSIRComponentPhasors:
    """Collocated surface phasors for one Cartesian field component."""

    component: str
    surface: KSIRComponentSurface
    field: npt.NDArray[np.complexfloating]
    normal_derivative: npt.NDArray[np.complexfloating]
    compatibility_signature: str


@dataclass(frozen=True)
class KSIRFrequencyResult:
    """In-memory production KSIR far-field result."""

    name: str
    frequencies: npt.NDArray[np.floating]
    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    directions: npt.NDArray[np.floating]
    origin: npt.NDArray[np.floating]
    range_normalized_fields: Mapping[str, npt.NDArray[np.complexfloating]]
    electric_cartesian: Optional[npt.NDArray[np.complexfloating]]
    electric_spherical: Optional[npt.NDArray[np.complexfloating]]
    magnetic_cartesian: Optional[npt.NDArray[np.complexfloating]]
    magnetic_spherical: Optional[npt.NDArray[np.complexfloating]]
    radiation_intensity: Optional[npt.NDArray[np.floating]]
    transversality_error: Optional[npt.NDArray[np.floating]]
    radiated_power: Optional[npt.NDArray[np.floating]]
    directivity: Optional[npt.NDArray[np.floating]]
    maximum_directivity: Optional[npt.NDArray[np.floating]]
    incident_electric: Optional[npt.NDArray[np.complexfloating]]
    bistatic_rcs: Optional[npt.NDArray[np.floating]]
    face_contributions: Mapping[str, npt.NDArray[np.complexfloating]]
    cancellation_indicator: Mapping[str, npt.NDArray[np.floating]]
    active_area: Mapping[str, float]
    missing_area_fraction: Mapping[str, float]
    closure: str
    mathematically_closed: bool


@dataclass(frozen=True)
class KSIRSavedFarField:
    """Far-field components reevaluated from a saved surface DFT."""

    frequencies: npt.NDArray[np.floating]
    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    directions: npt.NDArray[np.floating]
    origin: npt.NDArray[np.floating]
    range_normalized_fields: Mapping[str, npt.NDArray[np.complexfloating]]
    closure: str
    mathematically_closed: bool


def _evaluate_component_with_closure(
    surface: KSIRComponentSurface,
    field: npt.ArrayLike,
    normal_derivative: npt.ArrayLike,
    closure: ResolvedKSIRClosure,
    frequencies: npt.ArrayLike,
    directions: npt.ArrayLike,
    wave_speed: float,
    origin: npt.ArrayLike,
    nthreads: int = 1,
    retain_face_contributions: bool = True,
) -> tuple[
    npt.NDArray[np.complexfloating],
    Optional[npt.NDArray[np.complexfloating]],
]:
    """Evaluate one component and retain contributions by physical face."""

    field_values = np.asarray(field)
    derivative_values = np.asarray(normal_derivative)
    result_dtype = np.result_type(field_values.dtype, derivative_values.dtype)
    if result_dtype.kind != "c":
        raise ValueError("KSIR surface phasors must use a complex dtype")
    frequency_values = np.asarray(frequencies)
    direction_values = np.asarray(directions)
    total = np.zeros(
        (frequency_values.size, direction_values.shape[0]), dtype=result_dtype
    )
    face_contributions = (
        np.zeros(
            (frequency_values.size, direction_values.shape[0], len(FACES)),
            dtype=result_dtype,
        )
        if retain_face_contributions
        else None
    )
    for (
        face_id,
        positions,
        normals,
        areas,
        image_field,
        image_derivative,
    ) in closure.transformed_faces(
        surface,
        field_values,
        derivative_values,
    ):
        contribution = evaluate_far_zone_patches(
            positions,
            normals,
            areas,
            frequencies,
            directions,
            image_field,
            image_derivative,
            wave_speed=wave_speed,
            origin=origin,
            nthreads=nthreads,
        )
        total += contribution
        if face_contributions is not None:
            face_contributions[:, :, FACES.index(face_id)] += contribution
    return _readonly(total), (
        None if face_contributions is None else _readonly(face_contributions)
    )


class _ComponentDFTAccumulator:
    """Running raw inside/outside DFTs for one component surface."""

    def __init__(
        self,
        surface: KSIRComponentSurface,
        frequencies: npt.NDArray[np.floating],
        dt: float,
        iterations: int,
        sample_time_offset_steps: float,
        window_values: npt.NDArray[np.floating],
        window_name: str,
        closure_signature: str,
        real_dtype: npt.DTypeLike,
        dtype: npt.DTypeLike,
        nthreads: int,
    ):
        self.surface = surface
        self.frequencies = frequencies
        self.dt = dt
        self.iterations = iterations
        self.sample_time_offset_steps = sample_time_offset_steps
        self.window_values = window_values
        self.window_name = window_name
        self.closure_signature = closure_signature
        self.real_dtype = np.dtype(real_dtype)
        self.dtype = np.dtype(dtype)
        self.nthreads = nthreads
        shape = (frequencies.size, surface.npatches)
        self.inside_dft = np.zeros(shape, dtype=self.dtype)
        self.outside_dft = np.zeros(shape, dtype=self.dtype)
        self._phase = _dft_phase_at_time(
            frequencies, sample_time_offset_steps * dt, self.dtype
        )
        self._step = _dft_phase_at_time(frequencies, dt, self.dtype)
        self._inside_indices = np.ascontiguousarray(
            np.concatenate([face.inside_flat_indices for face in surface.faces]),
            dtype=np.int64,
        )
        self._outside_indices = np.ascontiguousarray(
            np.concatenate([face.outside_flat_indices for face in surface.faces]),
            dtype=np.int64,
        )
        self._next_iteration = 0
        self._finalised = False

    def sampling_multiplier(self, iteration: int) -> npt.NDArray[np.complexfloating]:
        """Return this sample's configured-precision DFT multiplier and advance."""

        if self._finalised:
            raise RuntimeError("cannot observe a finalised KSIR DFT accumulator")
        if iteration != self._next_iteration:
            raise ValueError(
                f"expected KSIR iteration {self._next_iteration}, received {iteration}"
            )

        multiplier = np.asarray(
            self.dt * self.window_values[iteration] * self._phase,
            dtype=self.dtype,
        )

        self._next_iteration += 1
        if self._next_iteration % DFT_PHASE_REANCHOR_INTERVAL == 0:
            physical_time = (
                self._next_iteration + self.sample_time_offset_steps
            ) * self.dt
            self._phase = _dft_phase_at_time(
                self.frequencies, physical_time, self.dtype
            )
        else:
            self._phase *= self._step
        return multiplier

    def observe(self, iteration: int, field: npt.ArrayLike) -> None:
        multiplier = self.sampling_multiplier(iteration)
        values = np.asarray(field)
        if values.shape != self.surface.field_shape:
            raise ValueError(
                f"field shape {values.shape} does not match surface field shape "
                f"{self.surface.field_shape}"
            )
        # Solver arrays already have this dtype and remain zero-copy. The cast
        # also keeps the public monitor robust when fed integer/test arrays.
        flat = np.ascontiguousarray(values, dtype=self.real_dtype).ravel()
        if _accumulate_surface_dft is not None:
            _accumulate_surface_dft(
                self.nthreads,
                self._inside_indices,
                self._outside_indices,
                flat,
                multiplier,
                self.inside_dft,
                self.outside_dft,
            )
        else:
            inside = np.take(flat, self._inside_indices)
            outside = np.take(flat, self._outside_indices)
            contribution = multiplier[:, np.newaxis]
            self.inside_dft += contribution * inside[np.newaxis, :]
            self.outside_dft += contribution * outside[np.newaxis, :]

    def load_device_dfts(
        self, inside: npt.ArrayLike, outside: npt.ArrayLike
    ) -> None:
        """Load completed raw DFTs accumulated by a device backend."""

        if self._finalised:
            raise RuntimeError("cannot load a finalised KSIR DFT accumulator")
        if self._next_iteration != self.iterations:
            raise RuntimeError(
                f"KSIR component {self.surface.component} received "
                f"{self._next_iteration} of {self.iterations} expected samples"
            )
        expected = self.inside_dft.shape
        inside_values = np.asarray(inside)
        outside_values = np.asarray(outside)
        if inside_values.shape != expected or outside_values.shape != expected:
            raise ValueError(f"device DFT arrays must have shape {expected}")
        if inside_values.dtype != self.dtype or outside_values.dtype != self.dtype:
            raise ValueError(f"device DFT arrays must use dtype {self.dtype}")
        self.inside_dft[...] = inside_values
        self.outside_dft[...] = outside_values

    def finalise(
        self, background_er: float, background_mr: float
    ) -> KSIRComponentPhasors:
        if self._finalised:
            raise RuntimeError("KSIR DFT accumulator has already been finalised")
        if self._next_iteration != self.iterations:
            raise RuntimeError(
                f"KSIR component {self.surface.component} received "
                f"{self._next_iteration} of {self.iterations} expected samples"
            )
        field = 0.5 * (self.outside_dft + self.inside_dft)
        derivative_parts = []
        start = 0
        difference = self.outside_dft - self.inside_dft
        for face in self.surface.faces:
            stop = start + face.npatches
            derivative_parts.append(difference[:, start:stop] / face.normal_spacing)
            start = stop
        derivative = np.concatenate(derivative_parts, axis=1)
        _readonly(field)
        _readonly(derivative)
        signature = surface_compatibility_signature(
            self.surface,
            self.frequencies,
            self.dt,
            self.iterations,
            self.sample_time_offset_steps * self.dt,
            self.window_name,
            background_er,
            background_mr,
            self.closure_signature,
            self.dtype,
        )
        self._finalised = True
        return KSIRComponentPhasors(
            component=self.surface.component,
            surface=self.surface,
            field=field,
            normal_derivative=derivative,
            compatibility_signature=signature,
        )


class KSIRFrequencyDomainMonitor:
    """Streaming closed-surface frequency-domain KSIR monitor."""

    def __init__(
        self,
        name: str,
        surfaces: Mapping[str, KSIRComponentSurface],
        frequencies: npt.ArrayLike,
        theta: npt.ArrayLike,
        phi: npt.ArrayLike,
        dt: float,
        iterations: int,
        *,
        real_dtype,
        complex_dtype,
        nthreads: int = 1,
        solver_backend: str = "cpu",
        origin: Optional[npt.ArrayLike] = None,
        window: str = "rectangular",
        wave_speed: Optional[float] = None,
        impedance: Optional[float] = None,
        save_surface_dft: bool = True,
        incident_surface_file: Optional[Path | str] = None,
        incident_monitor_name: Optional[str] = None,
        exterior_index_bounds: Optional[Sequence[Sequence[int]]] = None,
        closure: Optional[ResolvedKSIRClosure] = None,
        allow_external_sources: bool = False,
    ):
        if not name:
            raise ValueError("KSIR monitor name must not be empty")
        if not surfaces:
            raise ValueError("at least one component surface is required")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and greater than zero")
        if not isinstance(iterations, (int, np.integer)) or iterations <= 0:
            raise ValueError("iterations must be an integer greater than zero")
        unknown = set(surfaces) - set(ALL_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown field components: {sorted(unknown)}")
        self.real_dtype = np.dtype(real_dtype)
        self.complex_dtype = np.dtype(complex_dtype)
        if self.real_dtype.kind != "f":
            raise ValueError("real_dtype must be a floating-point dtype")
        if (
            self.complex_dtype.kind != "c"
            or self.complex_dtype.itemsize != 2 * self.real_dtype.itemsize
        ):
            raise ValueError(
                "complex_dtype must be the matching complex dtype for real_dtype"
            )
        if not isinstance(nthreads, (int, np.integer)) or nthreads < 1:
            raise ValueError("nthreads must be an integer greater than zero")
        if not isinstance(window, str):
            raise ValueError("window must be 'rectangular' or 'hann'")
        if wave_speed is not None and (not np.isfinite(wave_speed) or wave_speed <= 0):
            raise ValueError("wave_speed must be finite and greater than zero")
        if impedance is not None and (not np.isfinite(impedance) or impedance <= 0):
            raise ValueError("impedance must be finite and greater than zero")

        self.name = name
        self.surfaces = MappingProxyType(dict(surfaces))
        self.components = tuple(surfaces)
        self.frequencies = _frequencies(frequencies, self.real_dtype)
        validate_nyquist_frequencies(frequencies, dt)
        self.theta_values = _angles("theta", theta, self.real_dtype)
        self.phi_values = _angles("phi", phi, self.real_dtype)
        theta_grid, phi_grid = np.meshgrid(
            self.theta_values, self.phi_values, indexing="ij"
        )
        self.theta = _readonly(theta_grid.ravel())
        self.phi = _readonly(phi_grid.ravel())
        self.directions = _readonly(
            spherical_directions(self.theta, self.phi, degrees=True)
        )
        if origin is None:
            centres = np.stack([surface.centre for surface in surfaces.values()])
            reference_origin = np.mean(centres, axis=0)
        else:
            reference_origin = np.asarray(origin, dtype=self.real_dtype)
        if reference_origin.shape != (3,) or not np.all(np.isfinite(reference_origin)):
            raise ValueError("origin must contain exactly three finite values")
        self.origin = _readonly(reference_origin.copy())

        self.dt = float(dt)
        self.iterations = int(iterations)
        self.window_name = window.lower().replace("-", "_")
        self.window_values = _window(
            self.window_name, self.iterations, self.real_dtype
        )
        if self.window_name in ("boxcar", "none"):
            self.window_name = "rectangular"
        elif self.window_name == "hanning":
            self.window_name = "hann"
        self.precision = "single" if self.real_dtype.itemsize == 4 else "double"
        if solver_backend not in ("cpu", "cuda", "opencl", "metal"):
            raise ValueError("solver_backend is not a supported gprMax solver")
        self.solver_backend = solver_backend
        self.nthreads = int(nthreads)
        if solver_backend == "cpu":
            self.collection_backend = (
                "cython_openmp"
                if _accumulate_surface_dft is not None
                else "numpy_fallback"
            )
        else:
            self.collection_backend = f"{solver_backend}_device"
        self.save_surface_dft = bool(save_surface_dft)
        self.incident_surface_file = (
            None if incident_surface_file is None else Path(incident_surface_file)
        )
        self.incident_monitor_name = incident_monitor_name or name
        self.allow_external_sources = bool(allow_external_sources)
        self.wave_speed = None if wave_speed is None else float(wave_speed)
        self.impedance = None if impedance is None else float(impedance)
        self._wave_speed_override = wave_speed is not None
        self._impedance_override = impedance is not None
        self.background_er = None
        self.background_mr = None
        self.background_material_id = None
        self.background_material_name = None
        self.surface_material_id = None
        self.closure = (
            ResolvedKSIRClosure("closed", (), (), True, True)
            if closure is None
            else closure
        )
        for surface in surfaces.values():
            face_ids = tuple(face.face_id for face in surface.faces)
            if face_ids != self.closure.active_faces:
                raise ValueError(
                    f"{surface.component} surface faces {face_ids} do not match "
                    f"closure faces {self.closure.active_faces}"
                )
        if exterior_index_bounds is None:
            self.exterior_index_bounds = None
        else:
            bounds = np.asarray(exterior_index_bounds, dtype=np.int64)
            if bounds.shape != (3, 2) or np.any(bounds[:, 1] < bounds[:, 0]):
                raise ValueError("exterior_index_bounds must have shape (3, 2)")
            self.exterior_index_bounds = bounds

        self._accumulators = {
            component: _ComponentDFTAccumulator(
                surface,
                self.frequencies,
                self.dt,
                self.iterations,
                0.0 if component in ELECTRIC_COMPONENTS else 0.5,
                self.window_values,
                self.window_name,
                self.closure.signature,
                self.real_dtype,
                self.complex_dtype,
                self.nthreads,
            )
            for component, surface in surfaces.items()
        }
        self._surface_data = None
        self._result = None
        self._finalised = False

        self.associated_plane_wave = None
        self.plane_wave_metadata = None
        self._incident_reference_index = None
        self._incident_reference_positions = None
        self._incident_electric = None
        self._incident_phase = np.ones(
            self.frequencies.size, dtype=self.complex_dtype
        )
        self._incident_step = _dft_phase_at_time(
            self.frequencies, self.dt, self.complex_dtype
        )
        self._incident_next_iteration = 0

    @property
    def result(self) -> KSIRFrequencyResult:
        if self._result is None:
            raise RuntimeError(
                "KSIR result is not available until the solver has finalised"
            )
        return self._result

    @property
    def surface_data(self) -> Mapping[str, KSIRComponentPhasors]:
        if self._surface_data is None:
            raise RuntimeError("KSIR surface DFT is not available until finalisation")
        return self._surface_data

    def validate_materials(
        self, material_ids: npt.ArrayLike, id_lookup: Mapping[str, int]
    ) -> int:
        """Verify that all straddling samples use one homogeneous material ID."""

        ids = np.asarray(material_ids)
        sampled_ids = []
        for component, surface in self.surfaces.items():
            component_ids = ids[id_lookup[component]]
            for face in surface.faces:
                keep = self.closure.material_validation_mask(
                    surface,
                    face,
                    self.real_dtype,
                )
                if np.any(keep):
                    sampled_ids.append(
                        component_ids[tuple(face.inside_indices[keep].T)]
                    )
                    sampled_ids.append(
                        component_ids[tuple(face.outside_indices[keep].T)]
                    )
        if not sampled_ids:
            raise ValueError(
                f"KSIR monitor {self.name!r} has no off-symmetry samples "
                "for material validation"
            )
        unique_ids = np.unique(np.concatenate(sampled_ids))
        if unique_ids.size != 1:
            raise ValueError(
                f"KSIR monitor {self.name!r} surface straddles multiple material IDs: "
                f"{unique_ids.tolist()}"
            )
        self.surface_material_id = int(unique_ids[0])
        if self.exterior_index_bounds is not None:
            exterior_ids = []
            for component, surface in self.surfaces.items():
                slices = tuple(
                    slice(int(lower), int(upper) + 1)
                    for lower, upper in self.exterior_index_bounds
                )
                component_ids = ids[id_lookup[component]][slices]
                coordinate_axes = [
                    (
                        np.arange(lower, upper + 1, dtype=self.real_dtype)
                        + COMPONENT_OFFSETS[component][axis]
                    )
                    * surface.grid_spacing[axis]
                    for axis, (lower, upper) in enumerate(
                        self.exterior_index_bounds
                    )
                ]
                coordinates = np.meshgrid(*coordinate_axes, indexing="ij")
                outside = np.zeros(component_ids.shape, dtype=bool)
                for axis in range(3):
                    outside |= coordinates[axis] < surface.physical_lower[axis]
                    outside |= coordinates[axis] > surface.physical_upper[axis]
                on_symmetry_plane = np.zeros(component_ids.shape, dtype=bool)
                for plane in self.closure.symmetry_planes:
                    tolerance = (
                        16
                        * np.finfo(self.real_dtype).eps
                        * max(
                            abs(plane.coordinate),
                            surface.grid_spacing[plane.axis],
                        )
                    )
                    on_symmetry_plane |= np.isclose(
                        coordinates[plane.axis],
                        plane.coordinate,
                        rtol=0,
                        atol=tolerance,
                    )
                outside &= ~on_symmetry_plane
                for face in self.closure.omitted_faces:
                    axis = "xyz".index(face[0])
                    tolerance = (
                        16
                        * np.finfo(self.real_dtype).eps
                        * max(
                            abs(surface.physical_lower[axis]),
                            abs(surface.physical_upper[axis]),
                            surface.grid_spacing[axis],
                        )
                    )
                    if face.endswith("0"):
                        outside &= (
                            coordinates[axis]
                            > surface.physical_lower[axis] + surface.grid_spacing[axis] + tolerance
                        )
                    else:
                        outside &= (
                            coordinates[axis]
                            < surface.physical_upper[axis] - surface.grid_spacing[axis] - tolerance
                        )
                exterior_ids.append(component_ids[outside])
            exterior_unique = np.unique(np.concatenate(exterior_ids))
            if (
                exterior_unique.size != 1
                or int(exterior_unique[0]) != self.surface_material_id
            ):
                raise ValueError(
                    f"KSIR monitor {self.name!r} exterior region is not "
                    "homogeneous with its surface"
                )
        return self.surface_material_id

    def configure_background(self, materials: Sequence) -> None:
        """Resolve lossless homogeneous Green-function properties."""

        material = next(
            (item for item in materials if item.numID == self.surface_material_id),
            None,
        )
        if material is None:
            raise ValueError(
                f"KSIR monitor {self.name!r} cannot resolve material ID "
                f"{self.surface_material_id}"
            )
        if (
            not np.isfinite(material.er)
            or not np.isfinite(material.mr)
            or material.er <= 0
            or material.mr <= 0
            or material.se != 0
            or material.sm != 0
            or getattr(material, "poles", 0) != 0
        ):
            raise ValueError(
                f"KSIR monitor {self.name!r} requires a lossless, non-dispersive "
                f"background material; got {material.ID!r}"
            )
        self.background_er = float(material.er)
        self.background_mr = float(material.mr)
        self.background_material_id = int(material.numID)
        self.background_material_name = material.ID
        if (
            self.plane_wave_metadata is not None
            and self.plane_wave_metadata["material_id"]
            and self.plane_wave_metadata["material_id"] != material.ID
        ):
            raise ValueError(
                f"KSIR monitor {self.name!r} surface material {material.ID!r} "
                "does not match the associated plane-wave background "
                f"{self.plane_wave_metadata['material_id']!r}"
            )
        if not self._wave_speed_override:
            self.wave_speed = c / np.sqrt(self.background_er * self.background_mr)
        if not self._impedance_override:
            self.impedance = np.sqrt(
                mu_0 * self.background_mr / (epsilon_0 * self.background_er)
            )

    def associate_plane_wave(
        self, plane_wave, grid_spacing: npt.ArrayLike, index: int
    ) -> None:
        """Associate an enclosed DiscretePlaneWave and prepare incident DFT."""

        self.associated_plane_wave = plane_wave
        spacing = np.asarray(grid_spacing, dtype=self.real_dtype)
        reference_index = np.rint(self.origin / spacing).astype(np.int64)
        one_d_index = int(
            np.dot(plane_wave.m[:3], reference_index - plane_wave.origin)
        )
        if plane_wave.axial != 0:
            one_d_index += int(plane_wave.origin_axial)
        if one_d_index < 0 or one_d_index >= plane_wave.E_fields.shape[1]:
            raise ValueError(
                f"KSIR monitor {self.name!r} incident reference falls outside "
                "the associated plane-wave grid"
            )
        self._incident_reference_index = one_d_index
        self._incident_reference_positions = _readonly(
            np.stack(
                [
                    (reference_index + COMPONENT_OFFSETS[component]) * spacing
                    for component in ELECTRIC_COMPONENTS
                ]
            )
        )
        self._incident_electric = np.zeros(
            (self.frequencies.size, 3), dtype=self.complex_dtype
        )
        self.plane_wave_metadata = {
            "index": int(index),
            "corners": np.asarray(plane_wave.corners, dtype=np.int32),
            "waveform_id": plane_wave.waveformID,
            "material_id": (
                "" if plane_wave.materialID is None else plane_wave.materialID
            ),
            "actual_angles": np.asarray(
                plane_wave.actual_angles, dtype=self.real_dtype
            ),
            "polarisation_angle": float(plane_wave.psi),
            "integer_mapping": np.asarray(plane_wave.m[:3], dtype=np.int32),
            "start": float(plane_wave.start),
            "stop": float(plane_wave.stop),
            "reference_grid_index": reference_index,
            "reference_1d_index": one_d_index,
            "reference_positions": self._incident_reference_positions,
        }

    def observe_electric(
        self,
        iteration: int,
        Ex: npt.ArrayLike,
        Ey: npt.ArrayLike,
        Ez: npt.ArrayLike,
    ) -> None:
        fields = {"Ex": Ex, "Ey": Ey, "Ez": Ez}
        for component in ELECTRIC_COMPONENTS:
            if component in self._accumulators:
                self._accumulators[component].observe(iteration, fields[component])

        if self.associated_plane_wave is None:
            return
        multiplier = self.device_incident_sampling_multiplier(iteration)
        incident = self.associated_plane_wave.E_fields[
            :, self._incident_reference_index
        ]
        self._incident_electric += multiplier[:, np.newaxis] * incident[np.newaxis, :]

    def device_incident_sampling_multiplier(
        self, iteration: int
    ) -> npt.NDArray[np.complexfloating]:
        """Advance and return the incident-plane-wave DFT multiplier.

        Device collectors use this independently of the surface-component
        accumulators so the auxiliary one-dimensional plane wave remains on
        the accelerator throughout timestepping.
        """

        if self.associated_plane_wave is None:
            raise RuntimeError(
                f"KSIR monitor {self.name!r} has no associated incident plane wave"
            )
        if iteration != self._incident_next_iteration:
            raise ValueError(
                f"expected incident iteration {self._incident_next_iteration}, "
                f"received {iteration}"
            )
        multiplier = self.dt * self.window_values[iteration] * self._incident_phase
        self._incident_next_iteration += 1
        if self._incident_next_iteration % DFT_PHASE_REANCHOR_INTERVAL == 0:
            physical_time = self._incident_next_iteration * self.dt
            self._incident_phase = _dft_phase_at_time(
                self.frequencies, physical_time, self.complex_dtype
            )
        else:
            self._incident_phase *= self._incident_step
        return multiplier

    def load_device_incident_electric(self, values: npt.ArrayLike) -> None:
        """Load an incident electric-field DFT downloaded at finalisation."""

        if self.associated_plane_wave is None or self._incident_electric is None:
            raise RuntimeError(
                f"KSIR monitor {self.name!r} has no associated incident plane wave"
            )
        incident = np.asarray(values, dtype=self.complex_dtype)
        if incident.shape != self._incident_electric.shape:
            raise ValueError(
                f"incident electric DFT has shape {incident.shape}, expected "
                f"{self._incident_electric.shape}"
            )
        self._incident_electric[...] = incident

    def observe_magnetic(
        self,
        iteration: int,
        Hx: npt.ArrayLike,
        Hy: npt.ArrayLike,
        Hz: npt.ArrayLike,
    ) -> None:
        fields = {"Hx": Hx, "Hy": Hy, "Hz": Hz}
        for component in MAGNETIC_COMPONENTS:
            if component in self._accumulators:
                self._accumulators[component].observe(iteration, fields[component])

    def device_sampling_multiplier(
        self, component: str, iteration: int
    ) -> npt.NDArray[np.complexfloating]:
        """Advance and return a component multiplier for device accumulation."""

        try:
            accumulator = self._accumulators[component]
        except KeyError as exc:
            raise ValueError(
                f"component {component!r} is not active in monitor {self.name!r}"
            ) from exc
        return accumulator.sampling_multiplier(iteration)

    def load_device_component_dfts(
        self,
        component: str,
        inside: npt.ArrayLike,
        outside: npt.ArrayLike,
    ) -> None:
        """Load raw DFTs downloaded from an accelerator backend."""

        try:
            accumulator = self._accumulators[component]
        except KeyError as exc:
            raise ValueError(
                f"component {component!r} is not active in monitor {self.name!r}"
            ) from exc
        accumulator.load_device_dfts(inside, outside)

    def _subtract_incident_surface(
        self, data: Mapping[str, KSIRComponentPhasors]
    ) -> Mapping[str, KSIRComponentPhasors]:
        if self.incident_surface_file is None:
            return data
        with h5py.File(self.incident_surface_file, "r") as source:
            monitor_path = f"ntff/{self.incident_monitor_name}"
            if monitor_path not in source:
                raise ValueError(
                    "incident surface file has no monitor "
                    f"{self.incident_monitor_name!r}"
                )
            monitor_group = source[monitor_path]
            for attr, expected in (
                ("phasor_time_sign", PHASOR_TIME_DEPENDENCE),
                ("forward_transform_sign", FORWARD_TRANSFORM_KERNEL),
                ("green_radial_sign", OUTGOING_GREEN_RADIAL_FACTOR),
            ):
                actual = monitor_group.attrs.get(attr)
                if isinstance(actual, bytes):
                    actual = actual.decode()
                if actual != expected:
                    raise ValueError(
                        f"incident surface convention {attr!r} is incompatible"
                    )
            if "surface" not in monitor_group:
                raise ValueError("incident monitor does not contain saved surface DFTs")

            subtracted: Dict[str, KSIRComponentPhasors] = {}
            for component, current in data.items():
                if component not in monitor_group["surface"]:
                    raise ValueError(
                        f"incident monitor has no {component} surface DFT"
                    )
                group = monitor_group["surface"][component]
                signature = group.attrs.get("compatibility_signature")
                if isinstance(signature, bytes):
                    signature = signature.decode()
                if signature != current.compatibility_signature:
                    raise ValueError(
                        f"incident {component} surface DFT is incompatible"
                    )
                field = np.asarray(current.field) - group["psi_dft"][:]
                derivative = (
                    np.asarray(current.normal_derivative) - group["dn_psi_dft"][:]
                )
                _readonly(field)
                _readonly(derivative)
                subtracted[component] = KSIRComponentPhasors(
                    component=component,
                    surface=current.surface,
                    field=field,
                    normal_derivative=derivative,
                    compatibility_signature=current.compatibility_signature,
                )
        return MappingProxyType(subtracted)

    def _cartesian(
        self,
        fields: Mapping[str, npt.NDArray[np.complexfloating]],
        components: Tuple[str, str, str],
    ) -> Optional[npt.NDArray[np.complexfloating]]:
        if not any(component in fields for component in components):
            return None
        shape = (self.frequencies.size, self.directions.shape[0], 3)
        cartesian = np.full(
            shape, np.nan + 1j * np.nan, dtype=self.complex_dtype
        )
        for axis, component in enumerate(components):
            if component in fields:
                cartesian[:, :, axis] = fields[component]
        return _readonly(cartesian)

    def _integrated_metrics(
        self, radiation_intensity: npt.NDArray[np.floating]
    ) -> tuple[
        Optional[npt.NDArray[np.floating]],
        Optional[npt.NDArray[np.floating]],
        Optional[npt.NDArray[np.floating]],
    ]:
        complete_sphere = (
            self.theta_values.size >= 2
            and self.phi_values.size >= 2
            and np.isclose(self.theta_values[0], 0)
            and np.isclose(self.theta_values[-1], 180)
            and np.isclose(self.phi_values[-1] - self.phi_values[0], 360)
        )
        if not complete_sphere:
            return None, None, None
        shape = (
            self.frequencies.size,
            self.theta_values.size,
            self.phi_values.size,
        )
        intensity_grid = radiation_intensity.reshape(shape)
        theta_rad = np.deg2rad(self.theta_values)
        phi_rad = np.deg2rad(self.phi_values)
        theta_weighted = intensity_grid * np.sin(theta_rad)[np.newaxis, :, np.newaxis]
        phi_integral = np.trapezoid(theta_weighted, phi_rad, axis=2)
        power = np.trapezoid(phi_integral, theta_rad, axis=1)
        directivity = np.full_like(radiation_intensity, np.nan)
        valid = power > 0
        directivity[valid] = (
            4 * np.pi * radiation_intensity[valid] / power[valid, np.newaxis]
        )
        maximum = np.full(power.shape, np.nan, dtype=self.real_dtype)
        maximum[valid] = np.max(directivity[valid], axis=1)
        return _readonly(power), _readonly(directivity), _readonly(maximum)

    def finalise(self) -> None:
        if self._finalised:
            return
        if self.wave_speed is None or self.impedance is None:
            raise RuntimeError(
                f"KSIR monitor {self.name!r} background material was not configured"
            )
        if (
            self.associated_plane_wave is not None
            and self._incident_next_iteration != self.iterations
        ):
            raise RuntimeError(
                f"KSIR incident observer received {self._incident_next_iteration} "
                f"of {self.iterations} expected samples"
            )

        raw_data = MappingProxyType(
            {
                component: accumulator.finalise(
                    self.background_er, self.background_mr
                )
                for component, accumulator in self._accumulators.items()
            }
        )
        data = self._subtract_incident_surface(raw_data)
        self._surface_data = data

        fields: Dict[str, npt.NDArray[np.complexfloating]] = {}
        face_contributions: Dict[str, npt.NDArray[np.complexfloating]] = {}
        cancellation_indicators: Dict[str, npt.NDArray[np.floating]] = {}
        active_areas: Dict[str, float] = {}
        missing_fractions: Dict[str, float] = {}
        for component, component_data in data.items():
            values, contributions = _evaluate_component_with_closure(
                component_data.surface,
                component_data.field,
                component_data.normal_derivative,
                self.closure,
                self.frequencies,
                self.directions,
                self.wave_speed,
                self.origin,
                self.nthreads,
            )
            fields[component] = values
            face_contributions[component] = contributions
            numerator = np.sum(np.abs(contributions), axis=2)
            denominator = np.abs(values)
            indicator = np.full(
                denominator.shape, np.inf, dtype=self.real_dtype
            )
            np.divide(
                numerator,
                denominator,
                out=indicator,
                where=denominator > 0,
            )
            cancellation_indicators[component] = _readonly(indicator)
            active_areas[component] = float(
                np.sum(component_data.surface.area_weights)
            )
            extents = (
                component_data.surface.physical_upper
                - component_data.surface.physical_lower
            )
            full_area = 2 * (
                extents[0] * extents[1]
                + extents[0] * extents[2]
                + extents[1] * extents[2]
            )
            missing_fractions[component] = (
                max(0.0, 1.0 - active_areas[component] / full_area)
                if self.closure.name == "experimental_mask"
                else 0.0
            )
        field_mapping = MappingProxyType(fields)

        electric_cartesian = self._cartesian(fields, ELECTRIC_COMPONENTS)
        magnetic_cartesian = self._cartesian(fields, MAGNETIC_COMPONENTS)
        complete_electric = all(
            component in fields for component in ELECTRIC_COMPONENTS
        )
        complete_magnetic = all(
            component in fields for component in MAGNETIC_COMPONENTS
        )
        electric_spherical = (
            None
            if not complete_electric
            else _readonly(
                project_cartesian_to_spherical(
                    electric_cartesian, self.theta, self.phi, degrees=True
                )
            )
        )
        magnetic_spherical = (
            None
            if not complete_magnetic
            else _readonly(
                project_cartesian_to_spherical(
                    magnetic_cartesian, self.theta, self.phi, degrees=True
                )
            )
        )

        radiation_intensity = None
        transversality_error = None
        radiated_power = None
        directivity = None
        maximum_directivity = None
        bistatic_rcs = None
        if complete_electric:
            tangential_squared = (
                np.abs(electric_spherical[:, :, 1]) ** 2
                + np.abs(electric_spherical[:, :, 2]) ** 2
            )
            radiation_intensity = _readonly(
                np.asarray(
                    0.5 * tangential_squared / self.impedance,
                    dtype=self.real_dtype,
                )
            )
            tangential = np.sqrt(tangential_squared)
            transversality_error = np.full(
                tangential.shape, np.nan, dtype=self.real_dtype
            )
            nonzero = tangential > 0
            transversality_error[nonzero] = (
                np.abs(electric_spherical[:, :, 0][nonzero])
                / tangential[nonzero]
            )
            _readonly(transversality_error)
            if self.closure.mathematically_closed:
                radiated_power, directivity, maximum_directivity = (
                    self._integrated_metrics(radiation_intensity)
                )
            if self._incident_electric is not None:
                incident_power = np.sum(np.abs(self._incident_electric) ** 2, axis=1)
                bistatic_rcs = np.full(
                    tangential_squared.shape, np.nan, dtype=self.real_dtype
                )
                valid = incident_power > 0
                bistatic_rcs[valid] = (
                    4
                    * np.pi
                    * tangential_squared[valid]
                    / incident_power[valid, np.newaxis]
                )
                _readonly(bistatic_rcs)

        incident_electric = self._incident_electric
        if incident_electric is not None:
            _readonly(incident_electric)
        self._result = KSIRFrequencyResult(
            name=self.name,
            frequencies=self.frequencies,
            theta=self.theta,
            phi=self.phi,
            directions=self.directions,
            origin=self.origin,
            range_normalized_fields=field_mapping,
            electric_cartesian=electric_cartesian,
            electric_spherical=electric_spherical,
            magnetic_cartesian=magnetic_cartesian,
            magnetic_spherical=magnetic_spherical,
            radiation_intensity=radiation_intensity,
            transversality_error=transversality_error,
            radiated_power=radiated_power,
            directivity=directivity,
            maximum_directivity=maximum_directivity,
            incident_electric=incident_electric,
            bistatic_rcs=bistatic_rcs,
            face_contributions=MappingProxyType(face_contributions),
            cancellation_indicator=MappingProxyType(cancellation_indicators),
            active_area=MappingProxyType(active_areas),
            missing_area_fraction=MappingProxyType(missing_fractions),
            closure=self.closure.name,
            mathematically_closed=self.closure.mathematically_closed,
        )
        self._finalised = True

    def write_hdf5(self, base_group) -> None:
        """Write results and optionally reusable surface DFTs."""

        result = self.result
        ntff_group = base_group.require_group("ntff")
        if self.name in ntff_group:
            raise ValueError(f"duplicate NTFF output group {self.name!r}")
        group = ntff_group.create_group(self.name)
        group.attrs["formulation"] = "KSIR"
        group.attrs["phasor_time_sign"] = PHASOR_TIME_DEPENDENCE
        group.attrs["forward_transform_sign"] = FORWARD_TRANSFORM_KERNEL
        group.attrs["green_radial_sign"] = OUTGOING_GREEN_RADIAL_FACTOR
        group.attrs["background_eps_r"] = self.background_er
        group.attrs["background_mu_r"] = self.background_mr
        group.attrs["background_material"] = self.background_material_name
        group.attrs["background_material_id"] = self.background_material_id
        group.attrs["wave_speed"] = self.wave_speed
        group.attrs["impedance"] = self.impedance
        group.attrs["origin"] = self.origin
        first_surface = next(iter(self.surfaces.values()))
        group.attrs["logical_bounds"] = (*first_surface.lower, *first_surface.upper)
        group.attrs["closure"] = self.closure.name
        group.attrs["mathematically_closed"] = self.closure.mathematically_closed
        group.attrs["closure_exact"] = self.closure.exact
        group.attrs["omitted_faces"] = np.asarray(
            self.closure.omitted_faces, dtype="S5"
        )
        group.attrs["symmetry_plane_faces"] = np.asarray(
            [plane.face for plane in self.closure.symmetry_planes], dtype="S5"
        )
        group.attrs["symmetry_plane_types"] = np.asarray(
            [plane.boundary_type for plane in self.closure.symmetry_planes],
            dtype="S3",
        )
        group.attrs["symmetry_plane_coordinates"] = np.asarray(
            [plane.coordinate for plane in self.closure.symmetry_planes],
            dtype=self.real_dtype,
        )
        group.attrs["symmetry_image_count"] = self.closure.image_count
        group.attrs["precision"] = self.precision
        group.attrs["real_dtype"] = self.real_dtype.name
        group.attrs["complex_dtype"] = self.complex_dtype.name
        group.attrs["window"] = self.window_name
        group.attrs["solver"] = self.solver_backend
        group.attrs["collection_backend"] = self.collection_backend
        group.attrs["phase_reanchor_interval"] = DFT_PHASE_REANCHOR_INTERVAL
        if self.solver_backend == "cpu":
            group.attrs["openmp_threads"] = self.nthreads
        group.attrs["dt"] = self.dt
        group.attrs["iterations"] = self.iterations
        group.attrs["components"] = np.asarray(self.components, dtype="S2")
        group.attrs["sample_time_offsets"] = np.asarray(
            [
                0.0 if item in ELECTRIC_COMPONENTS else 0.5 * self.dt
                for item in self.components
            ]
        )
        group.attrs["incident_surface_subtracted"] = (
            self.incident_surface_file is not None
        )
        group.attrs["allow_external_sources"] = self.allow_external_sources
        if self.incident_surface_file is not None:
            group.attrs["incident_surface_file"] = str(self.incident_surface_file)
            group.attrs["incident_monitor_name"] = self.incident_monitor_name

        group["frequencies"] = result.frequencies
        group["theta"] = result.theta
        group["phi"] = result.phi
        group["theta_values"] = self.theta_values
        group["phi_values"] = self.phi_values
        group["directions"] = result.directions
        range_group = group.create_group("range_normalized_fields")
        for component, values in result.range_normalized_fields.items():
            range_group[component] = values
        contribution_group = group.create_group("face_contributions")
        cancellation_group = group.create_group("cancellation_indicator")
        diagnostics_group = group.create_group("closure_diagnostics")
        diagnostics_group.attrs["face_order"] = np.asarray(FACES, dtype="S5")
        for component in self.components:
            contribution_group[component] = result.face_contributions[component]
            cancellation_group[component] = result.cancellation_indicator[component]
            component_group = diagnostics_group.create_group(component)
            component_group.attrs["active_area"] = result.active_area[component]
            component_group.attrs["missing_area_fraction"] = (
                result.missing_area_fraction[component]
            )
        for name, values in (
            ("E_cartesian", result.electric_cartesian),
            ("E_spherical", result.electric_spherical),
            ("H_cartesian", result.magnetic_cartesian),
            ("H_spherical", result.magnetic_spherical),
            ("radiation_intensity", result.radiation_intensity),
            ("transversality_error", result.transversality_error),
            ("Prad", result.radiated_power),
            ("directivity", result.directivity),
            ("Dmax", result.maximum_directivity),
            ("incident_E", result.incident_electric),
            ("bistatic_rcs", result.bistatic_rcs),
        ):
            if values is not None:
                group[name] = values

        if self.plane_wave_metadata is not None:
            plane_wave_group = group.create_group("plane_wave")
            for name, value in self.plane_wave_metadata.items():
                plane_wave_group.attrs[name] = value

        if self.save_surface_dft:
            surface_group = group.create_group("surface")
            for component, data in self.surface_data.items():
                component_group = surface_group.create_group(component)
                component_group.attrs["compatibility_signature"] = (
                    data.compatibility_signature
                )
                component_group.attrs["logical_lower"] = data.surface.lower
                component_group.attrs["logical_upper"] = data.surface.upper
                component_group.attrs["physical_lower"] = data.surface.physical_lower
                component_group.attrs["physical_upper"] = data.surface.physical_upper
                component_group.attrs["grid_spacing"] = data.surface.grid_spacing
                component_group.attrs["field_shape"] = data.surface.field_shape
                component_group["patch_xyz"] = data.surface.patch_positions
                component_group["normals"] = data.surface.normals
                component_group["area_weight"] = data.surface.area_weights
                component_group["psi_dft"] = data.field
                component_group["dn_psi_dft"] = data.normal_derivative


def evaluate_saved_surface_dft(
    filename: Path | str,
    monitor_name: str,
    theta: npt.ArrayLike,
    phi: npt.ArrayLike,
    *,
    origin: Optional[npt.ArrayLike] = None,
) -> KSIRSavedFarField:
    """Evaluate new directions from a saved production KSIR surface DFT."""

    with h5py.File(filename, "r") as source:
        path = f"ntff/{monitor_name}"
        if path not in source:
            raise ValueError(f"saved file has no NTFF monitor {monitor_name!r}")
        group = source[path]
        for attr, expected in (
            ("phasor_time_sign", PHASOR_TIME_DEPENDENCE),
            ("forward_transform_sign", FORWARD_TRANSFORM_KERNEL),
            ("green_radial_sign", OUTGOING_GREEN_RADIAL_FACTOR),
        ):
            actual = group.attrs.get(attr)
            if isinstance(actual, bytes):
                actual = actual.decode()
            if actual != expected:
                raise ValueError(f"saved surface convention {attr!r} is incompatible")
        if "surface" not in group:
            raise ValueError("saved monitor does not contain surface DFTs")
        stored_real_dtype = group.attrs.get(
            "real_dtype", group["frequencies"].dtype.name
        )
        if isinstance(stored_real_dtype, bytes):
            stored_real_dtype = stored_real_dtype.decode()
        real_dtype = np.dtype(stored_real_dtype)
        theta_values = _angles("theta", theta, real_dtype)
        phi_values = _angles("phi", phi, real_dtype)
        theta_grid, phi_grid = np.meshgrid(
            theta_values, phi_values, indexing="ij"
        )
        paired_theta = _readonly(theta_grid.ravel())
        paired_phi = _readonly(phi_grid.ravel())
        directions = _readonly(
            spherical_directions(paired_theta, paired_phi, degrees=True)
        )
        closure_name = group.attrs.get("closure", "closed")
        if isinstance(closure_name, bytes):
            closure_name = closure_name.decode()

        def decoded_strings(name):
            values = np.atleast_1d(group.attrs.get(name, ()))
            return tuple(
                value.decode() if isinstance(value, bytes) else str(value)
                for value in values
            )

        closure = closure_from_metadata(
            closure_name,
            decoded_strings("omitted_faces"),
            decoded_strings("symmetry_plane_faces"),
            decoded_strings("symmetry_plane_types"),
            np.atleast_1d(
                group.attrs.get("symmetry_plane_coordinates", ())
            ),
        )
        frequencies = _readonly(group["frequencies"][:])
        reference_origin = np.asarray(
            group.attrs["origin"] if origin is None else origin,
            dtype=real_dtype,
        )
        if reference_origin.shape != (3,) or not np.all(np.isfinite(reference_origin)):
            raise ValueError("origin must contain exactly three finite values")
        _readonly(reference_origin)
        wave_speed = float(group.attrs["wave_speed"])
        fields: Dict[str, npt.NDArray[np.complexfloating]] = {}
        for component, component_group in group["surface"].items():
            surface = closure.apply_quadrature(
                build_component_surface(
                    component,
                    component_group.attrs["logical_lower"],
                    component_group.attrs["logical_upper"],
                    component_group.attrs["grid_spacing"],
                    component_group.attrs["field_shape"],
                    excluded_faces=closure.omitted_faces,
                    real_dtype=real_dtype,
                )
            )
            if not np.array_equal(
                surface.patch_positions, component_group["patch_xyz"][:]
            ):
                raise ValueError(
                    f"saved {component} surface geometry is inconsistent"
                )
            values, _ = _evaluate_component_with_closure(
                surface,
                component_group["psi_dft"][:],
                component_group["dn_psi_dft"][:],
                closure,
                frequencies,
                directions,
                wave_speed,
                reference_origin,
            )
            fields[component] = values

    return KSIRSavedFarField(
        frequencies=frequencies,
        theta=paired_theta,
        phi=paired_phi,
        directions=directions,
        origin=reference_origin,
        range_normalized_fields=MappingProxyType(fields),
        closure=closure.name,
        mathematically_closed=closure.mathematically_closed,
    )
