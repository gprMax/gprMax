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

"""Advanced-time KSIR field extension for CPU and device collectors."""

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Deque, Dict, Mapping, Tuple

import numpy as np
import numpy.typing as npt
from scipy.constants import c

from .closures import ResolvedKSIRClosure
from .surfaces import KSIRComponentSurface

try:
    from gprMax.cython.ntff import (
        deposit_time_domain_surface as _deposit_time_domain_surface,
        gather_time_domain_surface as _gather_time_domain_surface,
    )
except ImportError:  # Source-tree use before extensions are rebuilt.
    _deposit_time_domain_surface = None
    _gather_time_domain_surface = None


ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")
MAGNETIC_COMPONENTS = ("Hx", "Hy", "Hz")


@dataclass(frozen=True)
class KSIRTimeDomainResult:
    """In-memory time histories reconstructed at exterior points."""

    name: str
    times: npt.NDArray[np.floating]
    points: npt.NDArray[np.floating]
    fields: Mapping[str, npt.NDArray[np.floating]]
    sample_time_offsets: Mapping[str, float]
    time_origin: str
    time_origins: npt.NDArray[np.floating]
    valid_lengths: npt.NDArray[np.int64]
    collection_backend: str
    closure: str
    mathematically_closed: bool

    def point_times(self, point_index: int) -> npt.NDArray[np.floating]:
        """Return physical times for one point's valid output samples."""

        length = int(self.valid_lengths[point_index])
        return self.time_origins[point_index] + self.times[:length]

    def point_field(
        self, component: str, point_index: int
    ) -> npt.NDArray[np.floating]:
        """Return one point's valid field trace without padded trailing bins."""

        length = int(self.valid_lengths[point_index])
        return self.fields[component][point_index, :length]


class _ComponentAccumulator:
    """Streaming time derivative and advanced-time deposition for one component."""

    def __init__(
        self,
        surface: KSIRComponentSurface,
        points: npt.NDArray[np.floating],
        dt: float,
        iterations: int,
        wave_speed: float,
        sample_time_offset_steps: float,
        real_dtype,
        nthreads: int,
        closure: ResolvedKSIRClosure,
    ):
        self.surface = surface
        self.points = points
        self.dt = dt
        self.iterations = iterations
        self.wave_speed = wave_speed
        self.sample_time_offset_steps = sample_time_offset_steps
        self.real_dtype = np.dtype(real_dtype)
        self.nthreads = nthreads
        self._recent: Deque[Tuple[int, npt.NDArray, npt.NDArray]] = deque(maxlen=3)
        self._next_iteration = 0
        self._last_deposited = -1
        self._finalised = False

        base_positions = surface.patch_positions
        base_normals = surface.normals
        base_areas = surface.area_weights
        base_indices = np.arange(surface.npatches, dtype=np.int64)
        positions = []
        normals = []
        areas = []
        source_patch_indices = []
        parities = []
        for image in closure.component_images(surface.component):
            image_positions, image_normals = image.transform(
                base_positions, base_normals
            )
            positions.append(image_positions)
            normals.append(image_normals)
            areas.append(base_areas)
            source_patch_indices.append(base_indices)
            parities.append(
                np.full(surface.npatches, image.parity, dtype=self.real_dtype)
            )
        positions = np.concatenate(positions)
        normals = np.concatenate(normals)
        areas = np.concatenate(areas)
        parity = np.concatenate(parities)
        self._source_patch_index = np.ascontiguousarray(
            np.concatenate(source_patch_indices), dtype=np.int64
        )
        # Point validation must use the rectangular support of the surface
        # patches, not only their centre coordinates.  On a symmetry axis the
        # physical half-surface ends at the reflection plane; reflecting that
        # clipped cuboid gives the support of the completed surface.
        support_lower = surface.physical_lower.copy()
        support_upper = surface.physical_upper.copy()
        for plane in closure.symmetry_planes:
            if plane.face.endswith("0"):
                support_lower[plane.axis] = plane.coordinate
            else:
                support_upper[plane.axis] = plane.coordinate
        support_axes = [
            np.asarray((lower, upper), dtype=self.real_dtype)
            for lower, upper in zip(support_lower, support_upper)
        ]
        support_corners = np.stack(
            np.meshgrid(*support_axes, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        completed_corners = []
        corner_normals = np.zeros_like(support_corners)
        for image in closure.component_images(surface.component):
            image_corners, _ = image.transform(support_corners, corner_normals)
            completed_corners.append(image_corners)
        completed_corners = np.concatenate(completed_corners)
        self.completed_physical_lower = np.min(completed_corners, axis=0)
        self.completed_physical_upper = np.max(completed_corners, axis=0)
        displacement = points[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distance = np.linalg.norm(displacement, axis=2)
        if np.any(distance == 0):
            raise ValueError(
                "observation points must not coincide with surface patches"
            )
        direction = displacement / distance[:, :, np.newaxis]
        normal_projection = np.sum(normals[np.newaxis, :, :] * direction, axis=2)

        self._normal_derivative_weight = np.ascontiguousarray(
            -areas[np.newaxis, :] * parity[np.newaxis, :]
            / (4 * np.pi * distance),
            dtype=self.real_dtype,
        )
        self._field_weight = np.ascontiguousarray(
            areas[np.newaxis, :]
            * parity[np.newaxis, :]
            * normal_projection
            / (4 * np.pi * distance**2),
            dtype=self.real_dtype,
        )
        self._time_derivative_weight = np.ascontiguousarray(
            areas[np.newaxis, :]
            * parity[np.newaxis, :]
            * normal_projection
            / (4 * np.pi * wave_speed * distance),
            dtype=self.real_dtype,
        )

        delay = sample_time_offset_steps + distance / (wave_speed * dt)
        self._integer_delay = np.ascontiguousarray(
            np.floor(delay), dtype=np.int64
        )
        self._fractional_delay = np.ascontiguousarray(
            delay - self._integer_delay, dtype=self.real_dtype
        )
        self.minimum_delay_steps = np.min(delay, axis=1)
        self.maximum_integer_delay_steps = np.max(self._integer_delay, axis=1)
        self._inside_indices = np.ascontiguousarray(
            np.concatenate(
                [face.inside_flat_indices for face in self.surface.faces]
            ),
            dtype=np.int64,
        )
        self._outside_indices = np.ascontiguousarray(
            np.concatenate(
                [face.outside_flat_indices for face in self.surface.faces]
            ),
            dtype=np.int64,
        )
        self._normal_spacing = np.ascontiguousarray(
            np.concatenate(
                [
                    np.full(face.npatches, face.normal_spacing)
                    for face in self.surface.faces
                ]
            ),
            dtype=self.real_dtype,
        )
        self._surface_buffer = np.empty(
            self.surface.npatches, dtype=self.real_dtype
        )
        self._derivative_buffer = np.empty_like(self._surface_buffer)
        self._time_origin_steps: npt.NDArray[np.int64]
        self.output: npt.NDArray[np.floating] | None

    def allocate(
        self,
        output_length: int,
        time_origin_steps: npt.NDArray[np.int64],
        *,
        allocate_output: bool = True,
    ) -> None:
        self._time_origin_steps = np.ascontiguousarray(
            time_origin_steps, dtype=np.int64
        )
        if allocate_output:
            self.output = np.zeros(
                (self.points.shape[0], output_length), dtype=self.real_dtype
            )
        else:
            self.output = None

    def _surface_values(
        self, field: npt.ArrayLike
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        values_array = np.asarray(field)
        if values_array.shape != self.surface.field_shape:
            raise ValueError(
                f"field shape {values_array.shape} does not match surface field shape "
                f"{self.surface.field_shape}"
            )
        if _gather_time_domain_surface is not None:
            flat = np.ascontiguousarray(
                values_array, dtype=self.real_dtype
            ).ravel()
            _gather_time_domain_surface(
                self.nthreads,
                self._inside_indices,
                self._outside_indices,
                self._normal_spacing,
                flat,
                self._surface_buffer,
                self._derivative_buffer,
            )
            return self._surface_buffer.copy(), self._derivative_buffer.copy()

        values = []
        derivatives = []
        for face in self.surface.faces:
            inside, outside = face.sample(field)
            surface_value, normal_derivative = face.collocate(inside, outside)
            values.append(np.asarray(surface_value, dtype=self.real_dtype))
            derivatives.append(np.asarray(normal_derivative, dtype=self.real_dtype))
        return np.concatenate(values), np.concatenate(derivatives)

    def _deposit(
        self,
        sample_index: int,
        surface_value: npt.NDArray,
        normal_derivative: npt.NDArray,
        time_derivative: npt.NDArray,
    ) -> None:
        if sample_index != self._last_deposited + 1:
            raise RuntimeError(
                "KSIR samples must be deposited exactly once in time order"
            )

        if _deposit_time_domain_surface is not None:
            _deposit_time_domain_surface(
                self.nthreads,
                sample_index,
                np.ascontiguousarray(surface_value, dtype=self.real_dtype),
                np.ascontiguousarray(normal_derivative, dtype=self.real_dtype),
                np.ascontiguousarray(time_derivative, dtype=self.real_dtype),
                self._source_patch_index,
                self._normal_derivative_weight,
                self._field_weight,
                self._time_derivative_weight,
                self._integer_delay,
                self._fractional_delay,
                self._time_origin_steps,
                self.output,
            )
            self._last_deposited = sample_index
            return

        source = self._source_patch_index
        integrand = (
            self._normal_derivative_weight
            * normal_derivative[source][np.newaxis, :]
            + self._field_weight * surface_value[source][np.newaxis, :]
            + self._time_derivative_weight
            * time_derivative[source][np.newaxis, :]
        )
        destination = (
            sample_index
            + self._integer_delay
            - self._time_origin_steps[:, np.newaxis]
        )
        for point_index in range(self.points.shape[0]):
            np.add.at(
                self.output[point_index],
                destination[point_index],
                (1 - self._fractional_delay[point_index]) * integrand[point_index],
            )
            np.add.at(
                self.output[point_index],
                destination[point_index] + 1,
                self._fractional_delay[point_index] * integrand[point_index],
            )
        self._last_deposited = sample_index

    def observe(self, iteration: int, field: npt.ArrayLike) -> None:
        """Observe one completed component time level."""

        if self._finalised:
            raise RuntimeError("cannot observe a finalised KSIR accumulator")
        if iteration != self._next_iteration:
            raise ValueError(
                f"expected KSIR iteration {self._next_iteration}, received {iteration}"
            )

        surface_value, normal_derivative = self._surface_values(field)
        self._recent.append((iteration, surface_value, normal_derivative))
        self._next_iteration += 1

        if len(self._recent) < 3:
            return

        previous, centre, following = self._recent
        if iteration == 2:
            forward_derivative = (
                -3 * previous[1] + 4 * centre[1] - following[1]
            ) / (2 * self.dt)
            self._deposit(previous[0], previous[1], previous[2], forward_derivative)

        centred_derivative = (following[1] - previous[1]) / (2 * self.dt)
        self._deposit(centre[0], centre[1], centre[2], centred_derivative)

    def finalise(self) -> None:
        """Deposit the final endpoint using a second-order backward derivative."""

        if self._finalised:
            return
        if self._next_iteration != self.iterations:
            raise RuntimeError(
                f"KSIR component {self.surface.component} received "
                f"{self._next_iteration} "
                f"of {self.iterations} expected samples"
            )

        if len(self._recent) == 1:
            only = self._recent[0]
            self._deposit(only[0], only[1], only[2], np.zeros_like(only[1]))
        elif len(self._recent) == 2:
            first, last = self._recent
            derivative = (last[1] - first[1]) / self.dt
            self._deposit(first[0], first[1], first[2], derivative)
            self._deposit(last[0], last[1], last[2], derivative)
        elif len(self._recent) == 3:
            before_previous, previous, last = self._recent
            backward_derivative = (
                3 * last[1] - 4 * previous[1] + before_previous[1]
            ) / (2 * self.dt)
            self._deposit(last[0], last[1], last[2], backward_derivative)

        self.output.setflags(write=False)
        self._finalised = True


class KSIRTimeDomainMonitor:
    """Advanced-time field reconstruction at explicit exterior points.

    CPU collection uses configured-dtype Cython/OpenMP kernels, with a NumPy
    fallback before the extensions are compiled. Accelerator collection and
    compact output storage remain device-resident until finalisation.
    """

    def __init__(
        self,
        name: str,
        surfaces: Mapping[str, KSIRComponentSurface],
        points: npt.ArrayLike,
        dt: float,
        iterations: int,
        *,
        real_dtype,
        wave_speed: float = c,
        nthreads: int = 1,
        time_origin: str = "simulation",
        device_backend: str | None = None,
        closure: ResolvedKSIRClosure | None = None,
    ):
        if not name:
            raise ValueError("KSIR monitor name must not be empty")
        if not surfaces:
            raise ValueError("at least one component surface is required")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and greater than zero")
        if not isinstance(iterations, (int, np.integer)) or iterations <= 0:
            raise ValueError("iterations must be an integer greater than zero")
        if not np.isfinite(wave_speed) or wave_speed <= 0:
            raise ValueError("wave_speed must be finite and greater than zero")
        if not isinstance(nthreads, (int, np.integer)) or nthreads < 1:
            raise ValueError("nthreads must be an integer greater than zero")
        if not isinstance(time_origin, str) or time_origin not in (
            "simulation",
            "first_arrival",
        ):
            raise ValueError(
                "time_origin must be 'simulation' or 'first_arrival'"
            )
        if device_backend not in (None, "cuda", "opencl", "metal"):
            raise ValueError(
                "device_backend must be None, 'cuda', 'opencl', or 'metal'"
            )
        self.real_dtype = np.dtype(real_dtype)
        if self.real_dtype.kind != "f":
            raise ValueError("real_dtype must be a floating-point dtype")
        point_array = np.asarray(points, dtype=self.real_dtype)
        if point_array.ndim == 1:
            point_array = point_array[np.newaxis, :]
        if (
            point_array.ndim != 2
            or point_array.shape[0] == 0
            or point_array.shape[1] != 3
        ):
            raise ValueError("points must have shape (npoints, 3)")
        if not np.all(np.isfinite(point_array)):
            raise ValueError("points must contain only finite values")

        components = tuple(surfaces)
        unknown = set(components) - set(ELECTRIC_COMPONENTS + MAGNETIC_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown field components: {sorted(unknown)}")

        self.name = name
        self.points = point_array.copy()
        self.points.setflags(write=False)
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.wave_speed = float(wave_speed)
        self.nthreads = int(nthreads)
        self.time_origin = time_origin
        self.device_backend = device_backend
        if device_backend is None:
            self.collection_backend = (
                "cython_openmp"
                if _gather_time_domain_surface is not None
                and _deposit_time_domain_surface is not None
                else "numpy_fallback"
            )
        else:
            self.collection_backend = f"{device_backend}_device"
        self.components = components
        self.surfaces = MappingProxyType(dict(surfaces))
        self.closure = closure or ResolvedKSIRClosure(
            "closed", (), (), True, True
        )
        if not self.closure.mathematically_closed:
            raise ValueError(
                "advanced-time KSIR requires a closed or symmetry-completed surface"
            )
        for surface in surfaces.values():
            face_ids = tuple(face.face_id for face in surface.faces)
            if face_ids != self.closure.active_faces:
                raise ValueError(
                    f"{surface.component} surface faces {face_ids} do not match "
                    f"closure faces {self.closure.active_faces}"
                )
        self._accumulators: Dict[str, _ComponentAccumulator] = {}
        self._finalised = False
        self._result = None
        self.surface_material_id = None

        for component, surface in surfaces.items():
            offset_steps = 0.0 if component in ELECTRIC_COMPONENTS else 0.5
            self._accumulators[component] = _ComponentAccumulator(
                surface,
                self.points,
                self.dt,
                self.iterations,
                self.wave_speed,
                offset_steps,
                self.real_dtype,
                self.nthreads,
                self.closure,
            )

        for component, accumulator in self._accumulators.items():
            completed_scale = max(
                1.0,
                np.max(np.abs(accumulator.completed_physical_lower)),
                np.max(np.abs(accumulator.completed_physical_upper)),
            )
            tolerance = 10 * np.finfo(self.real_dtype).eps * completed_scale
            on_or_inside = np.all(
                (
                    point_array
                    >= accumulator.completed_physical_lower - tolerance
                )
                & (
                    point_array
                    <= accumulator.completed_physical_upper + tolerance
                ),
                axis=1,
            )
            if np.any(on_or_inside):
                raise ValueError(
                    "observation points must be strictly outside the completed "
                    f"{component} surface"
                )

        minimum_delay = np.min(
            np.stack(
                [
                    accumulator.minimum_delay_steps
                    for accumulator in self._accumulators.values()
                ]
            ),
            axis=0,
        )
        maximum_integer_delay = np.max(
            np.stack(
                [
                    accumulator.maximum_integer_delay_steps
                    for accumulator in self._accumulators.values()
                ]
            ),
            axis=0,
        )
        if self.time_origin == "first_arrival":
            time_origin_steps = np.floor(minimum_delay).astype(np.int64)
        else:
            time_origin_steps = np.zeros(self.points.shape[0], dtype=np.int64)
        valid_lengths = (
            self.iterations
            + maximum_integer_delay
            - time_origin_steps
            + 1
        )
        self.time_origin_steps = np.ascontiguousarray(
            time_origin_steps, dtype=np.int64
        )
        self.time_origin_steps.setflags(write=False)
        self.time_origins = np.asarray(
            self.time_origin_steps * self.dt, dtype=self.real_dtype
        )
        self.time_origins.setflags(write=False)
        self.valid_lengths = np.ascontiguousarray(valid_lengths, dtype=np.int64)
        self.valid_lengths.setflags(write=False)
        self.output_length = int(np.max(self.valid_lengths))
        for accumulator in self._accumulators.values():
            accumulator.allocate(
                self.output_length,
                self.time_origin_steps,
                allocate_output=self.device_backend is None,
            )

    @property
    def result(self) -> KSIRTimeDomainResult:
        if self._result is None:
            raise RuntimeError(
                "KSIR result is not available until the solver has finalised"
            )
        return self._result

    def validate_materials(
        self, material_ids: npt.ArrayLike, id_lookup: Mapping[str, int]
    ) -> int:
        """Verify that all straddling samples use one homogeneous material ID."""

        ids = np.asarray(material_ids)
        sampled_ids = []
        for component, surface in self.surfaces.items():
            component_ids = ids[id_lookup[component]]
            for face in surface.faces:
                keep = np.ones(face.npatches, dtype=bool)
                for plane in self.closure.symmetry_planes:
                    tolerance = (
                        16
                        * np.finfo(self.real_dtype).eps
                        * max(
                            abs(plane.coordinate),
                            surface.grid_spacing[plane.axis],
                        )
                    )
                    keep &= ~np.isclose(
                        face.patch_positions[:, plane.axis],
                        plane.coordinate,
                        rtol=0,
                        atol=tolerance,
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
        return self.surface_material_id

    def observe_electric(
        self,
        iteration: int,
        Ex: npt.ArrayLike,
        Ey: npt.ArrayLike,
        Ez: npt.ArrayLike,
    ) -> None:
        if self.device_backend is not None:
            raise RuntimeError("device KSIR monitors must be observed on the device")
        fields = {"Ex": Ex, "Ey": Ey, "Ez": Ez}
        for component in ELECTRIC_COMPONENTS:
            if component in self._accumulators:
                self._accumulators[component].observe(iteration, fields[component])

    def observe_magnetic(
        self,
        iteration: int,
        Hx: npt.ArrayLike,
        Hy: npt.ArrayLike,
        Hz: npt.ArrayLike,
    ) -> None:
        if self.device_backend is not None:
            raise RuntimeError("device KSIR monitors must be observed on the device")
        fields = {"Hx": Hx, "Hy": Hy, "Hz": Hz}
        for component in MAGNETIC_COMPONENTS:
            if component in self._accumulators:
                self._accumulators[component].observe(iteration, fields[component])

    def finalise(self) -> None:
        if self._finalised:
            return
        if self.device_backend is None:
            for accumulator in self._accumulators.values():
                accumulator.finalise()
        elif any(
            accumulator.output is None
            for accumulator in self._accumulators.values()
        ):
            raise RuntimeError("not all device KSIR component outputs were loaded")

        times = self.dt * np.arange(self.output_length, dtype=self.real_dtype)
        times.setflags(write=False)
        fields = MappingProxyType(
            {
                component: self._accumulators[component].output
                for component in self.components
            }
        )
        offsets = MappingProxyType(
            {
                component: (0.0 if component in ELECTRIC_COMPONENTS else 0.5 * self.dt)
                for component in self.components
            }
        )
        self._result = KSIRTimeDomainResult(
            name=self.name,
            times=times,
            points=self.points,
            fields=fields,
            sample_time_offsets=offsets,
            time_origin=self.time_origin,
            time_origins=self.time_origins,
            valid_lengths=self.valid_lengths,
            collection_backend=self.collection_backend,
            closure=self.closure.name,
            mathematically_closed=self.closure.mathematically_closed,
        )
        self._finalised = True

    def load_device_component_output(
        self, component: str, output: npt.ArrayLike
    ) -> None:
        """Attach one configured-dtype history downloaded at finalisation."""

        if self.device_backend is None:
            raise RuntimeError("CPU KSIR monitors do not accept device output")
        if self._finalised:
            raise RuntimeError("cannot load output into a finalised KSIR monitor")
        if component not in self._accumulators:
            raise ValueError(f"component {component!r} is not monitored")
        values = np.asarray(output)
        expected_shape = (self.points.shape[0], self.output_length)
        if values.shape != expected_shape:
            raise ValueError(
                f"device output for {component} must have shape {expected_shape}"
            )
        if values.dtype != self.real_dtype:
            raise ValueError(
                f"device output for {component} must use dtype {self.real_dtype}"
            )
        values = np.ascontiguousarray(values)
        values.setflags(write=False)
        self._accumulators[component].output = values

    def write_hdf5(self, base_group) -> None:
        """Write compact time-domain histories to the normal model output."""

        result = self.result
        ntff_group = base_group.require_group("ntff")
        if self.name in ntff_group:
            raise ValueError(f"duplicate NTFF output group {self.name!r}")
        group = ntff_group.create_group(self.name)
        first_surface = next(iter(self.surfaces.values()))
        group.attrs["formulation"] = "KSIR"
        group.attrs["output_type"] = "time_domain_field_extension"
        group.attrs["domain"] = "time"
        group.attrs["mathematically_closed"] = self.closure.mathematically_closed
        group.attrs["closure"] = self.closure.name
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
        group.attrs["logical_bounds"] = (
            *first_surface.lower,
            *first_surface.upper,
        )
        group.attrs["wave_speed"] = self.wave_speed
        group.attrs["precision"] = (
            "single" if self.real_dtype.itemsize == 4 else "double"
        )
        group.attrs["real_dtype"] = self.real_dtype.name
        group.attrs["solver"] = self.device_backend or "cpu"
        group.attrs["collection_backend"] = self.collection_backend
        if self.device_backend is None:
            group.attrs["openmp_threads"] = self.nthreads
        group.attrs["dt"] = self.dt
        group.attrs["iterations"] = self.iterations
        group.attrs["components"] = np.asarray(self.components, dtype="S2")
        group.attrs["sample_time_offsets"] = np.asarray(
            [result.sample_time_offsets[item] for item in self.components],
            dtype=self.real_dtype,
        )
        group.attrs["time_origin"] = self.time_origin
        if self.surface_material_id is not None:
            group.attrs["background_material_id"] = self.surface_material_id

        group["points"] = result.points
        group["times"] = result.times
        group["time_origins"] = result.time_origins
        group["valid_lengths"] = result.valid_lengths
        fields_group = group.create_group("fields")
        for component, values in result.fields.items():
            fields_group[component] = values
