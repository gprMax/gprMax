# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
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
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Giannopoulos et al. (1997) one-step time-domain far-field transform."""

import logging
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Mapping

import numpy as np
import numpy.typing as npt

from .equivalent_currents import ELECTRIC_COMPONENTS, MAGNETIC_COMPONENTS, _common_face_geometry
from .evaluator import spherical_basis, spherical_directions
from .surfaces import COMPONENT_OFFSETS, FACES
from .time_domain import TERMINAL_DECAY_THRESHOLD, TERMINAL_DECAY_WINDOW_SAMPLES

try:
    from gprMax.cython.ntff import (
        deposit_equivalent_current_time as _deposit_equivalent_current_time,
    )
    from gprMax.cython.ntff import (
        gather_equivalent_current_component as _gather_equivalent_current_component,
    )
except ImportError:  # Source-tree use before extensions are rebuilt.
    _deposit_equivalent_current_time = None
    _gather_equivalent_current_component = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EquivalentCurrentTimeResult:
    """Range-normalised electric far fields on a reduced-time axis."""

    name: str
    times: npt.NDArray[np.floating]
    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    directions: npt.NDArray[np.floating]
    fields: Mapping[str, npt.NDArray[np.floating]]
    terminal_field_ratios: npt.NDArray[np.floating]
    terminal_decay_ok: npt.NDArray[np.bool_]
    terminal_decay_threshold: float
    terminal_decay_window_samples: int
    collection_backend: str


def _readonly(values, dtype=None):
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


class EquivalentCurrentTimeMonitor:
    """Stream the 1997 one-step Love-current far-field construction."""

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
        *,
        real_dtype,
        wave_speed,
        impedance,
        nthreads=1,
        device_backend=None,
        mpi_grid=None,
    ):
        self.name = name
        self.lower = np.asarray(lower, dtype=np.int64)
        self.upper = np.asarray(upper, dtype=np.int64)
        self.spacing = np.asarray(spacing, dtype=real_dtype)
        self.global_field_shape = tuple(int(value) for value in field_shape)
        self.mpi_grid = mpi_grid
        self.mpi_comm = None if mpi_grid is None else mpi_grid.comm
        self.field_shape = (
            self.global_field_shape
            if mpi_grid is None
            else tuple(int(value + 1) for value in mpi_grid.size)
        )
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.real_dtype = np.dtype(real_dtype)
        self.wave_speed = float(wave_speed)
        self.impedance = float(impedance)
        self.nthreads = int(nthreads)
        self.origin = np.asarray(origin, dtype=self.real_dtype)
        self.allow_external_sources = False
        self.managed_output = True
        if device_backend not in (None, "cuda", "opencl", "metal"):
            raise ValueError("device_backend must be None, 'cuda', 'opencl', or 'metal'")
        self.device_backend = device_backend
        self.collection_backend = (
            f"{device_backend}_device"
            if device_backend is not None
            else (
                "cython_openmp"
                if _gather_equivalent_current_component is not None
                and _deposit_equivalent_current_time is not None
                else "numpy_fallback"
            )
        )
        if self.mpi_comm is not None:
            if self.device_backend is not None:
                raise ValueError("MPI NTFF collection is available with the CPU solver")
            self.collection_backend = f"mpi_{self.collection_backend}"
        if self.lower.shape != (3,) or self.upper.shape != (3,):
            raise ValueError("equivalent-current bounds must have shape (3,)")
        if np.any(self.upper <= self.lower):
            raise ValueError("equivalent-current upper bounds must exceed lower bounds")
        if self.origin.shape != (3,) or not np.all(np.isfinite(self.origin)):
            raise ValueError("origin must contain three finite values")
        if self.dt <= 0 or self.iterations < 3:
            raise ValueError("time transform requires positive dt and at least 3 iterations")
        if self.wave_speed <= 0 or self.impedance <= 0:
            raise ValueError("wave speed and impedance must be positive")
        if self.nthreads < 1:
            raise ValueError("nthreads must be positive")

        theta_values, phi_values = np.broadcast_arrays(
            np.asarray(theta, dtype=self.real_dtype),
            np.asarray(phi, dtype=self.real_dtype),
        )
        if theta_values.size == 0 or np.any(theta_values < 0) or np.any(theta_values > 180):
            raise ValueError("theta must contain values between 0 and 180 degrees")
        if not np.all(np.isfinite(theta_values)) or not np.all(np.isfinite(phi_values)):
            raise ValueError("far-field angles must be finite")
        self.theta = _readonly(theta_values.ravel(), self.real_dtype)
        self.phi = _readonly(phi_values.ravel(), self.real_dtype)
        self.directions = _readonly(
            spherical_directions(self.theta, self.phi, degrees=True), self.real_dtype
        )
        _, theta_basis, phi_basis = spherical_basis(self.theta, self.phi, degrees=True)
        self.theta_basis = _readonly(theta_basis, self.real_dtype)
        self.phi_basis = _readonly(phi_basis, self.real_dtype)

        positions = []
        normals = []
        areas = []
        normal_axes = []
        for face_id in FACES:
            face_positions, face_normals, face_areas, normal_axis = _common_face_geometry(
                self.lower,
                self.upper,
                self.spacing,
                face_id,
                self.real_dtype,
            )
            positions.append(face_positions)
            normals.append(face_normals)
            areas.append(face_areas)
            normal_axes.append(np.full(face_positions.shape[0], normal_axis, dtype=np.int8))
        positions = np.concatenate(positions)
        normals = np.concatenate(normals)
        areas = np.concatenate(areas)
        normal_axes = np.concatenate(normal_axes)
        if self.mpi_grid is not None:
            anchor = np.floor(positions / self.spacing).astype(np.int32)
            positive = normals[np.arange(normals.shape[0]), normal_axes] > 0
            anchor[np.flatnonzero(positive), normal_axes[positive]] -= 1
            owners = np.fromiter(
                (self.mpi_grid.get_rank_from_coordinate(item) for item in anchor),
                dtype=np.int32,
                count=anchor.shape[0],
            )
            keep = owners == self.mpi_grid.rank
            positions = positions[keep]
            normals = normals[keep]
            areas = areas[keep]
            normal_axes = normal_axes[keep]
        self.positions = _readonly(positions, self.real_dtype)
        self.normals = _readonly(normals, self.real_dtype)
        self.area_weights = _readonly(areas, self.real_dtype)
        self.normal_axes = _readonly(normal_axes, np.int8)
        self.npatches = self.positions.shape[0]
        self._stencils = self._build_stencils()
        self._gather_buffer = np.empty(self.npatches, dtype=self.real_dtype)

        shift = -((self.positions - self.origin) @ self.directions.T).T / (
            self.wave_speed * self.dt
        )
        self._delay_maps = {
            0.0: self._delay_map(shift, 0.0),
            0.5: self._delay_map(shift, 0.5),
        }
        local_minimum = float(np.min(shift)) if self.npatches else np.inf
        local_maximum = float(np.max(shift)) if self.npatches else -np.inf
        if self.mpi_comm is not None:
            bounds = self.mpi_comm.allgather((local_minimum, local_maximum))
            local_minimum = min(item[0] for item in bounds)
            local_maximum = max(item[1] for item in bounds)
        if not np.isfinite(local_minimum) or not np.isfinite(local_maximum):
            raise RuntimeError("MPI equivalent-current surface has no owned patches")
        self._time_origin_step = int(np.floor(local_minimum)) - 2
        last_step = self.iterations - 1 + int(np.ceil(local_maximum)) + 2
        self._raw_length = last_step - self._time_origin_step + 1
        self._theta_output = (
            np.zeros((self.directions.shape[0], self._raw_length), dtype=self.real_dtype)
            if device_backend is None
            else None
        )
        self._phi_output = None if self._theta_output is None else np.zeros_like(self._theta_output)
        self._complete_start_step = int(np.ceil(local_maximum + 1))
        self._complete_stop_step = int(np.floor(local_minimum + self.iterations - 1))
        if self._complete_stop_step < self._complete_start_step:
            raise ValueError(
                "time window is too short to contain one complete retarded surface history"
            )

        self._previous_electric = None
        self._previous_magnetic = None
        self._next_electric = 0
        self._next_magnetic = 0
        self._finalised = False
        self._result = None
        self.surface_material_id = None

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("equivalent-current time result is unavailable before finalisation")
        return self._result

    def _build_stencils(self):
        # Surface coordinates and Yee offsets are stored in the configured
        # field precision. A fixed 1e-9 tolerance is tighter than one ULP for
        # typical single-precision grid coordinates and can reject an exactly
        # aligned surface after division by ``spacing``.
        alignment_tolerance = max(1e-9, 32 * np.finfo(self.real_dtype).eps)
        result = {}
        for component in ELECTRIC_COMPONENTS + MAGNETIC_COMPONENTS:
            component_axis = "xyz".index(component[1].lower())
            groups = []
            for face_axis in range(3):
                if face_axis == component_axis:
                    continue
                patch_indices = np.flatnonzero(self.normal_axes == face_axis)
                target = self.positions[patch_indices]
                shifts_by_axis = []
                offset = np.asarray(COMPONENT_OFFSETS[component], dtype=self.real_dtype)
                for axis in range(3):
                    sample_index = target[:, axis] / self.spacing[axis] - offset[axis]
                    nearest = np.rint(sample_index)
                    if np.allclose(sample_index, nearest, rtol=0, atol=alignment_tolerance):
                        shifts_by_axis.append((0.0,))
                    elif np.allclose(
                        np.abs(sample_index - nearest),
                        0.5,
                        rtol=0,
                        atol=alignment_tolerance,
                    ):
                        shifts_by_axis.append((-0.5, 0.5))
                    else:
                        raise RuntimeError(f"cannot form symmetric {component} stencil")
                flat_indices = []
                for shifts in product(*shifts_by_axis):
                    position = target + np.asarray(shifts) * self.spacing
                    indices_float = position / self.spacing - offset
                    indices = np.rint(indices_float).astype(np.int64)
                    if not np.allclose(
                        indices_float,
                        indices,
                        rtol=0,
                        atol=alignment_tolerance,
                    ):
                        raise RuntimeError(f"{component} stencil is not on Yee samples")
                    if self.mpi_grid is not None:
                        indices -= np.asarray(self.mpi_grid.lower_extent, dtype=np.int64)
                    if np.any(indices < 0) or np.any(
                        indices >= np.asarray(self.field_shape)[np.newaxis, :]
                    ):
                        raise ValueError(f"{component} stencil lies outside the field array")
                    flat_indices.append(np.ravel_multi_index(indices.T, self.field_shape))
                groups.append(
                    (
                        _readonly(patch_indices, np.int64),
                        _readonly(np.stack(flat_indices), np.int64),
                    )
                )
            result[component] = tuple(groups)
        return MappingProxyType(result)

    def _delay_map(self, shift, offset):
        coordinate = shift + offset
        lower = np.floor(coordinate).astype(np.int64)
        return (
            _readonly(lower, np.int64),
            _readonly(coordinate - lower, self.real_dtype),
        )

    def _gather_vector(self, components, fields):
        values = np.zeros((self.npatches, 3), dtype=self.real_dtype)
        for axis, component in enumerate(components):
            flat = np.ascontiguousarray(fields[axis], dtype=self.real_dtype).ravel()
            for patch_indices, stencil in self._stencils[component]:
                if _gather_equivalent_current_component is None:
                    gathered = np.mean(flat[stencil], axis=0, dtype=self.real_dtype)
                else:
                    gathered = self._gather_buffer[: patch_indices.size]
                    _gather_equivalent_current_component(self.nthreads, stencil, flat, gathered)
                values[patch_indices, axis] = gathered
        return values

    def _deposit(self, sample_index, offset, current, theta_basis, phi_basis):
        integer_delay, fraction = self._delay_maps[offset]
        first = sample_index + int(np.min(integer_delay)) - self._time_origin_step
        last = sample_index + int(np.max(integer_delay)) - self._time_origin_step + 1
        if first < 0 or last >= self._raw_length:
            raise RuntimeError("equivalent-current time output buffer is too short")
        current = np.ascontiguousarray(current, dtype=self.real_dtype)
        if _deposit_equivalent_current_time is not None:
            _deposit_equivalent_current_time(
                self.nthreads,
                sample_index,
                current,
                theta_basis,
                phi_basis,
                integer_delay,
                fraction,
                self.area_weights,
                self._time_origin_step,
                self._theta_output,
                self._phi_output,
            )
            return
        theta_values = current @ theta_basis.T
        phi_values = current @ phi_basis.T
        for direction in range(self.directions.shape[0]):
            destination = sample_index + integer_delay[direction] - self._time_origin_step
            weight = self.area_weights
            np.add.at(
                self._theta_output[direction],
                destination,
                (1 - fraction[direction]) * weight * theta_values[:, direction],
            )
            np.add.at(
                self._theta_output[direction],
                destination + 1,
                fraction[direction] * weight * theta_values[:, direction],
            )
            np.add.at(
                self._phi_output[direction],
                destination,
                (1 - fraction[direction]) * weight * phi_values[:, direction],
            )
            np.add.at(
                self._phi_output[direction],
                destination + 1,
                fraction[direction] * weight * phi_values[:, direction],
            )

    def validate_materials(self, material_ids, id_lookup):
        ids = np.asarray(material_ids)
        sampled = []
        for component in ELECTRIC_COMPONENTS + MAGNETIC_COMPONENTS:
            component_ids = ids[id_lookup[component]].ravel()
            for _, stencil in self._stencils[component]:
                sampled.append(component_ids[stencil.ravel()])
        if self.mpi_comm is not None:
            from .mpi import distributed_unique

            local = np.concatenate(sampled) if sampled else np.empty(0, dtype=np.int64)
            unique = distributed_unique(local, self.mpi_comm)
        else:
            unique = np.unique(np.concatenate(sampled))
        if unique.size != 1:
            raise ValueError(
                "equivalent-current surface must lie in one material; "
                f"found IDs {unique.tolist()}"
            )
        self.surface_material_id = int(unique[0])
        return self.surface_material_id

    def observe_electric(self, iteration, Ex, Ey, Ez):
        if self.device_backend is not None:
            raise RuntimeError("device equivalent-current monitors are observed by the backend")
        if iteration != self._next_electric:
            raise ValueError(f"expected electric iteration {self._next_electric}, got {iteration}")
        electric = self._gather_vector(ELECTRIC_COMPONENTS, (Ex, Ey, Ez))
        magnetic_current = -np.cross(self.normals, electric)
        if self._previous_electric is not None:
            derivative = (magnetic_current - self._previous_electric) / self.dt
            self._deposit(
                iteration - 1,
                0.5,
                derivative,
                self.phi_basis,
                -self.theta_basis,
            )
        self._previous_electric = magnetic_current
        self._next_electric += 1

    def observe_magnetic(self, iteration, Hx, Hy, Hz):
        if self.device_backend is not None:
            raise RuntimeError("device equivalent-current monitors are observed by the backend")
        if iteration != self._next_magnetic:
            raise ValueError(f"expected magnetic iteration {self._next_magnetic}, got {iteration}")
        magnetic = self._gather_vector(MAGNETIC_COMPONENTS, (Hx, Hy, Hz))
        electric_current = self.impedance * np.cross(self.normals, magnetic)
        if self._previous_magnetic is not None:
            derivative = (electric_current - self._previous_magnetic) / self.dt
            self._deposit(
                iteration,
                0.0,
                derivative,
                self.theta_basis,
                self.phi_basis,
            )
        self._previous_magnetic = electric_current
        self._next_magnetic += 1

    def load_device_far_field_output(self, theta_output, phi_output):
        """Install completed device traces before normal finalisation."""

        if self.device_backend is None:
            raise RuntimeError("CPU equivalent-current monitors cannot load device output")
        expected = (self.directions.shape[0], self._raw_length)
        theta_output = np.asarray(theta_output, dtype=self.real_dtype)
        phi_output = np.asarray(phi_output, dtype=self.real_dtype)
        if theta_output.shape != expected or phi_output.shape != expected:
            raise ValueError(
                "equivalent-current device output has the wrong shape: "
                f"expected {expected}, got {theta_output.shape} and {phi_output.shape}"
            )
        self._theta_output = np.ascontiguousarray(theta_output)
        self._phi_output = np.ascontiguousarray(phi_output)
        self._next_electric = self.iterations
        self._next_magnetic = self.iterations

    def finalise(self):
        if self._finalised:
            return
        if self._next_electric != self.iterations or self._next_magnetic != self.iterations:
            raise RuntimeError("equivalent-current monitor did not receive every time step")
        if self._theta_output is None or self._phi_output is None:
            raise RuntimeError("equivalent-current device output was not loaded")
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
        scale = -1 / (4 * np.pi * self.wave_speed)
        electric_theta = _readonly(scale * self._theta_output[:, start:stop], self.real_dtype)
        electric_phi = _readonly(scale * self._phi_output[:, start:stop], self.real_dtype)
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
                "Equivalent-current time monitor %r has not decayed below %.1e "
                "at the end of its complete retarded interval (direction %d "
                "ratio %.3e). Increase the simulation time window.",
                self.name,
                TERMINAL_DECAY_THRESHOLD,
                worst,
                ratios[worst],
            )
        self._finalised = True
