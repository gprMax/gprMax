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

"""Conventional Love-current far-zone transformation on a Yee grid."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt

from .surfaces import COMPONENT_OFFSETS, FACES

try:
    from gprMax.cython.ntff import (
        evaluate_equivalent_current_far_zone as _evaluate_equivalent_current_cython,
    )
except ImportError:  # Source-tree use before extensions are rebuilt.
    _evaluate_equivalent_current_cython = None


ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")
MAGNETIC_COMPONENTS = ("Hx", "Hy", "Hz")
ALL_COMPONENTS = ELECTRIC_COMPONENTS + MAGNETIC_COMPONENTS
UNIT_VECTORS = np.eye(3)
TARGET_WORKING_SET_BYTES = 32 * 1024 * 1024
MAX_DIRECTION_BLOCK = 1024


@dataclass(frozen=True)
class EquivalentCurrentPhasors:
    """Love currents collocated at common cell-face centres."""

    positions: npt.NDArray[np.floating]
    normals: npt.NDArray[np.floating]
    area_weights: npt.NDArray[np.floating]
    electric_current: npt.NDArray[np.complexfloating]
    magnetic_current: npt.NDArray[np.complexfloating]


def _readonly(values: npt.ArrayLike, dtype=None) -> npt.NDArray:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _common_face_geometry(lower, upper, spacing, face_id, real_dtype):
    normal_axis = "xyz".index(face_id[0])
    normal_sign = -1 if face_id.endswith("0") else 1
    tangential_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    axes = [
        (np.arange(lower[axis], upper[axis], dtype=real_dtype) + 0.5) * spacing[axis]
        for axis in tangential_axes
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    positions = np.empty((mesh[0].size, 3), dtype=real_dtype)
    positions[:, normal_axis] = (
        lower[normal_axis] if normal_sign < 0 else upper[normal_axis]
    ) * spacing[normal_axis]
    for axis, values in zip(tangential_axes, mesh):
        positions[:, axis] = values.ravel()
    normal = np.zeros(3, dtype=real_dtype)
    normal[normal_axis] = normal_sign
    normals = np.broadcast_to(normal, positions.shape).copy()
    area = spacing[tangential_axes[0]] * spacing[tangential_axes[1]]
    areas = np.full(positions.shape[0], area, dtype=real_dtype)
    return positions, normals, areas, normal_axis


def _component_face_values(data, face_id: str):
    start = 0
    selected = None
    for face in data.surface.faces:
        stop = start + face.npatches
        if face.face_id == face_id:
            selected = face, slice(start, stop)
            break
        start = stop
    if selected is None:
        raise ValueError(f"component surface has no face {face_id!r}")

    face, face_slice = selected
    tangential_axes = tuple(axis for axis in range(3) if axis != face.normal_axis)
    face_shape = tuple(np.unique(face.inside_indices[:, axis]).size for axis in tangential_axes)
    leading_shape = np.asarray(data.field).shape[:-1]
    field = np.asarray(data.field)[..., face_slice].reshape((*leading_shape, *face_shape))
    derivative = np.asarray(data.normal_derivative)[..., face_slice].reshape(
        (*leading_shape, *face_shape)
    )
    return face, tangential_axes, field, derivative


def _collocate_tangential_component(data, face_id: str) -> npt.NDArray:
    """Arithmetic-collocate one tangential Yee component on a common face."""

    component = data.surface.component
    face, tangential_axes, field, derivative = _component_face_values(data, face_id)
    component_axis = "xyz".index(component[1].lower())
    if component_axis == face.normal_axis:
        raise ValueError(f"{component} is normal, not tangential, on {face_id}")

    inside = field - 0.5 * face.normal_spacing * derivative
    outside = field + 0.5 * face.normal_spacing * derivative
    normal_samples = (inside,) if component.startswith("E") else (inside, outside)
    average_axis = next(
        axis for axis in tangential_axes if COMPONENT_OFFSETS[component][axis] == 0.0
    )
    array_axis = len(np.asarray(data.field).shape[:-1]) + tangential_axes.index(average_axis)
    stencil_values = []
    for samples in normal_samples:
        low = [slice(None)] * samples.ndim
        high = [slice(None)] * samples.ndim
        low[array_axis] = slice(0, -1)
        high[array_axis] = slice(1, None)
        stencil_values.extend((samples[tuple(low)], samples[tuple(high)]))
    result = np.mean(np.asarray(stencil_values), axis=0, dtype=np.asarray(data.field).dtype)
    return result.reshape((*result.shape[: len(np.asarray(data.field).shape[:-1])], -1))


def collocate_love_currents(
    surface_data: Mapping[str, object],
) -> EquivalentCurrentPhasors:
    """Form ``J = n x H`` and ``M = -n x E`` on common active faces."""

    missing = set(ALL_COMPONENTS) - set(surface_data)
    if missing:
        raise ValueError(f"surface data is missing components {sorted(missing)}")
    first = surface_data["Ex"]
    field = np.asarray(first.field)
    if field.ndim != 2 or field.dtype.kind != "c":
        raise ValueError("surface phasors must have shape (nfrequencies, npatches)")
    lower = np.asarray(first.surface.lower, dtype=np.int64)
    upper = np.asarray(first.surface.upper, dtype=np.int64)
    spacing = np.asarray(first.surface.grid_spacing)
    real_dtype = spacing.dtype
    complex_dtype = field.dtype
    active_faces = tuple(face.face_id for face in first.surface.faces)
    if not active_faces or any(face not in FACES for face in active_faces):
        raise ValueError("equivalent-current surface has invalid active faces")
    if active_faces != tuple(face for face in FACES if face in active_faces):
        raise ValueError("equivalent-current surface faces must use canonical order")
    for component in ALL_COMPONENTS:
        data = surface_data[component]
        if (
            tuple(data.surface.lower) != tuple(lower)
            or tuple(data.surface.upper) != tuple(upper)
            or np.asarray(data.field).shape[0] != field.shape[0]
            or np.asarray(data.field).dtype != complex_dtype
        ):
            raise ValueError("surface DFT components are not compatible")
        if tuple(face.face_id for face in data.surface.faces) != active_faces:
            raise ValueError(
                "equivalent-current surface components must use the same active faces"
            )

    all_positions = []
    all_normals = []
    all_areas = []
    all_electric_current = []
    all_magnetic_current = []
    for face_id in active_faces:
        positions, normals, areas, normal_axis = _common_face_geometry(
            lower, upper, spacing, face_id, real_dtype
        )
        electric = np.zeros((field.shape[0], positions.shape[0], 3), dtype=complex_dtype)
        magnetic = np.zeros_like(electric)
        for component_axis in range(3):
            if component_axis == normal_axis:
                continue
            electric[:, :, component_axis] = _collocate_tangential_component(
                surface_data[ELECTRIC_COMPONENTS[component_axis]], face_id
            )
            magnetic[:, :, component_axis] = _collocate_tangential_component(
                surface_data[MAGNETIC_COMPONENTS[component_axis]], face_id
            )
        all_positions.append(positions)
        all_normals.append(normals)
        all_areas.append(areas)
        all_electric_current.append(np.cross(normals[np.newaxis, :, :], magnetic))
        all_magnetic_current.append(-np.cross(normals[np.newaxis, :, :], electric))

    return EquivalentCurrentPhasors(
        positions=_readonly(np.concatenate(all_positions), real_dtype),
        normals=_readonly(np.concatenate(all_normals), real_dtype),
        area_weights=_readonly(np.concatenate(all_areas), real_dtype),
        electric_current=_readonly(np.concatenate(all_electric_current, axis=1), complex_dtype),
        magnetic_current=_readonly(np.concatenate(all_magnetic_current, axis=1), complex_dtype),
    )


def _evaluate_numpy(
    currents: EquivalentCurrentPhasors,
    wavenumbers: npt.NDArray,
    directions: npt.NDArray,
    impedance: float,
) -> npt.NDArray:
    complex_dtype = currents.electric_current.dtype
    output = np.zeros((wavenumbers.size, directions.shape[0], 3), dtype=complex_dtype)
    complex_bytes = np.dtype(complex_dtype).itemsize
    bytes_per_direction = max(1, currents.positions.shape[0]) * complex_bytes
    block_size = max(
        1,
        min(MAX_DIRECTION_BLOCK, TARGET_WORKING_SET_BYTES // bytes_per_direction),
    )
    weighted_j = currents.area_weights[np.newaxis, :, np.newaxis] * currents.electric_current
    weighted_m = currents.area_weights[np.newaxis, :, np.newaxis] * currents.magnetic_current
    for frequency, wavenumber in enumerate(wavenumbers):
        for start in range(0, directions.shape[0], block_size):
            stop = min(start + block_size, directions.shape[0])
            direction = directions[start:stop]
            phase = np.exp(1j * wavenumber * (currents.positions @ direction.T)).astype(
                complex_dtype, copy=False
            )
            electric_transform = phase.T @ weighted_j[frequency]
            magnetic_transform = phase.T @ weighted_m[frequency]
            transverse_electric = (
                electric_transform
                - direction * np.sum(direction * electric_transform, axis=1)[:, np.newaxis]
            )
            output[frequency, start:stop] = (
                -1j
                * wavenumber
                / (4 * np.pi)
                * (impedance * transverse_electric - np.cross(direction, magnetic_transform))
            )
    return output


def evaluate_equivalent_current_far_zone(
    surface_data: Mapping[str, object],
    frequencies: npt.ArrayLike,
    directions: npt.ArrayLike,
    *,
    origin: npt.ArrayLike,
    wave_speed: float,
    impedance: float,
    nthreads: int = 1,
) -> npt.NDArray[np.complexfloating]:
    """Return Cartesian ``r exp(+jkr) E`` using the engineering convention."""

    currents = collocate_love_currents(surface_data)
    frequency_values = np.asarray(frequencies, dtype=currents.positions.dtype)
    direction_values = np.asarray(directions, dtype=currents.positions.dtype)
    origin_values = np.asarray(origin, dtype=currents.positions.dtype)
    if frequency_values.ndim != 1 or frequency_values.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional array")
    if direction_values.ndim != 2 or direction_values.shape[1] != 3:
        raise ValueError("directions must have shape (ndirections, 3)")
    if not np.allclose(np.linalg.norm(direction_values, axis=1), 1, rtol=1e-6, atol=1e-7):
        raise ValueError("directions must contain unit vectors")
    if origin_values.shape != (3,) or not np.all(np.isfinite(origin_values)):
        raise ValueError("origin must contain three finite values")
    if not np.isfinite(wave_speed) or wave_speed <= 0:
        raise ValueError("wave_speed must be finite and positive")
    if not np.isfinite(impedance) or impedance <= 0:
        raise ValueError("impedance must be finite and positive")
    if not isinstance(nthreads, (int, np.integer)) or nthreads < 1:
        raise ValueError("nthreads must be an integer greater than zero")

    relative_currents = EquivalentCurrentPhasors(
        positions=_readonly(currents.positions - origin_values, currents.positions.dtype),
        normals=currents.normals,
        area_weights=currents.area_weights,
        electric_current=currents.electric_current,
        magnetic_current=currents.magnetic_current,
    )
    wavenumbers = np.ascontiguousarray(
        2 * np.pi * frequency_values / wave_speed, dtype=currents.positions.dtype
    )
    direction_values = np.ascontiguousarray(direction_values)
    if _evaluate_equivalent_current_cython is None:
        output = _evaluate_numpy(relative_currents, wavenumbers, direction_values, impedance)
    else:
        output = np.empty(
            (wavenumbers.size, direction_values.shape[0], 3),
            dtype=currents.electric_current.dtype,
        )
        _evaluate_equivalent_current_cython(
            nthreads,
            relative_currents.positions,
            relative_currents.area_weights,
            wavenumbers,
            direction_values,
            relative_currents.electric_current,
            relative_currents.magnetic_current,
            impedance,
            output,
        )
    return _readonly(output, currents.electric_current.dtype)
