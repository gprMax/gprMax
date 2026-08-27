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

"""Yee-aligned closed surfaces for the KSIR NTFF formulation."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
"""Field components supported by the Cartesian KSIR formulation."""

COMPONENT_OFFSETS = {
    "Ex": (0.5, 0.0, 0.0),
    "Ey": (0.0, 0.5, 0.0),
    "Ez": (0.0, 0.0, 0.5),
    "Hx": (0.0, 0.5, 0.5),
    "Hy": (0.5, 0.0, 0.5),
    "Hz": (0.5, 0.5, 0.0),
}
"""Yee sample offsets in units of the grid spacing."""

FACES = ("x0", "xmax", "y0", "ymax", "z0", "zmax")
_FACE_SPECS = {
    "x0": (0, -1),
    "xmax": (0, 1),
    "y0": (1, -1),
    "ymax": (1, 1),
    "z0": (2, -1),
    "zmax": (2, 1),
}


def _readonly(array: npt.NDArray) -> npt.NDArray:
    """Mark geometry arrays read-only before exposing them to callers."""

    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class KSIRSurfaceFace:
    """One non-overlapping midpoint-patch face of a component surface."""

    component: str
    face_id: str
    normal_axis: int
    normal_sign: int
    normal: npt.NDArray[np.floating]
    inside_indices: npt.NDArray[np.int32]
    outside_indices: npt.NDArray[np.int32]
    inside_flat_indices: npt.NDArray[np.int64]
    outside_flat_indices: npt.NDArray[np.int64]
    patch_positions: npt.NDArray[np.floating]
    area_weights: npt.NDArray[np.floating]
    normal_spacing: float
    field_shape: Tuple[int, int, int]
    global_patch_indices: Optional[npt.NDArray[np.int64]] = None

    @property
    def npatches(self) -> int:
        return self.patch_positions.shape[0]

    def sample(self, field: npt.ArrayLike) -> Tuple[npt.NDArray, npt.NDArray]:
        """Gather the inside and outside samples from a gprMax field array."""

        values = np.asarray(field)
        if values.shape != self.field_shape:
            raise ValueError(
                f"field shape {values.shape} does not match surface field shape {self.field_shape}"
            )
        inside = values[tuple(self.inside_indices.T)]
        outside = values[tuple(self.outside_indices.T)]
        return inside, outside

    def collocate(
        self, inside: npt.ArrayLike, outside: npt.ArrayLike
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Collocate same-component samples and calculate the outward derivative.

        The sample arrays may have any leading dimensions, but their final
        dimension must enumerate this face's patches.
        """

        inside_values = np.asarray(inside)
        outside_values = np.asarray(outside)
        if inside_values.shape != outside_values.shape:
            raise ValueError("inside and outside samples must have the same shape")
        if inside_values.ndim == 0 or inside_values.shape[-1] != self.npatches:
            raise ValueError("the final sample dimension must match the number of face patches")
        surface_value = 0.5 * (outside_values + inside_values)
        normal_derivative = (outside_values - inside_values) / self.normal_spacing
        return surface_value, normal_derivative


@dataclass(frozen=True)
class KSIRComponentSurface:
    """Closed cuboid used to transform one Cartesian field component."""

    component: str
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]
    grid_spacing: Tuple[float, float, float]
    field_shape: Tuple[int, int, int]
    physical_lower: npt.NDArray[np.floating]
    physical_upper: npt.NDArray[np.floating]
    faces: Tuple[KSIRSurfaceFace, ...]

    @property
    def centre(self) -> npt.NDArray[np.floating]:
        return 0.5 * (self.physical_lower + self.physical_upper)

    @property
    def npatches(self) -> int:
        return sum(face.npatches for face in self.faces)

    @property
    def patch_positions(self) -> npt.NDArray[np.floating]:
        return np.concatenate([face.patch_positions for face in self.faces], axis=0)

    @property
    def normals(self) -> npt.NDArray[np.floating]:
        return np.concatenate(
            [np.broadcast_to(face.normal, (face.npatches, 3)) for face in self.faces], axis=0
        )

    @property
    def area_weights(self) -> npt.NDArray[np.floating]:
        return np.concatenate([face.area_weights for face in self.faces], axis=0)

    def face(self, face_id: str) -> KSIRSurfaceFace:
        """Return a face by its stable gprMax boundary identifier."""

        try:
            return next(face for face in self.faces if face.face_id == face_id)
        except StopIteration as exc:
            raise KeyError(face_id) from exc


def _triplet(name: str, values: Sequence, dtype) -> npt.NDArray:
    try:
        array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly three numeric values") from exc
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return array


def _index_triplet(name: str, values: Sequence) -> npt.NDArray[np.int64]:
    numeric = _triplet(name, values, np.dtype(float))
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{name} must contain exactly three integer values")
    return numeric.astype(np.int64)


def _component_index_range(lower: int, upper: int, offset: float) -> npt.NDArray[np.int32]:
    stop = upper if offset == 0.5 else upper + 1
    return np.arange(lower, stop, dtype=np.int32)


def _build_face(
    component: str,
    face_id: str,
    lower: npt.NDArray[np.int64],
    upper: npt.NDArray[np.int64],
    spacing: npt.NDArray[np.floating],
    field_shape: Tuple[int, int, int],
) -> KSIRSurfaceFace:
    real_dtype = spacing.dtype
    offsets = np.asarray(COMPONENT_OFFSETS[component], dtype=real_dtype)
    normal_axis, normal_sign = _FACE_SPECS[face_id]
    tangential_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    tangential_ranges = [
        _component_index_range(lower[axis], upper[axis], offsets[axis]) for axis in tangential_axes
    ]
    tangential_mesh = np.meshgrid(*tangential_ranges, indexing="ij")
    npatches = tangential_mesh[0].size

    inside_indices = np.empty((npatches, 3), dtype=np.int32)
    for axis, mesh in zip(tangential_axes, tangential_mesh):
        inside_indices[:, axis] = mesh.ravel()

    if normal_sign < 0:
        inside_normal_index = lower[normal_axis]
    elif offsets[normal_axis] == 0.5:
        inside_normal_index = upper[normal_axis] - 1
    else:
        inside_normal_index = upper[normal_axis]
    inside_indices[:, normal_axis] = inside_normal_index

    outside_indices = inside_indices.copy()
    outside_indices[:, normal_axis] += normal_sign

    shape_array = np.asarray(field_shape, dtype=np.int64)
    if np.any(inside_indices < 0) or np.any(inside_indices >= shape_array):
        raise ValueError(f"{component} {face_id} inside samples lie outside field_shape")
    if np.any(outside_indices < 0) or np.any(outside_indices >= shape_array):
        raise ValueError(f"{component} {face_id} outside samples lie outside field_shape")

    patch_positions = (inside_indices.astype(real_dtype) + offsets) * spacing
    outside_positions = (outside_indices.astype(real_dtype) + offsets) * spacing
    patch_positions[:, normal_axis] = 0.5 * (
        patch_positions[:, normal_axis] + outside_positions[:, normal_axis]
    )

    normal = np.zeros(3, dtype=real_dtype)
    normal[normal_axis] = normal_sign
    patch_area = spacing[tangential_axes[0]] * spacing[tangential_axes[1]]
    area_weights = np.full(npatches, patch_area, dtype=real_dtype)

    inside_flat = np.ravel_multi_index(inside_indices.T, field_shape).astype(np.int64)
    outside_flat = np.ravel_multi_index(outside_indices.T, field_shape).astype(np.int64)

    return KSIRSurfaceFace(
        component=component,
        face_id=face_id,
        normal_axis=normal_axis,
        normal_sign=normal_sign,
        normal=_readonly(normal),
        inside_indices=_readonly(inside_indices),
        outside_indices=_readonly(outside_indices),
        inside_flat_indices=_readonly(inside_flat),
        outside_flat_indices=_readonly(outside_flat),
        patch_positions=_readonly(patch_positions),
        area_weights=_readonly(area_weights),
        normal_spacing=float(spacing[normal_axis]),
        field_shape=field_shape,
    )


def build_component_surface(
    component: str,
    lower: Sequence[int],
    upper: Sequence[int],
    grid_spacing: Sequence[float],
    field_shape: Sequence[int],
    *,
    excluded_faces: Sequence[str] = (),
    real_dtype=None,
) -> KSIRComponentSurface:
    """Build a KSIR surface for one Yee field component.

    ``lower`` and ``upper`` identify a common logical cuboid in grid-line
    indices. The component's Yee offsets expand axes whose component samples
    lie on grid lines by half a cell on each side. Consequently all six
    component surfaces have the same centre while every face remains exactly
    halfway between two samples of that same component.

    Faces may be excluded only as part of an explicit closure policy. This
    low-level builder does not decide whether the resulting surface is exact.
    """

    if component not in COMPONENT_OFFSETS:
        raise ValueError(f"unknown field component {component!r}; expected one of {COMPONENTS}")
    excluded = tuple(excluded_faces)
    unknown_faces = set(excluded) - set(FACES)
    if unknown_faces:
        raise ValueError(f"unknown excluded faces: {sorted(unknown_faces)}")
    if len(set(excluded)) != len(excluded):
        raise ValueError("excluded_faces must not contain duplicates")
    if len(excluded) == len(FACES):
        raise ValueError("at least one surface face must remain active")

    lower_array = _index_triplet("lower", lower)
    upper_array = _index_triplet("upper", upper)
    if real_dtype is None:
        candidate = np.asarray(grid_spacing).dtype
        real_dtype = candidate if candidate.kind == "f" else np.dtype(float)
    real_dtype = np.dtype(real_dtype)
    if real_dtype.kind != "f":
        raise ValueError("real_dtype must be a floating-point dtype")
    spacing = _triplet("grid_spacing", grid_spacing, real_dtype)
    shape_array = _index_triplet("field_shape", field_shape)
    if np.any(upper_array <= lower_array):
        raise ValueError("upper bounds must be greater than lower bounds on every axis")
    if np.any(lower_array < 0):
        raise ValueError("lower bounds must be non-negative")
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("grid_spacing must contain finite values greater than zero")
    if np.any(shape_array <= 0):
        raise ValueError("field_shape values must be greater than zero")

    shape_tuple = tuple(int(value) for value in shape_array)
    faces = tuple(
        _build_face(component, face_id, lower_array, upper_array, spacing, shape_tuple)
        for face_id in FACES
        if face_id not in excluded
    )

    offsets = np.asarray(COMPONENT_OFFSETS[component], dtype=real_dtype)
    half_extension = np.where(offsets == 0.0, 0.5, 0.0).astype(real_dtype)
    physical_lower = (lower_array.astype(real_dtype) - half_extension) * spacing
    physical_upper = (upper_array.astype(real_dtype) + half_extension) * spacing

    return KSIRComponentSurface(
        component=component,
        lower=tuple(int(value) for value in lower_array),
        upper=tuple(int(value) for value in upper_array),
        grid_spacing=tuple(float(value) for value in spacing),
        field_shape=shape_tuple,
        physical_lower=_readonly(physical_lower),
        physical_upper=_readonly(physical_upper),
        faces=faces,
    )


def build_all_component_surfaces(
    lower: Sequence[int],
    upper: Sequence[int],
    grid_spacing: Sequence[float],
    field_shape: Sequence[int],
    *,
    excluded_faces: Sequence[str] = (),
    real_dtype=None,
) -> Mapping[str, KSIRComponentSurface]:
    """Build surfaces for all six Cartesian field components."""

    surfaces: Dict[str, KSIRComponentSurface] = {}
    for component in COMPONENTS:
        surfaces[component] = build_component_surface(
            component,
            lower,
            upper,
            grid_spacing,
            field_shape,
            excluded_faces=excluded_faces,
            real_dtype=real_dtype,
        )
    return surfaces
