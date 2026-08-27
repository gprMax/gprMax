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

"""MPI partitioning helpers for near-to-far-field integration surfaces."""

from types import MappingProxyType
from typing import Mapping

import numpy as np

from .surfaces import KSIRComponentSurface, KSIRSurfaceFace


def _readonly(values, dtype=None):
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _patch_owner_ranks(grid, inside_indices):
    """Return the unique rank owning each inside Yee sample.

    The outside sample differs by one cell in the face-normal direction and
    is therefore available in the selected rank's existing field halo.
    """

    return np.fromiter(
        (grid.get_rank_from_coordinate(index) for index in inside_indices),
        dtype=np.int32,
        count=inside_indices.shape[0],
    )


def localise_component_surface(surface: KSIRComponentSurface, grid) -> KSIRComponentSurface:
    """Partition a global component surface and map its indices to one rank."""

    local_shape = tuple(int(value + 1) for value in grid.size)
    lower_extent = np.asarray(grid.lower_extent, dtype=np.int32)
    local_faces = []
    global_offset = 0
    for face in surface.faces:
        owners = _patch_owner_ranks(grid, face.inside_indices)
        keep = owners == grid.rank
        global_indices = global_offset + np.flatnonzero(keep)
        inside = np.asarray(face.inside_indices[keep] - lower_extent, dtype=np.int32)
        outside = np.asarray(face.outside_indices[keep] - lower_extent, dtype=np.int32)
        shape = np.asarray(local_shape, dtype=np.int32)
        if np.any(inside < 0) or np.any(inside >= shape):
            raise RuntimeError(
                f"MPI NTFF {face.component} {face.face_id} inside sample is not local"
            )
        if np.any(outside < 0) or np.any(outside >= shape):
            raise RuntimeError(
                f"MPI NTFF {face.component} {face.face_id} outside sample is not in the halo"
            )
        inside_flat = np.ravel_multi_index(inside.T, local_shape).astype(np.int64)
        outside_flat = np.ravel_multi_index(outside.T, local_shape).astype(np.int64)
        local_faces.append(
            KSIRSurfaceFace(
                component=face.component,
                face_id=face.face_id,
                normal_axis=face.normal_axis,
                normal_sign=face.normal_sign,
                normal=face.normal,
                inside_indices=_readonly(inside, np.int32),
                outside_indices=_readonly(outside, np.int32),
                inside_flat_indices=_readonly(inside_flat, np.int64),
                outside_flat_indices=_readonly(outside_flat, np.int64),
                patch_positions=_readonly(face.patch_positions[keep], face.patch_positions.dtype),
                area_weights=_readonly(face.area_weights[keep], face.area_weights.dtype),
                normal_spacing=face.normal_spacing,
                field_shape=local_shape,
                global_patch_indices=_readonly(global_indices, np.int64),
            )
        )
        global_offset += face.npatches

    return KSIRComponentSurface(
        component=surface.component,
        lower=surface.lower,
        upper=surface.upper,
        grid_spacing=surface.grid_spacing,
        field_shape=local_shape,
        physical_lower=surface.physical_lower,
        physical_upper=surface.physical_upper,
        faces=tuple(local_faces),
    )


def localise_surfaces(surfaces: Mapping[str, KSIRComponentSurface], grid):
    """Return rank-local views of a set of globally defined surfaces."""

    return MappingProxyType(
        {
            component: localise_component_surface(surface, grid)
            for component, surface in surfaces.items()
        }
    )


def global_patch_indices(surface: KSIRComponentSurface):
    """Return concatenated global patch indices for a localised surface."""

    values = [face.global_patch_indices for face in surface.faces]
    if any(value is None for value in values):
        raise RuntimeError("surface does not contain MPI global patch indices")
    return np.ascontiguousarray(np.concatenate(values), dtype=np.int64)


def distributed_unique(values, comm):
    """Return the sorted union of small rank-local integer arrays."""

    local = np.unique(np.asarray(values, dtype=np.int64))
    gathered = comm.allgather(local)
    nonempty = [item for item in gathered if item.size]
    if not nonempty:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(nonempty))
