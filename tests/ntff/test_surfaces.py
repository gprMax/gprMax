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

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from gprMax.ntff.surfaces import (
    COMPONENTS,
    COMPONENT_OFFSETS,
    FACES,
    build_all_component_surfaces,
    build_component_surface,
)


LOWER = np.array((2, 3, 4))
UPPER = np.array((6, 8, 10))
SPACING = np.array((0.1, 0.2, 0.3))
FIELD_SHAPE = (12, 13, 14)
FACE_SPECS = {
    "x0": (0, -1),
    "xmax": (0, 1),
    "y0": (1, -1),
    "ymax": (1, 1),
    "z0": (2, -1),
    "zmax": (2, 1),
}


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("face_id", FACES)
def test_all_36_component_faces_have_correct_geometry_and_outward_derivative(
    component, face_id
):
    surface = build_component_surface(component, LOWER, UPPER, SPACING, FIELD_SHAPE)
    face = surface.face(face_id)
    offsets = np.asarray(COMPONENT_OFFSETS[component])
    normal_axis, normal_sign = FACE_SPECS[face_id]
    tangential_axes = [axis for axis in range(3) if axis != normal_axis]

    assert face.component == component
    assert face.normal_axis == normal_axis
    assert face.normal_sign == normal_sign
    expected_normal = np.zeros(3)
    expected_normal[normal_axis] = normal_sign
    assert_array_equal(face.normal, expected_normal)

    index_step = face.outside_indices - face.inside_indices
    assert_array_equal(index_step, np.broadcast_to(expected_normal, index_step.shape))
    assert np.all(face.inside_indices >= 0)
    assert np.all(face.outside_indices >= 0)
    assert np.all(face.inside_indices < np.asarray(FIELD_SHAPE))
    assert np.all(face.outside_indices < np.asarray(FIELD_SHAPE))

    expected_normal_position = (
        surface.physical_lower[normal_axis]
        if normal_sign < 0
        else surface.physical_upper[normal_axis]
    )
    assert_allclose(face.patch_positions[:, normal_axis], expected_normal_position)
    for axis in tangential_axes:
        expected_positions = (face.inside_indices[:, axis] + offsets[axis]) * SPACING[axis]
        assert_allclose(face.patch_positions[:, axis], expected_positions)

    tangential_counts = [
        UPPER[axis] - LOWER[axis] + (1 if offsets[axis] == 0.0 else 0)
        for axis in tangential_axes
    ]
    assert face.npatches == np.prod(tangential_counts)
    expected_face_area = np.prod(
        surface.physical_upper[tangential_axes]
        - surface.physical_lower[tangential_axes]
    )
    assert_allclose(np.sum(face.area_weights), expected_face_area)

    indices = np.indices(FIELD_SHAPE).reshape(3, -1).T
    positions = (indices + offsets) * SPACING
    gradient = np.array((1.7, -0.8, 0.45))
    intercept = -0.23
    linear_field = (positions @ gradient + intercept).reshape(FIELD_SHAPE)
    inside, outside = face.sample(linear_field)
    collocated, normal_derivative = face.collocate(inside, outside)

    assert_allclose(collocated, face.patch_positions @ gradient + intercept)
    assert_allclose(normal_derivative, expected_normal @ gradient)
    assert_array_equal(inside, linear_field.ravel()[face.inside_flat_indices])
    assert_array_equal(outside, linear_field.ravel()[face.outside_flat_indices])


def test_all_component_surfaces_share_centre_and_stable_face_order():
    surfaces = build_all_component_surfaces(LOWER, UPPER, SPACING, FIELD_SHAPE)
    expected_centre = 0.5 * (LOWER + UPPER) * SPACING

    assert tuple(surfaces) == COMPONENTS
    for surface in surfaces.values():
        assert_allclose(surface.centre, expected_centre)
        assert tuple(face.face_id for face in surface.faces) == FACES
        assert surface.patch_positions.shape == (surface.npatches, 3)
        assert surface.normals.shape == (surface.npatches, 3)
        assert surface.area_weights.shape == (surface.npatches,)


def test_face_collocation_accepts_frequency_by_patch_arrays():
    face = build_component_surface("Hz", LOWER, UPPER, SPACING, FIELD_SHAPE).face("zmax")
    inside = np.arange(2 * face.npatches).reshape(2, face.npatches)
    outside = inside + 3.0

    field, derivative = face.collocate(inside, outside)

    assert field.shape == (2, face.npatches)
    assert_allclose(field, inside + 1.5)
    assert_allclose(derivative, 3.0 / SPACING[2])


@pytest.mark.parametrize(
    "component,lower,upper,spacing,shape,match",
    [
        ("Qx", LOWER, UPPER, SPACING, FIELD_SHAPE, "unknown field component"),
        ("Ex", (0, 3, 4), UPPER, SPACING, FIELD_SHAPE, "outside samples"),
        ("Ex", LOWER, (12, 8, 10), SPACING, FIELD_SHAPE, "outside samples"),
        ("Ex", LOWER, LOWER, SPACING, FIELD_SHAPE, "greater than"),
        ("Ex", (2.5, 3, 4), UPPER, SPACING, FIELD_SHAPE, "integer values"),
        ("Ex", LOWER, UPPER, (0.1, 0.0, 0.3), FIELD_SHAPE, "grid_spacing"),
        ("Ex", LOWER[:2], UPPER, SPACING, FIELD_SHAPE, "exactly three"),
    ],
)
def test_surface_builder_rejects_invalid_geometry(
    component, lower, upper, spacing, shape, match
):
    with pytest.raises(ValueError, match=match):
        build_component_surface(component, lower, upper, spacing, shape)


def test_surface_geometry_arrays_are_read_only():
    face = build_component_surface("Ey", LOWER, UPPER, SPACING, FIELD_SHAPE).face("x0")

    with pytest.raises(ValueError, match="read-only"):
        face.patch_positions[0, 0] = 1.0


def test_surface_geometry_uses_requested_real_precision():
    surface = build_component_surface(
        "Hy", LOWER, UPPER, SPACING, FIELD_SHAPE, real_dtype=np.float32
    )

    assert surface.physical_lower.dtype == np.float32
    assert surface.physical_upper.dtype == np.float32
    for face in surface.faces:
        assert face.normal.dtype == np.float32
        assert face.patch_positions.dtype == np.float32
        assert face.area_weights.dtype == np.float32
