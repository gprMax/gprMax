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
from numpy.testing import assert_allclose
from scipy.constants import c

from gprMax.ntff.closures import (
    ExperimentalMask,
    HuygensOpenSurface,
    ResolvedKSIRClosure,
    SymmetryCompletion,
    SymmetryPlane,
    closure_from_metadata,
    component_parity,
    resolve_closure,
)
from gprMax.ntff.evaluator import (
    evaluate_far_zone_patches,
    spherical_directions,
)
from gprMax.ntff.surfaces import FACES, build_component_surface


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("boundary_type", ("pec", "pmc"))
def test_component_parity_matches_electric_edge_image_theory(axis, boundary_type):
    for family in ("E", "H"):
        for component_axis, letter in enumerate("xyz"):
            component = family + letter
            normal = component_axis == axis
            if boundary_type == "pec":
                expected_even = normal if family == "E" else not normal
            else:
                expected_even = not normal if family == "E" else normal
            assert component_parity(component, axis, boundary_type) == (1 if expected_even else -1)


def test_resolver_uses_only_declared_electric_grid_line_planes():
    closure = resolve_closure(
        SymmetryCompletion(),
        {"x0": "pmc", "ymax": "pec"},
        (0, 3, 2),
        (8, 10, 9),
        (12, 10, 11),
        (0.1, 0.2, 0.3),
    )

    assert closure.omitted_faces == ("x0", "ymax")
    assert [plane.axis for plane in closure.symmetry_planes] == [0, 1]
    assert [plane.coordinate for plane in closure.symmetry_planes] == [0.0, 2.0]
    assert [plane.boundary_type for plane in closure.symmetry_planes] == [
        "pmc",
        "pec",
    ]
    assert closure.image_count == 4


def test_resolver_rejects_unresolved_or_untouched_symmetry_assertions():
    kwargs = dict(
        symmetry_boundaries={"x0": "pmc"},
        lower=(0, 2, 2),
        upper=(8, 8, 8),
        grid_size=(12, 12, 12),
        grid_spacing=(0.1, 0.1, 0.1),
    )
    with pytest.raises(ValueError, match="not a declared"):
        resolve_closure(SymmetryCompletion(("y0",)), **kwargs)
    with pytest.raises(ValueError, match="does not touch"):
        resolve_closure(
            SymmetryCompletion(("x0",)),
            symmetry_boundaries={"x0": "pmc"},
            lower=(1, 2, 2),
            upper=(8, 8, 8),
            grid_size=(12, 12, 12),
            grid_spacing=(0.1, 0.1, 0.1),
        )


@pytest.mark.parametrize(
    "boundaries,match",
    [
        ({"left": "pec"}, "unknown faces"),
        ({"x0": "open"}, "types"),
    ],
)
def test_resolver_rejects_invalid_boundary_maps(boundaries, match):
    with pytest.raises(ValueError, match=match):
        resolve_closure(
            SymmetryCompletion(),
            boundaries,
            (0, 2, 2),
            (8, 8, 8),
            (12, 12, 12),
            (0.1, 0.1, 0.1),
        )


@pytest.mark.parametrize(
    "face,lower,upper,plane_coordinate,boundary_index",
    [
        ("x0", (0, 2, 2), (6, 7, 7), 0.0, 0),
        ("xmax", (2, 2, 2), (8, 7, 7), 0.8, 8),
    ],
)
@pytest.mark.parametrize("real_dtype", [np.dtype("f4"), np.dtype("f8")])
def test_on_plane_electric_edge_patches_receive_half_area(
    face, lower, upper, plane_coordinate, boundary_index, real_dtype
):
    plane = SymmetryPlane(face, 0, plane_coordinate, "pmc")
    closure = ResolvedKSIRClosure("symmetry", (face,), (plane,), True, True)
    surface = build_component_surface(
        "Ey",
        lower,
        upper,
        (0.1, 0.1, 0.1),
        (10, 10, 10),
        excluded_faces=(face,),
        real_dtype=real_dtype,
    )
    adjusted = closure.apply_quadrature(surface)
    tangential_face = adjusted.face("y0")
    on_plane = tangential_face.inside_indices[:, 0] == boundary_index

    assert on_plane.any()
    assert tangential_face.area_weights.dtype == real_dtype
    assert_allclose(tangential_face.area_weights[on_plane], 0.005)
    assert_allclose(tangential_face.area_weights[~on_plane], 0.01)


def _point_source_phasors(surface, frequency, sources):
    positions = surface.patch_positions
    normals = surface.normals
    wavenumber = 2 * np.pi * frequency / c
    field = np.zeros(surface.npatches, dtype=np.complex128)
    derivative = np.zeros_like(field)
    for source_position, amplitude in sources:
        displacement = positions - source_position
        radius = np.linalg.norm(displacement, axis=1)
        radial_direction = displacement / radius[:, np.newaxis]
        source_field = amplitude * np.exp(-1j * wavenumber * radius) / (4 * np.pi * radius)
        radial_derivative = (-1j * wavenumber - 1 / radius) * source_field
        field += source_field
        derivative += radial_derivative * np.sum(normals * radial_direction, axis=1)
    return field[np.newaxis, :], derivative[np.newaxis, :]


def _evaluate_with_closure(surface, field, derivative, closure, frequency, directions, origin):
    """Evaluate all physical and virtual patches using only the core API."""

    result = np.zeros((1, directions.shape[0]), dtype=field.dtype)
    for _, positions, normals, areas, image_field, image_derivative in closure.transformed_faces(
        surface, field, derivative
    ):
        result += evaluate_far_zone_patches(
            positions,
            normals,
            areas,
            [frequency],
            directions,
            image_field,
            image_derivative,
            origin=origin,
        )
    return result


@pytest.mark.parametrize(
    "component,boundary_type,sources",
    [
        ("Ey", "pmc", (((1.1, 1.0, 1.0), 1.0),)),
        (
            "Ex",
            "pmc",
            (
                ((1.3, 1.0, 1.0), 1.0),
                ((0.9, 1.0, 1.0), -1.0),
            ),
        ),
    ],
)
def test_half_surface_image_completion_matches_full_surface(component, boundary_type, sources):
    frequency = 250e6
    spacing = (0.1, 0.1, 0.1)
    shape = (25, 22, 22)
    full = build_component_surface(component, (2, 4, 4), (20, 16, 16), spacing, shape)
    plane = SymmetryPlane("x0", 0, 1.1, boundary_type)
    symmetry = ResolvedKSIRClosure("symmetry", ("x0",), (plane,), True, True)
    half = symmetry.apply_quadrature(
        build_component_surface(
            component,
            (11, 4, 4),
            (20, 16, 16),
            spacing,
            shape,
            excluded_faces=("x0",),
        )
    )
    closed = ResolvedKSIRClosure("closed", (), (), True, True)
    directions = spherical_directions([20, 55, 90, 125, 160], [10, 80, 155, 240, 320], degrees=True)
    full_field, full_derivative = _point_source_phasors(full, frequency, sources)
    half_field, half_derivative = _point_source_phasors(half, frequency, sources)

    expected = _evaluate_with_closure(
        full,
        full_field,
        full_derivative,
        closed,
        frequency,
        directions,
        (1.1, 1.0, 1.0),
    )
    actual = _evaluate_with_closure(
        half,
        half_field,
        half_derivative,
        symmetry,
        frequency,
        directions,
        (1.1, 1.0, 1.0),
    )

    assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_triple_reflection_matches_full_surface_and_uses_eight_images():
    frequency = 300e6
    spacing = (0.1, 0.1, 0.1)
    shape = (24, 24, 24)
    component = "Ex"
    full = build_component_surface(component, (2, 2, 2), (20, 20, 20), spacing, shape)
    planes = (
        SymmetryPlane("x0", 0, 1.1, "pec"),
        SymmetryPlane("y0", 1, 1.1, "pmc"),
        SymmetryPlane("z0", 2, 1.1, "pmc"),
    )
    symmetry = ResolvedKSIRClosure("symmetry", ("x0", "y0", "z0"), planes, True, True)
    octant = symmetry.apply_quadrature(
        build_component_surface(
            component,
            (11, 11, 11),
            (20, 20, 20),
            spacing,
            shape,
            excluded_faces=("x0", "y0", "z0"),
        )
    )
    source = (((1.1, 1.1, 1.1), 1.0),)
    directions = spherical_directions([30, 65, 100, 145], [15, 110, 220, 310], degrees=True)
    full_field, full_derivative = _point_source_phasors(full, frequency, source)
    octant_field, octant_derivative = _point_source_phasors(octant, frequency, source)
    closed = ResolvedKSIRClosure("closed", (), (), True, True)

    expected = _evaluate_with_closure(
        full,
        full_field,
        full_derivative,
        closed,
        frequency,
        directions,
        (1.1, 1.1, 1.1),
    )
    actual = _evaluate_with_closure(
        octant,
        octant_field,
        octant_derivative,
        symmetry,
        frequency,
        directions,
        (1.1, 1.1, 1.1),
    )

    assert symmetry.image_count == 8
    assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)


def test_experimental_mask_is_explicitly_open_and_removes_only_requested_faces():
    mask = ExperimentalMask(("zmax",))
    closure = resolve_closure(
        mask,
        {},
        (2, 2, 2),
        (5, 5, 5),
        (8, 8, 8),
        (0.02, 0.02, 0.02),
    )
    surface = build_component_surface(
        "Ex",
        (2, 2, 2),
        (5, 5, 5),
        (0.02, 0.02, 0.02),
        (9, 9, 9),
        excluded_faces=closure.omitted_faces,
    )

    assert closure.name == "experimental_mask"
    assert closure.omitted_faces == ("zmax",)
    assert closure.active_faces == tuple(face for face in FACES if face != "zmax")
    assert not closure.mathematically_closed
    assert not closure.exact
    assert tuple(face.face_id for face in surface.faces) == closure.active_faces


def test_huygens_open_surface_selects_any_nonempty_face_subset_and_round_trips():
    closure = resolve_closure(
        HuygensOpenSurface(("z0", "xmax", "x0")),
        {},
        (2, 2, 2),
        (5, 5, 5),
        (8, 8, 8),
        (0.02, 0.02, 0.02),
    )
    restored = closure_from_metadata(
        closure.name,
        closure.omitted_faces,
        (),
        (),
        (),
    )

    assert closure.name == "huygens_open"
    assert closure.omitted_faces == ("x0", "xmax", "z0")
    assert closure.active_faces == ("y0", "ymax", "zmax")
    assert not closure.mathematically_closed
    assert restored == closure
    with pytest.raises(ValueError, match="unknown faces"):
        HuygensOpenSurface(("feed",))
    with pytest.raises(ValueError, match="leave at least one active face"):
        HuygensOpenSurface(FACES)


def test_saved_symmetry_metadata_reconstructs_the_same_closure():
    closure = resolve_closure(
        SymmetryCompletion(("x0", "zmax")),
        {"x0": "pmc", "zmax": "pec"},
        (0, 2, 2),
        (8, 8, 10),
        (12, 12, 10),
        (0.1, 0.2, 0.3),
    )

    restored = closure_from_metadata(
        closure.name,
        closure.omitted_faces,
        [plane.face for plane in closure.symmetry_planes],
        [plane.boundary_type for plane in closure.symmetry_planes],
        [plane.coordinate for plane in closure.symmetry_planes],
    )

    assert restored == closure
    assert restored.signature == closure.signature


@pytest.mark.parametrize(
    "args,match",
    [
        (("unknown", (), (), (), ()), "unknown saved"),
        (("closed", ("x0",), (), (), ()), "must not omit"),
        (("symmetry", ("x0",), (), (), ()), "requires"),
        (("symmetry", ("x0",), ("y0",), ("pmc",), (0.0,)), "must match"),
        (("experimental_mask", ("zmax",), ("x0",), ("pmc",), (0.0,)), "no symmetry"),
        (("symmetry", ("x0",), ("x0",), ("open",), (0.0,)), "boundary_type"),
    ],
)
def test_saved_closure_metadata_rejects_inconsistent_state(args, match):
    with pytest.raises(ValueError, match=match):
        closure_from_metadata(*args)


@pytest.mark.parametrize(
    "args",
    [
        ("Qx", 0, "pec"),
        ("Ex", 3, "pec"),
        ("Ex", 0, "open"),
    ],
)
def test_component_parity_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        component_parity(*args)


@pytest.mark.parametrize(
    "args,match",
    [
        (("left", 0, 0.0, "pec"), "unknown symmetry face"),
        (("x0", 1, 0.0, "pec"), "must use axis"),
        (("x0", 0, np.inf, "pec"), "finite"),
        (("x0", 0, 0.0, "open"), "boundary_type"),
    ],
)
def test_symmetry_plane_rejects_invalid_geometry(args, match):
    with pytest.raises(ValueError, match=match):
        SymmetryPlane(*args)


@pytest.mark.parametrize("faces", [(), ("x0", "x0"), FACES])
def test_experimental_mask_requires_a_nonempty_proper_face_subset(faces):
    with pytest.raises(ValueError):
        ExperimentalMask(faces)
