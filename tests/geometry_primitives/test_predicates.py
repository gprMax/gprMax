"""Unit tests for the pure geometric predicates in
``gprMax/cython/geometry_primitives.pyx``.

These helpers underpin the shape rasterisers: ``are_clockwise``,
``is_within_radius`` and ``is_inside_sector`` decide which cells a
cylindrical sector covers, and ``point_in_polygon`` serves the
fractal-surface machinery. All are deterministic maths with no side
effects, so every test is a straight input → boolean check.

The functions take C ``float`` (single-precision) arguments, so test
values are chosen to be exactly representable in float32 and to sit
well clear of decision boundaries where the trigonometry of the
sector arms is inexact.
"""

import numpy as np
import pytest

from gprMax.cython.geometry_primitives import (
    are_clockwise,
    is_inside_sector,
    is_within_radius,
    point_in_polygon,
)

PI = np.pi


class TestAreClockwise:
    """``are_clockwise(v1x, v1y, v2x, v2y)`` is True iff v2 lies strictly
    clockwise of v1 (negative z-component of the 2D cross product)."""

    @pytest.mark.parametrize(
        "v1, v2, expected",
        [
            # v2 anti-clockwise of v1 -> False
            ((1.0, 0.0), (0.0, 1.0), False),
            # v2 clockwise of v1 -> True
            ((0.0, 1.0), (1.0, 0.0), True),
            ((1.0, 1.0), (1.0, -1.0), True),
            # Collinear (same direction) — cross product is exactly zero,
            # and the comparison is strict, so not clockwise.
            ((1.0, 0.0), (2.0, 0.0), False),
            # Anti-parallel — still zero cross product.
            ((1.0, 0.0), (-1.0, 0.0), False),
            # Zero vector never registers as clockwise.
            ((0.0, 0.0), (1.0, 1.0), False),
            ((1.0, 1.0), (0.0, 0.0), False),
        ],
    )
    def test_sign_convention(self, v1, v2, expected):
        assert are_clockwise(v1[0], v1[1], v2[0], v2[1]) is expected


class TestIsWithinRadius:
    """``is_within_radius(vx, vy, radius)`` — inclusive circle test on the
    vector from the circle centre to the point."""

    @pytest.mark.parametrize(
        "vx, vy, radius, expected",
        [
            (0.0, 0.0, 1.0, True),  # centre always inside
            (3.0, 4.0, 5.0, True),  # exactly on the boundary (3-4-5) — inclusive
            (3.0, 4.0, 4.5, False),  # just outside
            (1.0, 1.0, 1.5, True),  # sqrt(2) < 1.5
            (0.0, 0.0, 0.0, True),  # zero radius keeps only the centre
            (0.5, 0.0, 0.0, False),
        ],
    )
    def test_membership(self, vx, vy, radius, expected):
        assert is_within_radius(vx, vy, radius) is expected


class TestIsInsideSector:
    """``is_inside_sector(px, py, ctrx, ctry, start, angle, radius)``.

    The sector is defined anti-clockwise from the start arm. Sectors up
    to pi use the AND of the two arm tests; reflex sectors (> pi) use
    the OR branch. Points are placed at 45-degree diagonals so the
    verdict is robust to float32 trig error in the arm coordinates.
    """

    @pytest.mark.parametrize(
        "px, py, expected",
        [
            (0.5, 0.5, True),  # 45 deg — inside quadrant I
            (-0.5, 0.5, False),  # 135 deg — past the end arm
            (0.5, -0.5, False),  # -45 deg — behind the start arm
            (-0.5, -0.5, False),  # 225 deg
            (1.5, 1.5, False),  # right direction, outside the radius
        ],
    )
    def test_quarter_sector_at_origin(self, px, py, expected):
        # Start arm on +x axis, opening pi/2 towards +y: quadrant I.
        assert is_inside_sector(px, py, 0.0, 0.0, 0.0, PI / 2, 1.0) is expected

    @pytest.mark.parametrize(
        "px, py, expected",
        [
            (0.0, 0.5, True),  # straight up — inside the upper half-plane
            (0.5, 0.5, True),
            (-0.5, 0.5, True),
            (0.0, -0.5, False),  # lower half-plane
            (0.5, -0.5, False),
        ],
    )
    def test_half_plane_sector(self, px, py, expected):
        # angle == pi is the largest sector still using the AND branch.
        assert is_inside_sector(px, py, 0.0, 0.0, 0.0, PI, 1.0) is expected

    @pytest.mark.parametrize(
        "px, py, expected",
        [
            (0.5, 0.5, True),  # 45 deg
            (-0.5, 0.5, True),  # 135 deg
            (-0.5, -0.5, True),  # 225 deg
            (0.5, -0.5, False),  # 315 deg — the excluded quadrant
            (-1.5, 1.5, False),  # inside the arc but outside the radius
        ],
    )
    def test_reflex_sector_uses_or_branch(self, px, py, expected):
        # 270-degree sector from the +x axis: quadrants I, II, III.
        assert is_inside_sector(px, py, 0.0, 0.0, 0.0, 3 * PI / 2, 1.0) is expected

    @pytest.mark.parametrize(
        "px, py, expected",
        [
            (-0.5, 0.5, True),  # quadrant II
            (0.5, 0.5, False),  # quadrant I — behind the start arm
        ],
    )
    def test_nonzero_start_angle(self, px, py, expected):
        # Start arm on +y axis, opening pi/2 towards -x: quadrant II.
        assert is_inside_sector(px, py, 0.0, 0.0, PI / 2, PI / 2, 1.0) is expected

    def test_offset_centre(self):
        # The point is tested relative to (ctrx, ctry), not the origin.
        assert is_inside_sector(2.5, 3.5, 2.0, 3.0, 0.0, PI / 2, 1.0) is True
        assert is_inside_sector(1.5, 3.5, 2.0, 3.0, 0.0, PI / 2, 1.0) is False

    def test_radius_limits_diagonal_point(self):
        # (0.5, 0.5) is 0.707 from the centre — outside a 0.5 radius.
        assert is_inside_sector(0.5, 0.5, 0.0, 0.0, 0.0, PI / 2, 0.5) is False


SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
TRIANGLE = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
# L-shape: unit-4 square with the (x < 2, y > 2) corner notched out.
L_SHAPE = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 4.0), (2.0, 2.0), (0.0, 2.0)]


class TestPointInPolygon:
    """``point_in_polygon(px, py, polycoords)`` — ray-casting with explicit
    vertex and horizontal-boundary short-circuits."""

    @pytest.mark.parametrize(
        "px, py, expected",
        [
            (0.5, 0.5, True),  # dead centre
            (1.5, 0.5, False),  # right of the square
            (0.5, -0.5, False),  # below
            (100.0, 100.0, False),  # far field
        ],
    )
    def test_square_interior_and_exterior(self, px, py, expected):
        assert point_in_polygon(px, py, SQUARE) is expected

    def test_vertex_counts_as_inside(self):
        assert point_in_polygon(1.0, 1.0, SQUARE) is True
        assert point_in_polygon(0.0, 0.0, SQUARE) is True

    def test_point_on_horizontal_edge_counts_as_inside(self):
        # Bottom edge y == 0 and top edge y == 1 both hit the explicit
        # boundary check (p1y == p2y == py with px strictly between).
        assert point_in_polygon(0.5, 0.0, SQUARE) is True
        assert point_in_polygon(0.5, 1.0, SQUARE) is True

    @pytest.mark.parametrize(
        "px, py, expected",
        [
            (1.0, 0.5, True),  # centred low in the triangle
            (1.9, 1.9, False),  # beyond the right slanted edge
            (0.2, 1.5, False),  # beyond the left slanted edge
            (1.0, 2.5, False),  # above the apex
        ],
    )
    def test_triangle(self, px, py, expected):
        assert point_in_polygon(px, py, TRIANGLE) is expected

    @pytest.mark.parametrize(
        "px, py, expected",
        [
            (1.0, 1.0, True),  # bottom-left arm of the L
            (3.0, 3.0, True),  # right column of the L
            (1.0, 3.0, False),  # inside the notch
            (5.0, 3.0, False),  # right of everything
        ],
    )
    def test_concave_polygon(self, px, py, expected):
        # Concavity exercises the multi-crossing path of the ray-caster.
        assert point_in_polygon(px, py, L_SHAPE) is expected
