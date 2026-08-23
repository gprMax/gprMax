"""Unit tests for the shared helpers in
``gprMax/user_objects/cmds_geometry/cmds_geometry.py``.

``check_averaging`` converts the hash-command averaging flag,
``rotate_point`` / ``rotate_2point_object`` implement the rotation the
``RotatableMixin`` classes apply before rasterising, and
``rotate_polarisation`` remaps a point-plus-polarisation object (used
by sources/receivers-style rotations). All pure functions.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.user_objects.cmds_geometry.cmds_geometry import (
    check_averaging,
    rotate_2point_object,
    rotate_point,
    rotate_polarisation,
)


class TestCheckAveraging:
    @pytest.mark.parametrize("value", ["y", "Y"])
    def test_yes_maps_to_true(self, value):
        assert check_averaging(value) is True

    @pytest.mark.parametrize("value", ["n", "N"])
    def test_no_maps_to_false(self, value):
        assert check_averaging(value) is False


class TestRotatePoint:
    def test_90_degrees_about_z(self):
        p = rotate_point(np.array([1.0, 0.0, 0.0]), "z", 90)
        assert np.allclose(p, [0.0, 1.0, 0.0])

    def test_90_degrees_about_x(self):
        p = rotate_point(np.array([0.0, 1.0, 0.0]), "x", 90)
        assert np.allclose(p, [0.0, 0.0, 1.0])

    def test_90_degrees_about_y(self):
        p = rotate_point(np.array([0.0, 0.0, 1.0]), "y", 90)
        assert np.allclose(p, [1.0, 0.0, 0.0])

    def test_rotation_about_an_offset_origin(self):
        # (2, 1, 0) is one unit along +x from origin (1, 1, 0); rotating
        # 90 degrees about z moves it one unit along +y from that origin.
        p = rotate_point(np.array([2.0, 1.0, 0.0]), "z", 90, origin=(1.0, 1.0, 0.0))
        assert np.allclose(p, [1.0, 2.0, 0.0])

    def test_360_degrees_is_identity(self):
        p = rotate_point(np.array([0.3, 0.7, 0.2]), "z", 360)
        assert np.allclose(p, [0.3, 0.7, 0.2])


class TestRotate2PointObject:
    def test_90_about_z_with_explicit_origin(self):
        # An x-aligned segment from (2, 2, 2) to (5, 2, 2) rotated about
        # its first point becomes y-aligned; the result is re-sorted to
        # (lower-left, upper-right) form.
        pts = np.array([[2.0, 2.0, 2.0], [5.0, 2.0, 2.0]])
        new_pts = rotate_2point_object(pts, "z", 90, origin=(2.0, 2.0, 0.0))
        assert np.allclose(new_pts, [[2.0, 2.0, 2.0], [2.0, 5.0, 2.0]])

    def test_default_origin_is_the_object_centre(self):
        # Rotating a box 90 degrees about z around its own centre swaps
        # its x- and y-extents while keeping the centre fixed.
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]])
        new_pts = rotate_2point_object(pts, "z", 90)
        assert np.allclose(new_pts, [[-1.0, 1.0, 0.0], [3.0, 3.0, 0.0]])

    def test_coordinates_along_the_rotation_axis_are_preserved(self):
        # Rotation about x cannot change the x-extents of the object.
        pts = np.array([[1.0, 2.0, 2.0], [3.0, 5.0, 2.0]])
        new_pts = rotate_2point_object(pts, "x", 90)
        assert np.isclose(new_pts[0, 0], 1.0)
        assert np.isclose(new_pts[1, 0], 3.0)

    def test_result_is_sorted_lower_left_to_upper_right(self):
        pts = np.array([[2.0, 2.0, 0.0], [6.0, 4.0, 0.0]])
        new_pts = rotate_2point_object(pts, "z", 180)
        assert np.all(new_pts[0] <= new_pts[1])

    @pytest.mark.parametrize("angle", [45, 91, 30])
    def test_non_multiple_of_90_raises(self, angle):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with pytest.raises(ValueError):
            rotate_2point_object(pts, "z", angle)

    @pytest.mark.parametrize("angle", [-90, 450])
    def test_angle_outside_0_360_raises(self, angle):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with pytest.raises(ValueError):
            rotate_2point_object(pts, "z", angle)

    def test_invalid_axis_raises(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with pytest.raises(ValueError):
            rotate_2point_object(pts, "w", 90)


class TestRotatePolarisation:
    GRID = SimpleNamespace(dx=0.001, dy=0.002, dz=0.003)

    @pytest.mark.parametrize(
        "polarisation, axis, expected",
        [
            ("x", "y", "z"),
            ("x", "z", "y"),
            ("y", "x", "z"),
            ("y", "z", "x"),
            ("z", "x", "y"),
            ("z", "y", "x"),
        ],
    )
    def test_90_degree_polarisation_remap(self, polarisation, axis, expected):
        _, new_polarisation = rotate_polarisation(
            (0.01, 0.02, 0.03), polarisation, axis, 90, self.GRID
        )
        assert new_polarisation == expected

    def test_returns_point_pair_one_cell_along_the_polarisation(self):
        # The second point extends the first by one cell in the current
        # polarisation direction — the segment the rotation then acts on.
        pts, _ = rotate_polarisation((0.01, 0.02, 0.03), "y", "z", 90, self.GRID)
        assert np.allclose(pts[0], [0.01, 0.02, 0.03])
        assert np.allclose(pts[1], [0.01, 0.022, 0.03])

    def test_uppercase_polarisation_accepted(self):
        _, new_polarisation = rotate_polarisation((0.0, 0.0, 0.0), "X", "z", 90, self.GRID)
        assert new_polarisation == "y"


pytestmark = pytest.mark.unit
