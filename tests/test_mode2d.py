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

"""Shared live-plane geometry for reduced two-dimensional modes."""

import pytest

from gprMax.mode2d import mode2d_geometry


@pytest.mark.parametrize(
    "mode,axis,live,electric,magnetic,strides",
    (
        ("2D TMx", 0, 0, ("Ex",), ("Hy", "Hz"), (1, 1, 1)),
        ("2D TMy", 1, 0, ("Ey",), ("Hx", "Hz"), (1, 1, 1)),
        ("2D TMz", 2, 0, ("Ez",), ("Hx", "Hy"), (1, 1, 1)),
        ("2D TEx", 0, 1, ("Ey", "Ez"), ("Hx",), (0, 1, 1)),
        ("2D TEy", 1, 1, ("Ex", "Ez"), ("Hy",), (1, 0, 1)),
        ("2D TEz", 2, 1, ("Ex", "Ey"), ("Hz",), (1, 1, 0)),
    ),
)
def test_mode2d_geometry(mode, axis, live, electric, magnetic, strides):
    geometry = mode2d_geometry(mode)

    assert geometry.invariant_axis == axis
    assert geometry.live_index == live
    assert geometry.active_electric == electric
    assert geometry.active_magnetic == magnetic
    assert geometry.collocation_strides == strides


def test_mode2d_geometry_returns_none_for_3d():
    assert mode2d_geometry("3D") is None
