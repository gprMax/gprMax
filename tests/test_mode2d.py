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
