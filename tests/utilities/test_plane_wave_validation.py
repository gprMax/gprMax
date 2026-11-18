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

"""Unit tests for the plane-wave TFSF background check."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.materials import Material
from gprMax.utilities.validation import _material_signature, validate_plane_wave_tfsf_background

FREE_SPACE = 0
VACUUM = 1
BLOCK = 2
CORNERS = [1, 1, 0, 7, 7, 1]
OWNED_LOWER = [0, 0, 0]
OWNED_UPPER = [10, 10, 2]
PROJECTIONS = np.ones(6)


def _materials():
    free_space = Material(FREE_SPACE, "free_space")
    vacuum = Material(VACUUM, "vacuum")
    block = Material(BLOCK, "block")
    block.er = 4.0
    return [free_space, vacuum, block]


def _grid():
    return SimpleNamespace(
        ID=np.zeros((6, 10, 10, 2), dtype=np.uint32),
        materials=_materials(),
        rigidE=np.zeros((12, 9, 9, 2), dtype=np.int8),
        rigidH=np.zeros((6, 9, 9, 2), dtype=np.int8),
    )


def _grid_3d():
    return SimpleNamespace(
        ID=np.zeros((6, 9, 9, 9), dtype=np.uint32),
        materials=_materials(),
        rigidE=np.zeros((12, 8, 8, 8), dtype=np.int8),
        rigidH=np.zeros((6, 8, 8, 8), dtype=np.int8),
    )


def _validate(grid, background=FREE_SPACE, skip_axis=2, projections=PROJECTIONS):
    validate_plane_wave_tfsf_background(
        grid,
        CORNERS,
        OWNED_LOWER,
        OWNED_UPPER,
        grid.materials[background],
        projections,
        skip_axis,
    )


def test_a_homogeneous_boundary_is_accepted():
    _validate(_grid())


def test_a_differently_named_material_with_the_same_properties_is_accepted():
    grid = _grid()
    grid.ID[:, 1:8, 1:8, 0] = VACUUM

    _validate(grid)


def test_a_different_material_on_the_upper_x_face_is_rejected():
    grid = _grid()
    grid.ID[:, 7, 1:8, 0] = BLOCK

    with pytest.raises(ValueError, match="block"):
        _validate(grid)


def test_a_material_fully_inside_the_box_is_accepted():
    grid = _grid()
    grid.ID[:, 3:5, 3:5, 0] = BLOCK

    _validate(grid)


def test_non_averaging_magnetic_components_on_a_homogeneous_boundary_are_accepted():
    grid = _grid()
    grid.materials[FREE_SPACE].averagable = False
    grid.rigidH[2, 7, 3, 0] = 1  # Hy at the upper x face

    _validate(grid)


def test_non_averaging_electric_components_on_a_homogeneous_boundary_are_accepted():
    grid = _grid()
    grid.materials[FREE_SPACE].averagable = False
    grid.rigidE[8, 7, 3, 0] = 1  # Ez at the upper x face

    _validate(grid)


def test_y_face_hz_correction_uses_the_ex_projection():
    grid = _grid()
    grid.ID[5, 3, 0, 0] = BLOCK  # Hz one cell outside the lower y face
    projections = np.zeros(6)
    projections[5] = 1.0  # Hz is live, but its y-face correction uses Ex.

    _validate(grid, projections=projections)

    projections[0] = 1.0
    with pytest.raises(ValueError, match="block"):
        _validate(grid, projections=projections)


def test_a_different_material_at_the_adjacent_lower_face_node_is_rejected():
    grid = _grid()
    grid.ID[4, 0, 3, 0] = BLOCK  # Hy one cell outside the lower x face

    with pytest.raises(ValueError, match="block"):
        _validate(grid)


def test_a_different_material_at_the_adjacent_lower_face_e_node_is_rejected():
    grid = _grid()
    grid.ID[2, 0, 3, 0] = BLOCK  # Ez one cell outside the lower x face

    with pytest.raises(ValueError, match="block"):
        _validate(grid)


def _validate_3d(grid, background=FREE_SPACE):
    validate_plane_wave_tfsf_background(
        grid,
        [1, 1, 1, 7, 7, 7],
        [0, 0, 0],
        [9, 9, 9],
        grid.materials[background],
        PROJECTIONS,
        -1,
    )


def test_a_differently_named_material_in_3d_is_accepted():
    grid = _grid_3d()
    grid.ID[:, 1:8, 1:8, 1:8] = VACUUM

    _validate_3d(grid)


def test_a_3d_material_on_the_boundary_is_rejected():
    grid = _grid_3d()
    grid.ID[:, 7, 1:8, 1:8] = BLOCK

    with pytest.raises(ValueError, match="block"):
        _validate_3d(grid)


def test_a_3d_enclosed_scatterer_is_accepted():
    grid = _grid_3d()
    grid.ID[:, 3:5, 3:5, 3:5] = BLOCK

    _validate_3d(grid)


def test_dispersive_pole_order_does_not_change_the_signature(monkeypatch):
    class _PoleMaterial:
        def __init__(self, name, poles):
            self.ID = name
            self.poles = poles
            self.directional_materials = None

    def fake_entry(material):
        return {
            "name": material.ID,
            "model": "debye",
            "base": {},
            "averagable": True,
            "poles": material.poles,
        }

    monkeypatch.setattr("gprMax.material_database.material_to_database_entry", fake_entry)
    first = _PoleMaterial("a", [{"e": 1.0, "t": 2.0}, {"e": 3.0, "t": 4.0}])
    second = _PoleMaterial("b", [{"e": 3.0, "t": 4.0}, {"e": 1.0, "t": 2.0}])

    assert _material_signature(first) == _material_signature(second)
