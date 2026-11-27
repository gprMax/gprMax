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

"""Regression tests for triangles and cylindrical sectors that round just
below an MPI rank's lower face.

On a 1 mm grid with the rank starting at 9 mm, a lower coordinate of
8.49 mm and a thickness of 0.49 mm discretise to level -1 and thickness 0.
check_thickness used to clamp that empty intersection to level 0, so both
primitives built an extra zero-thickness sheet on the lower halo plane
that the serial model does not contain.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from gprMax import config
from gprMax.user_objects.cmds_geometry import cylindrical_sector as sector_module
from gprMax.user_objects.cmds_geometry import triangle as triangle_module

DL = 1e-3
RANK_START = 9
GLOBAL_CELLS = 30
LOCAL_CELLS = 20
LOWER = 8.49e-3
THICKNESS = 0.49e-3

_MATERIAL = SimpleNamespace(numID=2, averagable=False, is_pec=False)


class _MPIGrid(SimpleNamespace):
    def __init__(self):
        lower_extent = np.full(3, RANK_START, dtype=np.int32)
        super().__init__(
            dl=np.full(3, DL),
            size=np.full(3, LOCAL_CELLS, dtype=np.int32),
            global_size=np.full(3, GLOBAL_CELLS, dtype=np.int32),
            lower_extent=lower_extent,
            is_distributed=True,
            averagevolumeobjects=True,
            geometry_rasterisation_records=[],
            dx=DL,
            dy=DL,
            dz=DL,
            solid=np.zeros((LOCAL_CELLS, LOCAL_CELLS, LOCAL_CELLS), dtype=np.uint32),
            rigidE=np.zeros((12, LOCAL_CELLS, LOCAL_CELLS, LOCAL_CELLS), dtype=np.int8),
            rigidH=np.zeros((6, LOCAL_CELLS, LOCAL_CELLS, LOCAL_CELLS), dtype=np.int8),
            ID=np.zeros(
                (6, LOCAL_CELLS + 1, LOCAL_CELLS + 1, LOCAL_CELLS + 1), dtype=np.uint32
            ),
        )

    def global_to_local_coordinate(self, point):
        return point - self.lower_extent

    def local_to_global_coordinate(self, point):
        return point + self.lower_extent

    def within_bounds(self, local_point):
        global_point = self.local_to_global_coordinate(local_point)
        for index, dimension in enumerate("xyz"):
            if global_point[index] < 0 or global_point[index] > self.global_size[index]:
                raise ValueError(dimension)
        return True


def _triangle_points(axis):
    transverse = [index for index in range(3) if index != axis]
    p1 = [0.010, 0.010, 0.010]
    p2 = [0.010, 0.010, 0.010]
    p3 = [0.010, 0.010, 0.010]
    p1[axis] = p2[axis] = p3[axis] = LOWER
    p2[transverse[0]] = 0.012
    p3[transverse[1]] = 0.012
    return tuple(p1), tuple(p2), tuple(p3)


@pytest.mark.parametrize("axis", range(3))
def test_triangle_below_lower_rank_face_is_not_built(monkeypatch, axis):
    grid = _MPIGrid()
    p1, p2, p3 = _triangle_points(axis)
    calls = []

    monkeypatch.setattr(
        triangle_module, "resolve_geometry_materials", lambda *args, **kwargs: [_MATERIAL]
    )
    monkeypatch.setattr(
        triangle_module, "build_triangle", lambda *args, **kwargs: calls.append(args) or 0
    )

    triangle_module.Triangle(p1=p1, p2=p2, p3=p3, thickness=THICKNESS, material_id="metal").build(
        grid
    )

    assert calls == []
    assert grid.geometry_rasterisation_records


@pytest.mark.parametrize("axis", range(3))
def test_cylindrical_sector_below_lower_rank_face_is_not_built(monkeypatch, axis):
    grid = _MPIGrid()
    calls = []

    monkeypatch.setattr(
        sector_module, "resolve_geometry_materials", lambda *args, **kwargs: [_MATERIAL]
    )
    monkeypatch.setattr(
        sector_module,
        "build_cylindrical_sector",
        lambda *args, **kwargs: calls.append(args) or 0,
    )

    with mock.patch.object(
        config, "get_model_config", return_value=SimpleNamespace(mode="3D")
    ):
        sector_module.CylindricalSector(
            normal="xyz"[axis],
            ctr1=0.010,
            ctr2=0.010,
            extent1=LOWER,
            extent2=LOWER + THICKNESS,
            r=0.004,
            start=0,
            end=180,
            material_id="metal",
        ).build(grid)

    assert calls == []
    assert grid.geometry_rasterisation_records
