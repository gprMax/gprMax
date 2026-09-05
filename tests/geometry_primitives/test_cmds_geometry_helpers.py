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

"""Unit tests for the shared helpers in
``gprMax/user_objects/cmds_geometry/cmds_geometry.py``.

``check_averaging`` converts the hash-command averaging flag, while the
rasterisation helpers combine per-rank object occupancy.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.user_objects.cmds_geometry.cmds_geometry import (
    check_averaging,
    validate_distributed_geometry_rasterisation,
    validate_geometry_rasterisation,
)


class TestCheckAveraging:
    @pytest.mark.parametrize("value", ["y", "Y"])
    def test_yes_maps_to_true(self, value):
        assert check_averaging(value) is True

    @pytest.mark.parametrize("value", ["n", "N"])
    def test_no_maps_to_false(self, value):
        assert check_averaging(value) is False

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_boolean_values_are_accepted(self, value):
        assert check_averaging(value) is bool(value)

    @pytest.mark.parametrize("value", ["yes", "", 1, None])
    def test_invalid_values_are_rejected(self, value):
        with pytest.raises(ValueError, match="Averaging should be"):
            check_averaging(value)


class _SingleRankComm:
    def allgather(self, value):
        return [value]

    def Allreduce(self, send, receive):
        receive[:] = send


class TestGeometryRasterisationReporting:
    def test_distributed_counts_are_deferred_then_validated(self):
        grid = SimpleNamespace(is_distributed=True, comm=_SingleRankComm())

        validate_geometry_rasterisation(grid, 2, geometry="#sphere")
        validate_distributed_geometry_rasterisation(grid)

        assert grid.geometry_rasterisation_records == [(2, "#sphere")]

    def test_distributed_empty_object_is_rejected(self):
        grid = SimpleNamespace(is_distributed=True, comm=_SingleRankComm())
        validate_geometry_rasterisation(grid, 0, geometry="#sphere")

        with pytest.raises(ValueError, match="does not occupy any Yee cells or faces"):
            validate_distributed_geometry_rasterisation(grid)


pytestmark = pytest.mark.unit
