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

``check_averaging`` converts the hash-command averaging flag.
"""

import pytest

from gprMax.user_objects.cmds_geometry.cmds_geometry import check_averaging


class TestCheckAveraging:
    @pytest.mark.parametrize("value", ["y", "Y"])
    def test_yes_maps_to_true(self, value):
        assert check_averaging(value) is True

    @pytest.mark.parametrize("value", ["n", "N"])
    def test_no_maps_to_false(self, value):
        assert check_averaging(value) is False


pytestmark = pytest.mark.unit
