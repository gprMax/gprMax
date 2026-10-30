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

"""Asking for a material in an official database that has none yet says so."""

import pytest

from gprMax.material_database import load_material_spec


@pytest.mark.parametrize("database", ["gpr", "antenna", "bioem"])
def test_a_reserved_database_says_it_has_no_materials_yet(database):
    with pytest.raises(
        KeyError, match="has no materials yet .*reserved for curated material models"
    ):
        load_material_spec(database, "concrete")


def test_a_database_with_materials_lists_them():
    with pytest.raises(KeyError, match="Available entries: pec, pmc, vacuum"):
        load_material_spec("fundamental", "concrete")
