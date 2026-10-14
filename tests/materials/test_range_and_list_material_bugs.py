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

"""Material-bin reuse must preserve indexing; missing list entries must fail clearly."""
import pytest

from gprMax.materials import ListMaterial, Material, RangeMaterial


def test_range_material_appends_one_entry_per_bin_even_when_a_later_bin_reuses_a_material():
    materials = []

    class _Grid:
        pass

    grid = _Grid()
    grid.materials = materials
    # Populate the later bin through the range builder, with actual er=2.5
    # rather than a material whose name alone appears to specify that value.
    previous = RangeMaterial("previous", (2.5, 2.5), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0))
    previous.calculate_properties(1, grid)
    existing = grid.materials[previous.matID[0]]

    rm = RangeMaterial(
        ID="range1",
        er_range=(1.0, 3.0),
        se_range=(0.0, 0.0),
        mr_range=(1.0, 1.0),
        sm_range=(0.0, 0.0),
    )
    rm.calculate_properties(2, grid)

    assert len(rm.matID) == 2
    assert rm.matID[1] == existing.numID  # bin1 correctly reused, not dropped
    # bin0 must be a genuinely new material, distinct from the reused one
    assert rm.matID[0] != existing.numID
    new_material = next(m for m in grid.materials if m.numID == rm.matID[0])
    assert new_material.er == 1.5
    assert existing.er == 2.5


def test_list_material_missing_material_raises_valueerror_not_attributeerror():
    class _Grid:
        materials = []

    lm = ListMaterial(ID="list1", listofmaterials=["nonexistent_material"])

    with pytest.raises(ValueError):
        lm.calculate_properties(1, _Grid())


def test_list_material_existing_materials_resolve_correctly():
    existing = Material(numID=5, ID="my_material")

    class _Grid:
        materials = [existing]

    lm = ListMaterial(ID="list1", listofmaterials=["my_material"])
    lm.calculate_properties(1, _Grid())

    assert lm.matID == [5]


def test_list_material_nbins_exceeding_list_length_raises_valueerror():
    existing = Material(numID=5, ID="my_material")

    class _Grid:
        materials = [existing]

    lm = ListMaterial(ID="list1", listofmaterials=["my_material"])

    with pytest.raises(ValueError, match="exceeds list length"):
        lm.calculate_properties(2, _Grid())
    # Rejected call leaves previous state alone
    assert lm.matID == []


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.5, "2", None])
def test_list_material_non_positive_or_non_integer_nbins_raises_valueerror(bad):
    existing = Material(numID=5, ID="my_material")

    class _Grid:
        materials = [existing]

    lm = ListMaterial(ID="list1", listofmaterials=["my_material"])

    with pytest.raises(ValueError, match="positive integer nbins"):
        lm.calculate_properties(bad, _Grid())
    assert lm.matID == []


def test_list_material_missing_material_message_is_in_exception_args():
    class _Grid:
        materials = []

    lm = ListMaterial(ID="list1", listofmaterials=["nonexistent_material"])

    with pytest.raises(ValueError) as excinfo:
        lm.calculate_properties(1, _Grid())
    assert "nonexistent_material" in str(excinfo.value)
    assert excinfo.value.args[0] != ""
