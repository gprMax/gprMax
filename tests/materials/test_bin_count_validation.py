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

"""Bin-count validation for fractal material mixtures."""

import numpy as np
import pytest

from gprMax.materials import CrimMixture, ListMaterial, PeplinskiSoil, RangeMaterial

pytestmark = pytest.mark.unit


def _range():
    return RangeMaterial("rg", (2.0, 6.0), (0.0, 0.1), (1.0, 1.0), (0.0, 0.0))


def _peplinski():
    return PeplinskiSoil("soil", 0.5, 0.5, 2.0, 2.66, (0.001, 0.25))


def _crim():
    return CrimMixture(
        ID="wetsand",
        matrix_id="sand",
        matrix_fraction=0.6,
        dispersive_id="water",
        fraction_lower=0.02,
        fraction_upper=0.35,
        f_min=1e6,
        f_max=3e9,
        a=0.5,
    )


class TestBinCount:
    @pytest.mark.parametrize("nbins", [0, -1, 2.5, "3", None, True])
    def test_range_rejects_bad_bin_counts(self, fake_grid, nbins):
        with pytest.raises(ValueError, match="positive integer bin count"):
            _range().calculate_properties(nbins, fake_grid(materials=[]))

    @pytest.mark.parametrize("nbins", [0, -2, 1.5, "4", None, False])
    def test_peplinski_rejects_bad_bin_counts(self, fake_grid, nbins):
        with pytest.raises(ValueError, match="positive integer bin count"):
            _peplinski().calculate_properties(nbins, fake_grid(materials=[]))

    @pytest.mark.parametrize("nbins", [0, -1, 2.0, "8", None, True])
    def test_crim_rejects_bad_bin_counts(self, make_material, make_dispersive, fake_grid, nbins):
        matrix = make_material(ID="sand", er=5.0, se=0.0)
        water = make_dispersive(ID="water", model="debye", er=4.9, se=0.0, poles=[(73.2, 9.231e-12, 0.0)])
        with pytest.raises(ValueError, match="positive integer bin count"):
            _crim().calculate_properties(nbins, fake_grid(materials=[matrix, water]))

    def test_list_rejects_more_bins_than_materials(self, make_material, fake_grid):
        grid = fake_grid(materials=[make_material(ID="a", numID=0)])
        with pytest.raises(ValueError, match="contains 1 material"):
            ListMaterial("li", ["a"]).calculate_properties(5, grid)
    @pytest.mark.parametrize("nbins", [0, -1, 2.5, "3", None, True])
    def test_list_rejects_bad_bin_counts(self, make_material, fake_grid, nbins):
        grid = fake_grid(materials=[make_material(ID="a", numID=0)])
        with pytest.raises(ValueError, match="positive integer bin count"):
            ListMaterial("li", ["a"]).calculate_properties(nbins, grid)

    def test_numpy_integer_bin_count_still_accepted(self, fake_grid):
        mixture = _range()
        mixture.calculate_properties(np.int64(3), fake_grid(materials=[]))
        assert len(mixture.matID) == 3


class TestValidBehaviourPreserved:
    def test_range_builds_one_material_per_bin(self, fake_grid):
        mixture = _range()
        mixture.calculate_properties(3, fake_grid(materials=[]))
        assert len(mixture.matID) == 3

    def test_list_builds_exact_length_mapping(self, make_material, fake_grid):
        grid = fake_grid(materials=[make_material(ID="a", numID=0)])
        mixture = ListMaterial("li", ["a"])
        mixture.calculate_properties(1, grid)
        assert mixture.matID == [0]

    def test_peplinski_builds_one_material_per_bin(self, fake_grid):
        mixture = _peplinski()
        mixture.calculate_properties(2, fake_grid(materials=[]))
        assert len(mixture.matID) == 2
