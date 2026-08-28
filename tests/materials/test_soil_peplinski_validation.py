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

"""Regression tests for SoilPeplinski validation gaps (Codex-reported):
PeplinskiSoil.calculate_properties() (materials.py) divides by
sand_density (self.rs) and by the water-fraction bin midpoint - zero
sand density, or a water-fraction range collapsed to zero width at
exactly 0, both risk ZeroDivisionError/NaN/Inf. The old validation only
rejected values `< 0`, missing zero sand density, a zero-width/all-zero
water fraction range, a reversed range, and fractions above 1.
"""
from types import SimpleNamespace

import pytest

from gprMax.user_objects.cmds_multiuse import SoilPeplinski


def _fake_grid():
    return SimpleNamespace(mixingmodels=[])


def _build(**overrides):
    kwargs = dict(
        sand_fraction=0.5,
        clay_fraction=0.5,
        bulk_density=2.0,
        sand_density=2.66,
        water_fraction_lower=0.001,
        water_fraction_upper=0.25,
        id="soil1",
    )
    kwargs.update(overrides)
    SoilPeplinski(**kwargs).build(_fake_grid())


def test_valid_parameters_build_successfully():
    _build()


def test_zero_sand_density_rejected():
    with pytest.raises(ValueError):
        _build(sand_density=0)


def test_zero_water_fraction_range_rejected():
    with pytest.raises(ValueError):
        _build(water_fraction_lower=0, water_fraction_upper=0)


def test_reversed_water_fraction_range_rejected():
    with pytest.raises(ValueError):
        _build(water_fraction_lower=0.3, water_fraction_upper=0.1)


@pytest.mark.parametrize("field", ["sand_fraction", "clay_fraction"])
def test_fraction_above_one_rejected(field):
    with pytest.raises(ValueError):
        _build(**{field: 1.5})
