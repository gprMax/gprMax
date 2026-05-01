# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import pytest

from gprMax.utilities import round_value, round32


class TestRoundValueInteger:
    """Tests for round_value with decimalplaces=0 (ROUND_HALF_DOWN mode)."""

    @pytest.mark.parametrize("value, expected", [
        # Half values are rounded downwards
        (1.5, 1),
        (2.5, 2),
        (0.5, 0),
        (3.5, 3),
        (99.5, 99),
        (100.5, 100),
        # Below half rounds down
        (1.4, 1),
        (0.4, 0),
        (2.4, 2),
        # Above half rounds up
        (1.6, 2),
        (0.6, 1),
        (2.6, 3),
        # Exact integers remain unchanged
        (0, 0),
        (1, 1),
        (3, 3),
        (100, 100),
        # Negative half values round towards zero (ROUND_HALF_DOWN)
        (-1.5, -1),
        (-2.5, -2),
        (-0.5, 0),
        # Negative non-half values
        (-1.4, -1),
        (-1.6, -2),
        (-3, -3),
    ])
    def test_integer_rounding(self, value, expected):
        assert round_value(value) == expected

    @pytest.mark.parametrize("value, expected", [
        (1.5, 1),
        (2.5, 2),
        (-1.5, -1),
    ])
    def test_explicit_zero_decimalplaces(self, value, expected):
        """Passing decimalplaces=0 explicitly should behave identically."""
        assert round_value(value, decimalplaces=0) == expected

    def test_return_type_is_int(self):
        result = round_value(1.5)
        assert isinstance(result, int)

    def test_return_type_is_int_for_whole_number(self):
        result = round_value(3.0)
        assert isinstance(result, int)


class TestRoundValueDecimal:
    """Tests for round_value with decimalplaces > 0 (ROUND_FLOOR mode)."""

    @pytest.mark.parametrize("value, decimalplaces, expected", [
        # 2 decimal places - positive values floor down
        (1.239, 2, 1.23),
        (1.231, 2, 1.23),
        (1.235, 2, 1.23),
        (0.999, 2, 0.99),
        (0.001, 2, 0.0),
        # 2 decimal places - negative values floor away from zero
        (-1.239, 2, -1.24),
        (-1.231, 2, -1.24),
        (-0.001, 2, -0.01),
        # 1 decimal place
        (1.9, 1, 1.9),
        (1.99, 1, 1.9),
        (1.95, 1, 1.9),
        (1.11, 1, 1.1),
        (-1.11, 1, -1.2),
        # 3 decimal places
        (1.2345, 3, 1.234),
        (1.2349, 3, 1.234),
        (-1.2345, 3, -1.235),
        # Exact values remain unchanged
        (1.23, 2, 1.23),
        (0.0, 2, 0.0),
    ])
    def test_decimal_rounding(self, value, decimalplaces, expected):
        assert round_value(value, decimalplaces=decimalplaces) == expected

    def test_return_type_is_float(self):
        result = round_value(1.239, decimalplaces=2)
        assert isinstance(result, float)

    def test_return_type_is_float_for_whole_number(self):
        result = round_value(1.0, decimalplaces=1)
        assert isinstance(result, float)


class TestRound32:
    """Tests for round32 - rounds up to nearest multiple of 32."""

    @pytest.mark.parametrize("value, expected", [
        # Exact multiples of 32 remain unchanged
        (0, 0),
        (32, 32),
        (64, 64),
        (96, 96),
        (128, 128),
    ])
    def test_exact_multiples(self, value, expected):
        assert round32(value) == expected

    @pytest.mark.parametrize("value, expected", [
        # Values round up to the next multiple of 32
        (1, 32),
        (16, 32),
        (31, 32),
        (33, 64),
        (63, 64),
        (65, 96),
        (100, 128),
    ])
    def test_rounds_up(self, value, expected):
        assert round32(value) == expected

    @pytest.mark.parametrize("value, expected", [
        # Negative values - np.ceil rounds towards zero
        (-1, 0),
        (-31, 0),
        (-32, -32),
        (-33, -32),
        (-64, -64),
        (-65, -64),
    ])
    def test_negative_values(self, value, expected):
        assert round32(value) == expected

    @pytest.mark.parametrize("value, expected", [
        # Float inputs
        (32.0, 32),
        (32.1, 64),
        (31.9, 32),
        (0.1, 32),
    ])
    def test_float_inputs(self, value, expected):
        assert round32(value) == expected

    def test_return_type_is_int(self):
        result = round32(33)
        assert isinstance(result, int)

    def test_return_type_is_int_for_float_input(self):
        result = round32(33.5)
        assert isinstance(result, int)
