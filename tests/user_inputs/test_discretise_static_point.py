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

"""discretise_static_point() must never leak a raw decimal error.

Every user coordinate funnels through np.vectorize(round_int); these tests
pin the clean-ValueError contract at the funnel, independent of individual
commands.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.user_inputs import MainGridUserInput

INF = float("inf")

pytestmark = pytest.mark.unit


def _uip():
    return MainGridUserInput(SimpleNamespace(dl=np.array([1e-3, 1e-3, 1e-3])))


@pytest.mark.parametrize("point", [(INF, 0.0, 0.0), (0.0, -INF, 0.0), (0.0, 0.0, INF)])
def test_inf_point_raises_the_clean_valueerror(point):
    with pytest.raises(ValueError, match="round_int requires a finite value"):
        _uip().discretise_static_point(point)


def test_nan_point_raises_the_clean_valueerror():
    with pytest.raises(ValueError, match="round_int requires a finite value"):
        _uip().discretise_static_point((0.0, float("nan"), 0.0))


def test_finite_point_is_unchanged():
    np.testing.assert_array_equal(
        _uip().discretise_static_point((1.6e-3, 0.4e-3, 2.4e-3)), [2, 0, 2]
    )


def test_round_to_grid_static_point_shares_the_contract():
    with pytest.raises(ValueError, match="round_int requires a finite value"):
        _uip().round_to_grid_static_point((INF, 0.0, 0.0))
