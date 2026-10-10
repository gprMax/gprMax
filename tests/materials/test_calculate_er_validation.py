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

"""Tests for frequency validation in permittivity calculations."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestBaseMaterialFrequency:
    @pytest.mark.parametrize("freq", [0, 0.0, -1e9, np.inf, -np.inf, np.nan, None, "bad", [], 1e9 + 1j])
    def test_rejects_non_positive_or_non_finite_frequency(self, make_material, freq):
        m = make_material(er=6.0, se=0.5)
        with pytest.raises(ValueError, match="finite evaluation frequency greater than zero"):
            m.calculate_er(freq)

    @pytest.mark.parametrize(
        "freq",
        [
            np.array([1e9, 0.0]),
            np.array([1e9, -2e9]),
            np.array([1e9, np.nan]),
            np.array([1e9 + 1j]),
        ],
    )
    def test_rejects_arrays_containing_bad_frequencies(self, make_material, freq):
        m = make_material(er=6.0, se=0.5)
        with pytest.raises(ValueError, match="finite evaluation frequency greater than zero"):
            m.calculate_er(freq)

    def test_accepts_positive_frequency(self, make_material):
        m = make_material(er=6.0, se=0.5)
        assert m.calculate_er(1e9) == 6.0


class TestDispersiveFrequency:
    @pytest.mark.parametrize("freq", [0, 0.0, -1e9, np.inf, -np.inf, np.nan, None, "bad", [], 1e9 + 1j])
    def test_debye_rejects_non_positive_or_non_finite_frequency(self, make_dispersive, freq):
        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        with pytest.raises(ValueError, match="finite evaluation frequency greater than zero"):
            m.calculate_er(freq)

    @pytest.mark.parametrize("model", ["lorentz", "drude"])
    @pytest.mark.parametrize("freq", [0, -1e9, np.inf, np.nan])
    def test_other_families_reject_bad_frequencies(self, make_dispersive, model, freq):
        m = make_dispersive(model=model, er=2.0, se=0.0, poles=[(1.0, 1e9, 0.5e9)])
        with pytest.raises(ValueError, match="finite evaluation frequency greater than zero"):
            m.calculate_er(freq)

    def test_rejects_array_containing_zero(self, make_dispersive):
        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        with pytest.raises(ValueError, match="finite evaluation frequency greater than zero"):
            m.calculate_er(np.array([1e9, 0.0, 2e9]))


class TestValidBehaviourPreserved:
    def test_debye_matches_closed_form_at_positive_frequency(self, make_dispersive):
        from gprMax import config

        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        frequency = 1e9
        w = 2 * np.pi * frequency
        expected = 2.0 + 0.1 / (1j * w * config.e0) + 1.0 / (1 + 1j * w * 1e-12)
        assert m.calculate_er(frequency) == pytest.approx(expected)

    def test_conductive_debye_has_passive_sign_at_positive_frequency(self, make_dispersive):
        """Ensure conductivity produces the expected loss sign."""
        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        assert m.calculate_er(1e9).imag < 0
