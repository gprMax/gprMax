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
    @pytest.mark.parametrize("freq", [0, 0.0, -1e9, 1e9])
    def test_accepts_dc_negative_and_positive_frequency(self, make_material, freq):
        m = make_material(er=6.0, se=0.5)
        assert m.calculate_er(freq) == 6.0

    def test_accepts_array_with_dc_and_negative_frequencies(self, make_material):
        m = make_material(er=6.0, se=0.5)
        result = m.calculate_er(np.array([-2e9, 0.0, 1e9]))
        assert result == 6.0

    @pytest.mark.parametrize("freq", [np.inf, -np.inf, np.nan, None, "bad", [], 1e9 + 1j])
    def test_rejects_non_finite_or_invalid_frequency(self, make_material, freq):
        m = make_material(er=6.0, se=0.5)
        with pytest.raises(ValueError, match="finite real evaluation frequency"):
            m.calculate_er(freq)

    @pytest.mark.parametrize(
        "freq",
        [
            np.array([1e9, np.nan]),
            np.array([1e9, np.inf]),
            np.array([1e9 + 1j]),
        ],
    )
    def test_rejects_arrays_containing_bad_frequencies(self, make_material, freq):
        m = make_material(er=6.0, se=0.5)
        with pytest.raises(ValueError, match="finite real evaluation frequency"):
            m.calculate_er(freq)


class TestDispersiveDCBehaviour:
    def test_conductive_debye_dc_raises_clear_error(self, make_dispersive):
        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        with pytest.raises(ValueError, match="singular at zero frequency"):
            m.calculate_er(0)

    def test_conductive_debye_dc_array_raises_clear_error(self, make_dispersive):
        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        with pytest.raises(ValueError, match="singular at zero frequency"):
            m.calculate_er(np.array([1e9, 0.0, 2e9]))

    def test_lossless_debye_dc_returns_static_permittivity(self, make_dispersive):
        m = make_dispersive(model="debye", er=2.0, se=0.0, poles=[(1.0, 1e-12, 0.0)])
        assert m.calculate_er(0) == pytest.approx(3.0)

    def test_lossless_lorentz_dc_returns_static_permittivity(self, make_dispersive):
        m = make_dispersive(model="lorentz", er=2.0, se=0.0, poles=[(1.0, 1e9, 0.5e9)])
        assert m.calculate_er(0) == pytest.approx(3.0)

    def test_drude_dc_raises_clear_error(self, make_dispersive):
        m = make_dispersive(model="drude", er=2.0, se=0.0, poles=[(1.0, 1e9, 0.5e9)])
        with pytest.raises(ValueError, match="singular at zero frequency"):
            m.calculate_er(0)

    def test_lossless_debye_dc_array_stays_finite(self, make_dispersive):
        m = make_dispersive(model="debye", er=2.0, se=0.0, poles=[(1.0, 1e-12, 0.0)])
        result = m.calculate_er(np.array([0.0, 1e9]))
        assert np.all(np.isfinite(result))
        assert result[0] == pytest.approx(3.0)

    @pytest.mark.parametrize("freq", [np.inf, -np.inf, np.nan, None, "bad", [], 1e9 + 1j])
    def test_dispersive_rejects_non_finite_or_invalid_frequency(self, make_dispersive, freq):
        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        with pytest.raises(ValueError, match="finite real evaluation frequency"):
            m.calculate_er(freq)


class TestConjugateSymmetry:
    @pytest.mark.parametrize(
        "model, poles",
        [
            ("debye", [(1.0, 1e-12, 0.0)]),
            ("lorentz", [(1.0, 1e9, 0.5e9)]),
            ("drude", [(1.0, 2e9, 0.5e9)]),
        ],
    )
    def test_negative_frequency_is_conjugate_of_positive(self, make_dispersive, model, poles):
        m = make_dispersive(model=model, er=2.0, se=0.1, poles=poles)
        frequency = 1.5e9
        assert m.calculate_er(-frequency) == pytest.approx(np.conj(m.calculate_er(frequency)))

    @pytest.mark.parametrize(
        "model, poles",
        [
            ("debye", [(1.0, 1e-12, 0.0)]),
            ("lorentz", [(1.0, 1e9, 0.5e9)]),
            ("drude", [(1.0, 2e9, 0.5e9)]),
        ],
    )
    def test_mixed_sign_array_preserves_symmetry(self, make_dispersive, model, poles):
        m = make_dispersive(model=model, er=2.0, se=0.1, poles=poles)
        frequencies = np.array([-2e9, -0.5e9, 0.5e9, 2e9])
        result = m.calculate_er(frequencies)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, np.conj(result[::-1]), rtol=1e-12)


class TestValidBehaviourPreserved:
    def test_debye_matches_closed_form_at_positive_frequency(self, make_dispersive):
        from gprMax import config

        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        frequency = 1e9
        w = 2 * np.pi * frequency
        expected = 2.0 + 0.1 / (1j * w * config.e0) + 1.0 / (1 + 1j * w * 1e-12)
        assert m.calculate_er(frequency) == pytest.approx(expected)

    def test_conductive_debye_has_passive_sign_at_positive_frequency(self, make_dispersive):
        """Conductivity must contribute negative imaginary permittivity."""
        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        assert m.calculate_er(1e9).imag < 0

    def test_conductive_debye_has_passive_sign_at_negative_frequency(self, make_dispersive):
        """Loss sign flips with frequency so the response stays passive."""
        m = make_dispersive(model="debye", er=2.0, se=0.1, poles=[(1.0, 1e-12, 0.0)])
        assert m.calculate_er(-1e9).imag > 0
