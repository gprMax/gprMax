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

"""Unit tests for voltage-source S11 and impedance calculations."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.materials import Material
from gprMax.ntff.conventions import engineering_dft
from gprMax.ports import (
    _finite_source_gap_admittance,
    _safe_complex_divide,
    admittance_from_s11,
    correct_s11_for_parallel_gap,
    engineering_rfft,
    impedance_from_s11,
    minimum_wavelength_sampling,
    validate_spectrum_limit,
)


@pytest.fixture
def port_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={"float_or_double": np.float64, "complex": np.complex128},
            em_consts={
                "c": 299792458.0,
                "e0": 8.8541878128e-12,
                "m0": 1.25663706212e-6,
            },
        ),
    )


@pytest.mark.parametrize("value, expected", [(10, 10.0), (12.5, 12.5), ("nyquist", "nyquist")])
def test_spectrum_limit_accepts_numeric_or_explicit_nyquist(value, expected):
    assert validate_spectrum_limit(value) == expected


@pytest.mark.parametrize("value", [2.9, 0, -1, np.inf, np.nan, True, "full"])
def test_spectrum_limit_rejects_ambiguous_or_nonphysical_values(value):
    with pytest.raises(ValueError):
        validate_spectrum_limit(value)


def test_engineering_rfft_matches_reference_dft(port_config):
    dt = 2.5e-11
    time_offset = 0.5 * dt
    samples = np.linspace(-0.8, 1.2, 17)

    frequencies, spectrum = engineering_rfft(samples, dt, time_offset=time_offset)
    reference = engineering_dft(
        samples,
        frequencies,
        dt,
        time_offset=time_offset,
    )

    np.testing.assert_allclose(spectrum, reference, rtol=1e-13, atol=1e-24)


def test_gap_deembedding_recovers_known_load_and_impedance(port_config):
    reference_impedance = 50.0
    antenna_impedance = np.asarray([75 + 20j, 32 - 8j], dtype=np.complex128)
    background_admittance = np.asarray([0.002 + 0.003j, 0.005 + 0.001j])
    total_admittance = 1 / antenna_impedance + background_admittance
    source_impedance = 1 / total_admittance
    source_s11 = (source_impedance - reference_impedance) / (source_impedance + reference_impedance)

    corrected_s11, correction_valid = correct_s11_for_parallel_gap(
        source_s11,
        reference_impedance * background_admittance,
    )
    corrected_impedance, impedance_valid = impedance_from_s11(corrected_s11, reference_impedance)
    corrected_admittance, admittance_valid = admittance_from_s11(corrected_s11, reference_impedance)

    assert correction_valid.all()
    assert impedance_valid.all()
    assert admittance_valid.all()
    np.testing.assert_allclose(corrected_impedance, antenna_impedance, rtol=1e-13)
    np.testing.assert_allclose(corrected_admittance, 1 / antenna_impedance, rtol=1e-13)


def test_dispersive_gap_admittance_uses_complete_complex_permittivity(port_config):
    dt = 1e-11
    frequency = np.asarray([0.0, 2e9, 8e9])
    area = 2e-6
    dl = 1e-3
    conductivity = 0.2
    epsilon0 = config.sim_config.em_consts["e0"]

    def calculate_er(values):
        omega = 2 * np.pi * np.asarray(values)
        return 3.5 + 5 / (1 + 1j * omega * 8e-11) + conductivity / (1j * omega * epsilon0)

    output = SimpleNamespace(
        background_is_dispersive=True,
        background_material=SimpleNamespace(calculate_er=calculate_er),
        background_conductance=conductivity * area / dl,
        area=area,
        dl=dl,
    )
    actual = _finite_source_gap_admittance(output, frequency, dt, np.complex128)

    omega_discrete = (2 / dt) * np.tan(np.pi * frequency[1:] * dt)
    expected = np.empty(frequency.shape, dtype=np.complex128)
    expected[0] = conductivity * area / dl
    expected[1:] = 1j * omega_discrete * epsilon0 * calculate_er(omega_discrete / (2 * np.pi)) * area / dl
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-18)


def test_safe_complex_divide_keeps_independent_frequency_bins(port_config):
    numerator = np.asarray([1 + 0j, 1 + 0j, 1 + 0j], dtype=np.complex64)
    denominator = np.asarray([1 + 0j, 1e30 + 0j, 0 + 0j], dtype=np.complex64)

    result, valid = _safe_complex_divide(numerator, denominator, np.complex64)

    np.testing.assert_array_equal(valid, (True, True, False))
    np.testing.assert_allclose(result[:2], (1, 1e-30), rtol=1e-6)
    assert np.isnan(result[2])


def test_open_and_short_keep_s11_valid_but_mask_singular_secondary_quantity(port_config):
    s11 = np.asarray([1 + 0j, -1 + 0j])

    impedance, impedance_valid = impedance_from_s11(s11, 50.0)
    admittance, admittance_valid = admittance_from_s11(s11, 50.0)

    assert not impedance_valid[0]
    assert np.isnan(impedance[0])
    assert impedance_valid[1]
    assert impedance[1] == 0
    assert admittance_valid[0]
    assert admittance[0] == 0
    assert not admittance_valid[1]
    assert np.isnan(admittance[1])


def test_default_material_limit_uses_shortest_wavelength(port_config):
    free_space = Material(0, "free_space")
    high_er = Material(1, "high_er")
    high_er.er = 9
    generated_source = Material(2, "generated_source")
    generated_source.er = 1000
    generated_source.type = "voltage-source"
    grid = SimpleNamespace(
        dx=0.01,
        dy=0.005,
        dz=0.002,
        materials=[free_space, high_er, generated_source],
    )

    cells, limiting = minimum_wavelength_sampling(grid, np.asarray([0.0, 1e9]))

    assert np.isinf(cells[0])
    assert limiting[1] == "high_er"
    assert cells[1] == pytest.approx(299792458.0 / (1e9 * 3 * 0.01))
