"""Unit tests for voltage-source S11 and impedance calculations."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.materials import Material
from gprMax.ntff.conventions import engineering_dft
from gprMax.ports import (
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
