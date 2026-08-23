"""Focused coverage for the copper rectangular-guide S21 validation."""

import logging

import numpy as np
import pytest

from testing.validation.validate_impedance_copper_waveguide_s21 import (
    ANCHORS,
    FMAX,
    FMIN,
    GUIDE_HEIGHT,
    GUIDE_WIDTH,
    MAX_ATTENUATION_RELATIVE_L2_ERROR,
    MAX_SOURCE_REFLECTION_DB,
    METAL_PRESET,
    PORT1_X,
    PORT2_X,
    REFERENCE_PLANE_SPACING,
    SOURCE_X,
    TIME_WINDOW,
    analyse_s21,
    build_scene,
    continuum_beta,
    copper_surface_impedance,
    earliest_wall_end_round_trip,
    magnitude_db,
    next_rectangular_mode_cutoff,
    perturbation_coefficient,
    run_validation,
    te10_cutoff,
    theoretical_s21,
    yee_beta,
)


@pytest.fixture(autouse=True)
def restore_package_logging():
    """Do not leak ``gprMax.run``'s application logger into later tests."""

    yield
    logger = logging.getLogger("gprMax")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_copper_benchmark_is_above_cutoff_and_causally_isolated():
    assert FMIN > te10_cutoff()
    assert FMAX < next_rectangular_mode_cutoff()
    assert max(ANCHORS) < next_rectangular_mode_cutoff()
    assert earliest_wall_end_round_trip() > TIME_WINDOW
    assert SOURCE_X < PORT1_X < PORT2_X
    assert REFERENCE_PLANE_SPACING == pytest.approx(PORT2_X - PORT1_X)
    assert GUIDE_WIDTH > GUIDE_HEIGHT


def test_copper_impedance_drives_equal_first_order_loss_and_phase_shift():
    frequency = np.linspace(FMIN, FMAX, 9)
    impedance = copper_surface_impedance(frequency)
    np.testing.assert_allclose(impedance.real, impedance.imag, rtol=2e-15)
    assert np.all(np.diff(impedance.real) > 0)

    coefficient = perturbation_coefficient(frequency)
    predicted = theoretical_s21(frequency, numerical_dispersion=False)
    alpha = -np.log(np.abs(predicted)) / REFERENCE_PLANE_SPACING
    phase_correction = np.angle(
        predicted / np.exp(-1j * continuum_beta(frequency) * REFERENCE_PLANE_SPACING)
    )
    np.testing.assert_allclose(alpha, coefficient * impedance.real, rtol=2e-13)
    np.testing.assert_allclose(
        phase_correction,
        -coefficient * impedance.imag * REFERENCE_PLANE_SPACING,
        rtol=1e-11,
        atol=2e-16,
    )
    insertion_loss_db = -magnitude_db(predicted)
    assert np.min(insertion_loss_db) >= 0.1


def test_yee_reference_exposes_the_continuum_phase_error():
    frequency = np.linspace(FMIN, FMAX, 11)
    assert np.all(yee_beta(frequency) > continuum_beta(frequency))
    assert np.max(
        np.abs(
            np.angle(
                theoretical_s21(frequency)
                / theoretical_s21(frequency, numerical_dispersion=False)
            )
        )
    ) > np.deg2rad(2)


def test_s21_analysis_recovers_an_exact_complex_reference():
    frequency = np.linspace(FMIN, FMAX, 11)
    analytical = theoretical_s21(frequency)
    near = 0.73 * np.exp(1j * np.linspace(0.1, 0.7, frequency.size))
    result = analyse_s21(frequency, near, near * analytical)
    np.testing.assert_allclose(result["measured_s21"], analytical, rtol=4e-16)
    assert result["maximum_magnitude_error_db"] < 2e-15
    assert result["maximum_phase_error_deg"] < 2e-14
    assert result["attenuation_relative_l2_error"] < 2e-13
    assert result["complex_relative_l2_error"] < 2e-16


def test_build_scene_uses_copper_and_two_passive_reference_ports():
    scene = build_scene()
    objects = (*scene.grid_objects, *scene.geometry_objects)
    names = [type(item).__name__ for item in objects]
    surface = next(item for item in objects if type(item).__name__ == "SurfaceImpedance")
    assert surface.preset == METAL_PRESET
    assert names.count("SurfaceImpedance") == 1
    assert names.count("ImpedanceBox") == 4
    assert names.count("EigenmodePort") == 3
    assert names.count("EigenmodeExcitation") == 1


@pytest.mark.integration
def test_copper_waveguide_fdfd_loss_and_injection_match_theory(tmp_path):
    summary = run_validation(tmp_path, threads=4)
    metrics = summary["metrics"]
    assert metrics["maximum_source_reflection_db"] < MAX_SOURCE_REFLECTION_DB
    assert (
        metrics["fdfd_physical_alpha_relative_l2_error"]
        < MAX_ATTENUATION_RELATIVE_L2_ERROR
    )
    assert metrics["physical_insertion_loss_db_min"] >= 0.1
    assert summary["reference_plane_spacing_m"] == pytest.approx(
        REFERENCE_PLANE_SPACING
    )
    assert summary["acceptance"]["passed"]
    assert (tmp_path / "impedance_copper_waveguide_s21.csv").is_file()
    assert (tmp_path / "impedance_copper_waveguide_s21.png").is_file()
    assert (tmp_path / "summary.json").is_file()
