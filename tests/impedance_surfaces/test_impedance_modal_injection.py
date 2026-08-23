"""Focused coverage for the surface-impedance modal-injection benchmark."""

import logging

import numpy as np
import pytest

from testing.validation.validate_impedance_modal_injection import (
    FAR_MONITOR_X,
    FMAX,
    FMIN,
    GUIDE_HEIGHT,
    GUIDE_WIDTH,
    MAX_ALPHA_RELATIVE_L2_ERROR,
    MAX_SOURCE_REFLECTION_DB,
    MONITOR_SPACING,
    NEAR_MONITOR_X,
    SOURCE_X,
    SURFACE_RESISTANCE,
    TIME_WINDOW,
    analyse_modal_coefficients,
    build_scene,
    earliest_wall_end_round_trip,
    magnitude_db,
    perturbation_alpha,
    run_validation,
    te10_cutoff,
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


def test_benchmark_is_above_cutoff_and_causally_isolated():
    assert FMIN > te10_cutoff()
    assert earliest_wall_end_round_trip() > TIME_WINDOW
    assert SOURCE_X < NEAR_MONITOR_X < FAR_MONITOR_X
    assert MONITOR_SPACING == pytest.approx(FAR_MONITOR_X - NEAR_MONITOR_X)


def test_te10_perturbation_alpha_is_positive_and_linear_in_resistance():
    frequency = np.linspace(FMIN, FMAX, 7)
    alpha = perturbation_alpha(frequency)
    assert np.all(alpha > 0)
    np.testing.assert_allclose(
        perturbation_alpha(frequency, 2 * SURFACE_RESISTANCE),
        2 * alpha,
        rtol=2e-15,
        atol=0,
    )
    with pytest.raises(ValueError, match="above cutoff"):
        perturbation_alpha(np.asarray((te10_cutoff(),)))


def test_modal_coefficient_analysis_recovers_prescribed_mismatch_and_alpha():
    frequency = np.linspace(FMIN, FMAX, 7)
    alpha = perturbation_alpha(frequency)
    phase = np.linspace(0.2, 0.8, frequency.size)
    a1 = np.exp(0.1j * phase)
    reflection = 0.02 * np.exp(-0.3j * phase)
    near = 0.7 * np.exp(-1.2j * phase)
    ratio = np.exp(-alpha * MONITOR_SPACING - 0.4j * phase)
    result = analyse_modal_coefficients(
        frequency,
        a1,
        reflection * a1,
        near,
        near * ratio,
    )
    np.testing.assert_allclose(result["source_reflection"], reflection)
    np.testing.assert_allclose(result["measured_alpha_per_m"], alpha, rtol=2e-14)
    assert result["maximum_source_reflection_db"] == pytest.approx(
        float(magnitude_db(np.asarray((0.02,)))[0])
    )
    assert result["alpha_relative_l2_error"] < 2e-14


def test_build_scene_contains_one_drive_and_three_ports():
    scene = build_scene()
    names = [
        type(item).__name__
        for item in (*scene.grid_objects, *scene.geometry_objects)
    ]
    assert names.count("SurfaceImpedance") == 1
    assert names.count("ImpedanceBox") == 4
    assert names.count("EigenmodePort") == 3
    assert names.count("EigenmodeExcitation") == 1
    assert GUIDE_WIDTH > GUIDE_HEIGHT


@pytest.mark.integration
def test_impedance_modal_injection_has_low_source_reflection(tmp_path):
    summary = run_validation(tmp_path, threads=1)
    assert summary["metrics"]["maximum_source_reflection_db"] < MAX_SOURCE_REFLECTION_DB
    assert summary["metrics"]["alpha_relative_l2_error"] < MAX_ALPHA_RELATIVE_L2_ERROR
    assert summary["acceptance"]["passed"]
    assert (tmp_path / "impedance_modal_injection.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
