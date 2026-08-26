"""Focused coverage for the copper-wall waveguide validation."""

import logging

import numpy as np
import pytest

from testing.validation.impedance_surface import _wall_waveguide_common as plot_common
from testing.validation.impedance_surface.validate_copper_wall_waveguide import (
    DFT_POINTS,
    DOMAIN,
    EXCITATION_FMAX,
    EXCITATION_FMIN,
    EXCITATION_TRANSITION,
    FMAX,
    FMIN,
    GUIDE_HEIGHT,
    GUIDE_WIDTH,
    MAX_ATTENUATION_RELATIVE_L2_ERROR,
    MAX_FDTD_ATTENUATION_RELATIVE_L2_ERROR,
    MAX_SOURCE_REFLECTION_DB,
    METAL_PRESET,
    PORT1_X,
    PORT2_X,
    PROPAGATION_ANCHORS,
    REFERENCE_PLANE_SPACING,
    SOURCE_ANCHORS,
    SOURCE_X,
    TIME_WINDOW,
    VALIDATION_POINTS,
    analyse_s21,
    build_scene,
    continuum_beta,
    copper_surface_impedance,
    earliest_wall_end_round_trip,
    fitted_surface_impedance,
    magnitude_db,
    next_rectangular_mode_cutoff,
    perturbation_coefficient,
    run_validation,
    te10_cutoff,
    theoretical_s21,
    wall_surface_impedance,
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
    assert DOMAIN == pytest.approx((0.210, 0.0028, 0.002))
    assert FMIN > te10_cutoff()
    assert FMAX < next_rectangular_mode_cutoff()
    assert max(SOURCE_ANCHORS) < next_rectangular_mode_cutoff()
    assert TIME_WINDOW == pytest.approx(0.500e-9)
    assert earliest_wall_end_round_trip() - TIME_WINDOW > 80e-12
    assert SOURCE_X == pytest.approx(0.090)
    assert PORT1_X == pytest.approx(0.105)
    assert PORT2_X == pytest.approx(0.145)
    assert SOURCE_X < PORT1_X < PORT2_X
    assert REFERENCE_PLANE_SPACING == pytest.approx(PORT2_X - PORT1_X)
    assert GUIDE_WIDTH > GUIDE_HEIGHT
    assert EXCITATION_FMIN == pytest.approx(120e9)
    assert EXCITATION_FMAX == pytest.approx(150e9)
    assert EXCITATION_TRANSITION == pytest.approx(20e9)
    assert DFT_POINTS == 31
    assert VALIDATION_POINTS == 21
    assert len(SOURCE_ANCHORS) == len(set(SOURCE_ANCHORS)) == 35
    assert np.all(np.diff(SOURCE_ANCHORS) > 0)
    assert len([frequency for frequency in SOURCE_ANCHORS if FMIN <= frequency <= FMAX]) == 21
    assert len(PROPAGATION_ANCHORS) == len(set(PROPAGATION_ANCHORS)) == 11
    assert np.all(np.diff(PROPAGATION_ANCHORS) > 0)
    assert (
        len(
            [
                frequency
                for frequency in PROPAGATION_ANCHORS
                if EXCITATION_FMIN <= frequency <= EXCITATION_FMAX
            ]
        )
        == 7
    )


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


def test_copper_foster_fit_tracks_actual_surface_impedance():
    surface = wall_surface_impedance()
    assert surface.fit_order == "auto"
    assert surface.fit_pole_count == 3
    assert surface.fit_meets_tolerance
    frequency = np.linspace(FMIN, FMAX, 101)
    actual = copper_surface_impedance(frequency)
    fitted = fitted_surface_impedance(frequency)
    assert np.linalg.norm(fitted - actual) / np.linalg.norm(actual) < 2e-3


def test_yee_reference_exposes_the_continuum_phase_error():
    frequency = np.linspace(FMIN, FMAX, 11)
    assert np.all(yee_beta(frequency) > continuum_beta(frequency))
    assert np.max(
        np.abs(
            np.angle(
                theoretical_s21(frequency) / theoretical_s21(frequency, numerical_dispersion=False)
            )
        )
    ) > np.deg2rad(2)


def test_s21_analysis_recovers_an_exact_complex_reference():
    frequency = np.linspace(FMIN, FMAX, 11)
    analytical = theoretical_s21(frequency)
    near = 0.73 * np.exp(1j * np.linspace(0.1, 0.7, frequency.size))
    result = analyse_s21(frequency, near, near * analytical)
    np.testing.assert_allclose(result["measured_s21"], analytical, rtol=4e-16)
    assert result["maximum_magnitude_error_db"] < 4e-15
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
    assert names.count("Box") == 4
    assert names.count("EigenmodePort") == 3
    assert names.count("EigenmodeExcitation") == 1
    domain = next(item for item in scene.single_use_objects if type(item).__name__ == "Domain")
    time_window = next(
        item for item in scene.single_use_objects if type(item).__name__ == "TimeWindow"
    )
    band = next(item for item in scene.grid_objects if type(item).__name__ == "EigenmodeBand")
    ports = sorted(
        (item for item in scene.grid_objects if type(item).__name__ == "EigenmodePort"),
        key=lambda item: item.kwargs["port"],
    )
    assert domain.domain_size == pytest.approx(DOMAIN)
    assert time_window.time == pytest.approx(TIME_WINDOW)
    assert band.kwargs["fmin"] == pytest.approx(EXCITATION_FMIN)
    assert band.kwargs["fmax"] == pytest.approx(EXCITATION_FMAX)
    assert band.kwargs["points"] == DFT_POINTS
    assert band.kwargs["transition"] == pytest.approx(EXCITATION_TRANSITION)
    np.testing.assert_array_equal(ports[0].kwargs["anchors"], SOURCE_ANCHORS)
    np.testing.assert_array_equal(ports[1].kwargs["anchors"], PROPAGATION_ANCHORS)
    np.testing.assert_array_equal(ports[2].kwargs["anchors"], PROPAGATION_ANCHORS)


def test_common_plot_reports_impedance_s11_fdfd_and_fdtd(tmp_path, monkeypatch):
    frequency = np.asarray((130e9, 140e9, 150e9))
    actual_impedance = np.asarray((0.09 + 0.09j, 0.10 + 0.10j, 0.11 + 0.11j))
    fitted_impedance = actual_impedance * (1 + 1e-4j)
    fdfd_alpha = np.asarray((0.67, 0.62, 0.59))
    fdtd_alpha = np.asarray((0.69, 0.64, 0.61))
    result = {
        "frequency_hz": frequency,
        "impedance_frequency_hz": frequency,
        "target_impedance_ohm": actual_impedance,
        "fitted_impedance_ohm": fitted_impedance,
        "source_reflection_db": np.asarray((-35.0, -40.0, -45.0)),
        "fdfd_theory_alpha_per_m": np.asarray((0.68, 0.63, 0.60)),
        "fdfd_alpha_per_m": fdfd_alpha,
        "fdtd_theory_alpha_per_m": np.asarray((0.68, 0.63, 0.60)),
        "fdtd_alpha_per_m": fdtd_alpha,
    }
    captured = {}
    original_subplots = plot_common.plt.subplots
    original_close = plot_common.plt.close

    def record_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        captured["figure"] = figure
        captured["axes"] = axes
        return figure, axes

    monkeypatch.setattr(plot_common.plt, "subplots", record_subplots)
    monkeypatch.setattr(plot_common.plt, "close", lambda figure: None)
    output = tmp_path / "copper_wall_waveguide.png"
    try:
        plot_common.plot_wall_waveguide_validation(
            output,
            result,
            title="Copper-wall rectangular waveguide",
            maximum_source_reflection_db=MAX_SOURCE_REFLECTION_DB,
        )
        axes = captured["axes"]
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), actual_impedance.real)
        np.testing.assert_array_equal(axes[0].lines[1].get_ydata(), fitted_impedance.real)
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), result["source_reflection_db"])
        np.testing.assert_array_equal(axes[2].lines[1].get_ydata(), fdfd_alpha)
        np.testing.assert_array_equal(axes[3].lines[1].get_ydata(), fdtd_alpha)
        assert axes[2].get_ylim()[0] == pytest.approx(0)
        assert axes[3].get_ylim()[0] == pytest.approx(0)
        assert output.is_file()
        assert output.stat().st_size > 1000
    finally:
        if "figure" in captured:
            original_close(captured["figure"])


@pytest.mark.integration
def test_copper_wall_waveguide_fdfd_and_fdtd_outputs(tmp_path):
    summary = run_validation(tmp_path, threads=4)
    metrics = summary["metrics"]
    assert metrics["maximum_source_reflection_db"] < MAX_SOURCE_REFLECTION_DB
    assert metrics["fdfd_physical_alpha_relative_l2_error"] < MAX_ATTENUATION_RELATIVE_L2_ERROR
    assert metrics["fdtd_physical_alpha_relative_l2_error"] < MAX_FDTD_ATTENUATION_RELATIVE_L2_ERROR
    assert metrics["physical_insertion_loss_db_min"] >= 0.1
    assert summary["reference_plane_spacing_m"] == pytest.approx(REFERENCE_PLANE_SPACING)
    assert summary["fit_order"] == "auto"
    assert summary["fit_pole_count"] == 3
    assert summary["modal_dft_points"] == 31
    assert summary["source_modal_anchor_points"] == 35
    assert summary["passive_modal_anchor_points"] == 11
    assert summary["fdtd_integration_duration_s"] >= TIME_WINDOW
    assert summary["wall_return_margin_s"] > 80e-12
    assert set(summary["acceptance"]["checks"]) == {
        "source_reflection",
        "fdfd_physical_attenuation",
        "fdtd_physical_attenuation",
    }
    assert summary["acceptance"]["passed"]
    assert (tmp_path / "copper_wall_waveguide.csv").is_file()
    assert (tmp_path / "copper_wall_waveguide.png").is_file()
    assert (tmp_path / "summary.json").is_file()
