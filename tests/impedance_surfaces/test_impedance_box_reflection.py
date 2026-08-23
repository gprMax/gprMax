"""Analytical and end-to-end coverage for the impedance-box reflection benchmark."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.impedance_surfaces import (
    SurfaceImpedanceModel,
    _check_plane_wave_compatibility,
)
from testing.validation.validate_impedance_box_reflection import (
    ETA0,
    FREQUENCIES,
    ORIENTATIONS,
    algorithmic_impedance,
    algorithmic_reflection,
    benchmark_model,
    run_benchmark,
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


def test_algorithmic_impedance_is_continuous_model_at_bilinear_frequency():
    model = benchmark_model()
    dt = 1.83e-12
    calculated = algorithmic_impedance(model, FREQUENCIES, dt)
    warped_s = 2j / dt * np.tan(np.pi * FREQUENCIES * dt)
    expected = np.empty(FREQUENCIES.shape, dtype=np.complex128)
    identity = np.eye(model.order)
    for index, value in np.ndenumerate(warped_s):
        expected[index] = model.D + model.C @ np.linalg.solve(
            value * identity - model.A, model.B
        )
    np.testing.assert_allclose(calculated, expected, rtol=3e-14, atol=3e-14)


def test_algorithmic_reflection_has_continuous_static_limit():
    resistance = 83.0
    model = SurfaceImpedanceModel("constant", D=resistance)
    calculated = algorithmic_reflection(model, np.asarray((0.0,)), 1.9e-12)[0]
    expected = (resistance - ETA0) / (resistance + ETA0)
    assert calculated == pytest.approx(expected, rel=2e-15, abs=2e-15)


def test_six_orientation_definitions_are_transverse_and_inward_propagating():
    assert set(ORIENTATIONS) == {"-x", "+x", "-y", "+y", "-z", "+z"}
    for orientation in ORIENTATIONS.values():
        propagation = np.asarray(orientation.propagation)
        assert propagation[orientation.normal_axis] == -orientation.normal_sign
        assert np.count_nonzero(propagation) == 1
        assert orientation.electric_axis != orientation.normal_axis
        assert orientation.electric_component in {"Ex", "Ey", "Ez"}


def test_plane_wave_guard_accepts_strict_tfsf_enclosure_and_rejects_unsafe_sources():
    boundary = {(2, 10, 12, 14), (1, 15, 17, 19)}
    vector_wave = SimpleNamespace(axial=0, corners=np.asarray((5, 6, 7, 20, 21, 22)))
    _check_plane_wave_compatibility([vector_wave], boundary)

    touching = SimpleNamespace(axial=0, corners=np.asarray((10, 6, 7, 20, 21, 22)))
    with pytest.raises(ValueError, match="strictly inside its TFSF box"):
        _check_plane_wave_compatibility([touching], boundary)

    axial = SimpleNamespace(axial=1, corners=vector_wave.corners)
    with pytest.raises(ValueError, match="axial plane waves sample the geometry"):
        _check_plane_wave_compatibility([axial], boundary)


@pytest.mark.integration
def test_normal_incidence_reflection_matches_exact_discrete_boundary(tmp_path):
    summary = run_benchmark(tmp_path, orientation_names=("-x",), threads=1)
    metrics = summary["metrics"]["-x"]
    assert metrics["magnitude_rmse"] < 0.005
    assert metrics["phase_rmse_degrees"] < 0.7
    assert metrics["complex_relative_l2_error"] < 0.01
    assert summary["acceptance"]["passed"]
    assert (tmp_path / "mx_reflection.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
