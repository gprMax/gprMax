"""End-to-end checks for the reusable-KSIR validation utilities."""

import numpy as np
import pytest

from testing.validation.validate_ntff import (
    _solver_options,
    run_accelerator_parity,
    run_dipole_validation,
    run_mie_sweep_validation,
    run_near_field_validation,
)

try:
    import pycuda.driver as _cuda_driver

    _cuda_driver.init()
    HAS_CUDA = _cuda_driver.Device.count() > 0
except Exception:
    HAS_CUDA = False


def test_validation_solver_options_cover_all_backends():
    assert _solver_options("cpu", 0, "double") == {"cpu_precision": "double"}
    assert _solver_options("cuda", 2, "single")["gpu"] == [2]
    assert _solver_options("opencl", 1, "single")["opencl"] == [1]
    assert _solver_options("metal", 3, "single")["metal"] == [3]


def test_cpu_dipole_validation_matches_closed_form(tmp_path):
    result = run_dipole_validation(
        tmp_path,
        backend="cpu",
        device=0,
        precision="double",
        threads=2,
    )

    assert result["collection_backend"] == "cython_openmp"
    assert result["e_plane_rms_error"] < 0.01
    assert result["e_plane_maximum_error"] < 0.02
    assert result["pole_maximum"] < 1e-8
    assert result["h_plane_rms_deviation_from_unity"] < 0.01


def test_cpu_near_field_validation_matches_direct_receivers(tmp_path):
    result = run_near_field_validation(
        tmp_path,
        backend="cpu",
        device=0,
        precision="double",
        threads=2,
    )

    assert result["collection_backend"] == "cython_openmp"
    assert np.all(np.diff(result["distance_m"]) > 0)
    for metrics in result["metrics"]:
        assert metrics["relative_l2_error_significant"] < 0.05
        assert metrics["correlation_significant"] > 0.999
        assert metrics["cross_correlation_lag_samples"] == 0


def test_cpu_mie_sweep_exercises_backscatter_resonances(tmp_path):
    result = run_mie_sweep_validation(
        tmp_path,
        precision="double",
        threads=2,
    )

    assert result["collection_backend"] == "cython_openmp"
    assert np.all(np.isfinite(result["simulated_rcs_m2"]))
    assert np.all(result["simulated_rcs_m2"] > 0)
    assert result["overall_rms_error_db"] < 5
    assert result["band_errors"][0]["rms_error_db"] < 2
    assert len(result["simulated_resonances"]["peaks"]) > 0
    assert len(result["simulated_resonances"]["nulls"]) > 0


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
def test_cuda_time_domain_validation_matches_cpu(tmp_path):
    result = run_accelerator_parity(
        tmp_path,
        backend="cuda",
        device=0,
        precision="single",
        threads=2,
    )

    assert result["cpu_collection_backend"] == "cython_openmp"
    assert result["device_collection_backend"] == "cuda_device"
    for metrics in result["point_metrics"]:
        assert metrics["relative_l2_error"] < 1e-4
        assert metrics["maximum_error_normalized_to_peak"] < 1e-4
