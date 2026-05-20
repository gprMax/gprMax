# Tests for analytical_solutions.py
# Tests the hertzian_dipole_fs function for output shape,
# physical validity, and edge cases.

import numpy as np
import pytest

from tests.analytical_solutions import hertzian_dipole_fs


# Shared test parameters
ITERATIONS = 100
DT = 1e-11
DXDYDZ = (0.001, 0.001, 0.001)
RX = (0.1, 0.0, 0.1)


class TestOutputShape:
    """Tests for output shape and type."""

    def test_returns_numpy_array(self):
        result = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, RX)
        assert isinstance(result, np.ndarray)

    def test_output_shape(self):
        """Output should have shape (iterations, 6) for 6 field components."""
        result = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, RX)
        assert result.shape == (ITERATIONS, 6)

    def test_output_shape_scales_with_iterations(self):
        result = hertzian_dipole_fs(200, DT, DXDYDZ, RX)
        assert result.shape == (200, 6)


class TestFieldValidity:
    """Tests for physical validity of field values."""

    def test_no_nan_values(self):
        result = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, RX)
        assert not np.any(np.isnan(result)), "Fields contain NaN values"

    def test_no_inf_values(self):
        result = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, RX)
        assert not np.any(np.isinf(result)), "Fields contain Inf values"

    def test_hz_is_always_zero(self):
        """Hz (index 5) is always zero for a z-directed Hertzian dipole."""
        result = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, RX)
        assert np.all(result[:, 5] == 0), "Hz field should always be zero"

    def test_fields_not_all_zero(self):
        """At least some field values should be nonzero."""
        result = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, RX)
        assert np.any(result != 0), "All field values are zero"


class TestPhysics:
    """Tests for physical behaviour of the dipole solution."""

    def test_field_decreases_with_distance(self):
        """Ez field should be stronger closer to the dipole."""
        rx_near = (0.05, 0.0, 0.05)
        rx_far = (0.2, 0.0, 0.2)

        result_near = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, rx_near)
        result_far = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, rx_far)

        max_near = np.amax(np.abs(result_near[:, 2]))  # Ez
        max_far = np.amax(np.abs(result_far[:, 2]))    # Ez

        assert max_near > max_far, "Field should be stronger closer to the dipole"

    def test_symmetry_ex_ey_on_diagonal(self):
        """For receiver on the diagonal (x=y), Ex and Ey should be equal."""
        rx_diag = (0.1, 0.1, 0.1)
        result = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, rx_diag)
        np.testing.assert_allclose(result[:, 0], result[:, 1], rtol=1e-5)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_iteration(self):
        result = hertzian_dipole_fs(1, DT, DXDYDZ, RX)
        assert result.shape == (1, 6)

    def test_z_zero_receiver(self):
        """z=0 receiver should not cause errors."""
        rx_z0 = (0.1, 0.0, 0.0)
        result = hertzian_dipole_fs(ITERATIONS, DT, DXDYDZ, rx_z0)
        assert result.shape == (ITERATIONS, 6)
        assert not np.any(np.isnan(result))

    def test_nonuniform_spatial_resolution(self):
        dxdydz = (0.002, 0.001, 0.003)
        result = hertzian_dipole_fs(ITERATIONS, DT, dxdydz, RX)
        assert result.shape == (ITERATIONS, 6)
        assert not np.any(np.isnan(result))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])