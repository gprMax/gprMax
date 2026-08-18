"""Mass-based 1 g/10 g SAR spatial-averaging tests."""

import numpy as np
import pytest

from gprMax.sar_averaging import (
    INVALID,
    VALID,
    _bounds_for_cube,
    _centered_shells_touch_tissue,
    _spatial_average_sar_python,
    spatial_average_sar,
)


@pytest.mark.parametrize(
    "orientation,axis,expected_lower,expected_upper",
    (
        (0, 0, 0.5, 2.5),
        (1, 0, -0.5, 1.5),
        (2, 1, 0.5, 2.5),
        (3, 1, -0.5, 1.5),
        (4, 2, 0.5, 2.5),
        (5, 2, -0.5, 1.5),
    ),
)
def test_face_orientation_uses_iec_negative_then_positive_order(
    orientation, axis, expected_lower, expected_upper
):
    lower, upper = _bounds_for_cube(np.ones(3), 2.0, orientation, np.ones(3))

    assert lower[axis] == pytest.approx(expected_lower)
    assert upper[axis] == pytest.approx(expected_upper)


def test_centered_cube_checks_every_intermediate_shell_face():
    tissue = np.ones((7, 7, 7), dtype=bool)
    tissue[2, 2:5, 2:5] = False

    assert not _centered_shells_touch_tissue(tissue, np.ones(3), (3, 3, 3), 4.0, 0)
    tissue[2, 3, 3] = True
    assert _centered_shells_touch_tissue(tissue, np.ones(3), (3, 3, 3), 4.0, 0)


@pytest.mark.parametrize("target_mass", (0.001, 0.01))
def test_uniform_tissue_preserves_constant_local_sar(target_mass):
    density = np.full((14, 14, 14), 1000.0)
    local_sar = np.full(density.shape, 2.5)

    result = spatial_average_sar(
        density,
        local_sar,
        (0.002, 0.002, 0.002),
        target_mass,
    )

    assert result.peak_sar == pytest.approx(2.5)
    np.testing.assert_allclose(result.sar[np.isfinite(result.sar)], 2.5)
    np.testing.assert_allclose(
        result.averaging_mass[np.isfinite(result.averaging_mass)],
        target_mass,
        rtol=2e-9,
    )


def test_fractional_cells_use_piecewise_constant_density_and_mass_weighted_sar():
    density = np.full((10, 10, 10), 1000.0)
    density[5:, :, :] = 2000.0
    local_sar = np.ones_like(density)
    local_sar[5:, :, :] = 3.0

    result = spatial_average_sar(density, local_sar, (0.001, 0.001, 0.001), 0.001)

    cell = (4, 5, 5)
    side = result.averaging_volume[cell] ** (1 / 3)
    upper_x = (cell[0] + 0.5) * 0.001 + 0.5 * side
    high_length = max(0.0, upper_x - 0.005)
    low_length = side - high_length
    expected = (1000 * low_length * 1 + 2000 * high_length * 3) / (
        1000 * low_length + 2000 * high_length
    )
    assert result.sar[cell] == pytest.approx(expected, rel=2e-9)


def test_background_cells_are_not_given_spatial_sar():
    density = np.full((12, 12, 12), np.nan)
    density[2:10, 2:10, 2:10] = 1000.0
    local_sar = np.zeros_like(density)
    local_sar[np.isfinite(density)] = 4.0

    result = spatial_average_sar(density, local_sar, (0.002, 0.002, 0.002), 0.001)

    assert np.all(result.status[~np.isfinite(density)] == INVALID)
    assert np.all(np.isnan(result.sar[~np.isfinite(density)]))
    assert result.peak_sar == pytest.approx(4.0)


def test_insufficient_tissue_mass_produces_no_average():
    density = np.full((4, 4, 4), 1000.0)
    local_sar = np.ones_like(density)

    result = spatial_average_sar(density, local_sar, (0.001, 0.001, 0.001), 0.001)

    assert np.all(np.isnan(result.sar))
    assert np.isnan(result.peak_sar)
    assert not np.any(result.status == VALID)


def test_invalid_density_is_rejected():
    density = np.full((3, 3, 3), 1000.0)
    density[1, 1, 1] = 0
    with pytest.raises(ValueError, match="density must be positive"):
        spatial_average_sar(density, np.ones_like(density), (0.001,) * 3, 0.001)


def test_compiled_spatial_plan_is_deterministic_across_thread_counts():
    density = np.full((18, 17, 16), np.nan)
    density[2:16, 2:15, 2:14] = 950.0
    density[9:16, 2:15, 2:14] = 1300.0
    local_sar = np.zeros_like(density)
    local_sar[np.isfinite(density)] = np.linspace(0.5, 3.5, np.count_nonzero(np.isfinite(density)))

    serial = spatial_average_sar(density, local_sar, (0.0015, 0.0015, 0.0015), 0.001, nthreads=1)
    parallel = spatial_average_sar(density, local_sar, (0.0015, 0.0015, 0.0015), 0.001, nthreads=4)

    np.testing.assert_array_equal(parallel.status, serial.status)
    np.testing.assert_array_equal(parallel.orientation, serial.orientation)
    np.testing.assert_allclose(parallel.sar, serial.sar, equal_nan=True)
    np.testing.assert_allclose(parallel.averaging_mass, serial.averaging_mass, equal_nan=True)


def test_compiled_spatial_plan_matches_python_reference():
    density = np.full((12, 11, 10), np.nan)
    density[1:11, 1:10, 1:9] = 980.0
    density[6:11, 2:9, 2:8] = 1250.0
    local_sar = np.zeros_like(density)
    local_sar[np.isfinite(density)] = np.linspace(0.25, 4.0, np.count_nonzero(np.isfinite(density)))
    arguments = (
        density,
        local_sar,
        (0.0015, 0.0015, 0.0015),
        0.001,
    )

    reference = _spatial_average_sar_python(*arguments)
    compiled = spatial_average_sar(*arguments, nthreads=4)

    np.testing.assert_array_equal(compiled.status, reference.status)
    np.testing.assert_array_equal(compiled.orientation, reference.orientation)
    np.testing.assert_allclose(compiled.sar, reference.sar, equal_nan=True)
    np.testing.assert_allclose(compiled.averaging_mass, reference.averaging_mass, equal_nan=True)
    np.testing.assert_allclose(
        compiled.averaging_volume, reference.averaging_volume, equal_nan=True
    )
    assert compiled.peak_sar == pytest.approx(reference.peak_sar)
    assert compiled.peak_cell == reference.peak_cell
