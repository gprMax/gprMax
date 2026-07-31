"""Tests for the Cython/OpenMP far-zone KSIR evaluator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax.ntff.evaluator as evaluator


@pytest.mark.parametrize(
    "real_dtype,complex_dtype,rtol,atol",
    [
        (np.float32, np.complex64, 2e-5, 2e-5),
        (np.float64, np.complex128, 2e-13, 2e-13),
    ],
)
@pytest.mark.parametrize("nthreads", [1, 2])
def test_cython_far_zone_matches_numpy_fallback(
    monkeypatch,
    real_dtype,
    complex_dtype,
    rtol,
    atol,
    nthreads,
):
    compiled = evaluator._evaluate_far_zone_patches_cython
    if compiled is None:
        pytest.skip("Cython NTFF extension is not built")
    rng = np.random.default_rng(4815)
    npatches = 137
    nfrequencies = 3
    ndirections = 17
    positions = rng.uniform(-0.2, 0.2, (npatches, 3)).astype(real_dtype)
    normals = rng.normal(size=(npatches, 3)).astype(real_dtype)
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    areas = rng.uniform(1e-5, 4e-4, npatches).astype(real_dtype)
    frequencies = np.asarray((0.0, 3e8, 7e8), dtype=real_dtype)
    directions = rng.normal(size=(ndirections, 3)).astype(real_dtype)
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    field = (
        rng.normal(size=(nfrequencies, npatches)) + 1j * rng.normal(size=(nfrequencies, npatches))
    ).astype(complex_dtype)
    derivative = (
        rng.normal(size=(nfrequencies, npatches)) + 1j * rng.normal(size=(nfrequencies, npatches))
    ).astype(complex_dtype)

    actual = evaluator.evaluate_far_zone_patches(
        positions,
        normals,
        areas,
        frequencies,
        directions,
        field,
        derivative,
        origin=(0.03, -0.02, 0.01),
        nthreads=nthreads,
    )
    monkeypatch.setattr(evaluator, "_evaluate_far_zone_patches_cython", None)
    expected = evaluator.evaluate_far_zone_patches(
        positions,
        normals,
        areas,
        frequencies,
        directions,
        field,
        derivative,
        origin=(0.03, -0.02, 0.01),
        direction_block_size=5,
        patch_block_size=31,
    )

    assert actual.dtype == np.dtype(complex_dtype)
    assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_cython_far_zone_supports_many_frequencies_and_few_directions(monkeypatch):
    if evaluator._evaluate_far_zone_patches_cython is None:
        pytest.skip("Cython NTFF extension is not built")
    rng = np.random.default_rng(621)
    npatches = 53
    nfrequencies = 47
    positions = rng.uniform(-0.1, 0.1, (npatches, 3))
    normals = rng.normal(size=(npatches, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    areas = rng.uniform(1e-5, 2e-4, npatches)
    frequencies = np.linspace(1e7, 2e9, nfrequencies)
    directions = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    field = rng.normal(size=(nfrequencies, npatches)) + 1j * rng.normal(
        size=(nfrequencies, npatches)
    )
    derivative = rng.normal(size=(nfrequencies, npatches)) + 1j * rng.normal(
        size=(nfrequencies, npatches)
    )

    actual = evaluator.evaluate_far_zone_patches(
        positions,
        normals,
        areas,
        frequencies,
        directions,
        field,
        derivative,
        nthreads=4,
    )
    monkeypatch.setattr(evaluator, "_evaluate_far_zone_patches_cython", None)
    expected = evaluator.evaluate_far_zone_patches(
        positions,
        normals,
        areas,
        frequencies,
        directions,
        field,
        derivative,
        patch_block_size=17,
    )

    assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
