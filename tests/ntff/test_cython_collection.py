"""Tests for the Cython/OpenMP CPU KSIR surface collector."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax.ntff.frequency_domain as frequency_domain
from gprMax.cython.ntff import accumulate_surface_dft
from gprMax.ntff.frequency_domain import KSIRFrequencyDomainMonitor
from gprMax.ntff.surfaces import build_component_surface


@pytest.mark.parametrize(
    "real_dtype,complex_dtype,rtol",
    [
        (np.dtype("f4"), np.dtype("c8"), 2e-6),
        (np.dtype("f8"), np.dtype("c16"), 2e-14),
    ],
)
@pytest.mark.parametrize("nthreads", [1, 2])
def test_cython_surface_dft_matches_numpy(
    real_dtype, complex_dtype, rtol, nthreads
):
    rng = np.random.default_rng(8472)
    nfield = 4096
    npatches = 733
    nfrequencies = 7
    field = rng.standard_normal(nfield).astype(real_dtype)
    inside_indices = rng.integers(
        0, nfield, size=npatches, dtype=np.int64
    )
    outside_indices = rng.integers(
        0, nfield, size=npatches, dtype=np.int64
    )
    multiplier = (
        rng.standard_normal(nfrequencies)
        + 1j * rng.standard_normal(nfrequencies)
    ).astype(complex_dtype)
    initial_inside = (
        rng.standard_normal((nfrequencies, npatches))
        + 1j * rng.standard_normal((nfrequencies, npatches))
    ).astype(complex_dtype)
    initial_outside = (
        rng.standard_normal((nfrequencies, npatches))
        + 1j * rng.standard_normal((nfrequencies, npatches))
    ).astype(complex_dtype)
    inside_dft = initial_inside.copy()
    outside_dft = initial_outside.copy()

    accumulate_surface_dft(
        nthreads,
        inside_indices,
        outside_indices,
        field,
        multiplier,
        inside_dft,
        outside_dft,
    )

    expected_inside = initial_inside + (
        multiplier[:, np.newaxis] * field[inside_indices][np.newaxis, :]
    )
    expected_outside = initial_outside + (
        multiplier[:, np.newaxis] * field[outside_indices][np.newaxis, :]
    )
    assert_allclose(inside_dft, expected_inside, rtol=rtol)
    assert_allclose(outside_dft, expected_outside, rtol=rtol)


def _monitor(name, surface, nthreads):
    return KSIRFrequencyDomainMonitor(
        name,
        {"Ex": surface},
        [1.25e9, 2.5e9],
        [90.0],
        [0.0],
        1e-11,
        5,
        real_dtype=np.dtype("f8"),
        complex_dtype=np.dtype("c16"),
        nthreads=nthreads,
    )


def test_monitor_cython_path_matches_source_tree_numpy_fallback(monkeypatch):
    shape = (9, 10, 8)
    surface = build_component_surface(
        "Ex", (2, 2, 2), (5, 6, 5), (0.03, 0.04, 0.05), shape
    )
    cython_monitor = _monitor("cython", surface, 2)
    fields = [
        np.asarray(
            (iteration + 1) * (np.indices(shape)[0] - 0.2 * np.indices(shape)[2]),
            dtype=np.float64,
        )
        for iteration in range(5)
    ]
    zeros = np.zeros(shape)

    for iteration, field in enumerate(fields):
        cython_monitor.observe_electric(iteration, field, zeros, zeros)
    monkeypatch.setattr(frequency_domain, "_accumulate_surface_dft", None)
    fallback_monitor = _monitor("fallback", surface, 1)
    for iteration, field in enumerate(fields):
        fallback_monitor.observe_electric(iteration, field, zeros, zeros)

    cython_accumulator = cython_monitor._accumulators["Ex"]
    fallback_accumulator = fallback_monitor._accumulators["Ex"]
    assert cython_monitor.nthreads == 2
    assert cython_monitor.collection_backend == "cython_openmp"
    assert fallback_monitor.collection_backend == "numpy_fallback"
    assert_allclose(
        cython_accumulator.inside_dft, fallback_accumulator.inside_dft
    )
    assert_allclose(
        cython_accumulator.outside_dft, fallback_accumulator.outside_dft
    )


@pytest.mark.parametrize("nthreads", [0, -1, 1.5, None])
def test_monitor_rejects_invalid_openmp_thread_count(nthreads):
    surface = build_component_surface(
        "Ex", (2, 2, 2), (4, 4, 4), (0.1, 0.1, 0.1), (8, 8, 8)
    )
    with pytest.raises(ValueError, match="nthreads"):
        _monitor("invalid", surface, nthreads)
