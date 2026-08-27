# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Tests for Cython/OpenMP advanced-time KSIR kernels."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax.ntff.time_domain as time_domain
from gprMax.cython.ntff import (
    deposit_time_domain_surface,
    gather_time_domain_surface,
)
from gprMax.ntff.closures import SymmetryCompletion, resolve_closure
from gprMax.ntff.surfaces import build_component_surface
from gprMax.ntff.time_domain import KSIRTimeDomainMonitor


@pytest.mark.parametrize("dtype,rtol", [(np.float32, 1e-5), (np.float64, 5e-14)])
@pytest.mark.parametrize("nthreads", [1, 2])
def test_cython_gather_and_deposit_match_numpy(dtype, rtol, nthreads):
    rng = np.random.default_rng(7351)
    nfield = 512
    npatches = 83
    neffective = 2 * npatches
    npoints = 5
    field = rng.standard_normal(nfield).astype(dtype)
    inside = rng.integers(0, nfield, npatches, dtype=np.int64)
    outside = rng.integers(0, nfield, npatches, dtype=np.int64)
    spacing = rng.uniform(0.01, 0.04, npatches).astype(dtype)
    surface = np.empty(npatches, dtype=dtype)
    derivative = np.empty(npatches, dtype=dtype)

    gather_time_domain_surface(
        nthreads, inside, outside, spacing, field, surface, derivative
    )

    expected_surface = 0.5 * (field[inside] + field[outside])
    expected_derivative = (field[outside] - field[inside]) / spacing
    assert_allclose(surface, expected_surface, rtol=rtol)
    assert_allclose(derivative, expected_derivative, rtol=rtol)

    time_derivative = rng.standard_normal(npatches).astype(dtype)
    weights = [
        rng.standard_normal((npoints, neffective)).astype(dtype)
        for _ in range(3)
    ]
    integer_delay = rng.integers(
        20, 35, (npoints, neffective), dtype=np.int64
    )
    fractional_delay = rng.random((npoints, neffective)).astype(dtype)
    origins = np.asarray((15, 17, 19, 21, 23), dtype=np.int64)
    sample_index = 4
    output = np.zeros((npoints, 30), dtype=dtype)
    source_patch_index = np.tile(
        np.arange(npatches, dtype=np.int64), 2
    )

    deposit_time_domain_surface(
        nthreads,
        sample_index,
        surface,
        derivative,
        time_derivative,
        source_patch_index,
        weights[0],
        weights[1],
        weights[2],
        integer_delay,
        fractional_delay,
        origins,
        output,
    )

    contribution = (
        weights[0] * derivative[source_patch_index][np.newaxis, :]
        + weights[1] * surface[source_patch_index][np.newaxis, :]
        + weights[2] * time_derivative[source_patch_index][np.newaxis, :]
    )
    expected = np.zeros_like(output)
    destination = sample_index + integer_delay - origins[:, np.newaxis]
    for point in range(npoints):
        np.add.at(
            expected[point],
            destination[point],
            (1 - fractional_delay[point]) * contribution[point],
        )
        np.add.at(
            expected[point],
            destination[point] + 1,
            fractional_delay[point] * contribution[point],
        )
    assert_allclose(output, expected, rtol=rtol, atol=rtol)


def _monitor(name, surface, nthreads, closure=None):
    return KSIRTimeDomainMonitor(
        name,
        {"Ex": surface},
        ((0.8, 0.4, 0.4), (0.4, 0.9, 0.5)),
        0.01,
        6,
        real_dtype=np.dtype("f8"),
        wave_speed=1.0,
        nthreads=nthreads,
        closure=closure,
    )


def test_monitor_cython_path_matches_numpy_fallback(monkeypatch):
    shape = (10, 10, 10)
    surface = build_component_surface(
        "Ex", (2, 2, 2), (6, 6, 6), (0.08, 0.08, 0.08), shape
    )
    indices = np.indices(shape)
    fields = [
        np.asarray((iteration + 1) * (indices[0] - 0.3 * indices[2]))
        for iteration in range(6)
    ]
    zeros = np.zeros(shape)
    cython_monitor = _monitor("cython", surface, 2)
    for iteration, field in enumerate(fields):
        cython_monitor.observe_electric(iteration, field, zeros, zeros)
    cython_monitor.finalise()

    monkeypatch.setattr(time_domain, "_gather_time_domain_surface", None)
    monkeypatch.setattr(time_domain, "_deposit_time_domain_surface", None)
    fallback_monitor = _monitor("fallback", surface, 1)
    for iteration, field in enumerate(fields):
        fallback_monitor.observe_electric(iteration, field, zeros, zeros)
    fallback_monitor.finalise()

    assert cython_monitor.collection_backend == "cython_openmp"
    assert fallback_monitor.collection_backend == "numpy_fallback"
    assert_allclose(
        cython_monitor.result.fields["Ex"],
        fallback_monitor.result.fields["Ex"],
        rtol=2e-14,
        atol=2e-14,
    )

def test_symmetry_image_mapping_matches_numpy_fallback(monkeypatch):
    shape = (10, 10, 10)
    closure = resolve_closure(
        SymmetryCompletion(),
        {"x0": "pmc"},
        (0, 2, 2),
        (6, 6, 6),
        (9, 9, 9),
        (0.08, 0.08, 0.08),
    )
    surface = closure.apply_quadrature(
        build_component_surface(
            "Ex",
            (0, 2, 2),
            (6, 6, 6),
            (0.08, 0.08, 0.08),
            shape,
            excluded_faces=closure.omitted_faces,
        )
    )
    indices = np.indices(shape)
    fields = [
        np.asarray((iteration + 1) * (indices[0] - 0.3 * indices[2]))
        for iteration in range(6)
    ]
    zeros = np.zeros(shape)
    cython_monitor = _monitor("cython_symmetry", surface, 2, closure)
    for iteration, field in enumerate(fields):
        cython_monitor.observe_electric(iteration, field, zeros, zeros)
    cython_monitor.finalise()

    monkeypatch.setattr(time_domain, "_gather_time_domain_surface", None)
    monkeypatch.setattr(time_domain, "_deposit_time_domain_surface", None)
    fallback_monitor = _monitor("fallback_symmetry", surface, 1, closure)
    for iteration, field in enumerate(fields):
        fallback_monitor.observe_electric(iteration, field, zeros, zeros)
    fallback_monitor.finalise()

    assert_allclose(
        cython_monitor.result.fields["Ex"],
        fallback_monitor.result.fields["Ex"],
        rtol=2e-14,
        atol=2e-14,
    )
