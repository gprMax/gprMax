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

"""Regression test for gprMax/gprMax#435 ("PML bug when using magnetic
materials", intermittent reflections, reproducible per-geometry but not
across all geometries - roughly half the time).

Root cause found while investigating that issue: `pml_average_er_mr()` and
`pml_sum_er_mr()` (gprMax/cython/pml_build.pyx) declared their accumulators
as `cdef double sumer, summr` with NO initial value, then accumulated into
them with `+=` inside a `prange(nogil=True)` loop. Cython/OpenMP treats a
scalar `+=` inside a `prange` loop as a reduction - and a reduction combines
the per-thread partial sums with whatever value the variable held BEFORE
entering the parallel region, not with an implicit zero. Since the variable
was never set to 0.0, the result depended on whatever garbage happened to be
on the stack at that point - explaining the reported symptom exactly:
consistent for a given geometry (same code path -> same leftover stack
content) but not consistent across different geometries/models (different
code paths leave different garbage), and no correlation with the "magnetic
material" itself (any model could have been affected; a very large mr like
steel's 1000 just makes the PML profile much more sensitive to a polluted
average, making the resulting reflection far more noticeable).

This average/sum feeds directly into PML.calculate_sigmamax()/
calculate_update_coeffs() (gprMax/pml.py), which is computed ONCE at build
time and baked into that PML slab's update coefficients for the entire run -
so a corrupted average silently produces a wrong-but-consistent (for that
run) impedance mismatch and spurious reflections, with no error or warning
anywhere.

Fixed by explicitly zero-initializing `sumer`/`summr` in both functions.

This test verifies exact correctness of both functions against a manually
computed reference, and repeats the call many times with different
stack-polluting operations interleaved beforehand to make a reintroduced
uninitialized-accumulator regression more likely to be caught (not a
guaranteed catch without a memory sanitizer, but a real empirical check).
"""
import numpy as np
import pytest

from gprMax.cython.pml_build import pml_average_er_mr, pml_sum_er_mr


def _pollute_stack():
    """Recurse with large local arrays to disturb whatever the C stack
    currently holds, then return - maximising the chance that a
    (re-)introduced uninitialized-accumulator bug shows up as nonzero
    garbage in the very next call to the functions under test."""
    junk = np.full(4096, 12345.6789, dtype=np.float64)
    return float(junk.sum())


@pytest.mark.parametrize("nthreads", [1, 2, 4])
def test_pml_average_er_mr_matches_manual_reference(nthreads):
    n1, n2 = 5, 7
    rng = np.random.default_rng(42)
    solid = rng.integers(0, 3, size=(n1, n2)).astype(np.uint32)
    ers = np.array([1.0, 5.96926, 9.47588], dtype=np.float64)
    mrs = np.array([1.0, 1.0, 1000.0], dtype=np.float64)  # last one steel-like

    expected_er = ers[solid].sum() / (n1 * n2)
    expected_mr = mrs[solid].sum() / (n1 * n2)

    for _ in range(50):
        _pollute_stack()
        averageer, averagemr = pml_average_er_mr(n1, n2, nthreads, solid, ers, mrs)
        assert averageer == pytest.approx(expected_er, rel=1e-12)
        assert averagemr == pytest.approx(expected_mr, rel=1e-12)


@pytest.mark.parametrize("nthreads", [1, 2, 4])
def test_pml_sum_er_mr_matches_manual_reference(nthreads):
    n1, n2 = 6, 4
    rng = np.random.default_rng(7)
    solid = rng.integers(0, 2, size=(n1, n2)).astype(np.uint32)
    ers = np.array([1.0, 1400000.0], dtype=np.float64)
    mrs = np.array([1.0, 1000.0], dtype=np.float64)

    expected_sumer = ers[solid].sum()
    expected_summr = mrs[solid].sum()

    for _ in range(50):
        _pollute_stack()
        sumer, summr = pml_sum_er_mr(n1, n2, nthreads, solid, ers, mrs)
        assert sumer == pytest.approx(expected_sumer, rel=1e-12)
        assert summr == pytest.approx(expected_summr, rel=1e-12)
