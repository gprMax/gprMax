"""Unit tests for ``gprMax/cython/fractals_generate.pyx``.

``generate_fractal2D`` / ``generate_fractal3D`` are the whole fractal:
given the FFT of a white-noise array, they divide each spectral
coefficient by ``distance_from_centre ** D``, suppressing the fine detail
and leaving a correlated (1/f-noise) landscape once the caller transforms
back.

Both are pure arithmetic — no grid, no config, no randomness — so every
test here hands the kernel a hand-built array and compares against the
same formula evaluated in numpy.

The formula, for each cell:

    v2   = weighting * ((index + offset + global_size // 2) % global_size)
    rr   = ||v2 - v1||
    B    = rr ** D                (B = 0.9 if B == 0)
    out  = A / B
"""

import numpy as np
import pytest

from gprMax.cython.fractals_generate import generate_fractal2D, generate_fractal3D

NTHREADS = 1


def reference_2d(A, ox, oy, gx, gy, D, weighting, v1):
    """The documented formula, evaluated in numpy."""
    nx, ny = A.shape
    sx, sy = gx // 2, gy // 2
    i = np.arange(nx)[:, None]
    j = np.arange(ny)[None, :]
    v2x = weighting[0] * ((i + ox + sx) % gx)
    v2y = weighting[1] * ((j + oy + sy) % gy)
    rr = np.sqrt((v2x - v1[0]) ** 2 + (v2y - v1[1]) ** 2)
    B = rr**D
    B = np.where(B == 0, 0.9, B)
    return A / B


def reference_3d(A, ox, oy, oz, gx, gy, gz, D, weighting, v1):
    nx, ny, nz = A.shape
    sx, sy, sz = gx // 2, gy // 2, gz // 2
    i = np.arange(nx)[:, None, None]
    j = np.arange(ny)[None, :, None]
    k = np.arange(nz)[None, None, :]
    v2x = weighting[0] * ((i + ox + sx) % gx)
    v2y = weighting[1] * ((j + oy + sy) % gy)
    v2z = weighting[2] * ((k + oz + sz) % gz)
    rr = np.sqrt((v2x - v1[0]) ** 2 + (v2y - v1[1]) ** 2 + (v2z - v1[2]) ** 2)
    B = rr**D
    B = np.where(B == 0, 0.9, B)
    return A / B


def random_complex(shape, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=np.complex128
    )


def run_2d(A, D, weighting, v1, ox=0, oy=0, gx=None, gy=None):
    nx, ny = A.shape
    gx = nx if gx is None else gx
    gy = ny if gy is None else gy
    out = np.zeros((nx, ny), dtype=np.complex128)
    generate_fractal2D(
        nx,
        ny,
        ox,
        oy,
        gx,
        gy,
        NTHREADS,
        D,
        np.asarray(weighting, dtype=np.float64),
        np.asarray(v1, dtype=np.float64),
        A,
        out,
    )
    return out


def run_3d(A, D, weighting, v1, ox=0, oy=0, oz=0, gx=None, gy=None, gz=None):
    nx, ny, nz = A.shape
    gx = nx if gx is None else gx
    gy = ny if gy is None else gy
    gz = nz if gz is None else gz
    out = np.zeros((nx, ny, nz), dtype=np.complex128)
    generate_fractal3D(
        nx,
        ny,
        nz,
        ox,
        oy,
        oz,
        gx,
        gy,
        gz,
        NTHREADS,
        D,
        np.asarray(weighting, dtype=np.float64),
        np.asarray(v1, dtype=np.float64),
        A,
        out,
    )
    return out


class TestGenerateFractal2D:
    def test_matches_the_reference_formula(self):
        A = random_complex((8, 8))
        weighting = [1.0, 1.0]
        v1 = [4.0, 4.0]
        out = run_2d(A, 1.5, weighting, v1)
        expected = reference_2d(A, 0, 0, 8, 8, 1.5, weighting, v1)
        assert np.allclose(out, expected)

    def test_zero_dimension_is_the_identity(self):
        # rr ** 0 == 1 for every cell, including the centre, so the array
        # passes through untouched.
        A = random_complex((8, 8))
        out = run_2d(A, 0.0, [1.0, 1.0], [4.0, 4.0])
        assert np.allclose(out, A)

    def test_centre_cell_uses_the_zero_norm_fallback(self):
        # With gx = 8 the shift is sx = 4, so cell i = 4 maps to position
        # 0. Placing v1 at the origin makes rr == 0 there, hence B == 0,
        # which the kernel replaces with 0.9.
        A = np.ones((8, 8), dtype=np.complex128)
        out = run_2d(A, 1.0, [1.0, 1.0], [0.0, 0.0])
        assert out[4, 4] == pytest.approx(1.0 / 0.9)

    def test_only_the_centre_cell_uses_the_fallback(self):
        A = np.ones((8, 8), dtype=np.complex128)
        out = run_2d(A, 1.0, [1.0, 1.0], [0.0, 0.0])
        fallback = np.isclose(np.abs(out), 1.0 / 0.9)
        assert set(map(tuple, np.argwhere(fallback))) == {(4, 4)}

    def test_higher_dimension_suppresses_high_frequencies_harder(self):
        # The shift puts DC at index 0 and the highest frequency at index
        # n // 2 (which maps to position 0, furthest from v1 at the
        # centre). That far cell is divided by a larger number as D rises.
        A = np.ones((16, 16), dtype=np.complex128)
        v1 = [8.0, 8.0]
        low = run_2d(A, 1.0, [1.0, 1.0], v1)
        high = run_2d(A, 2.5, [1.0, 1.0], v1)
        far = (8, 8)
        assert abs(high[far]) < abs(low[far])

    def test_weighting_stretches_the_distance_metric_per_axis(self):
        A = np.ones((8, 8), dtype=np.complex128)
        isotropic = run_2d(A, 2.0, [1.0, 1.0], [4.0, 4.0])
        # Weighting x by 2 doubles every x-distance (and v1 with it), so a
        # cell offset from the centre along x is suppressed harder while
        # one offset along y is untouched.
        stretched = run_2d(A, 2.0, [2.0, 1.0], [8.0, 4.0])
        assert abs(stretched[4, 0]) < abs(isotropic[4, 0])
        assert abs(stretched[0, 4]) == pytest.approx(abs(isotropic[0, 4]))

    def test_offsets_select_a_sub_block_of_the_full_domain(self):
        # The ox/oy/gx/gy parameters exist so an MPI rank holding a
        # sub-block computes the same values the serial code would.
        full_A = random_complex((8, 8), seed=3)
        weighting = [1.0, 1.0]
        v1 = [4.0, 4.0]
        full = run_2d(full_A, 1.5, weighting, v1)

        block = np.ascontiguousarray(full_A[4:8, 2:6])
        out = run_2d(block, 1.5, weighting, v1, ox=4, oy=2, gx=8, gy=8)
        assert np.allclose(out, full[4:8, 2:6])

    def test_non_square_arrays(self):
        A = random_complex((4, 10))
        weighting = [1.0, 1.0]
        v1 = [2.0, 5.0]
        out = run_2d(A, 1.5, weighting, v1)
        assert np.allclose(out, reference_2d(A, 0, 0, 4, 10, 1.5, weighting, v1))

    def test_output_dtype_is_preserved(self):
        A = random_complex((4, 4))
        out = run_2d(A, 1.5, [1.0, 1.0], [2.0, 2.0])
        assert out.dtype == np.complex128
        assert np.iscomplexobj(out)

    def test_input_array_is_not_mutated(self):
        A = random_complex((4, 4))
        before = A.copy()
        run_2d(A, 1.5, [1.0, 1.0], [2.0, 2.0])
        assert np.array_equal(A, before)


class TestGenerateFractal3D:
    def test_matches_the_reference_formula(self):
        A = random_complex((6, 6, 6))
        weighting = [1.0, 1.0, 1.0]
        v1 = [3.0, 3.0, 3.0]
        out = run_3d(A, 1.5, weighting, v1)
        expected = reference_3d(A, 0, 0, 0, 6, 6, 6, 1.5, weighting, v1)
        assert np.allclose(out, expected)

    def test_zero_dimension_is_the_identity(self):
        A = random_complex((4, 4, 4))
        out = run_3d(A, 0.0, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        assert np.allclose(out, A)

    def test_centre_cell_uses_the_zero_norm_fallback(self):
        A = np.ones((8, 8, 8), dtype=np.complex128)
        out = run_3d(A, 1.0, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        assert out[4, 4, 4] == pytest.approx(1.0 / 0.9)

    def test_only_the_centre_cell_uses_the_fallback(self):
        A = np.ones((8, 8, 8), dtype=np.complex128)
        out = run_3d(A, 1.0, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        fallback = np.isclose(np.abs(out), 1.0 / 0.9)
        assert set(map(tuple, np.argwhere(fallback))) == {(4, 4, 4)}

    def test_higher_dimension_suppresses_high_frequencies_harder(self):
        A = np.ones((8, 8, 8), dtype=np.complex128)
        v1 = [4.0, 4.0, 4.0]
        low = run_3d(A, 1.0, [1.0, 1.0, 1.0], v1)
        high = run_3d(A, 2.5, [1.0, 1.0, 1.0], v1)
        far = (4, 4, 4)
        assert abs(high[far]) < abs(low[far])

    def test_offsets_select_a_sub_block_of_the_full_domain(self):
        full_A = random_complex((8, 8, 8), seed=5)
        weighting = [1.0, 1.0, 1.0]
        v1 = [4.0, 4.0, 4.0]
        full = run_3d(full_A, 1.5, weighting, v1)

        block = np.ascontiguousarray(full_A[2:6, 0:4, 4:8])
        out = run_3d(block, 1.5, weighting, v1, ox=2, oy=0, oz=4, gx=8, gy=8, gz=8)
        assert np.allclose(out, full[2:6, 0:4, 4:8])

    def test_non_cubic_arrays(self):
        A = random_complex((3, 5, 7))
        weighting = [1.0, 1.0, 1.0]
        v1 = [1.5, 2.5, 3.5]
        out = run_3d(A, 1.2, weighting, v1)
        assert np.allclose(out, reference_3d(A, 0, 0, 0, 3, 5, 7, 1.2, weighting, v1))

    def test_weighting_scales_each_axis_independently(self):
        A = np.ones((8, 8, 8), dtype=np.complex128)
        weighting = [1.0, 2.0, 3.0]
        v1 = [4.0, 8.0, 12.0]
        out = run_3d(A, 1.5, weighting, v1)
        assert np.allclose(out, reference_3d(A, 0, 0, 0, 8, 8, 8, 1.5, weighting, v1))

    def test_input_array_is_not_mutated(self):
        A = random_complex((4, 4, 4))
        before = A.copy()
        run_3d(A, 1.5, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        assert np.array_equal(A, before)
