"""Unit tests for ``gprMax/fractals/fractal_surface.py`` (``FractalSurface``).

A fractal surface is the rough height-map applied to one face of a
fractal box. ``generate_fractal_surface()`` fills white noise, FFTs it,
divides each spectral coefficient by ``distance ** dimension`` (the
Cython kernel), inverse-FFTs, and linearly rescales the result so its
minimum and maximum land exactly on the requested ``fractalrange``.

Fractal *values* are random, but the contract around them is not: the
shape follows the plane the surface lives on, the output range is exact
arithmetic, and a given seed always reproduces the same landscape. Those
are what we pin. The ``dimension`` and ``weighting`` knobs get
directional assertions only — smoother/rougher, not a target number.

``MPIFractalSurface`` is out of scope: it needs a live MPI communicator
and the optional ``mpi4py_fft`` package.
"""

import numpy as np
import pytest

from gprMax.fractals.fractal_surface import FractalSurface

SEED = 42
DIMENSION = 1.5


def make_surface(xs=0, xf=0, ys=0, yf=8, zs=0, zf=8, dimension=DIMENSION, seed=SEED,
                 fractalrange=(4, 12), weighting=(1.0, 1.0)):
    """A surface on a plane, ready to generate."""
    surface = FractalSurface(xs, xf, ys, yf, zs, zf, dimension, seed)
    surface.fractalrange = fractalrange
    surface.weighting = np.array(weighting, dtype=np.float64)
    return surface


def roughness(surface):
    """Mean absolute difference between neighbouring cells — how jagged
    the height-map is."""
    return np.mean(np.abs(np.diff(surface.fractalsurface, axis=0)))


class TestConstruction:
    def test_start_and_stop_are_stored_as_int32(self):
        surface = FractalSurface(1, 2, 3, 4, 5, 6, DIMENSION, SEED)
        assert np.array_equal(surface.start, [1, 3, 5])
        assert np.array_equal(surface.stop, [2, 4, 6])
        assert surface.start.dtype == np.int32

    def test_dimension_and_seed_are_stored(self):
        surface = FractalSurface(0, 0, 0, 8, 0, 8, 2.0, 7)
        assert surface.dimension == 2.0
        assert surface.seed == 7

    def test_defaults(self):
        surface = FractalSurface(0, 0, 0, 8, 0, 8, DIMENSION, SEED)
        assert surface.ID is None
        assert surface.surfaceID is None
        assert surface.fractalrange == (0, 0)
        assert surface.filldepth == 0
        assert surface.grass == []
        assert np.array_equal(surface.weighting, [1, 1])
        assert surface.dtype == np.complex128

    def test_surface_ids_are_the_six_faces(self):
        assert FractalSurface.surfaceIDs == [
            "xminus",
            "xplus",
            "yminus",
            "yplus",
            "zminus",
            "zplus",
        ]


class TestCoordinateProperties:
    def test_getters_read_through_to_start_and_stop(self):
        surface = FractalSurface(1, 2, 3, 4, 5, 6, DIMENSION, SEED)
        assert (surface.xs, surface.ys, surface.zs) == (1, 3, 5)
        assert (surface.xf, surface.yf, surface.zf) == (2, 4, 6)

    @pytest.mark.parametrize(
        "attribute, index, array",
        [
            ("xs", 0, "start"),
            ("ys", 1, "start"),
            ("zs", 2, "start"),
            ("xf", 0, "stop"),
            ("yf", 1, "stop"),
            ("zf", 2, "stop"),
        ],
    )
    def test_setters_write_through(self, attribute, index, array):
        surface = FractalSurface(1, 2, 3, 4, 5, 6, DIMENSION, SEED)
        setattr(surface, attribute, 9)
        assert getattr(surface, array)[index] == 9

    def test_size_and_extents(self):
        surface = FractalSurface(1, 5, 2, 8, 3, 6, DIMENSION, SEED)
        assert np.array_equal(surface.size, [4, 6, 3])
        assert (surface.nx, surface.ny, surface.nz) == (4, 6, 3)


class TestGetSurfaceDims:
    def test_x_plane_surface_spans_y_and_z(self):
        assert make_surface(xs=3, xf=3, ys=0, yf=6, zs=0, zf=4).get_surface_dims() == (6, 4)

    def test_y_plane_surface_spans_x_and_z(self):
        assert make_surface(xs=0, xf=6, ys=3, yf=3, zs=0, zf=4).get_surface_dims() == (6, 4)

    def test_z_plane_surface_spans_x_and_y(self):
        assert make_surface(xs=0, xf=6, ys=0, yf=4, zs=3, zf=3).get_surface_dims() == (6, 4)


class TestGenerateFractalSurface:
    def test_returns_true(self):
        assert make_surface().generate_fractal_surface() is True

    @pytest.mark.parametrize(
        "bounds, expected_shape",
        [
            ((3, 3, 0, 6, 0, 4), (6, 4)),  # x-plane
            ((0, 6, 3, 3, 0, 4), (6, 4)),  # y-plane
            ((0, 6, 0, 4, 3, 3), (6, 4)),  # z-plane
        ],
    )
    def test_shape_follows_the_plane(self, bounds, expected_shape):
        xs, xf, ys, yf, zs, zf = bounds
        surface = make_surface(xs, xf, ys, yf, zs, zf)
        surface.generate_fractal_surface()
        assert surface.fractalsurface.shape == expected_shape

    def test_output_dtype_comes_from_config(self, fractal_config):
        # Computed in complex128, delivered as the model's float type.
        surface = make_surface()
        surface.generate_fractal_surface()
        assert surface.fractalsurface.dtype == fractal_config.sim_config.dtypes["float_or_double"]

    def test_output_spans_the_requested_range_exactly(self):
        surface = make_surface(fractalrange=(4, 12))
        surface.generate_fractal_surface()
        assert np.amin(surface.fractalsurface) == pytest.approx(4.0)
        assert np.amax(surface.fractalsurface) == pytest.approx(12.0)

    @pytest.mark.parametrize("fractalrange", [(0, 1), (5, 6), (10, 40)])
    def test_range_is_exact_for_any_limits(self, fractalrange):
        surface = make_surface(fractalrange=fractalrange)
        surface.generate_fractal_surface()
        assert np.amin(surface.fractalsurface) == pytest.approx(float(fractalrange[0]))
        assert np.amax(surface.fractalsurface) == pytest.approx(float(fractalrange[1]))

    def test_every_value_lies_within_the_range(self):
        surface = make_surface(fractalrange=(4, 12))
        surface.generate_fractal_surface()
        assert np.all(surface.fractalsurface >= 4.0)
        assert np.all(surface.fractalsurface <= 12.0)

    def test_the_surface_is_not_flat(self):
        surface = make_surface()
        surface.generate_fractal_surface()
        assert np.unique(surface.fractalsurface).size > 1

    def test_same_seed_reproduces_the_same_surface(self):
        # The property gprMax users depend on: a published model must
        # regenerate byte-identically.
        first, second = make_surface(seed=SEED), make_surface(seed=SEED)
        first.generate_fractal_surface()
        second.generate_fractal_surface()
        assert np.array_equal(first.fractalsurface, second.fractalsurface)

    def test_different_seed_gives_a_different_surface(self):
        first, second = make_surface(seed=SEED), make_surface(seed=SEED + 1)
        first.generate_fractal_surface()
        second.generate_fractal_surface()
        assert not np.array_equal(first.fractalsurface, second.fractalsurface)

    def test_generating_twice_gives_the_same_surface(self):
        # The RNG is created fresh inside the method, so a second call on
        # the same object does not advance any shared stream.
        surface = make_surface()
        surface.generate_fractal_surface()
        first = surface.fractalsurface.copy()
        surface.generate_fractal_surface()
        assert np.array_equal(surface.fractalsurface, first)

    def test_higher_dimension_gives_a_smoother_surface(self):
        # A bigger dimension divides the high-frequency coefficients by
        # more, so the cell-to-cell variation falls.
        rough = make_surface(ys=0, yf=32, zs=0, zf=32, dimension=1.0)
        smooth = make_surface(ys=0, yf=32, zs=0, zf=32, dimension=3.0)
        rough.generate_fractal_surface()
        smooth.generate_fractal_surface()
        assert roughness(smooth) < roughness(rough)

    def test_weighting_is_not_mutated_by_generation(self):
        surface = make_surface(weighting=(1.0, 2.0))
        surface.generate_fractal_surface()
        assert np.array_equal(surface.weighting, [1.0, 2.0])

    def test_weighting_changes_the_surface(self):
        isotropic = make_surface(weighting=(1.0, 1.0))
        anisotropic = make_surface(weighting=(1.0, 4.0))
        isotropic.generate_fractal_surface()
        anisotropic.generate_fractal_surface()
        assert not np.array_equal(isotropic.fractalsurface, anisotropic.fractalsurface)

    def test_a_single_cell_wide_surface_still_generates(self):
        surface = make_surface(ys=0, yf=1, zs=0, zf=8)
        surface.generate_fractal_surface()
        assert surface.fractalsurface.shape == (1, 8)
        assert np.amin(surface.fractalsurface) == pytest.approx(4.0)
        assert np.amax(surface.fractalsurface) == pytest.approx(12.0)
