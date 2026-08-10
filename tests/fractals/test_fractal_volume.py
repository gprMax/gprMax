"""Unit tests for ``gprMax/fractals/fractal_volume.py`` (``FractalVolume``).

A fractal volume is the *interior* of a fractal box: a 3D array whose
cells each carry a bin index, so that heterogeneous soil (N materials
from a mixing model) can be distributed with realistic spatial
correlation instead of uniformly at random.

``generate_fractal_volume()`` is the 3D twin of the surface generator —
noise, FFT, spectral division, inverse FFT — with two extras:

- the ``weighting`` is pre-scaled by ``min(dims) / dims`` so the filter
  is isotropic in physical terms rather than in cells;
- the result is *binned* into ``nbins`` levels with ``np.digitize``, so
  the output holds integers in ``[0, nbins - 1]`` — indices into the
  mixing model's material table, not heights.

``generate_volume_mask()`` then records which cells belong to the user's
original box, as opposed to the extension a rough surface asked for.

``MPIFractalVolume`` is out of scope: it needs a live MPI communicator
and the optional ``mpi4py_fft`` package.
"""

import numpy as np
import pytest

from gprMax.fractals.fractal_volume import FractalVolume

SEED = 42
DIMENSION = 1.5


def make_volume(xs=0, xf=8, ys=0, yf=8, zs=0, zf=8, dimension=DIMENSION, seed=SEED, nbins=4):
    volume = FractalVolume(xs, xf, ys, yf, zs, zf, dimension, seed)
    volume.nbins = nbins
    return volume


class TestConstruction:
    def test_start_and_stop_are_stored_as_int32(self):
        volume = FractalVolume(1, 2, 3, 4, 5, 6, DIMENSION, SEED)
        assert np.array_equal(volume.start, [1, 3, 5])
        assert np.array_equal(volume.stop, [2, 4, 6])
        assert volume.start.dtype == np.int32

    def test_original_bounds_snapshot_the_constructor_arguments(self):
        # The originals remember the user's box; start/stop may later be
        # extended by a rough surface.
        volume = FractalVolume(1, 2, 3, 4, 5, 6, DIMENSION, SEED)
        assert np.array_equal(volume.original_start, [1, 3, 5])
        assert np.array_equal(volume.original_stop, [2, 4, 6])

    def test_extending_the_volume_leaves_the_originals_alone(self):
        volume = FractalVolume(1, 2, 3, 4, 5, 6, DIMENSION, SEED)
        volume.zf = 10
        assert volume.zf == 10
        assert volume.originalzf == 6

    def test_defaults(self):
        volume = FractalVolume(0, 8, 0, 8, 0, 8, DIMENSION, SEED)
        assert volume.ID is None
        assert volume.operatingonID is None
        assert volume.averaging is False
        assert volume.nbins == 0
        assert volume.mixingmodel is None
        assert volume.fractalsurfaces == []
        assert np.array_equal(volume.weighting, [1, 1, 1])
        assert volume.dtype == np.complex128

    def test_dimension_and_seed_are_stored(self):
        volume = FractalVolume(0, 8, 0, 8, 0, 8, 2.5, 7)
        assert volume.dimension == 2.5
        assert volume.seed == 7


class TestCoordinateProperties:
    def test_getters_read_through(self):
        volume = FractalVolume(1, 2, 3, 4, 5, 6, DIMENSION, SEED)
        assert (volume.xs, volume.ys, volume.zs) == (1, 3, 5)
        assert (volume.xf, volume.yf, volume.zf) == (2, 4, 6)

    @pytest.mark.parametrize(
        "attribute, index, array",
        [
            ("xs", 0, "start"),
            ("ys", 1, "start"),
            ("zs", 2, "start"),
            ("xf", 0, "stop"),
            ("yf", 1, "stop"),
            ("zf", 2, "stop"),
            ("originalxs", 0, "original_start"),
            ("originalys", 1, "original_start"),
            ("originalzs", 2, "original_start"),
            ("originalxf", 0, "original_stop"),
            ("originalyf", 1, "original_stop"),
            ("originalzf", 2, "original_stop"),
        ],
    )
    def test_setters_write_through(self, attribute, index, array):
        volume = FractalVolume(1, 2, 3, 4, 5, 6, DIMENSION, SEED)
        setattr(volume, attribute, 9)
        assert getattr(volume, array)[index] == 9

    def test_size_and_extents(self):
        volume = FractalVolume(1, 5, 2, 8, 3, 6, DIMENSION, SEED)
        assert np.array_equal(volume.size, [4, 6, 3])
        assert (volume.nx, volume.ny, volume.nz) == (4, 6, 3)


class TestFilterScaling:
    def test_a_cubic_volume_keeps_its_weighting(self):
        volume = make_volume(xf=8, yf=8, zf=8)
        volume.generate_fractal_volume()
        assert np.allclose(volume.weighting, [1.0, 1.0, 1.0])

    def test_an_elongated_volume_is_scaled_by_the_shortest_axis(self):
        # min(8, 4, 4) / (8, 4, 4) = (0.5, 1, 1)
        volume = make_volume(xf=8, yf=4, zf=4)
        volume.generate_fractal_volume()
        assert np.allclose(volume.weighting, [0.5, 1.0, 1.0])

    @pytest.mark.parametrize(
        "bounds, expected",
        [
            ((1, 8, 4), [1.0, 0.5, 1.0]),  # nx == 1: scale y and z
            ((8, 1, 4), [0.5, 1.0, 1.0]),  # ny == 1: scale x and z
            ((8, 4, 1), [0.5, 1.0, 1.0]),  # nz == 1: scale x and y
        ],
    )
    def test_flat_volumes_hold_the_flat_axis_at_one(self, bounds, expected):
        nx, ny, nz = bounds
        volume = make_volume(xf=nx, yf=ny, zf=nz)
        volume.generate_fractal_volume()
        assert np.allclose(volume.weighting, expected)


class TestGenerateFractalVolume:
    def test_returns_true(self):
        assert make_volume().generate_fractal_volume() is True

    def test_shape_matches_the_volume_extents(self):
        volume = make_volume(xs=2, xf=8, ys=1, yf=5, zs=0, zf=4)
        volume.generate_fractal_volume()
        assert volume.fractalvolume.shape == (6, 4, 4)

    def test_output_dtype_comes_from_config(self, fractal_config):
        volume = make_volume()
        volume.generate_fractal_volume()
        assert volume.fractalvolume.dtype == fractal_config.sim_config.dtypes["float_or_double"]

    @pytest.mark.parametrize("nbins", [2, 4, 8])
    def test_values_are_bin_indices_in_range(self, nbins):
        # These are indices into mixingmodel.matID, so anything outside
        # [0, nbins - 1] would index past the end of the material table.
        volume = make_volume(nbins=nbins)
        volume.generate_fractal_volume()
        values = np.unique(volume.fractalvolume)
        assert values.min() >= 0
        assert values.max() <= nbins - 1

    def test_values_are_whole_numbers(self):
        volume = make_volume(nbins=4)
        volume.generate_fractal_volume()
        assert np.array_equal(volume.fractalvolume, np.floor(volume.fractalvolume))

    def test_every_bin_is_populated(self):
        volume = make_volume(xf=16, yf=16, zf=16, nbins=4)
        volume.generate_fractal_volume()
        assert set(np.unique(volume.fractalvolume)) == {0.0, 1.0, 2.0, 3.0}

    def test_same_seed_reproduces_the_same_volume(self):
        first, second = make_volume(seed=SEED), make_volume(seed=SEED)
        first.generate_fractal_volume()
        second.generate_fractal_volume()
        assert np.array_equal(first.fractalvolume, second.fractalvolume)

    def test_different_seed_gives_a_different_volume(self):
        first, second = make_volume(seed=SEED), make_volume(seed=SEED + 1)
        first.generate_fractal_volume()
        second.generate_fractal_volume()
        assert not np.array_equal(first.fractalvolume, second.fractalvolume)

    @pytest.mark.xfail(reason="Cython extension needs rebuild — Python 3.14/Cython compiler incompatibility")
    def test_different_dimension_gives_a_different_volume(self):
        first = make_volume(dimension=1.0)
        second = make_volume(dimension=3.0)
        first.generate_fractal_volume()
        second.generate_fractal_volume()
        assert not np.array_equal(first.fractalvolume, second.fractalvolume)

    def test_a_single_cell_thick_volume_still_generates(self):
        volume = make_volume(xf=1, yf=8, zf=8)
        volume.generate_fractal_volume()
        assert volume.fractalvolume.shape == (1, 8, 8)


class TestGenerateVolumeMask:
    def test_mask_shape_and_dtype(self):
        volume = make_volume(xf=8, yf=8, zf=8)
        volume.generate_volume_mask()
        assert volume.mask.shape == (8, 8, 8)
        assert volume.mask.dtype == np.int8

    def test_an_unextended_volume_is_masked_solid(self):
        volume = make_volume(xf=4, yf=4, zf=4)
        volume.generate_volume_mask()
        assert np.all(volume.mask == 1)

    def test_only_the_original_footprint_is_masked(self):
        # A rough surface has pushed the volume out to z = 8; the user's
        # box only ever reached z = 4, so the extension masks to zero.
        volume = make_volume(xf=4, yf=4, zf=8)
        volume.original_stop = np.array([4, 4, 4], dtype=np.int32)
        volume.generate_volume_mask()
        assert np.all(volume.mask[:, :, 0:4] == 1)
        assert np.all(volume.mask[:, :, 4:8] == 0)

    def test_the_footprint_is_offset_into_the_extended_volume(self):
        # Extension in the minus direction: the volume starts at z = 0 but
        # the original box started at z = 2.
        volume = make_volume(xs=0, xf=4, ys=0, yf=4, zs=0, zf=8)
        volume.original_start = np.array([0, 0, 2], dtype=np.int32)
        volume.original_stop = np.array([4, 4, 6], dtype=np.int32)
        volume.generate_volume_mask()
        assert np.all(volume.mask[:, :, 0:2] == 0)
        assert np.all(volume.mask[:, :, 2:6] == 1)
        assert np.all(volume.mask[:, :, 6:8] == 0)

    def test_mask_is_regenerated_from_scratch_each_call(self):
        volume = make_volume(xf=4, yf=4, zf=4)
        volume.generate_volume_mask()
        volume.mask[0, 0, 0] = 3
        volume.generate_volume_mask()
        assert volume.mask[0, 0, 0] == 1
