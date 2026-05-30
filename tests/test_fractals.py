import numpy as np

from gprMax.fractals import _bin_fractal_values


def test_bin_fractal_values_range():
    """Ensure binning produces indices in 0..nbins-1 and handles simple array."""
    arr = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    nbins = 5
    out = _bin_fractal_values(arr, nbins)
    assert out.min() >= 0
    assert out.max() < nbins
    # ensure that bin indices are non-decreasing (i.e. successive values
    # should map to same or higher bin)
    assert np.all(np.diff(out) >= 0)


def test_bin_fractal_values_edgecases():
    """Check behaviour at array edges and with repeated values."""
    arr = np.array([0.0, 0.0, 1.0, 1.0])
    nbins = 3
    out = _bin_fractal_values(arr, nbins)
    assert out.shape == arr.shape
    assert np.all(out >= 0) and np.all(out < nbins)
    # identical input values should map to identical output indices
    assert out[0] == out[1] and out[2] == out[3]
