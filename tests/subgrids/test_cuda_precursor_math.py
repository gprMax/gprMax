"""CPU-side setup mathematics for the CUDA HSG precursor kernels."""

import itertools

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from gprMax.subgrids.cuda_precursor_nodes import _descriptor, _spline_axis_weights

pytestmark = pytest.mark.unit


def test_cuda_descriptor_signature_preserves_64_bit_offsets_on_windows():
    from gprMax.cuda_opencl.knl_subgrid_precursors import gather_taps

    # Host arguments are np.int64. C/C++ long is only 32 bits on Windows;
    # OpenCL long, unlike C/C++ long, is specified as a 64-bit type.
    signature = gather_taps["args_cuda"].template
    for name in ("base0", "base1", "base2", "base3", "stride_a", "stride_b"):
        assert f"const long long {name}" in signature


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("index", [0, -1, 2])
@pytest.mark.parametrize("step", [1, 2, -1, -2])
def test_descriptor_matches_numpy_without_allocating_volume(axis, index, step):
    shape = (5, 6, 7)
    slices = [slice(None, None, step)] * 3
    slices[axis] = index
    view = np.arange(np.prod(shape)).reshape(shape)[tuple(slices)]
    base, sa, sb, na, nb = _descriptor(shape, tuple(slices))
    assert (na, nb) == view.shape
    np.testing.assert_array_equal(base + sa * np.arange(na)[:, None] + sb * np.arange(nb), view)


def test_descriptor_large_shape_uses_only_scalar_arithmetic(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Do not allocate a volume-sized temporary")

    monkeypatch.setattr(np, "arange", forbidden)
    shape = (100_001, 100_002, 100_003)
    base, sa, sb, na, nb = _descriptor(shape, (99_999, slice(2, 5), slice(3, 8, 2)))
    assert base == 99_999 * shape[1] * shape[2] + 2 * shape[2] + 3
    assert (sa, sb, na, nb) == (shape[2], 2, 3, 3)


def test_descriptor_singleton_axis_has_zero_stride():
    assert _descriptor((4, 5, 6), (1, slice(2, 3), slice(4, 5))) == (46, 0, 0, 1, 1)


@pytest.mark.parametrize(
    "slc", [(0, 1, slice(None)), (0, slice(1, 1), slice(None)), (0, slice(None)), (0, None, slice(None))]
)
def test_descriptor_rejects_non_2d_or_empty_views(slc):
    with pytest.raises((ValueError, TypeError)):
        _descriptor((5, 6, 7), slc)


@pytest.mark.parametrize("order", range(1, 6))
@pytest.mark.parametrize("size", [6, 13])
def test_separable_spline_weights_match_cpu_fitpack_including_edges(order, size):
    rng = np.random.default_rng(834)
    x, y = np.arange(size) + 0.5, np.arange(size + 2)
    # Include clamped exterior points and exact endpoints on both axes.
    tx, ty = np.linspace(-0.5, size + 0.5, 3 * size), np.linspace(-1, size + 2, 3 * size + 5)
    field = rng.normal(size=(len(x), len(y)))
    reference = RectBivariateSpline(x, y, field, kx=order, ky=order)(tx, ty)
    wx, wy = _spline_axis_weights(x, tx, order), _spline_axis_weights(y, ty, order)
    np.testing.assert_allclose(wx @ field @ wy.T, reference, rtol=1e-12, atol=2e-13)
    np.testing.assert_allclose(wx.sum(axis=1), 1, rtol=1e-13, atol=1e-13)


def test_cpu_api_actually_accepts_all_five_spline_degrees():
    from gprMax.subgrids.user_objects import SubGridHSG
    from gprMax.subgrids.precursor_nodes import PrecursorNodes

    rng = np.random.default_rng(5)
    field = rng.normal(size=(8, 9))
    results = []
    for order in range(1, 6):
        declaration = SubGridHSG(p1=(0.03,) * 3, p2=(0.06,) * 3, id="fine", interpolation=order)
        assert declaration.kwargs["interpolation"] == order
        precursor = PrecursorNodes.__new__(PrecursorNodes)
        precursor.interpolation, precursor.ratio, precursor.d = order, 3, 1 / 6
        coords = precursor.create_interpolated_coords(True, field)
        results.append(precursor.interpolate_to_sub_grid(field, coords))
    # Confirm that the CPU really uses the option, not only that it parses it.
    for a, b in itertools.combinations(results, 2):
        assert np.max(np.abs(a - b)) > 0.01
