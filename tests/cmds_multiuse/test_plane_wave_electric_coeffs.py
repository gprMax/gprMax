"""Regression tests for the DPW bulk-update electric curl coefficients.

updateElectricFields (gprMax/cython/plane_wave.pyx, equation 9 of Tan & Potter)
must pair the z-shifted H_x difference in the E_y row with CBz
(updatecoeffsE[3]), exactly like the axial and dispersive variants and the
CUDA/OpenCL/Metal kernel. The wrong pairing (CBx) is numerically invisible on
cubic grids, where dx == dz makes CBx == CBz, so these tests recover the
coefficient directly on synthetic arrays with distinct metric coefficients
instead of running a model. Both the production Cython bulk update and the
retained development-only Python bulk update must use the derivative's axis
to select its metric coefficient; this does not assert full solver parity
for the Python fallback.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.cython.plane_wave import updatePlaneWave_electric
from gprMax.sources import DiscretePlaneWave

pytestmark = pytest.mark.unit


def _run_electric_bulk_update(dtype, updatecoeffs, m, *, h_component=0, backend="cython"):
    """Advance one electric bulk update with only the wanted curl term active."""
    n = 20

    h_fields = np.zeros((3, n), dtype=dtype)
    e_fields = np.zeros((3, n), dtype=dtype)
    h_fields[h_component, :] = np.linspace(0.1, 0.5, n)

    integrals = tuple(np.zeros((2, 1), dtype=dtype) for _ in range(3))
    pml_coeffs = tuple(np.zeros((4, 1), dtype=dtype) for _ in range(6))
    fields3d = tuple(np.zeros((6, 6, 6), dtype=dtype) for _ in range(6))
    projections = np.zeros(6, dtype=np.float64)
    waveforms = np.zeros((2, 3, n), dtype=dtype)
    origin = np.zeros(3, dtype=np.int32)
    corners = np.zeros(8, dtype=np.int32)
    owned_lower = np.zeros(3, dtype=np.int32)
    owned_upper = np.zeros(3, dtype=np.int32)

    if backend == "python":
        grid = SimpleNamespace(
            ID=np.zeros((6, 1, 1, 1), dtype=np.uint32),
            updatecoeffsE=updatecoeffs.astype(dtype)[None, :],
        )
        dpw = DiscretePlaneWave.__new__(DiscretePlaneWave)
        dpw.m = m
        dpw.length = n
        dpw.corners = corners
        dpw.H_fields = h_fields
        dpw.E_fields = e_fields
        dpw.projections = projections
        dpw.waveformvalues_wholedt = waveforms
        dpw.update_electric_field_1D(grid, iteration=0, precompute=True)
        return h_fields, e_fields

    updatePlaneWave_electric(
        n,
        0,
        1,
        -1,
        h_fields,
        e_fields,
        *integrals,
        updatecoeffs.astype(dtype),
        np.zeros(5, dtype=dtype),
        *pml_coeffs,
        *fields3d,
        projections,
        waveforms,
        waveforms.copy(),
        m,
        origin,
        corners,
        owned_lower,
        owned_upper,
        True,
        0,
        1e-12,
        0.01,
        0.01,
        0.02,
        0.01,
        3e8,
        0.0,
        1e-9,
        1e9,
        b"gaussian",
    )
    return h_fields, e_fields


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["cython", "python"])
def test_ey_first_term_uses_cbz(dtype, backend):
    """The z-shifted H_x difference in the E_y row must scale with CBz."""
    # Distinct, exactly representable binary fractions prevent a wrong
    # coefficient from passing by accident.
    updatecoeffs = np.array([0.5, 0.375, 0.25, 0.125, 0.0])
    # m_z = 1 makes the H_x difference active; m_x = 0 zeroes the H_z term.
    h_fields, e_fields = _run_electric_bulk_update(
        dtype, updatecoeffs, np.array([0, 0, 1, 1], dtype=np.int32), backend=backend
    )

    d_hx = h_fields[0, 1:-1].astype(np.float64) - h_fields[0, :-2].astype(np.float64)
    assert np.all(d_hx != 0)
    np.testing.assert_allclose(e_fields[1, 1:-1], updatecoeffs[3] * d_hx, rtol=1e-5)
    np.testing.assert_array_equal(e_fields[0], 0)
    np.testing.assert_array_equal(e_fields[2], 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["cython", "python"])
def test_ey_second_term_uses_cbx(dtype, backend):
    """The x-shifted H_z difference in the E_y row must keep scaling with CBx."""
    updatecoeffs = np.array([0.5, 0.375, 0.25, 0.125, 0.0])
    # m_x = 1 makes the H_z difference active; m_z = 0 zeroes the H_x term.
    h_fields, e_fields = _run_electric_bulk_update(
        dtype,
        updatecoeffs,
        np.array([1, 0, 0, 1], dtype=np.int32),
        h_component=2,
        backend=backend,
    )

    d_hz = h_fields[2, 1:-1].astype(np.float64) - h_fields[2, :-2].astype(np.float64)
    assert np.all(d_hz != 0)
    np.testing.assert_allclose(e_fields[1, 1:-1], -updatecoeffs[1] * d_hz, rtol=1e-5)
    np.testing.assert_array_equal(e_fields[0], 0)
    np.testing.assert_array_equal(e_fields[2], 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["cython", "python"])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize(
    "e_component,h_component,axis,sign",
    [
        (0, 2, 1, 1),  # Ex: +dHz/dy
        (0, 1, 2, -1),  # Ex: -dHy/dz
        (1, 0, 2, 1),  # Ey: +dHx/dz
        (1, 2, 0, -1),  # Ey: -dHz/dx
        (2, 1, 0, 1),  # Ez: +dHy/dx
        (2, 0, 1, -1),  # Ez: -dHx/dy
    ],
)
def test_electric_curl_uses_derivative_axis(
    dtype, backend, direction, e_component, h_component, axis, sign
):
    """Isolate all six curl terms with distinct coefficients and signed shifts."""
    updatecoeffs = np.array([0.5, 0.375, 0.25, 0.125, 0.0])
    m = np.array([0, 0, 0, 1], dtype=np.int32)
    m[axis] = direction
    h_fields, e_fields = _run_electric_bulk_update(
        dtype, updatecoeffs, m, h_component=h_component, backend=backend
    )

    indices = np.arange(1, h_fields.shape[1] - 1)
    h = h_fields[h_component].astype(np.float64)
    difference = h[indices] - h[indices - direction]
    expected = np.zeros((3, len(indices)))
    expected[e_component] = sign * updatecoeffs[axis + 1] * difference
    assert np.all(expected[e_component] != 0)
    np.testing.assert_allclose(e_fields[:, 1:-1], expected, rtol=1e-5, atol=0)
