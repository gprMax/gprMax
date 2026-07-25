"""Tests for the complex-pole dispersive (Lorentz/Drude) per-iteration PMC
ghost-node E update (gprMax.cython.symmetry_boundaries_dispersive_complex),
the face-interior (non-edge) counterpart of
test_symmetry_boundary_pmc_dispersive_updates.py (Debye/real-pole) for
materials with complex ADE poles.

The only difference from the real-pole file: Tx/Ty/Tz and
updatecoeffsdispersive are complex here, so phi (which must stay real - it
feeds directly into a real E-field update) accumulates
Real(updatecoeffsdispersive[...]) * Real(T[...]) per pole - the real part of
each factor individually, matching
fields_updates_dispersive_template.jinja's own complex branch exactly (not
"fixed" to Real(a*b), which would be a different, not-implemented, formula -
see gprMax/cython/symmetry_boundaries_dispersive_complex.pyx's module
docstring). The T-array recursion itself and Phase B are unchanged in form
from the real-pole case - ordinary complex arithmetic handles them directly.

maxpoles=2 with genuinely complex (non-zero imaginary part) per-pole
coefficients is used throughout, specifically to exercise that the .real
extraction is happening correctly (a bug that silently used the full
complex value, or zero, or the imaginary part, would not be caught by
real-valued test coefficients).
"""
import numpy as np
import pytest

from gprMax.cython import symmetry_boundaries_dispersive_complex as sbdc
from gprMax.grid.fdtd_grid import FDTDGrid

nx, ny, nz = 6, 5, 4
maxpoles = 2
ca, cb1, cb2, cb3, ce = 0.7, 0.11, 0.22, 0.33, 0.15
# Per-pole complex dispersive coefficients: alpha (phi weight), beta (T
# decay), gamma (E-driving) - all genuinely complex.
alpha = [0.05 + 0.09j, 0.03 - 0.06j]
beta = [0.9 - 0.2j, 0.8 + 0.15j]
gamma = [0.02 + 0.01j, 0.01 - 0.03j]

_FACE_FUNCS = {
    "x0": sbdc.update_symmetry_boundary_electric_dispersive_x0,
    "xmax": sbdc.update_symmetry_boundary_electric_dispersive_xmax,
    "y0": sbdc.update_symmetry_boundary_electric_dispersive_y0,
    "ymax": sbdc.update_symmetry_boundary_electric_dispersive_ymax,
    "z0": sbdc.update_symmetry_boundary_electric_dispersive_z0,
    "zmax": sbdc.update_symmetry_boundary_electric_dispersive_zmax,
}

_FACE_FUNCS_B = {
    "x0": sbdc.update_symmetry_boundary_electric_dispersive_b_x0,
    "xmax": sbdc.update_symmetry_boundary_electric_dispersive_b_xmax,
    "y0": sbdc.update_symmetry_boundary_electric_dispersive_b_y0,
    "ymax": sbdc.update_symmetry_boundary_electric_dispersive_b_ymax,
    "z0": sbdc.update_symmetry_boundary_electric_dispersive_b_z0,
    "zmax": sbdc.update_symmetry_boundary_electric_dispersive_b_zmax,
}


def _update_face_interior(grid, face):
    _FACE_FUNCS[face](
        grid.nx, grid.ny, grid.nz, 1, maxpoles,
        grid.updatecoeffsE, grid.updatecoeffsdispersive, grid.ID,
        grid.Tx, grid.Ty, grid.Tz,
        grid.Ex, grid.Ey, grid.Ez,
        grid.Hx, grid.Hy, grid.Hz,
    )


def _update_face_interior_b(grid, face):
    _FACE_FUNCS_B[face](
        grid.nx, grid.ny, grid.nz, 1, maxpoles,
        grid.updatecoeffsdispersive, grid.ID,
        grid.Tx, grid.Ty, grid.Tz,
        grid.Ex, grid.Ey, grid.Ez,
    )


def _make_grid():
    grid = FDTDGrid()
    grid.nx, grid.ny, grid.nz = nx, ny, nz
    grid.ID = np.ones((6, nx + 1, ny + 1, nz + 1), dtype=np.uint32)
    grid.updatecoeffsE = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [ca, cb1, cb2, cb3, ce],
        ]
    )
    grid.updatecoeffsdispersive = np.array(
        [
            [0.0] * (3 * maxpoles),
            [alpha[0], beta[0], gamma[0], alpha[1], beta[1], gamma[1]],
        ],
        dtype=np.complex128,
    )

    def field(offset, shape=None, dtype=np.float64):
        shape = shape or (nx + 1, ny + 1, nz + 1)
        arr = np.zeros(shape, dtype=dtype)
        it = np.nditer(arr, flags=["multi_index", "refs_ok"])
        for _ in it:
            idx = it.multi_index
            arr[idx] = sum(1000**p * v for p, v in enumerate(idx)) + offset
        return arr

    grid.Ex, grid.Ey, grid.Ez = field(1), field(2), field(3)
    grid.Hx, grid.Hy, grid.Hz = field(4), field(5), field(6)
    # Complex T arrays - give them a genuinely non-zero imaginary part too
    # (offset*1j on top of the real ramp), not just real numbers cast wide.
    tshape = (maxpoles, nx + 1, ny + 1, nz + 1)
    grid.Tx = field(7, tshape, dtype=np.complex128) + 0.5j * field(0.7, tshape, dtype=np.complex128)
    grid.Ty = field(8, tshape, dtype=np.complex128) + 0.5j * field(0.8, tshape, dtype=np.complex128)
    grid.Tz = field(9, tshape, dtype=np.complex128) + 0.5j * field(0.9, tshape, dtype=np.complex128)
    return grid


def _phi_and_new_t(t_old_poles, e_old):
    """Independently re-derives phi (real: Re(alpha)*Re(T) per pole, NOT
    Re(alpha*T)) and the new per-pole complex T values for a single grid
    position - matches the Cython pole loop exactly."""
    phi = 0.0
    t_new = []
    for p in range(maxpoles):
        phi += alpha[p].real * t_old_poles[p].real
        t_new.append(beta[p] * t_old_poles[p] + gamma[p] * e_old)
    return phi, t_new


def _expected_x0(grid):
    Ey, Ez = grid.Ey.copy(), grid.Ez.copy()
    Ty, Tz = grid.Ty.copy(), grid.Tz.copy()
    Ty_b, Tz_b = grid.Ty.copy(), grid.Tz.copy()
    for j in range(ny):
        for k in range(1, nz):
            phi, t_new = _phi_and_new_t(grid.Ty[:, 0, j, k], grid.Ey[0, j, k])
            Ty[:, 0, j, k] = t_new
            Ey[0, j, k] = (
                ca * grid.Ey[0, j, k]
                + cb3 * (grid.Hx[0, j, k] - grid.Hx[0, j, k - 1])
                - cb1 * (2 * grid.Hz[0, j, k])
                - ce * phi
            )
            Ty_b[:, 0, j, k] = [t_new[p] - gamma[p] * Ey[0, j, k] for p in range(maxpoles)]
    for j in range(1, ny):
        for k in range(nz):
            phi, t_new = _phi_and_new_t(grid.Tz[:, 0, j, k], grid.Ez[0, j, k])
            Tz[:, 0, j, k] = t_new
            Ez[0, j, k] = (
                ca * grid.Ez[0, j, k]
                - cb2 * (grid.Hx[0, j, k] - grid.Hx[0, j - 1, k])
                + cb1 * (2 * grid.Hy[0, j, k])
                - ce * phi
            )
            Tz_b[:, 0, j, k] = [t_new[p] - gamma[p] * Ez[0, j, k] for p in range(maxpoles)]
    return Ey, Ez, Ty, Tz, Ty_b, Tz_b


def _expected_xmax(grid):
    Ey, Ez = grid.Ey.copy(), grid.Ez.copy()
    Ty, Tz = grid.Ty.copy(), grid.Tz.copy()
    Ty_b, Tz_b = grid.Ty.copy(), grid.Tz.copy()
    for j in range(ny):
        for k in range(1, nz):
            phi, t_new = _phi_and_new_t(grid.Ty[:, nx, j, k], grid.Ey[nx, j, k])
            Ty[:, nx, j, k] = t_new
            Ey[nx, j, k] = (
                ca * grid.Ey[nx, j, k]
                + cb3 * (grid.Hx[nx, j, k] - grid.Hx[nx, j, k - 1])
                + cb1 * (2 * grid.Hz[nx - 1, j, k])
                - ce * phi
            )
            Ty_b[:, nx, j, k] = [t_new[p] - gamma[p] * Ey[nx, j, k] for p in range(maxpoles)]
    for j in range(1, ny):
        for k in range(nz):
            phi, t_new = _phi_and_new_t(grid.Tz[:, nx, j, k], grid.Ez[nx, j, k])
            Tz[:, nx, j, k] = t_new
            Ez[nx, j, k] = (
                ca * grid.Ez[nx, j, k]
                - cb2 * (grid.Hx[nx, j, k] - grid.Hx[nx, j - 1, k])
                - cb1 * (2 * grid.Hy[nx - 1, j, k])
                - ce * phi
            )
            Tz_b[:, nx, j, k] = [t_new[p] - gamma[p] * Ez[nx, j, k] for p in range(maxpoles)]
    return Ey, Ez, Ty, Tz, Ty_b, Tz_b


def _expected_y0(grid):
    Ex, Ez = grid.Ex.copy(), grid.Ez.copy()
    Tx, Tz = grid.Tx.copy(), grid.Tz.copy()
    Tx_b, Tz_b = grid.Tx.copy(), grid.Tz.copy()
    for i in range(nx):
        for k in range(1, nz):
            phi, t_new = _phi_and_new_t(grid.Tx[:, i, 0, k], grid.Ex[i, 0, k])
            Tx[:, i, 0, k] = t_new
            Ex[i, 0, k] = (
                ca * grid.Ex[i, 0, k]
                - cb3 * (grid.Hy[i, 0, k] - grid.Hy[i, 0, k - 1])
                + cb2 * (2 * grid.Hz[i, 0, k])
                - ce * phi
            )
            Tx_b[:, i, 0, k] = [t_new[p] - gamma[p] * Ex[i, 0, k] for p in range(maxpoles)]
    for i in range(1, nx):
        for k in range(nz):
            phi, t_new = _phi_and_new_t(grid.Tz[:, i, 0, k], grid.Ez[i, 0, k])
            Tz[:, i, 0, k] = t_new
            Ez[i, 0, k] = (
                ca * grid.Ez[i, 0, k]
                + cb1 * (grid.Hy[i, 0, k] - grid.Hy[i - 1, 0, k])
                - cb2 * (2 * grid.Hx[i, 0, k])
                - ce * phi
            )
            Tz_b[:, i, 0, k] = [t_new[p] - gamma[p] * Ez[i, 0, k] for p in range(maxpoles)]
    return Ex, Ez, Tx, Tz, Tx_b, Tz_b


def _expected_ymax(grid):
    Ex, Ez = grid.Ex.copy(), grid.Ez.copy()
    Tx, Tz = grid.Tx.copy(), grid.Tz.copy()
    Tx_b, Tz_b = grid.Tx.copy(), grid.Tz.copy()
    for i in range(nx):
        for k in range(1, nz):
            phi, t_new = _phi_and_new_t(grid.Tx[:, i, ny, k], grid.Ex[i, ny, k])
            Tx[:, i, ny, k] = t_new
            Ex[i, ny, k] = (
                ca * grid.Ex[i, ny, k]
                - cb3 * (grid.Hy[i, ny, k] - grid.Hy[i, ny, k - 1])
                - cb2 * (2 * grid.Hz[i, ny - 1, k])
                - ce * phi
            )
            Tx_b[:, i, ny, k] = [t_new[p] - gamma[p] * Ex[i, ny, k] for p in range(maxpoles)]
    for i in range(1, nx):
        for k in range(nz):
            phi, t_new = _phi_and_new_t(grid.Tz[:, i, ny, k], grid.Ez[i, ny, k])
            Tz[:, i, ny, k] = t_new
            Ez[i, ny, k] = (
                ca * grid.Ez[i, ny, k]
                + cb1 * (grid.Hy[i, ny, k] - grid.Hy[i - 1, ny, k])
                + cb2 * (2 * grid.Hx[i, ny - 1, k])
                - ce * phi
            )
            Tz_b[:, i, ny, k] = [t_new[p] - gamma[p] * Ez[i, ny, k] for p in range(maxpoles)]
    return Ex, Ez, Tx, Tz, Tx_b, Tz_b


def _expected_z0(grid):
    Ex, Ey = grid.Ex.copy(), grid.Ey.copy()
    Tx, Ty = grid.Tx.copy(), grid.Ty.copy()
    Tx_b, Ty_b = grid.Tx.copy(), grid.Ty.copy()
    for i in range(nx):
        for j in range(1, ny):
            phi, t_new = _phi_and_new_t(grid.Tx[:, i, j, 0], grid.Ex[i, j, 0])
            Tx[:, i, j, 0] = t_new
            Ex[i, j, 0] = (
                ca * grid.Ex[i, j, 0]
                + cb2 * (grid.Hz[i, j, 0] - grid.Hz[i, j - 1, 0])
                - cb3 * (2 * grid.Hy[i, j, 0])
                - ce * phi
            )
            Tx_b[:, i, j, 0] = [t_new[p] - gamma[p] * Ex[i, j, 0] for p in range(maxpoles)]
    for i in range(1, nx):
        for j in range(ny):
            phi, t_new = _phi_and_new_t(grid.Ty[:, i, j, 0], grid.Ey[i, j, 0])
            Ty[:, i, j, 0] = t_new
            Ey[i, j, 0] = (
                ca * grid.Ey[i, j, 0]
                - cb1 * (grid.Hz[i, j, 0] - grid.Hz[i - 1, j, 0])
                + cb3 * (2 * grid.Hx[i, j, 0])
                - ce * phi
            )
            Ty_b[:, i, j, 0] = [t_new[p] - gamma[p] * Ey[i, j, 0] for p in range(maxpoles)]
    return Ex, Ey, Tx, Ty, Tx_b, Ty_b


def _expected_zmax(grid):
    Ex, Ey = grid.Ex.copy(), grid.Ey.copy()
    Tx, Ty = grid.Tx.copy(), grid.Ty.copy()
    Tx_b, Ty_b = grid.Tx.copy(), grid.Ty.copy()
    for i in range(nx):
        for j in range(1, ny):
            phi, t_new = _phi_and_new_t(grid.Tx[:, i, j, nz], grid.Ex[i, j, nz])
            Tx[:, i, j, nz] = t_new
            Ex[i, j, nz] = (
                ca * grid.Ex[i, j, nz]
                + cb2 * (grid.Hz[i, j, nz] - grid.Hz[i, j - 1, nz])
                + cb3 * (2 * grid.Hy[i, j, nz - 1])
                - ce * phi
            )
            Tx_b[:, i, j, nz] = [t_new[p] - gamma[p] * Ex[i, j, nz] for p in range(maxpoles)]
    for i in range(1, nx):
        for j in range(ny):
            phi, t_new = _phi_and_new_t(grid.Ty[:, i, j, nz], grid.Ey[i, j, nz])
            Ty[:, i, j, nz] = t_new
            Ey[i, j, nz] = (
                ca * grid.Ey[i, j, nz]
                - cb1 * (grid.Hz[i, j, nz] - grid.Hz[i - 1, j, nz])
                - cb3 * (2 * grid.Hx[i, j, nz - 1])
                - ce * phi
            )
            Ty_b[:, i, j, nz] = [t_new[p] - gamma[p] * Ey[i, j, nz] for p in range(maxpoles)]
    return Ex, Ey, Tx, Ty, Tx_b, Ty_b


_FACES = {
    "x0": (_expected_x0, ("Ey", "Ez"), ("Ty", "Tz")),
    "xmax": (_expected_xmax, ("Ey", "Ez"), ("Ty", "Tz")),
    "y0": (_expected_y0, ("Ex", "Ez"), ("Tx", "Tz")),
    "ymax": (_expected_ymax, ("Ex", "Ez"), ("Tx", "Tz")),
    "z0": (_expected_z0, ("Ex", "Ey"), ("Tx", "Ty")),
    "zmax": (_expected_zmax, ("Ex", "Ey"), ("Tx", "Ty")),
}


@pytest.mark.parametrize("face", _FACES.keys())
def test_face_interior_matches_independent_hand_derivation(face):
    expected_fn, e_components, t_components = _FACES[face]
    grid = _make_grid()
    expected_e1, expected_e2, expected_t1, expected_t2, _, _ = expected_fn(grid)

    _update_face_interior(grid, face)

    assert np.allclose(getattr(grid, e_components[0]), expected_e1), f"{face}/{e_components[0]} mismatch"
    assert np.allclose(getattr(grid, e_components[1]), expected_e2), f"{face}/{e_components[1]} mismatch"
    assert np.allclose(getattr(grid, t_components[0]), expected_t1), f"{face}/{t_components[0]} mismatch"
    assert np.allclose(getattr(grid, t_components[1]), expected_t2), f"{face}/{t_components[1]} mismatch"
    # E must stay strictly real (float64), not accidentally promoted to
    # complex by having picked up a complex intermediate somewhere.
    assert not np.iscomplexobj(getattr(grid, e_components[0]))
    assert not np.iscomplexobj(getattr(grid, e_components[1]))


@pytest.mark.parametrize("face", _FACES.keys())
def test_face_interior_does_not_touch_the_other_components(face):
    _, e_components, t_components = _FACES[face]
    all_e = ("Ex", "Ey", "Ez")
    all_t = ("Tx", "Ty", "Tz")
    untouched_e = [c for c in all_e if c not in e_components]
    untouched_t = [c for c in all_t if c not in t_components]

    grid = _make_grid()
    before = {c: getattr(grid, c).copy() for c in untouched_e + untouched_t + ["Hx", "Hy", "Hz"]}

    _update_face_interior(grid, face)

    for c in before:
        assert np.array_equal(getattr(grid, c), before[c]), f"{face}: {c} changed unexpectedly"


@pytest.mark.parametrize("face", _FACES.keys())
def test_phase_b_corrects_t_using_the_e_value_current_when_it_runs(face):
    """Same regression guard as the real-pole file's equivalent test: phase
    B's correction (T -= gamma*E) must use whatever (real) E value is
    current at the time phase B runs - here, phase A's own output E."""
    expected_fn, e_components, t_components = _FACES[face]

    grid = _make_grid()
    _, _, _, _, expected_t1_after_b, expected_t2_after_b = expected_fn(grid)

    _update_face_interior(grid, face)
    _update_face_interior_b(grid, face)

    assert np.allclose(getattr(grid, t_components[0]), expected_t1_after_b), (
        f"{face}: phase B's T ({t_components[0]}) does not match independent hand derivation"
    )
    assert np.allclose(getattr(grid, t_components[1]), expected_t2_after_b), (
        f"{face}: phase B's T ({t_components[1]}) does not match independent hand derivation"
    )
