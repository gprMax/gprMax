"""Tests for the dispersive (Debye) per-iteration PMC ghost-node E update on
the 12 domain edges (gprMax.cython.symmetry_boundaries_dispersive), the edge
counterpart of test_symmetry_boundary_pmc_dispersive_updates.py, mirroring
test_symmetry_boundary_pmc_edges.py's structure for the non-dispersive case.

Design decision under direct test here (see
gprMax/cython/symmetry_boundaries_dispersive.pyx's module docstring): the
phi/T-array bookkeeping runs UNCONDITIONALLY at every edge cell, regardless
of the a_pmc/b_pmc flags - only the E-field self/ghost terms are gated by
them, exactly as in the non-dispersive edge kernels. This is what lets a
single-PMC-neighbour edge (where the other side has already been forced to
"pec" elsewhere, giving Ca=Cb=Ce=0 and a zero updatecoeffsdispersive row for
that material) degenerate correctly to zero without any explicit
PEC-transparency branch - not directly exercised here (that's the
build-time/material-ID concern covered by test_symmetry_boundary.py and the
end-to-end tests), just confirmed structurally: T still updates even when
both flags are False (never happens via the real per-iteration dispatch,
which drops such edges entirely, but is part of the documented per-cell
contract).
"""
import numpy as np
import pytest

from gprMax.cython import symmetry_boundaries_dispersive as sbd

nx, ny, nz = 6, 5, 4
maxpoles = 2
ca, cb1, cb2, cb3, ce = 0.7, 0.11, 0.22, 0.33, 0.15
alpha = [0.05, 0.03]
beta = [0.9, 0.8]
gamma = [0.02, 0.01]

_CALL_H_ORDER = {"Ez": ("Hx", "Hy"), "Ey": ("Hx", "Hz"), "Ex": ("Hy", "Hz")}
_T_FOR_E = {"Ez": "Tz", "Ey": "Ty", "Ex": "Tx"}


def _make_arrays():
    ID = np.ones((6, nx + 1, ny + 1, nz + 1), dtype=np.uint32)
    C = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [ca, cb1, cb2, cb3, ce]])
    D = np.array([[0.0] * (3 * maxpoles), [alpha[0], beta[0], gamma[0], alpha[1], beta[1], gamma[1]]])

    def field(offset, shape=(nx + 1, ny + 1, nz + 1)):
        arr = np.zeros(shape)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            arr[idx] = sum(1000**p * v for p, v in enumerate(idx)) + offset
        return arr

    Ex, Ey, Ez = field(1), field(2), field(3)
    Hx, Hy, Hz = field(4), field(5), field(6)
    tshape = (maxpoles, nx + 1, ny + 1, nz + 1)
    Tx, Ty, Tz = field(7, tshape), field(8, tshape), field(9, tshape)
    return ID, C, D, {"Ex": Ex, "Ey": Ey, "Ez": Ez}, {"Hx": Hx, "Hy": Hy, "Hz": Hz}, {"Tx": Tx, "Ty": Ty, "Tz": Tz}


def _phi_and_new_t(t_old_poles, e_old):
    phi = 0.0
    t_new = []
    for p in range(maxpoles):
        phi += alpha[p] * t_old_poles[p]
        t_new.append(beta[p] * t_old_poles[p] + gamma[p] * e_old)
    return phi, t_new


# Ghost-term helpers - identical formulas to test_symmetry_boundary_pmc_edges.py.
def _gx0_ez(H, j, k):
    return 2 * cb1 * H["Hy"][0, j, k]


def _gxmax_ez(H, j, k):
    return -2 * cb1 * H["Hy"][nx - 1, j, k]


def _gy0_ez(H, i, k):
    return -2 * cb2 * H["Hx"][i, 0, k]


def _gymax_ez(H, i, k):
    return 2 * cb2 * H["Hx"][i, ny - 1, k]


def _gx0_ey(H, j, k):
    return -2 * cb1 * H["Hz"][0, j, k]


def _gxmax_ey(H, j, k):
    return 2 * cb1 * H["Hz"][nx - 1, j, k]


def _gz0_ey(H, i, j):
    return 2 * cb3 * H["Hx"][i, j, 0]


def _gzmax_ey(H, i, j):
    return -2 * cb3 * H["Hx"][i, j, nz - 1]


def _gy0_ex(H, i, k):
    return 2 * cb2 * H["Hz"][i, 0, k]


def _gymax_ex(H, i, k):
    return -2 * cb2 * H["Hz"][i, ny - 1, k]


def _gz0_ex(H, i, j):
    return -2 * cb3 * H["Hy"][i, j, 0]


def _gzmax_ex(H, i, j):
    return 2 * cb3 * H["Hy"][i, j, nz - 1]


# Each entry: (name, phase_a_func, phase_b_func, position(t)->(i,j,k), free
# range, E component, ghostA(H,t), ghostB(H,t)).
_EDGES = [
    ("Ez_X0_Y0", sbd.update_symmetry_boundary_electric_dispersive_Ez_X0_Y0,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ez_X0_Y0,
     lambda k: (0, 0, k), range(nz), "Ez",
     lambda H, k: _gx0_ez(H, 0, k), lambda H, k: _gy0_ez(H, 0, k)),
    ("Ez_X0_YMax", sbd.update_symmetry_boundary_electric_dispersive_Ez_X0_YMax,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ez_X0_YMax,
     lambda k: (0, ny, k), range(nz), "Ez",
     lambda H, k: _gx0_ez(H, ny, k), lambda H, k: _gymax_ez(H, 0, k)),
    ("Ez_XMax_Y0", sbd.update_symmetry_boundary_electric_dispersive_Ez_XMax_Y0,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ez_XMax_Y0,
     lambda k: (nx, 0, k), range(nz), "Ez",
     lambda H, k: _gxmax_ez(H, 0, k), lambda H, k: _gy0_ez(H, nx, k)),
    ("Ez_XMax_YMax", sbd.update_symmetry_boundary_electric_dispersive_Ez_XMax_YMax,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ez_XMax_YMax,
     lambda k: (nx, ny, k), range(nz), "Ez",
     lambda H, k: _gxmax_ez(H, ny, k), lambda H, k: _gymax_ez(H, nx, k)),
    ("Ey_X0_Z0", sbd.update_symmetry_boundary_electric_dispersive_Ey_X0_Z0,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ey_X0_Z0,
     lambda j: (0, j, 0), range(ny), "Ey",
     lambda H, j: _gx0_ey(H, j, 0), lambda H, j: _gz0_ey(H, 0, j)),
    ("Ey_X0_ZMax", sbd.update_symmetry_boundary_electric_dispersive_Ey_X0_ZMax,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ey_X0_ZMax,
     lambda j: (0, j, nz), range(ny), "Ey",
     lambda H, j: _gx0_ey(H, j, nz), lambda H, j: _gzmax_ey(H, 0, j)),
    ("Ey_XMax_Z0", sbd.update_symmetry_boundary_electric_dispersive_Ey_XMax_Z0,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ey_XMax_Z0,
     lambda j: (nx, j, 0), range(ny), "Ey",
     lambda H, j: _gxmax_ey(H, j, 0), lambda H, j: _gz0_ey(H, nx, j)),
    ("Ey_XMax_ZMax", sbd.update_symmetry_boundary_electric_dispersive_Ey_XMax_ZMax,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ey_XMax_ZMax,
     lambda j: (nx, j, nz), range(ny), "Ey",
     lambda H, j: _gxmax_ey(H, j, nz), lambda H, j: _gzmax_ey(H, nx, j)),
    ("Ex_Y0_Z0", sbd.update_symmetry_boundary_electric_dispersive_Ex_Y0_Z0,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ex_Y0_Z0,
     lambda i: (i, 0, 0), range(nx), "Ex",
     lambda H, i: _gy0_ex(H, i, 0), lambda H, i: _gz0_ex(H, i, 0)),
    ("Ex_Y0_ZMax", sbd.update_symmetry_boundary_electric_dispersive_Ex_Y0_ZMax,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ex_Y0_ZMax,
     lambda i: (i, 0, nz), range(nx), "Ex",
     lambda H, i: _gy0_ex(H, i, nz), lambda H, i: _gzmax_ex(H, i, 0)),
    ("Ex_YMax_Z0", sbd.update_symmetry_boundary_electric_dispersive_Ex_YMax_Z0,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ex_YMax_Z0,
     lambda i: (i, ny, 0), range(nx), "Ex",
     lambda H, i: _gymax_ex(H, i, 0), lambda H, i: _gz0_ex(H, i, ny)),
    ("Ex_YMax_ZMax", sbd.update_symmetry_boundary_electric_dispersive_Ex_YMax_ZMax,
     sbd.update_symmetry_boundary_electric_dispersive_b_Ex_YMax_ZMax,
     lambda i: (i, ny, nz), range(nx), "Ex",
     lambda H, i: _gymax_ex(H, i, nz), lambda H, i: _gzmax_ex(H, i, ny)),
]


@pytest.mark.parametrize("edge", _EDGES, ids=[e[0] for e in _EDGES])
@pytest.mark.parametrize("a_pmc,b_pmc", [(True, False), (False, True), (True, True)])
def test_edge_matches_independent_hand_derivation(edge, a_pmc, b_pmc):
    name, func, func_b, pos_fn, free_range, comp, ghost_a, ghost_b = edge
    ID, C, D, E, H, T = _make_arrays()
    comp_arr = E[comp]
    t_arr = T[_T_FOR_E[comp]]
    h1_name, h2_name = _CALL_H_ORDER[comp]

    expected_e = comp_arr.copy()
    expected_t_after_a = t_arr.copy()
    expected_t_after_b = t_arr.copy()
    for t in free_range:
        pos = pos_fn(t)
        phi, t_new = _phi_and_new_t(t_arr[(slice(None),) + pos], comp_arr[pos])
        expected_t_after_a[(slice(None),) + pos] = t_new
        if a_pmc or b_pmc:
            expected_e[pos] = ca * comp_arr[pos] - ce * phi
        if a_pmc:
            expected_e[pos] = expected_e[pos] + ghost_a(H, t)
        if b_pmc:
            expected_e[pos] = expected_e[pos] + ghost_b(H, t)
        expected_t_after_b[(slice(None),) + pos] = [
            t_new[p] - gamma[p] * expected_e[pos] for p in range(maxpoles)
        ]

    func(nx, ny, nz, 1, a_pmc, b_pmc, maxpoles, C, D, ID, t_arr, comp_arr, H[h1_name], H[h2_name])

    assert np.allclose(comp_arr, expected_e), f"{name} a_pmc={a_pmc} b_pmc={b_pmc} E mismatch"
    assert np.allclose(t_arr, expected_t_after_a), f"{name} a_pmc={a_pmc} b_pmc={b_pmc} T (phase A) mismatch"

    func_b(nx, ny, nz, 1, maxpoles, D, ID, t_arr, comp_arr)

    assert np.allclose(t_arr, expected_t_after_b), f"{name} a_pmc={a_pmc} b_pmc={b_pmc} T (phase B) mismatch"


@pytest.mark.parametrize("edge", _EDGES, ids=[e[0] for e in _EDGES])
def test_edge_neither_pmc_leaves_e_untouched_but_still_updates_t(edge):
    """With both flags False (never actually reached via the real
    per-iteration dispatch, which drops such edges entirely), E must stay
    untouched - but T still updates, since the phi/T bookkeeping is
    unconditional. This is the structural property that lets a
    single-PMC-neighbour edge's T correctly keep tracking a real (non-pec)
    dispersive material's state even when momentarily viewed with both
    flags False."""
    name, func, func_b, pos_fn, free_range, comp, ghost_a, ghost_b = edge
    ID, C, D, E, H, T = _make_arrays()
    comp_arr = E[comp]
    t_arr = T[_T_FOR_E[comp]]
    h1_name, h2_name = _CALL_H_ORDER[comp]
    e_before = comp_arr.copy()

    expected_t_after_a = t_arr.copy()
    for t in free_range:
        pos = pos_fn(t)
        _, t_new = _phi_and_new_t(t_arr[(slice(None),) + pos], comp_arr[pos])
        expected_t_after_a[(slice(None),) + pos] = t_new

    func(nx, ny, nz, 1, False, False, maxpoles, C, D, ID, t_arr, comp_arr, H[h1_name], H[h2_name])

    assert np.array_equal(comp_arr, e_before), f"{name}: E changed even though neither face is PMC"
    assert np.allclose(t_arr, expected_t_after_a), f"{name}: T did not update despite unconditional bookkeeping"
