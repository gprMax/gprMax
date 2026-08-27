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

"""Tests for the per-iteration PMC ghost-node E update on the 12 domain
edges (gprMax.cython.symmetry_boundaries.update_symmetry_boundary_electric_Ez_X0_Y0
and its 11 siblings) - the edge piece of the #symmetry_boundary PMC feature,
complementing the face-interior tests in test_symmetry_boundary_pmc_updates.py.

The simplified per-edge scheme applies the self term (Ca*E) once if EITHER
bordering face is PMC; each face then
separately, additively contributes its own doubled ghost term only if THAT
SPECIFIC face is PMC. A single-PMC-neighbour edge reduces correctly to zero
on its own once the edge's ID has been forced to pec elsewhere (Ca=Cb=0) -
not exercised directly here (that's a build-time concern, see
test_symmetry_boundary.py), just confirmed structurally: passing only one of
the two flags produces exactly the single-face contribution, nothing more.

Each expected-value computation below is a plain Python loop (not
vectorized), independently re-deriving the same formula the Cython
implementation computes - so a transcription bug in one wouldn't be masked
by the same bug in the other.
"""
import numpy as np
import pytest

from gprMax.cython import symmetry_boundaries as sb

nx, ny, nz = 6, 5, 4
ca, cb1, cb2, cb3 = 0.7, 0.11, 0.22, 0.33

# The two H arrays each cython edge function takes, in its actual positional
# order (matching gprMax/cython/symmetry_boundaries.pyx exactly).
_CALL_H_ORDER = {"Ez": ("Hx", "Hy"), "Ey": ("Hx", "Hz"), "Ex": ("Hy", "Hz")}


def _make_arrays():
    ID = np.ones((6, nx + 1, ny + 1, nz + 1), dtype=np.uint32)
    C = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [ca, cb1, cb2, cb3, 0.0]])

    def field(offset):
        arr = np.zeros((nx + 1, ny + 1, nz + 1))
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    arr[i, j, k] = 1000 * i + 100 * j + 10 * k + offset
        return arr

    Ex, Ey, Ez = field(1), field(2), field(3)
    Hx, Hy, Hz = field(4), field(5), field(6)
    return ID, C, {"Ex": Ex, "Ey": Ey, "Ez": Ez}, {"Hx": Hx, "Hy": Hy, "Hz": Hz}


# Ghost-term helpers, each taking the full Hx/Hy/Hz dict and the fixed/free
# indices - independently re-deriving the formulas already verified for the
# face-interior update, referenced by name (not position) to avoid any
# argument-order ambiguity.
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


# Each entry: (name, cython_func, position(t)->(i,j,k), free range, component,
# ghostA(H,t), ghostB(H,t)).
_EDGES = [
    ("Ez_X0_Y0", sb.update_symmetry_boundary_electric_Ez_X0_Y0,
     lambda k: (0, 0, k), range(nz), "Ez",
     lambda H, k: _gx0_ez(H, 0, k), lambda H, k: _gy0_ez(H, 0, k)),
    ("Ez_X0_YMax", sb.update_symmetry_boundary_electric_Ez_X0_YMax,
     lambda k: (0, ny, k), range(nz), "Ez",
     lambda H, k: _gx0_ez(H, ny, k), lambda H, k: _gymax_ez(H, 0, k)),
    ("Ez_XMax_Y0", sb.update_symmetry_boundary_electric_Ez_XMax_Y0,
     lambda k: (nx, 0, k), range(nz), "Ez",
     lambda H, k: _gxmax_ez(H, 0, k), lambda H, k: _gy0_ez(H, nx, k)),
    ("Ez_XMax_YMax", sb.update_symmetry_boundary_electric_Ez_XMax_YMax,
     lambda k: (nx, ny, k), range(nz), "Ez",
     lambda H, k: _gxmax_ez(H, ny, k), lambda H, k: _gymax_ez(H, nx, k)),
    ("Ey_X0_Z0", sb.update_symmetry_boundary_electric_Ey_X0_Z0,
     lambda j: (0, j, 0), range(ny), "Ey",
     lambda H, j: _gx0_ey(H, j, 0), lambda H, j: _gz0_ey(H, 0, j)),
    ("Ey_X0_ZMax", sb.update_symmetry_boundary_electric_Ey_X0_ZMax,
     lambda j: (0, j, nz), range(ny), "Ey",
     lambda H, j: _gx0_ey(H, j, nz), lambda H, j: _gzmax_ey(H, 0, j)),
    ("Ey_XMax_Z0", sb.update_symmetry_boundary_electric_Ey_XMax_Z0,
     lambda j: (nx, j, 0), range(ny), "Ey",
     lambda H, j: _gxmax_ey(H, j, 0), lambda H, j: _gz0_ey(H, nx, j)),
    ("Ey_XMax_ZMax", sb.update_symmetry_boundary_electric_Ey_XMax_ZMax,
     lambda j: (nx, j, nz), range(ny), "Ey",
     lambda H, j: _gxmax_ey(H, j, nz), lambda H, j: _gzmax_ey(H, nx, j)),
    ("Ex_Y0_Z0", sb.update_symmetry_boundary_electric_Ex_Y0_Z0,
     lambda i: (i, 0, 0), range(nx), "Ex",
     lambda H, i: _gy0_ex(H, i, 0), lambda H, i: _gz0_ex(H, i, 0)),
    ("Ex_Y0_ZMax", sb.update_symmetry_boundary_electric_Ex_Y0_ZMax,
     lambda i: (i, 0, nz), range(nx), "Ex",
     lambda H, i: _gy0_ex(H, i, nz), lambda H, i: _gzmax_ex(H, i, 0)),
    ("Ex_YMax_Z0", sb.update_symmetry_boundary_electric_Ex_YMax_Z0,
     lambda i: (i, ny, 0), range(nx), "Ex",
     lambda H, i: _gymax_ex(H, i, 0), lambda H, i: _gz0_ex(H, i, ny)),
    ("Ex_YMax_ZMax", sb.update_symmetry_boundary_electric_Ex_YMax_ZMax,
     lambda i: (i, ny, nz), range(nx), "Ex",
     lambda H, i: _gymax_ex(H, i, nz), lambda H, i: _gzmax_ex(H, i, ny)),
]


@pytest.mark.parametrize("edge", _EDGES, ids=[e[0] for e in _EDGES])
@pytest.mark.parametrize("a_pmc,b_pmc", [(True, False), (False, True), (True, True), (False, False)])
def test_edge_matches_independent_hand_derivation(edge, a_pmc, b_pmc):
    name, func, pos_fn, free_range, comp, ghost_a, ghost_b = edge
    ID, C, E, H = _make_arrays()
    comp_arr = E[comp]
    h1_name, h2_name = _CALL_H_ORDER[comp]

    expected = comp_arr.copy()
    for t in free_range:
        pos = pos_fn(t)
        if a_pmc or b_pmc:
            expected[pos] = ca * comp_arr[pos]
        if a_pmc:
            expected[pos] = expected[pos] + ghost_a(H, t)
        if b_pmc:
            expected[pos] = expected[pos] + ghost_b(H, t)

    func(nx, ny, nz, 1, a_pmc, b_pmc, C, ID, comp_arr, H[h1_name], H[h2_name])

    assert np.allclose(comp_arr, expected), f"{name} a_pmc={a_pmc} b_pmc={b_pmc} mismatch"


@pytest.mark.parametrize("edge", _EDGES, ids=[e[0] for e in _EDGES])
def test_edge_neither_pmc_leaves_field_untouched(edge):
    """Sanity check: with both flags False, nothing should change at all -
    matches the real pipeline, which never calls an edge function unless at
    least one of its two faces is actually a declared PMC boundary."""
    name, func, pos_fn, free_range, comp, ghost_a, ghost_b = edge
    ID, C, E, H = _make_arrays()
    comp_arr = E[comp]
    h1_name, h2_name = _CALL_H_ORDER[comp]
    before = comp_arr.copy()

    func(nx, ny, nz, 1, False, False, C, ID, comp_arr, H[h1_name], H[h2_name])

    assert np.array_equal(comp_arr, before)
