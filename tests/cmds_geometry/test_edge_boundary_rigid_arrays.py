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

"""Regression tests for Edge/MagneticEdge writing outside the rigid arrays
(Codex-reported, "Need to make sure we can build the edges properly"):

Grid coordinates are allowed to equal the far domain boundary (e.g. an
x-directed #magnetic_edge at y == ny is a legitimate, physically
meaningful position - a wire running along the domain's far edge).
MagneticEdge.build() then calls build_magnetic_edge_x(i, ny, k, ...),
whose set_rigid_Hx() (gprMax/cython/yee_cell_setget_rigid.pyx) wrote
directly to rigidH[0, i, j, k] with NO bounds guard at all. rigidH has
purely cell-centred dimensions (nx, ny, nz) (gprMax/grid/fdtd_grid.py),
so j == ny is one past the last valid index. Cython is compiled with
boundscheck=False globally (setup.py), so this was not a clean exception
but an out-of-bounds memory WRITE - undefined behaviour, potentially
silent heap corruption rather than a crash.

The regular electric Edge has the identical problem via
set_rigid_Ex/Ey/Ez - except those already had a "j, k may run to
rigidE.shape[2]/[3]" comment and correct guards on the READ side
(get_rigid_Ex/Ey/Ez), just not on the WRITE side. The H-component read
functions (get_rigid_Hx/Hy/Hz) had the identical latent gap too (only
guarded their own axis, not the transverse ones) - unexploited by any
current caller (yee_cell_build.pyx's material-averaging loops always
iterate the transverse axes safely within [0, n)), but fixed for
symmetry with the now-fixed write path.

Fixed by adding upper-bound guards on the transverse axis/axes to all
six set_rigid_{E,H}{x,y,z} functions (and the three get_rigid_H{x,y,z}
functions), mirroring the guards get_rigid_Ex/Ey/Ez already had.

This file tests two things:
1. Unit-level, directly against the compiled Cython functions with
   minimally-sized (no slack/padding) arrays: a transverse coordinate at
   the far wall must not write anything (the guard fires), and ordinary
   interior coordinates must still behave exactly as before (no
   regression in normal geometry building).
2. End-to-end, via real Edge/#edge and MagneticEdge/#magnetic_edge
   commands positioned exactly on each of the 6 domain faces, for every
   run-axis/transverse-boundary combination - confirming the model
   builds without exception and the ID array reflects the intended
   material assignment.
"""
import numpy as np
import pytest

import gprMax
from gprMax.cython.geometry_primitives import (
    build_edge_x,
    build_edge_y,
    build_edge_z,
    build_magnetic_edge_x,
    build_magnetic_edge_y,
    build_magnetic_edge_z,
)

NX, NY, NZ = 5, 6, 7  # deliberately asymmetric to catch axis-mixups
SHAPE3 = (NX, NY, NZ)


def _make_arrays():
    rigidE = np.zeros((12, NX, NY, NZ), dtype=np.int8)
    rigidH = np.zeros((6, NX, NY, NZ), dtype=np.int8)
    ID = np.zeros((6, NX + 1, NY + 1, NZ + 1), dtype=np.uint32)
    return rigidE, rigidH, ID


def _expected_e_writes(own_idx, tr1_idx, tr2_idx, offsets, i, j, k, shape):
    """Reference re-implementation (independent of the Cython fix, but
    mirroring its exact guard logic) of which of set_rigid_E{x,y,z}'s 4
    write terms are valid for the fixed axis-role mapping of a specific
    component. own_idx/tr1_idx/tr2_idx are which of (i,j,k) plays the
    "own" (undecremented) vs the two transverse roles for this component
    (e.g. for Ex, own=i, transverse pair=(j,k)). `offsets` is the
    (own, tr1_decrement, tr2_decrement, both_decrement) component-index
    offsets - Ex/Ez use (0,1,3,2), but Ey's real (pre-existing, correct)
    layout swaps the two transverse terms' indices: (0,3,1,2).

    Returns a set of (component, i, j, k) tuples that must be True.
    """
    own_off, tr1_off, tr2_off, both_off = offsets
    coords = [i, j, k]
    own = coords[own_idx]
    t1 = coords[tr1_idx]
    t2 = coords[tr2_idx]
    s1 = shape[tr1_idx]
    s2 = shape[tr2_idx]

    def make(own_v, t1_v, t2_v):
        c = [0, 0, 0]
        c[own_idx] = own_v
        c[tr1_idx] = t1_v
        c[tr2_idx] = t2_v
        return tuple(c)

    expected = set()
    if t1 < s1 and t2 < s2:
        expected.add((own_off,) + make(own, t1, t2))
    if t1 != 0 and t2 < s2:
        expected.add((tr1_off,) + make(own, t1 - 1, t2))
    if t2 != 0 and t1 < s1:
        expected.add((tr2_off,) + make(own, t1, t2 - 1))
    if t1 != 0 and t2 != 0:
        expected.add((both_off,) + make(own, t1 - 1, t2 - 1))
    return expected


def _actual_true_positions(rigid):
    return {tuple(pos) for pos in np.argwhere(rigid != 0)}


# --- Electric edges (build_edge_x/y/z): own axis / transverse pair / component offsets ---
# offsets = (own, tr1_decrement, tr2_decrement, both_decrement) - see
# set_rigid_Ex/Ey/Ez in yee_cell_setget_rigid.pyx. Ex/Ez use (0,1,3,2);
# Ey's pre-existing (correct, unchanged) layout swaps the two transverse
# terms to (0,3,1,2) - see _expected_e_writes' docstring.
_E_CASES = [
    (build_edge_x, 0, 1, 2, (0, 1, 3, 2)),  # Ex: own=i, transverse=(j,k)
    (build_edge_y, 1, 0, 2, (4, 7, 5, 6)),  # Ey: own=j, transverse=(i,k)
    (build_edge_z, 2, 0, 1, (8, 9, 11, 10)),  # Ez: own=k, transverse=(i,j)
]


@pytest.mark.parametrize("builder,own_idx,tr1_idx,tr2_idx,offsets", _E_CASES)
@pytest.mark.parametrize("t1_at_max", [False, True])
@pytest.mark.parametrize("t2_at_max", [False, True])
def test_electric_edge_transverse_boundary_matches_reference(
    builder, own_idx, tr1_idx, tr2_idx, offsets, t1_at_max, t2_at_max
):
    """No exception/crash, and the resulting rigidE state EXACTLY matches
    an independent, guard-mirroring reference computation - not just
    "stays zero" (some neighbour-offset terms remain valid even when one
    transverse coordinate is at the far wall)."""
    rigidE, rigidH, ID = _make_arrays()
    coords = [1, 1, 1]
    coords[own_idx] = 1  # interior, safe
    coords[tr1_idx] = SHAPE3[tr1_idx] if t1_at_max else 1
    coords[tr2_idx] = SHAPE3[tr2_idx] if t2_at_max else 1
    i, j, k = coords

    builder(i, j, k, 9, rigidE, rigidH, ID)

    expected = _expected_e_writes(own_idx, tr1_idx, tr2_idx, offsets, i, j, k, SHAPE3)
    assert _actual_true_positions(rigidE) == expected


@pytest.mark.parametrize("builder", [build_edge_x, build_edge_y, build_edge_z])
def test_electric_edge_interior_position_still_sets_rigid_flag(builder):
    rigidE, rigidH, ID = _make_arrays()
    builder(1, 1, 1, 9, rigidE, rigidH, ID)
    assert np.any(rigidE != 0)


# --- Magnetic edges (build_magnetic_edge_x/y/z): own axis / two transverse axes ---
_H_CASES = [
    (build_magnetic_edge_x, 0, 1, 2, 0),  # Hx: own=i, transverse=(j,k), components 0-1
    (build_magnetic_edge_y, 1, 0, 2, 2),  # Hy: own=j, transverse=(i,k), components 2-3
    (build_magnetic_edge_z, 2, 0, 1, 4),  # Hz: own=k, transverse=(i,j), components 4-5
]


@pytest.mark.parametrize("builder,own_idx,tr1_idx,tr2_idx,comp_base", _H_CASES)
@pytest.mark.parametrize("t1_at_max", [False, True])
@pytest.mark.parametrize("t2_at_max", [False, True])
def test_magnetic_edge_transverse_boundary_matches_reference(
    builder, own_idx, tr1_idx, tr2_idx, comp_base, t1_at_max, t2_at_max
):
    """Unlike E components, H's transverse axes have no neighbour-offset
    term at all - both writes use the SAME raw transverse coordinates,
    so if EITHER is at the far wall, the guard must block BOTH writes
    entirely (there's no partially-valid case here, unlike E)."""
    rigidH_arrays_zero = np.zeros((6, NX, NY, NZ), dtype=np.int8)
    rigidE, rigidH, ID = _make_arrays()
    coords = [1, 1, 1]
    coords[own_idx] = 1
    coords[tr1_idx] = SHAPE3[tr1_idx] if t1_at_max else 1
    coords[tr2_idx] = SHAPE3[tr2_idx] if t2_at_max else 1
    i, j, k = coords

    builder(i, j, k, 9, rigidH, ID)

    if t1_at_max or t2_at_max:
        assert np.array_equal(rigidH, rigidH_arrays_zero)
    else:
        assert np.any(rigidH != 0)


@pytest.mark.parametrize(
    "builder,run_axis_size",
    [
        (build_magnetic_edge_x, NX),
        (build_magnetic_edge_y, NY),
        (build_magnetic_edge_z, NZ),
    ],
)
def test_magnetic_edge_interior_position_still_sets_rigid_flag(builder, run_axis_size):
    """No regression: an ordinary interior edge position must still set
    the expected rigid flag exactly as before."""
    rigidE, rigidH, ID = _make_arrays()
    builder(1, 1, 1, 9, rigidH, ID)
    assert np.any(rigidH != 0)


# --- End-to-end: real #edge / #magnetic_edge commands at each domain face ---


def _base_scene(domain=(0.02, 0.02, 0.02), dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=1, se=1, mr=1, sm=0, id="wire"))
    return scene


def _build_and_capture(monkeypatch, scene):
    import gprMax.model as model_mod

    captured = {}
    orig_build = model_mod.Model.build

    def patched(self):
        orig_build(self)
        captured["ID"] = self.G.ID.copy()
        captured["rigidE"] = self.G.rigidE.copy()
        captured["rigidH"] = self.G.rigidH.copy()
        captured["wire_numID"] = next(m.numID for m in self.G.materials if m.ID == "wire")

    monkeypatch.setattr(model_mod.Model, "build", patched)
    return captured


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("boundary", ["near", "far"])
def test_magnetic_edge_builds_at_every_transverse_boundary(monkeypatch, tmp_path, axis, boundary):
    dl = 1e-3
    domain = (0.02, 0.02, 0.02)
    n = 20  # cells per axis (0.02 / 1e-3)
    scene = _base_scene(domain, dl)

    # Run the edge along `axis`; place it at the near (0) or far (n) wall
    # on BOTH transverse axes simultaneously - the worst case, since it
    # exercises both fixed coordinates at the boundary at once.
    coord = 0.0 if boundary == "near" else n * dl
    p1 = [0.005, 0.005, 0.005]
    p2 = [0.005, 0.005, 0.005]
    run_index = "xyz".index(axis)
    p1[run_index] = 0.005
    p2[run_index] = 0.015
    for i in range(3):
        if i != run_index:
            p1[i] = coord
            p2[i] = coord

    scene.add(gprMax.MagneticEdge(p1=tuple(p1), p2=tuple(p2), material_id="wire"))

    captured = _build_and_capture(monkeypatch, scene)
    # Must not raise/crash - the previously vulnerable code path.
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True,
        outputfile=tmp_path / f"medge_{axis}_{boundary}", hide_progress_bars=True,
    )

    # ID (padded nx+1/ny+1/nz+1, so a boundary coordinate is always a
    # valid index there) must reflect the material assignment along the
    # whole run of the edge - confirms the edge was genuinely built at
    # the boundary, not silently skipped as "outside the grid".
    id_component = 3 + run_index  # Hx/Hy/Hz -> ID indices 3/4/5
    lo_cell, hi_cell = 5, 14  # 0.005m..0.015m at dl=1e-3
    fixed = n if boundary == "far" else 0
    id_slice = [slice(None)] * 3
    id_slice[run_index] = slice(lo_cell, hi_cell + 1)
    for i in range(3):
        if i != run_index:
            id_slice[i] = fixed
    assert np.all(captured["ID"][id_component][tuple(id_slice)] == captured["wire_numID"])

    # rigidH must never contain a flag at a transverse coordinate that
    # equals the array's own far-wall size (out of range for a purely
    # cell-centred (nx,ny,nz) array) - the guard must have prevented any
    # write there, for every one of the 6 magnetic-edge flag planes.
    assert captured["rigidH"].shape == (6, n, n, n)
    if boundary == "far":
        assert np.count_nonzero(captured["rigidH"]) == 0


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("boundary", ["near", "far"])
def test_electric_edge_builds_at_every_transverse_boundary(monkeypatch, tmp_path, axis, boundary):
    dl = 1e-3
    domain = (0.02, 0.02, 0.02)
    n = 20
    scene = _base_scene(domain, dl)

    coord = 0.0 if boundary == "near" else n * dl
    p1 = [0.005, 0.005, 0.005]
    p2 = [0.005, 0.005, 0.005]
    run_index = "xyz".index(axis)
    p1[run_index] = 0.005
    p2[run_index] = 0.015
    for i in range(3):
        if i != run_index:
            p1[i] = coord
            p2[i] = coord

    scene.add(gprMax.Edge(p1=tuple(p1), p2=tuple(p2), material_id="wire"))

    captured = _build_and_capture(monkeypatch, scene)
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True,
        outputfile=tmp_path / f"edge_{axis}_{boundary}", hide_progress_bars=True,
    )

    id_component = run_index  # Ex/Ey/Ez -> ID indices 0/1/2
    lo_cell, hi_cell = 5, 14
    fixed = n if boundary == "far" else 0
    id_slice = [slice(None)] * 3
    id_slice[run_index] = slice(lo_cell, hi_cell + 1)
    for i in range(3):
        if i != run_index:
            id_slice[i] = fixed
    assert np.all(captured["ID"][id_component][tuple(id_slice)] == captured["wire_numID"])

    # Unlike MagneticEdge/rigidH, rigidE's "both transverse coordinates
    # decremented" term (see test_electric_edge_transverse_boundary_
    # matches_reference) stays genuinely valid even when both transverse
    # coordinates are at the far wall, so rigidE is NOT expected to be
    # all-zero here - the array shape itself (no resize/corruption) and
    # the correct ID assignment above are what matter; the unit-level
    # reference test above already covers the exact expected pattern.
    assert captured["rigidE"].shape == (12, n, n, n)
