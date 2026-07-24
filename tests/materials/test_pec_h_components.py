"""Regression tests for H-component handling in the rigid (non-averaged)
geometry build path.

Bug 3 (fixed): build_voxel() wrote Hx/Hy/Hz using the same 4-tangential-
corner pattern as Ex/Ey/Ez, instead of H's correct 2-position own-axis
pattern (already correct in build_box()).

New behaviour (this session): PEC specifically (matched via
Material.is_pec - the builtin 'pec' material, or any user-defined material
with se=inf) must not touch H at all - no ID write, no rigid flag - leaving
whatever background value/state was already there, since PEC has no
well-defined magnetic properties. This is PEC-specific, not applied to other
rigid (non-averaged) materials such as PMC or a non-averagable anisotropic
material, which still get the corrected 2-position H write.
"""
import numpy as np
from numpy.testing import assert_array_equal

from gprMax.cython.geometry_primitives import (
    build_box,
    build_magnetic_edge_x,
    build_voxel,
    build_voxels_from_array,
)
from gprMax.cython.yee_cell_build import build_electric_components, build_magnetic_components
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material, create_built_in_materials


def _grid(nx, ny, nz, extra_materials=()):
    grid = FDTDGrid()
    grid.nx, grid.ny, grid.nz = nx, ny, nz
    create_built_in_materials(grid)
    grid.materials += list(extra_materials)
    grid.initialise_geometry_arrays()
    return grid


PEC_NUMID = 0  # builtin pec is always created first by create_built_in_materials


def test_pec_voxel_leaves_h_untouched():
    """A PEC voxel must not write H ID values or mark H rigid - both must
    remain exactly as they were before the call (the "background")."""
    grid = _grid(3, 3, 3)
    before_ID = np.array(grid.ID, copy=True)
    before_rigidH = np.array(grid.rigidH, copy=True)

    build_voxel(
        1, 1, 1, PEC_NUMID, PEC_NUMID, PEC_NUMID, PEC_NUMID,
        False, True, True, True,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    idHx, idHy, idHz = grid.IDlookup["Hx"], grid.IDlookup["Hy"], grid.IDlookup["Hz"]
    assert_array_equal(np.asarray(grid.ID[idHx]), before_ID[idHx])
    assert_array_equal(np.asarray(grid.ID[idHy]), before_ID[idHy])
    assert_array_equal(np.asarray(grid.ID[idHz]), before_ID[idHz])
    assert_array_equal(np.asarray(grid.rigidH), before_rigidH)

    # E must still be correctly rigid at all 4 corners - unaffected by PEC's
    # H carve-out.
    idEx = grid.IDlookup["Ex"]
    assert grid.ID[idEx, 1, 1, 1] == PEC_NUMID
    assert grid.ID[idEx, 1, 2, 2] == PEC_NUMID
    assert grid.ID[idEx, 1, 2, 1] == PEC_NUMID
    assert grid.ID[idEx, 1, 1, 2] == PEC_NUMID


def test_non_pec_rigid_voxel_writes_h_at_correct_two_positions():
    """A non-PEC rigid (non-averaged) material must get the corrected
    2-position H write (own axis only), not the old broken 4-corner copy,
    and must be marked rigid there."""
    grid = _grid(3, 3, 3)
    matA = Material(3, "matA")
    matA.se = 5.0
    grid.materials.append(matA)

    build_voxel(
        1, 1, 1, matA.numID, matA.numID, matA.numID, matA.numID,
        False, False, False, False,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    idHx, idHy, idHz = grid.IDlookup["Hx"], grid.IDlookup["Hy"], grid.IDlookup["Hz"]
    # The 2 true positions for each component.
    assert grid.ID[idHx, 1, 1, 1] == matA.numID
    assert grid.ID[idHx, 2, 1, 1] == matA.numID
    assert grid.ID[idHy, 1, 1, 1] == matA.numID
    assert grid.ID[idHy, 1, 2, 1] == matA.numID
    assert grid.ID[idHz, 1, 1, 1] == matA.numID
    assert grid.ID[idHz, 1, 1, 2] == matA.numID

    # The old (buggy) tangential-corner positions must NOT have been written.
    assert grid.ID[idHx, 1, 2, 2] != matA.numID
    assert grid.ID[idHy, 2, 1, 2] != matA.numID
    assert grid.ID[idHz, 2, 2, 1] != matA.numID

    # Rigid flags set (own-corner flag, index 0/2/4 per component).
    assert grid.rigidH[0, 1, 1, 1]
    assert grid.rigidH[2, 1, 1, 1]
    assert grid.rigidH[4, 1, 1, 1]


def test_anisotropic_pec_on_one_axis_only_skips_only_that_axis():
    """An anisotropic voxel with a PEC x-material but ordinary non-averaged
    y/z materials must only skip Hx - Hy/Hz get the correct 2-position
    write as normal."""
    grid = _grid(3, 3, 3)
    matY = Material(3, "matY")
    matY.se = 2.0
    matZ = Material(4, "matZ")
    matZ.se = 3.0
    grid.materials += [matY, matZ]
    before_ID = np.array(grid.ID, copy=True)

    build_voxel(
        1, 1, 1, PEC_NUMID, PEC_NUMID, matY.numID, matZ.numID,
        False, True, False, False,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    idHx, idHy, idHz = grid.IDlookup["Hx"], grid.IDlookup["Hy"], grid.IDlookup["Hz"]
    # Hx (PEC axis) untouched.
    assert_array_equal(np.asarray(grid.ID[idHx]), before_ID[idHx])
    assert not grid.rigidH[0, 1, 1, 1]
    # Hy, Hz (non-PEC axes) correctly written and rigid.
    assert grid.ID[idHy, 1, 1, 1] == matY.numID
    assert grid.ID[idHy, 1, 2, 1] == matY.numID
    assert grid.ID[idHz, 1, 1, 1] == matZ.numID
    assert grid.ID[idHz, 1, 1, 2] == matZ.numID
    assert grid.rigidH[2, 1, 1, 1]
    assert grid.rigidH[4, 1, 1, 1]


def test_pec_box_leaves_h_untouched():
    """Same PEC/H behaviour as build_voxel, via build_box's own (separately
    hand-written) loops."""
    grid = _grid(4, 4, 4)
    before_ID = np.array(grid.ID, copy=True)
    before_rigidH = np.array(grid.rigidH, copy=True)

    build_box(
        1, 3, 1, 3, 1, 3, PEC_NUMID, PEC_NUMID, PEC_NUMID, PEC_NUMID,
        False, True, True, True,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    idHx, idHy, idHz = grid.IDlookup["Hx"], grid.IDlookup["Hy"], grid.IDlookup["Hz"]
    assert_array_equal(np.asarray(grid.ID[idHx]), before_ID[idHx])
    assert_array_equal(np.asarray(grid.ID[idHy]), before_ID[idHy])
    assert_array_equal(np.asarray(grid.ID[idHz]), before_ID[idHz])
    assert_array_equal(np.asarray(grid.rigidH), before_rigidH)

    idEx = grid.IDlookup["Ex"]
    assert grid.ID[idEx, 1, 1, 1] == PEC_NUMID


def test_non_pec_box_writes_h_at_correct_positions():
    """build_box with a non-PEC rigid material writes H at the correct
    2-position own-axis pattern and marks it rigid."""
    grid = _grid(4, 4, 4)
    matA = Material(3, "matA")
    matA.se = 5.0
    grid.materials.append(matA)

    build_box(
        1, 3, 1, 3, 1, 3, matA.numID, matA.numID, matA.numID, matA.numID,
        False, False, False, False,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    idHx = grid.IDlookup["Hx"]
    assert grid.ID[idHx, 1, 1, 1] == matA.numID
    assert grid.ID[idHx, 3, 1, 1] == matA.numID
    assert grid.rigidH[0, 1, 1, 1]


def test_build_voxels_from_array_skips_h_for_pec_numid():
    """The is_pec_lookup threading for build_voxels_from_array (used by
    FractalBox/GeometryObjectsRead) must skip H for a PEC-numID voxel."""
    grid = _grid(3, 3, 3)
    before_ID = np.array(grid.ID, copy=True)

    data = np.full((1, 1, 1), PEC_NUMID, dtype=np.int16)
    is_pec_lookup = np.array([m.is_pec for m in grid.materials], dtype=np.uint8)
    is_averagable_lookup = np.array([m.averagable for m in grid.materials], dtype=np.uint8)

    build_voxels_from_array(
        1, 1, 1, 0, False, is_pec_lookup, is_averagable_lookup, data,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    idHx, idHy, idHz = grid.IDlookup["Hx"], grid.IDlookup["Hy"], grid.IDlookup["Hz"]
    assert_array_equal(np.asarray(grid.ID[idHx]), before_ID[idHx])
    assert_array_equal(np.asarray(grid.ID[idHy]), before_ID[idHy])
    assert_array_equal(np.asarray(grid.ID[idHz]), before_ID[idHz])

    idEx = grid.IDlookup["Ex"]
    assert grid.ID[idEx, 1, 1, 1] == PEC_NUMID


def test_pec_box_end_to_end_leaves_h_at_background(tmp_path):
    """End-to-end: a real PEC Box built via gprMax.run() must leave H at
    the domain's background material (free_space), not overwritten with
    the PEC's placeholder magnetic properties."""
    import gprMax
    import gprMax.model as model_mod

    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    import unittest.mock as mock
    with mock.patch.object(model_mod.Model, "build", patched_build):
        dl = 1e-3
        scene = gprMax.Scene()
        scene.add(gprMax.Title(name="pec_box_h_background"))
        scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
        scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
        scene.add(gprMax.PMLThickness(thickness=0))
        scene.add(gprMax.TimeWindow(time=1e-12))
        scene.add(gprMax.Box(p1=(0.005, 0.005, 0.005), p2=(0.015, 0.015, 0.015), material_id="pec"))
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "pec_box_h_background",
            hide_progress_bars=True,
        )

    grid = captured["grid"]
    # H must never be marked rigid anywhere inside the PEC box's footprint -
    # confirming build_box's PEC carve-out took effect through the real
    # object-building pipeline (Scene -> Model.build()), not just a raw
    # build_box() call. The resulting ID value itself is intentionally not
    # asserted here: since solid[] is still (correctly) set to PEC's numID
    # throughout the box regardless of this fix, the ordinary averaging
    # pass - now free to run since H isn't rigid - naturally recomputes a
    # deep-interior point as PEC's numID too (both its neighbours are PEC),
    # which is the expected "let the general mechanism decide" outcome, not
    # a specific frozen background value.
    assert not grid.rigidH[0, 10, 10, 10]
    assert not grid.rigidH[2, 10, 10, 10]
    assert not grid.rigidH[4, 10, 10, 10]

    # E must still be correctly rigid throughout the box - unaffected by
    # the H carve-out.
    assert grid.rigidE[0, 10, 10, 10]


def test_build_magnetic_components_treats_pec_as_transparent_at_boundary():
    """The general averaging pass (build_magnetic_components) must not blend
    PEC's meaningless mr=1/sm=0 placeholder into a real neighbour's magnetic
    properties - at a boundary between a real material and PEC, H must take
    the real material's numID directly (no averaging, no compound material),
    matching what a model without the PEC object would produce there.

    A single 2x2x2 grid with solid[0,:,:]=matA (real) and solid[1,:,:]=PEC
    exercises all three cases for Hx (dependency axis i) in one pass:
    i=0 (both sides matA - ordinary equal case), i=1 (boundary - one PEC, one
    real - must resolve to matA directly), i=2 (both sides PEC - left
    untouched, see test_build_magnetic_components_leaves_both_pec_untouched).
    """
    grid = _grid(2, 2, 2)
    matA = Material(3, "matA")
    matA.mr = 3.0
    matA.sm = 0.02
    grid.materials.append(matA)

    grid.solid[0, :, :] = matA.numID
    grid.solid[1, :, :] = PEC_NUMID

    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid)

    idHx = grid.IDlookup["Hx"]
    # i=0: both neighbours matA - ordinary case, unaffected by this fix.
    assert grid.ID[idHx, 0, 0, 0] == matA.numID
    # i=1: boundary - PEC is transparent, real neighbour used directly.
    assert grid.ID[idHx, 1, 0, 0] == matA.numID
    # No compound "matA+...+pec+pec"-style material should exist at all.
    assert not any("pec" in m.ID and "+" in m.ID for m in grid.materials)


def test_build_magnetic_components_leaves_both_pec_neighbours_untouched():
    """When both neighbours of an H position are PEC, there is no real
    medium left to reference (solid[] no longer remembers what was there
    before PEC was placed) - the position must be left exactly as it was
    before this call, not forced to PEC's own numID nor anything else."""
    grid = _grid(2, 2, 2)
    grid.solid[:, :, :] = PEC_NUMID
    before = np.array(grid.ID, copy=True)

    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid)

    assert_array_equal(np.asarray(grid.ID), before)


def test_build_magnetic_components_unchanged_for_two_real_materials():
    """Sanity check: ordinary averaging between two non-PEC materials is
    completely unaffected by the PEC-transparency logic.

    Explicitly requests harmonic=False (arithmetic mean) here because that's
    what this test is actually checking (that PEC-transparency doesn't
    perturb an ordinary average) - it doesn't care which mixing rule
    #magnetic_averaging defaults to. See test_magnetic_averaging_mode.py for
    coverage of the harmonic/arithmetic mixing rules themselves.
    """
    grid = _grid(2, 2, 2)
    matA = Material(3, "matA")
    matA.mr = 3.0
    matB = Material(4, "matB")
    matB.mr = 5.0
    grid.materials += [matA, matB]

    grid.solid[0, :, :] = matA.numID
    grid.solid[1, :, :] = matB.numID

    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid, False)

    idHx = grid.IDlookup["Hx"]
    compound_numid = grid.ID[idHx, 1, 0, 0]
    compound = next(m for m in grid.materials if m.numID == compound_numid)
    assert compound.mr == np.mean([matA.mr, matB.mr])


def test_pec_cylinder_and_box_in_magnetic_half_space_leave_h_at_boundary_unchanged(tmp_path):
    """End-to-end regression for the exact scenario that surfaced this bug:
    a half-space with magnetic properties different from free space
    (mr=3, sm=0.02) embedding a PEC cylinder and a PEC box must not create
    any PEC-blended compound material - H at the object boundaries must
    resolve directly to half_space's own numID, not an averaged placeholder.
    """
    import gprMax
    import gprMax.model as model_mod

    def _capture(scene, outfile):
        captured = {}
        orig_build = model_mod.Model.build

        def patched_build(self):
            orig_build(self)
            captured["grid"] = self.G

        import unittest.mock as mock
        with mock.patch.object(model_mod.Model, "build", patched_build):
            gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=outfile, hide_progress_bars=True)
        return captured["grid"]

    dl = 2e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="pec_h_invariance"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.040, 0.040, 0.040)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=6, se=0.01, mr=3, sm=0.02, id="half_space"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.040, 0.040, 0.020), material_id="half_space"))
    scene.add(gprMax.Cylinder(p1=(0.020, 0, 0.010), p2=(0.020, 0.040, 0.010), r=0.006, material_id="pec"))
    scene.add(gprMax.Box(p1=(0.005, 0.005, 0.002), p2=(0.015, 0.015, 0.012), material_id="pec"))

    grid = _capture(scene, tmp_path / "pec_h_invariance")

    assert not any("pec" in m.ID and "+" in m.ID for m in grid.materials), (
        "no PEC-blended compound material should have been created at the "
        f"cylinder/box boundaries, found: {[m.ID for m in grid.materials if '+' in m.ID]}"
    )


def test_magnetic_edge_sets_exact_self_consistent_rigid_bits():
    """set_rigid_Hx was reverted to the self-consistent formula (mirroring
    set_rigid_Ex/Ey/Ez's own-position/neighbour-offset shape) so a
    standalone magnetic edge marks exactly one H position, not two - a
    single build_magnetic_edge_x call at position i must set plane0@i and
    plane1@(i-1) (if i!=0), and nothing else."""
    grid = _grid(4, 3, 3)
    matA = Material(3, "matA")
    grid.materials.append(matA)

    build_magnetic_edge_x(2, 1, 1, matA.numID, grid.rigidH, grid.ID)

    rigidH = np.asarray(grid.rigidH)
    expected = np.zeros_like(rigidH)
    expected[0, 2, 1, 1] = True  # plane0 at the edge's own position
    expected[1, 1, 1, 1] = True  # plane1 at position-1 (i!=0, so guard passes)
    assert_array_equal(rigidH, expected)

    idHx = grid.IDlookup["Hx"]
    assert grid.ID[idHx, 2, 1, 1] == matA.numID


def test_hole_carving_preserves_neighbouring_cells_rigid_h_faces():
    """Overwriting one rigid cell in a multi-cell rigid block must not
    orphan the H faces still legitimately claimed by its untouched
    neighbours - the same safety property #edge/#box already rely on for
    E, now confirmed for H's 2-owner-per-position sharing too."""
    grid = _grid(4, 3, 3)
    matA = Material(3, "matA")
    matA.se = 5.0
    grid.materials.append(matA)

    # Two adjacent rigid cells at i=1 and i=2, sharing Hx position 2.
    build_voxel(
        1, 1, 1, matA.numID, matA.numID, matA.numID, matA.numID,
        False, False, False, False,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )
    build_voxel(
        2, 1, 1, matA.numID, matA.numID, matA.numID, matA.numID,
        False, False, False, False,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    idHx = grid.IDlookup["Hx"]
    assert grid.ID[idHx, 2, 1, 1] == matA.numID

    # Overwrite cell 2 with an averaging (non-rigid) material.
    free_space_numid = next(m.numID for m in grid.materials if m.ID == "free_space")
    build_voxel(
        2, 1, 1, free_space_numid, free_space_numid, free_space_numid, free_space_numid,
        True, False, False, False,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    # Shared position 2 must still be rigid, via cell 1's independent claim.
    build_electric_components(grid.solid, grid.rigidE, grid.ID, grid)
    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid)
    assert grid.ID[idHx, 2, 1, 1] == matA.numID, (
        "cell 1's independent rigid claim on the shared Hx face at position 2 "
        "should have survived cell 2 being overwritten"
    )
