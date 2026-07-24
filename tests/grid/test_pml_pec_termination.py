"""Tests for FDTDGrid._terminate_pmls_with_pec - a PML is always implicitly
backed by a PEC wall at the domain's outer edge (the existing non-update of
tangential E there already gives the correct physics); this makes that
explicit in the ID array, purely for bookkeeping/inspection (e.g. fine
geometry views), the same way a `#symmetry_boundary ... pec` face's ID is
forced. A face with neither an active PML nor a symmetry boundary is
deliberately left unforced, to leave room for a future non-PEC termination
there.
"""
import numpy as np

import gprMax
import gprMax.model as model_mod

FACES = ("x0", "y0", "z0", "xmax", "ymax", "zmax")

# (component index, tangential-component name) pairs per face, matching
# FDTDGrid._force_pec_tangential_e's own face -> tangential-component map.
TANGENTIAL = {
    "x0": (1, 2),
    "xmax": (1, 2),
    "y0": (0, 2),
    "ymax": (0, 2),
    "z0": (0, 1),
    "zmax": (0, 1),
}
NORMAL = {"x0": 0, "xmax": 0, "y0": 1, "ymax": 1, "z0": 2, "zmax": 2}


def _capture_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _scene():
    # Domain sized to 30 cells/axis (dl=1e-3) so the default 10-cell PML
    # comfortably fits (2*10 < 30) on every face - this file specifically
    # exercises real PML-driven PEC termination, so the domain must be
    # large enough for genuine PML physics rather than disabling PML.
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


def _face_slice(grid, comp_idx, face):
    """Exactly mirrors the valid-range slices FDTDGrid._force_pec_tangential_e
    writes (component's own dependency-axis range only, not the padded
    array's full shape - e.g. Ey's j only spans 0:ny, not 0:ny+1)."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    slices = {
        ("x0", 1): grid.ID[1, 0, 0:ny, 0 : nz + 1],
        ("x0", 2): grid.ID[2, 0, 0 : ny + 1, 0:nz],
        ("xmax", 1): grid.ID[1, nx, 0:ny, 0 : nz + 1],
        ("xmax", 2): grid.ID[2, nx, 0 : ny + 1, 0:nz],
        ("y0", 0): grid.ID[0, 0:nx, 0, 0 : nz + 1],
        ("y0", 2): grid.ID[2, 0 : nx + 1, 0, 0:nz],
        ("ymax", 0): grid.ID[0, 0:nx, ny, 0 : nz + 1],
        ("ymax", 2): grid.ID[2, 0 : nx + 1, ny, 0:nz],
        ("z0", 0): grid.ID[0, 0:nx, 0 : ny + 1, 0],
        ("z0", 1): grid.ID[1, 0 : nx + 1, 0:ny, 0],
        ("zmax", 0): grid.ID[0, 0:nx, 0 : ny + 1, nz],
        ("zmax", 1): grid.ID[1, 0 : nx + 1, 0:ny, nz],
        # Normal component at (or nearest to, for an upper wall - E has no
        # own-axis position exactly on an upper wall) this face - must never
        # be forced by this face's own mechanism. Restricted to the interior
        # of the perpendicular axes (excluding this plane's own edges, which
        # are legitimately tangential-forced by the *other* four faces they
        # border) so this check isn't confounded by that separate, correct
        # forcing.
        ("x0", 0): grid.ID[0, 0, 1:ny, 1:nz],
        ("xmax", 0): grid.ID[0, nx - 1, 1:ny, 1:nz],
        ("y0", 1): grid.ID[1, 1:nx, 0, 1:nz],
        ("ymax", 1): grid.ID[1, 1:nx, ny - 1, 1:nz],
        ("z0", 2): grid.ID[2, 1:nx, 1:ny, 0],
        ("zmax", 2): grid.ID[2, 1:nx, 1:ny, nz - 1],
    }
    return slices[(face, comp_idx)]


def test_default_pml_on_all_faces_forces_tangential_e_to_pec(monkeypatch, tmp_path):
    """Default domain (PML active on all 6 faces, no symmetry boundaries):
    every face's two tangential E components must be forced to pec across
    their whole plane; the normal component must be untouched."""
    scene = _scene()

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    pec_numid = next(m.numID for m in grid.materials if m.ID == "pec")

    for face in FACES:
        for comp_idx in TANGENTIAL[face]:
            assert np.all(_face_slice(grid, comp_idx, face) == pec_numid), face
        assert not np.any(_face_slice(grid, NORMAL[face], face) == pec_numid), face


def test_no_pml_no_symmetry_boundary_leaves_face_unforced(monkeypatch, tmp_path):
    """PML disabled everywhere, no symmetry boundary declared: the physics
    at the domain wall is still silently PEC-like (unchanged, established
    elsewhere), but the ID array must NOT be explicitly forced to pec -
    leaving room for a future non-PEC termination there."""
    scene = _scene()
    scene.add(gprMax.PMLThickness(thickness=0))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    pec_numid = next(m.numID for m in grid.materials if m.ID == "pec")

    for face in FACES:
        for comp_idx in TANGENTIAL[face]:
            assert not np.any(_face_slice(grid, comp_idx, face) == pec_numid), face


def test_pml_termination_coexists_with_symmetry_boundary_on_other_faces(monkeypatch, tmp_path):
    """A pec/pmc symmetry boundary on some faces must not stop the other,
    still-PML-active faces from getting their own independent PEC
    termination."""
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pec"))
    scene.add(gprMax.SymmetryBoundary(face="ymax", type="pmc"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    pec_numid = next(m.numID for m in grid.materials if m.ID == "pec")

    # x0 has its own explicit pec symmetry boundary (already covered by
    # test_symmetry_boundary.py); ymax is pmc, so untouched. The remaining
    # faces (y0, z0, xmax, zmax) still have their default active PML and
    # must be independently PEC-terminated by this new mechanism.
    for face in ("y0", "z0", "xmax", "zmax"):
        for comp_idx in TANGENTIAL[face]:
            assert np.all(_face_slice(grid, comp_idx, face) == pec_numid), face

    # ymax (pmc, PML disabled there) must still have no ID forcing of its
    # own - checked on the interior only, since its own face slice's edges
    # are legitimately forced by the bordering x0 (pec symmetry boundary)
    # and z0/zmax (still-PML-active) faces, not by ymax itself.
    nx, nz = grid.nx, grid.nz
    ex_ymax_interior = grid.ID[0, 1:nx, grid.ny, 1:nz]
    ez_ymax_interior = grid.ID[2, 1:nx, grid.ny, 1:nz]
    assert not np.any(ex_ymax_interior == pec_numid)
    assert not np.any(ez_ymax_interior == pec_numid)
