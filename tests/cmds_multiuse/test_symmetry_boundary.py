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

"""Tests for the #symmetry_boundary command / SymmetryBoundary class - PEC
and PMC symmetry-plane boundaries that replace the PML on a domain face.

Covers the command-structure/registration parts of the feature (face
registration, PML thickness disabling, PEC-ID forcing at build time), and
confirms all three material regimes (non-dispersive, Debye, Lorentz/Drude)
are accepted at a PMC face without error - the per-iteration PMC ghost-node
field update itself is tested separately under
tests/materials/test_symmetry_boundary_pmc_*.py and
tests/updates/test_symmetry_boundary_solve_loop.py.
"""
from pathlib import Path

import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod

INF = float("inf")


def _capture_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _scene(dl=1e-3, pml_thickness=10):
    # Domain is sized to 25 cells/axis (not 10) so that the default PML
    # thickness (10 cells/side) fits without tripping FDTDGrid's
    # "PML has too many cells for the domain size" check on the faces
    # that are NOT under test (i.e. that keep their default thickness) -
    # see test_pec_face_registers_and_disables_pml, which explicitly
    # asserts those faces stay at the default 10.
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.025, 0.025, 0.025)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    if pml_thickness != 10:
        scene.add(gprMax.PMLThickness(thickness=pml_thickness))
    return scene


def test_pec_face_registers_and_disables_pml(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pec"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    assert grid.symmetry_boundaries == {"x0": "pec"}
    assert grid.pmls["thickness"]["x0"] == 0
    # Other faces keep their default PML thickness.
    for face in ("y0", "z0", "xmax", "ymax", "zmax"):
        assert grid.pmls["thickness"][face] == 10


def test_pmc_face_registers_and_disables_pml_but_no_id_changes(monkeypatch, tmp_path):
    """PML disabled on every other face so this test isolates the pmc face's
    own (lack of) ID forcing from the separate PML-termination-with-pec
    behaviour, which would otherwise also force pec at the other, still-PML
    -active faces this slice's border touches."""
    scene = _scene(pml_thickness=0)
    scene.add(gprMax.SymmetryBoundary(face="ymax", type="pmc"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    assert grid.symmetry_boundaries == {"ymax": "pmc"}
    assert grid.pmls["thickness"]["ymax"] == 0

    pec_numid = next(m.numID for m in grid.materials if m.ID == "pec")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    ex_ymax = grid.ID[0, 0:nx, ny, 0 : nz + 1]
    ez_ymax = grid.ID[2, 0 : nx + 1, ny, 0:nz]
    assert not np.any(ex_ymax == pec_numid)
    assert not np.any(ez_ymax == pec_numid)


def test_pec_face_forces_tangential_e_ids_to_pec(monkeypatch, tmp_path):
    """x0 is x-normal, so its tangential components are Ey and Ez - both
    should be forced to pec across the full face; Ex (normal to the face)
    must be untouched, and the adjacent x=1 plane must be untouched. PML
    disabled on every other face so this test isolates the symmetry
    boundary's own ID forcing from the separate PML-termination-with-pec
    behaviour, which would otherwise also force pec at the other, still-PML
    -active faces these slices' borders touch."""
    scene = _scene(pml_thickness=0)
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pec"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    pec_numid = next(m.numID for m in grid.materials if m.ID == "pec")
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    ey_x0 = grid.ID[1, 0, 0:ny, 0 : nz + 1]
    ez_x0 = grid.ID[2, 0, 0 : ny + 1, 0:nz]
    assert np.all(ey_x0 == pec_numid)
    assert np.all(ez_x0 == pec_numid)

    ex_x0 = grid.ID[0, 0, 0 : ny + 1, 0 : nz + 1]
    assert not np.any(ex_x0 == pec_numid)

    ey_x1 = grid.ID[1, 1, 0:ny, 0 : nz + 1]
    assert not np.any(ey_x1 == pec_numid)


def test_pec_and_pmc_sharing_an_edge_forces_shared_e_component_pec(monkeypatch, tmp_path):
    """x0 (pec) and ymax (pmc) share the edge x=0,y=ny. Ez is tangential to
    both faces, so the pec face's own forcing must reach that shared edge -
    this is the safeguard the feature is specifically for (a PMC ghost-node
    update must never see a live value where E has to stay exactly zero).
    PML disabled on every other face so this test isolates the symmetry
    boundaries' own ID forcing from the separate PML-termination-with-pec
    behaviour, which would otherwise also force pec at the other, still-PML
    -active faces this slice's border touches."""
    scene = _scene(pml_thickness=0)
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pec"))
    scene.add(gprMax.SymmetryBoundary(face="ymax", type="pmc"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    pec_numid = next(m.numID for m in grid.materials if m.ID == "pec")
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    ez_ymax = grid.ID[2, 0 : nx + 1, ny, 0:nz]
    pec_mask = ez_ymax == pec_numid
    # Only the i=0 shared-edge line is pec, not the rest of the ymax face.
    assert np.all(np.where(pec_mask)[0] == 0)
    assert pec_mask.sum() == nz


def test_duplicate_face_declaration_rejected(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pec"))
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))

    with pytest.raises(ValueError):
        gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)


def test_invalid_face_rejected(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="w0", type="pec"))

    with pytest.raises(ValueError):
        gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)


def test_invalid_type_rejected(monkeypatch, tmp_path):
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="foo"))

    with pytest.raises(ValueError):
        gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)


def test_rejected_in_2d_mode(monkeypatch, tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pec"))

    with pytest.raises(ValueError):
        gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)


def test_lorentz_material_at_pmc_face_is_allowed(monkeypatch, tmp_path):
    """Lorentz/Drude (complex-pole) dispersive materials ARE supported by
    the per-iteration PMC ghost-node update - see
    gprMax/cython/symmetry_boundaries_dispersive_complex.pyx. An earlier
    version of this feature rejected this combination with a build-time
    guard (Stage 1 of the PMC-dispersive work); Stage 2 added the
    complex-pole Cython path and lifted it."""
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="metal"))
    scene.add(
        gprMax.AddLorentzDispersion(
            poles=1, er_delta=[2.0], omega=[1e9], delta=[1e8], material_ids=["metal"]
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    assert captured["grid"].symmetry_boundaries == {"x0": "pmc"}


def test_debye_material_at_pmc_face_is_allowed(monkeypatch, tmp_path):
    """Debye (real-pole) dispersive materials are also supported at a PMC
    face, via the separate real-pole Cython path."""
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="soil"))
    scene.add(gprMax.AddDebyeDispersion(poles=1, er_delta=[2.0], tau=[1e-10], material_ids=["soil"]))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    assert captured["grid"].symmetry_boundaries == {"x0": "pmc"}


def test_lorentz_material_with_pec_symmetry_boundary_is_allowed(monkeypatch, tmp_path):
    """PEC symmetry boundaries need no per-iteration code at all (dispersive
    or otherwise), so a Lorentz/Drude material present elsewhere in the
    model must not be rejected there either."""
    scene = _scene()
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pec"))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="metal"))
    scene.add(
        gprMax.AddLorentzDispersion(
            poles=1, er_delta=[2.0], omega=[1e9], delta=[1e8], material_ids=["metal"]
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    assert captured["grid"].symmetry_boundaries == {"x0": "pec"}


def test_text_command_parses_correctly(monkeypatch, tmp_path: Path):
    """Exercises the hash_cmds_multiuse.py text-parsing path specifically."""
    infile = tmp_path / "symmetry_boundary.in"
    infile.write_text(
        "#title: symmetry boundary text parsing\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.025 0.025 0.025\n"
        "#time_window: 1e-12\n"
        "#symmetry_boundary: x0 pec\n"
        "#symmetry_boundary: ymax pmc\n"
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(inputfile=str(infile), n=1, geometry_only=True, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]

    assert grid.symmetry_boundaries == {"x0": "pec", "ymax": "pmc"}
    assert grid.pmls["thickness"]["x0"] == 0
    assert grid.pmls["thickness"]["ymax"] == 0
