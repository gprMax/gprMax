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

"""Regression tests for #magnetic_averaging (harmonic vs. arithmetic mixing
of mu_r/sigma* at Yee-cell H-component boundaries).

Each H component (Hx, Hy, Hz) is averaged from the two neighbouring cells
stacked along the component's own axis - i.e. normal to any interface
between them. Normal B is continuous across a material interface, so the
harmonic mean of mu_r (and, for consistency, sigma*) is the physically
correct mixing rule there, unlike the tangential 4-cell average used for
E-field smoothing (arithmetic mean, unaffected by this command).

New default (this change): harmonic. 'arithmetic' is available via
#magnetic_averaging for byte-for-byte backwards compatibility with older
gprMax versions.

Implementation note this file guards against: Material.create_compound_id()
deliberately duplicates a 2-material call into a 4-part name ("A+A+B+B")
so a 2-cell magnetic average collides with (and reuses) a 4-cell electric
average of the same 2 materials - correct only because both used the same
(arithmetic) mixing rule before this change. Harmonic magnetic averages now
use a distinct "Hmag_" prefix to avoid silently reusing the electric
material's arithmetic-mean mu_r/sigma*. That prefix must never be or
contain ':' - hash_cmds_file.py's command-line parser does a bare
`line.split(":")` and keeps only cmd[1], so a second ':' anywhere in a
material ID (reachable via a #geometry_objects_write / #geometry_objects_read
round-trip, which writes material.ID verbatim into a #material: line)
silently truncates the name there.
"""
import numpy as np
import pytest

import gprMax
import gprMax.config as config
import gprMax.model as model_mod
from gprMax.cython.yee_cell_build import build_magnetic_components
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material, create_built_in_materials
from gprMax.user_objects.cmds_singleuse import MagneticAveraging


def _grid(nx, ny, nz, extra_materials=()):
    grid = FDTDGrid()
    grid.nx, grid.ny, grid.nz = nx, ny, nz
    create_built_in_materials(grid)
    grid.materials += list(extra_materials)
    grid.initialise_geometry_arrays()
    return grid


def _capture_built_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def test_default_mode_is_harmonic(tmp_path, monkeypatch):
    """No #magnetic_averaging command present - config must default to
    'harmonic', not the old 'arithmetic'."""
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["mode"] = config.get_model_config().magnetic_averaging_mode

    monkeypatch.setattr(model_mod.Model, "build", patched_build)

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "default_mode",
        geometry_only=True,
        hide_progress_bars=True,
    )

    assert captured["mode"] == "harmonic"


def test_magnetic_averaging_command_rejects_invalid_mode():
    grid = _grid(2, 2, 2)
    with pytest.raises(ValueError, match="harmonic.*arithmetic"):
        MagneticAveraging(mode="bogus").build(grid)


@pytest.mark.parametrize("mode", ["harmonic", "HARMONIC", "Arithmetic"])
def test_magnetic_averaging_command_accepts_case_insensitive_valid_modes(tmp_path, monkeypatch, mode):
    # Routed through a real gprMax.run() (like test_default_mode_is_harmonic)
    # rather than calling .build() on a bare grid: MagneticAveraging.build()
    # reads/writes config.get_model_config(), which only exists once a full
    # SimulationConfig has been created - i.e. inside an actual run, not for
    # a hand-built grid in isolation.
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["mode"] = config.get_model_config().magnetic_averaging_mode

    monkeypatch.setattr(model_mod.Model, "build", patched_build)

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-11))
    scene.add(gprMax.MagneticAveraging(mode=mode))
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / f"mode_{mode}",
        geometry_only=True,
        hide_progress_bars=True,
    )

    assert captured["mode"] == mode.lower()


def test_harmonic_mean_matches_hand_computed_value_including_zero_sigma_star():
    """mr=(1,4) -> harmonic mean 1.6 (not the arithmetic 2.5). sm=(0,0.4) ->
    harmonic mean 0, not a ZeroDivisionError/NaN - free_space (sm=0) next to
    any ordinary lossy magnetic material is the common case this must
    handle cleanly.
    """
    matA = Material(3, "matA")
    matA.mr, matA.sm = 1.0, 0.0
    matB = Material(4, "matB")
    matB.mr, matB.sm = 4.0, 0.4
    grid = _grid(2, 2, 2, extra_materials=[matA, matB])
    grid.solid[0, :, :] = matA.numID
    grid.solid[1, :, :] = matB.numID

    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid, True)

    idHx = grid.IDlookup["Hx"]
    numid = grid.ID[idHx, 1, 0, 0]
    averaged = next(m for m in grid.materials if m.numID == numid)
    assert averaged.mr == pytest.approx(1.6)
    assert averaged.sm == pytest.approx(0.0)
    assert ":" not in averaged.ID


def test_arithmetic_mean_still_available_and_matches_old_formula():
    matA = Material(3, "matA")
    matA.mr, matA.sm = 1.0, 0.0
    matB = Material(4, "matB")
    matB.mr, matB.sm = 4.0, 0.4
    grid = _grid(2, 2, 2, extra_materials=[matA, matB])
    grid.solid[0, :, :] = matA.numID
    grid.solid[1, :, :] = matB.numID

    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid, False)

    idHx = grid.IDlookup["Hx"]
    numid = grid.ID[idHx, 1, 0, 0]
    averaged = next(m for m in grid.materials if m.numID == numid)
    assert averaged.mr == np.mean([matA.mr, matB.mr])
    assert averaged.sm == np.mean([matA.sm, matB.sm])


def test_harmonic_and_arithmetic_use_independent_materials_not_stale_reuse():
    """Building the same 2-material boundary once in each mode must not
    let one mode's material leak into the other via the compound-ID
    lookup (the exact bug this change had to fix: harmonic silently
    reusing the arithmetic-mean electric-average material)."""
    matA = Material(3, "matA")
    matA.mr = 2.0
    matB = Material(4, "matB")
    matB.mr = 8.0
    grid = _grid(2, 2, 2, extra_materials=[matA, matB])
    grid.solid[0, :, :] = matA.numID
    grid.solid[1, :, :] = matB.numID

    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid, True)
    harmonic_numid = grid.ID[grid.IDlookup["Hx"], 1, 0, 0]
    harmonic_mat = next(m for m in grid.materials if m.numID == harmonic_numid)
    assert harmonic_mat.mr == pytest.approx(3.2)  # 2*2*8/10

    # Reset the Hx ID at this position and rebuild in arithmetic mode -
    # must produce (or reuse) a *different* material with the arithmetic value.
    grid.ID[grid.IDlookup["Hx"], 1, 0, 0] = 0
    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid, False)
    arithmetic_numid = grid.ID[grid.IDlookup["Hx"], 1, 0, 0]
    arithmetic_mat = next(m for m in grid.materials if m.numID == arithmetic_numid)
    assert arithmetic_mat.mr == pytest.approx(5.0)  # (2+8)/2
    assert arithmetic_mat.numID != harmonic_mat.numID


def test_end_to_end_default_harmonic_differs_from_explicit_arithmetic(tmp_path, monkeypatch):
    """A real Box boundary, built through the full gprMax.run() pipeline,
    must give different (correct, harmonic) mu_r by default and reproduce
    the old arithmetic value when #magnetic_averaging: arithmetic is set.
    """

    def _run(mode):
        captured = _capture_built_grid(monkeypatch)
        scene = gprMax.Scene()
        scene.add(gprMax.Discretisation(p1=(0.01, 0.01, 0.01)))
        scene.add(gprMax.Domain(p1=(0.1, 0.1, 0.1)))
        scene.add(gprMax.PMLThickness(thickness=0))
        scene.add(gprMax.TimeWindow(time=1e-11))
        if mode is not None:
            scene.add(gprMax.MagneticAveraging(mode=mode))
        scene.add(gprMax.Material(er=1, se=0, mr=4, sm=0.4, id="mag"))
        scene.add(gprMax.Box(p1=(0.0, 0.0, 0.0), p2=(0.05, 0.1, 0.1), material_id="mag"))
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=tmp_path / f"end_to_end_{mode}",
            geometry_only=True,
            hide_progress_bars=True,
        )
        grid = captured["grid"]
        idHx = grid.IDlookup["Hx"]
        numid = grid.ID[idHx, 5, 0, 0]
        return next(m for m in grid.materials if m.numID == numid)

    default_mat = _run(None)
    harmonic_mat = _run("harmonic")
    arithmetic_mat = _run("arithmetic")

    assert default_mat.mr == pytest.approx(1.6)  # harmonic mean of (1, 4)
    assert harmonic_mat.mr == pytest.approx(1.6)
    assert arithmetic_mat.mr == pytest.approx(2.5)  # (1 + 4) / 2, the old behaviour
    assert ":" not in harmonic_mat.ID
