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

"""Regression tests for per-voxel averaging gating in build_voxels_from_array
and build_voxels_from_array_mask (used by FractalBox and GeometryObjectsRead).

Bug: unlike Box/Cylinder/etc. (which compute
`averaging = materials[0].averagable and user_flag`, forcing the rigid path
whenever the material is non-averagable, e.g. PEC/PMC or any custom
se=inf/sm=inf material), FractalBox.build() set `self.volume.averaging`
purely from the user/grid default flag, never checking whether any of its
mixing model's constituent materials (resolvable since the 2023-04-21
#material_list/#material_range mixing models let a fractal box reference
arbitrary predefined materials by name) are non-averagable. This let a PEC
"bin" in a fractal box's mixing model get its E properties blended with a
real neighbour's via the ordinary averaging pass, producing a physically
meaningless compound material with se=inf.

Fixed by gating averaging per-voxel inside build_voxels_from_array/_mask:
`voxel_averaging = averaging and is_averagable_lookup[numID]`, so a
non-averagable bin always takes the rigid path (with the existing PEC/H
carve-out still applying there) regardless of the object-level averaging
request, while other (averagable) bins continue to be smoothed normally with
each other.
"""
import numpy as np

from gprMax.cython.geometry_primitives import build_voxels_from_array
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material, create_built_in_materials


def _grid(nx, ny, nz, extra_materials=()):
    grid = FDTDGrid()
    grid.nx, grid.ny, grid.nz = nx, ny, nz
    create_built_in_materials(grid)
    grid.materials += list(extra_materials)
    grid.initialise_geometry_arrays()
    return grid


PEC_NUMID = 0


def test_pec_voxel_in_array_takes_rigid_path_even_when_averaging_requested():
    """A PEC-numID voxel in the data array must be rigid (E marked rigid,
    solid[] set directly) even though averaging=True was requested for the
    whole call - the per-voxel gate must override it for this one voxel."""
    grid = _grid(2, 1, 1)
    data = np.array([[[PEC_NUMID]], [[PEC_NUMID]]], dtype=np.int16).reshape(2, 1, 1)
    is_pec_lookup = np.array([m.is_pec for m in grid.materials], dtype=np.uint8)
    is_averagable_lookup = np.array([m.averagable for m in grid.materials], dtype=np.uint8)

    build_voxels_from_array(
        0, 0, 0, 0, True, is_pec_lookup, is_averagable_lookup, data,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    # Rigid path was taken despite averaging=True: E is marked rigid.
    assert grid.rigidE[0, 0, 0, 0]
    assert grid.rigidE[0, 1, 0, 0]


def test_averagable_voxel_in_array_still_averages_when_requested():
    """An ordinary averagable material in the same call must still take the
    averaging path (E unmarked rigid, deferred to the general averaging
    pass) - the PEC gate must not disable averaging for everything."""
    grid = _grid(2, 1, 1)
    matA = Material(3, "matA")
    matA.se = 5.0
    grid.materials.append(matA)

    data = np.full((2, 1, 1), matA.numID, dtype=np.int16)
    is_pec_lookup = np.array([m.is_pec for m in grid.materials], dtype=np.uint8)
    is_averagable_lookup = np.array([m.averagable for m in grid.materials], dtype=np.uint8)

    build_voxels_from_array(
        0, 0, 0, 0, True, is_pec_lookup, is_averagable_lookup, data,
        grid.solid, grid.rigidE, grid.rigidH, grid.ID,
    )

    assert not grid.rigidE[0, 0, 0, 0]
    assert not grid.rigidE[0, 1, 0, 0]


def test_fractal_box_with_pec_in_mixing_model_creates_no_infinite_conductivity_compound(tmp_path):
    """End-to-end regression for the exact scenario that surfaced this bug:
    a FractalBox using a #material_list mixing model that includes 'pec'
    must not create any compound material with se=inf, regardless of the
    requested averaging setting - while ordinary averaging between the
    other (real) materials in the mixing model must still occur normally.
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
    scene.add(gprMax.Title(name="fractal_pec_averaging_gate"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=6, se=0.01, mr=3, sm=0.02, id="matA"))
    scene.add(gprMax.Material(er=10, se=0.02, mr=1, sm=0, id="matB"))
    scene.add(gprMax.MaterialList(id="mymix", list_of_materials=["matA", "matB", "pec"]))
    scene.add(gprMax.FractalBox(
        p1=(0, 0, 0), p2=(0.02, 0.02, 0.02),
        frac_dim=1.5, weighting=(1, 1, 1), n_materials=3,
        mixing_model_id="mymix", id="myfractalbox", seed=1,
        averaging="y",
    ))

    grid = _capture(scene, tmp_path / "fractal_pec_averaging_gate")

    inf_compounds = [m.ID for m in grid.materials if "+" in m.ID and m.se == float("inf")]
    assert inf_compounds == [], f"found infinite-conductivity compound material(s): {inf_compounds}"

    # Ordinary averaging between the two real materials must still work.
    normal_compounds = [m.ID for m in grid.materials if "+" in m.ID and "pec" not in m.ID]
    assert normal_compounds, "expected ordinary matA/matB averaging to still produce compound materials"
