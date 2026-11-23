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

"""End-to-end tests for `inf`/NaN in #sphere and #ellipsoid centres.

Sphere and Ellipsoid were originally left out of the `inf` wiring (see the
original note in test_primitives_inf.py) because their centres have no
"fill to the wall" range semantics the way a Box/Cylinder's boundary corners
do. The single-point convention (resolve_inf_point, role=None - the same
rule as #rx/#hertzian_dipole) applies cleanly: on the invariant axis of an
active 2D mode the centre resolves to the interior reference layer; on a
transverse axis -inf/+inf snap the centre to the axis faces (a wall-clipped
half shape, permitted by the "may extend outwith the domain" rule); in 3D
`inf` raises the standard clean error. NaN is never valid.

Before the fix both commands crashed with a raw decimal.InvalidOperation
from round_int() (or the cryptic "cannot convert NaN to integer").
"""

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


def _run(monkeypatch, tmp_path, label, scene):
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )
    return captured["grid"]


def _scene_2d(mode, dl=1e-3, extent=0.02):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Domain(p1=(extent, extent, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    return scene


def _scene_3d(dl=1e-3, extent=0.02):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(extent, extent, extent)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    return scene


# --- Sphere -----------------------------------------------------------


def test_sphere_inf_centre_on_tm_invariant_axis_builds_at_interior_layer(monkeypatch, tmp_path):
    scene = _scene_2d("TM")
    scene.add(gprMax.Sphere(p1=(0.01, 0.01, INF), r=0.005, material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "sph_tm", scene)
    # z resolves to the single TM cell (index 0); 3 is diel's solid value in
    # these scenes (same convention as test_primitives_inf.py).
    assert grid.solid[10, 10, 0] == 3
    assert grid.solid[15, 10, 0] != 3


def test_sphere_inf_centre_on_te_invariant_axis_builds_at_interior_layer(monkeypatch, tmp_path):
    scene = _scene_2d("TE")
    scene.add(gprMax.Sphere(p1=(0.01, 0.01, INF), r=0.005, material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "sph_te", scene)
    # z resolves to the TE interior layer (index 1); the sphere is thick
    # enough to also cover the forced boundary layer at index 0.
    assert grid.solid[10, 10, 1] == 3
    assert grid.solid[10, 10, 0] == 3


@pytest.mark.parametrize("sign, wall_cell", [(1, 19), (-1, 0)])
def test_sphere_inf_centre_on_transverse_axis_snaps_to_the_face(
    monkeypatch, tmp_path, sign, wall_cell
):
    scene = _scene_2d("TM")
    scene.add(gprMax.Sphere(p1=(sign * INF, 0.01, 0.0), r=0.002, material_id="diel"))

    grid = _run(monkeypatch, tmp_path, f"sph_x_{sign}", scene)
    # +inf snaps the centre to the x extent (upper face node), -inf to 0:
    # a wall-clipped half sphere, same face rule as every point command.
    assert grid.solid[wall_cell, 10, 0] == 3
    if sign == 1:
        assert grid.solid[17, 10, 0] != 3  # only the wall-adjacent sliver


def test_sphere_3d_with_inf_centre_is_rejected(monkeypatch, tmp_path):
    scene = _scene_3d()
    scene.add(gprMax.Sphere(p1=(0.01, 0.01, INF), r=0.005, material_id="diel"))

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, "sph_3d", scene)


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_sphere_nan_centre_is_rejected(monkeypatch, tmp_path, mode):
    scene = _scene_2d(mode)
    scene.add(gprMax.Sphere(p1=(0.01, float("nan"), 0.0), r=0.005, material_id="diel"))

    with pytest.raises(ValueError, match="centre coordinates must be finite"):
        _run(monkeypatch, tmp_path, f"sph_nan_{mode}", scene)


def test_sphere_finite_centre_is_unchanged(monkeypatch, tmp_path):
    scene = _scene_2d("TM")
    scene.add(gprMax.Sphere(p1=(0.01, 0.01, 0.0), r=0.005, material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "sph_ctrl", scene)
    assert grid.solid[10, 10, 0] == 3


def test_sphere_wholly_outside_the_2d_slab_reports_empty_rasterisation(monkeypatch, tmp_path):
    """The original sphere_inf probe used a z centre of 0.02 in a model whose
    invariant axis is one cell (0.004 m): the resolved centre is valid but
    the sphere intersects no cell, which must surface as the existing
    empty-rasterisation error rather than a crash."""
    scene = _scene_2d("TM", dl=4e-3, extent=0.04)
    scene.add(gprMax.Sphere(p1=(INF, 0.02, 0.02), r=0.008, material_id="diel"))

    with pytest.raises(ValueError, match="does not occupy any Yee cells"):
        _run(monkeypatch, tmp_path, "sph_empty", scene)


# --- Ellipsoid --------------------------------------------------------


def test_ellipsoid_inf_centre_on_tm_invariant_axis_builds(monkeypatch, tmp_path):
    scene = _scene_2d("TM")
    scene.add(
        gprMax.Ellipsoid(p1=(0.01, 0.01, INF), xr=0.005, yr=0.005, zr=0.005, material_id="diel")
    )

    grid = _run(monkeypatch, tmp_path, "ell_tm", scene)
    assert grid.solid[10, 10, 0] == 3


def test_ellipsoid_3d_with_inf_centre_is_rejected(monkeypatch, tmp_path):
    scene = _scene_3d()
    scene.add(
        gprMax.Ellipsoid(p1=(0.01, 0.01, INF), xr=0.005, yr=0.005, zr=0.005, material_id="diel")
    )

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, "ell_3d", scene)


def test_ellipsoid_nan_centre_is_rejected(monkeypatch, tmp_path):
    scene = _scene_2d("TM")
    scene.add(
        gprMax.Ellipsoid(
            p1=(0.01, 0.01, float("nan")), xr=0.005, yr=0.005, zr=0.005, material_id="diel"
        )
    )

    with pytest.raises(ValueError, match="centre coordinates must be finite"):
        _run(monkeypatch, tmp_path, "ell_nan", scene)


def test_ellipsoid_finite_centre_is_unchanged(monkeypatch, tmp_path):
    scene = _scene_2d("TM")
    scene.add(
        gprMax.Ellipsoid(p1=(0.01, 0.01, 0.0), xr=0.005, yr=0.005, zr=0.005, material_id="diel")
    )

    grid = _run(monkeypatch, tmp_path, "ell_ctrl", scene)
    assert grid.solid[10, 10, 0] == 3
