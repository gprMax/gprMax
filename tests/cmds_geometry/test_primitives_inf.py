"""End-to-end tests for `inf` coordinates in the other geometry
primitives wired up alongside Box: Cylinder, Cone, Edge, Plate,
Triangle, and GeometryObjectsRead's placement offset.

Deliberately not covered here:
  - Sphere, Ellipsoid: centre+radius shapes with no axis of constant
    cross-section, so a centre snapped to a domain wall has no clean
    "fill to the wall" meaning the way a Box/Cylinder's boundary does.
  - CylindricalSector and fractal objects, which have dedicated test
    modules for their specialised 2D handling.

Key design points under test:
  - `inf` is only allowed in an active 2D mode (TM/TE) - a 3D model
    (including any subgrid, which can never be 2D) rejects it outright.
    This was originally a general "snap to domain edge" 3D convenience
    too, but that scope was walked back - see
    gprMax/user_inputs.py's resolve_inf_point() docstring for why.
  - Cylinder: p1/p2 are the two end-face centres, resolved like a Box's
    corners (role="lower"/"upper", sign ignored) since spanning the
    cylinder's own axis from wall to wall is a legitimate, common
    construct (a target running through the full invariant thickness).
  - Cone can never use `inf` at all: blocked in 3D by the general
    2D-only guard, and blocked in 2D by its own guard (its
    cross-section varies along its axis, unlike a cylinder's constant
    one - not invariant).
  - Plate is blocked in 3D by the general 2D-only guard. In 2D, only
    the orientation normal to the invariant axis is rejected (it would
    sit exactly on the already-forced PEC/PMC domain wall, so a
    material plate there is moot) - a plate normal to either
    transverse axis is a genuinely meaningful 2D shape (a thin sheet
    standing in the cross-section) and is allowed, with its extent
    along the invariant axis resolved the same way as Box's corners.
  - Edge keeps the plain role-based resolution (no special-casing for
    its two flat axes - if a user mistakenly puts `inf` there, Edge's
    own existing "not specified correctly" validation catches it).
  - Triangle: the vertex axis shared (=`inf`) across all 3 vertices is
    the extrusion normal; it resolves to the axis origin, and `thickness`
    resolves to the full invariant-axis extent if also `inf`.
"""
import tempfile
from pathlib import Path

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


def _scene_with_diel(dl=1e-3):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="diel"))
    return scene


# --- Cylinder ---------------------------------------------------------


def test_cylinder_spans_full_te_invariant_axis(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Cylinder(p1=(0.01, 0.01, INF), p2=(0.01, 0.01, INF), r=0.005, material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "cyl_te", scene)
    assert grid.solid[10, 10, :].tolist() == [3, 3]


def test_cylinder_spans_full_tm_invariant_axis(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Cylinder(p1=(0.01, 0.01, INF), p2=(0.01, 0.01, INF), r=0.005, material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "cyl_tm", scene)
    assert grid.solid[10, 10, :].tolist() == [3]


def test_cylinder_3d_with_inf_is_rejected(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.Cylinder(p1=(0.01, 0.01, INF), p2=(0.01, 0.01, INF), r=0.005, material_id="diel"))

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, "cyl_3d", scene)


# --- Cone ---------------------------------------------------------------


def test_cone_3d_with_inf_is_rejected(monkeypatch, tmp_path):
    """Cone can never use `inf` now: 3D usage is rejected by the
    2D-only guard, and 2D usage is separately rejected by Cone's own
    guard (non-invariant cross-section, see test_cone_rejected_in_2d_mode)."""
    scene = _scene_with_diel()
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(
        gprMax.Cone(p1=(0.01, 0.01, INF), p2=(0.01, 0.01, INF), r1=0.002, r2=0.008, material_id="diel")
    )

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, "cone_3d", scene)


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_cone_rejected_in_2d_mode(monkeypatch, tmp_path, mode):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(
        gprMax.Cone(p1=(0.01, 0.01, INF), p2=(0.01, 0.01, INF), r1=0.002, r2=0.008, material_id="diel")
    )

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, f"cone_{mode}", scene)


# --- Edge -----------------------------------------------------------------


def test_edge_3d_with_inf_is_rejected(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.Edge(p1=(0.005, 0.005, -INF), p2=(0.005, 0.005, INF), material_id="diel"))

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, "edge_3d", scene)


def test_edge_tmz_invariant_axis_spans_the_single_cell(monkeypatch, tmp_path):
    """In TM, an edge is only meaningful running along the invariant
    axis - `inf` there spans the full 1-cell thickness like a Box's
    role-based endpoints (0..1), not the mode-aware single-index
    override used for single points."""
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Edge(p1=(0.004, 0.004, INF), p2=(0.004, 0.004, INF), material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "edge_tmz", scene)
    assert grid.ID[2, 4, 4, :].tolist() == [3, 2]


def test_edge_tez_invariant_axis_is_constant_at_interior_layer(monkeypatch, tmp_path):
    """In TE, an edge is only meaningful running along a NON-invariant
    axis. Even though the user writes `inf` on the invariant axis for
    both endpoints, it must resolve to the SAME interior reference
    index (1), not diverge to 0/axis-max - otherwise the edge would not
    be flat on that axis at all."""
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Edge(p1=(0.004, INF, INF), p2=(0.004, INF, INF), material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "edge_tez", scene)
    # y spans 0..19 (half-open range), z is constant at the interior
    # layer (index 1) - matching the exact example worked through with
    # the user: 4,inf,inf / 4,inf,inf -> (4,0,1) to (4,max,1).
    assert grid.ID[1, 4, :, 1].tolist() == [3] * 20 + [2]
    # z=0 is untouched by the edge (whatever's there - pec from tez()'s
    # own defensive forcing - is unrelated to this edge).
    assert (grid.ID[1, 4, :, 0] != 3).all()


def test_edge_rejected_in_tm_when_not_along_invariant_axis(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.Edge(p1=(0.004, 0.0, 0.0), p2=(0.004, 0.02, 0.0), material_id="diel"))

    with pytest.raises(ValueError, match="invariant axis"):
        _run(monkeypatch, tmp_path, "edge_tmz_bad", scene)


def test_edge_rejected_in_te_when_along_invariant_axis(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.Edge(p1=(0.004, 0.004, 0.0), p2=(0.004, 0.004, 0.002), material_id="diel"))

    with pytest.raises(ValueError, match="invariant axis"):
        _run(monkeypatch, tmp_path, "edge_tez_bad", scene)


# --- Plate ------------------------------------------------------------


def test_plate_3d_with_inf_is_rejected(monkeypatch, tmp_path):
    """Plate can never use `inf` now: 3D usage is rejected by the
    2D-only guard, and 2D usage is separately rejected by Plate's own
    guard (see test_plate_rejected_in_2d_mode)."""
    scene = _scene_with_diel()
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.Plate(p1=(INF, INF, 0.005), p2=(INF, INF, 0.005), material_id="diel"))

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, "plate_3d", scene)


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_plate_rejected_in_2d_mode_when_normal_to_invariant_axis(monkeypatch, tmp_path, mode):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
    scene.add(gprMax.Plate(p1=(0.002, 0.002, 0), p2=(0.008, 0.008, 0), material_id="diel"))

    with pytest.raises(ValueError, match="2D"):
        _run(monkeypatch, tmp_path, f"plate_{mode}", scene)


def test_plate_tez_normal_to_transverse_axis_lives_at_interior_layer(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Plate(p1=(0.01, 0.005, INF), p2=(0.01, 0.015, INF), material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "plate_te_transverse", scene)
    # build_face_yz (via the z-extent loop) writes Ey at all three z nodes
    # (0, 1, 2), but tez() then re-forces the outer wall nodes (0 and 2)
    # back to pec regardless of what the plate wrote - only the interior
    # node (1) is the genuinely live plane in TEz, matching the same
    # "interior reference layer" convention already used by Edge/
    # CylindricalSector. Ez (also set by build_face_yz) is TEz's dead
    # component throughout, forced to pec everywhere, so it is not
    # checked here.
    assert grid.ID[1, 10, 5, 0] == 0
    assert grid.ID[1, 10, 5, 1] == 3
    assert grid.ID[1, 10, 5, 2] == 0


def test_plate_tmz_normal_to_transverse_axis_spans_single_cell(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Plate(p1=(0.01, 0.005, INF), p2=(0.01, 0.015, INF), material_id="diel"))

    grid = _run(monkeypatch, tmp_path, "plate_tm_transverse", scene)
    assert grid.ID[2, 10, 5, 0] == 3


# --- Triangle ---------------------------------------------------------


def test_triangle_spans_full_te_invariant_axis_via_normal_and_thickness(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Domain(p1=(INF, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(
        gprMax.Triangle(
            p1=(INF, 0.0, 0.0),
            p2=(INF, 0.02, 0.0),
            p3=(INF, 0.0, 0.02),
            thickness=INF,
            material_id="diel",
        )
    )

    grid = _run(monkeypatch, tmp_path, "tri_te", scene)
    assert grid.solid[:, 5, 5].tolist() == [3, 3]


def test_triangle_spans_full_tm_invariant_axis_via_normal_and_thickness(monkeypatch, tmp_path):
    scene = _scene_with_diel()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Domain(p1=(INF, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(
        gprMax.Triangle(
            p1=(INF, 0.0, 0.0),
            p2=(INF, 0.02, 0.0),
            p3=(INF, 0.0, 0.02),
            thickness=INF,
            material_id="diel",
        )
    )

    grid = _run(monkeypatch, tmp_path, "tri_tm", scene)
    assert grid.solid[:, 5, 5].tolist() == [3]


# --- GeometryObjectsRead ------------------------------------------------


def test_geometry_objects_read_offset_resolves_inf_in_2d_mode(monkeypatch):
    """Smoke-test only: confirm resolve_inf_point is wired into
    GeometryObjectsRead.build() and resolves `inf` correctly ahead of
    the discretise_point() call - full read behaviour is covered by
    tests/geometry_objects/test_geometry_objects_read.py."""
    import gprMax.config as config
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.user_objects.cmds_geometry.geometry_objects_read import GeometryObjectsRead

    class _FakeModelConfig:
        mode = "2D TEx"

    monkeypatch.setattr(config, "get_model_config", lambda: _FakeModelConfig())

    grid = FDTDGrid()
    grid.dl = (1e-3, 1e-3, 1e-3)
    grid.nx, grid.ny, grid.nz = (2, 10, 10)

    gor = GeometryObjectsRead(p1=(-INF, 0, 0), geofile="dummy.h5", matfile="dummy.txt")
    uip = gor._create_uip(grid)
    resolved = uip.resolve_inf_point(gor.kwargs["p1"])
    # x is the invariant axis of an active 2D mode and this is a single
    # point (role=None), so it redirects to the TE interior reference
    # layer (index 1), not the plain sign-based axis origin/extent.
    assert resolved == (0.001, 0, 0)


def test_geometry_objects_read_offset_with_inf_in_3d_is_rejected(monkeypatch):
    import gprMax.config as config
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.user_objects.cmds_geometry.geometry_objects_read import GeometryObjectsRead

    class _FakeModelConfig:
        mode = "3D"

    monkeypatch.setattr(config, "get_model_config", lambda: _FakeModelConfig())

    grid = FDTDGrid()
    grid.dl = (1e-3, 1e-3, 1e-3)
    grid.nx, grid.ny, grid.nz = (10, 10, 10)

    gor = GeometryObjectsRead(p1=(-INF, 0, 0), geofile="dummy.h5", matfile="dummy.txt")
    uip = gor._create_uip(grid)
    with pytest.raises(ValueError, match="2D"):
        uip.resolve_inf_point(gor.kwargs["p1"])
