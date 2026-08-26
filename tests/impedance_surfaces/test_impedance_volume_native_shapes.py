"""End-to-end compilation of native volumetric impedance shapes."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import gprMax


@pytest.fixture(autouse=True)
def restore_package_logging():
    """Do not leak ``gprMax.run``'s application logger into later tests."""

    yield
    logger = logging.getLogger("gprMax")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def _shape(kind: str):
    common = {"material_id": "wall", "averaging": "n", "tag": "body"}
    if kind == "box":
        return gprMax.Box(p1=(0.006, 0.007, 0.008), p2=(0.014, 0.016, 0.018), **common)
    if kind == "sphere":
        return gprMax.Sphere(p1=(0.015, 0.015, 0.015), r=0.005, **common)
    if kind == "ellipsoid":
        return gprMax.Ellipsoid(p1=(0.015, 0.015, 0.015), xr=0.006, yr=0.004, zr=0.005, **common)
    if kind == "cylinder":
        return gprMax.Cylinder(
            p1=(0.015, 0.015, 0.007),
            p2=(0.015, 0.015, 0.023),
            r=0.003,
            **common,
        )
    if kind == "oblique_cylinder":
        return gprMax.Cylinder(
            p1=(0.008, 0.009, 0.010),
            p2=(0.022, 0.019, 0.018),
            r=0.003,
            **common,
        )
    if kind == "cone":
        return gprMax.Cone(
            p1=(0.008, 0.009, 0.010),
            p2=(0.022, 0.019, 0.018),
            r1=0.004,
            r2=0.002,
            **common,
        )
    if kind == "sector":
        return gprMax.CylindricalSector(
            normal="z",
            ctr1=0.015,
            ctr2=0.015,
            extent1=0.010,
            extent2=0.017,
            start=20,
            end=140,
            r=0.007,
            **common,
        )
    if kind == "prism":
        return gprMax.Triangle(
            p1=(0.008, 0.008, 0.010),
            p2=(0.023, 0.008, 0.010),
            p3=(0.008, 0.022, 0.010),
            thickness=0.006,
            **common,
        )
    raise AssertionError(f"unknown test shape {kind!r}")


def _scene(kind: str) -> gprMax.Scene:
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.032, 0.032, 0.032)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.SurfaceImpedance(id="wall", resistance=50.0))
    scene.add(_shape(kind))
    return scene


@pytest.mark.integration
@pytest.mark.parametrize(
    "kind",
    ("box", "sphere", "ellipsoid", "cylinder", "cone", "sector", "prism"),
)
def test_native_volumetric_shape_compiles_to_sparse_impedance_boundary(kind, tmp_path, monkeypatch):
    import gprMax.impedance_surfaces as implementation

    captured = {}
    original = implementation.compile_impedance_surfaces

    def capture(grid):
        system = original(grid)
        captured["grid"] = grid
        captured["system"] = system
        return system

    monkeypatch.setattr(implementation, "compile_impedance_surfaces", capture)
    gprMax.run(
        scenes=[_scene(kind)],
        outputfile=tmp_path / kind,
        geometry_only=True,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    grid = captured["grid"]
    system = captured["system"]
    tag_id = grid.geometry_tag_registry.id_for("body")
    tagged_cells = grid.geometry_tag_map.data == tag_id
    assert np.count_nonzero(tagged_cells) > 0
    marker_id = next(iter(grid.impedance_marker_models))
    assert np.all(grid.solid[tagged_cells] == marker_id)
    assert system is grid.impedance_surfaces
    assert system.model_ids == ("wall",)
    assert system.edge_count > 0
    assert system.port_count > 0
    assert not np.isin(grid.ID, tuple(grid.impedance_marker_models)).any()

    sentinels = {
        material.impedance_role: material.numID
        for material in grid.materials
        if hasattr(material, "impedance_role")
    }
    assert set(sentinels) == {"surface-hold", "volume-void"}
    assert np.count_nonzero(grid.ID[:3] == sentinels["volume-void"]) > 0
    assert np.count_nonzero(grid.ID[3:] == sentinels["volume-void"]) > 0
    for component, i, j, k, *_ in system.edge_info:
        assert grid.ID[component, i, j, k] == sentinels["surface-hold"]


@pytest.mark.integration
def test_non_manifold_oblique_cylinder_voxelisation_is_rejected(tmp_path):
    """A thin oblique cylinder can leave diagonally touching occupied cells.

    There is no unique closed boundary normal at the resulting Yee edge, so
    silently compiling it would create an ambiguous surface-current port.
    """

    with pytest.raises(
        ValueError, match="impedance-volume voxel topology is non-manifold at a Yee edge"
    ):
        gprMax.run(
            scenes=[_scene("oblique_cylinder")],
            outputfile=tmp_path / "oblique_cylinder",
            geometry_only=True,
            hide_progress_bars=True,
            cpu_precision="double",
        )
