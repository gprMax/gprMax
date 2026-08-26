"""Direct surface-impedance geometry material tests."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.impedance_surfaces import _sentinel_material
from gprMax.materials import Material
from gprMax.user_objects.cmds_geometry.box import Box
from gprMax.user_objects.cmds_geometry.cone import Cone
from gprMax.user_objects.cmds_geometry.cylinder import Cylinder
from gprMax.user_objects.cmds_geometry.cylindrical_sector import CylindricalSector
from gprMax.user_objects.cmds_geometry.edge import Edge
from gprMax.user_objects.cmds_geometry.ellipsoid import Ellipsoid
from gprMax.user_objects.cmds_geometry.magnetic_edge import MagneticEdge
from gprMax.user_objects.cmds_geometry.plate import Plate
from gprMax.user_objects.cmds_geometry.sphere import Sphere
from gprMax.user_objects.cmds_geometry.triangle import Triangle
from gprMax.user_objects.cmds_multiuse import Material as MaterialObject
from gprMax.user_objects.cmds_multiuse import SurfaceImpedance


@pytest.fixture(autouse=True)
def three_dimensional_model(monkeypatch):
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="3D"))


@pytest.fixture
def geometry_grid():
    size = np.asarray((32, 32, 32), dtype=np.int32)

    class StubGrid(SimpleNamespace):
        def within_bounds(self, point):
            if any(value < 0 or value > limit for value, limit in zip(point, self.size)):
                raise ValueError("point lies outside the grid")
            return True

    pec = Material(0, "pec")
    pec.se = float("inf")
    pec.averagable = False
    pmc = Material(1, "pmc")
    pmc.sm = float("inf")
    pmc.averagable = False
    free_space = Material(2, "free_space")
    bulk = Material(3, "bulk")
    return StubGrid(
        name="",
        nx=int(size[0]),
        ny=int(size[1]),
        nz=int(size[2]),
        size=size,
        global_size=size,
        dx=0.001,
        dy=0.001,
        dz=0.001,
        dl=np.asarray((0.001, 0.001, 0.001)),
        averagevolumeobjects=True,
        materials=[pec, pmc, free_space, bulk],
        solid=np.full(tuple(size), free_space.numID, dtype=np.uint32),
        rigidE=np.zeros((12, *size), dtype=np.int8),
        rigidH=np.zeros((6, *size), dtype=np.int8),
        ID=np.full((6, *(size + 1)), free_space.numID, dtype=np.uint32),
        geometry_tag_registry=None,
        geometry_tag_map=None,
        surface_impedance_models={"wall": SimpleNamespace(ID="wall")},
        impedance_marker_models={},
    )


def _box(material_id="wall"):
    return Box(
        p1=(0.006, 0.007, 0.008),
        p2=(0.014, 0.016, 0.018),
        material_id=material_id,
        averaging="y",
    )


def _sphere(material_id="wall"):
    return Sphere(
        p1=(0.015, 0.015, 0.015),
        r=0.005,
        material_id=material_id,
        averaging="y",
    )


def _ellipsoid(material_id="wall"):
    return Ellipsoid(
        p1=(0.015, 0.015, 0.015),
        xr=0.006,
        yr=0.004,
        zr=0.005,
        material_id=material_id,
        averaging="y",
    )


def _cylinder(material_id="wall"):
    return Cylinder(
        p1=(0.008, 0.009, 0.010),
        p2=(0.022, 0.019, 0.018),
        r=0.003,
        material_id=material_id,
        averaging="y",
    )


def _cone(material_id="wall"):
    return Cone(
        p1=(0.008, 0.009, 0.010),
        p2=(0.022, 0.019, 0.018),
        r1=0.004,
        r2=0.002,
        material_id=material_id,
        averaging="y",
    )


def _sector(material_id="wall"):
    return CylindricalSector(
        normal="z",
        ctr1=0.015,
        ctr2=0.015,
        extent1=0.010,
        extent2=0.017,
        start=20,
        end=140,
        r=0.007,
        material_id=material_id,
        averaging="y",
    )


def _prism(material_id="wall"):
    return Triangle(
        p1=(0.008, 0.008, 0.010),
        p2=(0.023, 0.008, 0.010),
        p3=(0.008, 0.022, 0.010),
        thickness=0.006,
        material_id=material_id,
        averaging="y",
    )


@pytest.mark.parametrize(
    "shape_factory",
    (_box, _sphere, _ellipsoid, _cylinder, _cone, _sector, _prism),
    ids=("box", "sphere", "ellipsoid", "cylinder", "cone", "sector", "prism"),
)
def test_cell_voxelisers_accept_surface_impedance_as_material(geometry_grid, shape_factory):
    shape_factory().build(geometry_grid)

    marker_id = next(iter(geometry_grid.impedance_marker_models))
    assert geometry_grid.impedance_marker_models[marker_id] == "wall"
    assert np.count_nonzero(geometry_grid.solid == marker_id) > 0
    marker = geometry_grid.materials[marker_id]
    assert marker.averagable is False


def test_direct_material_preserves_last_object_wins_and_cavities(geometry_grid):
    _box().build(geometry_grid)
    marker_id = next(iter(geometry_grid.impedance_marker_models))

    Box(
        p1=(0.009, 0.010, 0.011),
        p2=(0.012, 0.013, 0.014),
        material_id="free_space",
        averaging="n",
    ).build(geometry_grid)
    assert np.all(geometry_grid.solid[9:12, 10:13, 11:14] == 2)

    Box(
        p1=(0.006, 0.007, 0.008),
        p2=(0.008, 0.009, 0.010),
        material_id="bulk",
        averaging="n",
    ).build(geometry_grid)
    assert np.all(geometry_grid.solid[6:8, 7:9, 8:10] == 3)
    assert np.count_nonzero(geometry_grid.solid == marker_id) > 0


@pytest.mark.parametrize(
    "sheet",
    (
        Plate(
            p1=(0.008, 0.008, 0.010),
            p2=(0.020, 0.020, 0.010),
            material_id="wall",
        ),
        Edge(
            p1=(0.008, 0.010, 0.010),
            p2=(0.020, 0.010, 0.010),
            material_id="wall",
        ),
        MagneticEdge(
            p1=(0.008, 0.010, 0.010),
            p2=(0.020, 0.010, 0.010),
            material_id="wall",
        ),
        Triangle(
            p1=(0.008, 0.008, 0.010),
            p2=(0.020, 0.008, 0.010),
            p3=(0.008, 0.020, 0.010),
            thickness=0,
            material_id="wall",
        ),
        CylindricalSector(
            normal="z",
            ctr1=0.015,
            ctr2=0.015,
            extent1=0.010,
            extent2=0.010,
            start=20,
            end=140,
            r=0.007,
            material_id="wall",
        ),
    ),
    ids=("plate", "edge", "magnetic-edge", "triangle", "sector"),
)
def test_sheet_and_edge_geometry_reject_surface_impedance(geometry_grid, sheet):
    with pytest.raises(ValueError, match="closed, cell-occupying volume"):
        sheet.build(geometry_grid)


@pytest.mark.parametrize("upper_x", (0.006, 0.0064))
def test_flat_or_subcell_box_rejects_surface_impedance(geometry_grid, upper_x):
    flat = Box(
        p1=(0.006, 0.007, 0.008),
        p2=(upper_x, 0.016, 0.018),
        material_id="wall",
        averaging="n",
    )
    with pytest.raises(ValueError, match="positive cell extent"):
        flat.build(geometry_grid)


@pytest.mark.parametrize(
    "material_ids",
    (
        ("wall",),
        ("wall", "free_space", "free_space"),
        ("wall", "wall", "wall"),
    ),
)
def test_directional_surface_impedance_is_rejected(geometry_grid, material_ids):
    anisotropic = Box(
        p1=(0.006, 0.007, 0.008),
        p2=(0.014, 0.016, 0.018),
        material_ids=material_ids,
        averaging="n",
    )
    with pytest.raises(ValueError, match="isotropic material_id"):
        anisotropic.build(geometry_grid)


def test_ambiguous_bulk_and_surface_impedance_id_is_rejected(geometry_grid):
    geometry_grid.materials.append(Material(len(geometry_grid.materials), "wall"))
    with pytest.raises(ValueError, match="ambiguous"):
        _box().build(geometry_grid)


def test_surface_impedance_id_cannot_reuse_existing_material_id(geometry_grid):
    with pytest.raises(ValueError, match="conflicts with an existing material ID"):
        SurfaceImpedance(id="bulk", resistance=50.0).build(geometry_grid)


def test_material_id_cannot_reuse_existing_surface_impedance_id(geometry_grid):
    material = MaterialObject(er=2.0, se=0.0, mr=1.0, sm=0.0, id="wall")
    with pytest.raises(ValueError):
        material.build(geometry_grid)


def test_public_ids_cannot_enter_private_impedance_namespace(geometry_grid):
    with pytest.raises(ValueError, match="reserved prefix"):
        SurfaceImpedance(id="__impedance_custom", resistance=50.0).build(geometry_grid)

    material = MaterialObject(
        er=2.0,
        se=0.0,
        mr=1.0,
        sm=0.0,
        id="__impedance_custom",
    )
    with pytest.raises(ValueError):
        material.build(geometry_grid)


def test_private_marker_id_collision_is_rejected(geometry_grid):
    geometry_grid.materials.append(
        Material(len(geometry_grid.materials), "__impedance_volume__wall")
    )
    with pytest.raises(ValueError, match="reserved internal"):
        _box().build(geometry_grid)


def test_private_sentinel_id_collision_is_rejected(geometry_grid):
    geometry_grid.materials.append(
        Material(len(geometry_grid.materials), "__impedance_surface_hold__")
    )
    with pytest.raises(ValueError, match="reserved internal"):
        _sentinel_material(geometry_grid, "surface-hold")


def test_ordinary_material_geometry_is_unchanged(geometry_grid):
    _sphere(material_id="bulk").build(geometry_grid)
    assert np.count_nonzero(geometry_grid.solid == 3) > 0
    assert not geometry_grid.impedance_marker_models
