"""Geometry-general surface-impedance volume marking tests."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.geometry_tags import GeometryTagMap, GeometryTagRegistry
from gprMax.materials import Material
from gprMax.user_objects.cmds_geometry.box import Box
from gprMax.user_objects.cmds_geometry.cone import Cone
from gprMax.user_objects.cmds_geometry.cylinder import Cylinder
from gprMax.user_objects.cmds_geometry.cylindrical_sector import CylindricalSector
from gprMax.user_objects.cmds_geometry.ellipsoid import Ellipsoid
from gprMax.user_objects.cmds_geometry.impedance_volume import ImpedanceVolume
from gprMax.user_objects.cmds_geometry.plate import Plate
from gprMax.user_objects.cmds_geometry.sphere import Sphere
from gprMax.user_objects.cmds_geometry.triangle import Triangle


@pytest.fixture(autouse=True)
def three_dimensional_model(monkeypatch):
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="3D"))


@pytest.fixture
def tagged_grid():
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
    metal = Material(3, "metal")
    grid = StubGrid(
        nx=int(size[0]),
        ny=int(size[1]),
        nz=int(size[2]),
        size=size,
        dx=0.001,
        dy=0.001,
        dz=0.001,
        dl=np.asarray((0.001, 0.001, 0.001)),
        averagevolumeobjects=True,
        materials=[pec, pmc, free_space, metal],
        solid=np.full(tuple(size), free_space.numID, dtype=np.uint32),
        rigidE=np.zeros((12, *size), dtype=np.int8),
        rigidH=np.zeros((6, *size), dtype=np.int8),
        ID=np.full((6, *(size + 1)), free_space.numID, dtype=np.uint32),
    )
    registry = GeometryTagRegistry()
    registry.register("body")
    registry.freeze()
    grid.geometry_tag_registry = registry
    grid.geometry_tag_map = GeometryTagMap(tuple(grid.size), registry)
    grid.surface_impedance_models = {"wall": SimpleNamespace(ID="wall")}
    grid.impedance_marker_models = {}
    grid.impedance_volume_specs = []
    return grid


def _box():
    return Box(
        p1=(0.006, 0.007, 0.008),
        p2=(0.014, 0.016, 0.018),
        material_id="metal",
        averaging="n",
        tag="body",
    )


def _sphere():
    return Sphere(
        p1=(0.015, 0.015, 0.015),
        r=0.005,
        material_id="metal",
        averaging="n",
        tag="body",
    )


def _ellipsoid():
    return Ellipsoid(
        p1=(0.015, 0.015, 0.015),
        xr=0.006,
        yr=0.004,
        zr=0.005,
        material_id="metal",
        averaging="n",
        tag="body",
    )


def _cylinder():
    return Cylinder(
        p1=(0.008, 0.009, 0.010),
        p2=(0.022, 0.019, 0.018),
        r=0.003,
        material_id="metal",
        averaging="n",
        tag="body",
    )


def _cone():
    return Cone(
        p1=(0.008, 0.009, 0.010),
        p2=(0.022, 0.019, 0.018),
        r1=0.004,
        r2=0.002,
        material_id="metal",
        averaging="n",
        tag="body",
    )


def _cylindrical_sector():
    return CylindricalSector(
        normal="z",
        ctr1=0.015,
        ctr2=0.015,
        extent1=0.010,
        extent2=0.017,
        start=20,
        end=140,
        r=0.007,
        material_id="metal",
        averaging="n",
        tag="body",
    )


def _triangular_prism():
    return Triangle(
        p1=(0.008, 0.008, 0.010),
        p2=(0.023, 0.008, 0.010),
        p3=(0.008, 0.022, 0.010),
        thickness=0.006,
        material_id="metal",
        averaging="n",
        tag="body",
    )


@pytest.mark.parametrize(
    "shape_factory",
    (
        _box,
        _sphere,
        _ellipsoid,
        _cylinder,
        _cone,
        _cylindrical_sector,
        _triangular_prism,
    ),
    ids=("box", "sphere", "ellipsoid", "cylinder", "cone", "sector", "prism"),
)
def test_all_cell_voxelising_primitives_use_the_same_tag_conversion(tagged_grid, shape_factory):
    shape_factory().build(tagged_grid)
    tag_id = tagged_grid.geometry_tag_registry.id_for("body")
    selected = tagged_grid.geometry_tag_map.data == tag_id
    assert np.any(selected)
    before = tagged_grid.solid.copy()

    ImpedanceVolume(geometry_tag="body", surface_impedance_id="wall").build(tagged_grid)

    marker_id = next(iter(tagged_grid.impedance_marker_models))
    assert np.all(tagged_grid.solid[selected] == marker_id)
    np.testing.assert_array_equal(tagged_grid.solid[~selected], before[~selected])
    spec = tagged_grid.impedance_volume_specs[-1]
    occupied = np.argwhere(selected)
    assert spec == {
        "kind": "tagged",
        "model_id": "wall",
        "geometry_tag": "body",
        "cell_count": int(np.count_nonzero(selected)),
        "lower": tuple(int(value) for value in occupied.min(axis=0)),
        "upper": tuple(int(value) for value in occupied.max(axis=0) + 1),
    }


def test_tag_overwrite_and_later_geometry_preserve_last_object_wins(tagged_grid):
    _box().build(tagged_grid)
    inner = Box(
        p1=(0.009, 0.010, 0.011),
        p2=(0.012, 0.013, 0.014),
        material_id="free_space",
        averaging="n",
    )
    inner.build(tagged_grid)
    tag_id = tagged_grid.geometry_tag_registry.id_for("body")
    surviving = tagged_grid.geometry_tag_map.data == tag_id

    ImpedanceVolume(geometry_tag="body", surface_impedance_id="wall").build(tagged_grid)
    marker_id = next(iter(tagged_grid.impedance_marker_models))
    assert np.all(tagged_grid.solid[surviving] == marker_id)
    assert np.all(tagged_grid.solid[9:12, 10:13, 11:14] == 2)

    later = Box(
        p1=(0.006, 0.007, 0.008),
        p2=(0.008, 0.009, 0.010),
        material_id="metal",
        averaging="n",
    )
    later.build(tagged_grid)
    assert np.all(tagged_grid.solid[6:8, 7:9, 8:10] == 3)


def test_shape_agnostic_conversion_accepts_an_irregular_tagged_voxel_set(tagged_grid):
    tag_id = tagged_grid.geometry_tag_registry.id_for("body")
    cells = np.asarray(((4, 5, 6), (4, 5, 7), (9, 11, 13), (10, 11, 13)))
    tagged_grid.geometry_tag_map.data[tuple(cells.T)] = tag_id
    tagged_grid.solid[tuple(cells.T)] = 3

    ImpedanceVolume(geometry_tag="body", surface_impedance_id="wall").build(tagged_grid)

    marker_id = next(iter(tagged_grid.impedance_marker_models))
    assert np.all(tagged_grid.solid[tuple(cells.T)] == marker_id)
    assert tagged_grid.impedance_volume_specs[-1]["cell_count"] == len(cells)


@pytest.mark.parametrize("geometry_tag", ("", "has space", "untagged", None))
def test_invalid_or_missing_target_tag_is_rejected(geometry_tag):
    with pytest.raises((TypeError, ValueError)):
        ImpedanceVolume(geometry_tag=geometry_tag, surface_impedance_id="wall")


def test_empty_target_tag_is_rejected(tagged_grid):
    with pytest.raises(ValueError, match="has no occupied cells"):
        ImpedanceVolume(geometry_tag="body", surface_impedance_id="wall").build(tagged_grid)


def test_unknown_impedance_model_uses_the_standard_lookup_error(tagged_grid):
    tagged_grid.geometry_tag_map.data[5, 6, 7] = tagged_grid.geometry_tag_registry.id_for("body")
    with pytest.raises(ValueError, match="there is no surface impedance"):
        ImpedanceVolume(geometry_tag="body", surface_impedance_id="missing").build(tagged_grid)


def test_missing_tag_map_is_rejected(tagged_grid):
    tagged_grid.geometry_tag_map = None
    with pytest.raises(ValueError, match="geometry tag 'body' is unavailable"):
        ImpedanceVolume(geometry_tag="body", surface_impedance_id="wall").build(tagged_grid)


def test_plate_has_no_cells_for_closed_volume_conversion(tagged_grid):
    Plate(
        p1=(0.008, 0.008, 0.010),
        p2=(0.020, 0.020, 0.010),
        material_id="metal",
        tag="body",
    ).build(tagged_grid)
    with pytest.raises(ValueError, match="has no occupied cells"):
        ImpedanceVolume(geometry_tag="body", surface_impedance_id="wall").build(tagged_grid)


def test_zero_thickness_triangle_rejects_cell_tag_before_conversion(tagged_grid):
    patch = Triangle(
        p1=(0.008, 0.008, 0.010),
        p2=(0.020, 0.008, 0.010),
        p3=(0.008, 0.020, 0.010),
        thickness=0,
        material_id="metal",
        tag="body",
    )
    with pytest.raises(ValueError, match="cell-centred tag requires a volumetric prism"):
        patch.build(tagged_grid)
