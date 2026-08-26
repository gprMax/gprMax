"""Local topology checks for volumetric surface-impedance geometry."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import gprMax
from gprMax.impedance_surfaces import _electric_quadrants, _validate_impedance_voxel_topology

_EDGE_DIAGONALS = frozenset((0b0101, 0b1010))


def _empty_owner(shape=(8, 8, 8)) -> np.ndarray:
    return np.full(shape, -1, dtype=np.int32)


def _edge_mask_owner(axis: int, mask: int) -> np.ndarray:
    """Place one four-bit occupancy mask around a central Yee edge."""

    owner = _empty_owner()
    coordinate = (4, 4, 4)
    _, offsets = _electric_quadrants(owner, axis)
    for quadrant, offset in enumerate(offsets):
        if mask & (1 << quadrant):
            cell = tuple(coordinate[dimension] + offset[dimension] for dimension in range(3))
            owner[cell] = quadrant % 2
    return owner


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
@pytest.mark.parametrize("mask", range(16), ids=lambda value: f"mask-{value:04b}")
def test_every_four_quadrant_edge_mask_rejects_only_alternating_diagonals(axis, mask):
    owner = _edge_mask_owner(axis, mask)

    if mask in _EDGE_DIAGONALS:
        direction = "xyz"[axis]
        with pytest.raises(
            ValueError,
            match=(
                rf"non-manifold at a Yee edge: {direction}-directed edge \(4, 4, 4\); "
                r"connect .* through a cell face"
            ),
        ):
            _validate_impedance_voxel_topology(owner)
    else:
        _validate_impedance_voxel_topology(owner)


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
@pytest.mark.parametrize("model_ids", ((0, 0), (0, 1)), ids=("same-model", "different-models"))
def test_face_contact_is_valid_for_same_or_different_surface_models(axis, model_ids):
    owner = _empty_owner()
    first = np.asarray((3, 3, 3), dtype=np.int32)
    second = first.copy()
    second[axis] += 1
    owner[tuple(first)] = model_ids[0]
    owner[tuple(second)] = model_ids[1]

    _validate_impedance_voxel_topology(owner)


@pytest.mark.parametrize(
    "offset",
    ((1, 1, 0), (1, 0, 1), (0, 1, 1)),
    ids=("z-edge", "y-edge", "x-edge"),
)
def test_edge_only_contact_is_rejected(offset):
    owner = _empty_owner()
    first = np.asarray((3, 3, 3), dtype=np.int32)
    owner[tuple(first)] = 0
    owner[tuple(first + offset)] = 1

    with pytest.raises(ValueError, match=r"non-manifold at a Yee edge: [xyz]-directed edge"):
        _validate_impedance_voxel_topology(owner)


@pytest.mark.parametrize(
    ("first_offset", "second_offset"),
    (
        ((0, 0, 0), (1, 1, 1)),
        ((0, 0, 1), (1, 1, 0)),
        ((0, 1, 0), (1, 0, 1)),
        ((0, 1, 1), (1, 0, 0)),
    ),
    ids=("body-diagonal-0", "body-diagonal-1", "body-diagonal-2", "body-diagonal-3"),
)
@pytest.mark.parametrize("model_ids", ((0, 0), (0, 1)), ids=("same-model", "different-models"))
def test_vertex_only_contact_is_rejected(first_offset, second_offset, model_ids):
    owner = _empty_owner()
    base = np.asarray((3, 3, 3), dtype=np.int32)
    owner[tuple(base + first_offset)] = model_ids[0]
    owner[tuple(base + second_offset)] = model_ids[1]

    with pytest.raises(
        ValueError,
        match=(
            r"non-manifold at grid vertex \(4, 4, 4\).*"
            r"not face-connected.*connect through cell faces"
        ),
    ):
        _validate_impedance_voxel_topology(owner)


def test_vertex_pinched_retained_region_is_rejected():
    owner = _empty_owner()
    base = np.asarray((3, 3, 3), dtype=np.int32)
    owner[3:5, 3:5, 3:5] = 0
    owner[tuple(base)] = -1
    owner[tuple(base + 1)] = -1

    with pytest.raises(
        ValueError,
        match=(
            r"non-manifold at grid vertex \(4, 4, 4\): the retained incident cells "
            r"are not face-connected"
        ),
    ):
        _validate_impedance_voxel_topology(owner)


def test_far_separated_impedance_bodies_need_not_be_globally_connected():
    owner = _empty_owner()
    owner[2, 2, 2] = 0
    owner[5, 5, 5] = 1

    _validate_impedance_voxel_topology(owner)


@pytest.mark.parametrize(
    "cells",
    (
        ((3, 3, 3), (4, 3, 3), (4, 4, 3)),
        ((3, 3, 3), (4, 3, 3), (4, 4, 3), (4, 4, 4)),
    ),
    ids=("edge-contact", "vertex-contact"),
)
def test_face_connected_bridge_repairs_local_contact(cells):
    owner = _empty_owner()
    for index, cell in enumerate(cells):
        owner[cell] = index % 2

    _validate_impedance_voxel_topology(owner)


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


def _scene_with_boxes(boxes) -> gprMax.Scene:
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.030, 0.030, 0.030)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    scene.add(
        gprMax.SurfaceImpedance(
            id="wall",
            preset="copper",
            fit_frequency_range=(8e9, 12e9),
            fit_order="auto",
        )
    )
    for lower, upper in boxes:
        scene.add(
            gprMax.Box(
                p1=lower,
                p2=upper,
                material_id="wall",
                averaging="n",
            )
        )
    return scene


@pytest.mark.integration
def test_topology_preflight_failure_does_not_mutate_grid_component_ids(tmp_path, monkeypatch):
    import gprMax.impedance_surfaces as implementation

    captured = {}
    original = implementation.compile_impedance_surfaces

    def capture(grid):
        captured["before"] = grid.ID.copy()
        captured["materials_before"] = tuple(material.ID for material in grid.materials)
        try:
            return original(grid)
        finally:
            captured["after"] = grid.ID.copy()
            captured["materials_after"] = tuple(material.ID for material in grid.materials)

    monkeypatch.setattr(implementation, "compile_impedance_surfaces", capture)
    scene = _scene_with_boxes(
        (
            ((0.010, 0.010, 0.010), (0.011, 0.011, 0.011)),
            ((0.011, 0.011, 0.010), (0.012, 0.012, 0.011)),
        )
    )

    with pytest.raises(ValueError, match=r"non-manifold at a Yee edge: z-directed edge"):
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / "edge_contact",
            geometry_only=True,
            hide_progress_bars=True,
            cpu_precision="double",
        )

    np.testing.assert_array_equal(captured["after"], captured["before"])
    assert captured["materials_after"] == captured["materials_before"]


@pytest.mark.integration
def test_valid_staircase_keeps_reentrant_quarter_cell_clipped_h_update(tmp_path, monkeypatch):
    import gprMax.impedance_surfaces as implementation

    captured = {}
    original = implementation.compile_impedance_surfaces

    def capture(grid):
        system = original(grid)
        captured["system"] = system
        return system

    monkeypatch.setattr(implementation, "compile_impedance_surfaces", capture)
    scene = _scene_with_boxes(
        (
            ((0.010, 0.010, 0.010), (0.016, 0.016, 0.016)),
            ((0.016, 0.013, 0.010), (0.019, 0.016, 0.016)),
        )
    )
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "staircase",
        geometry_only=True,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    system = captured["system"]
    reentrant = np.flatnonzero(system.edge_fraction == 0.25)
    assert reentrant.size > 0
    for edge_index in reentrant:
        edge = system.edge_info[edge_index]
        assert edge[5] == 2
        assert edge[7] == 2
        h_slice = slice(edge[4], edge[4] + edge[5])
        port_slice = slice(edge[6], edge[6] + edge[7])
        np.testing.assert_allclose(np.abs(system.h_weight[h_slice]), 0.0005)
        np.testing.assert_allclose(system.port_g[port_slice], -0.0005)
        assert len(set(system.port_normal[port_slice, 0])) == 2
