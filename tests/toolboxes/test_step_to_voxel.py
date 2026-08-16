import csv
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from toolboxes.STEPtoVoxel.converter import _read_assignments
from toolboxes.STEPtoVoxel.grouping import suggest_material_groups
from toolboxes.STEPtoVoxel.markers import (
    classify_marker_name,
    load_markers,
    marker_record,
)
from toolboxes.STEPtoVoxel.step_metadata import StepMetadata, parse_step_entities
from toolboxes.STEPtoVoxel.voxeliser import (
    GridSpec,
    resolve_sweep_axis,
    voxelise_solid_scanline,
    write_gprmax_hdf5,
)


def _thin_box_mesh():
    vertices = np.array(
        [
            [1.0, 1.0, 2.00],
            [4.0, 1.0, 2.00],
            [4.0, 4.0, 2.00],
            [1.0, 4.0, 2.00],
            [1.0, 1.0, 2.25],
            [4.0, 1.0, 2.25],
            [4.0, 4.0, 2.25],
            [1.0, 4.0, 2.25],
        ]
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, triangles


def _box_mesh(lower=(1.2, 1.2, 1.2), upper=(4.2, 4.2, 4.2)):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    vertices, triangles = _thin_box_mesh()
    unit_vertices = vertices.copy()
    unit_vertices[:, 0] = (vertices[:, 0] - 1.0) / 3.0
    unit_vertices[:, 1] = (vertices[:, 1] - 1.0) / 3.0
    unit_vertices[:, 2] = (vertices[:, 2] - 2.0) / 0.25
    return lower + unit_vertices * (upper - lower), triangles


def test_thin_closed_solid_is_preserved_when_between_slice_centres():
    vertices, triangles = _thin_box_mesh()
    absent = voxelise_solid_scanline(
        vertices,
        triangles,
        (6, 6, 6),
        preserve_thin_features=False,
    )
    preserved = voxelise_solid_scanline(
        vertices,
        triangles,
        (6, 6, 6),
        preserve_thin_features=True,
    )

    assert not absent.any()
    assert preserved.any()
    assert np.count_nonzero(preserved[:, :, 2]) > 0
    assert np.count_nonzero(preserved[:, :, :2]) == 0
    assert np.count_nonzero(preserved[:, :, 3:]) == 0


@pytest.mark.parametrize("thin_axis", (0, 1, 2))
def test_thin_closed_solid_preservation_is_axis_independent(thin_axis):
    vertices, triangles = _thin_box_mesh()
    permutation = [axis for axis in range(3) if axis != 2]
    permutation.insert(thin_axis, 2)
    vertices = vertices[:, permutation]

    preserved = voxelise_solid_scanline(
        vertices,
        triangles,
        (6, 6, 6),
        preserve_thin_features=True,
        sweep_axis="z",
    )

    assert preserved.any()
    occupied_layers = np.flatnonzero(np.any(preserved, axis=tuple(axis for axis in range(3) if axis != thin_axis)))
    np.testing.assert_array_equal(occupied_layers, (2,))


def test_voxel_centre_classification_is_independent_of_sweep_axis():
    vertices, triangles = _box_mesh()
    results = {
        axis: voxelise_solid_scanline(
            vertices,
            triangles,
            (7, 7, 7),
            sweep_axis=axis,
            supersample=1,
        )
        for axis in "xyz"
    }

    np.testing.assert_array_equal(results["x"], results["y"])
    np.testing.assert_array_equal(results["x"], results["z"])
    assert resolve_sweep_axis((10, 4, 8), "auto") == "y"
    assert resolve_sweep_axis((8, 8, 8), "auto") == "z"


def test_symmetric_supersampling_is_independent_of_sweep_axis():
    vertices, triangles = _box_mesh(lower=(1.1, 1.1, 1.1), upper=(4.4, 4.4, 4.4))
    results = {
        axis: voxelise_solid_scanline(
            vertices,
            triangles,
            (7, 7, 7),
            sweep_axis=axis,
            supersample=2,
        )
        for axis in "xyz"
    }

    np.testing.assert_array_equal(results["x"], results["y"])
    np.testing.assert_array_equal(results["x"], results["z"])


@pytest.mark.parametrize("supersample", (1, 2))
def test_oblique_closed_solid_is_independent_of_sweep_axis(supersample):
    vertices = np.array(
        [
            [1.1, 3.2, 3.4],
            [5.4, 3.2, 3.4],
            [3.2, 1.0, 3.4],
            [3.2, 5.6, 3.4],
            [3.2, 3.2, 1.3],
            [3.2, 3.2, 5.2],
        ]
    )
    triangles = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    results = {
        axis: voxelise_solid_scanline(
            vertices,
            triangles,
            (7, 7, 7),
            sweep_axis=axis,
            supersample=supersample,
        )
        for axis in "xyz"
    }

    np.testing.assert_array_equal(results["x"], results["y"])
    np.testing.assert_array_equal(results["x"], results["z"])


def test_gprmax_hdf5_schema_preserves_background(tmp_path):
    data = np.full((3, 4, 2), -1, dtype=np.int16)
    data[1, 2, 0] = 0
    grid = GridSpec(
        origin_world=np.array((-1e-3, 2e-3, 3e-3)),
        dxyz_world=np.array((1e-3, 2e-3, 3e-3)),
        nxyz=np.array(data.shape, dtype=np.int32),
    )
    path = tmp_path / "geometry.h5"
    write_gprmax_hdf5(path, data, grid)

    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["data"][:], data)
        np.testing.assert_allclose(f.attrs["dx_dy_dz"], grid.dxyz_world)
        assert f["data"].dtype == np.dtype(np.int16)


def test_repeated_material_assignments_are_compacted_by_value(tmp_path):
    path = tmp_path / "materials.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "part_name",
                "include",
                "priority",
                "material_name",
                "relative_permittivity",
                "conductivity",
                "relative_permeability",
                "magnetic_loss",
            )
        )
        writer.writerow(("patch", "y", 2, "pec", 1, "inf", 1, 0))
        writer.writerow(("ground", "y", 1, "pec", 1, "inf", 1, 0))

    assignments = _read_assignments(path)
    assert assignments["patch"].material == assignments["ground"].material


def _part(name, entity_id, *, volume=1.0, area=6.0, moments=(1.0, 1.0, 1.0)):
    return SimpleNamespace(
        name=name,
        step_entity_id=entity_id,
        cad={
            "vol_m3": volume,
            "area_m2": area,
            "bbox_dims_xyz": (1.0, 1.0, 1.0),
            "principal_moments_m5": moments,
            "topology_counts": {
                "solids": 1,
                "shells": 1,
                "faces": 6,
                "edges": 12,
                "vertices": 8,
            },
        },
    )


def test_exact_grouping_combines_repeated_step_instances_only():
    parts = (_part("screw_1", 10), _part("screw_2", 10), _part("plastic_copy", 20))
    groups = suggest_material_groups(parts, mode="exact")

    repeated = next(group for group in groups if group.confidence == "exact_instance")
    assert repeated.part_names == ("screw_1", "screw_2")
    assert repeated.similar_group
    assert next(group for group in groups if group.part_names == ("plastic_copy",)).confidence == "unique"


def test_similar_grouping_is_explicit_and_uses_rotation_invariant_metrics():
    parts = (
        _part("part_a", 1, moments=(1.0, 2.0, 3.0)),
        _part("part_b", 2, volume=1.002, area=6.003, moments=(3.006, 1.002, 2.004)),
    )

    exact = suggest_material_groups(parts, mode="exact", relative_tolerance=0.01)
    similar = suggest_material_groups(parts, mode="similar", relative_tolerance=0.01)

    assert len(exact) == 2
    assert len(similar) == 1
    assert similar[0].confidence == "similar_candidate"


def test_grouped_material_csv_expands_to_per_part_assignments(tmp_path):
    path = tmp_path / "materials.csv"
    path.write_text(
        "group_id,group_confidence,similar_group,part_count,part_names,include,priority,"
        "material_name,relative_permittivity,conductivity,relative_permeability,magnetic_loss\n"
        "G001,exact_instance,S001,2,screw_1|screw_2,y,3,steel,1,1e7,1,0\n",
        encoding="utf-8",
    )

    assignments = _read_assignments(path)

    assert set(assignments) == {"screw_1", "screw_2"}
    assert assignments["screw_1"] == assignments["screw_2"]
    assert assignments["screw_1"].group_confidence == "exact_instance"


def test_step_name_resolution_uses_entity_references_not_order(tmp_path):
    step_file = tmp_path / "metadata.stp"
    step_file.write_text(
        """ISO-10303-21;
DATA;
#10=MANIFOLD_SOLID_BREP('SOLID1',#11);
#20=ADVANCED_BREP_SHAPE_REPRESENTATION(
  'assembly|PATCH',
  (#10),#30);
#40=MANIFOLD_SOLID_BREP('assembly|GROUND',#41);
#50=PRODUCT('internal-id','Readable product name','',());
ENDSEC;
END-ISO-10303-21;
""",
        encoding="utf-8",
    )

    entities = parse_step_entities(step_file)
    metadata = StepMetadata(entities)
    patch = metadata.resolve_name(10)
    ground = metadata.resolve_name(40)

    assert patch is not None
    assert patch.name == "PATCH"
    assert patch.source_entity_id == 20
    assert patch.graph_distance == 1
    assert ground is not None
    assert ground.name == "GROUND"
    assert ground.source_entity_id == 40
    assert entities[50].name == "Readable product name"


def test_cad_surface_marker_coordinates_and_model_translation(tmp_path):
    assert classify_marker_name("gprmax_source_tx1") == ("source", "tx1")
    assert classify_marker_name("rx2") == ("receiver", "2")
    assert classify_marker_name("port1") == ("port", "1")
    assert classify_marker_name("substrate") is None

    part = SimpleNamespace(
        name="port1",
        uid=7,
        cad={
            "bbox_xyzxyz": (1.0, 2.0, 3.0, 5.0, 6.0, 3.0),
            "vol_m3": 0.0,
            "area_m2": 16.0,
        },
        step_entity_id=108,
        name_source="SHELL_BASED_SURFACE_MODEL",
        name_confidence="exact",
    )
    vertices = np.array(((1, 2, 3), (5, 2, 3), (5, 6, 3), (1, 6, 3)), dtype=float)
    triangles = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    record = marker_record(
        part,
        grid_origin=(0, 0, 0),
        spacing=(1, 1, 1),
        vertices=vertices,
        triangles=triangles,
    )

    assert record is not None
    assert record["kind"] == "port"
    assert record["geometry"] == "surface"
    assert record["local_position_m"] == [3.0, 4.0, 3.0]
    assert record["axis"] == "z"

    markers_file = tmp_path / "markers.json"
    markers_file.write_text(json.dumps({"markers": [record]}), encoding="utf-8")
    marker = load_markers(markers_file)["port1"]
    assert marker.model_position((10, 20, 30)) == (13.0, 24.0, 33.0)
    assert marker.model_bounds((10, 20, 30)) == (11.0, 22.0, 33.0, 15.0, 26.0, 33.0)


def test_cad_edge_marker_preserves_endpoints_length_and_axis():
    part = SimpleNamespace(
        name="gprmax_source_feed",
        uid=8,
        cad={
            "bbox_xyzxyz": (1e-3, 2e-3, 3e-3, 1e-3, 2e-3, 4e-3),
            "vol_m3": 0.0,
            "area_m2": 0.0,
        },
        step_entity_id=120,
        name_source="GEOMETRIC_SET",
        name_confidence="exact",
    )
    vertices = np.array(((1e-3, 2e-3, 4e-3), (1e-3, 2e-3, 3e-3)))
    record = marker_record(
        part,
        grid_origin=(0, 0, 0),
        spacing=(0.1e-3, 0.1e-3, 0.1e-3),
        vertices=vertices,
        triangles=np.empty((0, 3), dtype=np.int32),
    )

    assert record is not None
    assert record["kind"] == "source"
    assert record["geometry"] == "line"
    assert record["axis"] == "z"
    assert record["direction"] == [0.0, 0.0, 1.0]
    assert record["cad_endpoints_m"] == [[1e-3, 2e-3, 3e-3], [1e-3, 2e-3, 4e-3]]
    assert record["length_m"] == pytest.approx(1e-3)
