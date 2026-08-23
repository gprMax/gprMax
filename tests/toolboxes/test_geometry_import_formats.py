from pathlib import Path

import h5py
import numpy as np
import pytest

from toolboxes.GeometryImport.mesh import convert_mesh, load_mesh_source, write_mesh_template
from toolboxes.GeometryImport.volume import load_label_volume


def _assert_test_label_volume(volume, labels):
    np.testing.assert_array_equal(volume.labels, labels)
    assert volume.spacing_m == (0.002, 0.003, 0.004)


@pytest.mark.integration
def test_nifti_preserves_axis_order_and_spacing(tmp_path):
    nib = pytest.importorskip("nibabel", reason="optional NIfTI dependency is not installed")
    labels = np.zeros((2, 3, 4), dtype=np.uint8)
    labels[1] = 2
    nifti = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.diag((2.0, 3.0, 4.0, 1.0))), nifti)
    _assert_test_label_volume(load_label_volume(nifti, unit="mm"), labels)


@pytest.mark.integration
def test_nrrd_preserves_axis_order_and_spacing(tmp_path):
    nrrd = pytest.importorskip("nrrd", reason="optional NRRD dependency is not installed")
    labels = np.zeros((2, 3, 4), dtype=np.uint8)
    labels[1] = 2

    nrrd_path = tmp_path / "labels.nrrd"
    nrrd.write(
        str(nrrd_path),
        labels,
        header={
            "space directions": np.diag((2.0, 3.0, 4.0)),
            "space origin": np.zeros(3),
            "space units": ["mm", "mm", "mm"],
        },
        index_order="F",
    )
    _assert_test_label_volume(load_label_volume(nrrd_path), labels)


@pytest.mark.integration
def test_metaimage_preserves_axis_order_and_spacing(tmp_path):
    sitk = pytest.importorskip(
        "SimpleITK", reason="optional MetaImage dependency is not installed"
    )
    labels = np.zeros((2, 3, 4), dtype=np.uint8)
    labels[1] = 2

    meta = tmp_path / "labels.mha"
    image = sitk.GetImageFromArray(np.transpose(labels, (2, 1, 0)))
    image.SetSpacing((2.0, 3.0, 4.0))
    sitk.WriteImage(image, str(meta))
    _assert_test_label_volume(load_label_volume(meta, unit="mm"), labels)


@pytest.mark.integration
def test_closed_vtp_surface_is_voxelised_and_tagged(tmp_path):
    pv = pytest.importorskip("pyvista", reason="optional VTK dependency is not installed")
    source_path = tmp_path / "sphere.vtp"
    pv.Sphere(radius=0.01, theta_resolution=12, phi_resolution=12).save(source_path)
    source = load_mesh_source(source_path, unit="m")
    assignments = write_mesh_template(source, tmp_path / "regions.csv")

    result = convert_mesh(
        source_path,
        assignments,
        tmp_path / "output",
        voxel_size=(0.002, 0.002, 0.002),
        unit="m",
    )

    assert result.kind == "surface"
    assert result.preview_file is not None
    assert result.preview_file.is_file()
    preview = pv.read(result.preview_file)
    assert set(preview.cell_data) == {"MaterialIndex", "TagID"}
    with h5py.File(result.geometry_file) as geometry:
        assert np.count_nonzero(geometry["data"][:] >= 0) > 0
        assert [value.decode() for value in geometry["tag_names"][:]] == [
            "untagged",
            "sphere",
        ]


@pytest.mark.integration
def test_vtu_volume_regions_are_sampled_at_cell_centres(tmp_path):
    pv = pytest.importorskip("pyvista", reason="optional VTK dependency is not installed")
    grid = pv.ImageData(dimensions=(3, 3, 3), spacing=(0.005, 0.005, 0.005))
    grid = grid.cast_to_unstructured_grid()
    grid.cell_data["region"] = np.where(grid.cell_centers().points[:, 0] < 0.005, 1, 2)
    source_path = tmp_path / "regions.vtu"
    grid.save(source_path)
    source = load_mesh_source(source_path, unit="m", region_array="region")
    assignments = write_mesh_template(source, tmp_path / "regions.csv")

    result = convert_mesh(
        source_path,
        assignments,
        tmp_path / "output",
        voxel_size=(0.0025, 0.0025, 0.0025),
        unit="m",
        region_array="region",
    )

    assert result.preview_file is not None
    assert result.preview_file.is_file()
    assert result.kind == "volume"
    with h5py.File(result.geometry_file) as geometry:
        tags, counts = np.unique(geometry["tag_data"][:], return_counts=True)
        assert dict(zip(tags, counts))[1] == 32
        assert dict(zip(tags, counts))[2] == 32


@pytest.mark.integration
def test_gmsh_physical_group_name_becomes_default_tag(tmp_path):
    pytest.importorskip("pyvista", reason="optional VTK dependency is not installed")
    meshio = pytest.importorskip("meshio", reason="optional mesh dependency is not installed")
    source_path = tmp_path / "tissue.msh"
    meshio.write(
        source_path,
        meshio.Mesh(
            np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1.0))),
            [("tetra", np.asarray(((0, 1, 2, 3),)))],
            cell_data={
                "gmsh:physical": [np.asarray((7,))],
                "gmsh:geometrical": [np.asarray((7,))],
            },
            field_data={"brain": np.asarray((7, 3))},
        ),
        file_format="gmsh22",
    )

    source = load_mesh_source(source_path, unit="mm")

    assert source.source_region_array == "gmsh:physical"
    assert source.regions[0].name == "brain"


@pytest.mark.integration
def test_gmsh_volume_ignores_lower_dimensional_boundary_groups(tmp_path):
    pytest.importorskip("pyvista", reason="optional VTK dependency is not installed")
    meshio = pytest.importorskip("meshio", reason="optional mesh dependency is not installed")
    source_path = tmp_path / "mixed_dimensions.msh"
    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1.0)))
    meshio.write(
        source_path,
        meshio.Mesh(
            points,
            [
                ("triangle", np.asarray(((0, 1, 2),))),
                ("tetra", np.asarray(((0, 1, 2, 3),))),
            ],
            cell_data={
                # Reusing physical tag 7 across dimensions is valid Gmsh and
                # verifies that the dimension disambiguates the group name.
                "gmsh:physical": [np.asarray((7,)), np.asarray((7,))],
                "gmsh:geometrical": [np.asarray((1,)), np.asarray((1,))],
            },
            field_data={
                "outer_boundary": np.asarray((7, 2)),
                "brain_volume": np.asarray((7, 3)),
            },
        ),
        file_format="gmsh22",
    )

    source = load_mesh_source(source_path, unit="mm")

    assert source.kind == "volume"
    assert source.dataset.n_cells == 1
    assert len(source.regions) == 1
    assert source.regions[0].name == "brain_volume"
    assert source.regions[0].cell_count == 1


@pytest.mark.parametrize(
    ("pad_cells", "supersample", "message"),
    ((-1, 1, "pad_cells"), (0, 0, "supersample")),
)
def test_mesh_conversion_rejects_invalid_grid_controls(
    tmp_path, pad_cells, supersample, message
):
    with pytest.raises(ValueError, match=message):
        convert_mesh(
            tmp_path / "not_read.vtp",
            tmp_path / "not_read.csv",
            tmp_path / "output",
            voxel_size=(0.001, 0.001, 0.001),
            unit="m",
            pad_cells=pad_cells,
            supersample=supersample,
        )
