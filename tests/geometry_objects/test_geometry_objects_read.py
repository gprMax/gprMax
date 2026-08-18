"""Regression tests for #geometry_objects_write / #geometry_objects_read
round-tripping, specifically around builtin materials (pec, pmc, free_space).

Bug: GeometryObjectsRead.build() used to assume grids always have exactly
two builtin materials in a fixed order (pec, free_space) - a scheme from
before `pmc` was added as a third builtin (gprMax/materials.py,
create_built_in_materials(): pec=0, pmc=1, free_space=2). Since
GeometryObject.write_hdf5() only writes materials actually present in the
exported region (gprMax/geometry_outputs/grid_view.py,
initialise_materials(filter_materials=True)), exporting a region that uses
PEC but not PMC produces a materials file with just [pec, free_space] - and
the old code's scalar index offset (`data + numexistmaterials`) silently
mapped every PEC voxel to PMC on read-back, with no error. See the
conversation/commit history around geometry_objects_read.py for the full
derivation; this test locks in the fix (matching materials by ID name
against the target grid rather than assuming a fixed builtin count/order).
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod


def _capture_built_grid(monkeypatch):
    """Monkeypatches Model.build to capture the FDTDGrid after a real
    build, so the test can inspect grid.solid/grid.materials directly.
    """
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _write_geometry(tmp_path: Path, boxes: list[tuple[tuple, tuple, str]], custom_materials=()):
    """Builds a small domain with the given (p1, p2, material_id) boxes and
    exports it via #geometry_objects_write. Returns (geofile, database file).
    """
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="write"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))

    for material in custom_materials:
        scene.add(material)
    for p1, p2, material_id in boxes:
        scene.add(gprMax.Box(p1=p1, p2=p2, material_id=material_id))

    outdir = tmp_path / "write"
    scene.add(
        gprMax.GeometryObjectsWrite(p1=(0.0, 0.0, 0.0), p2=(0.02, 0.02, 0.02), filename=str(outdir))
    )
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=outdir, hide_progress_bars=True)

    return outdir.with_suffix(".h5"), Path(f"{outdir}_materials.json")


def _read_geometry_and_get_grid(
    tmp_path: Path,
    geofile: Path,
    database_file: Path,
    monkeypatch,
    *,
    averaging="n",
):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="read"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0.0, 0.0, 0.0),
            geofile=str(geofile),
            material_database=database_file.stem,
            averaging=averaging,
        )
    )

    captured = _capture_built_grid(monkeypatch)
    outfile = tmp_path / "read"
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=outfile, hide_progress_bars=True)
    return captured["grid"]


def _material_histogram(grid):
    unique_ids, counts = np.unique(grid.solid, return_counts=True)
    by_name = {}
    for uid, cnt in zip(unique_ids, counts):
        name = next(m.ID for m in grid.materials if m.numID == uid)
        by_name[name] = int(cnt)
    return by_name


def _read_voxel_interface(tmp_path, monkeypatch, averaging, *, dispersive=False):
    """Import two adjacent dielectric regions from a voxel-only file."""

    dl = 1e-3
    data = np.zeros((4, 4, 4), dtype=np.int16)
    data[2:, :, :] = 1
    suffix = "_dispersive" if dispersive else ""
    geofile = tmp_path / f"voxel_interface_{averaging}{suffix}.h5"
    with h5py.File(geofile, "w") as geometry:
        geometry.attrs["dx_dy_dz"] = (dl, dl, dl)
        geometry["/data"] = data

    matfile = tmp_path / "voxel_interface_materials.txt"
    material_text = (
        "#material: 4 0 1 0 left_dielectric\n"
        "#material: 9 0 1 0 right_dielectric\n"
    )
    if dispersive:
        material_text = (
            "#material: 4 0 1 0 left_dielectric\n"
            "#add_dispersion_debye: 1 3 1e-11 left_dielectric\n"
            "#material: 9 0 1 0 right_dielectric\n"
            "#add_dispersion_debye: 1 5 2e-11 right_dielectric\n"
        )
    matfile.write_text(material_text)

    scene = gprMax.Scene()
    if dispersive:
        scene.add(gprMax.DispersiveAveraging(enabled=True))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(4 * dl, 4 * dl, 4 * dl)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0, 0, 0),
            geofile=str(geofile),
            matfile=str(matfile),
            averaging=averaging,
        )
    )

    captured = _capture_built_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"voxel_interface_{averaging}{suffix}",
        hide_progress_bars=True,
    )
    return captured["grid"]


def test_voxel_only_import_can_enable_interface_averaging(tmp_path, monkeypatch):
    grid = _read_voxel_interface(tmp_path, monkeypatch, "y")

    assert any(
        "left_dielectric" in material.ID and "right_dielectric" in material.ID
        for material in grid.materials
    )


def test_voxel_only_import_averaging_remains_off_by_default(tmp_path, monkeypatch):
    grid = _read_voxel_interface(tmp_path, monkeypatch, "n")

    assert not any(
        "left_dielectric" in material.ID and "right_dielectric" in material.ID
        for material in grid.materials
    )


def test_voxel_only_import_uses_global_dispersive_averaging(tmp_path, monkeypatch):
    grid = _read_voxel_interface(tmp_path, monkeypatch, "y", dispersive=True)

    compound = next(
        material
        for material in grid.materials
        if "left_dielectric" in material.ID and "right_dielectric" in material.ID
    )
    assert compound.poles == 2
    assert compound.tau == pytest.approx([1e-11, 2e-11])
    assert compound.deltaer == pytest.approx([1.5, 2.5])


def test_full_component_import_ignores_averaging_request(tmp_path, monkeypatch, capsys):
    geofile, database_file = _write_geometry(
        tmp_path, [((0.005, 0.005, 0.005), (0.015, 0.015, 0.015), "pec")]
    )

    _read_geometry_and_get_grid(
        tmp_path,
        geofile,
        database_file,
        monkeypatch,
        averaging="y",
    )
    output = capsys.readouterr().out

    assert "component arrays" in output
    assert "averaging='y' is ignored" in output


def test_pec_only_region_round_trips_as_pec(tmp_path, monkeypatch):
    """The originally-reported bug: a region using PEC but not PMC must not
    have its PEC voxels come back as PMC.
    """
    geofile, matfile = _write_geometry(
        tmp_path, [((0.005, 0.005, 0.005), (0.015, 0.015, 0.015), "pec")]
    )
    grid = _read_geometry_and_get_grid(tmp_path, geofile, matfile, monkeypatch)

    histogram = _material_histogram(grid)
    assert histogram.get("pec", 0) == 1000
    assert histogram.get("pmc", 0) == 0

    materials_by_id = {m.ID: m for m in grid.materials}
    assert materials_by_id["pec"].numID == 0
    assert materials_by_id["pmc"].numID == 1
    assert materials_by_id["free_space"].numID == 2
    # Reused builtins must not be mislabelled as freshly imported.
    assert materials_by_id["pmc"].type == "builtin"
    assert materials_by_id["free_space"].type == "builtin"


def test_semantic_tags_round_trip_and_are_remapped_by_name(tmp_path, monkeypatch):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.Box(
            p1=(0.004, 0.004, 0.004),
            p2=(0.012, 0.012, 0.012),
            material_id="pec",
            tag="housing",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0.006, 0.006, 0.006),
            p2=(0.010, 0.010, 0.010),
            material_id="free_space",
        )
    )
    outdir = tmp_path / "tagged_write"
    scene.add(
        gprMax.GeometryObjectsWrite(
            p1=(0.0, 0.0, 0.0), p2=(0.02, 0.02, 0.02), filename=str(outdir)
        )
    )
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True, outputfile=outdir, hide_progress_bars=True
    )

    grid = _read_geometry_and_get_grid(
        tmp_path,
        outdir.with_suffix(".h5"),
        Path(f"{outdir}_materials.json"),
        monkeypatch,
    )
    assert grid.geometry_tag_registry.names == ("untagged", "housing")
    tags = grid.geometry_tag_map.data
    assert np.all(tags[4:6, 4:6, 4:6] == 1)
    assert np.all(tags[6:10, 6:10, 6:10] == 0)


def test_pec_pmc_and_custom_material_round_trip_correctly(tmp_path, monkeypatch):
    """General case: PEC, PMC, free_space (background) and a custom
    material all present in the same exported region. Builtins must be
    reused (not duplicated); the custom material must get its own numID
    appended after the builtins.
    """
    custom = gprMax.Material(er=3.0, se=0.01, mr=1.0, sm=0.0, id="custom_mat")
    geofile, matfile = _write_geometry(
        tmp_path,
        boxes=[
            ((0.002, 0.002, 0.002), (0.008, 0.008, 0.008), "pec"),
            ((0.010, 0.010, 0.010), (0.016, 0.016, 0.016), "pmc"),
            ((0.017, 0.001, 0.001), (0.019, 0.003, 0.003), "custom_mat"),
        ],
        custom_materials=[custom],
    )
    grid = _read_geometry_and_get_grid(tmp_path, geofile, matfile, monkeypatch)

    histogram = _material_histogram(grid)
    assert histogram["pec"] == 216  # 6x6x6
    assert histogram["pmc"] == 216  # 6x6x6
    custom_name = next(name for name in histogram if name.startswith("custom_mat"))
    assert histogram[custom_name] == 8  # 2x2x2

    materials_by_id = {m.ID: m for m in grid.materials}
    assert materials_by_id["pec"].numID == 0
    assert materials_by_id["pec"].type == "builtin"
    assert materials_by_id["pmc"].numID == 1
    assert materials_by_id["pmc"].type == "builtin"
    assert materials_by_id["free_space"].numID == 2
    assert materials_by_id["free_space"].type == "builtin"

    # Exactly one custom material was created (not a duplicate per-box) and
    # it got a fresh numID after the three builtins.
    custom_materials = [
        m for m in grid.materials if m.ID.startswith("custom_mat") and "+" not in m.ID
    ]
    assert len(custom_materials) == 1
    assert custom_materials[0].numID == 3
    assert custom_materials[0].type == "imported"


def test_same_database_geometry_can_be_inserted_more_than_once(tmp_path, monkeypatch):
    """Repeated inserts reuse their namespaced material instead of colliding."""

    custom = gprMax.Material(er=3.0, se=0.01, mr=1.0, sm=0.0, id="custom_mat")
    geofile, database_file = _write_geometry(
        tmp_path,
        boxes=[((0.002, 0.002, 0.002), (0.006, 0.006, 0.006), "custom_mat")],
        custom_materials=[custom],
    )

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.04, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    for x in (0.0, 0.02):
        scene.add(
            gprMax.GeometryObjectsRead(
                p1=(x, 0.0, 0.0),
                geofile=str(geofile),
                material_database=database_file.stem,
            )
        )

    captured = _capture_built_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "read_twice",
        hide_progress_bars=True,
    )
    imported = [
        material
        for material in captured["grid"].materials
        if material.ID.startswith("custom_mat{") and "+" not in material.ID
    ]
    assert len(imported) == 1


def test_import_namespaces_a_conflicting_existing_material(tmp_path, monkeypatch):
    """A local material ID must not prevent importing a different CAD material."""

    custom = gprMax.Material(er=3.0, se=0.01, mr=1.0, sm=0.0, id="custom_mat")
    geofile, database_file = _write_geometry(
        tmp_path,
        boxes=[((0.002, 0.002, 0.002), (0.006, 0.006, 0.006), "custom_mat")],
        custom_materials=[custom],
    )

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Material(er=8.0, se=0.0, mr=1.0, sm=0.0, id="custom_mat"))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0.0, 0.0, 0.0),
            geofile=str(geofile),
            material_database=database_file.stem,
        )
    )

    captured = _capture_built_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "read_conflict",
        hide_progress_bars=True,
    )

    materials = {material.ID: material for material in captured["grid"].materials}
    assert materials["custom_mat"].er == pytest.approx(8.0)
    imported_id = f"custom_mat{{{database_file.stem}}}"
    assert materials[imported_id].er == pytest.approx(3.0)


def test_geometry_rejects_a_different_companion_database_name(tmp_path, monkeypatch):
    custom = gprMax.Material(er=3.0, se=0.01, mr=1.0, sm=0.0, id="custom_mat")
    geofile, _ = _write_geometry(
        tmp_path,
        boxes=[((0.002, 0.002, 0.002), (0.006, 0.006, 0.006), "custom_mat")],
        custom_materials=[custom],
    )

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0.0, 0.0, 0.0),
            geofile=str(geofile),
            material_database="another_database",
        )
    )

    with pytest.raises(ValueError, match="records material database.*not 'another_database'"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "database_mismatch",
            hide_progress_bars=True,
        )


def test_background_voxels_map_to_free_space_in_an_empty_model(tmp_path, monkeypatch):
    """In a model with no prior geometry, background (-1) voxels come back
    as free_space - because that's what's already occupying every cell
    before any geometry is built (see initialise_geometry_arrays()), not
    because -1 is special-cased to mean free_space specifically. See
    test_background_voxels_preserve_prior_geometry for the case where prior
    geometry means -1 must NOT become free_space.
    """
    geofile, matfile = _write_geometry(
        tmp_path, [((0.005, 0.005, 0.005), (0.007, 0.007, 0.007), "pec")]
    )
    grid = _read_geometry_and_get_grid(tmp_path, geofile, matfile, monkeypatch)

    histogram = _material_histogram(grid)
    assert histogram["pec"] == 8  # 2x2x2
    assert histogram["free_space"] == 20 * 20 * 20 - 8


@pytest.mark.parametrize(
    "with_rigid_and_id_arrays",
    [
        pytest.param(False, id="voxel_only-build_voxels_from_array"),
        pytest.param(True, id="full-read_data_and_read_ID"),
    ],
)
def test_background_voxels_preserve_prior_geometry(tmp_path, monkeypatch, with_rigid_and_id_arrays):
    """-1 in the file's /data array means "don't build anything here, leave
    whatever's already in the grid" (per the #geometry_objects_read
    documentation) - NOT "set this cell to free_space". A common real
    workflow is building a fractal/heterogeneous soil first, then importing
    a target (e.g. a buried pipe) whose file only marks the target's own
    voxels and leaves the rest -1, expecting the soil underneath the
    target's bounding box to remain untouched outside the target itself.

    Parametrized over both GeometryObjectsRead code paths: a "voxel only"
    file (no rigidE/rigidH/ID arrays - the common shape for a hand-made
    file, going through build_voxels_from_array, which already correctly
    skipped -1 before this fix) and a "full" file with those arrays present
    (the shape #geometry_objects_write always produces, going through
    ReadGeometryObject.read_data()/read_ID() - this path used to force -1
    to free_space unconditionally, which this test guards against).
    """
    dl = 1e-3
    n = 24

    matfile = tmp_path / "target_materials.txt"
    matfile.write_text("#material: 1 inf 1 0 pec\n")

    target_size = 6
    data = np.full((target_size,) * 3, -1, dtype=np.int16)
    data[2:4, 2:4, 2:4] = 0  # the target itself, in the middle of its bounding box

    geofile = tmp_path / "target.h5"
    with h5py.File(geofile, "w") as f:
        f.attrs["dx_dy_dz"] = (dl, dl, dl)
        f["/data"] = data
        if with_rigid_and_id_arrays:
            f["/rigidE"] = np.zeros((12, *data.shape), dtype=np.int8)
            f["/rigidH"] = np.zeros((6, *data.shape), dtype=np.int8)
            id_shape = (6, *(np.array(data.shape) + 1))
            f["/ID"] = np.full(id_shape, -1, dtype=np.int16)

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="preserve_prior"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(n * dl, n * dl, n * dl)))
    scene.add(gprMax.TimeWindow(time=1e-12))

    soil = gprMax.Material(er=6.0, se=0.01, mr=1.0, sm=0.0, id="soil")
    scene.add(soil)
    scene.add(
        gprMax.Box(
            p1=(2 * dl, 2 * dl, 2 * dl),
            p2=((n - 2) * dl, (n - 2) * dl, (n - 2) * dl),
            material_id="soil",
        )
    )
    # Placed so the imported 6x6x6 region sits fully inside the soil box.
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(8 * dl, 8 * dl, 8 * dl), geofile=str(geofile), matfile=str(matfile)
        )
    )

    captured = _capture_built_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "read",
        hide_progress_bars=True,
    )
    grid = captured["grid"]

    material_at = lambda i, j, k: next(
        m.ID for m in grid.materials if m.numID == grid.solid[i, j, k]
    )

    assert material_at(10, 10, 10) == "pec"  # the actual target voxel
    assert material_at(8, 8, 8) == "soil"  # -1 in the file, inside the imported bounding box
    assert material_at(5, 5, 5) == "soil"  # outside the imported region entirely, for reference


def test_reads_genuinely_external_materials_file(tmp_path, monkeypatch):
    """#geometry_objects_read must also work with a materials file that was
    NOT generated by #geometry_objects_write - e.g. the real
    toolboxes/AustinManWoman/AustinManWoman_materials.txt shipped in this
    repo, whose first material is a custom "Air" (not gprMax's builtin
    "free_space") and none of whose materials are named pec/pmc/free_space
    at all. The synthetic /data array below stands in for the real (large,
    externally-downloaded) AustinMan .h5 - only the materials file needs to
    be real for this test.
    """
    matfile = (
        Path(__file__).parents[2] / "toolboxes" / "AustinManWoman" / "AustinManWoman_materials.txt"
    )
    assert matfile.exists()

    dl = 2e-3
    n = 24
    data = np.full((n, n, n), -1, dtype=np.int16)
    data[0, 0, 0] = 0  # first material in the file ("Air")
    data[1, 1, 1] = 1  # second material in the file ("Aorta")

    geofile = tmp_path / "austinman_like.h5"
    with h5py.File(geofile, "w") as f:
        f.attrs["dx_dy_dz"] = (dl, dl, dl)
        f["/data"] = data

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="read"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(n * dl, n * dl, n * dl)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.GeometryObjectsRead(p1=(0.0, 0.0, 0.0), geofile=str(geofile), matfile=str(matfile))
    )

    captured = _capture_built_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "read",
        hide_progress_bars=True,
    )
    grid = captured["grid"]

    air_material = next(m for m in grid.materials if m.numID == grid.solid[0, 0, 0])
    aorta_material = next(m for m in grid.materials if m.numID == grid.solid[1, 1, 1])
    background_material = next(m for m in grid.materials if m.numID == grid.solid[10, 10, 10])

    assert air_material.ID.startswith("Air")
    assert aorta_material.ID.startswith("Aorta")
    assert aorta_material.er == pytest.approx(44.77)
    assert background_material.ID == "free_space"


def test_incomplete_materials_file_raises_clear_error(tmp_path, monkeypatch):
    """GitHub #497: a written region using two materials (pec + the
    implicit free_space background) paired with a hand-edited materials
    file that only declares one of them (matching the reporter's own
    materials.txt, which listed just "pec") used to crash deep inside numpy
    fancy-indexing with a bare `IndexError: index 1 is out of bounds for
    axis 0 with size 1` - confirmed still reproducible against current code
    before this fix. Must now raise a clear, actionable ValueError instead.
    """
    geofile, matfile = _write_geometry(
        tmp_path, [((0.005, 0.005, 0.005), (0.015, 0.015, 0.015), "pec")]
    )
    # Reproduce the reporter's own materials.txt: only "pec" declared, even
    # though the written region also used free_space as its background.
    incomplete_matfile = tmp_path / "incomplete_materials.txt"
    incomplete_matfile.write_text("#material: 1 inf 1 0 pec\n")

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="read"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.GeometryObjectsRead(
            p1=(0.0, 0.0, 0.0), geofile=str(geofile), matfile=str(incomplete_matfile)
        )
    )

    with pytest.raises(ValueError, match="only declares 1 material"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "read",
            hide_progress_bars=True,
        )
