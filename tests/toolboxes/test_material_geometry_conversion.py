"""One-time legacy migration and the single JSON geometry-import path."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.model as model_module
from gprMax.material_database import load_material_spec
from gprMax.toolboxes.MaterialDatabase import convert_geometry
from gprMax.toolboxes.MaterialDatabase.convert_geometry import _blocks

converter = importlib.import_module("gprMax.toolboxes.MaterialDatabase.convert_geometry")
pytestmark = pytest.mark.unit


def pair(tmp_path, text="#material: 4 0.01 1 0 tissue\n", *, full=False):
    geometry, materials = tmp_path / "legacy.h5", tmp_path / "legacy.txt"
    materials.write_text(text, encoding="utf-8")
    with h5py.File(geometry, "w") as output:
        output.attrs["dx, dy, dz"] = (0.001,) * 3
        output.attrs["Title"] = "Original export"
        output.create_dataset("data", data=np.zeros((4,) * 3, np.int16), compression="gzip")
        if full:
            output["ID"] = np.zeros((6, 5, 5, 5), np.uint32)
            output["rigidE"] = np.ones((12, 4, 4, 4), np.int8)
            output["rigidH"] = np.ones((6, 4, 4, 4), np.int8)
        output["tag_data"] = np.ones((4, 4, 4), np.uint8)
        output.create_dataset("tag_names", data=["untagged", "organ"], dtype=h5py.string_dtype())
        output["data"].attrs["description"] = "original cell materials"
    return geometry, materials


@pytest.mark.parametrize("full", (False, True))
def test_conversion_preserves_every_original_dataset_and_attribute(tmp_path, full):
    source, materials = pair(tmp_path, full=full)
    with h5py.File(source, "r+") as output:
        output["data"][0] = -1
        if full:
            output["ID"][:, 0] = np.iinfo(np.uint32).max  # Positive, not a -1 sentinel.
    if full:
        with pytest.raises(ValueError, match="material indices"):
            convert_geometry(source, materials)
        with h5py.File(source, "r+") as output:
            output["ID"][:, 0] = 0
    original_bytes = source.read_bytes(), materials.read_bytes()
    converted, database = convert_geometry(source, materials)
    assert (source.read_bytes(), materials.read_bytes()) == original_bytes
    with h5py.File(source) as before, h5py.File(converted) as after:
        for name in before:
            np.testing.assert_array_equal(before[name][:], after[name][:])
            assert before[name].dtype == after[name].dtype
            assert before[name].compression == after[name].compression
            assert dict(before[name].attrs) == dict(after[name].attrs)
        for name, value in before.attrs.items():
            np.testing.assert_array_equal(value, after.attrs[name])
        np.testing.assert_array_equal(after.attrs["dx_dy_dz"], [0.001] * 3)
        assert after.attrs["MaterialDatabase"] == database.stem


def test_all_native_poles_density_and_file_order_survive(tmp_path):
    text = (
        "## Material positions are file-local, not current builtin indices.\n"
        "#material: 1 INF 1 0 pec\n"
        "#material: 1 0 1 0 free_space\n"
        "#material: 4 0.01 1 0 debye\n"
        "#material: 5 0.02 1 0 lorentz\n"
        "#material: 6 0.03 1 0 drude\n"
        "#add_dispersion_debye: 2 2.5 1e-11 1.5 2e-9 debye\n"
        "#add_dispersion_lorentz: 2 3 1e9 1e8 4 2e9 2e8 lorentz\n"
        "#add_dispersion_drude: 2 1e9 1e8 2e9 2e8 drude\n"
        "#material_density: 1000 debye lorentz\n"
    )
    source, materials = pair(tmp_path, text)
    converted, database = convert_geometry(source, materials)
    with h5py.File(converted) as data:
        keys = list(data["material_keys"].asstr()[:])
    specs = [load_material_spec(database.stem, key, search_directory=tmp_path) for key in keys]
    assert [s.name for s in specs] == ["pec", "free_space", "debye", "lorentz", "drude"]
    assert specs[0].electric_conductivity == float("inf")
    assert specs[2].mass_density == specs[3].mass_density == 1000
    assert specs[4].mass_density is None
    assert [p["relaxation_time_s"] for p in specs[2].poles] == [1e-11, 2e-9]
    assert [p["relative_permittivity_difference"] for p in specs[2].poles] == [2.5, 1.5]
    assert [p["resonance_frequency_hz"] for p in specs[3].poles] == [1e9, 2e9]
    assert [p["damping_coefficient_per_s"] for p in specs[3].poles] == [1e8, 2e8]
    assert [p["relative_permittivity_difference"] for p in specs[3].poles] == [3, 4]
    assert [p["plasma_frequency_hz"] for p in specs[4].poles] == [1e9, 2e9]
    assert [p["collision_frequency_per_s"] for p in specs[4].poles] == [1e8, 2e8]


@pytest.mark.parametrize(
    "extra",
    (
        "#python: print('must not execute')\n",
        "#material: 9 0 1 0 tissue\n",
        "#add_dispersion_debye: 1 2 1e-9 absent\n",
        "#material_density: 1000 absent\n",
        "#material_density: 0 tissue\n",
        "#add_dispersion_lorentz: 2 1 1e9 tissue\n",
    ),
)
def test_invalid_text_stops_without_outputs(tmp_path, extra):
    source, materials = pair(tmp_path, "#material: 4 0 1 0 tissue\n" + extra)
    with pytest.raises(ValueError):
        convert_geometry(source, materials)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.h5", "legacy.txt"]


def test_utf8_bom_and_windows_line_endings_are_accepted(tmp_path):
    source, materials = pair(tmp_path)
    materials.write_bytes(b"\xef\xbb\xbf## Windows text file\r\n#material: 4 0.01 1 0 tissue\r\n")
    _, database = convert_geometry(source, materials)
    assert (
        load_material_spec(
            database.stem, "material_000_tissue", search_directory=tmp_path
        ).relative_permittivity
        == 4
    )


@pytest.mark.parametrize("external", ("link", "storage"))
def test_nonportable_hdf5_dependencies_are_rejected(tmp_path, external):
    source, materials = pair(tmp_path)
    with h5py.File(source, "r+") as data:
        if external == "link":
            data["external"] = h5py.ExternalLink("another.h5", "/data")
        else:
            data.create_dataset(
                "external", shape=(4, 4, 4), dtype="i2", external=[("mesh.raw", 0, 128)]
            )
    with pytest.raises(ValueError, match="self-contained"):
        convert_geometry(source, materials)
    assert not (tmp_path / "legacy_converted.h5").exists()


@pytest.mark.parametrize(
    "defect", ("negative", "float", "missing_material", "spacing", "partial", "tag")
)
def test_malformed_geometry_stops_without_outputs(tmp_path, defect):
    source, materials = pair(tmp_path)
    with h5py.File(source, "r+") as data:
        if defect == "negative":
            data["data"][0] = -2
        elif defect == "float":
            del data["data"]
            data["data"] = np.zeros((4, 4, 4), float)
        elif defect == "missing_material":
            data["data"][0] = 1
        elif defect == "spacing":
            data.attrs["dx, dy, dz"] = (0.001, 0, 0.001)
        elif defect == "partial":
            data["rigidE"] = np.zeros((12, 4, 4, 4), np.int8)
        else:
            data["tag_data"][0] = 2
    with pytest.raises(ValueError):
        convert_geometry(source, materials)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.h5", "legacy.txt"]


@pytest.mark.parametrize("values", ("nan 0 1 0", "4 -1 1 0", "0.5 0 1 0"))
def test_material_schema_is_checked_before_writing(tmp_path, values):
    source, materials = pair(tmp_path, f"#material: {values} tissue\n")
    with pytest.raises(ValueError):
        convert_geometry(source, materials)
    assert not (tmp_path / "legacy_converted.h5").exists()
    assert not (tmp_path / "legacy_materials.json").exists()


def test_overwrite_reserved_names_and_separated_outputs_are_rejected(tmp_path):
    source, materials = pair(tmp_path)
    with pytest.raises(ValueError, match="source file"):
        convert_geometry(source, materials, output_geometry=source)
    with pytest.raises(ValueError, match="source file"):
        convert_geometry(source, materials, output_geometry=materials)
    with pytest.raises(ValueError, match="same directory"):
        convert_geometry(source, materials, output_database=tmp_path / "elsewhere" / "db.json")
    with pytest.raises(ValueError, match="different files"):
        convert_geometry(
            source,
            materials,
            output_geometry=tmp_path / "pair.json",
            output_database=tmp_path / "pair.json",
        )
    with pytest.raises(ValueError, match="reserved"):
        convert_geometry(source, materials, output_database=tmp_path / "fundamental.json")
    converted, database = convert_geometry(source, materials)
    before = converted.read_bytes(), database.read_bytes()
    with pytest.raises(FileExistsError):
        convert_geometry(source, materials)
    assert (converted.read_bytes(), database.read_bytes()) == before


def test_output_geometry_option_keeps_the_default_json_beside_it(tmp_path):
    source, materials = pair(tmp_path)
    directory = tmp_path / "elsewhere"
    directory.mkdir()
    converted, database = convert_geometry(
        source, materials, output_geometry=directory / "model.h5"
    )
    assert converted.parent == database.parent == directory
    assert (
        load_material_spec(
            database.stem, "material_000_tissue", search_directory=directory
        ).relative_permittivity
        == 4
    )


def test_failed_second_write_cleans_up_both_new_outputs(monkeypatch, tmp_path):
    source, materials = pair(tmp_path)
    original = converter.write_database

    def fail(path, document):
        if path.parent == tmp_path:
            raise OSError("injected write failure")
        return original(path, document)

    monkeypatch.setattr(converter, "write_database", fail)
    with pytest.raises(OSError, match="injected"):
        convert_geometry(source, materials)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.h5", "legacy.txt"]


def test_conversion_does_not_overwrite_an_output_created_during_validation(monkeypatch, tmp_path):
    source, materials = pair(tmp_path)
    original = converter.validate_material_database
    target = tmp_path / "legacy_materials.json"

    def concurrent_output(*args, **kwargs):
        result = original(*args, **kwargs)
        target.write_text("another process owns this file")
        return result

    monkeypatch.setattr(converter, "validate_material_database", concurrent_output)
    with pytest.raises(FileExistsError):
        convert_geometry(source, materials)
    assert target.read_text() == "another process owns this file"
    assert not (tmp_path / "legacy_converted.h5").exists()


@pytest.mark.parametrize("model", ("can", "PMA", "PMN", "TS50"))
@pytest.mark.parametrize("spacing", ("1x1x1", "2x2x2"))
def test_shipped_legacy_geometry_files_convert(tmp_path, model, spacing):
    directory = Path(__file__).resolve().parents[2] / "gprMax" / "toolboxes" / "LandmineModels" / "legacy"
    source = directory / f"{model}_{spacing}.h5"
    converted, _ = convert_geometry(
        source,
        directory / f"{model}_materials.txt",
        output_geometry=tmp_path / "converted.h5",
        output_database=tmp_path / "converted_materials.json",
    )
    with h5py.File(source) as before, h5py.File(converted) as after:
        np.testing.assert_array_equal(before["data"][:], after["data"][:])


def test_blocks_are_bounded_even_for_a_large_contiguous_plane(tmp_path):
    with h5py.File(tmp_path / "blocks.h5", "w") as data:
        array = data.create_dataset("data", data=np.arange(420, dtype=np.uint64).reshape(2, 3, 70))
        blocks = list(_blocks(array, max_bytes=128))
        assert all(block.nbytes <= 128 for block in blocks)
        assert sum(block.size for block in blocks) == array.size
        assert sum(int(block.sum()) for block in blocks) == int(array[:].sum())


def test_python_legacy_argument_is_rejected_with_a_migration_command():
    with pytest.raises(ValueError, match="convert-geometry old.h5 old.txt"):
        gprMax.GeometryObjectsRead(p1=(0, 0, 0), geofile="old.h5", matfile="old.txt")


def test_unkeyed_hdf5_cannot_bypass_conversion(tmp_path):
    source, _ = pair(tmp_path)
    command = gprMax.GeometryObjectsRead(p1=(0, 0, 0), geofile=source, material_database="local")
    with pytest.raises(ValueError, match="convert-geometry"):
        command._build_database_material_map(None, source, "local")


def test_conversion_cli_prints_the_new_import_arguments(tmp_path):
    source, materials = pair(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gprMax.toolboxes.MaterialDatabase",
            "convert-geometry",
            str(source),
            str(materials),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "material_database='legacy_materials'" in result.stdout
    assert "Original files were not changed" in result.stdout


def _base_scene():
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.012,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.TimeWindow(iterations=80),
        gprMax.OMPThreads(n=1),
    ):
        scene.add(obj)
    return scene


@pytest.mark.integration
@pytest.mark.parametrize("existing_tau", (None, 1e-11, 2e-11))
def test_migrated_poles_and_density_survive_name_collisions_and_repeated_imports(
    tmp_path, monkeypatch, existing_tau
):
    source, materials = pair(
        tmp_path,
        "#material: 4 0.01 1 0 tissue\n#add_dispersion_debye: 1 2 1e-11 tissue\n#material_density: 1000 tissue\n",
    )
    geometry, database = convert_geometry(source, materials)
    scene = _base_scene()
    if existing_tau is not None:
        scene.add(gprMax.Material(er=4, se=0.01, mr=1, sm=0, id="tissue"))
        scene.add(
            gprMax.AddDebyeDispersion(
                poles=1, er_delta=[2], tau=[existing_tau], material_ids=["tissue"]
            )
        )
        scene.add(gprMax.MaterialDensity(density=1000, material_ids=["tissue"]))
    for point in ((0.002,) * 3, (0.006,) * 3):
        scene.add(
            gprMax.GeometryObjectsRead(p1=point, geofile=geometry, material_database=database.stem)
        )
    captured = []
    original = model_module.Model.build

    def build(model):
        original(model)
        captured.append(model.G)

    monkeypatch.setattr(model_module.Model, "build", build)
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "imported",
        hide_progress_bars=True,
        log_level=30,
    )
    grid = captured[0]
    first = grid.materials[grid.solid[3, 3, 3]]
    second = grid.materials[grid.solid[7, 7, 7]]
    assert first is second
    assert first.er == 4 and first.se == 0.01 and first.mass_density == 1000
    assert first.tau == [1e-11]
    assert first.deltaer == [2]
    assert grid.geometry_tag_registry.names[grid.geometry_tag_map.data[3, 3, 3]] == "organ"
    if existing_tau == 1e-11:
        assert first.ID == "tissue"
    elif existing_tau == 2e-11:
        assert first.ID != "tissue"


@pytest.mark.integration
@pytest.mark.parametrize("precision", ("single", "double"))
def test_converted_voxels_give_the_same_fields_as_the_direct_model(tmp_path, precision):
    source, materials = pair(tmp_path)
    geometry, database = convert_geometry(source, materials)
    fields = []
    for imported in (False, True):
        scene = _base_scene()
        if imported:
            scene.add(
                gprMax.GeometryObjectsRead(
                    p1=(0.002,) * 3, geofile=geometry, material_database=database.stem
                )
            )
        else:
            scene.add(gprMax.Material(er=4, se=0.01, mr=1, sm=0, id="tissue"))
            scene.add(
                gprMax.Box(p1=(0.002,) * 3, p2=(0.006,) * 3, material_id="tissue", averaging=False)
            )
        scene.add(gprMax.Waveform(wave_type="impulse", amp=0.001, freq=1e9, id="pulse"))
        scene.add(
            gprMax.HertzianDipole(p1=(0.003, 0.004, 0.004), polarisation="z", waveform_id="pulse")
        )
        scene.add(gprMax.Rx(p1=(0.007, 0.004, 0.004)))
        output = tmp_path / str(imported)
        gprMax.run(
            scenes=[scene],
            outputfile=output,
            cpu_precision=precision,
            hide_progress_bars=True,
            log_level=30,
        )
        with h5py.File(output.with_suffix(".h5")) as result:
            fields.append({name: dataset[:] for name, dataset in result["rxs/rx1"].items()})
    assert np.linalg.norm(fields[0]["Ez"]) > 0
    for component in fields[0]:
        np.testing.assert_array_equal(fields[0][component], fields[1][component])
