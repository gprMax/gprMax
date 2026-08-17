"""Material database schema, resolution, translation, and migration tests."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from gprMax.hash_cmds_file import get_user_objects
from gprMax.material_database import (
    build_material_from_spec,
    load_material_spec,
    material_matches_spec,
    material_to_database_entry,
    validate_material_database,
)
from gprMax.materials import DispersiveMaterial
from gprMax.user_objects.cmds_multiuse import MaterialFromDatabase
from toolboxes.MaterialDatabase import convert_geometry


def _database(materials, database_id="local"):
    return {
        "schema": "gprMax-material-database",
        "schema_version": 1,
        "database": {"id": database_id, "name": "Test", "version": "1.2.3"},
        "materials": materials,
    }


def _constant(er=4.0):
    return {
        "name": "Test dielectric",
        "model": "constant",
        "base": {
            "relative_permittivity": er,
            "electric_conductivity_s_per_m": 0.01,
            "relative_permeability": 1.0,
            "magnetic_conductivity_s_per_m": 0.0,
        },
        "metadata": {"validity": {"frequency_hz": [1e6, 1e9]}},
    }


def test_official_database_names_are_reserved(tmp_path):
    (tmp_path / "fundamental.json").write_text(json.dumps(_database({"fake": _constant()})))
    spec = load_material_spec("fundamental", "vacuum", search_directory=tmp_path)
    assert spec.source.official is True
    assert spec.key == "vacuum"


def test_local_filename_and_database_id_must_match(tmp_path):
    (tmp_path / "laboratory.json").write_text(
        json.dumps(_database({"dielectric": _constant()}, database_id="another_name"))
    )
    with pytest.raises(ValueError, match="declares ID 'another_name'.*selected as 'laboratory'"):
        load_material_spec("laboratory", "dielectric", search_directory=tmp_path)


def test_official_catalogues_validate_and_reserve_empty_subject_namespaces():
    fundamental = dict(validate_material_database("fundamental"))
    assert set(fundamental) == {"vacuum", "pec", "pmc"}

    # These names are reserved, but empirical entries require a separate
    # curation and provenance review before distribution with gprMax.
    for database in ("gpr", "antenna", "bioem"):
        assert validate_material_database(database) == ()


def test_local_constant_material_builds_with_provenance(tmp_path, fake_grid):
    (tmp_path / "local.json").write_text(json.dumps(_database({"dielectric": _constant()})))
    spec = load_material_spec("local", "dielectric", search_directory=tmp_path)
    grid = fake_grid(dt=1e-12)
    material = build_material_from_spec(grid, spec, "sample")
    assert material.er == 4.0
    assert material.se == 0.01
    assert material.database_provenance["database_version"] == "1.2.3"
    assert len(material.database_provenance["entry_sha256"]) == 64
    assert material_matches_spec(material, spec)

    material.er = 9.0
    assert not material_matches_spec(material, spec)


def test_debye_names_physical_quantities_explicitly(tmp_path, fake_grid):
    entry = _constant(er=2.0)
    entry.update(
        {
            "model": "debye",
            "poles": [
                {
                    "relative_permittivity_difference": 3.5,
                    "relaxation_time_s": 9e-12,
                }
            ],
        }
    )
    (tmp_path / "local.json").write_text(json.dumps(_database({"water_fit": entry})))
    spec = load_material_spec("local", "water_fit", search_directory=tmp_path)
    material = build_material_from_spec(fake_grid(dt=1e-13), spec, "water")
    assert isinstance(material, DispersiveMaterial)
    assert material.type == "debye"
    assert material.deltaer == [3.5]
    assert material.tau == [9e-12]


def test_general_poles_round_trip_exactly(tmp_path, fake_grid):
    entry = _constant(er=2.0)
    entry.update(
        {
            "model": "general",
            "inclusive_conductivity_s_per_m": 0.03,
            "poles": [{"w_per_s": [1.5, -2.5], "q_per_s": [-3.0, 4.0]}],
        }
    )
    (tmp_path / "local.json").write_text(json.dumps(_database({"mixed": entry})))
    spec = load_material_spec("local", "mixed", search_directory=tmp_path)
    material = build_material_from_spec(fake_grid(dt=1e-13), spec, "mixed_local")
    exported = material_to_database_entry(material)
    assert exported["model"] == "general"
    assert exported["poles"] == [{"w_per_s": [1.5, -2.5], "q_per_s": [-3.0, 4.0]}]
    assert exported["inclusive_conductivity_s_per_m"] == 0.03


def test_alias_cycles_are_rejected(tmp_path):
    document = _database(
        {
            "first": {"alias": {"material": "second"}},
            "second": {"alias": {"material": "first"}},
        }
    )
    (tmp_path / "local.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Circular material database alias"):
        load_material_spec("local", "first", search_directory=tmp_path)


def test_invalid_entry_reports_named_field(tmp_path):
    entry = _constant()
    entry["base"]["relative_permittivity"] = None
    (tmp_path / "local.json").write_text(json.dumps(_database({"bad": entry})))
    with pytest.raises(ValueError, match="relative_permittivity.*number"):
        load_material_spec("local", "bad", search_directory=tmp_path)


def test_unknown_constitutive_field_is_rejected(tmp_path):
    entry = _constant()
    entry["base"]["electric_conductivty_s_per_m"] = 0.02
    (tmp_path / "local.json").write_text(json.dumps(_database({"bad": entry})))
    with pytest.raises(ValueError, match="unknown fields.*electric_conductivty"):
        load_material_spec("local", "bad", search_directory=tmp_path)


def test_hash_command_accepts_optional_local_id():
    objects = get_user_objects(
        ["#material_from_database: antenna fr4_vendor grade_a\n"],
        checkessential=False,
    )
    assert len(objects) == 1
    assert isinstance(objects[0], MaterialFromDatabase)
    assert objects[0].kwargs == {
        "database": "antenna",
        "material": "fr4_vendor",
        "id": "grade_a",
    }


def test_legacy_conversion_copies_arrays_and_adds_keys(tmp_path):
    source = tmp_path / "legacy.h5"
    original = np.asarray([[[0, 1], [-1, 0]]], dtype=np.int16)
    with h5py.File(source, "w") as geometry:
        geometry.attrs["dx_dy_dz"] = (1e-3, 1e-3, 1e-3)
        geometry["data"] = original
        geometry["rigidE"] = np.zeros((12, *original.shape), dtype=np.int8)
        geometry["rigidH"] = np.zeros((6, *original.shape), dtype=np.int8)
        geometry["ID"] = np.full((6, 2, 3, 3), -1, dtype=np.int16)
    legacy = tmp_path / "materials.txt"
    legacy.write_text(
        "#material: 1 inf 1 0 pec\n"
        "#material: 4 0.01 1 0 soil-with.punctuation\n"
        "#add_dispersion_debye: 1 2.5 1e-9 soil-with.punctuation\n"
    )

    converted, database = convert_geometry(source, legacy)
    assert source.exists()
    with h5py.File(source, "r") as untouched:
        assert "material_keys" not in untouched
    with h5py.File(converted, "r") as result:
        np.testing.assert_array_equal(result["data"][:], original)
        keys = [value.decode() for value in result["material_keys"][:]]
    assert keys == ["material_000_pec", "material_001_soil-with.punctuation"]
    entries = json.loads(database.read_text())["materials"]
    assert entries[keys[1]]["model"] == "debye"
    assert entries[keys[1]]["metadata"]["original_id"] == "soil-with.punctuation"


pytestmark = pytest.mark.unit
