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
from gprMax.materials import DispersiveMaterial, Material
from gprMax.user_objects.cmds_multiuse import (
    AddDebyeDispersion,
    MaterialDensity,
    MaterialFromDatabase,
)
from toolboxes.MaterialDatabase import convert_geometry


def _database(materials, database_id="local"):
    return {
        "schema": "gprMax-material-database",
        "schema_version": 1,
        "database": {"id": database_id, "name": "Test", "version": "1.2.3"},
        "materials": materials,
    }


def _constant(er=4.0, mass_density=None):
    entry = {
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
    if mass_density is not None:
        entry["mass_density_kg_per_m3"] = mass_density
    return entry


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


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema", "wrong-schema"), ("schema_version", 2)),
)
def test_local_database_rejects_unsupported_schema(tmp_path, field, value):
    document = _database({"dielectric": _constant()})
    document[field] = value
    (tmp_path / "local.json").write_text(json.dumps(document))

    with pytest.raises(ValueError, match="must use.*schema version 1"):
        load_material_spec("local", "dielectric", search_directory=tmp_path)


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


def test_database_mass_density_builds_compares_and_round_trips(tmp_path, fake_grid):
    entry = _constant(mass_density=1040.0)
    (tmp_path / "local.json").write_text(json.dumps(_database({"tissue": entry})))
    spec = load_material_spec("local", "tissue", search_directory=tmp_path)
    material = build_material_from_spec(fake_grid(dt=1e-12), spec, "tissue")

    assert spec.mass_density == 1040.0
    assert material.mass_density == 1040.0
    assert material_matches_spec(material, spec)
    assert material_to_database_entry(material)["mass_density_kg_per_m3"] == 1040.0

    material.mass_density = 1030.0
    assert not material_matches_spec(material, spec)


@pytest.mark.parametrize("density", (0, -1, float("inf"), float("nan")))
def test_database_rejects_invalid_mass_density(tmp_path, density):
    (tmp_path / "local.json").write_text(
        json.dumps(_database({"bad": _constant(mass_density=density)}))
    )
    with pytest.raises(ValueError, match="mass_density_kg_per_m3"):
        load_material_spec("local", "bad", search_directory=tmp_path)


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


@pytest.mark.parametrize("damping_ratio", (1.0, 1.01))
def test_lorentz_database_rejects_critical_and_overdamped_poles(tmp_path, fake_grid, damping_ratio):
    frequency = 100e9
    entry = _constant(er=2.0)
    entry.update(
        {
            "model": "lorentz",
            "poles": [
                {
                    "relative_permittivity_difference": 3.5,
                    "resonance_frequency_hz": frequency,
                    "damping_coefficient_per_s": damping_ratio * 2.0 * np.pi * frequency,
                }
            ],
        }
    )
    (tmp_path / "local.json").write_text(json.dumps(_database({"critical": entry})))
    spec = load_material_spec("local", "critical", search_directory=tmp_path)

    with pytest.raises(ValueError, match=r"damping coefficient.*2 \* pi"):
        build_material_from_spec(fake_grid(dt=1e-12), spec, "critical")


@pytest.mark.parametrize(
    "model, frequency_field, rate_field",
    (
        ("lorentz", "resonance_frequency_hz", "damping_coefficient_per_s"),
        ("drude", "plasma_frequency_hz", "collision_frequency_per_s"),
    ),
)
def test_database_pole_frequency_limit_uses_hertz_not_angular_frequency(
    tmp_path, fake_grid, model, frequency_field, rate_field
):
    dt = 1e-12
    pole = {frequency_field: 1.01 / dt, rate_field: 1e9}
    if model == "lorentz":
        pole["relative_permittivity_difference"] = 3.5
    entry = _constant(er=2.0)
    entry.update({"model": model, "poles": [pole]})
    (tmp_path / "local.json").write_text(json.dumps(_database({"too_fast": entry})))
    spec = load_material_spec("local", "too_fast", search_directory=tmp_path)

    with pytest.raises(ValueError, match=r"frequency must be below 1 / dt"):
        build_material_from_spec(fake_grid(dt=dt), spec, "too_fast")


@pytest.mark.parametrize(
    "model, pole",
    (
        (
            "lorentz",
            {
                "relative_permittivity_difference": 3.5,
                "resonance_frequency_hz": 0.99e12,
                "damping_coefficient_per_s": 1e9,
            },
        ),
        (
            "drude",
            {
                "plasma_frequency_hz": 0.99e12,
                "collision_frequency_per_s": 1e9,
            },
        ),
    ),
)
def test_database_pole_frequency_just_below_timestep_limit_is_accepted(
    tmp_path, fake_grid, model, pole
):
    entry = _constant(er=2.0)
    entry.update({"model": model, "poles": [pole]})
    (tmp_path / "local.json").write_text(json.dumps(_database({"valid": entry})))
    spec = load_material_spec("local", "valid", search_directory=tmp_path)

    material = build_material_from_spec(fake_grid(dt=1e-12), spec, "valid")

    assert material.type == model


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


def test_material_density_hash_command_is_positional():
    objects = get_user_objects(
        ["#material_density: 1040 brain white_matter\n"],
        checkessential=False,
    )
    assert len(objects) == 1
    assert isinstance(objects[0], MaterialDensity)
    assert objects[0].kwargs == {
        "density": 1040.0,
        "material_ids": ["brain", "white_matter"],
    }


def test_material_density_assigns_without_changing_em_properties(fake_grid):
    grid = fake_grid(dt=1e-12)
    material = Material(0, "brain")
    material.er = 45.0
    material.se = 0.8
    grid.materials.append(material)

    MaterialDensity(density=1040.0, material_ids=["brain"]).build(grid)

    assert material.mass_density == 1040.0
    assert material.er == 45.0
    assert material.se == 0.8


@pytest.mark.parametrize("density", (0, -1, float("inf"), float("nan")))
def test_material_density_rejects_non_positive_or_non_finite_values(fake_grid, density):
    grid = fake_grid(dt=1e-12)
    grid.materials.append(Material(0, "brain"))
    with pytest.raises(ValueError, match="finite and greater than zero"):
        MaterialDensity(density=density, material_ids=["brain"]).build(grid)


def test_material_density_rejects_unknown_material(fake_grid):
    with pytest.raises(ValueError, match="do not exist"):
        MaterialDensity(density=1040.0, material_ids=["brain"]).build(fake_grid(dt=1e-12))


def test_dispersion_conversion_preserves_existing_mass_density(fake_grid):
    grid = fake_grid(dt=1e-12)
    material = Material(0, "brain")
    material.mass_density = 1040.0
    grid.materials.append(material)

    AddDebyeDispersion(
        poles=1,
        er_delta=[20.0],
        tau=[10e-12],
        material_ids=["brain"],
    ).build(grid)

    assert isinstance(grid.materials[0], DispersiveMaterial)
    assert grid.materials[0].mass_density == 1040.0


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
