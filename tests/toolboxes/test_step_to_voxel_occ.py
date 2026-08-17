import json
from pathlib import Path

import h5py
import pytest

pytest.importorskip("OCC", reason="optional pythonocc-core dependency is not installed")

from gprMax.material_database import load_material_spec
from toolboxes.STEPtoVoxel import (
    ConversionConfig,
    convert_step,
    inspect_step,
    write_material_template,
)


@pytest.mark.integration
def test_probe_fed_patch_step_conversion(tmp_path):
    example = Path(__file__).parents[2] / "toolboxes" / "STEPtoVoxel" / "examples" / "patch_antenna"
    step_file = example / "PROBE_FED.stp"
    parts = inspect_step(
        step_file,
        ConversionConfig(voxel_size=(0.2e-3, 0.2e-3, 0.2e-3)),
    )
    assert [part.name for part in parts] == [
        "PATCH",
        "SUB",
        "GROUND",
        "INNER",
        "DIE",
        "OUTER",
        "port1",
    ]
    assert all(part.name_confidence == "exact" for part in parts)
    assert parts[0].name_source == "MANIFOLD_SOLID_BREP"
    assert parts[-1].name_source == "SHELL_BASED_SURFACE_MODEL"
    assert parts[0].cad["topology_counts"]["faces"] > 0
    assert len(parts[0].cad["principal_moments_m5"]) == 3

    generated_materials = tmp_path / "generated_materials.csv"
    write_material_template(step_file, generated_materials, config=ConversionConfig())
    header = generated_materials.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("group_id,group_confidence,similar_group,part_count,part_names")

    result = convert_step(
        step_file,
        example / "materials.csv",
        tmp_path,
        ConversionConfig(voxel_size=(0.2e-3, 0.2e-3, 0.2e-3)),
        write_vtk=False,
    )

    assert result.component_cell_counts["PATCH"] > 0
    assert "port1" not in result.component_cell_counts
    assert result.vtk_file is None
    assert result.reference_geometry_cad_file is None
    with h5py.File(result.geometry_file) as f:
        assert f["data"].dtype == "int16"
        assert tuple(f.attrs["dx_dy_dz"]) == (0.2e-3, 0.2e-3, 0.2e-3)
        assert f.attrs["MaterialDatabase"] == "materials"
        material_keys = [value.decode() for value in f["material_keys"][:]]
    database = json.loads(result.materials_file.read_text(encoding="utf-8"))
    assert material_keys == list(database["materials"])
    assert database["schema"] == "gprMax-material-database"
    assert "Infinity" not in result.materials_file.read_text(encoding="utf-8")

    specs = {
        key: load_material_spec("materials", key, search_directory=tmp_path)
        for key in material_keys
    }
    pec_specs = [spec for spec in specs.values() if spec.electric_conductivity == float("inf")]
    assert len(pec_specs) == 1
    assert pec_specs[0].metadata["original_id"] == "pec"
    assert (
        specs[next(key for key in material_keys if key.endswith("substrate"))].relative_permittivity
        == 2.2
    )

    # The generated JSON is the user's runtime database. A repeated
    # conversion must not discard properties or metadata edited after the
    # initial STEP conversion.
    first_key = material_keys[0]
    database["materials"][first_key]["metadata"]["user_note"] = "preserve me"
    result.materials_file.write_text(json.dumps(database), encoding="utf-8")
    repeated = convert_step(
        step_file,
        example / "materials.csv",
        tmp_path,
        ConversionConfig(voxel_size=(0.2e-3, 0.2e-3, 0.2e-3)),
        write_vtk=False,
    )
    preserved = json.loads(repeated.materials_file.read_text(encoding="utf-8"))
    assert preserved["materials"][first_key]["metadata"]["user_note"] == "preserve me"

    markers = json.loads(result.markers_file.read_text(encoding="utf-8"))["markers"]
    assert len(markers) == 1
    assert markers[0]["name"] == "port1"
    assert markers[0]["kind"] == "port"
    assert markers[0]["axis"] == "z"

    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert manifest["sweep_axis"] == "auto"
    assert manifest["resolved_sweep_axis"] == "z"
    assert manifest["supersample"] == 1
    assert manifest["material_grouping"] == "exact"
    assert manifest["grouping_relative_tolerance"] == pytest.approx(0.01)
