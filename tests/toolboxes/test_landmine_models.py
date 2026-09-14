"""Check the shipped target data and run every model and resolution in gprMax."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.model as model_module
from gprMax.material_database import load_material_spec
from gprMax.toolboxes.LandmineModels.examples import free_space
from gprMax.toolboxes.MaterialDatabase.convert_geometry import parse_legacy_materials

TOOLBOX = Path(free_space.__file__).resolve().parents[1]
CASES = [(model, resolution) for model in free_space.MODEL_NAMES for resolution in (1, 2)]


@pytest.mark.unit
@pytest.mark.parametrize("model,resolution", CASES)
def test_current_files_preserve_original_geometry_and_materials(model, resolution):
    name = f"{model}_{resolution}x{resolution}x{resolution}.h5"
    legacy, current = TOOLBOX / "legacy" / name, TOOLBOX / name
    entries = parse_legacy_materials(TOOLBOX / "legacy" / f"{model}_materials.txt")
    with h5py.File(legacy) as before, h5py.File(current) as after:
        np.testing.assert_array_equal(before["data"][...], after["data"][...])
        assert before["data"].dtype == after["data"].dtype
        np.testing.assert_array_equal(before.attrs["dx_dy_dz"], after.attrs["dx_dy_dz"])
        np.testing.assert_array_equal(after.attrs["dx_dy_dz"], [resolution * 0.001] * 3)
        # File-local voxel IDs refer to this ordered table of plain material names.
        assert list(after["material_keys"].asstr()[...]) == [entry["name"] for _, entry in entries]
        database = after.attrs["MaterialDatabase"]
    for _, entry in entries:
        material = load_material_spec(database, entry["name"], search_directory=current.parent)
        base = entry["base"]
        assert material.relative_permittivity == base["relative_permittivity"]
        assert material.electric_conductivity == float(base["electric_conductivity_s_per_m"])
        assert material.relative_permeability == base["relative_permeability"]
        assert material.magnetic_conductivity == float(base["magnetic_conductivity_s_per_m"])


@pytest.mark.integration
@pytest.mark.parametrize("model,resolution", CASES)
def test_imported_materials_and_transparent_voxels_run_with_a_source(
    tmp_path, monkeypatch, model, resolution
):
    # A background distinct from stored air makes incorrect -1 handling visible.
    background_er = 2.25
    scene = free_space.build_model(model, resolution, background_er=background_er)
    with h5py.File(TOOLBOX / "legacy" / f"{model}_{resolution}x{resolution}x{resolution}.h5") as h:
        legacy = h["data"][...]
    entries = parse_legacy_materials(TOOLBOX / "legacy" / f"{model}_materials.txt")
    audit = {}
    original_build = model_module.Model.build

    def check_built_grid(instance):
        original_build(instance)
        grid = instance.G
        start = free_space.PADDING_CELLS
        region = tuple(slice(start, start + size) for size in legacy.shape)
        actual_ids = grid.solid[region]
        # Compare physical values, not file-local IDs against solver-global IDs.
        columns = [
            ("er", "relative_permittivity", background_er),
            ("se", "electric_conductivity_s_per_m", 0.0),
            ("mr", "relative_permeability", 1.0),
            ("sm", "magnetic_conductivity_s_per_m", 0.0),
        ]
        for attribute, key, background in columns:
            expected = np.full(legacy.shape, background, dtype=float)
            for local_id, (_, entry) in enumerate(entries):
                expected[legacy == local_id] = float(entry["base"][key])
            properties = np.array([getattr(material, attribute) for material in grid.materials])
            np.testing.assert_array_equal(properties[actual_ids], expected)
        audit.update(
            shape_cells=list(legacy.shape),
            transparent_cells=int(np.count_nonzero(legacy == -1)),
            all_voxel_materials_verified=True,
        )

    monkeypatch.setattr(model_module.Model, "build", check_built_grid)
    output = tmp_path / "target"
    gprMax.run(scenes=[scene], outputfile=output, hide_progress_bars=True, log_level=30)
    assert audit["all_voxel_materials_verified"]
    with h5py.File(output.with_suffix(".h5")) as h:
        fields = {name: dataset[...] for name, dataset in h["rxs/rx1"].items()}
        assert all(np.isfinite(values).all() for values in fields.values())
        assert np.linalg.norm(fields["Ex"]) > 0
        audit.update(
            model=model,
            resolution_mm=resolution,
            background_er=background_er,
            iterations=int(h.attrs["Iterations"]),
            peak_ex_v_per_m=float(np.max(np.abs(fields["Ex"]))),
            finite_receiver_fields=True,
        )
    (tmp_path / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
