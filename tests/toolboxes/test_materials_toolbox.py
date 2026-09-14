"""Verify the shipped Eccosorb fits and the documented material-loading routes."""

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.material_database import load_material_spec

TOOLBOX = Path(__file__).resolve().parents[2] / "gprMax" / "toolboxes" / "Materials"
GRADES = (14, 16, 18, 20, 22, 26, 28, 30)


def original_commands(key):
    """Read the original input lines independently of the JSON converter."""
    return [
        line
        for line in (TOOLBOX / "legacy/eccosorb.txt").read_text().splitlines()
        if line.startswith(("#material:", "#add_dispersion_debye:")) and line.split()[-1] == key
    ]


@pytest.mark.unit
@pytest.mark.parametrize("grade", GRADES)
def test_json_retains_all_original_material_and_pole_values(grade):
    key = f"eccosorb_ls{grade}"
    base_line, pole_line = original_commands(key)
    original_base = [float(value) for value in base_line.split()[1:-1]]
    original_poles = np.array([float(value) for value in pole_line.split()[2:-1]]).reshape(3, 2)
    spec = load_material_spec("eccosorb", key, search_directory=TOOLBOX)

    assert spec.model == "debye"
    np.testing.assert_array_equal(
        [
            spec.relative_permittivity,
            spec.electric_conductivity,
            spec.relative_permeability,
            spec.magnetic_conductivity,
        ],
        original_base,
    )
    np.testing.assert_array_equal(
        [
            (pole["relative_permittivity_difference"], pole["relaxation_time_s"])
            for pole in spec.poles
        ],
        original_poles,
    )


@pytest.mark.integration
@pytest.mark.parametrize("grade", GRADES)
def test_json_slab_fields_match_original_commands(tmp_path, grade):
    """A real solve checks pole translation and material lookup in geometry."""
    key = f"eccosorb_ls{grade}"
    shutil.copy2(TOOLBOX / "eccosorb.json", tmp_path / "eccosorb.json")
    current = (TOOLBOX / "eccosorb_slab.in").read_text().replace("eccosorb_ls22", key)
    legacy = current.replace(
        f"#material_from_database: eccosorb {key}", "\n".join(original_commands(key))
    )
    outputs = {}
    for variant, text in (("current", current), ("legacy", legacy)):
        inputfile = tmp_path / f"{variant}.in"
        inputfile.write_text(text)
        output = tmp_path / variant
        gprMax.run(inputfile=inputfile, outputfile=output, hide_progress_bars=True, log_level=30)
        outputs[variant] = output.with_suffix(".h5")

    with h5py.File(outputs["current"]) as current, h5py.File(outputs["legacy"]) as legacy:
        assert current.attrs["dt"] == legacy.attrs["dt"]
        for component in ("Ex", "Ey", "Ez"):
            field = current[f"rxs/rx1/{component}"][...]
            assert np.isfinite(field).all()
            np.testing.assert_array_equal(field, legacy[f"rxs/rx1/{component}"][...])
        assert np.linalg.norm(current["rxs/rx1/Ex"][...]) > 0
        # Only the requested grade is loaded, and its source is recorded.
        provenance = current["material_database_provenance"]
        assert len(provenance) == 1
        record = next(iter(provenance.values()))
        assert record.attrs["DatabaseID"] == "eccosorb"
        assert record.attrs["EntryKey"] == key


@pytest.mark.integration
def test_python_api_loads_the_database_from_the_working_directory(tmp_path, monkeypatch):
    """Output location must not affect the documented direct-API lookup rule."""
    shutil.copy2(TOOLBOX / "eccosorb.json", tmp_path / "eccosorb.json")
    monkeypatch.chdir(tmp_path)
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.048, 0.048, 0.048)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=2e-9))
    scene.add(gprMax.PMLThickness(thickness=6))
    scene.add(gprMax.OMPThreads(n=2))
    scene.add(gprMax.MaterialFromDatabase(database="eccosorb", material="eccosorb_ls22"))
    scene.add(
        gprMax.Box(
            p1=(0.010, 0.010, 0.022),
            p2=(0.038, 0.038, 0.028),
            material_id="eccosorb_ls22",
            averaging=False,
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=3e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(p1=(0.024, 0.024, 0.014), polarisation="x", waveform_id="pulse")
    )
    scene.add(gprMax.Rx(p1=(0.024, 0.024, 0.035), id="probe", outputs=["Ex", "Ey", "Ez"]))
    (tmp_path / "results").mkdir()
    output = tmp_path / "results/slab"
    gprMax.run(scenes=[scene], outputfile=output, hide_progress_bars=True, log_level=30)
    with h5py.File(output.with_suffix(".h5")) as handle:
        field = handle["rxs/rx1/Ex"][...]
        assert np.isfinite(field).all() and np.linalg.norm(field) > 0
        assert len(handle["material_database_provenance"]) == 1
