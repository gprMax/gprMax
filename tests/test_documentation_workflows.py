"""Keep source/port documentation examples tied to executable public interfaces."""

import importlib.util
import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.waveforms import Waveform
from gprMax.user_objects.user_objects import UserObject


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/source"
EXAMPLES = ROOT / "examples/features/studies"


def _module(name):
    spec = importlib.util.spec_from_file_location(f"doc_example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hash_waveform_list_covers_builtin_types():
    reference = (DOCS / "input_hash_cmds.rst").read_text()
    section = reference.split("#waveform:\n", 1)[1].split("#excitation_file:\n", 1)[0]
    for name in Waveform.types:
        if name != "user":  # User samples have their own file/API interface.
            assert f"``{name}``" in section


def test_port_output_contracts_are_shared_by_both_references():
    api = (DOCS / "input_api.rst").read_text()
    hash_reference = (DOCS / "input_hash_cmds.rst").read_text()
    for filename in (
        "voltage_port_outputs",
        "transmission_line_outputs",
        "frill_port_outputs",
        "network_port_outputs",
        "eigenmode_port_outputs",
        "dipole_outputs",
        "plane_wave_outputs",
    ):
        directive = f".. include:: _includes/{filename}.rstinc"
        assert directive in api
        assert directive in hash_reference
        assert (DOCS / "_includes" / f"{filename}.rstinc").is_file()


def test_public_model_objects_have_reference_entries():
    api = (DOCS / "input_api.rst").read_text()
    hash_reference = (DOCS / "input_hash_cmds.rst").read_text()
    for name, cls in inspect.getmembers(gprMax, inspect.isclass):
        if not issubclass(cls, UserObject) or cls is UserObject:
            continue
        assert name in api, f"Missing public API reference: {name}"
        if name not in {"SubGridHSG", "PMLProps"}:  # API-only / deprecated wrapper.
            command = cls.hash.fget(None)
            assert command in hash_reference, f"Missing hash reference: {command}"


def test_study_hash_reference_does_not_nest_cases_under_codebook():
    reference = (DOCS / "input_hash_cmds.rst").read_text()
    study = reference.split("#study:\n", 1)[1].split("#array_codebook:\n", 1)[0]
    for name in ("gpr", "source", "port", "eigenmode", "plane_wave"):
        assert f"``{name}``" in study
    assert "network_excitation_1" in study


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["passive", "source", "port", "plane_wave"])
def test_documented_hash_and_api_models_agree_and_plot(tmp_path, kind):
    scene, study = _module("run_study").build_model(kind)
    common = dict(hide_progress_bars=True, log_level=30, cpu_precision="double")
    gprMax.run(scenes=[scene], study=study, outputfile=tmp_path / "api", **common)
    gprMax.run(inputfile=EXAMPLES / f"{kind}.in", outputfile=tmp_path / "hash", **common)

    suffixes = ("",) if kind == "passive" else ("1", "2")
    for suffix in suffixes:
        with h5py.File(tmp_path / f"api{suffix}.h5") as api, h5py.File(tmp_path / f"hash{suffix}.h5") as hashed:
            if kind == "plane_wave":
                for direction in ("back_x", "back_y"):
                    path = f"ntff/surface/frequency/band/far_field/{direction}/fields/rcs"
                    np.testing.assert_allclose(api[path][...], hashed[path][...], rtol=1e-12, atol=0)
                    assert np.isfinite(api[path][...]).all()
            else:
                for name in ("feed", "receive"):
                    for dataset in ("time", "Vtotal", "Vgenerator", "valid_S11"):
                        path = f"ports/{name}/{dataset}"
                        np.testing.assert_allclose(api[path][...], hashed[path][...], rtol=1e-12, atol=0)
                assert np.max(np.abs(api["ports/receive/Vtotal"][...])) > 0
                if kind == "passive":
                    assert not api["ports/receive/valid_S11"][...].any()
                    assert not api["ports/receive/Vgenerator"][...].any()

    if kind == "port":
        with h5py.File(tmp_path / "api_study.h5") as api, h5py.File(tmp_path / "hash_study.h5") as hashed:
            np.testing.assert_allclose(api["S"][...], hashed["S"][...], rtol=1e-12, atol=0)
            np.testing.assert_array_equal(api["valid_S"][...], hashed["valid_S"][...])
            assert api["valid_S"][...].any()
            assert api["S"].shape[1:] == (2, 2)
    else:
        assert not (tmp_path / "api_study.h5").exists()

    plotter = _module("plot_results")
    for prefix in ("api", "hash"):
        image = plotter.plot_results(kind, tmp_path / prefix)
        assert image.stat().st_size > 1000
