"""Test the user workflow from an ordinary copied Python file, outside the repo."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gprMax.toolboxes.Optimisation import Real, optimise, simulate


def valid_model(parameters):
    return None  # Signature-only test; must not be executed.


def valid_objective(parameters, output):
    return 0.0


def wrong_objective(parameters, runs, context):
    return 0.0


def test_user_signature_errors_are_explained_before_creating_a_campaign(tmp_path):
    with pytest.raises(TypeError, match=r"evaluate\(parameters, output\)"):
        optimise(
            parameters={"x": Real(1, 2)},
            model=valid_model,
            objective=wrong_objective,
            directory=tmp_path / "bad",
        )
    with pytest.raises(TypeError, match="top level"):
        optimise(
            parameters={"x": Real(1, 2)},
            model=lambda p: None,
            objective=valid_objective,
            directory=tmp_path / "bad",
        )
    with pytest.raises(ValueError, match="at least 6"):
        optimise(
            parameters={"x": Real(1, 2)},
            model=valid_model,
            objective=valid_objective,
            directory=tmp_path / "bad",
            optimiser="ga",
            evaluations=2,
        )
    assert not (tmp_path / "bad").exists()


def _environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(Path(p).resolve()) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


@pytest.mark.parametrize("target, expected_trials", [(None, 4), (1e6, 1)])
def test_copy_rename_edit_and_run_without_import_reference_strings(
    tmp_path, target, expected_trials
):
    pytest.importorskip("optuna")
    from gprMax.toolboxes.Optimisation import read_receiver
    from gprMax.toolboxes.Optimisation.examples import start_here

    source = Path(start_here.__file__).read_text()
    # These are the edits an ordinary user makes, with no changes in framework code.
    source = source.replace('"permittivity"', '"my_eps"').replace('"probe"', '"my_receiver"')
    source = source.replace("def build_model(", "def make_scene(").replace(
        "model=build_model,", "model=make_scene,"
    )
    source = source.replace("def evaluate(", "def score_signal(").replace(
        "objective=evaluate,", "objective=score_signal,"
    )
    source = source.replace("TARGET_PEAK_V_PER_M = 50.0", "TARGET_PEAK_V_PER_M = 20.0").replace(
        "evaluations=12,", "evaluations=4,"
    )
    source = source.replace("seed=7,", f"seed=7, target_value={target!r},")
    script = tmp_path / "my own experiment.py"
    script.write_text(source)
    run = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"Evaluation {expected_trials}:" in run.stdout and "Best parameters:" in run.stdout
    directory = tmp_path / "peak_field_results"
    result = json.loads((directory / "optimiser/result.json").read_text())
    assert len(result["trials"]) == result["run_attempts"] == expected_trials
    values = []
    for trial in result["trials"]:
        assert set(trial["parameters"]) == {"my_eps"}
        candidate = directory / "candidates" / trial["candidate_id"]
        request = json.loads((candidate / "default/attempt-0001/request.json").read_text())
        assert request["scenario"]["settings"]["model"]["file"] == str(script)
        assert request["scenario"]["settings"]["model"]["name"] == "make_scene"
        trace = read_receiver(candidate / "default/attempt-0001/output.h5", "my_receiver", "Ez")
        assert trial["value"] == pytest.approx(abs(float(np.max(np.abs(trace.values))) - 20))
        processing = json.loads((candidate / "processing/processing.json").read_text())
        assert processing["evaluator"]["settings"]["objective"]["name"] == "score_signal"
        values.append(trial["value"])
    if expected_trials > 1:
        assert len(set(values)) > 1, "Changed parameter values must reach the actual gprMax models"
    else:
        assert result["stop_reason"] == "target_reached"


def test_reference_example_can_be_copied_and_switched_to_a_population_optimiser(tmp_path):
    pytest.importorskip("pymoo")
    from gprMax.toolboxes.Optimisation.examples import waveform_matching

    original = Path(waveform_matching.__file__)
    shutil.copy2(original.with_name("reference_waveform.npz"), tmp_path / "reference_waveform.npz")
    source = (
        original.read_text()
        .replace('optimiser="rf",', 'optimiser="ga",\n        population_size=4,')
        .replace("evaluations=12,", "evaluations=4,")
    )
    source = source.replace("import Real, optimise", "import Real, optimise, LocalPool")
    source = source.replace(
        "seed=7,", "seed=7,\n        execution=LocalPool(workers=2, cpu_threads_per_worker=1),"
    )
    script = tmp_path / "my_match.py"
    script.write_text(source)
    run = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    directory = tmp_path / "waveform_results"
    result = json.loads((directory / "optimiser/result.json").read_text())
    assert len(result["trials"]) == 4 and all(t["status"] == "complete" for t in result["trials"])
    for trial in result["trials"]:
        with np.load(
            directory / "candidates" / trial["candidate_id"] / "processing/comparison.npz"
        ) as data:
            score = float(
                np.sqrt(np.mean(data["residual"] ** 2)) / np.sqrt(np.mean(data["reference"] ** 2))
            )
            assert score == pytest.approx(trial["value"])


def model_with_magnetic_output(parameters):
    import gprMax

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(p1=(0.014, 0.014, 0.014), polarisation="z", waveform_id="pulse")
    )
    scene.add(gprMax.Rx(p1=(0.020, 0.016, 0.014), id="magnetic_measurement", outputs=["Hy"]))
    return scene


def custom_hdf5_objective(parameters, output):
    import h5py

    with h5py.File(output.file, "r") as handle:
        return float(handle.attrs["dt"]) * parameters["x"]


def test_simulate_and_custom_processing_require_no_ez_or_s11_fields(tmp_path):
    pytest.importorskip("optuna")
    output = simulate(
        model=model_with_magnetic_output, parameters={"x": 1.5}, directory=tmp_path / "one"
    )
    assert output.file.is_file()
    assert custom_hdf5_objective({"x": 1.5}, output) > 0
    assert isinstance(output.datasets(), list)
    result = optimise(
        parameters={"x": Real(1, 2)},
        model=model_with_magnetic_output,
        objective=custom_hdf5_objective,
        directory=tmp_path / "search",
        evaluations=2,
        progress=False,
    )
    assert result.best_value > 0 and "x" in result.best_parameters
