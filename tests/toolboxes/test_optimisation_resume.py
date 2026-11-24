"""Test resuming an interrupted optimise() run, and every refusal to resume.

The model and objectives are top-level functions in this file, as in a user's
script. Interruptions are made the way a user meets them: Ctrl+C raised while
the objective or a simulation runs, or a real kill of the launched script.
Each resumed run is compared with an uninterrupted run, not with saved numbers.
"""

import importlib.metadata
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import gprMax
from gprMax.toolboxes.Optimisation import LocalExecutor, LocalPool, Real, optimise, simulate
from gprMax.toolboxes.Optimisation.processing import PreparedEvaluator

PARAMETERS = {"permittivity": Real(2.0, 8.0)}
TARGET_PEAK = 250.0


def tiny_model(parameters):
    """A small dielectric block; each simulation takes a few seconds."""
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.040, 0.040, 0.040)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=0.6e-9))
    scene.add(gprMax.PMLThickness(thickness=4))
    scene.add(gprMax.Material(er=parameters["permittivity"], se=0, mr=1, sm=0, id="block"))
    scene.add(gprMax.Box(p1=(0.016, 0.012, 0.012), p2=(0.024, 0.028, 0.028), material_id="block"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=2e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(p1=(0.012, 0.020, 0.020), polarisation="z", waveform_id="pulse")
    )
    scene.add(gprMax.Rx(p1=(0.028, 0.020, 0.020), id="probe", outputs=["Ez"]))
    return scene


def other_model(parameters):
    """A different model function, used to show that a changed model is refused."""
    return tiny_model(parameters)


def peak_error(parameters, output):
    """Distance of the received peak from a target; it depends on the permittivity."""
    return abs(float(np.max(np.abs(output.receiver("probe", "Ez").values))) - TARGET_PEAK)


def other_objective(parameters, output):
    """A different objective function, used to show that a changed objective is refused."""
    return peak_error(parameters, output)


def failing_objective(parameters, output):
    """An objective that always fails, to check that a failed run is not continued."""
    raise RuntimeError("deliberate objective failure")


def _optimise(directory, **options):
    """Call optimise() with this file's model and objective, as a user script does."""
    arguments = {
        "parameters": PARAMETERS,
        "model": tiny_model,
        "objective": peak_error,
        "directory": directory,
    }
    return optimise(**{**arguments, **options})


@contextmanager
def _interrupt_objective_at(call):
    """Raise KeyboardInterrupt, as Ctrl+C would, when the objective is called the call-th time."""
    original = PreparedEvaluator.__call__
    calls = 0

    def interrupt(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == call:
            raise KeyboardInterrupt()
        return original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PreparedEvaluator, "__call__", interrupt)
        with pytest.raises(KeyboardInterrupt):
            yield


def _trials(result):
    """What an uninterrupted and a resumed run must agree on, trial by trial."""
    return [(t.candidate_id, dict(t.parameters), t.status, t.value) for t in result.trials]


def _attempts(directory):
    """Every simulation attempt directory of a campaign, relative to it."""
    return sorted(
        p.relative_to(directory) for p in Path(directory).glob("candidates/*/*/attempt-*")
    )


def _snapshot(directory):
    """The content of every file, to show that a refused call changed nothing."""
    directory = Path(directory)
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Resumed runs give the same answer as uninterrupted runs.


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, options, evaluations, stop_at",
    [
        # RF/TPE: stopped at the second member of the second batch of two.
        ("tpe", {"batch_size": 2}, 6, 4),
        ("rf", {"batch_size": 2}, 6, 4),
        # GA/PSO/DE: stopped part-way through the second population of four.
        ("ga", {"population_size": 4}, 8, 6),
        ("pso", {"population_size": 4}, 8, 6),
        ("de", {"population_size": 4}, 8, 6),
    ],
)
def test_resumed_run_matches_uninterrupted_run_for_every_optimiser(
    tmp_path, name, options, evaluations, stop_at
):
    pytest.importorskip({"rf": "skopt", "tpe": "optuna"}.get(name, "pymoo"))
    pytest.importorskip("cloudpickle")
    settings = dict(
        optimiser=name,
        evaluations=evaluations,
        seed=5,
        progress=False,
        # Four parallel workers, even where fewer CPUs are visible: an earlier
        # in-process gprMax run sets OMP_PROC_BIND, which can pin this process.
        execution=LocalPool(workers=4, cpu_threads_per_worker=1, allow_oversubscription=True),
        **options,
    )
    expected = _optimise(tmp_path / "straight", **settings)
    run = tmp_path / "interrupted"
    with _interrupt_objective_at(stop_at):
        _optimise(run, **settings)
    assert (run / "optimiser/checkpoint.bin").is_file()
    assert not list(run.rglob("*.tmp")), "Checkpoint and record writes leave no temporary files"
    # Only the documented arguments: the optimiser, its non-default settings,
    # the progress flag and the parallel execution are all restored.
    resumed = _optimise(run, resume=True)
    assert resumed.stop_reason == expected.stop_reason == "trial_budget"
    assert _trials(resumed) == _trials(expected)
    assert len(resumed.trials) == evaluations, "evaluations is the total, not extra evaluations"
    # The simulations completed before the interruption were reused, not repeated.
    assert _attempts(run) == _attempts(tmp_path / "straight")


@pytest.mark.slow
def test_repeated_interruptions_keep_checkpointing_and_evaluations_stays_the_total(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    expected = _optimise(tmp_path / "straight", evaluations=5, seed=7, progress=False)
    run = tmp_path / "run"
    with _interrupt_objective_at(2):
        _optimise(run, evaluations=3, seed=7, progress=False)
    first_checkpoint = (run / "optimiser/checkpoint.bin").read_bytes()
    # Raise the total budget from 3 to 5, then interrupt the resumed run too.
    with _interrupt_objective_at(3):
        _optimise(run, resume=True, evaluations=5)
    assert (
        run / "optimiser/checkpoint.bin"
    ).read_bytes() != first_checkpoint, "A resumed run must keep saving checkpoints"
    # Omitting evaluations now keeps the raised total of 5, not the original 3.
    resumed = _optimise(run, resume=True)
    assert len(resumed.trials) == 5
    assert _trials(resumed) == _trials(expected)


@pytest.mark.slow
def test_simulation_interrupted_part_way_runs_again_in_a_new_attempt(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    original = LocalExecutor.run
    calls = 0

    def stop_second_simulation(self, request, directory, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            # The worker had started this attempt when Ctrl+C arrived.
            directory.mkdir(parents=True)
            raise KeyboardInterrupt()
        return original(self, request, directory, **kwargs)

    run = tmp_path / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LocalExecutor, "run", stop_second_simulation)
        with pytest.raises(KeyboardInterrupt):
            _optimise(run, evaluations=2, seed=7, progress=False)
    result = _optimise(run, resume=True)
    assert [trial.status for trial in result.trials] == ["complete", "complete"]
    assert _attempts(run) == [
        Path("candidates/000001/default/attempt-0001"),
        Path("candidates/000002/default/attempt-0001"),
        Path("candidates/000002/default/attempt-0002"),
    ]
    # The incomplete attempt is kept for inspection; the rerun completed.
    second = run / "candidates/000002/default"
    assert not (second / "attempt-0001/result.json").exists()
    assert json.loads((second / "attempt-0002/result.json").read_text())["status"] == "complete"


def _environment():
    """Let a launched script import this gprMax checkout, as in test_optimisation_simple."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(Path(p).resolve()) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _stop_leftover_workers(folder):
    """Kill solver workers left running by the killed script, as a user must before resuming.

    Workers run in their own process group, so killing the script does not stop
    them. Each worker's command line contains its request file inside folder.
    """
    pids = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if str(folder).encode() in cmdline.read_bytes():
                pid = int(cmdline.parent.name)
                os.kill(pid, signal.SIGKILL)
                pids.append(pid)
        except (OSError, ValueError):
            pass
    deadline = time.monotonic() + 30
    while any(Path(f"/proc/{pid}").exists() for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.1)


@pytest.mark.slow
@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="needs Linux /proc")
def test_documented_restart_command_resumes_a_killed_starter_script(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    from gprMax.toolboxes.Optimisation.examples import start_here

    source = Path(start_here.__file__).read_text().replace("evaluations=12,", "evaluations=4,")
    straight, killed = tmp_path / "straight", tmp_path / "killed"
    for folder in (straight, killed):
        folder.mkdir()
        (folder / "my_model.py").write_text(source)

    def launch(folder, *arguments):
        return subprocess.run(
            [sys.executable, str(folder / "my_model.py"), *arguments],
            cwd=folder,
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=300,
        )

    run = launch(straight)
    assert run.returncode == 0, run.stdout + run.stderr

    process = subprocess.Popen(
        [sys.executable, str(killed / "my_model.py")],
        cwd=killed,
        env=_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output = []
        for line in process.stdout:
            output.append(line)
            if line.startswith("Evaluation 2:"):
                process.kill()  # SIGKILL: no Python clean-up runs.
                break
    finally:
        process.kill()
        process.wait(timeout=60)
    assert any(line.startswith("Evaluation 2:") for line in output), "".join(output)
    _stop_leftover_workers(killed)
    results = killed / "peak_field_results"
    assert (results / "optimiser/checkpoint.bin").is_file()
    assert not list(results.rglob(".checkpoint-*.tmp"))

    rerun = launch(killed)
    assert rerun.returncode != 0 and "--resume" in rerun.stderr, "Never restart silently"

    resumed = launch(killed, "--resume")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Resuming permittivity" in resumed.stdout
    numbers = [
        int(line.split(":")[0].split()[1])
        for line in resumed.stdout.splitlines()
        if line.startswith("Evaluation ")
    ]
    # Numbering continues after the interruption; it does not restart at 1.
    assert numbers and numbers[0] >= 2 and numbers == list(range(numbers[0], 5)), numbers

    def trials(folder):
        record = json.loads((folder / "peak_field_results/optimiser/result.json").read_text())
        return [(t["parameters"], t["status"], t["value"]) for t in record["trials"]]

    assert trials(killed) == trials(straight)
    assert len(trials(killed)) == 4
    # Each candidate has one completed attempt; the first one was never repeated.
    assert len(list(results.glob("candidates/000001/default/attempt-*"))) == 1
    for candidate in results.glob("candidates/*/default"):
        complete = [
            attempt
            for attempt in candidate.glob("attempt-*")
            if (attempt / "result.json").exists()
            and json.loads((attempt / "result.json").read_text())["status"] == "complete"
        ]
        assert len(complete) == 1, candidate


# ---------------------------------------------------------------------------
# Refusals, using copies of one interrupted run.


@pytest.fixture(scope="module")
def interrupted(tmp_path_factory):
    """A TPE run with a declared data file, stopped by Ctrl+C in its second evaluation."""
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    base = tmp_path_factory.mktemp("interrupted")
    data = base / "data.txt"
    data.write_text("reference data\n")
    with _interrupt_objective_at(2):
        _optimise(base / "run", evaluations=3, seed=7, progress=False, files=(data,))
    return base / "run", data


@pytest.fixture
def run(interrupted, tmp_path):
    """A private copy of the interrupted run, and the data file it depends on."""
    original, data = interrupted
    copy = tmp_path / "run"
    shutil.copytree(original, copy)
    return copy, data


def test_interrupted_run_has_a_checkpoint_by_default(run):
    directory, _ = run
    assert (directory / "optimiser/checkpoint.bin").is_file()
    assert json.loads((directory / "optimiser/result.json").read_text())["stop_reason"] == (
        "cancelled"
    )


def test_resume_with_the_same_settings_passed_explicitly_completes_the_run(run, capsys):
    directory, data = run
    result = _optimise(
        directory,
        resume=True,
        files=(data,),
        optimiser="tpe",
        seed=7,
        batch_size=1,
        population_size=6,
        target_value=None,
        stop_on_target=True,
        progress=False,
        evaluations=3,
    )
    assert result.stop_reason == "trial_budget" and len(result.trials) == 3
    assert all(trial.status == "complete" for trial in result.trials)
    # A finished run reports its saved result; nothing is asked or simulated again.
    attempts = _attempts(directory)
    again = _optimise(directory, resume=True, files=(data,))
    assert again.trials == result.trials and _attempts(directory) == attempts
    # progress=False was restored although it was omitted on resume.
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "setting",
    [
        {"optimiser": "ga"},
        {"seed": 3},
        {"population_size": 4},
        {"batch_size": 2},
        {"target_value": 1.0},
        {"stop_on_target": False},
        {"progress": True},
    ],
    ids=lambda setting: next(iter(setting)),
)
def test_resume_refuses_an_explicitly_different_setting(run, setting):
    directory, data = run
    before = _snapshot(directory)
    name = next(iter(setting))
    with pytest.raises(ValueError, match=f"{name}=.*original run"):
        _optimise(directory, resume=True, files=(data,), **setting)
    assert _snapshot(directory) == before


def test_resume_refuses_an_optimiser_object(run):
    from gprMax.toolboxes.Optimisation import make_optimiser

    directory, data = run
    before = _snapshot(directory)
    with pytest.raises(ValueError, match="saved optimiser is restored"):
        _optimise(directory, resume=True, files=(data,), optimiser=make_optimiser("tpe", seed=7))
    assert _snapshot(directory) == before


@pytest.mark.parametrize(
    "change, message",
    [
        ({"model": other_model}, "model definition or dependency changed"),
        ({"objective": other_objective}, "Evaluator definition changed"),
        ({"execution": LocalPool(workers=2)}, "original execution configuration"),
    ],
    ids=["model", "objective", "execution"],
)
def test_resume_refuses_a_changed_model_objective_or_execution(run, change, message):
    directory, data = run
    before = _snapshot(directory)
    with pytest.raises(ValueError, match=message):
        _optimise(directory, resume=True, files=(data,), **change)
    assert _snapshot(directory) == before


def test_resume_refuses_a_changed_dependency_file(run, tmp_path):
    directory, data = run
    before = _snapshot(directory)
    extra = tmp_path / "extra.txt"
    extra.write_text("another input\n")
    with pytest.raises(ValueError, match="dependency changed"):
        _optimise(directory, resume=True, files=(data, extra))
    original = data.read_text()
    try:
        data.write_text("edited reference data\n")
        with pytest.raises(ValueError, match="dependency changed"):
            _optimise(directory, resume=True, files=(data,))
    finally:
        data.write_text(original)
    assert _snapshot(directory) == before


def test_resume_refuses_a_changed_toolbox_or_optimiser_library(run, monkeypatch):
    from gprMax.toolboxes.Optimisation import optimisation

    directory, data = run
    before = _snapshot(directory)
    with monkeypatch.context() as patch:
        patch.setattr(optimisation, "sha256", lambda path: "edited source")
        with pytest.raises(ValueError, match="Toolbox source changed"):
            _optimise(directory, resume=True, files=(data,))
    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", lambda name: "0.0.0")
        with pytest.raises(ValueError, match="Optimiser library version changed"):
            _optimise(directory, resume=True, files=(data,))
    assert _snapshot(directory) == before


@pytest.mark.parametrize("damage", ["flipped byte", "truncated"])
def test_resume_refuses_a_corrupt_checkpoint(run, damage):
    directory, data = run
    checkpoint = directory / "optimiser/checkpoint.bin"
    content = bytearray(checkpoint.read_bytes())
    if damage == "flipped byte":
        content[-1] ^= 0xFF
    else:
        del content[len(content) // 2 :]
    checkpoint.write_bytes(bytes(content))
    before = _snapshot(directory)
    with pytest.raises(ValueError, match="corrupt or incomplete and cannot be used to resume"):
        _optimise(directory, resume=True, files=(data,))
    assert _snapshot(directory) == before


@pytest.mark.parametrize("cause", ["corrupt checkpoint", "library version"])
def test_refused_resume_does_not_record_a_new_evaluations_total(run, cause):
    directory, data = run
    if cause == "corrupt checkpoint":
        checkpoint = directory / "optimiser/checkpoint.bin"
        checkpoint.write_bytes(checkpoint.read_bytes()[:-1])
    before = _snapshot(directory)
    with pytest.MonkeyPatch.context() as patch:
        if cause == "library version":
            patch.setattr(importlib.metadata, "version", lambda name: "0.0.0")
        with pytest.raises(ValueError, match="corrupt|library version changed"):
            _optimise(directory, resume=True, files=(data,), evaluations=5)
    # The refused total of 5 must not become the saved total of the run.
    assert _snapshot(directory) == before


@pytest.mark.parametrize("content", ["{", "[]", '{"schema_version": 1}'])
def test_resume_refuses_a_damaged_settings_file(run, content):
    directory, data = run
    (directory / "optimise_settings.json").write_text(content)
    before = _snapshot(directory)
    with pytest.raises(ValueError, match="optimise_settings.json is damaged"):
        _optimise(directory, resume=True, files=(data,))
    assert _snapshot(directory) == before


def test_resume_refuses_a_total_below_the_completed_evaluations(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    directory = tmp_path / "run"
    with _interrupt_objective_at(3):
        _optimise(directory, evaluations=3, seed=7, progress=False)
    before = _snapshot(directory)
    with pytest.raises(ValueError, match="2 are already complete.*at least 2"):
        _optimise(directory, resume=True, evaluations=1)
    assert _snapshot(directory) == before
    # The refusal did not end the run: it still continues to its total.
    result = _optimise(directory, resume=True)
    assert result.stop_reason == "trial_budget" and len(result.trials) == 3


def test_resume_refuses_a_tampered_trial_record(run):
    directory, data = run
    record_path = directory / "optimiser/trial-000001.json"
    record = json.loads(record_path.read_text())
    record["proposal"]["parameters"]["permittivity"] = 2.5
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="did not reproduce its recorded proposal"):
        _optimise(directory, resume=True, files=(data,))


@pytest.mark.parametrize(
    "removed, message",
    [
        # checkpoint=False, or a run made before optimise() saved checkpoints.
        (["optimiser/checkpoint.bin"], "run without checkpoints"),
        (["optimiser/checkpoint.bin", "optimise_settings.json"], "run without checkpoints"),
        # A checkpointed run created through the Campaign interface.
        (["optimise_settings.json"], "created with the Campaign interface"),
        (["optimiser"], "no optimisation was started"),
        (["campaign.json"], "does not contain an optimisation campaign"),
    ],
    ids=["no-checkpoint", "pre-feature-run", "campaign-interface", "no-session", "no-campaign"],
)
def test_resume_explains_why_a_directory_cannot_be_resumed(run, removed, message):
    directory, data = run
    for name in removed:
        path = directory / name
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    before = _snapshot(directory)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        _optimise(directory, resume=True, files=(data,))
    assert _snapshot(directory) == before


def test_existing_directory_is_refused_without_resume(run, tmp_path):
    directory, data = run
    before = _snapshot(directory)
    with pytest.raises(FileExistsError, match="resume=True"):
        _optimise(directory, files=(data,), evaluations=3, seed=7, progress=False)
    assert _snapshot(directory) == before
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep me\n")
    with pytest.raises(FileExistsError, match="choose a new directory"):
        _optimise(unrelated)
    assert _snapshot(unrelated) == {Path("notes.txt"): b"keep me\n"}


def test_resume_without_cloudpickle_fails_before_running(run, monkeypatch):
    directory, data = run
    before = _snapshot(directory)
    monkeypatch.setitem(sys.modules, "cloudpickle", None)  # "import cloudpickle" now fails
    with pytest.raises(ImportError, match="requirements-optimisers.txt"):
        _optimise(directory, resume=True, files=(data,))
    assert _snapshot(directory) == before


# ---------------------------------------------------------------------------
# Defaults and argument checks.


def test_signature_still_shows_the_plain_defaults():
    signature = str(inspect.signature(optimise))
    for default in (
        "optimiser='tpe'",
        "evaluations=12",
        "seed=0",
        "population_size=6",
        "batch_size=1",
        "progress=True",
        "target_value=None",
        "stop_on_target=True",
        "checkpoint=True",
        "resume=False",
    ):
        assert default in signature


def test_invalid_checkpoint_and_resume_arguments_are_refused(tmp_path):
    with pytest.raises(ValueError, match="remove checkpoint=False"):
        _optimise(tmp_path / "new", resume=True, checkpoint=False)
    with pytest.raises(ValueError, match="True or False"):
        _optimise(tmp_path / "new", checkpoint="yes")
    with pytest.raises(FileNotFoundError, match="does not contain an optimisation campaign"):
        _optimise(tmp_path / "missing", resume=True)
    assert not list(tmp_path.iterdir()), "Nothing is created by a refused call"


def test_missing_cloudpickle_fails_before_creating_anything(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cloudpickle", None)
    with pytest.raises(ImportError, match="checkpoint=False"):
        _optimise(tmp_path / "new")
    # Mistakes in the user's own arguments are still reported first.
    with pytest.raises(TypeError, match=r"evaluate\(parameters, output\)"):
        _optimise(tmp_path / "new", objective=tiny_model)
    assert not (tmp_path / "new").exists()


def test_numpy_integer_for_a_setting_the_optimiser_ignores_is_saved(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    directory = tmp_path / "run"
    # TPE ignores population_size; the call worked before resume was added.
    _optimise(directory, evaluations=1, seed=7, population_size=np.int64(4), progress=False)
    saved = json.loads((directory / "optimise_settings.json").read_text())
    assert saved["population_size"] == 4


def test_run_without_checkpoints_needs_no_cloudpickle_and_cannot_be_resumed(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    directory = tmp_path / "run"
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "cloudpickle", None)
        result = _optimise(directory, checkpoint=False, evaluations=1, progress=False)
    assert result.trials[0].status == "complete"
    assert not (directory / "optimiser/checkpoint.bin").exists()
    with pytest.raises(ValueError, match="run without checkpoints"):
        _optimise(directory, resume=True)


def test_simulate_directory_is_not_mistaken_for_an_optimisation(tmp_path):
    pytest.importorskip("cloudpickle")
    simulate(model=tiny_model, parameters={"permittivity": 4.0}, directory=tmp_path / "one")
    before = _snapshot(tmp_path / "one")
    with pytest.raises(ValueError, match="no optimisation was started"):
        _optimise(tmp_path / "one", resume=True)
    assert _snapshot(tmp_path / "one") == before


# ---------------------------------------------------------------------------
# Finished runs keep their existing meaning on resume.


def test_resuming_a_run_that_reached_its_target_does_not_continue(tmp_path, capsys):
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    directory = tmp_path / "run"
    result = _optimise(directory, evaluations=3, seed=7, target_value=1e6)
    assert result.stop_reason == "target_reached" and len(result.trials) == 1
    attempts = _attempts(directory)
    capsys.readouterr()
    again = _optimise(directory, resume=True)
    assert again.stop_reason == "target_reached" and again.trials == result.trials
    assert _attempts(directory) == attempts
    output = capsys.readouterr().out
    assert "already finished (target_reached)" in output and "Resuming" not in output
    # A larger total does not extend a finished run, and is refused, not ignored.
    before = _snapshot(directory)
    with pytest.raises(ValueError, match=r"already finished \(target_reached, 1 evaluations\)"):
        _optimise(directory, resume=True, evaluations=5)
    assert _snapshot(directory) == before


def test_resuming_a_failed_run_reports_the_failure_again(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    directory = tmp_path / "run"
    with pytest.raises(RuntimeError, match="deliberate objective failure"):
        _optimise(directory, objective=failing_objective, evaluations=3, seed=7, progress=False)
    attempts = _attempts(directory)
    with pytest.raises(RuntimeError, match="deliberate objective failure"):
        _optimise(directory, objective=failing_objective, resume=True)
    assert _attempts(directory) == attempts
    record = json.loads((directory / "optimiser/result.json").read_text())
    assert record["stop_reason"] == "trial_failed" and len(record["trials"]) == 1


def test_custom_optimiser_object_is_restored_and_cannot_be_replaced(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    from gprMax.toolboxes.Optimisation import make_optimiser

    directory = tmp_path / "run"
    with _interrupt_objective_at(1):
        _optimise(directory, optimiser=make_optimiser("tpe", seed=7), evaluations=2, progress=False)
    with pytest.raises(ValueError, match="custom OptunaTPE"):
        _optimise(directory, resume=True, optimiser="tpe")
    result = _optimise(directory, resume=True)
    assert [trial.status for trial in result.trials] == ["complete", "complete"]
