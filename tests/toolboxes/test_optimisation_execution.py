"""Execution/processing contracts, including actual small FDTD simulations."""

import json
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from gprMax.toolboxes.Optimisation import (
    Campaign,
    Evaluation,
    Integer,
    LocalExecutor,
    LocalPool,
    ObjectiveResult,
    ParameterSpace,
    Problem,
    RunResult,
    RunTask,
    Scenario,
    evaluate_outputs,
    execution_from_profile,
)


def tiny_problem():
    from gprMax.toolboxes.Optimisation import Real
    from gprMax.toolboxes.Optimisation.examples import dielectric_block

    return Problem(
        ParameterSpace({"width": Real(0.008, 0.016), "permittivity": Real(2, 8)}),
        "gprMax.toolboxes.Optimisation.examples.dielectric_block:build_model",
        dependencies=(Path(dielectric_block.__file__),),
    )


def test_pool_bounds_order_device_assignment_and_failure(monkeypatch, tmp_path):
    lock, two_started = threading.Lock(), threading.Event()
    started, active = [], []
    peak = 0

    def worker(self, request, directory, *, cancel_event=None):
        nonlocal peak
        with lock:
            started.append((request["candidate_id"], self.device))
            active.append(self.device)
            assert len(set(active)) == len(active), "Two simulations share a GPU"
            peak = max(peak, len(active))
            if len(started) == 2:
                two_started.set()
        assert two_started.wait(3)
        if request["candidate_id"] == "0":
            time.sleep(0.08)
        status = "failed" if request.get("fail") else "complete"
        with lock:
            active.remove(self.device)
        return RunResult(
            request["candidate_id"], "default", directory, {"status": status, "run_attempts": 1}
        )

    monkeypatch.setattr(LocalExecutor, "run", worker)
    pool = LocalPool(solver="cuda", devices=[0, 2])
    tasks = [
        RunTask({"candidate_id": str(i), "scenario": {"id": "default"}}, tmp_path / str(i))
        for i in range(5)
    ]
    completion = []
    results = pool.map(tasks, on_result=lambda result: completion.append(result.candidate_id))
    assert peak == 2 and completion[0] == "1"
    assert [r.candidate_id for r in results] == [str(i) for i in range(5)]
    started.clear()
    tasks = [
        RunTask(
            dict(t.request, fail=t.request["candidate_id"] == "1"), tmp_path / "failed" / str(i)
        )
        for i, t in enumerate(tasks)
    ]
    results = pool.map(tasks, stop_on_failure=True)
    assert len(started) == 2
    assert [r.status for r in results] == [
        "complete",
        "failed",
        "cancelled",
        "cancelled",
        "cancelled",
    ]
    assert sum(r.record["run_attempts"] for r in results) == 2


def test_pool_configuration_profiles_and_cpu_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    if hasattr(__import__("os"), "sched_getaffinity"):
        monkeypatch.setattr("os.sched_getaffinity", lambda pid: {0, 1, 2, 3})
    with pytest.raises(ValueError, match="CPU threads"):
        LocalPool(workers=3, cpu_threads_per_worker=2)
    for kwargs in (
        {"solver": "cuda"},
        {"solver": "cuda", "devices": [0, 0]},
        {"solver": "cpu", "devices": [0]},
        {"solver": "cuda", "devices": [True]},
        {"solver": "metal", "devices": [0], "precision": "double"},
    ):
        with pytest.raises(ValueError):
            LocalPool(**kwargs)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"backend": "local", "workers": 2, "cpu_threads_per_worker": 2}))
    pool = execution_from_profile(path)
    assert pool.workers == 2 and pool.executors[0].cpu_threads == 2


def trace_result(tmp_path, values=(1, 3, 5)):
    directory = tmp_path / "run"
    directory.mkdir()
    path = directory / "output.h5"
    with h5py.File(path, "w") as output:
        receiver = output.create_group("rxs/rx1")
        receiver.attrs["Name"] = "received"
        data = receiver.create_dataset("Ez", data=values)
        data.attrs["SampleInterval"] = 1e-9
        data.attrs["TimeSampleOffset"] = 0
    from gprMax.toolboxes.Optimisation._storage import sha256

    return RunResult(
        "000001",
        "default",
        directory,
        {
            "status": "complete",
            "artifacts": {"output": {"path": "output.h5", "sha256": sha256(path)}},
        },
    )


def test_processing_interpolation_gate_artifacts_and_reprocessing(tmp_path):
    run = trace_result(tmp_path)
    reference = tmp_path / "reference.npz"
    # Predicted values at these times are 2, 4. Reference differs by one.
    np.savez(reference, time=[0.5e-9, 1.5e-9], values=[1, 3], unit="V/m")
    evaluator = Evaluation(
        "gprMax.toolboxes.Optimisation.examples.advanced.waveform_matching:evaluate_waveform",
        settings={"reference": str(reference)},
        dependencies=(reference,),
    )
    result = evaluate_outputs(evaluator, {}, {"default": run}, tmp_path / "processed")
    assert result.value == pytest.approx(1 / np.sqrt(5))
    record = json.loads((tmp_path / "processed/processing.json").read_text())
    assert record["status"] == "complete" and record["evaluator"]["dependencies"][0]["sha256"]
    assert record["artifacts"]["waveforms.npz"]["units"]["residual"] == "V/m"
    with np.load(tmp_path / "processed/waveforms.npz") as arrays:
        np.testing.assert_allclose(arrays["residual"], [1, 1])
    result2 = evaluate_outputs(evaluator, {}, {"default": run}, tmp_path / "reprocessed")
    assert result2 == result
    from gprMax.toolboxes.Optimisation.processing import PreparedEvaluator

    prepared = PreparedEvaluator(evaluator)
    reference.write_bytes(b"changed")
    with pytest.raises(ValueError, match="dependency changed"):
        prepared({}, {"default": run}, tmp_path / "changed")


def test_processing_rejects_invalid_units_extrapolation_and_artifact_paths(tmp_path):
    from gprMax.toolboxes.Optimisation import ProcessingContext

    run = trace_result(tmp_path)
    reference = tmp_path / "reference.npz"

    def evaluation(name, times, unit):
        np.savez(reference, time=times, values=[1, 3], unit=unit)
        return evaluate_outputs(
            Evaluation(
                "gprMax.toolboxes.Optimisation.examples.advanced.waveform_matching:evaluate_waveform",
                settings={"reference": str(reference)},
            ),
            {},
            {"default": run},
            tmp_path / name,
        )

    with pytest.raises(ValueError, match="units"):
        evaluation("units", [0, 1e-9], "A/m")
    with pytest.raises(ValueError, match="extrapolation"):
        evaluation("outside", [0, 3e-9], "V/m")
    context = ProcessingContext(tmp_path, {})
    with pytest.raises(ValueError, match="simple filenames"):
        context.save_json("../escaped.json", {})
    with pytest.raises(ValueError, match="non-object"):
        context.save_npz("objects.npz", objects=[{}])


@pytest.mark.parametrize("name", ["rf", "tpe"])
def test_surrogate_batches_preserve_pending_correspondence(name):
    pytest.importorskip("skopt" if name == "rf" else "optuna")
    from gprMax.toolboxes.Optimisation import TrialResult, make_optimiser

    adapter = make_optimiser(name, seed=7, batch_size=3)
    adapter.initialise(ParameterSpace({"x": Integer(1, 30)}))
    for generation in range(3):
        proposals = adapter.ask_batch()
        assert len(proposals) == 3 and len({p.token for p in proposals}) == 3
        results = tuple(
            TrialResult(
                p.token, p.parameters, str(i), "complete", float((p.parameters["x"] - 7) ** 2)
            )
            for i, p in enumerate(proposals)
        )
        with pytest.raises(ValueError, match="match"):
            adapter.tell_batch(proposals, tuple(reversed(results)))
        adapter.tell_batch(proposals, results)
    if name == "rf":
        assert len(adapter.optimiser.yi) == 9
    else:
        assert len(adapter.study.trials) == 9


def test_actual_cpu_pool_matches_serial_receiver_data(tmp_path):
    from gprMax.toolboxes.Optimisation import read_receiver

    candidates = [{"width": 0.008, "permittivity": 4}, {"width": 0.016, "permittivity": 6}]
    serial = Campaign(tiny_problem(), tmp_path / "serial").evaluate(candidates)
    parallel = Campaign(tiny_problem(), tmp_path / "pool", LocalPool(workers=2)).evaluate(
        candidates
    )
    for a, b in zip(serial, parallel):
        assert a[0].status == b[0].status == "complete"
        np.testing.assert_array_equal(
            read_receiver(a[0].output_file, "received", "Ez").values,
            read_receiver(b[0].output_file, "received", "Ez").values,
        )
    progress = json.loads((tmp_path / "pool/progress.json").read_text())
    assert progress["run_attempts"] == progress["completed"] == 2


def energy_objective(parameters, runs):
    from gprMax.toolboxes.Optimisation import read_receiver

    trace = read_receiver(runs["default"].output_file, "received", "Ez")
    return ObjectiveResult(float(np.mean(trace.values**2)))


def test_cache_coalesces_identical_pending_requests_and_reuses_saved_results(tmp_path):
    from gprMax.toolboxes.Optimisation import SimulationCache

    cache = SimulationCache(tmp_path / "cache", namespace="test-solver-build-1")
    campaign = Campaign(tiny_problem(), tmp_path / "first", LocalPool(workers=2), cache=cache)
    values = {"width": 0.012, "permittivity": 4}
    runs = campaign.evaluate([values, values, values])
    assert campaign.run_attempts == 1
    assert [r[0].record.get("cache", {}).get("hit", False) for r in runs] == [False, True, True]
    second = Campaign(tiny_problem(), tmp_path / "second", LocalPool(workers=2), cache=cache)
    assert second.evaluate_one(values)[0].record["cache"]["hit"]
    assert second.run_attempts == 0
    second.evaluate_one(values, seed=1)
    assert second.run_attempts == 1, "Simulation seed is part of cache identity"
    from gprMax.toolboxes.Optimisation import load_candidate

    parameters, recovered = load_candidate(campaign.storage / "candidates/000002")
    assert parameters == values and recovered["default"].output_file.is_file()


def test_checkpoint_recovers_pending_batch_without_repeating_completed_solves(
    tmp_path, monkeypatch
):
    pytest.importorskip("optuna")
    pytest.importorskip("cloudpickle")
    from gprMax.toolboxes.Optimisation import OptunaTPE
    from gprMax.toolboxes.Optimisation.processing import PreparedEvaluator

    campaign = Campaign(tiny_problem(), tmp_path / "campaign", LocalPool(workers=2))
    original = PreparedEvaluator.__call__
    calls = 0

    def interrupt_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PreparedEvaluator, "__call__", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        campaign.optimise(
            optimiser=OptunaTPE(seed=7, batch_size=2),
            evaluator=__name__ + ":energy_objective",
            n_trials=4,
            max_simulations=8,
            checkpoint=True,
        )
    assert campaign.run_attempts == 2
    monkeypatch.setattr(PreparedEvaluator, "__call__", original)
    resumed = Campaign.resume(tiny_problem(), campaign.storage)
    result = resumed.optimise(
        evaluator=__name__ + ":energy_objective", n_trials=4, max_simulations=8, resume=True
    )
    assert result.stop_reason == "trial_budget" and len(result.trials) == 4
    assert result.run_attempts == resumed.run_attempts == 4
    assert all(trial.status == "complete" for trial in result.trials)
    assert len(list(campaign.storage.glob("candidates/*/*/attempt-*"))) == 4
    # A terminal checkpoint can be opened without asking or simulating again.
    again = Campaign.resume(tiny_problem(), campaign.storage)
    assert (
        again.optimise(
            evaluator=__name__ + ":energy_objective", n_trials=4, max_simulations=8, resume=True
        ).trials
        == result.trials
    )


def test_retry_records_attempts_and_keeps_admission_within_budget(monkeypatch, tmp_path):
    pytest.importorskip("optuna")
    started = []

    def worker(self, request, directory, *, cancel_event=None):
        directory.mkdir(parents=True)
        started.append(directory.name)
        failed = directory.name == "attempt-0001"
        result = RunResult(
            request["candidate_id"],
            request["scenario"]["id"],
            directory,
            {
                "status": "failed" if failed else "complete",
                "run_attempts": 1,
                "failure": {"kind": "worker_exit"} if failed else {},
            },
        )
        (directory / "result.json").write_text(json.dumps(result.record))
        return result

    monkeypatch.setattr(LocalExecutor, "run", worker)
    pool = LocalPool(workers=1, max_retries=1)
    result = pool.run(
        {"candidate_id": "1", "scenario": {"id": "default"}}, tmp_path / "case/attempt-0001"
    )
    assert result.status == "complete" and result.record["run_attempts"] == 2
    assert started == ["attempt-0001", "attempt-0002"]
    assert (
        json.loads((tmp_path / "case/attempt-0001/result.json").read_text())["status"] == "failed"
    )
    from gprMax.toolboxes.Optimisation import OptunaTPE

    campaign = Campaign(tiny_problem(), tmp_path / "budget", LocalPool(workers=2, max_retries=1))
    result = campaign.optimise(
        optimiser=OptunaTPE(batch_size=2),
        evaluator=__name__ + ":energy_objective",
        n_trials=4,
        max_simulations=3,
    )
    assert result.stop_reason == "simulation_budget" and campaign.run_attempts == 0


@pytest.mark.parametrize("name", ["ga", "pso", "de", "rf", "tpe"])
def test_native_adapter_checkpoint_restores_next_proposals(tmp_path, name):
    pytest.importorskip({"rf": "skopt", "tpe": "optuna"}.get(name, "pymoo"))
    pytest.importorskip("cloudpickle")
    from gprMax.toolboxes.Optimisation import Real, TrialResult, make_optimiser
    from gprMax.toolboxes.Optimisation.checkpoint import load_checkpoint, save_checkpoint

    options = {"batch_size": 4} if name in ("rf", "tpe") else {"population_size": 4}
    adapter = make_optimiser(name, seed=7, **options)
    adapter.initialise(ParameterSpace({"x": Real(0, 1)}))
    for i in range(2):
        proposals = adapter.ask_batch()
        adapter.tell_batch(
            proposals,
            tuple(
                TrialResult(p.token, p.parameters, None, "complete", (p.parameters["x"] - 0.3) ** 2)
                for p in proposals
            ),
        )
    save_checkpoint(tmp_path, {"optimiser": adapter})
    expected = adapter.ask_batch()
    restored = load_checkpoint(tmp_path)["optimiser"].ask_batch()
    assert [(p.token, dict(p.parameters), dict(p.metadata)) for p in expected] == [
        (p.token, dict(p.parameters), dict(p.metadata)) for p in restored
    ]
