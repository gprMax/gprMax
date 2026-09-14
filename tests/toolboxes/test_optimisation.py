import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from gprMax.toolboxes.Optimisation import (
    Campaign,
    Categorical,
    Integer,
    LocalExecutor,
    ParameterSpace,
    Problem,
    Real,
    Scenario,
    read_receiver,
)


def test_physical_parameters_and_log_decoding():
    sigma = Real(1e-4, 1e2, "S/m", scale="log")
    assert sigma.from_unit(0.5) == pytest.approx(0.1)
    space = ParameterSpace(
        {"sigma": sigma, "layers": Integer(1, 3), "material": Categorical(("air", "soil"))}
    )
    assert space.validate({"sigma": 1.0, "layers": 2, "material": "soil"})["sigma"] == 1.0
    # Physical candidates are never decoded or log-transformed a second time.
    for bad in (float("nan"), float("inf"), True, "0.1", 1000):
        with pytest.raises(ValueError):
            space.validate({"sigma": bad, "layers": 2, "material": "soil"})
    with pytest.raises(ValueError):
        space.validate({"sigma": 1, "layers": True, "material": "soil"})
    with pytest.raises(ValueError):
        space.validate({"sigma": 1, "layers": 2, "material": "metal"})
    with pytest.raises(ValueError):
        space.validate({"sigma": 1, "layers": 2, "typo": "soil"})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Real(0, 10, scale="log"),
        lambda: Real(2, 1),
        lambda: Integer(1.5, 2),
        lambda: Categorical(("a", "a")),
        lambda: Categorical("air"),
        lambda: Scenario("../escape"),
        lambda: LocalExecutor(cpu_threads=True),
    ],
)
def test_invalid_definitions(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def problem(builder="missing_module:build_model", **kwargs):
    return Problem(ParameterSpace({"x": Real(0, 1)}), builder, **kwargs)


def test_import_does_not_load_optional_optimisers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gprMax.toolboxes.Optimisation; assert not {'optuna', 'pymoo', 'skopt', 'sklearn'} & sys.modules.keys()",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_validation_and_declared_dependency_changes_do_not_launch(tmp_path):
    asset = tmp_path / "asset.txt"
    asset.write_text("original")
    campaign = Campaign(problem(dependencies=(asset,)), tmp_path / "study")
    with pytest.raises(ValueError):
        campaign.evaluate_one({"x": 3})
    assert not (campaign.storage / "candidates").exists()
    asset.write_text("changed")
    with pytest.raises(ValueError, match="dependency changed"):
        campaign.evaluate_one({"x": 0.5})
    with pytest.raises(FileExistsError):
        Campaign(problem(), campaign.storage)


def test_worker_import_failure_is_recorded_and_does_not_abort_next_candidate(tmp_path):
    campaign = Campaign(problem(), tmp_path / "study")
    results = campaign.evaluate([{"x": 0.2}, {"x": 0.4}])
    assert [r[0].status for r in results] == ["failed", "failed"]
    assert results[0][0].record["failure"]["kind"] == "import_error"
    assert "ModuleNotFoundError" in (results[0][0].directory / "stderr.log").read_text()
    assert results[0][0].output_file is None
    assert results[0][0].directory != results[1][0].directory
    assert (
        json.loads((campaign.storage / "candidates/000002/evaluation.json").read_text())["status"]
        == "failed"
    )


def test_timeout_terminates_worker_and_records_failure(tmp_path):
    (tmp_path / "slow_builder.py").write_text(
        "import time\ntime.sleep(10)\ndef build_model(*args): pass\n"
    )
    executor = LocalExecutor(timeout=0.3, pythonpath=(tmp_path,))
    result = Campaign(
        problem("slow_builder:build_model"), tmp_path / "study", executor
    ).evaluate_one({"x": 0.5})[0]
    assert result.status == "failed"
    assert result.record["failure"]["kind"] == "timeout"
    assert result.record["returncode"] != 0
    assert result.output_file is None


def test_scenario_settings_are_snapshotted(tmp_path):
    scenario = Scenario("experiment", {"height": 1})
    campaign = Campaign(problem(scenarios=(scenario,)), tmp_path / "study")
    scenario.settings["height"] = 9
    run = campaign.evaluate_one({"x": 0.5})[0]
    request = json.loads((run.directory / "request.json").read_text())
    assert request["scenario"]["settings"] == {"height": 1}


def test_receiver_reader_preserves_half_step_and_rejects_ambiguity(tmp_path):
    path = tmp_path / "traces.h5"
    with h5py.File(path, "w") as output:
        rx = output.create_group("rxs/rx7")
        rx.attrs["Name"] = "magnetic"
        data = rx.create_dataset("Hy", data=[1.0, 2.0, 3.0])
        data.attrs["SampleInterval"] = 2e-12
        data.attrs["TimeSampleOffset"] = -1e-12
    trace = read_receiver(path, "magnetic", "Hy")
    np.testing.assert_allclose(trace.time, [-1e-12, 1e-12, 3e-12], rtol=1e-14, atol=0)
    assert trace.unit == "A/m"
    with pytest.raises(ValueError, match="found 0"):
        read_receiver(path, "absent", "Hy")
    with h5py.File(path, "a") as output:
        output.copy("rxs/rx7", "rxs/rx8")
    with pytest.raises(ValueError, match="found 2"):
        read_receiver(path, "magnetic", "Hy")


def test_port_reader_interpolates_complex_s11_without_crossing_invalid_bins(tmp_path):
    from gprMax.toolboxes.Optimisation import read_port

    path = tmp_path / "port.h5"
    with h5py.File(path, "w") as output:
        group = output.create_group("ports/feed")
        group.attrs["ReferenceImpedance"] = 50
        group.attrs["TailRelativeLevelDB"] = -80
        group.attrs["IndependentFrequencyResolution"] = 1e9
        group["frequency"] = [1e9, 2e9, 3e9]
        group["S11"] = [0.2 + 0.4j, 0.4 + 0.2j, np.nan + 0j]
        group["valid_S11"] = [1, 1, 0]
        group["Zin"] = [50 + 1j, 50 - 1j, np.nan + 0j]
        group["valid_Zin"] = [1, 1, 0]
    spectrum = read_port(path, "feed")
    assert spectrum.at(1.5e9) == pytest.approx(0.3 + 0.3j)
    assert spectrum.at(2e9) == pytest.approx(0.4 + 0.2j)
    assert spectrum.reference_impedance == 50
    assert spectrum.independent_frequency_resolution_hz == 1e9
    for frequency in (0.5e9, 2.5e9, 3e9, 4e9):
        with pytest.raises(ValueError):
            spectrum.at(frequency)
    with h5py.File(path, "a") as output:
        output["ports/feed/valid_S11"][2] = 1
    with pytest.raises(ValueError, match="Nonfinite"):
        read_port(path, "feed")


@pytest.mark.parametrize("optimum", [1, 6, 12])
def test_scalar_integer_search_keeps_distinct_physical_candidates(optimum):
    from gprMax.toolboxes.Optimisation import minimise_integer

    calls = []

    def objective(n):
        assert isinstance(n, int)
        calls.append(n)
        return (n - optimum) ** 2

    result = minimise_integer(objective, 1, 12, initial=5)
    assert result.best == optimum
    assert result.value == 0
    assert len(calls) == len(set(calls))
    assert len(calls) == len(result.evaluations)


def test_scalar_driver_rejects_invalid_objective():
    from gprMax.toolboxes.Optimisation import minimise_integer

    with pytest.raises(ValueError, match="not finite"):
        minimise_integer(lambda n: float("nan"), 1, 5)


def test_dipole_builder_changes_arms_without_shorting_the_feed():
    import gprMax
    from gprMax.toolboxes.Optimisation import BuildContext
    from gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole import build_model, settings

    config = settings()
    scenario = Scenario("free_space", config)
    context = BuildContext(Path("."), Path("output"), 0, 1)
    prepared = [
        build_model({"arm_cells": n}, scenario, context) for n in (config["lower"], config["upper"])
    ]
    sources = []
    domains = []
    for model in prepared:
        wires = model.scene.geometry_objects
        assert len(wires) == 2 and all(isinstance(wire, gprMax.ThinWire) for wire in wires)
        assert wires[1].kwargs["p1"][2] - wires[0].kwargs["p2"][2] == pytest.approx(config["dl"])
        assert 0 < wires[0].kwargs["radius"] < 0.5 * config["dl"]
        sources.append(
            next(x for x in model.scene.grid_objects if isinstance(x, gprMax.VoltageSource))
        )
        domains.append(
            next(x for x in model.scene.single_use_objects if isinstance(x, gprMax.Domain))
        )
    assert sources[0].point == sources[1].point
    assert domains[0].kwargs == domains[1].kwargs
    assert (
        prepared[0].effective_parameters["total_length_m"]
        < prepared[1].effective_parameters["total_length_m"]
    )


@pytest.fixture
def joint_evaluator(tmp_path, monkeypatch):
    module = tmp_path / "joint_objective.py"
    module.write_text(
        "from gprMax.toolboxes.Optimisation import ObjectiveResult\n"
        "def evaluate(parameters, runs):\n"
        "    values = {name: run.record['signal'] for name, run in runs.items()}\n"
        "    return ObjectiveResult(sum(values.values()), {'signals': values})\n"
        "def invalid(parameters, runs):\n"
        "    return ObjectiveResult(float('nan'))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("joint_objective", None)
    yield "joint_objective:evaluate"
    sys.modules.pop("joint_objective", None)


class RecordingExecutor:
    """Scheduler-only fixture; actual solver tests remain separate below."""

    def __init__(self, fail_scenario=None):
        self.requests = []
        self.fail_scenario = fail_scenario

    def to_dict(self):
        return {"backend": "test"}

    def run(self, request, directory):
        from gprMax.toolboxes.Optimisation import RunResult

        directory.mkdir(parents=True)
        self.requests.append(request)
        failed = request["scenario"]["id"] == self.fail_scenario
        return RunResult(
            request["candidate_id"],
            request["scenario"]["id"],
            directory,
            {
                "status": "failed" if failed else "complete",
                "signal": request["parameters"]["x"]
                * request["scenario"]["settings"].get("weight", 1),
                "failure": {"kind": "test_failure"} if failed else None,
            },
        )


class FeedbackOptimiser:
    """Proposals depend on the previous feedback, independently of any library."""

    def __init__(self):
        self.results = []
        self.asked = 0

    def initialise(self, parameters):
        return {"adapter": "test", "direction": "minimize"}

    def ask(self):
        from gprMax.toolboxes.Optimisation import Proposal

        assert self.asked == len(self.results), "Cannot propose the next candidate before feedback"
        value = 0.5 if not self.results else self.results[-1].value / 10
        self.asked += 1
        return Proposal(f"provider-{self.asked}", {"x": value})

    def tell(self, proposal, result):
        assert proposal.token == result.token
        self.results.append(result)


def test_joint_objective_feedback_and_budget_admission(tmp_path, joint_evaluator):
    scenarios = (Scenario("first", {"weight": 1}), Scenario("second", {"weight": 3}))
    executor = RecordingExecutor()
    adapter = FeedbackOptimiser()
    campaign = Campaign(problem(scenarios=scenarios), tmp_path / "campaign", executor)
    result = campaign.optimise(
        optimiser=adapter, evaluator=joint_evaluator, n_trials=10, max_simulations=5
    )
    # Two scenarios must fit before asking: the fifth slot cannot admit another candidate.
    assert result.stop_reason == "simulation_budget"
    assert result.run_attempts == len(executor.requests) == 4
    assert adapter.asked == 2
    assert [trial.value for trial in result.trials] == pytest.approx([2, 0.8])
    assert result.best.parameters == {"x": pytest.approx(0.2)}
    assert result.best.metrics["signals"] == {
        "first": pytest.approx(0.2),
        "second": pytest.approx(0.6),
    }
    for i, trial in enumerate(result.trials):
        record = json.loads((campaign.storage / "optimiser" / f"trial-{i:06d}.json").read_text())
        candidate = json.loads(
            (campaign.storage / "candidates" / trial.candidate_id / "candidate.json").read_text()
        )
        assert record["feedback"] == "delivered"
        assert candidate["trial"]["token"] == record["proposal"]["token"] == trial.token
    with pytest.raises(FileExistsError):
        campaign.optimise(optimiser=FeedbackOptimiser(), evaluator=joint_evaluator, n_trials=1)


def test_target_stops_after_feedback_and_budget_does_not_mean_success(tmp_path, joint_evaluator):
    adapter = FeedbackOptimiser()
    campaign = Campaign(problem(), tmp_path / "success", RecordingExecutor())
    result = campaign.optimise(
        optimiser=adapter, evaluator=joint_evaluator, n_trials=10, target_value=0.1
    )
    assert result.stop_reason == "target_reached" and result.target_met is True
    assert result.target_value == 0.1 and result.run_attempts == adapter.asked == 2
    assert len(adapter.results) == 2 and result.best.value == pytest.approx(0.05)
    ledger = json.loads((campaign.storage / "optimiser/trial-000001.json").read_text())
    assert ledger["feedback"] == "delivered"
    campaign = Campaign(problem(), tmp_path / "unmet", RecordingExecutor())
    result = campaign.optimise(
        optimiser=FeedbackOptimiser(), evaluator=joint_evaluator, n_trials=2, target_value=0.001
    )
    assert result.stop_reason == "trial_budget" and result.target_met is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_target_rejected_before_optimiser_starts(tmp_path, joint_evaluator, value):
    adapter = FeedbackOptimiser()
    campaign = Campaign(problem(), tmp_path / "campaign", RecordingExecutor())
    with pytest.raises(ValueError, match="finite real scalar"):
        campaign.optimise(
            optimiser=adapter, evaluator=joint_evaluator, n_trials=2, target_value=value
        )
    assert adapter.asked == 0 and campaign.run_attempts == 0


def synthetic_dip_spectrum():
    from gprMax.toolboxes.Optimisation import PortSpectrum

    frequency = np.arange(0.5e9, 1.51e9, 10e6)
    power = 0.01 + ((frequency - 1.006e9) / 100e6) ** 2
    return PortSpectrum(
        frequency,
        np.sqrt(power).astype(complex),
        np.ones(len(frequency), dtype=bool),
        50 + 1j * (frequency - 1e9) / 1e6,
        np.ones(len(frequency), dtype=bool),
        50,
        -80,
        Path("synthetic"),
        "/ports/feed",
        10e6,
    )


def test_resonance_objective_finds_frequency_between_bins():
    from gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole import resonance_metrics

    metrics = resonance_metrics(synthetic_dip_spectrum(), 1e9, 10e6)
    assert metrics["s11_minimum_hz"] == 1.01e9
    assert metrics["resonance_frequency_hz"] == pytest.approx(1.006e9)
    assert metrics["frequency_error_hz"] == pytest.approx(6e6)
    assert metrics["objective_unit"] == "Hz"


@pytest.mark.parametrize("case", ["invalid_band", "coarse_resolution", "boundary", "shallow"])
def test_resonance_objective_rejects_unresolved_dips(case):
    from dataclasses import replace

    from gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole import resonance_metrics

    spectrum = synthetic_dip_spectrum()
    if case == "invalid_band":
        spectrum.valid[15] = False
    elif case == "coarse_resolution":
        spectrum = replace(spectrum, independent_frequency_resolution_hz=40e6)
    elif case == "boundary":
        spectrum = replace(spectrum, s11=(spectrum.frequency / 1e9).astype(complex))
    else:
        spectrum = replace(spectrum, s11=np.sqrt(np.abs(spectrum.s11) ** 2 + 0.2).astype(complex))
    with pytest.raises(ValueError):
        resonance_metrics(spectrum, 1e9, 10e6)


def test_axial_refinement_preserves_radius_and_transverse_mesh():
    import gprMax
    from gprMax.toolboxes.Optimisation import BuildContext
    from gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole import build_model, settings

    coarse, fine = settings(), settings(axial_refinement=2, radius_m=0.000599584916)
    assert coarse["radius"] == fine["radius"] and coarse["dl"] == fine["dl"]
    assert fine["dz"] == coarse["dz"] / 2
    context = BuildContext(Path("."), Path("output"), 0, 1)
    model = build_model({"arm_cells": 45}, Scenario("free_space", fine), context)
    grid = next(x for x in model.scene.single_use_objects if isinstance(x, gprMax.Discretisation))
    pml = next(x for x in model.scene.single_use_objects if isinstance(x, gprMax.PMLThickness))
    assert grid.kwargs["p1"] == (coarse["dl"], coarse["dl"], coarse["dz"] / 2)
    assert pml.thickness == (8, 8, 16, 8, 8, 16)
    assert model.effective_parameters["total_length_m"] == pytest.approx(91 * fine["dz"])


@pytest.mark.parametrize("failure", ["simulation", "objective"])
def test_failed_trials_are_reported_without_numerical_penalties(tmp_path, joint_evaluator, failure):
    executor = RecordingExecutor("default" if failure == "simulation" else None)
    adapter = FeedbackOptimiser()
    campaign = Campaign(problem(), tmp_path / "campaign", executor)
    result = campaign.optimise(
        optimiser=adapter,
        evaluator="joint_objective:invalid" if failure == "objective" else joint_evaluator,
        n_trials=3,
    )
    assert result.stop_reason == "trial_failed"
    assert result.best is None and result.run_attempts == 1
    assert adapter.asked == 1 and len(adapter.results) == 1
    assert adapter.results[0].value is None
    assert adapter.results[0].status == "failed"
    assert adapter.results[0].failure["kind"].startswith(failure)


def test_optuna_native_parameters_and_failed_trial_state():
    optuna = pytest.importorskip("optuna")
    from gprMax.toolboxes.Optimisation import OptunaTPE, TrialResult

    space = ParameterSpace(
        {
            "sigma": Real(1e-4, 1e2, scale="log"),
            "n": Integer(2, 9),
            "material": Categorical(("air", "soil")),
            "length": Real(0.1, 0.2),
        }
    )
    adapter = OptunaTPE(seed=7)
    metadata = adapter.initialise(space)
    proposal = adapter.ask()
    assert space.validate(proposal.parameters) == proposal.parameters
    trial = adapter.study.trials[0]
    assert trial.distributions["sigma"].log
    assert trial.distributions["sigma"].low == 1e-4 and trial.distributions["sigma"].high == 1e2
    assert isinstance(trial.params["n"], int)
    assert metadata["capabilities"]["resume"]  # Via the optional campaign checkpoint.
    with pytest.raises(RuntimeError, match="pending"):
        adapter.ask()
    feedback = TrialResult(
        proposal.token,
        proposal.parameters,
        "000001",
        "failed",
        failure={"kind": "invalid_spectrum"},
    )
    adapter.tell(proposal, feedback)
    assert adapter.study.trials[0].state == optuna.trial.TrialState.FAIL
    assert adapter.study.trials[0].value is None
    assert adapter.study.trials[0].user_attrs["candidate_id"] == "000001"
    with pytest.raises(ValueError, match="pending"):
        adapter.tell(proposal, feedback)


def test_optuna_trial_mapping_and_duplicate_proposals(tmp_path, joint_evaluator):
    optuna = pytest.importorskip("optuna")
    from gprMax.toolboxes.Optimisation import OptunaTPE

    adapter = OptunaTPE(seed=0, n_startup_trials=2)
    # One mesh-representable design makes repeat-proposal semantics unambiguous.
    definition = Problem(ParameterSpace({"x": Integer(1, 1)}), "unused:builder")
    executor = RecordingExecutor()
    campaign = Campaign(definition, tmp_path / "campaign", executor)
    result = campaign.optimise(optimiser=adapter, evaluator=joint_evaluator, n_trials=4)
    assert len(result.trials) == result.run_attempts == 4
    assert len({r.candidate_id for r in result.trials}) == 4
    assert all(
        t.state == optuna.trial.TrialState.COMPLETE and t.value == 1 for t in adapter.study.trials
    )
    assert [t.user_attrs["candidate_id"] for t in adapter.study.trials] == [
        r.candidate_id for r in result.trials
    ]


def test_optuna_proposals_respond_to_objective_feedback():
    pytest.importorskip("optuna")
    from gprMax.toolboxes.Optimisation import OptunaTPE, TrialResult

    proposals = []
    for target in (0.1, 0.9):
        adapter = OptunaTPE(seed=4, n_startup_trials=2)
        adapter.initialise(ParameterSpace({"x": Real(0, 1)}))
        values = []
        for i in range(8):
            trial = adapter.ask()
            x = trial.parameters["x"]
            values.append(x)
            adapter.tell(
                trial,
                TrialResult(trial.token, trial.parameters, str(i), "complete", (x - target) ** 2),
            )
        proposals.append(values)
    assert proposals[0][:2] == proposals[1][:2]
    assert proposals[0][2:] != proposals[1][2:]


def test_feedback_delivery_error_preserves_committed_objective(tmp_path, joint_evaluator):
    class BrokenFeedback(FeedbackOptimiser):
        def tell(self, proposal, result):
            raise RuntimeError("provider unavailable")

    campaign = Campaign(problem(), tmp_path / "campaign", RecordingExecutor())
    with pytest.raises(RuntimeError, match="provider unavailable"):
        campaign.optimise(optimiser=BrokenFeedback(), evaluator=joint_evaluator, n_trials=3)
    record = json.loads((campaign.storage / "optimiser/trial-000000.json").read_text())
    assert record["result"]["status"] == "complete" and record["result"]["value"] == 0.5
    assert record["feedback"] == "error"
    assert (
        json.loads((campaign.storage / "optimiser/result.json").read_text())["stop_reason"]
        == "error"
    )


@pytest.mark.integration
def test_fresh_models_have_distinct_geometry_materials_and_repeatable_fields(tmp_path):
    from gprMax.toolboxes.Optimisation.examples import dielectric_block

    definition = Problem(
        ParameterSpace({"width": Real(0.008, 0.016), "permittivity": Real(2, 8)}),
        "gprMax.toolboxes.Optimisation.examples.dielectric_block:build_model",
        dependencies=(Path(dielectric_block.__file__),),
    )
    campaign = Campaign(definition, tmp_path / "block")
    a = {"width": 0.008, "permittivity": 4}
    b = {"width": 0.016, "permittivity": 6}
    results = [r[0] for r in campaign.evaluate([a, b, a])]
    for run in results:
        assert run.status == "complete", (run.record, (run.directory / "stderr.log").read_text())
        assert run.record["timings"]["solver_call_seconds"] > 0
        assert run.record["provenance"]["gprMax"]["version"]
        assert run.record["artifacts"]["output"]["sha256"]
    assert len({r.record["worker_pid"] for r in results}) == 3
    assert (
        results[0].record["model_metadata"]["block_end"][0]
        < results[1].record["model_metadata"]["block_end"][0]
    )
    assert results[0].record["effective_parameters"]["permittivity"] == 4
    assert results[1].record["effective_parameters"]["permittivity"] == 6
    traces = []
    for run in results:
        trace = read_receiver(run.output_file, "received", "Ez")
        with h5py.File(run.output_file) as output:
            receivers = [r for r in output["rxs"].values() if r.attrs["Name"] == "received"]
            assert len(receivers) == 1
            data = receivers[0]["Ez"]
            assert data.attrs["SampleInterval"] == output.attrs["dt"]
            traces.append(data[...])
            np.testing.assert_array_equal(trace.values, data[...])
            assert trace.unit == "V/m"
            assert np.isfinite(traces[-1]).all()
            assert np.max(np.abs(traces[-1])) > 0
    assert np.linalg.norm(traces[0] - traces[1]) > 0.01 * np.linalg.norm(traces[0])
    np.testing.assert_array_equal(traces[0], traces[2])


class RecordingBatch:
    batch_size = 3

    def __init__(self, generations=2):
        self.generations = generations
        self.asked = 0
        self.feedback = []

    def initialise(self, parameters):
        return {"adapter": "recording_batch", "capabilities": {"batches": True}}

    @property
    def finished(self):
        return len(self.feedback) >= self.generations

    def ask_batch(self):
        from gprMax.toolboxes.Optimisation import Proposal

        assert self.asked == len(self.feedback)
        self.asked += 1
        return tuple(Proposal(f"{self.asked}-{i}", {"x": x}) for i, x in enumerate((0.1, 0.5, 0.9)))

    def tell_batch(self, proposals, results):
        assert [p.token for p in proposals] == [r.token for r in results]
        self.feedback.append(results)


def test_population_target_is_checked_after_full_batch_feedback(tmp_path, joint_evaluator):
    adapter = RecordingBatch()
    campaign = Campaign(problem(), tmp_path / "batch", RecordingExecutor())
    result = campaign.optimise(
        optimiser=adapter, evaluator=joint_evaluator, n_trials=9, target_value=0.2
    )
    assert result.stop_reason == "target_reached" and result.target_met
    assert result.run_attempts == 3 and len(adapter.feedback) == 1
    records = [
        json.loads(p.read_text())
        for p in sorted((campaign.storage / "optimiser").glob("trial-*.json"))
    ]
    assert all(r["feedback"] == "delivered" for r in records)
    assert [r["member"] for r in records] == [0, 1, 2]


@pytest.mark.parametrize(
    "n_trials,max_simulations,attempts,reason",
    [(2, 100, 0, "trial_budget"), (9, 5, 0, "simulation_budget"), (9, 7, 6, "simulation_budget")],
)
def test_population_budget_admits_all_scenarios_before_ask(
    tmp_path, joint_evaluator, n_trials, max_simulations, attempts, reason
):
    adapter = RecordingBatch()
    campaign = Campaign(
        problem(scenarios=(Scenario("a"), Scenario("b"))), tmp_path / "batch", RecordingExecutor()
    )
    result = campaign.optimise(
        optimiser=adapter,
        evaluator=joint_evaluator,
        n_trials=n_trials,
        max_simulations=max_simulations,
    )
    assert result.stop_reason == reason and result.run_attempts == attempts
    assert adapter.asked == attempts // 6


def test_population_native_termination_and_fixed_budget_target(tmp_path, joint_evaluator):
    adapter = RecordingBatch(generations=2)
    campaign = Campaign(problem(), tmp_path / "batch", RecordingExecutor())
    result = campaign.optimise(
        optimiser=adapter,
        evaluator=joint_evaluator,
        n_trials=12,
        target_value=0.2,
        stop_on_target=False,
    )
    assert result.stop_reason == "optimiser_finished" and result.target_met
    assert result.run_attempts == 6


def test_failed_population_cancels_unstarted_members_without_penalties(tmp_path, joint_evaluator):
    adapter = RecordingBatch()
    campaign = Campaign(problem(), tmp_path / "batch", RecordingExecutor("default"))
    result = campaign.optimise(optimiser=adapter, evaluator=joint_evaluator, n_trials=6)
    assert result.stop_reason == "trial_failed" and result.run_attempts == 1
    assert [r.status for r in result.trials] == ["failed", "cancelled", "cancelled"]
    assert all(r.value is None for r in adapter.feedback[0])
    assert all(r.candidate_id is None for r in result.trials[1:])


def test_batch_delivery_error_keeps_all_committed_objectives(tmp_path, joint_evaluator):
    class BrokenBatch(RecordingBatch):
        def tell_batch(self, proposals, results):
            raise RuntimeError("batch provider failed")

    campaign = Campaign(problem(), tmp_path / "batch", RecordingExecutor())
    with pytest.raises(RuntimeError, match="batch provider failed"):
        campaign.optimise(optimiser=BrokenBatch(), evaluator=joint_evaluator, n_trials=6)
    records = [
        json.loads(p.read_text()) for p in (campaign.storage / "optimiser").glob("trial-*.json")
    ]
    assert len(records) == 3
    assert all(r["result"]["status"] == "complete" and r["feedback"] == "error" for r in records)


def test_numeric_coordinates_physical_scales_and_integer_rounding():
    from gprMax.toolboxes.Optimisation.population import NumericCoordinates

    space = ParameterSpace(
        {"sigma": Real(1e-4, 1e2, scale="log"), "n": Integer(1, 4), "fixed": Integer(2, 2)}
    )
    codec = NumericCoordinates(space)
    result = codec.decode([0.5, 0.5, 0.2])
    assert result == {"sigma": pytest.approx(0.1), "n": 3, "fixed": 2}
    assert codec.repair([0.5, 0.5, 0.2]) == pytest.approx([0.5, 2 / 3, 0.5])
    with pytest.raises(ValueError, match="categorical"):
        NumericCoordinates(ParameterSpace({"kind": Categorical(("a", "b"))}))


@pytest.mark.parametrize("name", ["ga", "pso", "de"])
def test_pymoo_population_feedback_controls_next_generation(name):
    pytest.importorskip("pymoo")
    from gprMax.toolboxes.Optimisation import TrialResult, make_optimiser

    sequences = []
    for target in (0.1, 0.9):
        adapter = make_optimiser(name, seed=8, population_size=6, generations=3)
        adapter.initialise(ParameterSpace({"x": Real(0, 1)}))
        sequence = []
        for generation in range(3):
            proposals = adapter.ask_batch()
            sequence.append([p.parameters["x"] for p in proposals])
            with pytest.raises(RuntimeError, match="pending"):
                adapter.ask_batch()
            feedback = tuple(
                TrialResult(
                    p.token, p.parameters, str(i), "complete", (p.parameters["x"] - target) ** 2
                )
                for i, p in enumerate(proposals)
            )
            with pytest.raises(ValueError, match="order"):
                adapter.tell_batch(proposals, tuple(reversed(feedback)))
            adapter.tell_batch(proposals, feedback)
            assert adapter.algorithm.evaluator.n_eval == 6 * (generation + 1)
        assert adapter.finished and adapter.ask_batch() == ()
        sequences.append(sequence)
    np.testing.assert_array_equal(sequences[0][0], sequences[1][0])
    assert not np.array_equal(sequences[0][1:], sequences[1][1:])


@pytest.mark.parametrize("name", ["ga", "pso", "de"])
def test_pymoo_numeric_mixed_dimensions_and_abort(name):
    pytest.importorskip("pymoo")
    from gprMax.toolboxes.Optimisation import TrialResult, make_optimiser

    space = ParameterSpace({"n": Integer(1, 20), "sigma": Real(1e-4, 1e2, scale="log")})
    adapter = make_optimiser(name, seed=3)
    adapter.initialise(space)
    proposals = adapter.ask_batch()
    assert all(space.validate(p.parameters) == p.parameters for p in proposals)
    for p in proposals:
        assert adapter.coordinates.decode(p.metadata["evaluated_coordinates"]) == p.parameters
    if name == "pso":
        assert all(
            p.metadata["proposed_coordinates"] == p.metadata["evaluated_coordinates"]
            for p in proposals
        )
    adapter.tell_batch(
        proposals,
        tuple(
            TrialResult(p.token, p.parameters, None, "failed", failure={"kind": "test"})
            for p in proposals
        ),
    )
    assert adapter.finished and adapter.algorithm.evaluator.n_eval == 0


def test_rf_native_dimensions_feedback_and_failure():
    pytest.importorskip("skopt")
    from gprMax.toolboxes.Optimisation import SkoptRF, TrialResult

    space = ParameterSpace(
        {
            "x": Real(0.001, 10, scale="log"),
            "n": Integer(1, 8),
            "kind": Categorical(("a", "b")),
            "fixed": Integer(2, 2),
        }
    )
    adapter = SkoptRF(seed=3, n_initial_points=2, n_points=100)
    adapter.initialise(space)
    for i in range(4):
        p = adapter.ask()
        assert space.validate(p.parameters) == p.parameters
        adapter.tell(
            p,
            TrialResult(
                p.token,
                p.parameters,
                str(i),
                "complete",
                p.parameters["x"] ** 2 + p.parameters["n"],
            ),
        )
    assert len(adapter.optimiser.models) == 3 and len(adapter.optimiser.yi) == 4
    p = adapter.ask()
    adapter.tell(p, TrialResult(p.token, p.parameters, None, "failed", failure={"kind": "test"}))
    assert len(adapter.optimiser.yi) == 4
    assert adapter.optimiser.space.dimensions[0].prior == "log-uniform"


def test_rf_proposals_depend_on_feedback():
    pytest.importorskip("skopt")
    from gprMax.toolboxes.Optimisation import SkoptRF, TrialResult

    sequences = []
    for target in (0.1, 0.9):
        adapter = SkoptRF(seed=8, n_initial_points=3, n_points=300)
        adapter.initialise(ParameterSpace({"x": Real(0, 1)}))
        values = []
        for i in range(7):
            p = adapter.ask()
            values.append(p.parameters["x"])
            adapter.tell(
                p,
                TrialResult(
                    p.token, p.parameters, str(i), "complete", (p.parameters["x"] - target) ** 2
                ),
            )
        sequences.append(values)
    assert sequences[0][:3] == sequences[1][:3]
    assert sequences[0][3:] != sequences[1][3:]


def test_explicit_one_mm_dipole_geometry():
    import gprMax
    from gprMax.toolboxes.Optimisation import BuildContext
    from gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole import build_model, settings

    config = settings(dz_m=0.001, cycles=100)
    assert config["dz"] == 0.001 and (config["lower"], config["upper"]) == (54, 83)
    model = build_model(
        {"arm_cells": 69},
        Scenario("free_space", config),
        BuildContext(Path("."), Path("output"), 0, 1),
    )
    assert model.effective_parameters["total_length_m"] == pytest.approx(0.139)
    assert model.effective_parameters["gap_m"] == 0.001
    grid = next(x for x in model.scene.single_use_objects if isinstance(x, gprMax.Discretisation))
    assert grid.kwargs["p1"][2] == 0.001
    pml = next(x for x in model.scene.single_use_objects if isinstance(x, gprMax.PMLThickness))
    assert pml.thickness == (8, 8, 24, 8, 8, 24)
    with pytest.raises(ValueError, match="either"):
        settings(dz_m=0.001, axial_refinement=2)
    for bad in (0, -0.001, float("nan"), 0.01):
        with pytest.raises(ValueError, match="dz_m"):
            settings(dz_m=bad)


@pytest.mark.parametrize("name", ["ga", "pso", "de", "rf"])
def test_adapters_reject_mutated_proposals(name):
    pytest.importorskip("pymoo" if name != "rf" else "skopt")
    from gprMax.toolboxes.Optimisation import TrialResult, make_optimiser

    adapter = make_optimiser(name, seed=7)
    adapter.initialise(ParameterSpace({"x": Real(0, 1)}))
    batch = adapter.ask_batch() if hasattr(adapter, "ask_batch") else (adapter.ask(),)
    batch[0].parameters["x"] = 0.314159
    results = tuple(TrialResult(p.token, p.parameters, None, "complete", 1.0) for p in batch)
    with pytest.raises(ValueError, match="modified"):
        if hasattr(adapter, "tell_batch"):
            adapter.tell_batch(batch, results)
        else:
            adapter.tell(batch[0], results[0])
