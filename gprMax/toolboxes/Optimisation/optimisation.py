"""Coordinator loop: proposed parameters -> simulations -> objective -> feedback.

Read run_optimisation to follow one complete search. It asks the selected
adapter for candidates, sends them to Campaign, runs PreparedEvaluator,
records TrialResults and returns matching scores to the adapter.
The numerical criterion is defined by the user's evaluator, not here.

Trial budgets count proposals; simulation budgets count worker attempts.
Batches are admitted and returned as complete groups. Checkpoints store
the adapter before ask so recovery can reproduce an interrupted proposal.
JSON records describe progress; they alone cannot restore library state.
"""

import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from ._storage import SCHEMA_VERSION, json_copy, sha256, write_json
from .optimisers import BatchOptimiser, ObjectiveResult, Optimiser, Proposal, TrialResult
from .processing import PreparedEvaluator


@dataclass(frozen=True)
class OptimisationResult:
    """Search summary, including the best successful trial and every trial status.

    stop_reason describes budget/target/native termination or a failure.
    best may be None if nothing succeeded. run_attempts counts actual worker
    attempts in this session, which can differ from len(trials) with retries,
    multiple scenarios or reuse. target_met reports the threshold comparison
    even when stop_on_target=False; it does not prove global convergence.
    """

    stop_reason: str
    best: TrialResult | None
    trials: tuple[TrialResult, ...]
    run_attempts: int
    optimiser: dict
    target_value: float | None = None
    target_met: bool | None = None

    @property
    def best_parameters(self):
        """Best evaluated parameter dictionary, or None if no trial succeeded."""
        return dict(self.best.parameters) if self.best is not None else None

    @property
    def best_value(self):
        """Lowest objective value observed, or None if no trial succeeded."""
        return self.best.value if self.best is not None else None


def run_optimisation(
    campaign,
    *,
    optimiser: Optimiser | BatchOptimiser,
    evaluator,
    n_trials: int,
    max_simulations: int | None = None,
    seed=0,
    on_failure="stop",
    target_value=None,
    stop_on_target=True,
    checkpoint=False,
    resume=False,
):
    """Propose -> simulate scenarios -> evaluate -> deliver matching feedback.

    Single-proposal adapters remain supported. Population adapters expose
    batch_size, finished, ask_batch and tell_batch. Admit the full declared batch
    before asking; complete it before target stopping. On failure with stop
    policy, cancel unstarted members and deliver the full batch without penalties.
    No partial population is submitted to an algorithm as a successful generation.

    n_trials counts proposals, max_simulations counts worker attempts. Exact
    duplicates may use an opt-in cache. Opt-in native checkpoints support recovery.
    stop_on_target=False permits fixed-budget comparisons while reporting target_met.
    """
    # 1. Validate stopping settings and prepare the objective callback.
    # These thresholds/budgets are separate from how the objective is calculated.
    for name, value in (("n_trials", n_trials), ("max_simulations", max_simulations)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if n_trials is None:
        raise ValueError("n_trials is required")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if on_failure not in ("stop", "continue"):
        raise ValueError("on_failure must be stop or continue")
    if not isinstance(stop_on_target, bool):
        raise ValueError("stop_on_target must be a boolean")
    if target_value is not None:
        target_value = ObjectiveResult(target_value).value
    evaluate = PreparedEvaluator(evaluator)
    evaluator_info = evaluate.info
    directory = campaign.storage / "optimiser"
    from .checkpoint import load_checkpoint, save_checkpoint

    state = None
    # 2. Restore a compatible native optimiser, or initialise a new one.
    # The checkpoint contains library state; JSON progress files do not.
    if resume:
        if not campaign._resuming:
            raise ValueError("Use Campaign.resume before resuming an optimiser")
        state = load_checkpoint(directory)
        if state["evaluator"] != evaluator_info:
            raise ValueError("Evaluator definition changed since checkpoint")
        expected = {
            "seed": seed,
            "on_failure": on_failure,
            "target_value": target_value,
            "stop_on_target": stop_on_target,
        }
        if state["settings"] != expected:
            raise ValueError("Resume requires the original optimisation settings")
        sources = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
        if state["toolbox_sources"] != sources:
            raise ValueError("Toolbox source changed since checkpoint; start a new campaign")
        optimiser, metadata = state["optimiser"], state["metadata"]
        if metadata.get("library") and metadata.get("version"):
            from importlib.metadata import version

            if version(metadata["library"]) != metadata["version"]:
                raise ValueError("Optimiser library version changed since checkpoint")
        checkpoint = True
    else:
        if directory.exists():
            raise FileExistsError(
                "This campaign already has an optimisation session; use explicit checkpoint recovery"
            )
        metadata = json_copy(optimiser.initialise(campaign.problem.parameters))
    batched = callable(getattr(optimiser, "ask_batch", None))
    if batched and not callable(getattr(optimiser, "tell_batch", None)):
        raise TypeError("A batch adapter needs tell_batch")
    if (
        on_failure == "continue"
        and metadata.get("capabilities", {}).get("continue_after_failure") is False
    ):
        raise ValueError("This optimiser requires on_failure='stop'")
    directory.mkdir(exist_ok=resume)
    if not resume:
        write_json(
            directory / "session.json",
            {
                "schema_version": SCHEMA_VERSION,
                "optimiser": metadata,
                "evaluator": evaluator_info,
                "n_trials": n_trials,
                "max_simulations": max_simulations,
                "simulation_seed": seed,
                "on_failure": on_failure,
                "target_value": target_value,
                "stop_on_target": stop_on_target,
                "policy": {
                    "duplicate_proposals": "cache" if campaign.cache else "fresh_runs",
                    "resume": checkpoint,
                    "batch_admission": "full_declared_size",
                    "target_check": "after_batch_feedback",
                },
            },
        )
    start_attempts = state["start_attempts"] if state else campaign.run_attempts
    results = state["results"] if state else []
    seen = state["seen"] if state else set()
    stop_reason, batch_number = "trial_budget", state["batch_number"] if state else 0

    def snapshot(finished_reason=None):
        """Save state before asking, or mark a fully finished session.

        The source hashes deliberately include comments/docstrings as file
        contents: any toolbox edit invalidates native checkpoint recovery.
        """
        if checkpoint:
            save_checkpoint(
                directory,
                {
                    "optimiser": optimiser,
                    "metadata": metadata,
                    "evaluator": evaluator_info,
                    "results": results,
                    "seen": seen,
                    "batch_number": batch_number,
                    "start_attempts": start_attempts,
                    "finished_reason": finished_reason,
                    "settings": {
                        "seed": seed,
                        "on_failure": on_failure,
                        "target_value": target_value,
                        "stop_on_target": stop_on_target,
                    },
                    "toolbox_sources": {
                        p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
                    },
                },
            )

    def save_summary(reason, error=None):
        """Commit the best successful score so far and return the corresponding result object."""
        completed = [r for r in results if r.status == "complete"]
        best = min(completed, key=lambda r: r.value) if completed else None
        target_met = (
            None if target_value is None else bool(best is not None and best.value <= target_value)
        )
        summary = OptimisationResult(
            reason,
            best,
            tuple(results),
            campaign.run_attempts - start_attempts,
            metadata,
            target_value,
            target_met,
        )
        record = asdict(summary)
        record.update(schema_version=SCHEMA_VERSION, error=error)
        write_json(directory / "result.json", record)
        return summary

    if state and state.get("finished_reason"):
        return save_summary(state["finished_reason"])
    try:
        while len(results) < n_trials:
            if batched and optimiser.finished:
                stop_reason = "optimiser_finished"
                break
            # 3. Reserve enough budget for a full batch and all its scenarios.
            # Asking first would advance an optimiser we might not be able to evaluate.
            size = optimiser.batch_size if batched else 1
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError("batch_size must be a positive integer")
            if len(results) + size > n_trials:
                stop_reason = "trial_budget"
                break
            used = campaign.run_attempts - start_attempts
            recovering = resume and (directory / f"trial-{len(results):06d}.json").exists()
            required = size * len(campaign._scenarios)
            if recovering:
                previous_tokens = [
                    json.loads((directory / f"trial-{len(results) + i:06d}.json").read_text())[
                        "proposal"
                    ]["token"]
                    for i in range(size)
                    if (directory / f"trial-{len(results) + i:06d}.json").exists()
                ]
                required = campaign.remaining_attempts(previous_tokens) + (
                    size - len(previous_tokens)
                ) * len(campaign._scenarios)
            # Admission is conservative: allow the configured retries for every task,
            # even though cache hits or successful first attempts may spend less.
            required *= getattr(campaign.execution, "max_attempts", 1)
            if max_simulations is not None and used + required > max_simulations:
                stop_reason = "simulation_budget"
                break
            snapshot()
            # 4. Ask for complete parameter combinations and preserve their tokens.
            # A token belongs to the optimiser; Campaign assigns a separate candidate ID.
            proposals = tuple(optimiser.ask_batch()) if batched else (optimiser.ask(),)
            if not proposals:
                stop_reason = "optimiser_finished"
                break
            if len(proposals) > size:
                raise ValueError("Optimiser exceeded its declared batch_size")
            batch_number += 1
            batch_path = directory / f"batch-{batch_number:06d}.json"
            batch_record = {
                "schema_version": SCHEMA_VERSION,
                "batch": batch_number,
                "proposals": [],
                "feedback": "pending",
            }
            records = []
            for offset, proposal in enumerate(proposals):
                if not isinstance(proposal, Proposal) or proposal.token in seen:
                    raise ValueError("Optimiser must return a Proposal with a unique trial token")
                seen.add(proposal.token)
                path = directory / f"trial-{len(results) + offset:06d}.json"
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "proposal": asdict(proposal),
                    "batch": batch_number,
                    "member": offset,
                    "result": None,
                    "feedback": "pending",
                }
                if recovering and path.exists():
                    previous = json.loads(path.read_text())
                    if previous["proposal"] != record["proposal"]:
                        raise ValueError(
                            "Restored optimiser did not reproduce its recorded proposal"
                        )
                records.append((path, record))
                batch_record["proposals"].append(proposal.token)
            # Commit the entire proposed population before starting any solver.
            write_json(batch_path, batch_record)
            for path, record in records:
                write_json(path, record)
            batch_results, interrupted, abort = [], None, False
            scheduled = None
            if callable(getattr(campaign.execution, "map", None)):
                # Keep allocation and ledger mutation on this coordinator.
                # Invalid proposals use the serial validation/error path below.
                try:
                    values = [campaign.problem.parameters.validate(p.parameters) for p in proposals]
                    evaluate.verify()
                except Exception:
                    pass
                else:
                    scheduled = campaign.evaluate_batch(
                        values,
                        seed=seed,
                        stop_on_failure=on_failure == "stop",
                        trials=[
                            {
                                "token": p.token,
                                "record": str(path.relative_to(campaign.storage)),
                                "batch": batch_number,
                                "metadata": p.metadata,
                            }
                            for p, (path, _) in zip(proposals, records)
                        ],
                    )
            # 5. Gather each candidate's scenarios, then call the user objective.
            # Parallel completion order must not change proposal-to-score identity.
            member_index = -1
            for proposal, (path, record) in zip(proposals, records):
                member_index += 1
                candidate_id = None
                phase = "validation"
                started = time.perf_counter()
                if abort and scheduled is None:
                    result = TrialResult(
                        proposal.token,
                        proposal.parameters,
                        None,
                        "cancelled",
                        failure={
                            "kind": "batch_aborted",
                            "message": "Earlier member failed or was interrupted",
                        },
                    )
                else:
                    try:
                        values = campaign.problem.parameters.validate(proposal.parameters)
                        evaluate.verify()
                        phase = "simulation"
                        runs = (
                            scheduled[member_index]
                            if scheduled is not None
                            else campaign.evaluate_one(
                                values,
                                seed=seed,
                                trial={
                                    "token": proposal.token,
                                    "record": str(path.relative_to(campaign.storage)),
                                    "batch": batch_number,
                                    "metadata": proposal.metadata,
                                },
                            )
                        )
                        candidate_id = runs[0].candidate_id
                        failures = {
                            run.scenario_id: run.record.get("failure")
                            for run in runs
                            if run.status != "complete"
                        }
                        if failures:
                            status = (
                                "cancelled"
                                if all(run.status == "cancelled" for run in runs)
                                else "failed"
                            )
                            result = TrialResult(
                                proposal.token,
                                values,
                                candidate_id,
                                status,
                                failure={"kind": "simulation_failed", "scenarios": failures},
                            )
                        else:
                            phase = "objective"
                            processing_dir = (
                                campaign.storage / "candidates" / candidate_id / "processing"
                            )
                            objective = evaluate.recover_or_run(
                                dict(values),
                                {run.scenario_id: run for run in runs},
                                processing_dir,
                                recover=resume,
                            )
                            if not isinstance(objective, ObjectiveResult):
                                raise TypeError("Evaluator must return ObjectiveResult")
                            result = TrialResult(
                                proposal.token,
                                values,
                                candidate_id,
                                "complete",
                                objective.value,
                                objective.metrics,
                            )
                    except BaseException as exc:
                        interrupted = exc if not isinstance(exc, Exception) else None
                        result = TrialResult(
                            proposal.token,
                            proposal.parameters,
                            candidate_id,
                            "cancelled" if interrupted else "failed",
                            failure={
                                "kind": phase + "_error",
                                "message": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                        )
                if interrupted or (result.status != "complete" and on_failure == "stop"):
                    abort = True
                results.append(result)
                batch_results.append(result)
                record.update(result=asdict(result), elapsed_seconds=time.perf_counter() - started)
                write_json(path, record)
                if candidate_id is not None:
                    write_json(
                        campaign.storage / "candidates" / candidate_id / "objective.json",
                        asdict(result),
                    )
                save_summary("running")
            # 6. Deliver feedback only after every member has a recorded outcome.
            # Even failed/cancelled members retain their place in the batch.
            # All objective records are durable before any population update.
            try:
                if batched:
                    optimiser.tell_batch(proposals, tuple(batch_results))
                else:
                    optimiser.tell(proposals[0], batch_results[0])
            except BaseException:
                batch_record.update(feedback="error", feedback_error=traceback.format_exc())
                write_json(batch_path, batch_record)
                for path, record in records:
                    record.update(feedback="error", feedback_error=batch_record["feedback_error"])
                    write_json(path, record)
                raise
            batch_record["feedback"] = "delivered"
            batch_record["outcome"] = (
                "complete"
                if all(r.status == "complete" for r in batch_results)
                else "failed_or_cancelled"
            )
            write_json(batch_path, batch_record)
            for path, record in records:
                record["feedback"] = "delivered"
                write_json(path, record)
            save_summary("running")
            if interrupted:
                raise interrupted
            if abort:
                stop_reason = "trial_failed"
                break
            # 7. Apply target stopping after the full batch update. Finding a good
            # early member does not submit an incomplete population to the library.
            if (
                stop_on_target
                and target_value is not None
                and any(r.status == "complete" and r.value <= target_value for r in batch_results)
            ):
                stop_reason = "target_reached"
                break
    except BaseException as exc:
        save_summary(
            "cancelled" if not isinstance(exc, Exception) else "error", traceback.format_exc()
        )
        raise
    snapshot(stop_reason)
    return save_summary(stop_reason)
