"""Own the simulations and files for one parameterised experiment.

Campaign evaluates an explicit table or receives candidates from the
optimiser loop. Each candidate is expanded into one task per Scenario.
Each task gets an attempt directory and a JSON request for a fresh Scene.
The execution backend returns RunResults; Campaign regroups them by
candidate without changing the original order.

This module owns directory IDs, progress, exact-request cache reuse and
recovery of completed simulations. optimisation.py owns the search and
processing.py calls the user's objective. See CODE_WALKTHROUGH.md for
the directory tree and the meanings of candidate, scenario and attempt.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._storage import SCHEMA_VERSION, json_copy, sha256, write_json
from .execution import ExecutionBackend, RunTask
from .models import Problem
from .runner import LocalExecutor


def _count_attempts(storage):
    """Reconstruct the consumed simulation count from durable attempt records.

    Cache/recovered results with run_attempts=0 do not spend new simulations.
    A started attempt without a final result still counts after interruption.
    """
    count = 0
    for path in (storage / "candidates").glob("*/*/attempt-*"):
        if (path / "result.json").exists():
            record = json.loads((path / "result.json").read_text())
            count += int(record.get("run_attempts", 1) > 0)
        elif (path / "state.json").exists():
            count += 1
    return count


class Campaign:
    """Coordinator for fresh-model evaluations and their saved results.

    Construct with a Problem, a new storage directory and an executor/pool.
    evaluate_one/evaluate_batch run explicit parameter combinations;
    optimise connects an adapter and objective. Only this coordinator
    allocates candidate IDs and updates shared campaign/progress records.
    """

    def __init__(
        self,
        problem: Problem,
        storage,
        execution: LocalExecutor | ExecutionBackend | None = None,
        *,
        cache=None,
    ):
        """Create a new campaign and snapshot its model/execution definition.

        storage must not already exist. Hashes describe the declared dependency
        files; later changes stop the run so different models/data are not mixed.
        Use Campaign.resume explicitly to reopen a compatible stopped campaign.
        """
        self.cache = cache
        self.problem = problem
        self.execution = execution or LocalExecutor()
        self.storage = Path(storage).expanduser().resolve()
        self._dependencies = [{"path": str(p), "sha256": sha256(p)} for p in problem.dependencies]
        self._scenarios = [s.to_dict() for s in problem.scenarios]
        definition = {
            "schema_version": SCHEMA_VERSION,
            "builder": problem.builder,
            "model_version": problem.version,
            "parameters": problem.parameters.to_dict(),
            "scenarios": self._scenarios,
            "dependencies": self._dependencies,
            "execution": self.execution.to_dict(),
            "policy": {
                "fresh_model": True,
                "cache": cache.to_dict() if cache else False,
                "resume": False,
            },
        }
        self.storage.mkdir(parents=True, exist_ok=False)
        write_json(self.storage / "campaign.json", definition)
        self._count = 0
        self.run_attempts = 0
        self._resuming = False
        self._recovered = {}

    @classmethod
    def resume(cls, problem, storage, execution=None, *, cache=None):
        """Reopen one's own stopped campaign. All previous workers must have exited.

        Resume uses the original execution configuration and verifies the model
        definition/dependencies. Completed HDF5 outputs are verified and reused;
        failed/incomplete tasks receive new attempt directories.
        """
        from .cache import SimulationCache
        from .execution import LocalPool

        self = cls.__new__(cls)
        self.storage = Path(storage).expanduser().resolve()
        definition = json.loads((self.storage / "campaign.json").read_text())
        self.problem = problem
        self._dependencies = [{"path": str(p), "sha256": sha256(p)} for p in problem.dependencies]
        self._scenarios = [s.to_dict() for s in problem.scenarios]
        current = {
            "builder": problem.builder,
            "model_version": problem.version,
            "parameters": problem.parameters.to_dict(),
            "scenarios": self._scenarios,
            "dependencies": self._dependencies,
        }
        if any(definition[key] != value for key, value in current.items()):
            raise ValueError(
                "Campaign model definition or dependency changed; start a new campaign"
            )
        saved = dict(definition["execution"])
        if execution is None:
            backend = saved.pop("backend", None)
            if backend == "local":
                execution = LocalPool(**saved)
            elif backend == "mpi":
                raise ValueError("Resume MPI inside an entered MPIPool context")
            else:
                execution = LocalExecutor(**saved)
        if json_copy(execution.to_dict()) != definition["execution"]:
            raise ValueError("Resume requires the original execution configuration")
        self.execution = execution
        saved_cache = definition["policy"].get("cache")
        self.cache = cache or (
            SimulationCache(saved_cache["directory"], namespace=saved_cache["namespace"])
            if saved_cache
            else None
        )
        if (self.cache.to_dict() if self.cache else False) != saved_cache:
            raise ValueError("Resume requires the original cache configuration")
        candidates = list((self.storage / "candidates").glob("[0-9]*"))
        self._count = max((int(p.name) for p in candidates), default=0)
        self.run_attempts = _count_attempts(self.storage)
        self._resuming, self._recovered = True, {}
        self._tokens = {}
        for path in candidates:
            record = json.loads((path / "candidate.json").read_text())
            trial = record.get("trial") or {}
            if trial.get("token"):
                self._tokens[trial["token"]] = (path, record)
        return self

    def remaining_attempts(self, tokens):
        """Count scenario results still needed when replaying recorded proposal tokens.

        Completed output hashes are checked before counting a scenario as
        reusable. The optimiser loop multiplies this count by the retry allowance
        when deciding whether the remaining simulation budget admits the batch.
        """
        required = 0
        for token in tokens:
            previous = self._tokens.get(token)
            for scenario in self._scenarios:
                reusable = False
                if previous:
                    for path in (previous[0] / scenario["id"]).glob("attempt-*/result.json"):
                        try:
                            record = json.loads(path.read_text())
                            artifact = record["artifacts"]["output"]
                            reusable |= (
                                record["status"] == "complete"
                                and sha256(path.parent / artifact["path"]) == artifact["sha256"]
                            )
                        except (OSError, ValueError, KeyError):
                            pass
                required += not reusable
        return required

    def _prepare(self, parameters, *, seed=0, trial=None):
        """Turn one parameter dictionary into a candidate directory and scenario tasks.

        trial optionally links the candidate to an optimiser token/batch record.
        Recovered candidates retain their ID and verified output; unfinished
        scenarios get a new attempt directory. No solver is launched here.
        """
        values = self.problem.parameters.validate(parameters)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        trial = json_copy(trial)
        for asset in self._dependencies:
            if sha256(asset["path"]) != asset["sha256"]:
                raise ValueError(
                    f"Declared dependency changed since campaign creation: {asset['path']}"
                )
        previous = self._tokens.get((trial or {}).get("token")) if self._resuming else None
        if previous:
            candidate_dir, previous_record = previous
            candidate_id = candidate_dir.name
            if previous_record["parameters"] != values or previous_record["seed"] != seed:
                raise ValueError("Recovered proposal differs from the recorded candidate")
        else:
            self._count += 1
            candidate_id = f"{self._count:06d}"
            candidate_dir = self.storage / "candidates" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=False)
            write_json(
                candidate_dir / "candidate.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": candidate_id,
                    "parameters": values,
                    "seed": seed,
                    "trial": trial,
                },
            )
        tasks = []
        # Every scenario uses the same candidate parameters but different fixed
        # settings. This is where one candidate becomes several simulation requests.
        for scenario in self._scenarios:
            request = {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "builder": self.problem.builder,
                "model_version": self.problem.version,
                "parameters": values,
                "scenario": json_copy(scenario),
                "seed": seed,
                "dependencies": self._dependencies,
                "execution": self.execution.to_dict(),
            }
            directory = candidate_dir / scenario["id"] / "attempt-0001"
            if previous:
                from .models import RunResult

                attempts = sorted((candidate_dir / scenario["id"]).glob("attempt-*"))
                for attempt in reversed(attempts):
                    try:
                        record = json.loads((attempt / "result.json").read_text())
                        artifact = record["artifacts"]["output"]
                        if (
                            record["status"] == "complete"
                            and sha256(attempt / artifact["path"]) == artifact["sha256"]
                        ):
                            directory = attempt
                            self._recovered[directory] = RunResult(
                                candidate_id,
                                scenario["id"],
                                directory,
                                dict(record, run_attempts=0, recovered=True),
                            )
                            break
                    except (OSError, ValueError, KeyError):
                        pass
                else:
                    number = max((int(p.name.split("-")[-1]) for p in attempts), default=0) + 1
                    directory = candidate_dir / scenario["id"] / f"attempt-{number:04d}"
            directory.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                directory.parent / "task.json",
                {"candidate_id": candidate_id, "scenario_id": scenario["id"], "status": "queued"},
            )
            tasks.append(RunTask(request, directory))
        return candidate_dir, tasks

    def evaluate_batch(self, candidates, *, seed=0, trials=None, stop_on_failure=False):
        """Run candidates across all scenarios and return rows in candidate order.

        The result is a tuple of rows; each row contains RunResults in declared
        scenario order. Tasks may complete out of order, but that cannot change
        the mapping between parameters, files and objective feedback.
        This method executes models only; it does not compute objective scores.
        """
        candidates = tuple(candidates)
        trials = (None,) * len(candidates) if trials is None else tuple(trials)
        if len(trials) != len(candidates):
            raise ValueError("One trial record is required per candidate")
        # Validate the whole explicit table before creating candidate directories.
        for candidate in candidates:
            self.problem.parameters.validate(candidate)
        prepared = [self._prepare(c, seed=seed, trial=t) for c, t in zip(candidates, trials)]
        # Flatten candidate/scenario rows for scheduling, then regroup the ordered
        # results below. A pool may finish tasks in any wall-clock order.
        tasks = tuple(task for _, row in prepared for task in row)
        completed = []

        def collect(result):
            """Record each completed task centrally and update actual attempts/progress."""
            completed.append(result)
            self.run_attempts += result.record.get("run_attempts", 1)
            write_json(
                result.directory.parent / "task.json",
                {
                    "candidate_id": result.candidate_id,
                    "scenario_id": result.scenario_id,
                    "status": result.status,
                    "attempt": result.directory.name,
                },
            )
            write_json(
                self.storage / "progress.json",
                {
                    "candidates_created": self._count,
                    "run_attempts": self.run_attempts,
                    "tasks_in_batch": len(tasks),
                    "tasks_finished": len(completed),
                    "completed": sum(r.status == "complete" for r in completed),
                    "failed": sum(r.status == "failed" for r in completed),
                    "cache_hits": sum(
                        bool(r.record.get("cache", {}).get("hit")) for r in completed
                    ),
                    "cancelled": sum(r.status == "cancelled" for r in completed),
                },
            )

        try:
            results = self._execute_tasks(tasks, stop_on_failure=stop_on_failure, on_result=collect)
        except BaseException:
            self.run_attempts = _count_attempts(self.storage)
            for directory, _ in prepared:
                write_json(
                    directory / "evaluation.json",
                    {
                        "status": "interrupted",
                        "runs": [
                            str(r.directory.relative_to(self.storage))
                            for r in completed
                            if r.candidate_id == directory.name
                        ],
                    },
                )
            raise
        if len(results) != len(tasks):
            raise ValueError("Executor must return one result for every submitted task")
        rows, offset = [], 0
        for directory, row in prepared:
            runs = tuple(results[offset : offset + len(row)])
            offset += len(row)
            for task, run in zip(row, runs):
                if (
                    run.candidate_id != task.request["candidate_id"]
                    or run.scenario_id != task.request["scenario"]["id"]
                    or run.directory.parent != task.directory.parent
                ):
                    raise ValueError("Executor returned results in the wrong task order")
            write_json(
                directory / "evaluation.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_id": directory.name,
                    "status": "complete" if all(r.status == "complete" for r in runs) else "failed",
                    "runs": [str(r.directory.relative_to(self.storage)) for r in runs],
                },
            )
            rows.append(runs)
        return tuple(rows)

    def _execute_tasks(self, tasks, *, stop_on_failure, on_result):
        """Resolve recovery/cache hits, dispatch remaining tasks and restore input order.

        With caching enabled, identical requests within this batch share one
        primary solve. aliases maps duplicate task indices to that primary;
        each duplicate still receives its own candidate result record.
        """
        if any(t.directory in self._recovered for t in tasks):
            recovered = {
                i: self._recovered.pop(t.directory)
                for i, t in enumerate(tasks)
                if t.directory in self._recovered
            }
            for result in recovered.values():
                on_result(result)
            fresh = iter(
                self._execute_tasks(
                    [t for i, t in enumerate(tasks) if i not in recovered],
                    stop_on_failure=stop_on_failure,
                    on_result=on_result,
                )
            )
            return tuple(recovered[i] if i in recovered else next(fresh) for i in range(len(tasks)))
        if self.cache is None:
            return self._submit(tasks, stop_on_failure=stop_on_failure, on_result=on_result)
        from .execution import _unsolved

        results = [None] * len(tasks)
        # primary maps a cache key to the first task index needing a solve.
        # aliases links later duplicates to it; keys lets each receive its own record.
        primary, aliases, keys = {}, {}, {}
        for i, task in enumerate(tasks):
            key = self.cache.key(task.request)
            keys[i] = key
            cached = self.cache.restore(key, task)
            if cached is not None:
                results[i] = cached
                on_result(cached)
            elif key in primary:
                aliases[i] = primary[key]
            else:
                primary[key] = i
        indices = list(primary.values())
        fresh = self._submit(
            [tasks[i] for i in indices], stop_on_failure=stop_on_failure, on_result=on_result
        )
        for index, result in zip(indices, fresh):
            results[index] = result
            self.cache.store(keys[index], result)
        for index, original in aliases.items():
            result = (
                self.cache.restore(keys[index], tasks[index])
                if results[original].status == "complete"
                else None
            )
            if result is None:
                result = _unsolved(
                    tasks[index],
                    "duplicate_not_available",
                    "Matching simulation did not produce a reusable output",
                )
            results[index] = result
            on_result(result)
        return tuple(results)

    def _submit(self, tasks, *, stop_on_failure, on_result):
        """Use a backend's ordered map, or fall back to serial run calls.

        stop_on_failure prevents queued serial work from starting; every skipped
        task still gets an explicit cancellation result in the returned sequence.
        """
        if callable(getattr(self.execution, "map", None)):
            return self.execution.map(tasks, stop_on_failure=stop_on_failure, on_result=on_result)
        from .execution import _unsolved

        results, stopped = [], False
        for task in tasks:
            result = (
                _unsolved(task) if stopped else self.execution.run(task.request, task.directory)
            )
            on_result(result)
            results.append(result)
            stopped |= stop_on_failure and result.status != "complete"
        return tuple(results)

    def evaluate_one(self, parameters, *, seed=0, trial=None):
        """Run one parameter combination and return one RunResult per declared scenario."""
        return self.evaluate_batch((parameters,), seed=seed, trials=(trial,))[0]

    def evaluate(self, candidates, *, seed=0):
        """Run an explicit table of parameter combinations without choosing them through an optimiser."""
        return self.evaluate_batch(candidates, seed=seed)

    def optimise(
        self,
        *,
        optimiser=None,
        evaluator,
        n_trials,
        max_simulations=None,
        seed=0,
        on_failure="stop",
        target_value=None,
        stop_on_target=True,
        checkpoint=False,
        resume=False,
    ):
        """Connect an adapter and objective to this campaign's simulation service.

        n_trials counts parameter proposals; max_simulations counts attempts.
        target_value/stop_on_target control stopping, not the objective formula.
        evaluator names the advanced callback or supplies an Evaluation object.
        checkpoint/resume enable explicit native optimiser recovery. Ordinary
        users normally use simple.optimise instead of configuring this layer.
        """
        from .optimisation import run_optimisation

        return run_optimisation(
            self,
            optimiser=optimiser,
            evaluator=evaluator,
            n_trials=n_trials,
            max_simulations=max_simulations,
            seed=seed,
            on_failure=on_failure,
            target_value=target_value,
            stop_on_target=stop_on_target,
            checkpoint=checkpoint,
            resume=resume,
        )
