"""User entry points for the three-part model/optimisation workflow.

1. Declare a dictionary of allowed parameter ranges.
2. Supply build_model(parameters), returning an unexecuted gprMax Scene.
3. Supply evaluate(parameters, output), returning one scalar to minimise.

optimise joins these functions to Campaign and the chosen optimiser;
simulate performs one concrete parameter combination without a search.
This module describes callbacks but never calls gprMax.run itself.
_simple.py loads the saved callbacks when the coordinator/worker needs
them. Use explicit Campaign contracts for multiple scenarios or recovery.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from ._storage import sha256
from .adapters import make_optimiser
from .campaign import Campaign
from .models import Problem, Scenario
from .parameters import ParameterSpace
from .processing import Evaluation
from .runner import LocalExecutor


def _describe_function(function, arguments, role):
    """Locate and fingerprint a saved user callback for later import.

    arguments checks the expected calling convention without executing the
    callback. The returned file/name/module/import_root/sha256 record is
    consumed by _simple._load_function in another process. Resolving both
    scripts and packages lets users copy or rename their example file.
    """
    if (
        not inspect.isfunction(function)
        or function.__qualname__ != function.__name__
        or function.__name__ == "<lambda>"
    ):
        raise TypeError(
            f"{role}: define a function at the top level of your Python file, then pass its name without parentheses"
        )
    try:
        inspect.signature(function).bind(*arguments)
    except TypeError as exc:
        expected = "build_model(parameters)" if role == "model" else "evaluate(parameters, output)"
        raise TypeError(f"{role}: use the function signature {expected}") from exc
    source = inspect.getsourcefile(function)
    if source is None or not Path(source).is_file():
        raise ValueError(f"Save your {role} function in a .py file before running optimisation")
    source = Path(source).resolve()
    module = inspect.getmodule(function)
    package = getattr(module, "__package__", None)
    module_name = function.__module__ if package and function.__module__ != "__main__" else None
    # For a package callback, also make its import root available to workers.
    import_root = source.parent
    if module_name:
        levels = len(module_name.split(".")) - (0 if source.name == "__init__.py" else 1)
        for _ in range(levels):
            import_root = import_root.parent
    return {
        "file": str(source),
        "name": function.__name__,
        "module": module_name,
        "import_root": str(import_root),
        "sha256": sha256(source),
    }


def _model_problem(parameters, model_spec, files):
    """Wrap the simple model callback in one advanced Problem/Scenario.

    The private bridge is the worker's builder. The user's actual callback
    description lives in scenario.settings["model"]. "default" is the
    bridge's internal scenario name, not a required physical source name.
    """
    dependencies = tuple(
        dict.fromkeys([Path(model_spec["file"]), *(Path(p).expanduser().resolve() for p in files)])
    )
    return Problem(
        parameters if isinstance(parameters, ParameterSpace) else ParameterSpace(parameters),
        "gprMax.toolboxes.Optimisation._simple:build",
        scenarios=(Scenario("default", {"model": model_spec}),),
        dependencies=dependencies,
    )


def optimise(
    *,
    parameters,
    model,
    objective,
    directory,
    optimiser="tpe",
    evaluations=12,
    seed=0,
    population_size=6,
    batch_size=1,
    execution=None,
    files=(),
    progress=True,
    target_value=None,
    stop_on_target=True,
):
    """Run an optimisation using two ordinary functions from your Python file.

    The three problem-specific inputs are:

    parameters
        A dictionary such as {"length": Real(0.10, 0.18, "m")}. Each proposed
        dictionary, with the same names, is passed to both user functions.
    model
        ``build_model(parameters) -> gprMax.Scene``. Build the complete model
        here; the toolbox runs it. Do not call gprMax.run inside this function.
    objective
        ``evaluate(parameters, output) -> number``. Read ``output.file`` or use
        ``output.receiver(name, component)`` / ``output.port(name)``. Return a
        finite number: smaller is better. ObjectiveResult adds optional metrics.

    ``directory`` must be a new results directory. ``evaluations`` is a maximum
    number of candidates, not a convergence claim. GA/PSO/DE evaluate complete
    populations and can leave a remainder unused. ``seed`` fixes the optimiser
    and simulation seeds. ``execution`` optionally selects a LocalPool or entered
    MPIPool. ``files`` lists input data/helper files to monitor for changes; the
    model and objective source files are tracked automatically.

    ``target_value`` is an optional acceptable objective score. For minimisation,
    a value less than or equal to it meets the target. ``stop_on_target=True``
    stops after the complete current population has received feedback. Set it
    to False for a fixed-budget comparison. Neither option changes the score
    formula in the user's objective.

    Define functions at file scope and put this call under
    ``if __name__ == "__main__":`` so workers can load the file safely. Functions
    may be copied/renamed: there are no user-written import-reference strings.

    Returns an OptimisationResult with ``best_parameters``, ``best_value``,
    ``trials`` and ``stop_reason``. See the advanced guide for caching, native
    checkpoint recovery, multiple scenarios and custom optimiser adapters.
    """
    if isinstance(evaluations, bool) or not isinstance(evaluations, int) or evaluations < 1:
        raise ValueError("evaluations must be a positive integer")
    # Describe the callbacks before creating a campaign: a worker will reload
    # these saved functions later, rather than receive live Python closures.
    model_spec = _describe_function(model, ({},), "model")
    objective_spec = _describe_function(objective, ({}, None), "objective")
    files = tuple(files)
    problem = _model_problem(parameters, model_spec, files)
    if isinstance(optimiser, str):
        options = {"batch_size": batch_size} if optimiser in ("rf", "tpe") else {}
        if optimiser not in ("rf", "tpe") and batch_size != 1:
            raise ValueError("Use population_size for GA/PSO/DE; batch_size applies to RF/TPE")
        adapter = make_optimiser(optimiser, seed=seed, population_size=population_size, **options)
    else:
        adapter = optimiser
    size = getattr(adapter, "batch_size", 1)
    if evaluations < size:
        raise ValueError(
            f"This optimiser proposes {size} candidates together; set evaluations to at least {size}"
        )
    evaluator = Evaluation(
        "gprMax.toolboxes.Optimisation._simple:evaluate",
        settings={"objective": objective_spec, "progress": bool(progress)},
        dependencies=tuple(
            dict.fromkeys(
                [Path(objective_spec["file"]), *(Path(p).expanduser().resolve() for p in files)]
            )
        ),
    )
    # This is the hand-off to the framework. Campaign owns simulation files;
    # run_optimisation owns the proposal -> score -> optimiser feedback loop.
    campaign = Campaign(problem, directory, execution or LocalExecutor())
    if progress:
        print(
            f"Optimising {', '.join(problem.parameters.to_dict())}; up to {evaluations} evaluations.\n"
            f"Model: {model.__name__}  Objective: {objective.__name__}\nResults: {campaign.storage}",
            flush=True,
        )
        if evaluations % size:
            print(
                f"Complete batches of {size} permit {evaluations // size * size} evaluations within this limit.",
                flush=True,
            )
    result = campaign.optimise(
        optimiser=adapter,
        evaluator=evaluator,
        n_trials=evaluations,
        seed=seed,
        target_value=target_value,
        stop_on_target=stop_on_target,
    )
    if result.stop_reason == "trial_failed":
        failed = next((trial for trial in result.trials if trial.status == "failed"), None)
        if failed is None:
            raise RuntimeError(
                f"Optimisation stopped after a failed batch. See {campaign.storage / 'optimiser/result.json'}"
            )
        failure = failed.failure or {}
        details = failure.get("message")
        if details is None:
            details = "; ".join(
                (item or {}).get("message", "Simulation failed")
                for item in failure.get("scenarios", {}).values()
            )
        raise RuntimeError(
            f"Evaluation {failed.candidate_id or failed.token} failed: {details or failure.get('kind')}\n"
            f"See {campaign.storage / 'optimiser/result.json'} and the candidate logs."
        )
    if progress:
        print(
            f"Best parameters: {result.best_parameters}\nBest objective: {result.best_value}\n"
            f"Stopped: {result.stop_reason} ({len(result.trials)} evaluations).",
            flush=True,
        )
    return result


def simulate(*, model, parameters, directory, execution=None, files=(), seed=0):
    """Run your model once before optimisation; return a SimulationOutput.

    Supply one concrete parameter dictionary, e.g. {"permittivity": 4.0}.
    Use the returned output with your own evaluation function to check its score.
    No optimiser is imported or initialised by this operation.
    """
    # A one-off simulation validates values without introducing search bounds.
    from numbers import Integral
    from numbers import Real as Number

    from .parameters import Categorical, Integer, Real
    from .results import SimulationOutput

    definitions = {}
    for name, value in parameters.items():
        if isinstance(value, bool):
            raise ValueError(f"Parameter {name} must be a number or string, not bool")
        if isinstance(value, Integral):
            definitions[name] = Integer(int(value), int(value))
        elif isinstance(value, Number):
            import math

            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Parameter {name} must be finite")
            # Reuse Real validation for one value, with finite neighbouring bounds.
            # Real requires distinct bounds. The neighbouring representable floats
            # allow validation of this one fixed value; simulate never searches them.
            lower, upper = math.nextafter(value, -math.inf), math.nextafter(value, math.inf)
            if not math.isfinite(lower):
                lower = value
            if not math.isfinite(upper):
                upper = value
            definitions[name] = Real(lower, upper)
        elif isinstance(value, str):
            definitions[name] = Categorical((value,))
        else:
            raise TypeError(f"Parameter {name} must be a number or string")
    spec = _describe_function(model, ({},), "model")
    problem = _model_problem(definitions, spec, files)
    run = Campaign(problem, directory, execution or LocalExecutor()).evaluate_one(
        parameters, seed=seed
    )[0]
    if run.status != "complete":
        raise RuntimeError(
            f"Model simulation failed: {(run.record.get('failure') or {}).get('message')}\nSee {run.directory}"
        )
    return SimulationOutput(run)
