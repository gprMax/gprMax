"""Private bridge between ordinary user functions and the Campaign contracts.

Users edit their parameter dictionary, model function and objective function.
This module handles loading those functions in isolated worker processes.
"""

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

from ._storage import sha256
from .models import PreparedModel
from .optimisers import ObjectiveResult
from .results import SimulationOutput


def _load_function(spec):
    """Load the callback described by simple._describe_function.

    Verify its source hash first, then import either its package module or
    its saved script. Cached imports avoid re-running top-level code for
    every objective evaluation. A changed or shadowed file is an error.
    """
    path = Path(spec["file"])
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"Your Python file changed during optimisation: {path}")
    # The generated module name is only an implementation detail. It is stable
    # within a run, distinct for different files, and never needs user editing.
    module_name = (
        spec["module"]
        or "_gprmax_user_" + hashlib.sha256((str(path) + spec["sha256"]).encode()).hexdigest()[:16]
    )
    for directory in (spec["import_root"], str(path.parent)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    if spec["module"]:
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != path:
            raise ValueError(f"A different module shadows your model file: {path}")
    elif module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return getattr(module, spec["name"])


def build(parameters, scenario, context):
    """Call build_model(parameters) and wrap its Scene for the worker.

    scenario/context belong to the advanced worker contract. The simple
    user function receives only the parameter dictionary. A user who needs
    explicit effective_parameters/metadata can already return PreparedModel.
    """
    spec = scenario.settings["model"]
    scene = _load_function(spec)(dict(parameters))
    if isinstance(scene, PreparedModel):
        return scene
    import gprMax

    if not isinstance(scene, gprMax.Scene):
        raise TypeError(
            f"{spec['name']}(parameters) must return a gprMax.Scene; do not call gprMax.run in the model function"
        )
    return PreparedModel(scene, metadata={"user_model": spec})


def evaluate(parameters, runs, context):
    """Call evaluate(parameters, output) after the simulation completes.

    runs["default"] is the single scenario created by simple._model_problem.
    SimulationOutput adapts that RunResult to the public file/readers API.
    ObjectiveResult wraps the scalar and optional diagnostics for every
    optimiser; the user's function does not need to know ask/tell mechanics.
    """
    spec = context.settings["objective"]
    run = runs["default"]
    value = _load_function(spec)(dict(parameters), SimulationOutput(run, context))
    try:
        result = value if isinstance(value, ObjectiveResult) else ObjectiveResult(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{spec['name']}(parameters, output) must return one finite number to minimise, or ObjectiveResult"
        ) from exc
    if context.settings["progress"]:
        print(
            f"Evaluation {int(run.candidate_id)}: {dict(parameters)} -> objective {result.value:.8g}",
            flush=True,
        )
    return result
