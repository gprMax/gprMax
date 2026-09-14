"""Run the user's objective and record how its score was produced.

The simple API adapts evaluate(parameters, output) to this layer.
Advanced users supply Evaluation(function, settings, dependencies) for
evaluate(parameters, runs, context), or a legacy import string for
evaluate(parameters, runs). runs maps scenario IDs to RunResult objects.

PreparedEvaluator loads and verifies the callback, checks simulation
artifacts, calls it, and commits processing.json. ProcessingContext
saves user-defined intermediate arrays/JSON beside that record.
These records support inspection and recovery; they do not prescribe
the user's signal-processing method or physical measurement names.
"""

import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ._storage import json_copy, sha256, write_json
from .models import check_reference
from .optimisers import ObjectiveResult


@dataclass(frozen=True)
class Evaluation:
    """An importable evaluate(parameters, runs, context) processing function.

    Declare measurement files and helper source files as dependencies. The legacy
    string evaluator remains evaluate(parameters, runs), without a context.
    """

    function: str
    settings: Mapping = field(default_factory=dict)
    dependencies: tuple[Path, ...] = ()

    def __post_init__(self):
        """Validate the callback reference and snapshot its settings/dependency paths."""
        check_reference(self.function)
        object.__setattr__(self, "settings", json_copy(dict(self.settings)))
        object.__setattr__(
            self, "dependencies", tuple(Path(p).expanduser().resolve() for p in self.dependencies)
        )


class ProcessingContext:
    """Per-evaluation artifact directory and snapshotted processing settings."""

    def __init__(self, directory, settings):
        """Bind one processing directory and immutable settings to this evaluation."""
        self.directory = Path(directory).resolve()
        self.settings = MappingProxyType(json_copy(settings))
        self.artifacts = {}

    def _path(self, name):
        """Require a new local filename so processing cannot overwrite an earlier artifact."""
        path = self.directory / name
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in ("", ".", "..", "processing.json")
        ):
            raise ValueError("Processing artifact names must be simple filenames")
        if path.exists():
            raise FileExistsError(path)
        return path

    def save_json(self, name, value):
        """Write JSON diagnostics and record their path, format and content hash."""
        path = self._path(name)
        write_json(path, value)
        self.artifacts[name] = {"path": name, "sha256": sha256(path), "format": "json"}
        return path

    def save_npz(self, name, *, units=None, **arrays):
        """Save numerical arrays without pickle and record optional unit labels.

        arrays and units use the user's names. Units are descriptive metadata,
        not conversions. Object arrays are rejected so the saved file can be
        loaded with allow_pickle=False.
        """
        import numpy as np

        path = self._path(name)
        if path.suffix != ".npz":
            raise ValueError("Array artifacts need a .npz extension")
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        if not arrays or any(value.dtype.hasobject for value in arrays.values()):
            raise ValueError("Use non-object arrays, readable without pickle")
        metadata = json_copy(dict(units or {}))
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        self.artifacts[name] = {
            "path": name,
            "sha256": sha256(path),
            "format": "npz",
            "units": metadata,
        }
        return path


class PreparedEvaluator:
    """A loaded callback plus its reproducibility and artifact records.

    Created by the optimisation coordinator, not by ordinary model authors.
    info fingerprints the callback/settings/dependencies; it is compared
    during recovery. The actual score remains whatever the callback returns.
    """

    def __init__(self, specification):
        """Load an advanced or legacy evaluator and record its source/dependencies."""
        self.contextual = isinstance(specification, Evaluation)
        spec = specification if self.contextual else Evaluation(specification)
        self.settings = json_copy(dict(spec.settings))
        module, name = spec.function.split(":")
        self.function = getattr(importlib.import_module(module), name)
        if not callable(self.function):
            raise TypeError("The evaluator must be callable")
        source = inspect.getsourcefile(self.function)
        self.info = {
            "reference": spec.function,
            "source": source,
            "sha256": sha256(source) if source else None,
            "interface": "context" if self.contextual else "legacy",
            "settings": self.settings,
            "dependencies": [{"path": str(p), "sha256": sha256(p)} for p in spec.dependencies],
        }

    def verify(self):
        """Reject changed objective code or declared inputs before evaluating another candidate."""
        assets = list(self.info["dependencies"])
        if self.info["source"]:
            assets.append({"path": self.info["source"], "sha256": self.info["sha256"]})
        for asset in assets:
            if sha256(asset["path"]) != asset["sha256"]:
                raise ValueError(f"Evaluator source or dependency changed: {asset['path']}")

    def recover_or_run(self, parameters, runs, directory, *, recover=False):
        """Reuse a committed score only when its recorded inputs still match.

        Matching covers the callback, parameters, simulation hashes and saved
        processing artifacts. An incomplete or mismatched record is preserved;
        a new processing-NNNN directory holds the recomputed result.
        """
        directory = Path(directory)
        if recover and directory.exists():
            self.verify()
            try:
                record = json.loads((directory / "processing.json").read_text())
                if (
                    record["status"] == "complete"
                    and record["evaluator"] == self.info
                    and record["parameters"] == parameters
                    and all(
                        name in runs and sha256(runs[name].output_file) == item["sha256"]
                        for name, item in record["inputs"].items()
                    )
                    and all(
                        sha256(directory / item["path"]) == item["sha256"]
                        for item in record["artifacts"].values()
                    )
                ):
                    return ObjectiveResult(record["value"], record["metrics"])
            except (OSError, KeyError, ValueError, TypeError):
                pass
            index = 2
            while directory.with_name(f"processing-{index:04d}").exists():
                index += 1
            directory = directory.with_name(f"processing-{index:04d}")
        return self(parameters, runs, directory)

    def __call__(self, parameters, runs, directory):
        """Verify all scenarios, call the objective and commit success or failure.

        Returns ObjectiveResult. Simulation data is read by the user's callback;
        this layer checks identity/integrity and manages processing.json. Its
        finally block preserves failure details even when the callback raises.
        """
        self.verify()
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        context = ProcessingContext(directory, self.settings)
        inputs = {}
        for name, run in runs.items():
            if run.status != "complete":
                raise ValueError(f"Cannot process unsuccessful scenario {name}")
            if run.output_file is not None:
                digest = sha256(run.output_file)
                if digest != run.record["artifacts"]["output"]["sha256"]:
                    raise ValueError(f"Simulation output integrity failure: {name}")
                inputs[name] = {"path": str(run.output_file), "sha256": digest}
        record = {
            "evaluator": self.info,
            "parameters": json_copy(dict(parameters)),
            "inputs": inputs,
            "status": "running",
            "artifacts": {},
        }
        write_json(directory / "processing.json", record)
        try:
            # Only this call chooses the numerical objective. All dictionary keys
            # above describe execution/provenance, not mandatory physical outputs.
            args = (dict(parameters), MappingProxyType(dict(runs)))
            result = self.function(*args, context) if self.contextual else self.function(*args)
            if not isinstance(result, ObjectiveResult):
                raise TypeError("Evaluator must return ObjectiveResult")
            record.update(status="complete", value=result.value, metrics=result.metrics)
            return result
        except BaseException as exc:
            record.update(
                status="failed", failure={"type": type(exc).__name__, "message": str(exc)}
            )
            raise
        finally:
            record["artifacts"] = context.artifacts
            write_json(directory / "processing.json", record)


def evaluate_outputs(evaluator, parameters, runs, directory):
    """Reprocess completed RunResults without launching gprMax.

    evaluator selects an advanced callback; parameters and runs provide
    its input. directory must be a new processing output directory.
    """
    return PreparedEvaluator(evaluator)(parameters, runs, directory)


def load_candidate(directory):
    """Load (parameter_dict, runs_by_scenario) from a candidate directory.

    Uses candidate.json/evaluation.json to locate each attempt's result.json.
    This reconstructs RunResult records; it neither reruns the solver nor
    loads HDF5 arrays. Useful for inspecting or reprocessing saved candidates.
    """
    from .models import RunResult

    directory = Path(directory).resolve()
    candidate = json.loads((directory / "candidate.json").read_text())
    evaluation = json.loads((directory / "evaluation.json").read_text())
    runs = {}
    for relative in evaluation["runs"]:
        path = (directory.parent.parent / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Run path must belong to the candidate")
        record = json.loads((path / "result.json").read_text())
        result = RunResult(candidate["id"], record["scenario_id"], path, record)
        runs[result.scenario_id] = result
    return candidate["parameters"], runs
