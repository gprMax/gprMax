"""Data passed between the campaign coordinator and solver worker.

A candidate is one complete parameter dictionary. Each candidate is run
once for every Scenario (fixed experimental settings). One execution
attempt produces one RunResult. These are distinct from optimiser
iterations or generations; a candidate can require several simulations.

Advanced builders receive (parameters, scenario, context) and return
PreparedModel. The simple API constructs these records automatically.
The coordinator sends JSON-compatible descriptions to the worker; the
worker builds and runs the actual Scene rather than sending it over MPI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ._storage import json_copy
from .parameters import ParameterSpace


def check_reference(reference):
    """Validate a saved function address such as 'my_model:build_model'.

    Workers import the module and look up the function by name. This accepts
    importable top-level functions, not arbitrary Python expressions.
    """
    if not isinstance(reference, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", reference
    ):
        raise ValueError("Use an import reference such as 'my_models.block:build_model'")


@dataclass(frozen=True)
class Scenario:
    """Fixed settings for one simulation of each candidate.

    id names both the result directory and the key in the advanced
    evaluator's runs dictionary. settings is user-defined JSON data, e.g.
    source orientation or material environment; it is not optimised.
    The simple API supplies one scenario named "default" automatically.
    """

    id: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate a directory-safe ID and detach settings from the caller."""
        if not isinstance(self.id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.id):
            raise ValueError("Scenario IDs must be simple nonempty names, without path separators")
        if not isinstance(self.settings, Mapping):
            raise TypeError("Scenario settings must be a mapping")
        object.__setattr__(self, "settings", json_copy(dict(self.settings)))

    def to_dict(self):
        """Return the JSON description sent to the worker and written to disk."""
        return {"id": self.id, "settings": json_copy(self.settings)}


@dataclass(frozen=True)
class BuildContext:
    """Execution settings allocated by the toolbox for one attempt.

    workdir is the attempt directory. output_stem is the path without an
    extension used by gprMax (normally workdir / "output"). seed controls
    reproducible model randomness; cpu_threads is the allocated CPU budget.
    Builders must respect the output location and thread allocation.
    """

    workdir: Path
    output_stem: Path
    seed: int
    cpu_threads: int


@dataclass
class PreparedModel:
    """A fresh, unexecuted Scene and optional records of its construction.

    scene is the gprMax model the worker will run. effective_parameters
    records the realised design after rounding or other model choices,
    e.g. a physical length snapped to cells. metadata holds other JSON
    descriptions chosen by the builder. Neither dictionary changes the
    optimiser's proposed parameters or defines the objective.
    """

    scene: Any
    effective_parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Problem:
    """Advanced description of the parameterised simulation experiment.

    parameters defines bounds/types; builder is an importable function
    returning PreparedModel. Every candidate runs all scenarios. version
    identifies the user's model definition; dependencies lists helper/data
    files whose contents must remain unchanged throughout the campaign.
    Objective selection belongs to Campaign.optimise, not this model record.
    """

    parameters: ParameterSpace
    builder: str
    scenarios: tuple[Scenario, ...] = (Scenario("default"),)
    version: str = "1"
    dependencies: tuple[Path, ...] = ()

    def __post_init__(self):
        """Check the model contract and resolve declared dependency file paths."""
        if not isinstance(self.parameters, ParameterSpace):
            raise TypeError("parameters must be a ParameterSpace")
        check_reference(self.builder)
        scenarios = tuple(self.scenarios)
        if not scenarios or any(not isinstance(x, Scenario) for x in scenarios):
            raise ValueError("At least one Scenario is required")
        if len({x.id for x in scenarios}) != len(scenarios):
            raise ValueError("Scenario IDs must be unique")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("A nonempty model version is required")
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(
            self, "dependencies", tuple(Path(p).expanduser().resolve() for p in self.dependencies)
        )


@dataclass(frozen=True)
class RunResult:
    """One candidate/scenario attempt, including failure or cancellation.

    directory contains request.json, logs, result.json and, on success,
    output.h5. record is the execution record, not a dictionary of physical
    measurements. Use output_file with a reader to obtain those measurements.
    A retry has its own directory but the same candidate/scenario identity.
    """

    candidate_id: str
    scenario_id: str
    directory: Path
    record: Mapping[str, Any]

    @property
    def status(self):
        """Return the recorded execution status: complete, failed or cancelled."""
        return self.record["status"]

    @property
    def output_file(self):
        """Resolve the saved output artifact, or return None when none was committed."""
        artifact = self.record.get("artifacts", {}).get("output")
        return self.directory / artifact["path"] if artifact else None
