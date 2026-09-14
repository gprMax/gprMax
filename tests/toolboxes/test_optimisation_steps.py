"""Physical increments must survive proposals, feedback and native checkpoints.

Cheap numerical scores test transport through each external optimiser; no
antenna simulation is needed to expose a units/index mapping error.
"""

import math

import numpy as np
import pytest

from gprMax.toolboxes.Optimisation import (
    Categorical,
    Integer,
    OptunaTPE,
    ParameterSpace,
    PymooDE,
    PymooGA,
    PymooPSO,
    Real,
    SkoptRF,
    TrialResult,
)
from gprMax.toolboxes.Optimisation.checkpoint import load_checkpoint, save_checkpoint


@pytest.mark.parametrize("step", [0, -1, True, 1.5])
def test_invalid_steps_are_rejected(step):
    with pytest.raises(ValueError, match="step"):
        Integer(26, 36, step=step)


def test_physical_lattice_bounds_values_and_ties():
    with pytest.raises(ValueError, match="steps"):
        Integer(26, 35, "mm", step=2)
    length = Integer(np.int64(26), np.int64(36), "mm", step=np.int64(2))
    assert length.validate(np.int64(30)) == 30
    for invalid in (25, 27, 37, 30.0, True):
        with pytest.raises(ValueError):
            length.validate(invalid)
    assert [length.from_unit(u) for u in (0, 0.1, 0.5, 0.9, 1)] == [26, 28, 32, 36, 36]
    assert Integer(-6, 6, step=4).from_unit(0.5) == 2  # Half-up also for negative bounds.
    assert Integer(7, 7, step=2).from_unit(0.8) == 7
    # Both unit and larger steps resolve a represented half-step upwards.
    assert Integer(1, 12).from_unit(np.nextafter(0.5, 0)) == 7
    # Ordinary off-boundary values retain the established nearest-integer mapping.
    plain = Integer(1, 12)
    for u in np.linspace(0, 1, 101):
        assert plain.from_unit(u) == min(12, math.floor(1 + u * 11 + 0.5))


def objective(parameters):
    return float((parameters["length_mm"] - 30) ** 2 + parameters["loss"])


def feedback(proposals):
    return tuple(
        TrialResult(p.token, p.parameters, None, "complete", objective(p.parameters))
        for p in proposals
    )


@pytest.mark.parametrize("kind", ["ga", "pso", "de", "tpe", "rf", "rf_batch"])
def test_steps_reach_models_and_feedback_in_physical_units(kind, tmp_path):
    pytest.importorskip("cloudpickle")
    library = "pymoo" if kind in ("ga", "pso", "de") else "optuna" if kind == "tpe" else "skopt"
    pytest.importorskip(library)
    definitions = {
        "length_mm": Integer(26, 36, "mm", step=2),
        "loss": Real(0.01, 0.1, scale="log"),
        "fixed_mm": Integer(8, 8, "mm", step=2),
    }
    if kind in ("tpe", "rf", "rf_batch"):
        definitions["material"] = Categorical(("a", "b"))
    space = ParameterSpace(definitions)
    if kind in ("ga", "pso", "de"):
        cls = {"ga": PymooGA, "pso": PymooPSO, "de": PymooDE}[kind]
        adapter = cls(seed=7, population_size=6, generations=6)
    elif kind == "tpe":
        adapter = OptunaTPE(seed=7, batch_size=2)
    else:
        adapter = SkoptRF(seed=7, batch_size=2 if kind == "rf_batch" else 1, n_points=100)
    adapter.initialise(space)
    if kind == "tpe":
        assert adapter._distributions["length_mm"].step == 2
    observations = []
    for _ in range(3):
        proposals = adapter.ask_batch()
        for proposal in proposals:
            values = proposal.parameters
            assert values["length_mm"] in (26, 28, 30, 32, 34, 36)
            assert values["fixed_mm"] == 8
            assert 0.01 <= values["loss"] <= 0.1
        observations.extend(proposals)
        adapter.tell_batch(proposals, feedback(proposals))
    if kind.startswith("rf"):
        index = adapter._names.index("length_mm")
        # Native RF uses interval indices but its responses remain physical scores.
        assert [point[index] for point in adapter.optimiser.Xi] == [
            (p.parameters["length_mm"] - 26) // 2 for p in observations
        ]
        assert adapter.optimiser.yi == [objective(p.parameters) for p in observations]
    # A checkpoint with a pending batch must keep future proposals identical.
    pending = adapter.ask_batch()
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    save_checkpoint(folder, {"optimiser": adapter})
    restored = load_checkpoint(folder)["optimiser"]
    adapter.tell_batch(pending, feedback(pending))
    restored.tell_batch(pending, feedback(pending))
    assert [p.parameters for p in adapter.ask_batch()] == [
        p.parameters for p in restored.ask_batch()
    ]
