"""Check that the public PSO adapter preserves native pymoo search behaviour."""
import numpy as np
import pytest

pytest.importorskip("pymoo")
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from gprMax.toolboxes.Optimisation import Integer, ParameterSpace, PymooPSO, TrialResult
from gprMax.toolboxes.Optimisation.checkpoint import load_checkpoint, save_checkpoint

SPACE = ParameterSpace(
    {"length": Integer(26, 36), "width": Integer(12, 44), "offset": Integer(1, 12)}
)


def physical(x):
    return np.floor([26, 12, 1] + np.asarray(x) * [10, 32, 11] + 0.5).astype(int)


def score(parameters):
    # A cheap numerical function tests transport only; it is not an antenna model.
    return float(np.sum((np.array(list(parameters.values())) - [30, 18, 4]) ** 2) - 50)


def advance(adapter):
    proposals = adapter.ask_batch()
    x = np.array([p.metadata["proposed_coordinates"] for p in proposals])
    np.testing.assert_array_equal(physical(x), [list(p.parameters.values()) for p in proposals])
    adapter.tell_batch(
        proposals,
        tuple(
            TrialResult(p.token, p.parameters, None, "complete", score(p.parameters))
            for p in proposals
        ),
    )
    return x


@pytest.mark.parametrize("options", [{}, {"adaptive": False, "pertube_best": False}])
def test_pso_matches_native_defaults_and_explicit_overrides(options):
    class Native(Problem):
        def __init__(self):
            super().__init__(n_var=3, n_obj=1, xl=np.zeros(3), xu=np.ones(3))
            self.positions = []

        def _evaluate(self, x, out, *args, **kwargs):
            self.positions.append(x.copy())
            out["F"] = (np.sum((physical(x) - [30, 18, 4]) ** 2, axis=1) - 50)[:, None]

    native = Native()
    result = minimize(native, PSO(pop_size=10, **options), termination=("n_gen", 24), seed=7)
    adapter = PymooPSO(population_size=10, generations=24, seed=7, options=options)
    adapter.initialise(SPACE)
    assert adapter.algorithm.adaptive is options.get("adaptive", True)
    assert adapter.algorithm.pertube_best is options.get("pertube_best", True)
    for x in native.positions:
        np.testing.assert_allclose(advance(adapter), x, rtol=0, atol=1e-14)
    assert adapter.finished
    np.testing.assert_allclose(adapter.algorithm.opt[0].F, result.F, rtol=0, atol=1e-13)


@pytest.mark.parametrize("restart_after", [None, 2])
def test_adaptive_pso_checkpoint_preserves_future_swarm(tmp_path, restart_after):
    pytest.importorskip("cloudpickle")
    adapter = PymooPSO(population_size=10, generations=24, seed=7, restart_after=restart_after)
    adapter.initialise(SPACE)
    for _ in range(7):
        advance(adapter)
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    save_checkpoint(folder, {"optimiser": adapter})
    restored = load_checkpoint(folder)["optimiser"]
    while not adapter.finished:
        np.testing.assert_array_equal(advance(adapter), advance(restored))
        np.testing.assert_array_equal(
            adapter.algorithm.pop.get("F"), restored.algorithm.pop.get("F")
        )
    assert restored.finished


def test_stagnation_restarts_explore_fresh_swarms_within_total_budget():
    adapter = PymooPSO(population_size=10, generations=8, seed=7, restart_after=2)
    adapter.initialise(SPACE)
    tokens, first_positions, swarms = [], {}, []
    while not adapter.finished:
        proposals = adapter.ask_batch()
        swarm = proposals[0].metadata["swarm"]
        first_positions.setdefault(swarm, proposals[0].metadata["proposed_coordinates"])
        swarms.append(swarm)
        tokens.extend(p.token for p in proposals)
        # Flat feedback deliberately provides no improvement, as on a rounded
        # model plateau. The adapter must explore again without extra budget.
        adapter.tell_batch(
            proposals,
            tuple(TrialResult(p.token, p.parameters, None, "complete", -5.0) for p in proposals),
        )
    assert len(tokens) == len(set(tokens)) == 80
    assert swarms == [1, 1, 1, 2, 2, 2, 3, 3]
    assert len({tuple(x) for x in first_positions.values()}) == 3
    assert [r["seed"] for r in adapter.restart_history] == [8, 9]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_restart_setting_is_rejected(value):
    with pytest.raises(ValueError, match="restart_after"):
        PymooPSO(restart_after=value)
