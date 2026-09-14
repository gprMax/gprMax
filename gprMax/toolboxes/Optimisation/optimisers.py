"""Common proposal/score records and the Optuna TPE adapter.

Ordinary users select an optimiser in optimise(...); they edit neither
ask nor tell. Adapter authors implement ask to propose a complete physical
parameter dictionary, then tell to receive its objective and status.
Batch variants exchange several candidates in matching order.

Proposal carries an optimiser token, distinct from Campaign's candidate
directory ID. TrialResult joins that token to the simulation and score.
ObjectiveResult contains one value to minimise plus optional diagnostics;
diagnostics are not additional objectives. Multiobjective/Pareto searches
would require a different feedback contract.
"""

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Mapping, Protocol

from ._storage import json_copy
from .parameters import ParameterSpace


@dataclass(frozen=True)
class Proposal:
    """One optimiser suggestion before simulation.

    token uniquely identifies the suggestion within the optimiser session.
    parameters holds all variables in their user-declared physical units.
    metadata records adapter details such as generation/member coordinates;
    it does not change the model parameters or define the objective.
    """

    token: str
    parameters: Mapping
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        """Check the token and detach parameter/metadata dictionaries for reliable records."""
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("An optimiser proposal needs a nonempty string token")
        object.__setattr__(self, "parameters", json_copy(dict(self.parameters)))
        object.__setattr__(self, "metadata", json_copy(dict(self.metadata)))


@dataclass(frozen=True)
class ObjectiveResult:
    """One finite scalar to minimise, with optional JSON diagnostics.

    value is the actual optimisation criterion returned by the user.
    metrics may record resonance, gain or intermediate errors, but adapters
    optimise value only. Returning a negative quantity expresses maximisation.
    Units and any weighting between physical errors belong to the objective.
    """

    value: float
    metrics: Mapping = field(default_factory=dict)

    def __post_init__(self):
        """Reject nonfinite/non-scalar scores and make diagnostics safe to record as JSON."""
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, Real)
            or not math.isfinite(self.value)
        ):
            raise ValueError("Objective must be a finite real scalar")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "metrics", json_copy(dict(self.metrics)))


@dataclass(frozen=True)
class TrialResult:
    """Feedback for one Proposal after simulation and objective processing.

    token/parameters must match the original proposal. candidate_id links
    to its saved simulations, or is None if no candidate was created.
    status distinguishes complete, failed and cancelled trials. Only a
    completed trial has a usable value; failure details are not fitness.
    """

    token: str
    parameters: Mapping
    candidate_id: str | None
    status: str
    value: float | None = None
    metrics: Mapping = field(default_factory=dict)
    failure: Mapping | None = None


class Optimiser(Protocol):
    """Serial single-objective minimisation. Each proposal receives one result.

    initialise returns JSON metadata including algorithm settings/capabilities.
    Adapters must handle failed/cancelled TrialResults without invented values.
    Instances are used for one optimisation session; recovery is not implied.
    """

    def initialise(self, parameters: ParameterSpace) -> dict:
        """Configure the search space once and return JSON library/settings metadata."""
        ...

    def ask(self) -> Proposal:
        """Propose one complete parameter combination with a new unique token."""
        ...

    def tell(self, proposal: Proposal, result: TrialResult) -> None:
        """Receive the matching completed/failed/cancelled trial before proposing again."""
        ...


class BatchOptimiser(Protocol):
    """Optional population interface with coordinated complete-batch feedback.

    batch_size is an upper bound used for admission BEFORE ask_batch. An empty
    batch signals native termination. A full admitted batch completes before
    target stopping; failures/interruptions cancel its remaining members.
    tell_batch receives every member in order, including explicit failures and
    cancellations. No numerical penalty or resumable state is implied.
    """

    batch_size: int

    def initialise(self, parameters: ParameterSpace) -> dict:
        """Configure a population/batch search and return its settings/capabilities."""
        ...

    @property
    def finished(self) -> bool:
        """Report native termination or an aborted adapter session."""
        ...

    def ask_batch(self) -> tuple[Proposal, ...]:
        """Propose at most batch_size candidates, or an empty tuple when finished."""
        ...

    def tell_batch(self, proposals: tuple[Proposal, ...], results: tuple[TrialResult, ...]) -> None:
        """Receive every member status/score in the original proposal order."""
        ...


class OptunaTPE:
    """Tree-structured Parzen Estimator using Optuna's ask/tell interface.

    Each trial proposes all real, log-real, integer and categorical variables.
    n_startup_trials sets initial exploration before TPE uses observed scores.
    batch_size > 1 enables pending-trial handling; feedback remains synchronous.
    This class stores library state, not solver state, and never runs gprMax.

    An Optuna Study is optimiser state; it is unrelated to gprMax's
    fixed-geometry Study. Recovery uses the campaign's optional native
    checkpoint. Trial pruning is not implemented by this adapter.
    """

    def __init__(self, *, seed=0, n_startup_trials=3, batch_size=1):
        """Validate search settings; defer importing/creating Optuna until initialise."""
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if (
            isinstance(n_startup_trials, bool)
            or not isinstance(n_startup_trials, int)
            or n_startup_trials < 1
        ):
            raise ValueError("n_startup_trials must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.seed = seed
        self.n_startup_trials = n_startup_trials
        self.study = None
        self._pending = {}

    def initialise(self, parameters):
        """Map named parameter definitions to Optuna distributions and create a minimising study."""
        if self.study is not None:
            raise RuntimeError("Use a new adapter for each optimisation session")
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "OptunaTPE requires the optional optuna package; tested with optuna==4.9.0"
            ) from exc
        distributions = {}
        for name, spec in parameters.to_dict().items():
            if spec["type"] == "Real":
                distributions[name] = optuna.distributions.FloatDistribution(
                    spec["lower"], spec["upper"], log=spec["scale"] == "log"
                )
            elif spec["type"] == "Integer":
                distributions[name] = optuna.distributions.IntDistribution(
                    spec["lower"], spec["upper"], step=spec.get("step", 1)
                )
            elif spec["type"] == "Categorical":
                distributions[name] = optuna.distributions.CategoricalDistribution(spec["choices"])
            else:
                raise ValueError(f"Unsupported Optuna parameter: {name}")
        self._parameters = parameters
        self._distributions = distributions
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=self.n_startup_trials,
                constant_liar=self.batch_size > 1,
            ),
        )
        return {
            "adapter": "gprMax.toolboxes.Optimisation.optimisers:OptunaTPE",
            "library": "optuna",
            "version": optuna.__version__,
            "sampler": "TPESampler",
            "seed": self.seed,
            "n_startup_trials": self.n_startup_trials,
            "direction": "minimize",
            "batch_size": self.batch_size,
            "constant_liar": self.batch_size > 1,
            "capabilities": {
                "multiple_parameters": True,
                "mixed_types": True,
                "parallel": True,
                "batches": True,
                "resume": True,
                "pruning": False,
            },
            "storage": "in_memory",
        }

    def ask(self):
        """Request one trial, rejecting a second ask while feedback is pending."""
        if self.study is None or self._pending:
            raise RuntimeError("Initialise first and finish the pending trial before asking again")
        return self._ask_one()

    def _ask_one(self):
        """Retain Optuna's trial object by token so later feedback updates the correct trial."""
        trial = self.study.ask(fixed_distributions=self._distributions)
        proposal = Proposal(str(trial.number), self._parameters.validate(trial.params))
        self._pending[proposal.token] = trial
        return proposal

    def tell(self, proposal, result):
        """Record diagnostics and tell Optuna the objective or failed status.

        Cancelled and failed framework trials become failed Optuna trials.
        No artificial large score is assigned to a failed simulation.
        """
        import optuna

        trial = self._pending.get(proposal.token)
        if (
            trial is None
            or result.token != proposal.token
            or dict(result.parameters) != dict(proposal.parameters)
        ):
            raise ValueError("Feedback must match a pending proposal")
        if dict(proposal.parameters) != trial.params:
            raise ValueError("Proposal parameters changed after ask")
        if result.status not in ("complete", "failed", "cancelled"):
            raise ValueError("Unsupported trial status")
        value = ObjectiveResult(result.value).value if result.status == "complete" else None
        trial.set_user_attr("candidate_id", result.candidate_id)
        trial.set_user_attr("metrics", json_copy(dict(result.metrics)))
        trial.set_user_attr("failure", json_copy(result.failure))
        self.study.tell(
            trial,
            values=value,
            state=optuna.trial.TrialState.COMPLETE
            if result.status == "complete"
            else optuna.trial.TrialState.FAIL,
        )
        del self._pending[proposal.token]

    @property
    def finished(self):
        """Leave stopping to campaign budgets/targets; this adapter has no native trial cap."""
        return False

    def ask_batch(self):
        """Create a synchronous group of proposals and preserve its token order."""
        if self.study is None or self._pending:
            raise RuntimeError("Finish pending feedback before asking a batch")
        proposals = tuple(self._ask_one() for _ in range(self.batch_size))
        self._batch_tokens = tuple(p.token for p in proposals)
        return proposals

    def tell_batch(self, proposals, results):
        """Validate the complete batch before updating any of its Optuna trials."""
        if (
            len(proposals) != len(results)
            or tuple(p.token for p in proposals) != self._batch_tokens
            or set(self._pending) != set(self._batch_tokens)
        ):
            raise ValueError("Feedback must match the complete pending batch in order")
        for p, r in zip(proposals, results):
            if (
                r.token != p.token
                or dict(r.parameters) != dict(p.parameters)
                or dict(p.parameters) != self._pending[p.token].params
                or r.status not in ("complete", "failed", "cancelled")
            ):
                raise ValueError("Feedback does not match the pending proposal")
            if r.status == "complete":
                ObjectiveResult(r.value)
        for p, r in zip(proposals, results):
            self.tell(p, r)
