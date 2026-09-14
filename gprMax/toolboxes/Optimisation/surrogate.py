"""Random-forest surrogate optimisation through scikit-optimize.

Each observation pairs a full physical parameter combination with the
user's scalar objective. The forest estimates promising unexplored
combinations; the acquisition rule chooses which models to simulate.
EI, PI and LCB mean expected improvement, probability of improvement,
and lower confidence bound. No gprMax model or reader is embedded here.

Numeric and categorical dimensions are supported. ask/tell handles a
single candidate; ask_batch/tell_batch handles synchronous batches.
Failed observations are excluded from model fitting, not assigned a
fabricated numerical penalty.
"""

from ._storage import json_copy
from .optimisers import ObjectiveResult, Proposal
from .population import _positive_integer


class SkoptRF:
    """Random-forest surrogate + expected improvement, with optional batches.

    Failed trials are excluded from training; no penalty is invented. Fixed
    integer/category dimensions are reinserted without exposing them to skopt.
    """

    def __init__(
        self, *, seed=0, n_initial_points=3, acquisition="EI", n_points=2000, batch_size=1
    ):
        """Record exploration, acquisition and batch settings before loading the optional library."""
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if acquisition not in ("EI", "PI", "LCB"):
            raise ValueError("RF acquisition must be EI, PI or LCB")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self._batch_pending = None
        self.seed, self.acquisition = seed, acquisition
        self.n_initial_points = _positive_integer(n_initial_points, "n_initial_points")
        self.n_points = _positive_integer(n_points, "n_points")
        self.optimiser = None
        self._pending = None
        self._number = 0

    def initialise(self, parameters):
        """Map parameter definitions to native skopt dimensions and create the forest search.

        Fixed integers/categories are kept in _fixed because skopt needs varying
        dimensions. _names preserves the order used in ask/tell coordinate lists;
        fixed values are reinserted before the candidate reaches the model.
        """
        if self.optimiser is not None:
            raise RuntimeError("Use a new adapter for each optimisation session")
        import skopt
        from skopt.space import Categorical, Integer, Real

        self.parameters = parameters
        dimensions = []
        self._names, self._fixed, self._steps = [], {}, {}
        for name, spec in parameters.to_dict().items():
            if spec["type"] == "Integer" and spec["lower"] == spec["upper"]:
                self._fixed[name] = spec["lower"]
                continue
            if spec["type"] == "Categorical" and len(spec["choices"]) == 1:
                self._fixed[name] = spec["choices"][0]
                continue
            if spec["type"] == "Real":
                dim = Real(
                    spec["lower"],
                    spec["upper"],
                    prior="log-uniform" if spec["scale"] == "log" else "uniform",
                    name=name,
                )
            elif spec["type"] == "Integer":
                step = spec.get("step", 1)
                if step == 1:
                    dim = Integer(spec["lower"], spec["upper"], name=name)
                else:
                    # skopt has no integer step argument. It searches interval
                    # indices; only this adapter sees those indices. Models and
                    # saved trial records continue to use physical values.
                    self._steps[name] = (spec["lower"], step)
                    dim = Integer(0, (spec["upper"] - spec["lower"]) // step, name=name)
            else:
                dim = Categorical(spec["choices"], name=name)
            self._names.append(name)
            dimensions.append(dim)
        if not dimensions:
            raise ValueError("SkoptRF needs at least one varying parameter")
        self.optimiser = skopt.Optimizer(
            dimensions,
            base_estimator="RF",
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition,
            acq_optimizer="sampling",
            random_state=self.seed,
            acq_optimizer_kwargs={"n_points": self.n_points},
        )
        return {
            "adapter": type(self).__name__,
            "library": "scikit-optimize",
            "version": skopt.__version__,
            "base_estimator": "RF",
            "acquisition": self.acquisition,
            "acquisition_optimizer": "sampling",
            "n_initial_points": self.n_initial_points,
            "n_points": self.n_points,
            "seed": self.seed,
            "direction": "minimize",
            "parameter_order": self._names,
            "fixed_parameters": self._fixed,
            "integer_step_encoding": self._steps,
            "storage": "in_memory",
            "failure_policy": "exclude_from_training",
            "batch_size": self.batch_size,
            "batch_strategy": "cl_min",
            "capabilities": {
                "batches": True,
                "multiple_parameters": True,
                "mixed_types": True,
                "parallel": True,
                "resume": True,
            },
        }

    def ask(self):
        """Propose one named physical combination and retain it until matching feedback arrives."""
        if self.optimiser is None or self._pending is not None or self._batch_pending is not None:
            raise RuntimeError("Initialise first and finish the pending trial before asking again")
        values = self.optimiser.ask()
        physical = self._decode_point(values)
        self._number += 1
        self._pending = Proposal(
            f"rf-{self._number:06d}",
            self.parameters.validate(physical),
            {"phase": "initial" if len(self.optimiser.yi) < self.n_initial_points else "surrogate"},
        )
        self._expected_parameters = json_copy(dict(self._pending.parameters))
        return self._pending

    def tell(self, proposal, result):
        """Fit the new successful observation; release failed/cancelled trials without training on them."""
        if (
            self._pending is None
            or proposal != self._pending
            or result.token != proposal.token
            or dict(result.parameters) != dict(proposal.parameters)
        ):
            raise ValueError("Feedback must match the pending proposal")
        if dict(proposal.parameters) != self._expected_parameters:
            raise ValueError("Pending proposal was modified after ask")
        if result.status not in ("complete", "failed", "cancelled"):
            raise ValueError("Unsupported trial status")
        if result.status == "complete":
            value = ObjectiveResult(result.value).value
            self.optimiser.tell(self._encode_point(proposal.parameters), value)
        self._pending = None

    @property
    def finished(self):
        """Leave stopping to campaign trial/simulation budgets and target policy."""
        return False

    def ask_batch(self):
        """Propose candidates together using skopt's cl_min batch strategy.

        The temporary optimistic values used to choose a batch are internal
        to skopt. Actual simulation scores arrive only in tell_batch.
        """
        if self.batch_size == 1:
            proposals = (self.ask(),)
        else:
            if (
                self.optimiser is None
                or self._pending is not None
                or self._batch_pending is not None
            ):
                raise RuntimeError("Finish pending feedback before asking a batch")
            points = self.optimiser.ask(n_points=self.batch_size, strategy="cl_min")
            proposals = []
            for point in points:
                physical = self._decode_point(point)
                self._number += 1
                proposals.append(
                    Proposal(
                        f"rf-{self._number:06d}",
                        self.parameters.validate(physical),
                        {
                            "phase": "initial"
                            if len(self.optimiser.yi) < self.n_initial_points
                            else "surrogate"
                        },
                    )
                )
            proposals = tuple(proposals)
        self._batch_pending = proposals
        self._batch_expected = [(p.token, json_copy(dict(p.parameters))) for p in proposals]
        return proposals

    def tell_batch(self, proposals, results):
        """Validate member identity/order, then train only on successful batch observations."""
        if (
            self._batch_pending is None
            or len(results) != len(proposals)
            or [(p.token, dict(p.parameters)) for p in proposals] != self._batch_expected
        ):
            raise ValueError(
                "Feedback must match the complete pending batch in order; a proposal may have been modified"
            )
        for p, r in zip(proposals, results):
            if (
                r.token != p.token
                or dict(r.parameters) != dict(p.parameters)
                or r.status not in ("complete", "failed", "cancelled")
            ):
                raise ValueError("Feedback does not match the pending proposal")
            if r.status == "complete":
                ObjectiveResult(r.value)
        if self.batch_size == 1:
            self.tell(proposals[0], results[0])
        else:
            successful = [(p, r) for p, r in zip(proposals, results) if r.status == "complete"]
            if successful:
                self.optimiser.tell(
                    [self._encode_point(p.parameters) for p, _ in successful],
                    [float(r.value) for _, r in successful],
                )
            # Failed batches must not leave skopt's cached proposals pending.
            self.optimiser.cache_ = {}
        self._batch_pending = None

    def _decode_point(self, point):
        """Convert skopt's scalar/index values into one physical dictionary."""
        physical = dict(self._fixed)
        for name, value in zip(self._names, point):
            value = value.item() if hasattr(value, "item") else value
            if name in self._steps:
                lower, step = self._steps[name]
                value = lower + step * int(value)
            physical[name] = value
        return physical

    def _encode_point(self, parameters):
        """Return matching skopt indices when feeding an evaluated score back."""
        values = []
        for name in self._names:
            value = parameters[name]
            if name in self._steps:
                lower, step = self._steps[name]
                value = (value - lower) // step
            values.append(value)
        return values
