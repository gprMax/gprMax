"""Adapt pymoo GA/PSO/DE populations to gprMax parameter dictionaries.

A population is a batch of candidate models. Each row contains every
optimisation parameter; columns follow ParameterSpace declaration order.
pymoo works on numerical coordinates in [0, 1]. NumericCoordinates maps
them to physical real/integer values before Campaign builds the model.

The coordinator runs simulations and supplies one scalar per row.
tell_batch attaches those scores to the same pymoo population, then
advances the algorithm. Nothing here reads S11 or assumes an antenna.
Categorical variables need the RF/TPE adapters in the current toolbox.
"""

import importlib
import math
from dataclasses import asdict

from ._storage import json_copy
from .models import check_reference
from .optimisers import ObjectiveResult, Proposal
from .parameters import Integer, Real


def _positive_integer(value, name, minimum=1):
    """Validate counts without accepting booleans or values below the required minimum."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class NumericCoordinates:
    """Unit coordinates with one explicit physical decode, including log scales.

    Integers use the nearest declared step, with half-up ties. PSO retains latent
    positions and velocities while evaluating the decoded integer geometry. Other adapters
    can repair integer coordinates onto their exact representable values.
    """

    def __init__(self, parameters):
        """Record numeric parameter order and reject unsupported categorical variables."""
        self.parameters = parameters
        self.specs = parameters.to_dict()
        if any(s["type"] == "Categorical" for s in self.specs.values()):
            raise ValueError(
                "This numeric population adapter does not support categorical parameters; use OptunaTPE or SkoptRF"
            )

    def decode(self, coordinates):
        """Convert one coordinate row to named physical parameter values.

        Real variables use their linear/log mapping. Integer variables round
        half up onto their declared step lattice inside inclusive bounds. The
        model still decides how those physical values enter the gprMax grid.
        """
        if len(coordinates) != len(self.specs):
            raise ValueError("Coordinate dimension mismatch")
        values = {}
        for (name, spec), coordinate in zip(self.specs.items(), coordinates):
            u = float(coordinate)
            if not math.isfinite(u) or not -1e-12 <= u <= 1 + 1e-12:
                raise ValueError("Optimiser coordinate is outside [0, 1]")
            u = min(1.0, max(0.0, u))
            if spec["type"] == "Integer":
                values[name] = Integer(
                    spec["lower"], spec["upper"], step=spec.get("step", 1)
                ).from_unit(u)
            else:
                values[name] = Real(spec["lower"], spec["upper"], scale=spec["scale"]).from_unit(u)
        return self.parameters.validate(values)

    def repair(self, coordinates):
        """Move integer coordinates onto the exact values that will be evaluated.

        GA/DE receive these repaired coordinates with their scores. PSO uses
        decode without this repair so its continuous positions/velocities survive
        while the model sees rounded integers.
        """
        values = self.decode(coordinates)
        repaired = list(map(float, coordinates))
        for i, (name, spec) in enumerate(self.specs.items()):
            if spec["type"] == "Integer":
                width = spec["upper"] - spec["lower"]
                repaired[i] = (values[name] - spec["lower"]) / width if width else 0.5
        return repaired


class PymooOptimiser:
    """Wrap a pymoo ask/tell factory with bounded population size.

    A custom importable factory accepts pop_size and JSON options and returns a
    compatible single-objective pymoo algorithm. It operates in unit numerical
    coordinates. No custom objective, model builder or callback is passed into
    pymoo. The campaign owns all simulation and objective evaluation.

    Failed/cancelled batches abort this adapter without invented fitness values.
    State is in memory; records are an audit trail, not a restart checkpoint.
    """

    def __init__(
        self,
        factory,
        *,
        population_size=6,
        seed=0,
        generations=None,
        options=None,
        integer_mode="repair",
    ):
        """Record a compatible pymoo factory and its population settings.

        factory is an import reference accepting pop_size and options.
        generations is an optional native cap; campaign budgets apply as well.
        integer_mode chooses repaired coordinates or latent PSO positions.
        """
        check_reference(factory)
        self.factory = factory
        self.batch_size = _positive_integer(population_size, "population_size", 2)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.seed = seed
        self.generations = (
            None if generations is None else _positive_integer(generations, "generations")
        )
        self.options = json_copy(dict(options or {}))
        if "pop_size" in self.options:
            raise ValueError("Set population_size explicitly")
        if integer_mode not in ("repair", "latent"):
            raise ValueError("integer_mode must be repair or latent")
        self.integer_mode = integer_mode
        self.algorithm = None
        self._pending = ()
        self._population = None
        self._aborted = False
        self._batch = 0

    def initialise(self, parameters):
        """Create a numerical single-objective problem and initialise the algorithm.

        There is no pymoo model-evaluation callback: gprMax runs externally.
        One coordinate column is created per declared parameter. The settings
        returned here are persisted by run_optimisation for later inspection.
        """
        if self.algorithm is not None:
            raise RuntimeError("Use a new adapter for each optimisation session")
        import numpy as np
        import pymoo
        from pymoo.core.problem import Problem
        from pymoo.core.termination import NoTermination

        self.coordinates = NumericCoordinates(parameters)
        # One column per parameter, one scalar objective for the whole model.
        # Parameter count does not multiply the simulations per candidate.
        self.problem = Problem(
            n_var=len(parameters.to_dict()),
            n_obj=1,
            xl=np.zeros(len(parameters.to_dict())),
            xu=np.ones(len(parameters.to_dict())),
        )
        module, name = self.factory.split(":")
        factory = getattr(importlib.import_module(module), name)
        self.algorithm = factory(pop_size=self.batch_size, **self.options)
        termination = NoTermination() if self.generations is None else ("n_gen", self.generations)
        self.algorithm.setup(self.problem, termination=termination, seed=self.seed, verbose=False)
        return {
            "adapter": type(self).__name__,
            "library": "pymoo",
            "version": pymoo.__version__,
            "factory": self.factory,
            "options": self.options,
            "population_size": self.batch_size,
            "generations": self.generations,
            "seed": self.seed,
            "direction": "minimize",
            "coordinate_encoding": "unit_linear_or_log; integer_declared_step_nearest_half_up",
            "integer_mode": self.integer_mode,
            "parameter_order": list(parameters.to_dict()),
            "failure_policy": "abort_batch_without_fitness",
            "storage": "in_memory",
            "capabilities": {
                "batches": True,
                "multiple_parameters": True,
                "numeric_mixed_types": True,
                "categorical": False,
                "parallel": True,
                "resume": True,
                "continue_after_failure": False,
            },
        }

    @property
    def finished(self):
        """Report native pymoo termination or a previous failed/cancelled batch."""
        return self._aborted or (self.algorithm is not None and not self.algorithm.has_next())

    def ask_batch(self):
        """Retain the proposed population and return its decoded physical candidates.

        _pending identifies proposals awaiting scores; _population is the same
        pymoo object that will receive those scores. _signature detects caller
        changes to proposal dictionaries before feedback is accepted.
        """
        import numpy as np

        if self.algorithm is None or self._pending:
            raise RuntimeError("Initialise first and finish the pending batch before asking again")
        if self.finished:
            return ()
        population = self.algorithm.ask()
        if population is None or not len(population):
            self._aborted = True
            return ()
        if len(population) > self.batch_size:
            raise ValueError("Pymoo returned more members than the declared batch_size")
        # X rows are candidates; columns follow parameter declaration order.
        # Keep proposed and evaluated coordinates so integer mapping is inspectable.
        raw = np.array(population.get("X"), dtype=float, copy=True)
        evaluated = (
            np.array([self.coordinates.repair(row) for row in raw])
            if self.integer_mode == "repair"
            else raw.copy()
        )
        population.set("X", evaluated)
        self._batch += 1
        generation = int(self.algorithm.n_gen)
        self._pending = tuple(
            Proposal(
                f"batch-{self._batch:06d}-member-{i:04d}",
                self.coordinates.decode(row),
                {
                    "batch": self._batch,
                    "generation": generation,
                    "member": i,
                    "proposed_coordinates": raw[i].tolist(),
                    "evaluated_coordinates": row.tolist(),
                },
            )
            for i, row in enumerate(evaluated)
        )
        self._population = population
        self._signature = json_copy([asdict(p) for p in self._pending])
        return self._pending

    def tell_batch(self, proposals, results):
        """Attach one scalar score to each retained population member and advance.

        Check all identities and statuses first. A failed/cancelled member
        aborts this adapter without inventing fitness or passing a shortened
        successful population to pymoo.
        """
        import numpy as np
        from pymoo.problems.static import StaticProblem

        if (
            not self._pending
            or tuple(proposals) != self._pending
            or len(results) != len(self._pending)
        ):
            raise ValueError("Feedback must match the pending batch")
        if [asdict(p) for p in proposals] != self._signature:
            raise ValueError("Pending proposal was modified after ask")
        for proposal, result in zip(self._pending, results):
            if result.token != proposal.token or dict(result.parameters) != dict(
                proposal.parameters
            ):
                raise ValueError("Feedback must match each pending proposal in order")
            if result.status not in ("complete", "failed", "cancelled"):
                raise ValueError("Unsupported trial status")
            if result.status == "complete":
                ObjectiveResult(result.value)
        if any(result.status != "complete" for result in results):
            self._aborted = True
        else:
            # pymoo expects F with shape (population members, objectives).
            # The single column contains user scores in the original member order.
            values = np.array([[result.value] for result in results])
            # Force this newly simulated batch to be counted even if a library
            # individual inherited evaluated attributes from its parent.
            self.algorithm.evaluator.eval(
                StaticProblem(self.problem, F=values),
                self._population,
                skip_already_evaluated=False,
            )
            self.algorithm.tell(infills=self._population)
        self._pending, self._population = (), None


class PymooGA(PymooOptimiser):
    """Genetic algorithm with integer coordinate repair before model evaluation."""

    def __init__(self, *, population_size=6, seed=0, generations=None, options=None):
        """Select pymoo GA; repeated candidates remain explicit evaluations unless caching is enabled."""
        settings = {"eliminate_duplicates": False, **(options or {})}
        super().__init__(
            "pymoo.algorithms.soo.nonconvex.ga:GA",
            population_size=population_size,
            seed=seed,
            generations=generations,
            options=settings,
            integer_mode="repair",
        )


class PymooPSO(PymooOptimiser):
    """Native adaptive PSO with continuous positions and rounded integer models.

    Adaptive coefficients and best-particle perturbation stay enabled, matching
    pymoo's defaults. Silently disabling them changes the search and can leave
    a quantised swarm repeatedly evaluating a small region. Explicit ``options``
    can still disable either feature for a controlled experiment.

    ``restart_after`` optionally starts a fresh swarm after that many complete
    populations without improvement. Campaign retains the best across every
    swarm. Restarts consume the existing budget; they do not add evaluations.
    """

    def __init__(
        self, *, population_size=6, seed=0, generations=None, options=None, restart_after=None
    ):
        """Select PSO with latent integer coordinates so position/velocity state is retained."""
        self.restart_after = (
            None if restart_after is None else _positive_integer(restart_after, "restart_after")
        )
        self._restart_count = 0
        self._stalled_batches = 0
        self._swarm_best = math.inf
        self.restart_history = []
        settings = {"adaptive": True, "pertube_best": True, **(options or {})}
        super().__init__(
            "pymoo.algorithms.soo.nonconvex.pso:PSO",
            population_size=population_size,
            seed=seed,
            generations=generations,
            options=settings,
            integer_mode="latent",
        )

    def initialise(self, parameters):
        """Record optional restart settings alongside the native PSO settings."""
        info = super().initialise(parameters)
        info.update(
            restart_after=self.restart_after,
            restart_policy="fresh_swarm; seed_plus_restart_index; shared_total_budget",
        )
        return info

    def ask_batch(self):
        """Restart a stagnant swarm before asking; keep global proposal numbering.

        A fresh swarm does not inherit the old attractor. This lets it explore
        another region. Campaign's result still includes all previous scores.
        Remaining native generations enforce the original total budget.
        """
        restart_after = getattr(self, "restart_after", None)
        if (
            restart_after is not None
            and self.algorithm is not None
            and not self._pending
            and not self.finished
            and self._stalled_batches >= restart_after
        ):
            from pymoo.algorithms.soo.nonconvex.pso import PSO
            from pymoo.core.termination import NoTermination

            self._restart_count += 1
            seed = (self.seed + self._restart_count) % 2**32
            remaining = None if self.generations is None else self.generations - self._batch
            termination = NoTermination() if remaining is None else ("n_gen", remaining)
            self.algorithm = PSO(pop_size=self.batch_size, **self.options)
            self.algorithm.setup(self.problem, termination=termination, seed=seed, verbose=False)
            self.restart_history.append({"after_batch": self._batch, "seed": seed})
            self._stalled_batches, self._swarm_best = 0, math.inf
        proposals = super().ask_batch()
        if restart_after is not None and proposals:
            # Include the swarm identity in durable candidate records. Refresh
            # the signature because these are new, fully specified proposals.
            self._pending = tuple(
                Proposal(
                    p.token,
                    p.parameters,
                    dict(
                        p.metadata,
                        swarm=self._restart_count + 1,
                        swarm_seed=(self.seed + self._restart_count) % 2**32,
                    ),
                )
                for p in proposals
            )
            self._signature = json_copy([asdict(p) for p in self._pending])
            return self._pending
        return proposals

    def tell_batch(self, proposals, results):
        """Track improvement only after a complete, successfully delivered batch."""
        super().tell_batch(proposals, results)
        if getattr(self, "restart_after", None) is not None and not self._aborted:
            best = min(result.value for result in results)
            if best < self._swarm_best:
                self._swarm_best, self._stalled_batches = best, 0
            else:
                self._stalled_batches += 1


class PymooDE(PymooOptimiser):
    """Differential evolution with integer repair and a minimum population of four."""

    def __init__(self, *, population_size=6, seed=0, generations=None, options=None):
        """Select the configured DE variant and validate its minimum population size."""
        _positive_integer(population_size, "population_size", 4)
        settings = {"variant": "DE/rand/1/bin", **(options or {})}
        super().__init__(
            "pymoo.algorithms.soo.nonconvex.de:DE",
            population_size=population_size,
            seed=seed,
            generations=generations,
            options=settings,
            integer_mode="repair",
        )
