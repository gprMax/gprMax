"""Translate a user-facing optimiser name into a fresh adapter object.

TPE = Tree-structured Parzen Estimator; RF = random-forest surrogate;
GA = genetic algorithm; PSO = particle swarm optimisation;
DE = differential evolution. Each uses the same model and objective.
The adapter proposes parameter combinations and receives scalar scores;
Campaign, not the external library, executes gprMax.
"""

from .optimisers import OptunaTPE
from .population import PymooDE, PymooGA, PymooPSO
from .surrogate import SkoptRF

OPTIMISERS = ("tpe", "ga", "pso", "rf", "de")


def make_optimiser(name, *, seed=0, population_size=6, n_initial_points=3, **options):
    """Create the selected adapter without starting a search or simulation.

    population_size applies to GA/PSO/DE. n_initial_points supplies the
    initial exploration count for RF/TPE. options forwards library-adapter
    settings such as batch_size; direct construction exposes further options.
    A new instance is required for each independent optimisation session.
    """
    if name == "tpe":
        return OptunaTPE(seed=seed, n_startup_trials=n_initial_points, **options)
    if name == "rf":
        return SkoptRF(seed=seed, n_initial_points=n_initial_points, **options)
    if name in ("ga", "pso", "de"):
        return {"ga": PymooGA, "pso": PymooPSO, "de": PymooDE}[name](
            seed=seed, population_size=population_size, **options
        )
    raise ValueError(f"Unknown optimiser {name!r}; choose from {OPTIMISERS}")
