"""Optional direct SciPy search for one bounded integer variable.

This helper calls objective(integer) itself and remembers previously
evaluated integers. It is separate from the general Campaign adapter
loop: the caller owns any simulation/storage needed by that objective.
Use the general adapters for multiple parameters or parallel populations.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarSearchResult:
    """Best evaluated integer and the complete integer-to-score table.

    solver_success/message describe SciPy's stopping result. best/value
    include the explicit endpoint and neighbour checks and do not establish
    a global optimum for a non-unimodal objective.
    """

    best: int
    value: float
    evaluations: dict
    solver_success: bool
    solver_message: str


def minimise_integer(objective, lower, upper, *, initial=None, maxiter=30):
    """Bounded SciPy search with exact integer memoisation and neighbour checks.

    Intended for a narrow, approximately unimodal search interval. The return
    value is the best evaluated integer, not a proof of a global minimum.
    Errors/nonfinite objectives stop the search rather than becoming penalties.
    """
    from scipy.optimize import minimize_scalar

    if any(isinstance(x, bool) or not isinstance(x, int) for x in (lower, upper, maxiter)):
        raise ValueError("Bounds and maxiter must be integers")
    if lower >= upper or maxiter <= 0:
        raise ValueError("Bounds must increase and maxiter must be positive")
    if initial is not None and (
        isinstance(initial, bool) or not isinstance(initial, int) or not lower <= initial <= upper
    ):
        raise ValueError("initial must be an integer within the bounds")
    evaluations = {}

    def evaluate(x):
        """Round one SciPy coordinate and reuse its score if that exact integer was already tested."""
        integer = min(upper, max(lower, int(math.floor(float(x) + 0.5))))
        if integer not in evaluations:
            value = float(objective(integer))
            if not math.isfinite(value):
                raise ValueError(f"Objective at {integer} is not finite")
            evaluations[integer] = value
        return evaluations[integer]

    if initial is not None:
        evaluate(initial)
    evaluate(lower)
    evaluate(upper)
    result = minimize_scalar(
        evaluate,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 0.4, "maxiter": maxiter},
    )
    best = min(evaluations, key=evaluations.get)
    # Rounding makes the objective piecewise constant to SciPy. Check the
    # best integer's immediate neighbours explicitly before reporting it.
    for value in (best - 1, best + 1):
        if lower <= value <= upper:
            evaluate(value)
    best = min(evaluations, key=evaluations.get)
    return ScalarSearchResult(
        best, evaluations[best], dict(evaluations), bool(result.success), str(result.message)
    )
