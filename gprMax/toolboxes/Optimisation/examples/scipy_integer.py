"""Direct SciPy example for one bounded integer; python scipy_integer.py results/scipy.

This small helper is separate from the general optimiser adapters. Here the
objective itself calls simulate(). Use the general adapters for several
parameters, parallel populations and campaign checkpoint recovery.
"""

import argparse
from pathlib import Path

from gprMax.toolboxes.Optimisation import minimise_integer, simulate

# 1-3. REUSE THE MODEL AND CRITERION, restricted here to integer permittivities.
from gprMax.toolboxes.Optimisation.examples.start_here import build_model, evaluate


# 4. RUN THE DIRECT SCALAR SEARCH.
def run_example(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)

    def score(relative_permittivity):
        # This callback runs directly in this process. Only the imported,
        # top-level build_model is loaded by a solver worker.
        parameters = {"permittivity": relative_permittivity}
        output = simulate(
            model=build_model,
            parameters=parameters,
            directory=directory / f"permittivity_{relative_permittivity}",
        )
        return evaluate(parameters, output)

    # The helper remembers evaluated integers so duplicate requests do not rerun
    # a model. This method is intended for an approximately unimodal interval.
    result = minimise_integer(score, 2, 8, initial=4, maxiter=12)
    print(f"Best permittivity: {result.best}; objective: {result.value}")
    print(f"Evaluated integer -> score: {result.evaluations}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run_example(parser.parse_args().directory)
