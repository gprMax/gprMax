"""Run the same model and objective with a different optimisation package.

From the gprMax checkout, for example:
    python -m gprMax.toolboxes.Optimisation.examples.optimiser_choices results/de --optimiser de
    python -m gprMax.toolboxes.Optimisation.examples.optimiser_choices results/rf --optimiser rf

Install the chosen package first; see ../OPTIMISERS.rst. Each command runs ONE
search. Use a fresh results directory for each comparison.
"""

import argparse
from pathlib import Path

from gprMax.toolboxes.Optimisation import optimise

# 1-3. REUSE the parameter ranges, model builder and scoring function.
# Replace this import with functions from your own saved model file.
from gprMax.toolboxes.Optimisation.examples.start_here import PARAMETERS, build_model, evaluate


# 4. CHANGE THE SEARCH SETTINGS; THE MODEL AND OBJECTIVE STAY THE SAME.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--optimiser", choices=("tpe", "rf", "ga", "pso", "de"), default="tpe")
    parser.add_argument("--evaluations", type=int, default=12)
    args = parser.parse_args()
    result = optimise(
        parameters=PARAMETERS,
        model=build_model,
        objective=evaluate,
        directory=args.directory,
        optimiser=args.optimiser,
        evaluations=args.evaluations,
        population_size=6,
        batch_size=1,
        seed=7,
    )
    print(f"Best parameters: {result.best_parameters}; objective: {result.best_value}")


if __name__ == "__main__":
    main()
