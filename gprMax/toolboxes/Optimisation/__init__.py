"""Public imports for parameter-driven gprMax optimisation.

Start with Real/Integer/Categorical, optimise and simulate. Your file
supplies build_model(parameters) and evaluate(parameters, output).
SimulationOutput gives the objective access to the saved HDF5 result.

The remaining exports support explicit campaigns, parallel execution,
custom processing and optimiser adapters. They are optional layers;
ordinary model authors need not construct their records themselves.
Importing this package loads the parent gprMax API but does not run a
simulation or import the optional optimiser backends.
See README.md for use and CODE_WALKTHROUGH.md for the execution path.
"""

from .adapters import make_optimiser
from .cache import SimulationCache
from .campaign import Campaign
from .execution import ExecutionBackend, LocalPool, MPIPool, RunTask, execution_from_profile
from .models import BuildContext, PreparedModel, Problem, RunResult, Scenario
from .optimisation import OptimisationResult
from .optimisers import BatchOptimiser, ObjectiveResult, Optimiser, OptunaTPE, Proposal, TrialResult
from .parameters import Categorical, Integer, ParameterSpace, Real
from .population import PymooDE, PymooGA, PymooOptimiser, PymooPSO
from .processing import Evaluation, ProcessingContext, evaluate_outputs, load_candidate
from .quantities import PortSpectrum, ReceiverTrace, read_port, read_receiver
from .results import SimulationOutput
from .runner import LocalExecutor
from .scalar import ScalarSearchResult, minimise_integer
from .simple import optimise, simulate
from .surrogate import SkoptRF

__all__ = [
    "optimise",
    "simulate",
    "SimulationOutput",
    "ExecutionBackend",
    "LocalPool",
    "MPIPool",
    "RunTask",
    "execution_from_profile",
    "Evaluation",
    "ProcessingContext",
    "evaluate_outputs",
    "load_candidate",
    "SimulationCache",
    "BuildContext",
    "Campaign",
    "Categorical",
    "Integer",
    "LocalExecutor",
    "ParameterSpace",
    "PreparedModel",
    "Problem",
    "Real",
    "RunResult",
    "Scenario",
    "ReceiverTrace",
    "read_receiver",
    "PortSpectrum",
    "read_port",
    "ScalarSearchResult",
    "minimise_integer",
    "make_optimiser",
    "BatchOptimiser",
    "PymooOptimiser",
    "PymooGA",
    "PymooPSO",
    "PymooDE",
    "SkoptRF",
    "ObjectiveResult",
    "Optimiser",
    "OptunaTPE",
    "Proposal",
    "TrialResult",
    "OptimisationResult",
]
