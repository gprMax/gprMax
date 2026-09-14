# Developer map

Ordinary users edit `PARAMETERS`, `build_model(parameters)` and
`evaluate(parameters, output)` in their own file. Begin at [README.md](README.md).
This document is for changing or extending the toolbox itself.

For a guided reading of the implementation, use
[Following one optimisation through the code](CODE_WALKTHROUGH.md). It explains
the internal terms, callback arguments, request/result fields and directory tree.
The source files also describe their role at module level, while method
docstrings and local comments explain identity, ordering and recovery decisions.

## Execution path

1. `simple.optimise` validates the two Python functions, automatically records
   their source paths/hashes, and constructs the existing advanced contracts.
2. `optimisation.run_optimisation` obtains proposals from an adapter and asks
   `campaign.Campaign` to evaluate them. The coordinator owns IDs and records.
3. `execution.LocalPool` / `MPIPool`, or the serial `runner.LocalExecutor`, dispatch
   simulation requests. Each solver runs in an isolated `_worker` subprocess.
4. `_simple.build` loads the user's saved function and calls
   `build_model(parameters)`. The worker executes the returned gprMax Scene.
5. `processing.PreparedEvaluator` checks the completed outputs and creates a
   processing directory. `_simple.evaluate` calls the user's
   `evaluate(parameters, SimulationOutput)` and converts a number to ObjectiveResult.
6. The coordinator records the score and delivers matching feedback to the
   optimiser. Library-specific behaviour stays in the adapter modules.

The private bridge adapts an ordinary Python file to the existing worker protocol;
it does not implement another optimiser loop. It supports saved top-level
functions, including copied/renamed scripts and package modules. Source files must
not change during a run. Runtime closures and anonymous functions are not supported
because their state is not present when an isolated worker loads the saved file.

## Module responsibilities

| Module | Responsibility | Who normally uses it |
|---|---|---|
| `simple.py` | `optimise` and `simulate` entry points | Users |
| `parameters.py` | Bounds, types and physical-value validation | Users choose definitions; adapters validate |
| `results.py` | `SimulationOutput.file`, optional readers and saved arrays | User objectives |
| `quantities.py` | Readers for the gprMax receiver/port file formats | Optional result extraction |
| `models.py` | Advanced Problem, Scenario, PreparedModel and RunResult contracts | Advanced users/backend authors |
| `processing.py` | Advanced callbacks, dependency checking and artifact records | Advanced processing and coordinator |
| `optimisers.py` | Proposal/feedback types and TPE adapter | Adapter authors |
| `population.py`, `surrogate.py` | pymoo and RF integration | Adapter authors |
| `adapters.py` | Select a named adapter without running a model | Entry points and advanced users |
| `scalar.py` | Optional direct SciPy search over one integer | Callers providing their own objective execution |
| `campaign.py`, `optimisation.py` | Simulation scheduling and feedback orchestration | Framework developers |
| `execution.py`, `runner.py`, `_worker.py` | Resource allocation and solver execution | Backend authors |
| `cache.py`, `checkpoint.py`, `_storage.py` | Optional reuse/recovery and atomic records | Framework developers |
| `_simple.py` | Load user functions and adapt their arguments/return values | Private implementation |

Receiver/port readers contain dataset names because they implement a known gprMax
file format. They are not a required objective schema. A user can process any saved
HDF5 quantity through `output.file`; only the final finite scalar is required by
the optimiser interface. The simulation worker currently expects a saved gprMax
HDF5 result and validates its basic grid/time metadata.

## Extending the framework

- New objective: edit the user's evaluation function; no framework changes.
- New geometry/material parametrisation: edit the user's model function.
- New specialised reader: add an optional helper; keep raw-file access available.
- New optimiser: implement the documented ask/tell or batch interface.
- New execution backend: implement the execution protocol and preserve task identity.
- Several scenarios or native checkpoints: use the explicit Campaign interface.

Tests cover both layers. The user-interface tests copy examples into another
folder, rename functions/parameter/receiver names, execute real gprMax simulations,
and recompute the returned objectives. Advanced tests cover ordering, failures,
budgets, caching and state restoration independently of that interface.

Editable examples put their parameter definitions, model, objective and launch
settings first. Supporting functions belong below that workflow and must be
signposted at each call with their purpose. The thin-wire example demonstrates
this layout: its resonance-estimation helper follows `run_search`, and the final
main guard launches only after all definitions have been loaded. Keep advanced
CLI/benchmark orchestration in `examples/advanced`.

## User documentation and examples

`README.rst` is the canonical user guide, including `OPTIMISERS.rst`,
`OUTPUT_READERS.rst` and `EXAMPLES.rst`. `README.md` is the repository landing page.
`docs/source/inc_Optimisation.rst` includes the same RST guide in the Sphinx manual;
keep user instructions beside the toolbox rather than duplicating them.

Keep reusable examples in four numbered sections: parameters/targets, model,
objective and launch. Put supporting calculations below that workflow, comment
why each helper is called, and keep the main guard at the end. Explain units and
model-specific dataset names where they enter the calculation. New readers must
document field shapes, units, validity masks and sampling/normalisation conventions;
raw dataset access must not be presented as an interpreted physical quantity.

Integer steps are public physical values (`lower + k * step`). Optuna receives its
native step; GA/PSO/DE use the numeric coordinate mapping; RF encodes stepped values
as integer interval indices internally and decodes them before issuing a proposal.
Models, scores and saved trial records always use the physical dictionary.
