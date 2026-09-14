# Following one optimisation through the code

Start with [examples/start_here.py](examples/start_here.py). Its four numbered sections cover allowed parameters, a gprMax model built from those parameters, a score calculated from the saved output, and the launch settings. This guide follows what happens after its `optimise(...)` call.

## What the user controls

| Question | Where to edit | Example |
|---|---|---|
| What may change? | `PARAMETERS` | `{"permittivity": Real(2, 8)}` |
| How do parameters change the model? | `build_model(parameters)` | Use `parameters["permittivity"]` in `gprMax.Material` |
| Which measurements are saved? | The same model function | Add an `Rx` with ID `probe`, saving `Ez` |
| What makes a candidate better? | `evaluate(parameters, output)` | Return an absolute error, RMS error or another finite scalar |
| Which algorithm chooses candidates? | `optimise(optimiser=...)` | `"tpe"`, `"rf"`, `"ga"`, `"pso"`, `"de"` |
| How much work is allowed? | Search/execution settings | `evaluations`, population size and allocated workers |

The optimisation criterion is the returned score. Smaller is better. To maximise a quantity, return its negative. If `ObjectiveResult(value, metrics)` is used, only `value` is optimised; `metrics` records additional information for inspection.

The dipole example calculates `abs(estimated_S11_dip_hz - TARGET_FREQUENCY_HZ)`. Its resonance helper measures the dip; the subtraction defines the criterion. `evaluations=12` is a budget, not a requirement to reach zero. Both `optimise` and the advanced `Campaign.optimise` interface expose `target_value` and `stop_on_target` for an optional acceptable-score threshold.

The [rectangular patch example](examples/rectangular_patch.py) uses length, width
and `feed_offset_mm`, measured from the actual patch centre toward the left
edge, as in the paper. Its builder positions the feed by subtracting that
distance from the centre. The default `Integer(..., step=2)` lengths and widths
keep the centre and feed centreline on the 1 mm design lattice. One-off checks
can still use odd millimetre lengths, which put the centre between nodes;
the signposted helper applies the 0.5 mm rounding and `evaluate()` saves both
requested and realised offsets. Earlier experiment records using
`feed_inset_mm` remain edge-referenced and must not simply be renamed.

Several parameters form one complete candidate dictionary. The builder receives them together, builds one model for each scenario, and the objective returns one score for that candidate. Adding parameters changes the search space; it does not automatically create a separate optimisation for each parameter.

## The vocabulary used inside the toolbox

| Term | Meaning |
|---|---|
| Parameter space | Names, types, bounds and scales of allowed variables |
| Proposal | A complete parameter combination suggested by an optimiser |
| Token | The optimiser's unique identifier for a proposal awaiting feedback |
| Candidate | The combination as recorded by the campaign, with a directory ID such as `000001` |
| Scenario | Fixed experimental settings applied to every candidate, e.g. a source orientation; the simple API uses one named `default` |
| Task | One candidate/scenario simulation request |
| Attempt | One execution of that task; retries use new attempt directories |
| Trial result | The proposal identity joined to its execution status and objective score |
| Batch/population | Several proposals evaluated before the next optimiser update |
| Artifact | A saved file, identified by path and content hash |
| Effective parameters | The actual design after model choices such as rounding a length to mesh cells |

A candidate with three scenarios normally needs three simulations. A retry adds an attempt. A cache hit reuses a simulation without adding an attempt. These distinctions explain why proposal counts and simulation counts can differ.

## Step 1: describe the experiment

[simple.py](simple.py) implements the public `optimise` and `simulate` functions. It checks the user's callbacks and records their saved source file, function name and content hash. This lets another Python process reload the same function later.

`_model_problem` constructs the advanced records automatically: a `ParameterSpace`, a `Problem`, and one `Scenario("default", ...)`. The internal `model` entry in that scenario holds the callback description. It is not a required geometry parameter or source name.

[parameters.py](parameters.py) validates names, bounds and types. Unit labels do not perform conversions. `Integer` validates inclusive bounds and an optional positive `step`; only the user's model decides whether it represents cells, layers or something else. [models.py](models.py) documents the records used by the advanced interface.

## Step 2: let the optimiser propose a design

[adapters.py](adapters.py) selects the adapter. The coordinator in [optimisation.py](optimisation.py) calls `initialise`, followed by `ask` or `ask_batch`. The resulting `Proposal` contains a token and physical parameter values.

| Adapter module | What it translates |
|---|---|
| [optimisers.py](optimisers.py) | Optuna TPE distributions/trials into proposals and feedback; also defines the shared record types |
| [population.py](population.py) | pymoo GA/PSO/DE coordinate rows into named real/integer values |
| [surrogate.py](surrogate.py) | Random-forest proposals and observations through scikit-optimize |

TPE means Tree-structured Parzen Estimator. GA means genetic algorithm; PSO means particle swarm optimisation; DE means differential evolution. RF uses a random-forest surrogate. All receive the user's same scalar objective.

Population algorithms work with one row per candidate and one column per parameter. Numeric coordinates are mapped to physical values once. GA/DE repair integer coordinates; PSO retains continuous positions/velocities while the model receives rounded integers. RF/TPE also support categorical values.

PSO keeps pymoo's adaptive coefficients and best-particle perturbation enabled.
The earlier wrapper disabled both by default, changing the native algorithm's
exploration. Explicit `PymooPSO(options={...})` settings override these defaults
and are recorded with the run. Cached repeated geometries save simulations;
they do not add new information or prove convergence to the best design.
Optional `restart_after` starts a new swarm after a specified number of
non-improving populations. Proposal records identify each swarm and seed;
checkpoint recovery retains the restart state. The total budget and the best
score across all swarms remain owned by the campaign.

The coordinator checks that a full declared batch and all its scenarios fit within the remaining budget before asking for proposals. A leftover budget smaller than a population can remain unused.

## Step 3: turn the proposal into simulation tasks

[campaign.py](campaign.py) assigns a candidate ID, writes its parameter record and creates one request per scenario. `_prepare` creates descriptions/directories; it does not run the solver.

Each request records the builder, parameters, scenario, seed, declared dependencies and execution settings. `BuildContext` gives the builder its working directory, output stem, random seed and CPU thread allocation. The advanced builder returns `PreparedModel(scene, effective_parameters, metadata)`.

[execution.py](execution.py) schedules tasks when a pool is selected. `LocalPool` allocates one active solver per CPU/GPU slot. `MPIPool` uses MPI to farm independent tasks across an existing allocation. This does not split a single FDTD model between those task-farm workers.

Tasks can finish out of order. The scheduler retains their original indices and identities so the correct output always returns to the correct proposal. Only the coordinator updates shared campaign progress.

## Step 4: build and run gprMax

[runner.py](runner.py) writes `request.json`, sets the allocated environment, and launches a fresh Python subprocess. It supervises timeout/cancellation and keeps `stdout.log` and `stderr.log`.

[_worker.py](_worker.py) is the child process entry point. Its numbered blocks validate the request, load the builder, construct the model, call `gprMax.run`, and verify the main HDF5 output. Through the simple API, [_simple.py](_simple.py) adapts the call to the user's ordinary `build_model(parameters)` function.

The worker currently records `output.h5`, its hash, grid/time metadata and execution provenance. It does not calculate the optimisation score or require a particular receiver or S11 field. The parent runner checks the child's return status and file integrity before exposing a successful `RunResult`.

## Step 5: read the result and calculate the score

[processing.py](processing.py) prepares the evaluator, verifies its declared inputs and creates a processing directory. The simple bridge calls the user's `evaluate(parameters, output)` with a [SimulationOutput](results.py).

The current conveniences are:

```python
trace = output.receiver("probe", "Ez")
# trace.values, trace.time, trace.unit

port = output.port("feed")
# port.frequency, port.s11, port.valid, port.impedance, port.impedance_valid

frill = output.port("frills/frill1")
# The same spectrum interface for a magnetic-frill equivalent coaxial feed.
```

[quantities.py](quantities.py) contains the current receiver and S11/Zin readers. The port reader handles `ports`, `tls` and `frills` groups. Select an exact unique ID or qualify it as `ports/feed`, `tls/tl1` or `frills/frill1`; ambiguous IDs are rejected. It uses gprMax's stored field names, reference impedance, timing and masks, and does not set the objective. The convenience view does not yet expose every terminal history or antenna metric.

`output.file` gives a `Path` to the main HDF5 result for custom processing. `output.datasets()` lists dataset paths without loading arrays; it does not explain every dataset's units or metadata. `output.save_npz(...)` can retain intermediate calculations beside the evaluation records.

The advanced evaluator receives `runs`, a dictionary keyed by scenario ID, so it can combine several simulations into one objective. `ProcessingContext` supplies settings and methods for saving intermediate data. `PreparedEvaluator` writes `processing.json` even if processing fails, preserving the cause.

## Step 6: return the score and decide whether to continue

The coordinator records each `TrialResult`, then calls `tell` or `tell_batch`. The adapter checks token/parameter identity before updating its library state. Failed or cancelled trials have statuses and explanations; the framework does not invent numerical penalties for them.

Successful full-batch feedback precedes target stopping. A good first member does not cause a partial population to be reported as a complete generation. The returned `OptimisationResult` includes the best successful candidate, all trial statuses and the stopping reason.

The separate [scalar.py](scalar.py) helper directly calls a user's integer objective through SciPy. It is a small one-variable search with memoisation and neighbour checks, not the general campaign adapter loop.

## Where the saved records come from

A typical single-scenario candidate has this structure:

```text
my_results/
  campaign.json                       model and execution definition
  progress.json                       current task counts
  optimiser/
    session.json                      optimiser/evaluator settings and budgets
    batch-000001.json                 proposal group and feedback status
    trial-000000.json                 one proposal, score and candidate link
    result.json                      best result so far and stopping reason
  candidates/
    000001/
      candidate.json                 complete proposed parameters and seed
      evaluation.json                paths/statuses of the scenario runs
      objective.json                 score and optional diagnostic metrics
      default/
        task.json                    latest task/attempt status
        attempt-0001/
          request.json               input to the worker
          stdout.log, stderr.log     solver/build diagnostics
          output.h5                  saved gprMax measurements
          worker_result.json         child's completion record
          result.json                parent's verified result
          state.json                 attempt status
      processing/
        processing.json              objective inputs, result and artifacts
        ...                          arrays/JSON saved by the user objective
```

Optional recovery, retries and standalone processing add sibling directories. The different filenames describe different stages, not different required physical outputs.

| Record field | Written by | Used by |
|---|---|---|
| `parameters` | Optimiser proposal, validated by ParameterSpace/Campaign | Model builder and evaluator |
| `scenario.settings` | User's advanced Problem, or the simple bridge | Builder |
| `execution` | Executor/pool configuration | Parent runner and worker |
| `effective_parameters`, `model_metadata` | Builder via PreparedModel | Records and optional user processing |
| `artifacts.output` | Worker, verified by parent runner | Readers, cache and evaluator integrity checks |
| `value`, `metrics` | User evaluator via ObjectiveResult | Optimiser consumes value; reporting retains metrics |
| `failure.kind` | Stage that failed | Logs, retry policy and search status handling |

## Reuse and recovery

[cache.py](cache.py) can reuse a verified HDF5 for an exactly matching deterministic request. The model/environment namespace and recorded inputs determine the key. Geometry equivalence after rounding is not inferred. The objective still runs on the reused output.

[checkpoint.py](checkpoint.py) saves native optimiser state and random-generator state before asking for a batch. During recovery, the coordinator reproduces the proposals and reuses verified completed simulations. A JSON history alone is not enough to reconstruct that state.

[_storage.py](_storage.py) writes records through a temporary file and atomic replacement, and hashes artifacts in bounded chunks. This keeps incomplete writes from looking like complete records.

Recovery deliberately checks source-file hashes, including comments and docstrings. Updating toolbox documentation therefore requires a new optimisation session when using native checkpoints created with the earlier source. Saved output files remain available for inspection and processing.

For execution profiles, dependencies and recovery examples, continue with [execution-and-processing.md](execution-and-processing.md). For implementing a new adapter, use [ADAPTERS.md](ADAPTERS.md). The short [developer map](DEVELOPER_GUIDE.md) lists extension points.
