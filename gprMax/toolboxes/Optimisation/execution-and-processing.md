# Advanced execution and result processing

Start with [the user workflow](README.md) for one model and one objective. This guide
explains explicit Campaign contracts, several scenarios, MPI, caching and recovery.

The user supplies a parameter space, a model builder and an objective. The
execution configuration controls where the simulations run. None of these APIs
requires changes to the gprMax solver or to fixed-geometry Studies.

## Choose the execution resources

```python
from gprMax.toolboxes.Optimisation import Campaign, LocalPool, PymooGA

execution = LocalPool(workers=4, cpu_threads_per_worker=2)
# Alternative: one independent solver per visible GPU.
# execution = LocalPool(solver="cuda", devices=[0, 1], cpu_threads_per_worker=2)
# execution = LocalPool(solver="opencl", devices=[0, 1])
# execution = LocalPool(solver="metal", devices=[0])

campaign = Campaign(problem, "results/new-campaign", execution)
result = campaign.optimise(
    optimiser=PymooGA(population_size=12, seed=7),
    evaluator="my_model:evaluate",
    n_trials=120,
    max_simulations=120,
    checkpoint=True,
)
```

Each worker builds and solves a complete model. Population size is independent
of worker count. The scheduler fills available slots and retains proposal IDs
when results finish out of order. Geometry is rebuilt for each fresh simulation.
This does not distribute one FDTD domain across several devices.

`solver` selects CPU/CUDA/OpenCL/Metal. `precision` defaults to `single`;
Metal accepts single precision and device 0 only. `timeout` bounds each worker
attempt. Local pools reject CPU thread allocations larger than the visible CPU
count unless `allow_oversubscription=True`. CPU affinity and cluster resource
allocation remain the launcher's responsibility. Memory must fit each model;
the toolbox does not predict or enforce per-model RAM/VRAM limits.

GPU IDs refer to devices visible inside the worker's environment. The toolbox
preserves `CUDA_VISIBLE_DEVICES`; it does not assume physical/global GPU numbers.
Backend packages and model features must be supported by the selected gprMax
backend. Unsupported execution is recorded as a failed run; there is no silent
CPU fallback. Model preparation and kernel setup may dominate very small models.

Reusable JSON profiles contain the constructor settings:

```json
{"backend": "local", "solver": "cpu", "workers": 4, "cpu_threads_per_worker": 2}
```

```python
from gprMax.toolboxes.Optimisation import execution_from_profile
execution = execution_from_profile("server.json")
```

## MPI and cluster allocations

Install optional execution dependencies with
`python -m pip install -r gprMax/toolboxes/Optimisation/requirements-execution.txt`
(`mpi4py` for MPI, `cloudpickle` for checkpoints), and use the site's
MPI runtime. Enter the pool on **every rank**; create the campaign only where the
context yields an execution object:

```python
from gprMax.toolboxes.Optimisation import MPIPool

with MPIPool(solver="cpu", cpu_threads_per_worker=2) as execution:
    if execution is not None:
        campaign = Campaign(problem, "shared/results/new-run", execution)
        result = campaign.optimise(
            optimiser=PymooGA(population_size=12, seed=7),
            evaluator="my_model:evaluate", n_trials=120, checkpoint=True,
        )
```

For one coordinator and four workers, launch that script with
`mpiexec -n 5 python my_optimisation.py`. On a cluster, run the same pool inside an
allocated scheduler job using the site's supported launcher. A worker pool serves
successive generations, avoiding a separate scheduler submission per candidate.

MPI workers use `mpi4py.futures.MPICommExecutor` over the existing allocation;
there is no dynamic MPI spawn and no nested gprMax task farm. Each worker launches
an isolated solver subprocess. The Python environment, toolbox, model modules,
dependencies and output directory must be accessible at the same absolute paths
on all nodes. Configure OpenMP allocations/binding with the cluster launcher.

For GPUs, explicitly supply each worker rank's **visible** device:

```python
# One node, two GPUs visible to both worker ranks; launch three ranks total.
with MPIPool(solver="cuda", device_by_rank={1: 0, 2: 1}) as execution:
    if execution is not None:
        ...
```

With scheduler-isolated visibility, each worker may instead see its allocated
GPU as device 0. The mapping must include every worker rank. The toolbox detects
duplicate assignments within a host/visibility namespace; the cluster allocation
must still provide exclusive devices. CPU and multi-GPU MPI execution are
different hardware configurations and should be validated on the target server.

## Convert outputs into an objective

The existing interface accepts arbitrary user processing:

```python
from gprMax.toolboxes.Optimisation import ObjectiveResult, read_receiver

def evaluate(parameters, runs):
    trace = read_receiver(runs["default"].output_file, "received", "Ez")
    # User-owned processing: windowing, filtering, FFT, comparison, inversion,
    # image reconstruction, feature extraction, or scenario aggregation.
    error, diagnostics = compare_with_measurement(trace.time, trace.values)
    return ObjectiveResult(error, diagnostics)
```

`runs` maps scenario IDs to `RunResult` objects, including output file paths,
effective parameters and simulation provenance. Readers expose physical axes and
units. Other gprMax outputs can be read directly using h5py or existing gprMax
postprocessing tools. The toolbox imposes no particular signal or image model.

For settings and saved arrays, use the explicit contextual interface:

```python
from pathlib import Path
from gprMax.toolboxes.Optimisation import Evaluation, ObjectiveResult

evaluation = Evaluation(
    "my_processing:compare",
    settings={"reference": "/shared/data/measured.npz", "time_gate_s": [2e-9, 8e-9]},
    dependencies=(Path("/shared/data/measured.npz"), Path("my_processing_helpers.py")),
)

def compare(parameters, runs, context):
    features, score, diagnostics = process_outputs(runs, context.settings)
    context.save_npz("features.npz", feature=features, units={"feature": "1"})
    return ObjectiveResult(score, diagnostics)

# campaign.optimise(..., evaluator=evaluation)
```

Use a two-argument callback for a string reference and a three-argument callback
for `Evaluation`. Processing settings and dependency hashes are snapshotted;
changes during a campaign are rejected. A `processing/processing.json` manifest
records input/output hashes, settings, objective and diagnostics. Large arrays
remain artifacts rather than entering the optimiser's scalar feedback.

The objective is one finite scalar to **minimise**. Maximisation can return the
negative utility. Weighted combinations must use the user's declared scaling.
Invalid simulation data or processing should raise an exception; the framework
records a failed trial instead of inventing a penalty. Native constraint vectors
and multiobjective feedback are not yet supported.

Processing currently runs on the coordinator after the simulation batch;
expensive postprocessing should be profiled before choosing larger worker pools.

## Complete waveform example

```bash
python -m gprMax.toolboxes.Optimisation.examples.advanced.waveform_matching results/waveform
```

This runs one reference model to create **synthetic** data, then eight RF proposals
in batches of two. Each candidate's `Ez` trace is interpolated onto the reference
times, optionally gated and mean-subtracted, and converted to normalised RMS
error. Extrapolation, mismatched units and zero reference RMS are rejected.
The reference, predicted waveform and residual are saved in `waveforms.npz`.

The same example accepts `--execution-profile server.json`. For MPI use an MPI
profile and launch all ranks with mpiexec. The simulator, processing callback and
optimiser are the same for local and distributed execution.

Reprocess saved data without new simulations:

```python
from gprMax.toolboxes.Optimisation import load_candidate, evaluate_outputs
parameters, runs = load_candidate("results/run/candidates/000001")
objective = evaluate_outputs(evaluation, parameters, runs, "results/reprocessed-000001")
```

This interface also supports generating ML labels. Splitting datasets, batching
training data and training surrogate models remain user/application concerns.

## Duplicate requests, budgets and failures

```python
from gprMax.toolboxes.Optimisation import SimulationCache
cache = SimulationCache("shared/cache", namespace="gprmax-build-and-environment-v1")
campaign = Campaign(problem, "results/cached-run", execution, cache=cache)
```

Caching is opt-in and assumes deterministic simulation for the declared inputs.
The namespace identifies the solver build and Python environment: change it when
either changes. The key also includes builder source, declared dependencies,
exact parameters, scenario, seed and execution configuration. Declare every
helper module and input asset on `Problem.dependencies`. The cache does not
automatically hash the complete gprMax build or infer equivalence between distinct
parameters that happen to produce the same geometry. Output integrity is checked
before reuse. Exact simultaneous duplicates share one successful simulation.

`n_trials` counts proposals. `max_simulations` counts worker attempts; cache hits
consume no attempts. Full batches and all scenarios are admitted conservatively
before asking. If retries are enabled with `max_retries`, admission reserves the
maximum attempt count; unused reservations are released after the batch. Cache
hits are not predicted before asking, so conservative admission can leave budget
unused. Budgets are not convergence criteria.

Retryable failures are timeout, worker exit and launch failure. Model-building,
solver-reported model errors and processing errors are recorded without automatic
retries. On failure, queued work is cancelled under `on_failure="stop"`; running
tasks finish and their results are retained. Local interruption terminates solver
process groups. MPI interruption may wait for running attempts to finish or time
out. Loss of an MPI rank can abort the allocation; recovery requires restarting it.

`progress.json`, per-attempt `state.json` and logs expose progress, attempts,
failures, cache hits and worker allocation. Worker results include host/device,
wall time, model preparation and solver-call timings. The solver-call timing
includes gprMax geometry preparation and accelerator setup; it is not kernel-only
timing or a GPU utilisation measurement.

## Resume a stopped optimisation

Enable `checkpoint=True` before the first run. After the old workers/allocation
have stopped:

```python
campaign = Campaign.resume(problem, "results/run", execution=execution)
result = campaign.optimise(
    evaluator=evaluation, n_trials=120, max_simulations=130, resume=True,
)
```

The saved adapter is restored, so do not construct another optimiser. Keep seed,
target and failure settings identical to the original call. Budgets may be raised
to allow recovery attempts. A terminal checkpoint returns the completed result;
it does not extend a finished experiment.

Recovery verifies the model, evaluator, toolbox source and optimiser version. It
requires the original execution configuration. It restores native optimiser and
random state from before an ask, verifies regenerated proposals against the
journal, reuses verified completed runs and processing artifacts, and reruns
failed/incomplete tasks in new attempt directories. The old attempt files remain.
An interrupted batch receives complete feedback in the restored in-memory state.

Checkpoint files contain pickle data: only resume your own trusted campaigns.
The supported in-memory adapters are tested for state restoration. Arbitrary
adapters that mutate external databases/services need a separate recovery
integration; replay is not an exactly-once transaction with external systems.

References: [MPI futures](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html),
[RF batch proposals](https://scikit-optimize.readthedocs.io/en/stable/modules/generated/skopt.Optimizer.html),
[Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).
