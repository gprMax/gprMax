# Advanced campaign and optimiser adapters

For the ordinary two-function workflow, start at [README.md](README.md).
For the explicit Problem/Campaign contracts, see [CAMPAIGN.rst](CAMPAIGN.rst).

Connect an external optimiser to a user-owned gprMax builder and objective. Each
candidate creates a fresh model and runs every declared scenario. The solver and
the fixed-geometry Study API are unchanged.

```python
from gprMax.toolboxes.Optimisation import PymooGA, PymooPSO, PymooDE, SkoptRF, OptunaTPE

# The same existing campaign.problem and evaluator work with each choice.
adapter = PymooPSO(population_size=6, seed=7)
result = campaign.optimise(
    optimiser=adapter,
    evaluator="my_model:evaluate",  # evaluate(parameters, runs) -> ObjectiveResult
    n_trials=24,
    max_simulations=24,
    target_value=1e7,
)
```

Use a **new campaign directory and adapter instance** for each experiment. The
objective is a finite scalar to minimise; diagnostics are arbitrary JSON metrics.
The user builder receives physical parameter values and returns `PreparedModel`,
including effective parameters and model metadata. Mesh spacing can be fixed,
derived from dimensions or varied by that builder; the coordinator has no mesh
assumptions.

## Supported adapters

| Adapter | Library | Parameter support | Feedback |
|---|---|---|---|
| `OptunaTPE` | Optuna | Real, log-real, integer, categorical | One candidate or synchronous batch |
| `SkoptRF` | scikit-optimize | Real, log-real, integer, categorical | RF + EI; one candidate or synchronous batch |
| `PymooGA` | pymoo | Real, log-real, integer, numeric mixtures | Complete generation |
| `PymooPSO` | pymoo | Real, log-real, integer, numeric mixtures | Complete swarm |
| `PymooDE` | pymoo | Real, log-real, integer, numeric mixtures | Complete generation |

`make_optimiser("ga"|"pso"|"de"|"rf"|"tpe", ...)` is a convenience factory; passing
an adapter object directly remains the general interface. Libraries are imported
only when initialising the selected adapter. Install optional packages using
`python -m pip install -r gprMax/toolboxes/Optimisation/requirements-optimisers.txt`.

Pymoo operates in unit coordinates. Real parameters decode linearly or
logarithmically exactly once; integers use the nearest value on their declared `Integer(..., step=...)` lattice,
with half-up ties. The default step is 1.
GA and DE repair integer coordinates before feedback. PSO retains continuous
positions and velocities and evaluates the corresponding rounded geometry. This
is continuous PSO on a discretised objective, not a specialised discrete PSO.
Categorical parameters are rejected by these numeric population wrappers before
simulation. TPE and RF support native categorical dimensions.

## Add another optimiser

Sequential adapters implement:

```python
initialise(parameter_space) -> dict  # JSON library/settings/capabilities
ask() -> Proposal                  # unique token, physical parameters, metadata
tell(proposal, trial_result)        # status, objective, metrics, candidate link
```

Population adapters implement `initialise` plus:

```python
batch_size: int             # upper bound required for budget admission
finished: bool              # property; native termination
ask_batch() -> tuple[Proposal, ...]  # empty also means finished
tell_batch(proposals, results)      # matching members in original order
```

`PymooOptimiser("my_algorithms:factory", population_size=..., options={...})`
wraps another compatible pymoo algorithm. The factory accepts `pop_size` and JSON
options and returns an algorithm exposing pymoo's ask/tell interface. It must
produce no more than the declared batch size, accept one unconstrained objective
and operate in unit numerical coordinates. A custom factory can configure native
operators. `integer_mode="repair"` or `"latent"` determines integer mapping.
Multiobjective algorithms and constraint vectors need a future protocol extension.

## Execution, budgets and failures

`LocalExecutor` runs serially. `LocalPool` runs candidates/scenarios concurrently on
CPU, CUDA, OpenCL or Metal workers. `MPIPool` distributes the same requests over an
existing MPI allocation. All required scenarios complete before the user evaluator
runs; full batches are returned to the optimiser in original proposal order.

See [execution and processing](execution-and-processing.md) for resource profiles,
custom signal/image processing, waveform matching, caching, retries and recovery.

```python
from gprMax.toolboxes.Optimisation import LocalPool
execution = LocalPool(solver="cuda", devices=[0, 1], cpu_threads_per_worker=2)
campaign = Campaign(problem, "results/new-run", execution=execution)
```

For RF/TPE use `batch_size=...` to propose several candidates together. RF uses
scikit-optimize's `cl_min` batch strategy; TPE enables constant-liar handling of
pending trials. These are synchronous batches; changing batch size changes the
proposal/feedback schedule and can change the optimisation trajectory.

The coordinator reserves the full declared population size and all its scenarios
**before asking**. Unused budget smaller than a population is left unspent. Once
admitted, a successful population completes before target stopping. Thus a target
found by the first member can still require the rest of that batch. A library's
own termination reports `optimiser_finished`; budget exhaustion never implies
convergence. The pymoo adapters default to campaign-controlled stopping;
`generations=...` adds a native generation cap. `stop_on_target=False` runs to the budget/native termination while
still reporting `target_met` for a fixed-budget comparison.

On a failure, the default policy cancels unstarted population members, records
all statuses, and delivers the full result set. Pymoo aborts the session without
supplying invented fitness values; `on_failure="continue"` is rejected for those
adapters. TPE marks failed trials as failed; RF excludes failed observations from
training. Retries default to zero. Pools accept `max_retries` for timeout, launch and worker-exit
failures; model/processing failures are not retried automatically. Attempts have
separate directories. No numerical failure penalty is invented. Already-running
tasks finish after a failure; queued work is cancelled.

Duplicate proposals receive fresh simulations by default. An opt-in
`SimulationCache` reuses exact requests and coalesces simultaneous identical
requests. It does not infer geometry equivalence from rounded dimensions.
`n_trials` counts proposals (including cancelled members); `max_simulations` counts
worker attempts. Keep both candidate and actual simulation counts when comparing
algorithms or multi-scenario problems.

## Audit records

`campaign.json`: builder, parameter space, scenarios, dependency hashes, executor.
`optimiser/session.json`: library/version, settings, evaluator hash, budgets, policy.
`optimiser/batch-*.json`: member tokens and population feedback delivery status.
`optimiser/trial-*.json`: proposal, coordinate mapping, generation/member, candidate,
objective, timings and feedback status. All proposals are committed before the
batch starts; all objective values are committed before optimiser feedback.
`optimiser/result.json`: incremental best result, stop reason and target status.

Set `checkpoint=True` to save the native in-memory optimiser before each ask.
`Campaign.resume(...)` plus `optimise(resume=True, ...)` recovers a stopped session,
verifies definitions and saved outputs, and replays an interrupted batch from the
saved optimiser state. Default audit files alone are not checkpoints. Only load
your own checkpoint files; native optimiser state uses pickle. Adapters with
external side effects need their own recovery integration. Constraint vectors,
multiobjective optimisation and asynchronous feedback remain future work.

## 1 GHz dipole comparison

For an editable example using ordinary model/objective functions, use
[examples/thin_wire_dipole.py](examples/thin_wire_dipole.py). The comparison below
uses the advanced CLI in `examples.advanced.thin_wire_dipole`.

```bash
python -m gprMax.toolboxes.Optimisation.examples.compare_dipole_optimisers \
  /absolute/path/to/new-comparison --optimisers ga pso rf de tpe \
  --dz-m 0.001 --cycles 100 --n-trials 18 --population-size 6 --cpu-threads 4
```

This deliberately uses a fixed evaluation budget to expose later feedback-driven
proposals even when an early candidate meets the target. Each algorithm uses the
same physical definition: 1 GHz target, 100 ns time window, 1 mm axial mesh,
approximately 2.998 mm transverse mesh, 1 mm feed gap and fixed wire radius.
Symmetric total-length increments are 2 mm. The objective is the absolute error
of the estimated S11-dip frequency, in Hz; tolerance is 10 MHz.

The one-cell feed gap differs from earlier, coarser examples. Comparisons between
optimisers use the same new gap, mesh, domain, excitation and objective. A fitted
frequency is an estimate within this model; meeting tolerance is not evidence of
mesh convergence or global optimality. A single seed on this trivial problem
demonstrates integration mechanics, not an algorithm performance ranking.

APIs: [pymoo external evaluation](https://pymoo.org/algorithms/usage.html),
[scikit-optimize Optimizer](https://scikit-optimize.readthedocs.io/en/stable/modules/generated/skopt.Optimizer.html).
