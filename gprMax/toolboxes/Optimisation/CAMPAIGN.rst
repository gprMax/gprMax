Advanced campaign reference
===========================

This toolbox runs fresh user-built gprMax models from physical parameter
values. It provides a common evaluation boundary for optimiser adapters
and subsequent labelled simulation datasets. It does not modify gprMax or
its existing fixed-geometry Studies.

Implemented: real/integer/string-category parameters, physical-value
validation, importable model builders, scenarios, serial or pooled CPU/GPU subprocesses,
explicit thread budgets, timeout handling, logs, atomic result manifests,
dependency hashes, named receiver traces and saved S11/Zin extraction.
Optional Optuna TPE, scikit-optimize RF, and pymoo GA/PSO/DE adapters connect
serial or population ask/tell to an importable joint evaluator. Population
budgets, generation/member records and explicit integer coordinate mapping are
implemented. See `the adapter guide <ADAPTERS.md>`_ for supported parameter types,
extension interfaces and the 1 mm dipole comparison. The earlier optional SciPy
integer scalar helper remains available separately.

Local and MPI pools, opt-in exact-request caching, bounded infrastructure retries,
native optimiser checkpoints and contextual result processing are implemented.
See `execution and processing <execution-and-processing.md>`_ for current usage.
New campaign directories are never overwritten; stopped campaigns require explicit
checkpoint recovery. Dataset export and fixed-geometry Study integration remain
separate future work.

Run the example
---------------

Activate a working gprMax v4/devel environment. From the repository root::

    python -m gprMax.toolboxes.Optimisation.examples.dielectric_block /absolute/path/to/new-campaign

The example builds two dielectric blocks in air with different widths and
relative permittivities. It prints each run's status and HDF5 location.
These are small integration examples, not calibrated GPR antenna models.

Use your own model
------------------

Put a top-level function in an importable user module::

    from gprMax.toolboxes.Optimisation import PreparedModel

    def build_model(parameters, scenario, context):
        import gprMax
        scene = gprMax.Scene()
        # Add Domain, Discretisation, TimeWindow, geometry and sources.
        # Use parameters["width"], parameters["permittivity"], etc.
        # Use scenario.settings for fixed experimental conditions.
        return PreparedModel(
            scene,
            effective_parameters=dict(parameters),
            metadata={"description": "my model"},
        )

The user owns all geometry and physical relationships. The builder receives
physical values, not normalised optimiser coordinates. ``Real.from_unit``
is an explicit optional conversion helper; candidate tables always contain
physical values. Bounds are inclusive. Log bounds must be positive and
categorical choices are strings. Coupled physical validation can raise
ValueError in the builder before the solver is launched.

Create a campaign::

    from pathlib import Path
    from gprMax.toolboxes.Optimisation import (
        Campaign, LocalExecutor, ParameterSpace, Problem, Real, Scenario,
        read_receiver,
    )

    problem = Problem(
        parameters=ParameterSpace({
            "width": Real(0.008, 0.016, unit="m"),
            "permittivity": Real(2, 8),
        }),
        builder="my_models.block:build_model",
        scenarios=(Scenario("free_space", {}),),
        version="1",
        dependencies=(Path("my_models/block.py"),),
    )
    execution = LocalExecutor(cpu_threads=1, timeout=300)
    campaign = Campaign(problem, Path("new-campaign"), execution)
    results = campaign.evaluate([
        {"width": 0.008, "permittivity": 4},
        {"width": 0.016, "permittivity": 6},
    ])
    for candidate in results:
        for run in candidate:
            if run.status == "complete":
                trace = read_receiver(run.output_file, "received", "Ez")
                # trace.values, trace.time, trace.unit
            else:
                print(run.record["failure"])

``evaluate_one(parameters, seed=...)`` is also available for external
drivers. It returns one RunResult per scenario; the caller decides how
to score outputs or report failed trials. No failure is silently converted
into an optimisation penalty.

Connect an optimiser
--------------------

Install the optional dependency in your chosen gprMax environment::

    python -m pip install "optuna==4.9.0"

Define an importable evaluator in your own module::

    from gprMax.toolboxes.Optimisation import ObjectiveResult, read_receiver

    def evaluate(parameters, runs):
        # runs maps every scenario ID to its completed RunResult.
        trace = read_receiver(runs["free_space"].output_file, "received", "Ez")
        loss = float((trace.values ** 2).mean())
        return ObjectiveResult(loss, metrics={"mean_squared_field": loss})

Then use the same campaign and model builder::

    from gprMax.toolboxes.Optimisation import OptunaTPE

    result = campaign.optimise(
        optimiser=OptunaTPE(seed=0, n_startup_trials=3),
        evaluator="my_models.objectives:evaluate",
        n_trials=12,
        max_simulations=24,
    )
    if result.best is not None:
        print(result.best.parameters, result.best.value)

The optimiser owns candidate selection. The campaign validates physical values,
builds every scenario in a fresh process, calls the evaluator once with all
scenario results, commits the result and feeds it back to the same trial.
The evaluator owns the scientific definition of the scalar to minimise and
may read any saved quantity. It must return ``ObjectiveResult`` containing a
finite scalar and JSON-compatible metrics. Simulation and objective failures
receive explicit failed feedback without a numerical objective. The default
``on_failure="stop"`` stops after feedback. ``"continue"`` permits further
trials for adapters that support it; the pymoo adapters require stopping and
abort failed populations without numerical fitness penalties.

``n_trials`` limits proposals. ``max_simulations`` limits worker attempts in
that optimisation invocation, conservatively counting failed launches. A
candidate is admitted only when all its scenarios fit the remaining budget.
Population adapters require room for their full declared batch before asking.
By default, repeated physical proposals create fresh candidate IDs and runs, preserving
the sampler's decisions. There is no automatic duplicate reuse in this path.

An optional ``target_value`` supplies a success threshold in the evaluator's
own objective units. After committing and delivering a completed result with
``value <= target_value``, the campaign stops with ``stop_reason="target_reached"``.
Population adapters complete and deliver the admitted batch before checking
the threshold. ``stop_on_target=False`` continues a fixed-budget comparison
while still reporting whether the target was met.
``OptimisationResult.target_met`` explicitly reports whether the threshold was
met; it is ``None`` when no target was supplied. Exhausting a trial or simulation
budget with an unmet target reports ``target_met=False``. This threshold is a
user-defined objective criterion, not proof of a global optimum.

``OptunaTPE`` uses native Optuna distributions; log-real parameters are not
decoded twice, and integers are proposed directly. The default uses random
proposals until three startup trials complete successfully; subsequent
proposals use TPE and the objective feedback.
Multiple parameters use the same interface. Its Optuna ``Study`` stores
optimiser state; this is unrelated to gprMax's fixed-geometry Study.
The adapter was tested with Optuna 4.9.0. Optuna and SciPy are not imported by
ordinary toolbox imports or fixed-table campaigns.

The ``Optimiser`` protocol has three methods: ``initialise(parameter_space)``
returns JSON metadata, ``ask()`` returns a ``Proposal(token, parameters)``, and
``tell(proposal, TrialResult)`` receives complete/failed/cancelled feedback.
This protocol supports serial single-objective minimisation. ``BatchOptimiser``
adds ``batch_size``, ``finished``, ``ask_batch`` and ``tell_batch`` for populations,
which can be evaluated by serial, local-pool or MPI execution. A new
adapter needs no model-specific code; callback-only libraries can continue to
use ``evaluate_one`` without pretending to implement ask/tell.

An ``optimiser/`` directory contains ``session.json``, one
``trial-000000.json`` per proposal, ``batch-*.json`` population membership and
feedback records, and an incrementally updated ``result.json``.
Tokens are saved before simulation and linked from ``candidate.json``.
Objective values are committed before feedback, and delivery status is
recorded separately. Algorithm settings, library version and evaluator source
hash are recorded. Delivery errors stop the loop and preserve the committed
objective. These files are an audit trail. Native state recovery requires opting
into campaign checkpoints before starting. Contextual evaluators declare their
transitive assets through ``Evaluation.dependencies``; model assets belong to
``Problem.dependencies``. See the current execution/processing guide for recovery
requirements and limitations.

Workers and outputs
-------------------

The default interpreter is the Python running the campaign. Supply
``LocalExecutor(python="/absolute/path/to/python")`` to use another solver
environment. User code must be installed there or explicitly exposed with
``pythonpath=(Path("/absolute/path/to/user-code"),)``. Do not pass notebook
closures, already-built Scenes or accelerator contexts between processes.
Builder code executes with the user's normal permissions; this is process
isolation, not a sandbox for untrusted model code.

Each worker has an independent working directory. Use ``context.workdir``
for auxiliary files and resolve input assets explicitly rather than relying
on the launching directory. Declare code/data files in ``dependencies``;
their hashes are checked before runs. The framework records the actual
builder source, solver module/version and available Git revision; it cannot
discover every transitive dependency of arbitrary Python code.

The backend reserves the main output name ``output.h5`` and adds a public
``OMPThreads`` object if the model has none. An incompatible thread count
or an OutputDir object is rejected. The default executor runs one CPU process at a time;
``n=1`` and ``geometry_fixed=False`` are enforced. Seeded Python/NumPy
randomness is provided, but model-specific generators must also honour
``context.seed``.

The campaign contains::

    campaign.json
    candidates/000001/candidate.json
    candidates/000001/evaluation.json
    candidates/000001/free_space/attempt-0001/
        request.json
        stdout.log
        stderr.log
        output.h5
        worker_result.json
        result.json

``result.json`` is the coordinator's committed run record. Its artifact
hash is verified before reporting completion. Completion verifies the
solver HDF5/grid metadata, not every possible physical observable. Requested
receiver extraction performs its own name, vector and timing validation.
Do not use a partial output from a failed attempt as a completed label.

Timings distinguish user builder time, total solver-call time and worker
wall time. Solver-call time includes internal model preparation. Finer
FDTD-only timing is not currently claimed. Matplotlib/font caches are
redirected into each attempt and can add noticeable startup cost to tiny
simulations when the caller has not supplied ``MPLCONFIGDIR`` and
``XDG_CACHE_HOME``. Explicit caller cache paths are preserved across workers.
No outputs are deleted automatically.

Thin-wire dipole optimisation
----------------------------

For a copyable example following the ordinary two-function interface, start with
`examples/thin_wire_dipole.py <examples/thin_wire_dipole.py>`_. The commands below
use the preserved advanced CLI and its explicit Campaign contracts.

The next example varies the length of a physical symmetric dipole, with two
``ThinWire`` arms separated by a one-cell voltage-source gap::

    python -m gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole /absolute/path/to/new-dipole-campaign --target-hz 1e9 --n-trials 12 --optimiser-seed 0

The default ``--objective match`` minimises reflected power ``abs(S11(target_frequency))**2``, using
the solver's saved gap-corrected S11 and validity mask, with a fixed 50 Ohm
reference. The terminal pulse must decay below -40 dB before scoring.
``--cycles`` controls the time window, ``--cells-per-wavelength`` the spatial
resolution, and ``--resistance`` the fixed feed reference in Ohms.

The parameter is an integer number of cells per arm. Both arms change
symmetrically; total end-to-end length includes the fixed feed gap. Default
length bounds cover approximately 0.36 to 0.56 free-space wavelengths. Mesh,
domain, radius and feed stay fixed within a search. The default radius is
0.2 transverse cell widths; ``--radius-m`` fixes it explicitly in metres.
``--axial-refinement`` divides the cell spacing along the wire while keeping
the transverse mesh fixed. Axial PML cell counts scale with refinement to
preserve their physical thickness. The actual values are recorded in
``optimisation.json``. The feed remains a single axial cell, so its physical
gap shrinks with axial refinement; this is not a study of mesh convergence
with identical physical feed geometry.

The example uses ``Campaign.optimise`` with ``OptunaTPE`` and the independent
``evaluate_s11`` evaluator. All candidates come from the optimiser; no prescribed
length table or half-wavelength answer is supplied. The default runs twelve
trials with three startup trials, and the plot shows evaluation order and best
S11 so far. The first optimiser proposal supplies the plotted initial spectrum.
The goal is to demonstrate the replaceable optimiser/model/objective connection;
the near-half-wavelength antenna is a physically interpretable test problem.
The finite trial budget does not establish a global optimum. Failed simulations
or invalid S11 stop this example after failed feedback is delivered.

To place the S11 minimum at a given frequency, select the independent resonance
evaluator and an explicit frequency tolerance::

    python -m gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole /absolute/path/to/new-resonance-campaign --objective resonance --target-hz 1e9 --cycles 100 --frequency-tolerance-hz 1e7 --axial-refinement 2 --radius-m 0.000599584916 --n-trials 24 --optimiser-seed 7 --cpu-threads 4

``evaluate_resonance`` returns the absolute S11-dip frequency error in Hz.
The same campaign/adapter interface is called with ``target_value=1e7`` to
stop within 10 MHz of 1 GHz. If the budget expires first, the result explicitly
reports that the target was not reached. The default matching objective remains
available; it answers a different scientific question.

The resonance evaluator uses a quadratic fit to reflected power at three bins
around the sampled minimum in the 0.6--1.4 target-frequency band. It requires
valid data throughout that band, an interior dip below -10 dB, a valid local
fit, sufficient terminal decay and independent frequency spacing no coarser
than the requested tolerance. The sampled minimum, fitted frequency, frequency
error and interpolation bracket are all saved. Interpolated frequencies are
estimates within the numerical model; they do not imply physical accuracy
beyond the mesh, time window and antenna model.

At 1 GHz, the command above retains a 100 ns window and the previous physical
wire radius, and reduces symmetric length increments from approximately 6 mm
to 3 mm. Its convergence plot shows the actual frequency-error objective and
the requested stopping threshold.

The framework incrementally writes trial/feedback records and each candidate's
``objective.json``. The example also writes ``search_history.json`` at completion,
final ``optimisation.json`` and ``dipole_optimisation.png``. Diagnostics include
the sampled S11 minimum and an interpolated input-reactance zero near the target,
which are distinct from best matching to the fixed reference impedance.
Interpolation at the target uses adjacent valid complex S11 bins; it never
bridges masked bins.

Length quantisation and finite time windows limit tuning precision. Read
the reported length increment and frequency-bin spacing. For a validated
antenna design, check time-window, mesh and boundary convergence at fixed
physical radius. The current gprMax ThinWire implementation documents that
its charge-based open-end correction is not implemented; the example is
an optimisation demonstrator, not an analytical-resonance certification.

Testing and next increments
--------------------------

From the repository root, in the gprMax environment::

    python -m pytest tests/toolboxes/test_optimisation.py -q

Tests cover physical/log parameter semantics, independent output paths,
invalid candidates, changed dependencies, worker failures/timeouts,
receiver timing/name selection, masked port spectra, integer search, dipole
feed geometry and actual CPU simulations with changed geometry/materials.
Optimiser tests cover joint scenario objectives, feedback affecting later
proposals, budgets, native mixed parameter distributions, repeated proposals,
failed trials, feedback delivery errors, threshold stopping versus budget
exhaustion, resonance estimation and rejection of unresolved spectra.
Optuna-specific tests skip when
the optional dependency is absent. A repeated candidate must produce identical
field arrays in the same environment.

Current execution, processing, caching and checkpoint interfaces are documented in
`execution and processing <execution-and-processing.md>`_. Dataset export,
constraint vectors, multiobjective and asynchronous optimisation remain future work.
