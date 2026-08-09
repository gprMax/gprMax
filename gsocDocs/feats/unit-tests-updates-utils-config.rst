Unit Tests — Updates, Solver, Config, Utilities and VTKHDF
==========================================================

**Branch:** ``feat/unit-tests-updates-utils-config``

**Modules under test:**
   - ``gprMax/updates/updates.py`` — the ``Updates`` ABC: the eleven-method
     contract every backend implements
   - ``gprMax/updates/cpu_updates.py`` — ``CPUUpdates``, the CPU orchestrator,
     including the sixteen-name dispersive kernel dispatch
   - ``gprMax/solvers.py`` — ``Solver.solve``, the canonical timestep running
     order, and ``create_solver``'s grid-type dispatch
   - ``gprMax/config.py`` — ``SimulationConfig`` and ``ModelConfig``: the
     module-level global every other module reads
   - ``gprMax/utilities/utilities.py`` — rounding that is deliberately not
     Python's rounding, natural sorting, the FFT power spectrum, the timer
   - ``gprMax/utilities/host_info.py`` — host, CPU and device probing on all
     three platforms, the OpenMP environment, and the memory checks
   - ``gprMax/utilities/logging.py`` — the custom ``BASIC`` level welded into
     the standard library at import, and the colour formatter
   - ``gprMax/vtkhdf_filehandlers/vtkhdf.py`` — ``VtkHdfFile``: file naming,
     root attributes, and every dataset write
   - ``gprMax/vtkhdf_filehandlers/vtk_image_data.py`` — ``VtkImageData``, the
     voxel writer
   - ``gprMax/vtkhdf_filehandlers/vtk_unstructured_grid.py`` —
     ``VtkUnstructuredGrid``, the line-geometry writer

**Covered transitively:**
   - ``gprMax/cython/fields_updates_normal.pyx`` ``update_electric`` /
     ``update_magnetic`` — driven as real kernels against real arrays, with the
     written region asserted cell by cell
   - ``gprMax/cython/fields_updates_dispersive`` — not executed, but **all
     sixteen** function names the dispatch can construct are resolved against
     the compiled extension
   - ``gprMax/updates/metal_updates.py`` — named only where the base class's
     contract touches it; see *Deliberately Untested Paths*

**Test files:**
   - ``tests/unit/updates/test_updates_base.py`` (57 tests)
   - ``tests/unit/updates/test_cpu_updates.py`` (91 tests)
   - ``tests/unit/updates/test_dispersive_dispatch.py`` (76 tests)
   - ``tests/unit/updates/test_solver.py`` (45 tests)
   - ``tests/unit/config/test_simulation_config.py`` (78 tests)
   - ``tests/unit/config/test_precision_dtypes.py`` (54 tests)
   - ``tests/unit/config/test_model_config.py`` (46 tests)
   - ``tests/unit/config/test_model_registry.py`` (24 tests)
   - ``tests/unit/config/test_output_paths.py`` (30 tests)
   - ``tests/unit/utilities/test_utilities.py`` (101 tests)
   - ``tests/unit/utilities/test_logging.py`` (51 tests)
   - ``tests/unit/utilities/test_host_info.py`` (66 tests)
   - ``tests/unit/utilities/test_omp_threads.py`` (33 tests)
   - ``tests/unit/utilities/test_mem_checks.py`` (35 tests)
   - ``tests/unit/utilities/test_device_detection.py`` (58 tests)
   - ``tests/unit/vtkhdf/test_vtkhdf_base.py`` (64 tests)
   - ``tests/unit/vtkhdf/test_vtkhdf_datasets.py`` (55 tests)
   - ``tests/unit/vtkhdf/test_vtk_image_data.py`` (51 tests)
   - ``tests/unit/vtkhdf/test_vtk_unstructured_grid.py`` (46 tests)

**Total: 1061 tests** from 848 test functions across 137
classes, all passing, **no** ``xfail`` **and no** ``skip``.

**Shared fixtures:** ``tests/unit/updates/conftest.py``,
``tests/unit/config/conftest.py``, ``tests/unit/utilities/conftest.py``,
``tests/unit/vtkhdf/conftest.py``

Scope
-----

Ten PRs have tested what a gprMax model *is* — waveforms, materials, sources,
receivers, the hash parser, user objects, geometry primitives, fractals, the
grid, the PML at the domain edge and everything that leaves as a file. None of
them tested the machinery that **runs** the thing.

This suite covers four layers of that machinery, and they are bound together by
one property: each is a *protocol* rather than a calculation. A timestep is an
order of operations. A configuration is a global read by name from a dozen
modules. A host probe is a shell command whose stdout is parsed by hand. A file
writer is a layout another program will read. None of them produces a number
you can check against a formula, and every one of them fails silently:

- a timestep in the wrong order runs to completion and writes a plausible file;
- a dispersive kernel name that no longer exists fails only for users with
  three-pole Lorentz materials in double precision;
- ``dtypes`` never being set surfaces as an ``AttributeError`` in a different
  file;
- a machine reporting 2 sockets instead of 12 just runs slowly;
- a transposed cell-data array opens fine in ParaView and shows the wrong
  thing.

All of it is nevertheless exactly assertable, which is why it is here: a call
sequence, a sixteen-name matrix, an eight-row dtype table, a rounding table, a
fixed set of dictionary keys, a byte-level file layout.

``config.py`` is the reason to do this now rather than later. Every one of the
ten existing directory conftests replaces ``config.sim_config`` and
``config.get_model_config`` with ``SimpleNamespace`` stand-ins — the right call
each time, but it means the real ``SimulationConfig``, the real ``ModelConfig``
and the real model-index bookkeeping have never been executed by a test.
``tests/unit/config/`` is the one suite in the project that builds and drives
them for real.

Diagnosing a Failure
--------------------

The traps that cost the most time are cross-cutting. Check these before reading
the per-test entries.

**A whole file errors at collection.** Wrong interpreter. The suite needs the
``gprMax`` conda environment — the base environment has no ``cython``, and
``gprMax-devel`` has no ``pytest``. ``python -m pytest`` from the wrong prompt
fails at ``import gprMax.config``.

**A patched kernel is not called, and the real one runs instead.**
``cpu_updates.py`` binds its kernels as **module globals at import**:

.. code-block:: python

   from gprMax.cython.fields_updates_normal import update_electric, update_magnetic

So a patch must target ``gprMax.updates.cpu_updates.update_electric``, never
``gprMax.cython.fields_updates_normal.update_electric``. The same applies to
``store_outputs`` and to ``timer``. This is the single most common way to write
a test in this suite that silently passes for the wrong reason.

**A config probe shells out to real hardware.** ``gprMax/config.py`` imports
``get_host_info``, ``detect_cuda_gpus``, ``detect_opencl``, ``detect_metal``
and ``get_terminal_width`` **by name**, so they must be patched on
``gprMax.config`` — patching them at their definition site in
``gprMax.utilities.host_info`` has no effect. The autouse ``no_host_probes``
fixture does this for the whole config suite.

**A test in an unrelated directory starts failing after a config test fails.**
``tests/unit/config/`` writes to ``gprMax.config.sim_config`` — the global every
other suite monkeypatches. The autouse ``restore_config_globals`` fixture saves
and restores it. If that fixture is removed, failures cascade across
directories in a way that depends on collection order.

**An ``OMP_*`` assertion behaves differently on your machine than on CI.**
``set_omp_threads`` branches on whether ``OMP_NUM_THREADS`` is already set. A
developer who exports it in their shell gets a different path. The autouse
``clean_omp_environment`` fixture deletes six variables before every test.

**A banner or padding assertion is off by a few characters.**
``get_terminal_width()`` reads the real terminal — variable interactively, 80
under pytest with no tty, different again on each of the three CI runners. Any
assertion on a padded string has to pin it. ``TERMINAL_WIDTH = 100`` in the
config conftest exists for this.

**A VTKHDF shape assertion looks transposed.** It is. ``_write_dataset``
transposes by default, because VTKHDF stores datasets ZYX-major, and
``numpy.transpose`` with no argument reverses **all** axes. Compare against the
raw on-disk array — the point of these tests is the on-disk order, so they
deliberately do not transpose it back.

**Fixing a known defect turns no test red.** That is by design. No test in this
suite asserts broken behaviour and none is marked ``xfail``; where a defect made
a contract untestable, the test was omitted. Every affected path is listed
under *Deliberately Untested Paths*, and each has a write-up carrying the tests
its fix should add.

The Running Order
-----------------

FDTD does not advance E and H together. It leapfrogs them: H is updated half a
timestep ahead of E, using the E field as it stands; then E is updated using
the H field that was just produced. Each is always reading the other at a
half-step offset, in both time and space. That is the Yee scheme, and its
consequence for this PR is that **a timestep is an order, not a set**.

``Solver.solve`` is where that order lives, and it lives nowhere else —
``CPUUpdates`` has no "run one timestep" method. Eleven calls per iteration:

.. code-block:: text

   store_outputs(i)                 ← sample receivers at time n
   store_snapshots(i)
   update_magnetic()                ← H: n-1/2 → n+1/2
   update_magnetic_pml()
   update_magnetic_sources(i)
   update_plane_waves_magnetic(i)   ← CPU backend only
   update_electric_a()              ← E: n → n+1, part one
   update_electric_pml()
   update_electric_sources(i)
   update_plane_waves_electric(i)   ← CPU backend only
   update_electric_b()              ← E: part two

bracketed by ``time_start()`` before the loop and
``finalise() → calculate_solve_time() → cleanup()`` after it.

**Why the electric update is split.** For a non-dispersive model
``update_electric_b`` does nothing at all. It exists for dispersive materials,
whose recursive convolution needs *both* the field before the update and the
field after it. Part A advances E and accumulates the dispersive contribution;
the PML and the sources then modify E; part B updates the pole accumulators
``Tx``/``Ty``/``Tz`` using the final value. Merging the two would use a stale E
for the accumulators. That is why part B takes 13 arguments and no
``updatecoeffsE`` and no H fields — it is not a field update.

**Why the kernels are all-positional.** Every call from ``cpu_updates.py`` into
Cython passes arguments positionally: 12 for ``update_magnetic``, 12 for the
non-dispersive ``update_electric_a``, 17 for the dispersive one, 13 for
``update_electric_b``. A swapped pair type-checks, compiles and runs. The
argument-order assertions in ``test_cpu_updates.py`` are the only thing in the
project that would notice.

**Sixteen names from four switches.** ``set_dispersive_updates`` builds a
kernel name by formatting four independent binary choices into one string::

   "update_electric_dispersive_{poles}pole_{half}_{precision}_{dispersion}"

with ``poles`` ∈ {``1``, ``multi``}, ``half`` ∈ {``A``, ``B``}, ``precision`` ∈
{``float``, ``double``} and ``dispersion`` ∈ {``real``, ``complex``} — sixteen
names, every one resolved by ``getattr`` on a module whose contents are
generated from a jinja template at build time. There is **no static link**
between the template and the dispatcher, so a rename on either side is caught
only at runtime, only for the user whose model happens to need that
combination. Parametrising the full matrix against the real compiled extension
is the single highest-value test in this PR.

The Global Config
-----------------

``gprMax.config`` is a module with a module-level variable in it:

.. code-block:: python

   sim_config: SimulationConfig = None

   def get_model_config() -> ModelConfig:
       return sim_config.get_model_config()

Everything downstream reaches for configuration through that no-argument call.
There is no key, no argument, and no object passed down the chain. The answer
comes from two pieces of mutable state: a list of ``ModelConfig`` slots and an
integer cursor into it. The context loop moves the cursor between models;
everything else silently follows.

Two construction-order facts govern the whole config suite.

**``ModelConfig`` cannot exist before ``SimulationConfig``.**
``ModelConfig.__init__`` reads the module-level ``sim_config`` for the banner
string, for ``model_end`` and for ``args.n``. So a ``ModelConfig`` built before
one is installed raises ``AttributeError`` on ``None``. The ``make_model_config``
fixture installs one first, always.

**Precision is a fork in the road.** One string in ``general["precision"]``
selects six ``dtypes`` entries at once — the NumPy real and complex types, the
two Cython shadow types, and two C type-name strings substituted into
accelerator kernel source. It also determines the ``float``/``double`` component
of every dispersive kernel name. Eight rows of precision × solver, asserted in
full, including the Cython types by identity (``is cython.float``) because they
are not comparable any other way.

The Host Probe
--------------

``get_host_info`` runs three or four external commands, parses their stdout
with string surgery, and returns nine keys that appear in the run banner, size
the OpenMP thread pool and gate the memory warnings. It is the most-shelled-out
function in the package and, until this PR, the least tested.

It also contains this project's own merged contribution. Microsoft removed
``wmic`` in Windows 11 25H2, so ``subprocess.check_output(["wmic", ...])``
raises ``FileNotFoundError`` — which the original ``except`` clause did not
catch — and gprMax crashed on startup before printing anything. The fix
(``ce2c456e``) widened three clauses and added a PowerShell ``Get-CimInstance``
fallback to each. ``gsocDocs/feats/setup-and-wmic-fix.rst`` records that it was
verified *by temporarily adding print statements inside each except block*.

That is the problem this suite solves. Every ``subprocess.check_output`` call is
served from a table keyed on argv, ``sys.platform`` is patched, and so are the
five ``platform`` lookups and the two ``psutil`` counts. Three properties
follow, and none is achievable any other way:

- **All three platform branches run on all three CI runners**, instead of two
  being dead code on each.
- **The wmic-absent path can be forced**, which is impossible on a machine that
  still has wmic — the exact situation in which the bug was missed.
- **The suite gives the same answer on every machine**, so a failure is a code
  change and never a hardware difference.

The fakes found seven more instances of the same blind spot, including
``lscpu``, which is absent from most minimal container images.

Test Infrastructure
-------------------

``tests/unit/updates/conftest.py``
   **``updates_config`` (autouse)** patches exactly five configuration keys,
   which is the module's entire surface: ``ompthreads``,
   ``materials["maxpoles"]``, ``materials["dispersivedtype"]``,
   ``general["precision"]`` and ``dtypes["complex"]``.

   **``make_wiring_grid``** returns a ``SimpleNamespace`` whose field
   attributes are *sentinel strings*, not arrays. The argument-order tests
   compare identity against those sentinels, so a swapped pair is caught by
   name rather than by a numerical coincidence.

   **``make_kernel_grid``** builds a small real ``FDTDGrid`` — 4×5×6, against
   the upstream sketch's 100³, which costs about 15 seconds for 25 tests.

   **``ramped_grid``** fills only the *source* field with a distinct
   per-component ramp. A uniformly filled grid has zero curl, so the kernels
   write nothing and every "which cells were updated" assertion passes
   vacuously. This fixture exists because the first version of those tests did
   exactly that.

``tests/unit/config/conftest.py``
   **The odd one out.** It builds the real classes rather than stand-ins, so it
   must not install the usual doubles.

   **``no_host_probes`` (autouse)** replaces ``get_host_info``,
   ``detect_cuda_gpus``, ``detect_opencl``, ``detect_metal`` and
   ``get_terminal_width`` **on** ``gprMax.config``, because that module imports
   them by name.

   **``restore_config_globals`` (autouse)** saves and restores
   ``config.sim_config``. Mutating it is precisely what these tests do, and
   every other directory depends on it.

   **``make_args``** starts from ``gprMax.args_defaults`` — the same dictionary
   the API and the CLI both fill in — so the defaults under test are the
   production defaults rather than a guess.

``tests/unit/utilities/conftest.py``
   **``fake_subprocess``** replaces ``subprocess.check_output`` with a lookup
   table keyed on argv, recording every command attempted. Anything not
   registered raises ``FileNotFoundError``, so an unanticipated command fails
   loudly rather than reaching the real shell. Several tests assert on the
   recorded calls rather than the return value — the point of the wmic fallback
   is *which command runs*.

   **``windows_host`` / ``macos_host`` / ``linux_host``** each patch
   ``sys.platform``, register that platform's commands with realistic output
   (header lines and ``\r\n`` included), and pin ``platform`` and ``psutil``.

   **``clean_omp_environment`` (autouse)** deletes six ``OMP_*`` variables
   before every test.

``tests/unit/vtkhdf/conftest.py``
   **``read_h5``** — the same helper PR 10's outputs suite uses, so the two
   sets of round-trip assertions are directly comparable.

   **``make_image_data``** defaults to an *anisotropic* ``(2, 3, 4)`` shape. A
   cubic default would hide every transposition bug this suite exists to catch.

   Every test writes into ``tmp_path``. The constructors open with mode ``"w"``,
   so a stray relative filename would truncate a real file in the working
   directory.

Closing the Upstream Sketch's Gaps
----------------------------------

``tests/updates/test_cpu_updates.py`` is an upstream sketch carrying six
``@pytest.mark.skip("test not implemented")`` stubs. **That file is left
exactly as it is** — the PR 9 precedent. The mapping below records which new
test now covers each stub, so the gap can be seen closed without this PR
touching someone else's file.

``test_update_electric_a_dispersive``
   ``test_cpu_updates.py::TestUpdateElectricA`` — the dispersive branch with
   its 17 positional arguments, plus the ``maxpoles`` switch that selects it.

``test_update_electric_b_dispersive``
   ``test_cpu_updates.py::TestUpdateElectricB`` — 13 arguments, no
   ``updatecoeffsE``, no H fields, and the no-op assertion for
   ``maxpoles == 0``.

``test_update_magnetic_sources``
   ``test_cpu_updates.py::TestSourceUpdateOrder`` and
   ``TestSourceUpdateArguments``.

``test_update_electric_sources``
   The same two classes — including the assertion that Hertzian dipoles are
   updated **last**, which the base docstring promises and only the
   concatenation order realises.

``test_dispersive_update_a`` / ``test_dispersive_update_b``
   ``test_dispersive_dispatch.py`` in its entirety: all sixteen names resolved
   against the real compiled extension, each of the four switches driven
   independently, and the binding onto the instance asserted.

The sketch also leaves two questions in comments, both answered by the Yee
stagger:

*"Why does fields_updates_normal use i+1, j+1 and k+1 everywhere?"*
   Because E and H sit half a cell apart. The curl of H at an E node needs the
   two H components straddling it, which in an array indexed from the same
   origin are at ``i`` and ``i+1``. The ``+1`` is the half-cell offset, made
   whole by the array indexing.

*"Why is there not a full (11×11×11) section of the grid being updated?"*
   Because a stencil reaching to ``i+1`` cannot run at the last index, and the
   two loop bounds differ between E and H. On a 4×5×6 grid the magnetic kernel
   writes 120 cells of each H component and the electric kernel writes 80, 75
   and 72 of ``Ex``, ``Ey`` and ``Ez`` respectively — asymmetric, because each
   component's stencil steps different axes. Those regions are asserted cell by
   cell in ``TestRealKernels``; they were determined empirically rather than
   derived, and the exact slices are recorded in that class.

Test Catalog — ``test_updates_base.py``
---------------------------------------

**57 tests** from 24 test functions across 5 classes.

The ``Updates`` abstract base class — ``gprMax/updates/updates.py``.

``Updates`` is the contract every solver backend implements. It carries no
logic at all: eleven ``@abstractmethod`` declarations, two concrete no-ops,
and an ``__init__`` that stores the grid. There is nothing to compute, so
every test here is about the *shape* of the contract.

That is worth testing precisely because the class exists to make a mistake
impossible. If a new abstract method is added upstream and one backend does
not implement it, that backend stops being instantiable — which is the ABC
working. But if a backend is written that never inherits from ``Updates`` at
all, the ABC is silently bypassed, and ``Solver.__init__``'s ``updates:
Updates`` annotation becomes a lie. That has already happened once; see
``TestBackendConformance``.

The eleven names are asserted as a frozen set rather than individually. A
test per method would pass unchanged if a twelfth were added, which is
precisely the event worth catching.

TestAbstractContract
^^^^^^^^^^^^^^^^^^^^

The set of abstract methods, and what it means to satisfy it.

``test_updates_is_an_abstract_base_class``
   ``Updates`` derives from ``ABC``, so the machinery is active.

   Inheriting ``abstractmethod`` declarations without an ``ABCMeta``
   metaclass silently does nothing — the decorators become documentation and
   every incomplete subclass instantiates happily.

``test_abstract_method_set_is_exactly_the_eleven``
   The contract is these eleven names and no others.

   A twelfth appearing here means every backend — including the three GPU
   ones this suite cannot run — needs a new implementation.

``test_there_are_eleven_abstract_methods``
   Count stated separately, so a rename plus an addition is caught.

``test_updates_cannot_be_instantiated``
   The base class is not usable directly.

``test_a_complete_subclass_can_be_instantiated``
   Implementing the eleven is sufficient — there is no hidden step.

``test_a_subclass_missing_any_one_method_cannot_be_instantiated``
   Each of the eleven is individually load-bearing.

   Builds a genuine subclass of ``Updates`` implementing ten of the eleven —
   the ABC machinery computes ``__abstractmethods__`` itself — and asserts
   instantiation fails naming the one left out.

``test_every_abstract_method_is_marked_abstract``
   ``__isabstractmethod__`` is set on each declaration.

TestConcreteMethods
^^^^^^^^^^^^^^^^^^^

``finalise`` and ``cleanup`` — the two hooks with a default.

``test_hook_is_not_abstract``
   Backends inherit these rather than being forced to write them.

``test_hook_returns_none``
   Both are no-ops on the base class.

``test_hook_takes_no_arguments_beyond_self``
   ``Solver.solve`` calls both with no arguments.

``test_cpu_updates_reimplements_both_hooks_identically``
   ``CPUUpdates`` re-declares both as byte-identical no-ops.

   Pinned as an observation rather than a defect: the overrides are
   harmless, but they are dead code, and anyone adding behaviour to the
   base-class hooks would find it silently ignored on the CPU path.

TestConstruction
^^^^^^^^^^^^^^^^

``__init__`` stores the grid and does nothing else.

``test_init_stores_the_grid``
   ``self.grid`` is the object passed in, not a copy.

``test_init_accepts_any_object``
   No validation whatsoever.

   The ``GridType`` bound is a typing construct with no runtime effect, so a
   stand-in grid is accepted — which is what makes most of this suite
   possible.

``test_init_sets_only_the_grid_attribute``
   Nothing else is initialised.

   This is why ``CPUUpdates.calculate_solve_time`` raises before
   ``time_start`` has run, and why ``update_electric_a`` raises on a
   dispersive model before ``set_dispersive_updates`` has run. Both are
   recorded in ``notes/bugs/``.

``test_init_signature_is_a_single_positional_grid``
   Backends are constructed as ``Backend(grid)`` throughout.

TestGenericParameter
^^^^^^^^^^^^^^^^^^^^

``Updates`` is generic over the grid type.

``test_updates_is_generic``
   ``Generic[GridType]`` is in the MRO's parameter list.

``test_grid_type_var_is_bound_to_fdtd_grid``
   Backends are parameterised by a grid, not by anything else.

``test_subscripting_updates_is_accepted``
   ``Updates[FDTDGrid]`` is a valid base, as CPUUpdates uses it.

TestBackendConformance
^^^^^^^^^^^^^^^^^^^^^^

Which backends actually implement the contract.

The CPU and MPI backends do. The three accelerator backends are checked here
for *conformance only* — none of them is executed, so no hardware is
required.

``test_cpu_updates_is_an_updates_subclass``
   Undocumented.

``test_cpu_updates_implements_the_whole_contract``
   No abstract methods remain, so it is instantiable.

``test_cpu_updates_can_be_constructed``
   Undocumented.

``test_cpu_updates_defines_every_contract_method``
   Each of the eleven is a real implementation, not inherited.

``test_cpu_updates_adds_three_methods_outside_the_contract``
   ``Solver.solve`` needs ``isinstance`` guards because of these.

   ``update_plane_waves_electric``, ``update_plane_waves_magnetic`` and
   ``set_dispersive_updates`` exist only on ``CPUUpdates``, so the solver
   loop cannot call them through the base type and has to type-check first.

``test_metal_updates_does_not_implement_the_contract``
   ``MetalUpdates`` is not an ``Updates`` subclass at all.

   It is a plain class that assigns ``self.grid = G`` by hand. Because
   ``Solver.__init__`` is annotated ``updates: Updates`` and
   ``create_solver`` hands it a ``MetalUpdates``, the Metal path violates
   the solver's own type contract — and the ABC cannot catch it, because
   nothing ever inherited from the ABC.

   Written up in ``notes/bugs/metal-updates-not-an-updates-subclass.md``.
   The import is guarded because the module imports ``Metal`` lazily but may
   still fail to import on a non-Apple platform.

When these fail
~~~~~~~~~~~~~~~

**The abstract-method count is wrong.** ``ABSTRACT_METHODS`` is a frozen set
of eleven names, asserted against ``Updates.__abstractmethods__``. Adding a
method to the ABC without adding it here fails immediately — which is the
point: every backend must then implement it, and the conformance tests will
say which does not.

**``TypeError: Can't instantiate abstract class``.** A backend is missing a
method the ABC requires. That is the failure this file exists to produce,
and it is *good* — the alternative is an ``AttributeError`` partway through
a timestep loop.

**``test_metal_updates_does_not_implement_the_contract`` goes green.**
``MetalUpdates`` has been made an ``Updates`` subclass. Delete that test and
add ``metal`` to the parametrisation of the conformance test instead. See
``notes/bugs/metal-updates-not-an-updates-subclass.md``.

**A subclass test looks circular.** It is deliberately not. The missing-one-
method test builds a genuine ``Updates`` subclass implementing ten of the
eleven and lets ``ABCMeta`` compute ``__abstractmethods__``; an earlier
version forced that attribute directly, which assumed exactly what it was
testing.

Test Catalog — ``test_cpu_updates.py``
--------------------------------------

**91 tests** from 77 test functions across 12 classes.

``CPUUpdates`` — ``gprMax/updates/cpu_updates.py``.

The CPU backend is a wiring layer. Almost nothing here computes: each method
reads a handful of attributes off the grid and hands them to a compiled
kernel, a source object, or a PML slab. So the properties worth asserting
are *which* collaborator was called, *with what*, and *in what order*.

Three of those are easy to get wrong and impossible to notice:

**Argument order.** Every kernel call is entirely positional — twelve
arguments for ``update_magnetic``, seventeen for the dispersive electric
update. Transposing two same-typed arrays produces a running simulation with
wrong numbers. Each call is asserted against the full expected tuple.

**Source ordering.** The base class promises to "update any Hertzian dipole
sources last". Nothing enforces that; it is an emergent property of the list
concatenation ``voltagesources + transmissionlines + hertziandipoles``. One
assertion pins it.

**The dispersive branch.** ``update_electric_a`` dispatches on ``maxpoles ==
0`` and ``update_electric_b`` on ``maxpoles > 0``. The second has no
``else``, so for a non-dispersive model it is a silent no-op.

Most tests here use the recorder grid from ``conftest.py`` rather than a
real one: the questions are about wiring, and sentinel strings make an
identity assertion possible that a real array would not. ``TestRealKernels``
runs the genuine compiled kernels on a small grid, to confirm the wiring
actually fits what the kernels expect.

TestConstruction
^^^^^^^^^^^^^^^^

``CPUUpdates(grid)`` and the attributes it does not set.

``test_stores_the_grid``
   Undocumented.

``test_construction_reads_no_configuration``
   A ``CPUUpdates`` can be built before ``sim_config`` exists.

   ``__init__`` is a bare ``super().__init__(G)``, so nothing is read.
   Asserted by removing the global entirely.

``test_construction_sets_only_the_grid``
   Neither the timer nor the dispersive functions are initialised.

``test_calculate_solve_time_before_time_start_raises``
   ``self.timestart`` only exists after ``time_start()``.

   ``Solver.solve`` always calls ``time_start()`` first, so this is latent
   rather than live — but it is the reason the two-step protocol exists.
   Written up in ``notes/bugs/cpu-updates-uninitialised-attributes.md``.

TestStoreOutputs
^^^^^^^^^^^^^^^^

``store_outputs`` delegates to ``fields_outputs.store_outputs``.

``test_delegates_to_the_module_level_function``
   Patched on ``cpu_updates``, where the name was bound at import.

``test_passes_grid_then_iteration_in_that_order``
   ``store_outputs(G, iteration)`` — a swap here would be silent.

   Both arguments are positional and neither is type-checked, so the
   transposed call would run and write field values into the wrong time-
   series slot.

``test_passes_no_keyword_arguments``
   Undocumented.

TestStoreSnapshots
^^^^^^^^^^^^^^^^^^

``store_snapshots`` — the deliberate off-by-one.

A snapshot requested for iteration *n* is stored when the loop counter reads
``n - 1``, because ``store_snapshots`` runs at the *top* of the iteration,
before the fields advance. The gate is ``snap.time == iteration + 1``.

``test_stores_a_snapshot_whose_time_is_one_past_the_iteration``
   Undocumented.

``test_does_not_store_when_times_are_equal``
   Equality with the raw iteration is exactly the wrong condition.

``test_does_not_store_on_any_other_iteration``
   Undocumented.

``test_stores_only_the_matching_snapshot``
   Several snapshots, one due.

``test_stores_every_snapshot_sharing_a_time``
   Two snapshots at the same iteration both fire.

``test_store_is_called_with_no_arguments``
   ``snap.store()`` takes nothing — the snapshot holds its own view.

``test_no_snapshots_is_a_no_op``
   Undocumented.

``test_snapshots_are_visited_in_list_order``
   Undocumented.

TestUpdateMagnetic
^^^^^^^^^^^^^^^^^^

``update_magnetic`` — twelve positional arguments to one kernel.

``test_calls_the_magnetic_kernel_once``
   Undocumented.

``test_passes_exactly_twelve_positional_arguments``
   Undocumented.

``test_argument_order_is_the_kernel_signature``
   ``nx, ny, nz, nthreads, updatecoeffsH, ID, Ex..Ez, Hx..Hz``.

   The six field arrays are all the same dtype and shape in production, so a
   transposition is invisible to the kernel and produces a silently wrong
   simulation. Sentinel values make it visible here.

``test_uses_the_magnetic_coefficients_not_the_electric_ones``
   The one coefficient array is ``updatecoeffsH``.

``test_thread_count_is_read_from_config_at_call_time``
   Changing ``ompthreads`` between calls changes the argument.

TestUpdateElectricA
^^^^^^^^^^^^^^^^^^^

``update_electric_a`` — the branch on ``maxpoles``.

``test_non_dispersive_model_calls_the_plain_kernel``
   Undocumented.

``test_plain_kernel_receives_twelve_positional_arguments``
   Same shape as the magnetic call, with ``updatecoeffsE``.

``test_dispersive_model_calls_the_dispersive_function``
   ``maxpoles > 0`` routes to ``self.dispersive_update_a``.

``test_dispersive_call_receives_seventeen_positional_arguments``
   Five more than the plain kernel: ``maxpoles``, ``updatecoeffsdispersive``
   and the three ``T`` memory arrays.

``test_dispersive_model_does_not_call_the_plain_kernel``
   The branch is exclusive.

``test_any_positive_pole_count_takes_the_dispersive_branch``
   Undocumented.

``test_dispersive_functions_unset_raises_attribute_error``
   A dispersive model built without ``set_dispersive_updates``.

   ``create_solver`` always calls it, so production is safe — but any other
   construction path fails here rather than at configuration time. Written
   up in ``notes/bugs/cpu-updates-uninitialised-attributes.md``.

TestUpdateElectricB
^^^^^^^^^^^^^^^^^^^

``update_electric_b`` — the second half of the dispersive update.

``test_non_dispersive_model_is_a_silent_no_op``
   ``maxpoles == 0`` does nothing, with no ``else`` and no log line.

   The method returns ``None`` having touched nothing, which is
   indistinguishable from a step that failed to run.

``test_dispersive_model_calls_the_dispersive_function``
   Undocumented.

``test_dispersive_call_receives_thirteen_positional_arguments``
   The B half takes **no** ``updatecoeffsE`` and **no** H fields.

   It closes the memory-variable loop using the already-updated electric
   field, so the magnetic field and the standard coefficients are not
   needed. Four fewer arguments than the A half.

``test_b_half_does_not_receive_the_magnetic_field``
   Undocumented.

TestPmlUpdates
^^^^^^^^^^^^^^

``update_electric_pml`` / ``update_magnetic_pml``.

``test_electric_visits_every_slab``
   Undocumented.

``test_magnetic_visits_every_slab``
   Undocumented.

``test_slabs_are_visited_in_list_order``
   Order is the ``pmls["slabs"]`` list, unsorted.

``test_no_slabs_is_a_no_op``
   A model with ``#pml_cells: 0`` has an empty slab list.

``test_slab_update_takes_no_arguments``
   Each slab holds its own coefficients and grid reference.

``test_electric_and_magnetic_read_the_same_slab_list``
   Undocumented.

TestSourceUpdateOrder
^^^^^^^^^^^^^^^^^^^^^

The concatenation order, which is the only thing enforcing the documented
"Hertzian dipoles last" rule.

``test_electric_sources_run_voltage_then_line_then_dipole``
   ``voltagesources + transmissionlines + hertziandipoles``.

   The base class docstring promises Hertzian dipoles are updated last.
   Nothing checks that — it is purely the order these three lists are
   concatenated in. Reordering the expression would be a silent physics
   change.

``test_hertzian_dipoles_are_updated_last``
   Stated as its own assertion, because it is the documented rule.

``test_magnetic_sources_run_line_then_dipole``
   ``transmissionlines + magneticdipoles`` — only two lists.

``test_transmission_lines_are_updated_by_both_paths``
   A transmission line appears in the electric *and* magnetic lists.

``test_voltage_sources_are_not_updated_magnetically``
   Undocumented.

``test_magnetic_dipoles_are_not_updated_electrically``
   Undocumented.

``test_no_sources_is_a_no_op``
   Undocumented.

``test_sources_within_a_list_keep_their_order``
   Undocumented.

TestSourceUpdateArguments
^^^^^^^^^^^^^^^^^^^^^^^^^

What each source receives — seven positional arguments.

``test_electric_source_argument_tuple``
   ``iteration, updatecoeffsE, ID, Ex, Ey, Ez, G``.

``test_magnetic_source_argument_tuple``
   ``iteration, updatecoeffsH, ID, Hx, Hy, Hz, G``.

``test_electric_sources_receive_seven_arguments``
   Undocumented.

``test_magnetic_sources_receive_seven_arguments``
   Undocumented.

``test_the_grid_itself_is_the_final_argument``
   Sources reach everything else through the grid they are handed.

``test_iteration_is_passed_through_unchanged``
   No off-by-one here, unlike ``store_snapshots``.

TestPlaneWaves
^^^^^^^^^^^^^^

``update_plane_waves_electric`` / ``_magnetic``.

These two are not on the base class, which is why ``Solver.solve`` guards
them with ``isinstance(self.updates, CPUUpdates)``.

``test_electric_plain_branch_is_taken_for_a_non_dispersive_wave``
   Undocumented.

``test_electric_dispersive_branch_is_taken_when_flagged``
   Undocumented.

``test_dispersive_wave_receives_the_dispersive_coefficients``
   The dispersive variant takes ``updatecoeffsdispersive`` as a fourth
   positional argument; the plain one does not.

``test_plain_wave_does_not_receive_the_dispersive_coefficients``
   Undocumented.

``test_both_electric_branches_pass_the_same_two_keywords``
   ``cythonize=True, precompute=True`` — hard-coded either way.

``test_magnetic_path_has_no_dispersive_branch``
   Asymmetric with the electric path, deliberately or otherwise.

   A wave flagged dispersive still takes the single magnetic path — the
   magnetic update has no dispersive variant to dispatch to.

``test_magnetic_wave_does_not_receive_dispersive_coefficients``
   Undocumented.

``test_every_wave_in_the_list_is_updated``
   Undocumented.

``test_no_waves_is_a_no_op``
   Undocumented.

``test_mixed_dispersive_and_plain_waves_each_take_their_branch``
   Undocumented.

TestTiming
^^^^^^^^^^

``time_start`` / ``calculate_solve_time``.

``test_time_start_records_the_clock``
   Undocumented.

``test_calculate_solve_time_returns_the_elapsed_difference``
   Two readings of the same clock, subtracted.

``test_solve_time_is_not_cached``
   Each call re-reads the clock, so it grows between calls.

``test_time_start_can_be_called_again_to_restart``
   Undocumented.

TestRealKernels
^^^^^^^^^^^^^^^

The compiled kernels, driven through ``CPUUpdates`` on a small grid.

Everything above uses recorders, which proves the wiring is *consistent* but
not that it *fits*. These tests run the genuine Cython kernels, so a
mismatch in argument count, order or dtype surfaces as a real error.

``test_update_magnetic_runs_on_a_zeroed_grid``
   A grid with no fields stays at zero — the curl of nothing.

``test_update_electric_a_runs_on_a_zeroed_grid``
   Undocumented.

``test_magnetic_update_touches_only_the_magnetic_field``
   H is computed from E; E must come back untouched.

``test_electric_update_touches_only_the_magnetic_field_readonly``
   E is computed from H; H must come back untouched.

``test_magnetic_update_writes_the_yee_staggered_region``
   ``Hx`` is written at ``[1:, :-1, :-1]`` and nowhere else.

   The 3-D kernel runs one fused loop over *cells* and writes ``Hx[i+1, j,
   k]``, so the touched region starts at 1 along x and stops one short along
   y and z. This is the answer to the question the upstream sketch left in a
   comment: the ``+1`` is each component's own half-cell Yee offset,
   restored after fusing three loops into one.

``test_each_magnetic_component_has_its_own_offset``
   The ``+1`` lands on a different axis for each component.

   ``Hx`` on the x-face, ``Hy`` on the y-face, ``Hz`` on the z-face — one
   fused cell loop, three different offsets.

``test_electric_update_skips_the_transverse_boundary_layer``
   ``Ex`` is written at ``[:-1, 1:-1, 1:-1]``.

   Full extent along its own axis — there are exactly ``nx`` x-edges — but
   only the interior nodes across y and z, because an ``Ex`` on a transverse
   boundary has no ``Hz``/``Hy`` on both sides to difference. The outermost
   layer is the PML's job, which is why the PML update runs immediately
   afterwards.

   This is the second question the upstream sketch left in a comment.

``test_each_electric_component_is_trimmed_on_its_transverse_axes``
   ``Ex`` is full along x and trimmed at both ends of y and z.

   The pattern rotates with the component. Note it is *not* symmetric the
   way the magnetic one is: ``Ex`` gets a dedicated edge loop for the ``i ==
   0`` face, so it spans all ``nx`` x-edges, whereas ``Ey`` and ``Ez`` lose
   their first x index as well.

``test_the_three_magnetic_components_cover_equal_cell_counts``
   Every magnetic component is written exactly ``nx*ny*nz`` times.

   One fused loop over cells, three writes per pass — so the counts must
   agree, however the offsets are arranged.

``test_the_three_electric_components_cover_different_cell_counts``
   The electric side is asymmetric, and the numbers say so.

   On a 4x5x6 grid: ``Ex`` 4x4x5 = 80, ``Ey`` 3x5x5 = 75, ``Ez`` 3x4x6 = 72.
   The differences come from which faces get a dedicated edge loop after the
   main interior pass.

``test_a_dtype_mismatch_is_rejected_by_the_kernel``
   Single-precision arrays against a double-precision config.

   The fused kernel signature binds one of ``float``/``double``, so a grid
   built at the wrong precision fails at the boundary rather than computing
   quietly in the wrong type.

``test_repeated_updates_are_stable``
   Ten alternating half-steps on a zeroed grid stay at zero.

   A cheap guard against uninitialised memory in the kernels — the failure
   mode PR 10 found in ``pml_build.pyx``.

When these fail
~~~~~~~~~~~~~~~

**A patched kernel is not called and the real one runs.** ``cpu_updates.py``
binds ``update_electric``, ``update_magnetic``, ``store_outputs`` and
``timer`` as module globals at import. Patch
``gprMax.updates.cpu_updates.<name>``, never the Cython module. This is the
most common way to write a test here that passes for the wrong reason.

**An argument-position test fails.** Every kernel call is all-positional: 12
arguments for ``update_magnetic``, 12 for the non-dispersive
``update_electric_a``, 17 for the dispersive one, 13 for
``update_electric_b``. The wiring grid supplies *sentinel strings* rather
than arrays precisely so a swapped pair is caught by identity rather than by
a numerical coincidence.

**A source-ordering test fails.** The base docstring promises Hertzian
dipoles are updated **last**, and that is realised only by the concatenation
``voltagesources + transmissionlines + hertziandipoles``. One assertion pins
a contract that is otherwise pure convention.

**A ``TestRealKernels`` region assertion fails.** These tests drive the real
compiled kernels on a 4×5×6 grid and assert which cells were written. Two
traps produced false passes in the first version: filling the grid uniformly
gives zero curl, so nothing is written and every assertion passes vacuously;
and indexing a boolean mask with an *empty* index set assigns the whole
array. The ``ramped_grid`` fixture and the plain ``arr != 0`` masks are the
fixes. The regions were determined empirically, not derived — H writes 120
cells per component, E writes 80, 75 and 72.

**A timing test fails on the clock.** ``timer`` is patched at
``gprMax.updates.cpu_updates.timer``; the tests assert on the *difference*
between two patched values, never on elapsed real time.

Test Catalog — ``test_dispersive_dispatch.py``
----------------------------------------------

**76 tests** from 25 test functions across 6 classes.

``set_dispersive_updates`` — ``gprMax/updates/cpu_updates.py:239``.

Twenty lines that build the *name* of a compiled kernel out of four switches
and then fetch it by string:

.. code-block:: python

   poles      = "multi" if maxpoles > 1 else "1"
   precision  = "float" if precision == "single" else "double"
   dispersion = "complex" if dispersivedtype == dtypes["complex"] else "real"

   update_f = "update_electric_dispersive_{}pole_{}_{}_{}"
   disp_a = update_f.format(poles, "A", precision, dispersion)
   disp_a_f = getattr(import_module("gprMax.cython.fields_updates_dispersive"), disp_a)

Four switches with two values each, times the A/B half, is **sixteen
possible names** — and every one must exist in a Cython module that is
*generated at build time* from ``fields_updates_dispersive_template.jinja``.

Nothing in the codebase checks that. If the template's naming scheme ever
drifts from this format string, the failure is an ``AttributeError`` at run
time, for the one combination of user settings nobody tried.
``TestEveryNameResolves`` closes that gap: it drives all sixteen against the
**real compiled module**, so a mismatch fails here rather than in a user's
simulation.

The remaining classes pin the switch logic itself, including the two silent
fallbacks — an unrecognised precision string yields ``"double"``, and a
``dispersivedtype`` that does not match yields ``"real"``.

TestEveryNameResolves
^^^^^^^^^^^^^^^^^^^^^

All sixteen constructible names exist in the compiled extension.

This is the test that justifies the file. The kernels are generated from a
jinja template at build time and imported by string at run time, with no
static link between the two. These sixteen cases are the entire space the
dispatcher can produce.

``test_there_are_exactly_sixteen_constructible_names``
   Four binary switches — no more, no fewer.

``test_name_exists_in_the_compiled_module``
   Every one is a real attribute of the built extension.

``test_name_resolves_to_something_callable``
   Undocumented.

``test_dispatcher_binds_both_halves_for_every_combination``
   The eight reachable configurations, driven end to end.

   Each binds an A and a B function, so this covers all sixteen names
   through the real code path rather than by string construction.

``test_bound_functions_are_the_ones_named_by_the_switches``
   The bound object is the module attribute the name predicts.

   Confirms the dispatcher is not merely finding *a* function, but the
   specific one implied by the configuration.

TestPoleSwitch
^^^^^^^^^^^^^^

``maxpoles > 1`` selects ``multipole``, otherwise ``1pole``.

``test_one_pole_selects_the_single_pole_kernel``
   Undocumented.

``test_more_than_one_pole_selects_the_multipole_kernel``
   Undocumented.

``test_zero_poles_still_binds_the_single_pole_kernel``
   ``maxpoles == 0`` is not rejected — it selects ``1pole``.

   ``create_solver`` guards the call with ``maxpoles != 0``, so this never
   happens in production. Called directly it succeeds silently, binding a
   kernel that ``update_electric_a`` will never reach because that method
   sends ``maxpoles == 0`` to the plain update instead.

``test_the_boundary_is_between_one_and_two``
   ``> 1``, so one pole is single and two are multi.

TestPrecisionSwitch
^^^^^^^^^^^^^^^^^^^

``precision == "single"`` selects ``float``; everything else selects
``double``.

``test_single_precision_selects_the_float_kernel``
   Undocumented.

``test_double_precision_selects_the_double_kernel``
   Undocumented.

``test_an_unrecognised_precision_silently_selects_double``
   The ternary tests equality with ``"single"`` and falls through.

   Any other string — including a capitalisation slip or the plausible
   ``"float"`` — silently produces the double-precision kernel. Since the
   grid's arrays were allocated from the same ``precision`` value elsewhere,
   this surfaces as a Cython buffer dtype mismatch rather than a message
   about the setting being wrong.

   Recorded in ``notes/bugs/config-precision-no-terminal-else.md``, which
   covers the same string's other silent failure.

TestDispersionSwitch
^^^^^^^^^^^^^^^^^^^^

``dispersivedtype == dtypes["complex"]`` selects ``complex``.

``test_matching_complex_dtype_selects_the_complex_kernel``
   Undocumented.

``test_a_real_dtype_selects_the_real_kernel``
   Undocumented.

``test_the_comparison_is_against_the_configured_complex_dtype``
   Not against ``np.complexfloating`` in general.

   A single-precision run compares against ``np.complex64``; handing it
   ``np.complex128`` — a complex type, but the wrong one — takes the *real*
   branch.

``test_unset_dispersive_dtype_silently_selects_real``
   ``dispersivedtype`` defaults to ``None`` until it is derived.

   ``ModelConfig`` initialises the key to ``None``, and only
   ``set_dispersive_material_types()`` fills it in. ``None == np.complex64``
   is ``False``, so calling ``set_dispersive_updates`` first binds the
   **real** kernel for what may be a complex-pole model — a wrong answer
   with no warning, produced purely by call order.

   Written up in ``notes/bugs/dispersive-dtype-default-none.md``.

TestBinding
^^^^^^^^^^^

How the resolved functions are attached to the instance.

``test_both_halves_are_set_together``
   Neither exists before the call; both exist after.

``test_the_two_halves_are_different_functions``
   Undocumented.

``test_a_and_b_differ_only_in_the_half_marker``
   The other three switches must agree between the halves.

``test_binding_is_per_instance_not_per_class``
   Two updaters can hold different kernels simultaneously.

   Subgrid runs do exactly this: ``create_solver`` configures the parent and
   every ``SubgridUpdater`` separately.

``test_calling_twice_rebinds_to_the_new_configuration``
   The method is idempotent in effect but not cached.

``test_bound_functions_are_plain_functions_not_methods``
   Assigned to the *instance*, so no ``self`` is passed.

   This is why ``update_electric_a`` calls ``self.dispersive_update_a(nx,
   ...)`` with the grid dimensions first and no leading ``self`` — an
   instance attribute holding a function is not a bound method.

TestModulePath
^^^^^^^^^^^^^^

The module the kernels are fetched from.

``test_the_module_path_is_hard_coded``
   Unlike the PML dispatcher, the module name is not formatted.

   Only the *function* name varies; the module is always
   ``gprMax.cython.fields_updates_dispersive``.

``test_the_module_is_imported_once_per_half``
   Two calls, one per half — the result is not reused.

   Harmless because ``import_module`` hits ``sys.modules``, but worth
   pinning so a future refactor does not assume a single import.

``test_a_missing_kernel_raises_attribute_error``
   There is no ``try``/``except`` around the ``getattr``.

   If the generated module ever lost a variant, the user would see a bare
   ``AttributeError`` naming a function they have never heard of, rather
   than a message about their material settings. Simulated here by importing
   a module that has none of the sixteen names.

When these fail
~~~~~~~~~~~~~~~

**One of the sixteen names does not resolve.** This is the failure the file
exists to catch. The kernels are generated from a jinja template at build
time and the dispatcher constructs their names by string formatting, with no
static link between the two. A failure means either the template changed or
the dispatcher's format string did — and only users with that particular
combination of pole count, precision and dispersion type would ever have
noticed.

**All sixteen fail at once.** The Cython extensions are stale or were built
without ``fields_updates_dispersive``. Rebuild with ``pip install -e .``.

**A switch test fails.** Each of the four binary choices is driven
independently, so the failing test names the switch: ``maxpoles > 1``
selects ``multi``, ``precision`` selects ``float``/``double``, and the
dispersive dtype compared against ``dtypes["complex"]`` selects
``complex``/``real``.

**The dispersion switch silently selects ``real``.** Check
``materials["dispersivedtype"]``. It defaults to ``None``, and ``None ==
np.complex64`` is ``False`` rather than an error — see
``notes/bugs/dispersive-dtype-default-none.md``. There is deliberately no
test asserting that fall-through.

Test Catalog — ``test_solver.py``
---------------------------------

**45 tests** from 38 test functions across 6 classes.

``Solver`` and ``create_solver`` — ``gprMax/solvers.py``.

This file holds the **running order of an FDTD timestep**, and it holds it
nowhere else. There is no "advance one step" method on ``CPUUpdates``; the
sequence exists only as eleven consecutive lines inside ``Solver.solve``.

That matters because the Yee scheme leapfrogs. Electric and magnetic fields
are staggered half a timestep apart and each is computed from the curl of
the other, so the order is the algorithm. Swap two calls and the simulation
still runs, still terminates, still writes a well-formed output file — with
wrong numbers. It is the archetype of the failure this project exists to
catch, and until now nothing asserted it.

Testing it needs neither a grid nor a kernel. ``Solver`` only ever calls
methods on the object it was handed, so a recorder that appends its own name
is a complete stand-in, and one iteration of the loop yields the sequence as
a plain list.

``create_solver`` is the other half: a six-way dispatch that decides which
backend a model gets. It uses ``type(grid) is FDTDGrid`` — **exact type
identity, not** ``isinstance`` — so a subclass falls through to a bare
``raise ValueError``.

TestSolverConstruction
^^^^^^^^^^^^^^^^^^^^^^

``Solver(updates)`` — three attributes, no logic.

``test_stores_the_updates_object``
   Undocumented.

``test_solve_time_starts_at_zero``
   Undocumented.

``test_memory_used_starts_at_zero``
   Undocumented.

``test_construction_sets_exactly_three_attributes``
   Undocumented.

``test_construction_does_not_call_the_updates_object``
   Nothing runs until ``solve``.

TestIterationOrder
^^^^^^^^^^^^^^^^^^

The eleven beats, in order. The heart of this file.

``test_one_iteration_produces_the_canonical_sequence``
   The full per-iteration order for a CPU run.

   If this test fails after a source change, the physics changed. There is
   no such thing as an incidental reordering here.

``test_magnetic_field_is_updated_before_the_electric_field``
   The leapfrog: H advances half a step, then E uses the new H.

``test_outputs_are_stored_before_anything_moves``
   Receivers record the state at the *top* of the iteration.

``test_snapshots_are_stored_before_anything_moves``
   Undocumented.

``test_outputs_are_stored_before_snapshots``
   Undocumented.

``test_each_half_step_runs_field_then_pml_then_sources``
   The PML correction follows the bulk update, then sources inject.

   The PML has to see the field the bulk kernel just produced, and a source
   must be added after both — otherwise the absorbing layer eats the
   excitation on the step it is applied.

``test_electric_b_is_the_last_step_of_the_iteration``
   The dispersive closing half runs after everything else.

   It needs both the old and the new electric field, so it cannot run until
   the PML and the sources have finished with E.

``test_electric_b_follows_the_electric_sources``
   Undocumented.

``test_plane_waves_follow_the_discrete_sources_in_each_half``
   Undocumented.

``test_there_are_eleven_steps_in_an_iteration``
   Undocumented.

TestLoopBracketing
^^^^^^^^^^^^^^^^^^

What happens once, outside the loop.

``test_time_start_runs_before_the_first_iteration``
   Undocumented.

``test_finalise_then_solve_time_then_cleanup``
   The exact teardown order.

   ``calculate_solve_time`` sits between the two hooks, so ``finalise`` can
   flush work that should be counted and ``cleanup`` can release resources
   that should not.

``test_time_start_is_called_exactly_once``
   Undocumented.

``test_teardown_step_is_called_exactly_once``
   Undocumented.

``test_solve_time_is_stored_on_the_solver``
   The return value of ``calculate_solve_time`` lands in ``solvetime``.

``test_solve_time_replaces_the_initial_zero``
   Undocumented.

TestIterationCount
^^^^^^^^^^^^^^^^^^

The loop runs once per item the iterator yields.

``test_body_repeats_once_per_iteration``
   Undocumented.

``test_an_empty_iterator_still_brackets_the_run``
   Zero iterations: the timer and the hooks still fire.

   A ``#time_window`` of zero produces this, and it must not crash.

``test_the_iterator_may_be_any_iterable``
   ``solve`` takes ``range()`` or ``tqdm()``; it only iterates.

   The loop variable is passed straight through to ``store_outputs`` and the
   source updates, so a non-range iterable works identically.

``test_iteration_values_are_passed_to_the_steps``
   Whatever the iterator yields reaches the iteration-taking steps.

TestBackendSpecificSteps
^^^^^^^^^^^^^^^^^^^^^^^^

Which steps a backend gets depends on its type.

``Solver.solve`` guards four of its calls with ``isinstance``, because the
methods are not on the ``Updates`` base class.

``test_a_non_cpu_backend_skips_the_plane_wave_steps``
   The two plane-wave calls are ``CPUUpdates``-only.

``test_a_non_cpu_backend_still_runs_the_nine_shared_steps``
   Everything on the base class still happens, in the same order.

``test_a_non_cpu_backend_is_still_bracketed``
   Undocumented.

``test_a_backend_inheriting_the_default_hooks_still_solves``
   ``finalise`` and ``cleanup`` are optional for a backend.

   ``Solver.solve`` calls both unconditionally, so the base class's no-op
   defaults are what make them optional. A backend that overrides neither
   runs to completion.

``test_the_cpu_guard_matches_subclasses``
   ``isinstance`` here, not exact type — unlike ``create_solver``.

   ``RecordingUpdates`` subclasses ``CPUUpdates`` and does get the plane-
   wave steps, which is what makes it a faithful stand-in.

TestCreateSolver
^^^^^^^^^^^^^^^^

``create_solver(model)`` — the backend dispatch.

``test_a_plain_grid_gets_cpu_updates``
   Undocumented.

``test_the_solver_wraps_the_updates_object``
   Undocumented.

``test_the_updates_object_holds_the_models_grid``
   Undocumented.

``test_a_non_dispersive_model_skips_dispersive_setup``
   ``maxpoles == 0`` leaves the dispersive functions unbound.

``test_a_dispersive_model_gets_dispersive_setup``
   ``maxpoles != 0`` triggers ``set_dispersive_updates()``.

   This is the only place in production that call is made, which is why
   constructing ``CPUUpdates`` any other way leaves those attributes
   missing.

``test_an_unknown_grid_type_raises_value_error``
   The terminal ``else`` logs and raises — with no message.

   ``raise ValueError`` bare, so the reason exists only in the log. The same
   pattern as ``check_kappamin`` in the PML, recorded in PR 10.

``test_a_subclass_of_fdtd_grid_is_rejected``
   ``type(grid) is FDTDGrid``, so inheritance does not qualify.

   Every branch of the dispatch uses exact type identity. A user or a test
   that subclasses ``FDTDGrid`` to add behaviour gets a bare ``ValueError``
   rather than the CPU backend, which is surprising — subclassing is
   otherwise how this codebase extends grids (``SubGridBaseGrid`` does
   exactly that).

   Written up in ``notes/bugs/create-solver-exact-type-dispatch.md``.

``test_the_error_is_logged_before_raising``
   The message the bare ``ValueError`` omits.

When these fail
~~~~~~~~~~~~~~~

**An ordering assertion fails.** ``CPU_ITERATION_ORDER`` is the eleven-call
sequence, in order, and it exists nowhere in the source except inside
``Solver.solve``'s loop body — ``CPUUpdates`` has no "run one timestep"
method. A reordering is silent wrong physics: E and H would be read at the
wrong half-step.

**The bracketing test fails.** ``time_start()`` before the loop;
``finalise() → calculate_solve_time() → cleanup()`` after it, in that order.
``calculate_solve_time`` reads ``self.timestart``, which only ``time_start``
sets, so the two are a pair — see ``notes/bugs/cpu-updates-uninitialised-
attributes.md``.

**A backend-specific step appears or disappears.** Four steps are guarded by
``isinstance`` checks inside the loop: the two plane-wave calls (CPU only),
the two halo swaps (MPI), the two HSG calls (subgrids), and the memory tally
(CUDA). The recorder subclasses are chosen to exercise each guard.

**``create_solver`` rejects a grid you expected it to accept.** The dispatch
is ``type(grid) is FDTDGrid`` — exact identity, not ``isinstance``. That is
deliberate and necessary, since every accelerator grid derives from
``FDTDGrid``, but it means a subclass falls through to "Unknown grid type".
See ``notes/bugs/create-solver-exact-type-dispatch.md``.

**A recorder's ``finalise`` or ``cleanup`` is not recorded.** Those two are
*concrete* no-ops on the ABC, so a recorder that does not override them
records nothing. Deleting them from a subclass requires assigning the
parent's version onto a fresh type — ``del`` raises, because the attribute
lives on the parent.

Test Catalog — ``test_simulation_config.py``
--------------------------------------------

**78 tests** from 59 test functions across 9 classes.

``SimulationConfig`` — ``gprMax/config.py:196``.

The object built once per run, from the argument namespace the CLI or the
API supplies, and then read by every module in the package. It is almost
entirely derivation: forty-odd attributes computed from twenty arguments,
plus five validation checks.

Two things make it worth testing carefully.

**``em_consts`` is a class attribute**, not an instance one, so every
``SimulationConfig`` in the process shares one dictionary. Mutating a key
through one instance changes it for all of them, including for tests that
ran earlier.

**The validation is uneven.** Four of the five checks raise a *bare*
``ValueError`` whose reason exists only in the log, and the fifth — the one
meant to stop two accelerators being selected at once — never fires at all,
because it tests ``count(True)`` against values the CLI supplies as lists.

Precision and dtype selection is large enough to have its own file; see
``test_precision_dtypes.py``.

TestDefaults
^^^^^^^^^^^^

A default run: no accelerator, no MPI, one model.

``test_constructs_from_the_shipped_argument_defaults``
   ``args_defaults`` plus an input file is a valid configuration.

``test_solver_defaults_to_cpu``
   Undocumented.

``test_precision_defaults_to_single``
   Note this differs from what the other test suites stub in.

   Every existing conftest supplies ``precision: "double"`` for its stand-in
   config; the real default is ``"single"``. Harmless, but it means those
   suites test the double-precision code path exclusively.

``test_general_has_exactly_four_keys``
   ``solver``, ``precision``, ``progressbars`` and ``subgrid``.

   The first three are set together; ``subgrid`` is added later in
   ``__init__``, which is why a stand-in that omits it still works for most
   consumers.

``test_subgrid_defaults_to_false``
   Undocumented.

``test_current_model_starts_at_zero``
   Undocumented.

``test_model_configs_is_sized_by_the_model_count``
   One slot per model, all empty until a model is built.

``test_arguments_are_kept_for_later_lookup``
   Several consumers read ``sim_config.args`` directly.

``test_argument_is_copied_onto_the_config``
   Each of these is a straight copy from the namespace.

``test_autotranslate_defaults_to_the_argument_value``
   Undocumented.

TestHostInfo
^^^^^^^^^^^^

The host probe runs unconditionally at construction.

``test_host_info_is_stored``
   Undocumented.

``test_host_info_is_probed_exactly_once``
   One probe per ``SimulationConfig``, not per read.

``test_host_info_is_probed_even_on_a_cpu_run``
   There is no way to opt out.

   A plain CPU run still shells out to ``wmic``/``sysctl``/``lscpu``, which
   is why the wmic removal broke gprMax at startup for every user rather
   than only for GPU users.

TestElectromagneticConstants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``em_consts`` — four values, shared by every instance.

``test_has_exactly_four_keys``
   Undocumented.

``test_impedance_of_free_space_is_derived_from_the_others``
   ``z0 = sqrt(m0 / e0)`` — about 376.73 ohms.

``test_speed_of_light_is_the_scipy_value``
   Undocumented.

``test_permittivity_is_the_scipy_value``
   Undocumented.

``test_permeability_is_the_scipy_value``
   Undocumented.

``test_em_consts_is_a_class_attribute_shared_by_all_instances``
   Every instance sees the same dictionary object.

   A test — or any caller — that mutated a key through one instance would
   change it for every other one, including instances created earlier.
   Nothing in the package does, but nothing prevents it either.

TestValidation
^^^^^^^^^^^^^^

The five guards in ``__init__``, and the one that does not work.

``test_taskfarm_with_fixed_geometry_is_rejected``
   Undocumented.

``test_taskfarm_alone_is_accepted``
   Undocumented.

``test_fixed_geometry_alone_is_accepted``
   Undocumented.

``test_showing_and_hiding_progress_bars_is_rejected``
   Undocumented.

``test_mpi_with_subgrids_is_rejected``
   Undocumented.

``test_subgrid_with_an_accelerator_is_rejected``
   Sub-gridding needs double precision, which the GPU paths force to single
   — so the combination is refused rather than silently downgraded.

``test_rejections_raise_a_bare_value_error``
   No message on the exception — only in the log.

   A user running through the API sees ``ValueError`` with an empty string.
   The same pattern as ``check_kappamin`` in the PML, recorded during PR 10.

``test_the_reason_is_logged``
   Undocumented.

``test_combined_accelerators_are_not_actually_rejected``
   The guard against selecting two accelerators never fires.

   ``[args.gpu, args.opencl, args.metal].count(True) > 1`` counts values
   equal to ``True``. The CLI parses ``-gpu`` with ``action="append",
   nargs="*"``, producing a *list*, and the API passes lists too — so
   ``count(True)`` is zero and the check passes.

   The three device branches that follow are independent ``if`` statements
   rather than a chain, so the last one wins: asking for both CUDA and
   OpenCL silently yields ``solver == "opencl"`` with a device dictionary
   built for it. No error, no warning.

   Written up in ``notes/bugs/config-combined-accelerator-guard.md``.

``test_the_guard_does_fire_for_literal_booleans``
   It works only for a caller that passes ``True`` itself.

   Which nothing in gprMax does — establishing that the check is not broken
   so much as guarding a shape the codebase never produces.

TestProgressBars
^^^^^^^^^^^^^^^^

``progressbars`` is derived from three arguments and the log level.

``test_on_by_default_at_the_default_log_level``
   ``log_level`` defaults to 20 (INFO), so bars are shown.

``test_off_when_the_log_level_is_above_info``
   Above INFO the bars would interleave with sparse output.

``test_on_when_explicitly_shown_even_at_a_high_log_level``
   ``show_progress_bars`` wins over the log-level heuristic.

``test_off_when_explicitly_hidden``
   Undocumented.

``test_on_at_or_below_info``
   Undocumented.

``test_off_above_info``
   25 is the custom ``BASIC`` level, which also suppresses bars.

TestSolverSelection
^^^^^^^^^^^^^^^^^^^

Which backend the arguments select.

``test_no_accelerator_argument_gives_cpu``
   Undocumented.

``test_gpu_argument_selects_cuda``
   Undocumented.

``test_opencl_argument_selects_opencl``
   Undocumented.

``test_metal_argument_selects_metal``
   Undocumented.

``test_every_accelerator_forces_single_precision``
   Both precisions work on a GPU; single is chosen for speed.

``test_subgrids_force_double_precision``
   The Huygens sub-grid coupling is too ill-conditioned for float32.

``test_an_empty_device_list_still_selects_the_accelerator``
   The branch tests ``is not None``, not truthiness.

   ``-gpu`` with no device ID parses to ``[]``, which still means "use CUDA"
   and defaults to device 0 later.

``test_accelerator_runs_get_a_devices_dictionary``
   Undocumented.

``test_a_cpu_run_has_no_devices_attribute``
   ``devices`` is only created on an accelerator path.

``test_cuda_devices_carry_compiler_options``
   Undocumented.

``test_non_cuda_devices_carry_compiler_options``
   Undocumented.

TestInputFilePath
^^^^^^^^^^^^^^^^^

``_set_input_file_path`` — where the model is read from.

``test_input_file_becomes_a_path``
   Undocumented.

``test_the_output_file_is_used_when_no_input_file_is_given``
   The API can supply a scene and an output name with no input file.

``test_neither_path_given_raises``
   ``Path(None)`` — a reachable combination of the shipped defaults.

   ``args_defaults`` has both ``inputfile`` and ``outputfile`` set to
   ``None``, so constructing straight from the defaults fails with a
   ``TypeError`` about ``NoneType`` rather than a message naming the missing
   argument.

TestModelStartAndEnd
^^^^^^^^^^^^^^^^^^^^

``_set_model_start_end`` — which model numbers this run covers.

``test_a_single_model_run_spans_zero_to_one``
   Undocumented.

``test_a_multi_model_run_spans_zero_to_n``
   Undocumented.

``test_a_restart_index_shifts_the_range``
   ``-i 3 -n 2`` resumes at model 3 and runs two models.

``test_a_restart_index_of_zero_is_treated_as_absent``
   The branch is a truthiness test, so ``0`` takes the else path.

   Model numbers are 1-based on the command line, so ``-i 0`` is not
   meaningful — but it is accepted and silently behaves like no ``-i``.

``test_the_model_config_list_is_not_resized_for_a_restart``
   ``model_configs`` is sized ``n`` but indices run to ``(i-1)+n``.

   With ``-i 5 -n 3`` the run iterates models 4, 5 and 6 while the list
   holds three slots, so storing the first config raises ``IndexError``.
   Asserted here as the arithmetic mismatch rather than by driving the
   failure, since that needs the whole context loop.

   Written up in ``notes/bugs/config-model-index-range-mismatch.md``.

TestSceneStorage
^^^^^^^^^^^^^^^^

``scenes`` — one per model, supplied by the API or left empty.

``test_scenes_default_to_one_empty_slot_per_model``
   Undocumented.

``test_supplied_scenes_are_kept``
   Undocumented.

``test_a_scene_can_be_retrieved_by_model_number``
   Undocumented.

``test_a_scene_can_be_stored_by_model_number``
   Undocumented.

``test_storing_a_scene_defaults_to_the_current_model``
   Undocumented.

When these fail
~~~~~~~~~~~~~~~

**Every test in the directory fails on a host probe.** The autouse
``no_host_probes`` fixture patches five names **on** ``gprMax.config``,
because that module imports them by name. Patching
``gprMax.utilities.host_info.get_host_info`` has no effect whatsoever.

**A test in an unrelated directory fails afterwards.**
``restore_config_globals`` is what prevents that. This is the only suite
that writes to ``gprMax.config.sim_config``, the global every other
directory monkeypatches.

**A default disagrees with the CLI.** ``make_args`` starts from
``gprMax.args_defaults``, so these tests check the production defaults
rather than a copy of them. A CLI change that alters a default will fail
here, which is intended.

**A validation test does not raise.** Four guards raise ``ValueError``; a
fifth — the combined-accelerator check — **never fires**, because it counts
``list.count(True)`` on arguments that are device-ID lists.
``test_combined_accelerators_are_not_actually_rejected`` pins that. See
``notes/bugs/config-combined-accelerator-guard.md``.

**An ``em_consts`` test fails after another test ran.** It is a **class**
attribute, shared by every instance. Mutating it in one test would change
every other; ``test_em_consts_is_a_class_attribute_shared_by_all_instances``
states that explicitly so nobody writes to it by accident.

Test Catalog — ``test_precision_dtypes.py``
-------------------------------------------

**54 tests** from 25 test functions across 7 classes.

``SimulationConfig._set_precision`` — ``gprMax/config.py:371``.

One string — ``"single"`` or ``"double"`` — chooses six different data
types, and through them the memory footprint of every field array, the
Cython fused type each kernel binds, and the C type name pasted into every
generated GPU kernel.

It also, indirectly, chooses the *name* of the compiled dispersive kernel
that will run: ``CPUUpdates.set_dispersive_updates`` reads the same
``general["precision"]`` string to decide between ``_float_`` and
``_double_``. Those two facts live in different files with nothing tying
them together, so ``TestConsistencyWithTheKernelDispatch`` asserts them side
by side.

The whole thing is one ``if``/``elif`` with **no terminal** ``else``, which
is the most consequential instance of that pattern found in this PR — see
``TestUnknownPrecision``.

TestDtypeKeys
^^^^^^^^^^^^^

The shape of the ``dtypes`` dictionary.

``test_has_exactly_six_keys``
   Six, at both precisions.

   Note the stand-in configs in every other test directory supply only two
   of these (``float_or_double`` and ``complex``). Anything reading a
   ``cython_*`` or ``C_*`` key would fail against those stubs — which is
   precisely the drift a config-level suite exists to pin.

``test_dtypes_is_created_during_construction``
   ``_set_precision`` runs unconditionally in ``__init__``.

``test_each_instance_gets_its_own_dictionary``
   Unlike ``em_consts``, ``dtypes`` is per instance.

TestSinglePrecision
^^^^^^^^^^^^^^^^^^^

The default: 32-bit fields.

``test_dtype_entry``
   Undocumented.

``test_field_arrays_would_be_float32``
   ``float_or_double`` is what every array allocation uses.

``test_complex_type_matches_the_real_one_in_width``
   A complex value is two floats of the chosen width.

``test_cython_type_is_the_float_shadow``
   Identity, not equality — the shadow types are singletons.

   ``cython.float`` and ``cython.double`` are distinct objects in the pure-
   Python shadow module, so an identity check is meaningful.

TestDoublePrecision
^^^^^^^^^^^^^^^^^^^

64-bit fields, reached by asking for sub-grids.

``test_dtype_entry``
   Undocumented.

``test_field_arrays_would_be_float64``
   Undocumented.

``test_cython_type_is_the_double_shadow``
   Undocumented.

``test_the_two_cython_shadows_are_distinct``
   Guards the identity assertions above from being vacuous.

TestPrecisionsDiffer
^^^^^^^^^^^^^^^^^^^^

Every entry actually changes between the two precisions.

``test_entry_differs_between_precisions``
   No key is accidentally shared, which would make it unswitchable.

``test_double_precision_doubles_the_field_footprint``
   Undocumented.

TestCComplexPerSolver
^^^^^^^^^^^^^^^^^^^^^

``C_complex`` — the one entry that also depends on the backend.

Each GPU toolchain spells a complex number differently, and the string here
is pasted verbatim into generated kernel source. A wrong value is a compile
error at run time, on hardware CI does not have.

``test_c_complex_matrix``
   All eight combinations of precision and backend.

   Half of these cannot be reached through the argument surface — every
   accelerator forces single precision — so the two settings are driven
   directly and ``_set_precision`` re-run. That exercises the real mapping
   rather than skipping the rows, and it is also how a caller using the
   Python API could reach them.

``test_real_dtypes_are_unaffected_by_the_solver``
   Only ``C_complex`` varies with the backend; the other five do not.

``test_cpu_leaves_c_complex_unset``
   Nothing generates C source on the CPU path, so there is no name.

``test_cuda_uses_the_pycuda_complex_template``
   Undocumented.

``test_opencl_uses_the_short_form``
   Undocumented.

``test_metal_uses_the_namespaced_form``
   Undocumented.

``test_the_real_c_type_is_independent_of_the_solver``
   Only the *complex* name varies; ``float`` is ``float`` everywhere.

TestUnknownPrecision
^^^^^^^^^^^^^^^^^^^^

The missing terminal ``else`` — the most consequential one found.

``_set_precision`` is ``if precision == "single" ... elif precision ==
"double"`` with nothing after it. Any other value leaves ``self.dtypes``
**never assigned at all**, and because the method is called at the very end
of ``__init__`` the object is returned looking complete.

The first symptom is an ``AttributeError`` about ``dtypes`` raised from
whichever module happens to read it first — typically an array allocation in
``FDTDGrid``, far from the setting that caused it.

No test asserts the broken behaviour. These tests establish the boundary:
the two valid values work, and the attribute's presence is what a caller
depends on. The defect is written up in ``notes/bugs/config-precision-no-
terminal-else.md``.

``test_a_recognised_precision_produces_a_complete_dtype_table``
   Undocumented.

``test_only_two_precision_values_are_reachable_from_the_arguments``
   There is no ``--precision`` flag.

   Single is the default and double is reached only via ``subgrid``, so no
   user input can select a third value. That is what keeps the missing
   ``else`` latent rather than live — it is reachable only by a caller
   setting ``general["precision"]`` directly, which the API permits.

``test_calling_set_precision_again_with_a_bad_value_leaves_stale_dtypes``
   Re-running the method does not clear what it fails to set.

   Demonstrates the failure shape without asserting a wrong *answer*: after
   an unrecognised precision the table still holds the previous run's types,
   so nothing signals that the request was ignored.

TestConsistencyWithTheKernelDispatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The knot between this file and ``cpu_updates.set_dispersive_updates``.

Both read ``general["precision"]``. This file turns it into array dtypes;
the dispatcher turns it into part of a compiled kernel's name. If they ever
disagreed, the arrays would be allocated at one width and handed to a kernel
compiled for the other — which Cython rejects at the call boundary with a
buffer dtype error rather than computing wrongly.

Nothing in the source ties the two together, so these tests are the tie.

``test_array_dtype_and_kernel_name_agree``
   Undocumented.

``test_the_c_type_name_matches_the_numpy_width``
   ``C_float_or_double`` and ``float_or_double`` describe one type.

When these fail
~~~~~~~~~~~~~~~

**A Cython type assertion fails.** ``cython_float_or_double`` and
``cython_complex`` are asserted by **identity** (``is cython.float``),
because Cython's shadow types are not comparable any other way and ``==`` on
them is not meaningful.

**A ``C_complex`` value is ``None`` where a string was expected.** That
entry is set by a *second*, solver-dependent branch inside each precision
block, so it is ``None`` for the CPU solver and a backend-specific type name
otherwise. Eight rows: two precisions × four solvers.

**A row is unreachable through the constructor.** Every accelerator argument
forces ``precision`` to ``"single"``, so the double-precision accelerator
rows cannot be produced by passing arguments. Those tests set
``general["precision"]`` and ``general["solver"]`` directly and re-run
``_set_precision()``. An earlier version skipped them; the suite now has
**zero skips**.

**``AttributeError: 'SimulationConfig' object has no attribute 'dtypes'``.**
The precision string is neither ``"single"`` nor ``"double"``. There is no
terminal ``else``, so the attribute is never created — see
``notes/bugs/config-precision-no-terminal-else.md``.

Test Catalog — ``test_model_config.py``
---------------------------------------

**46 tests** from 39 test functions across 8 classes.

``ModelConfig`` — ``gprMax/config.py:43``.

One of these exists per model in a run. It carries the per-model mutable
state: the material summary the update kernels dispatch on, the numerical
dispersion thresholds, the memory tally, and the banner printed at the top
of each model.

The construction-order constraint is the thing to understand first.
``__init__`` reads the module-level ``sim_config`` three times — for
``model_end`` and ``input_file_path`` in the banner, and for ``args.n`` when
deciding whether to number the output file. So a ``ModelConfig`` cannot be
built before a ``SimulationConfig`` has been installed as the global. The
object under test depends on the global it is part of, which is why this
suite builds real objects rather than the stand-ins every other directory
uses.

Output-path construction is large enough for its own file; see
``test_output_paths.py``. The registry that decides *which* ``ModelConfig``
``get_model_config()`` returns is in ``test_model_registry.py``.

TestConstruction
^^^^^^^^^^^^^^^^

What ``ModelConfig(n)`` sets, and what it needs first.

``test_constructs_with_a_simulation_config_installed``
   Undocumented.

``test_requires_the_global_simulation_config``
   Without the global, construction fails on attribute access.

   ``ModelConfig.__init__`` reads ``sim_config.model_end`` while building
   the banner string, so a ``None`` global raises here rather than at first
   use.

``test_model_number_is_stored_as_given``
   Zero-based internally; the banner adds one for display.

``test_mode_defaults_to_three_dimensional``
   ``2D`` is selected later, by the grid, if a dimension is one cell.

``test_grids_starts_empty``
   Undocumented.

``test_thread_count_starts_unset``
   ``set_omp_threads`` fills this in once the host is known.

   Until then it is ``None``, and passing ``None`` into a Cython ``int
   nthreads`` parameter raises ``TypeError`` — which is why every test suite
   that drives a kernel has to set it explicitly.

``test_a_cpu_run_has_no_device_attribute``
   ``device`` is only created on the CUDA/OpenCL/Metal paths.

   Worth pinning because the stand-in config in the PR 10 outputs suite
   supplies a ``device`` key unconditionally — the real object does not have
   one on a CPU run.

``test_an_accelerator_run_has_a_three_key_device_dictionary``
   ``dev``, ``deviceID`` and ``snapsgpu2cpu``.

``test_snapshot_transfer_starts_disabled``
   Enabled later only if snapshots would not fit in device memory.

TestMemoryTally
^^^^^^^^^^^^^^^

``mem_overhead`` / ``mem_use`` — the running estimate.

``test_overhead_is_sixty_five_megabytes``
   The comment above it says 50 MB; the value is 65e6.

``test_usage_starts_at_the_overhead``
   Estimates accumulate on top of a fixed baseline.

``test_usage_is_a_mutable_running_total``
   ``mem_check_run_all`` adds to this in place.

TestNumericalDispersion
^^^^^^^^^^^^^^^^^^^^^^^

``numdispersion`` — three thresholds for the dispersion analysis.

``test_has_exactly_three_keys``
   Undocumented.

``test_default_threshold``
   40 dB down from peak power, 2% phase error, 3 cells per wavelength.

TestMaterials
^^^^^^^^^^^^^

``materials`` — the summary the update dispatchers read.

``test_has_exactly_five_keys``
   Undocumented.

``test_pole_count_starts_at_zero``
   Which routes ``update_electric_a`` to the plain kernel.

``test_derived_entry_starts_as_none``
   All four are filled in by ``set_dispersive_material_types``.

   ``dispersivedtype`` starting as ``None`` is the reason
   ``set_dispersive_updates`` silently selects a *real* kernel when it runs
   first — see ``tests/unit/updates/test_dispersive_dispatch.py``.

TestSetDispersiveMaterialTypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``set_dispersive_material_types`` — real or complex poles.

``test_debye_materials_get_the_real_dtype``
   Debye poles are purely relaxational, so real arithmetic suffices.

``test_drude_or_lorentz_materials_get_the_complex_dtype``
   Those two have resonant poles, which need complex arithmetic.

``test_real_path_uses_an_empty_real_extraction``
   ``crealfunc`` is pasted into GPU kernel source.

   For a real dtype there is nothing to extract, so the substitution is the
   empty string rather than a no-op call.

``test_complex_path_extracts_the_real_component``
   Undocumented.

``test_the_dtype_matches_the_configured_precision``
   Double-precision runs get ``complex128``, not ``complex64``.

``test_the_c_dtype_is_set_alongside_the_numpy_one``
   Undocumented.

``test_an_unset_drudelorentz_flag_takes_the_real_path``
   ``None`` is falsy, so the default takes the Debye branch.

   Benign — a model with no dispersive materials at all never reaches a
   dispersive kernel — but it means the flag has three states and only two
   behaviours.

``test_the_result_agrees_with_the_kernel_dispatch``
   The knot with ``set_dispersive_updates``.

   The dispatcher decides ``real`` versus ``complex`` by comparing
   ``dispersivedtype`` against ``sim_config.dtypes["complex"]``. That
   comparison is only meaningful if this method wrote one of exactly those
   two values, which it does.

TestBanner
^^^^^^^^^^

``inputfilestr`` — the header printed before each model runs.

``test_contains_the_one_based_model_number``
   Displayed as ``Model 3/5`` for internal index 2.

``test_first_model_displays_as_one``
   Undocumented.

``test_contains_the_input_file_path``
   Undocumented.

``test_is_padded_to_the_terminal_width``
   The trailing rule fills the line.

   ``get_terminal_width`` is patched to a fixed value by the suite's autouse
   fixture — the real one differs between an interactive shell, pytest with
   no tty, and each CI runner, so an unpinned assertion here would be flaky
   across the three OSes.

``test_is_wrapped_in_colour_codes``
   Green, reset — colorama constants.

TestGeometryReuse
^^^^^^^^^^^^^^^^^

``reuse_geometry`` — skip rebuilding for later models.

``test_the_first_model_never_reuses``
   There is nothing to reuse yet.

``test_a_later_model_reuses_when_the_flag_is_set``
   Undocumented.

``test_a_later_model_does_not_reuse_by_default``
   Undocumented.

``test_every_model_after_the_first_reuses``
   Undocumented.

TestUserNamespace
^^^^^^^^^^^^^^^^^

``get_usernamespace`` — the names visible to deprecated ``#python`` blocks.

``test_includes_the_electromagnetic_constants``
   Undocumented.

``test_includes_the_run_counters``
   Undocumented.

``test_model_run_number_is_one_based``
   Undocumented.

``test_input_file_is_absolute``
   ``resolve()`` is called, so the path is made absolute.

   This touches the filesystem — the only such call on the read path of a
   ``ModelConfig``.

``test_has_exactly_seven_names``
   Undocumented.

When these fail
~~~~~~~~~~~~~~~

**``AttributeError: 'NoneType' object has no attribute ...`` at
construction.** ``ModelConfig.__init__`` reads the module-level
``sim_config`` for the banner, for ``model_end`` and for ``args.n``. A
``ModelConfig`` cannot exist before a ``SimulationConfig`` is installed as
the global; ``make_model_config`` installs one first, always.

**A banner assertion is off by a few characters.** The string is padded to
``get_terminal_width() - 1``, which is environment-dependent — 80 under
pytest, different on each CI runner. ``TERMINAL_WIDTH = 100`` in the
conftest pins it, and the colour codes (``\x1b[32m`` … ``\x1b[0m``) count as
characters but not columns.

**A dispersive-type test fails.** ``set_dispersive_material_types`` reads
``sim_config.dtypes``, so the *precision* determines the answer as much as
the Drude/Lorentz flag does. Complex for Drude or Lorentz, real for Debye,
at whichever width the run uses.

**A ``materials`` or ``numdispersion`` key count changes.** Both are
asserted as complete key sets rather than by individual lookups, so an added
key fails here first. That is where the reader should look for what the new
key means.

Test Catalog — ``test_model_registry.py``
-----------------------------------------

**24 tests** from 24 test functions across 6 classes.

The model-config registry — which ``ModelConfig`` the package sees.

A simulation is a *sequence* of models, but almost every module in gprMax
reaches for its configuration through a single no-argument call::

   from gprMax.config import get_model_config
   config = get_model_config()

There is no argument, no lookup key and no object passed down the call
chain. The answer comes from two pieces of mutable state on the module-level
``sim_config``: a list of ``ModelConfig`` slots (``model_configs``) and an
integer cursor into it (``current_model``). The context loop moves the
cursor between models; everything downstream silently follows.

That indirection is why this file exists. **Every other directory under
``tests/unit/`` replaces ``get_model_config`` with a stub**, precisely so it
does not have to care about the cursor — which means the real
``get_model_config`` / ``set_model_config`` / ``set_current_model`` triad
has never been executed by a test, despite being the single most-called
piece of configuration code in the package. A cursor left pointing at the
wrong slot would not raise: model 2 would quietly be written with model 1's
output path, materials and dispersive dtypes.

The tests below therefore assert three separate things:

- the plumbing — set a config, get it back, on the slot asked for;

- the *default* argument, which is where the cursor enters (both getters and
  both setters take ``model_num=None`` and mean "whatever ``current_model``
  says"), and which is the behaviour every caller in the package relies on;

- the failure mode — an unset slot raises ``ValueError`` and logs, rather
  than returning ``None`` for a caller to trip over later.

``sim_config`` is installed as the real module global here, not a double.
``restore_config_globals`` in ``conftest.py`` puts the original back.

TestTheCursor
^^^^^^^^^^^^^

``current_model`` — the integer that decides what everyone sees.

``test_starts_at_the_first_model``
   Undocumented.

``test_set_current_model_moves_it``
   Undocumented.

``test_the_cursor_is_not_range_checked``
   ``set_current_model`` is a bare assignment.

   Out-of-range values are accepted silently and only fail later, at the
   indexing in ``get_model_config``. Pinned so a future bounds check is a
   deliberate change rather than an accident.

TestStoringAndRetrieving
^^^^^^^^^^^^^^^^^^^^^^^^

``set_model_config`` / ``get_model_config`` on an explicit slot.

``test_a_config_can_be_stored_and_retrieved``
   Undocumented.

``test_slots_are_independent``
   Undocumented.

``test_storing_twice_replaces``
   Undocumented.

``test_the_stored_object_is_returned_by_identity``
   No copying, no wrapping — the caller gets the same object back.

   ``ModelConfig`` is mutated in place all through a model run (memory
   tallies, dispersive dtypes, the output path), so a copy anywhere in this
   path would silently discard those updates.

TestTheDefaultSlot
^^^^^^^^^^^^^^^^^^

``model_num=None`` means "the current model" — for both directions.

``test_getting_without_a_number_follows_the_cursor``
   Undocumented.

``test_storing_without_a_number_follows_the_cursor``
   Undocumented.

``test_moving_the_cursor_changes_the_answer``
   The whole point of the indirection, in one test.

   Nothing is passed between these two calls; only the cursor moved.

``test_slot_zero_is_not_confused_with_the_default``
   ``model_num=0`` is falsy, so the guard must test ``is None``.

   A truthiness check here would make model 0 unaddressable whenever the
   cursor sat elsewhere.

TestUnsetSlots
^^^^^^^^^^^^^^

An empty slot is an error, not a ``None``.

``test_every_slot_starts_empty``
   Undocumented.

``test_getting_an_unset_slot_raises``
   Undocumented.

``test_getting_an_unset_current_model_raises``
   Undocumented.

``test_the_missing_model_number_is_logged``
   The raise is bare, so the log line carries the whole diagnosis.

``test_a_cursor_past_the_end_raises_index_error_not_value_error``
   The out-of-range case is a different failure from the unset case.

   ``get_model_config`` indexes before it checks for ``None``, so a cursor
   beyond the list raises ``IndexError`` with no log line. That distinction
   matters when reading a traceback: ``ValueError`` means the model was
   never configured, ``IndexError`` means the *index arithmetic* is wrong —
   the ``-i``/``-n`` mismatch written up in ``notes/bugs/config-model-index-
   range-mismatch.md``.

TestTheModuleLevelHelper
^^^^^^^^^^^^^^^^^^^^^^^^

``config.get_model_config()`` — the function the whole package imports.

``test_it_delegates_to_the_installed_simulation_config``
   Undocumented.

``test_it_takes_no_arguments``
   Callers cannot ask for a specific model through this door.

   The only way to change the answer is to move the cursor, which is why
   ``set_current_model`` is called exactly once per model by the context
   loop and nowhere else.

``test_it_follows_the_cursor``
   Undocumented.

``test_it_reads_the_global_at_call_time``
   The lookup is late-bound, so replacing the global takes effect.

   This is what lets every other test directory swap in a stand-in
   ``sim_config`` and have the real ``get_model_config`` return it.

``test_it_fails_loudly_when_no_simulation_config_is_installed``
   The shipped value of the global is ``None``.

   Importing ``gprMax`` does not create a ``SimulationConfig``; the API and
   the CLI both build one. Calling before that point is an
   ``AttributeError`` on ``None``, not a helpful message.

TestTheCursorAlsoSelectsTheScene
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One cursor, two lists — scenes are indexed by the same integer.

``test_the_scene_follows_the_cursor``
   Undocumented.

``test_a_model_config_reads_its_scene_through_its_own_number``
   ``ModelConfig.get_scene`` passes ``self.model_num``, not the cursor.

   So a ``ModelConfig`` keeps pointing at its own scene even if the cursor
   has already moved on — the one place in this file where the answer does
   *not* depend on ``current_model``.

``test_scene_slots_and_config_slots_are_the_same_length``
   Both default to ``number_of_models``, so one cursor addresses both.

When these fail
~~~~~~~~~~~~~~~

**The whole file is about one integer.** ``current_model`` selects which
``ModelConfig`` the no-argument ``config.get_model_config()`` returns.
Nothing is passed between caller and callee; only the cursor moves. If an
ordering assertion here fails, the question is which cursor value was in
effect.

**``IndexError`` where ``ValueError`` was expected.** They mean different
things. ``ValueError`` means the model was never configured; ``IndexError``
means the cursor is outside the list, which is the ``-i``/``-n`` sizing
mismatch — see ``notes/bugs/config-model-index-range-mismatch.md``.

**Model 0 becomes unaddressable.** The default-argument guards must test
``model_num is None``, not truthiness. ``0`` is falsy, so a truthiness check
would redirect every request for the first model to the cursor.

**``config.get_model_config()`` returns something unexpected.** The lookup
is late-bound — it reads the global at call time. That is exactly what lets
the other ten directories swap in a stand-in ``sim_config`` and have the
real function return it.

Test Catalog — ``test_output_paths.py``
---------------------------------------

**30 tests** from 28 test functions across 6 classes.

Where a model's files end up.

Every artefact a run produces — the ``.h5`` output file, the snapshot
directory, the geometry files — is derived from one attribute,
``ModelConfig.output_file_path``, computed once in ``set_output_file_path``.
Three inputs feed it, in strict priority order:

1. ``outputdir``, passed by the ``#output_dir:`` input-file command; 2.
``args.outputfile``, from the API or ``-o`` on the command line; 3. failing
both, the *input* file path with its extension stripped.

Then two suffix operations are applied on top: the model number is appended
to the final path component (so a three-model run does not overwrite
itself), and ``.h5`` is attached to give ``output_file_path_ext``.

The reason to test this in isolation is that a mistake here is invisible
until the very end of a run. Nothing reads ``output_file_path`` while the
model is solving; it is consumed when the last timestep has already been
computed. A wrong path does not crash a simulation, it loses one — or,
worse, silently overwrites the previous model's results because the model
number was dropped.

Two of the tests below pin behaviour that is a known defect rather than a
contract, and say so in their docstrings. They exist because the current
behaviour is surprising and undocumented, and a reader hitting it needs to
find something in the suite that explains it.

TestTheDefaultPath
^^^^^^^^^^^^^^^^^^

No ``-o`` and no ``#output_dir:`` — the input file names the output.

``test_the_input_file_path_is_reused``
   Undocumented.

``test_the_input_file_extension_is_stripped``
   Undocumented.

``test_the_directory_of_the_input_file_is_kept``
   Undocumented.

``test_nothing_is_created_on_disk``
   Computing the path must not touch the filesystem.

   ``ModelConfig`` is constructed before the model is built, and may be
   constructed for a run that never completes. Only the explicit
   ``outputdir`` branch is allowed to create anything.

TestTheOutputFileArgument
^^^^^^^^^^^^^^^^^^^^^^^^^

``-o`` / ``outputfile=`` overrides the input file name.

``test_the_argument_wins_over_the_input_file``
   Undocumented.

``test_an_extension_on_the_argument_is_stripped``
   Users pass ``-o out.h5``; the ``.h5`` is added back later.

   Without the strip the file would be written as ``out.h5.h5``.

``test_the_argument_is_not_created_on_disk``
   Undocumented.

TestTheOutputDirectory
^^^^^^^^^^^^^^^^^^^^^^

``#output_dir:`` — the only branch with a filesystem side effect.

``test_the_directory_is_created``
   Undocumented.

``test_the_input_file_stem_is_placed_inside_it``
   Only the *stem* survives — the input file's own directory is dropped.

``test_an_existing_directory_is_accepted``
   ``exist_ok=True``, so re-running a multi-model simulation is fine.

``test_it_takes_priority_over_the_output_file_argument``
   Undocumented.

``test_a_missing_parent_directory_is_not_created``
   ``mkdir`` is called without ``parents=True``.

   ``#output_dir: results/2026/run_a`` therefore raises
   ``FileNotFoundError`` from deep inside configuration rather than
   reporting a bad path, even though the value came straight from user
   input. Pinned as the current behaviour, with the defect written up in
   ``notes/bugs/config-output-dir-no-parents.md``.

TestTheModelNumberSuffix
^^^^^^^^^^^^^^^^^^^^^^^^

A multi-model run must not overwrite itself.

``test_a_single_model_run_appends_nothing``
   Undocumented.

``test_a_multi_model_run_appends_a_one_based_number``
   Undocumented.

``test_the_number_is_the_model_index_plus_one``
   Undocumented.

``test_the_number_lands_on_the_file_name_not_the_directory``
   Undocumented.

``test_consecutive_models_get_distinct_paths``
   The property that actually matters, asserted directly.

``test_the_number_survives_an_output_directory``
   Undocumented.

TestTheExtendedPath
^^^^^^^^^^^^^^^^^^^

``output_file_path_ext`` — the name the ``.h5`` writer is handed.

``test_it_is_the_path_with_an_h5_suffix``
   Undocumented.

``test_it_includes_the_model_number``
   Undocumented.

``test_it_is_recomputed_when_the_path_changes``
   Both attributes are set together, so they cannot drift apart.

``test_a_dot_in_the_file_name_truncates_it``
   ``with_suffix`` treats everything after the last dot as an extension.

   ``v1.2_model.in`` becomes ``v1.h5``: ``.2_model`` is read as a suffix and
   replaced. Dots in file names are ordinary — version numbers, dimensions,
   dates — so this quietly writes two different models to the same file.
   Pinned as the current behaviour, with the defect written up in
   ``notes/bugs/config-output-path-with-suffix.md``.

TestTheSnapshotDirectory
^^^^^^^^^^^^^^^^^^^^^^^^

``set_snapshots_dir`` — a sibling directory, not a child.

``test_it_is_the_output_name_with_a_suffix``
   Undocumented.

``test_it_sits_beside_the_output_file``
   Undocumented.

``test_it_includes_the_model_number``
   Each model in a B-scan gets its own snapshot directory.

``test_it_follows_an_output_directory``
   Undocumented.

``test_it_does_not_create_anything``
   The name is computed here; the directory is made by the writer.

   Unlike ``set_output_file_path``'s ``outputdir`` branch, this one is pure,
   so calling it to *inspect* the path is safe.

``test_it_is_recomputed_from_the_current_output_path``
   Nothing is cached — the answer tracks later ``#output_dir:`` commands.

When these fail
~~~~~~~~~~~~~~~

**A path assertion fails on the model number.** The number is appended to
the final *component*, not the directory, and only when ``n != 1``. A
single-model run appends nothing, which is why ``model.in`` gives
``model.h5`` and a three-model run gives ``model1.h5``.

**A file or directory appears on disk unexpectedly.** Only the ``outputdir``
branch touches the filesystem. Every other path is pure, and
``test_nothing_is_created_on_disk`` guards that — ``ModelConfig`` is
constructed before the model is built and may be constructed for a run that
never completes.

**``FileNotFoundError`` from ``set_output_file_path``.** ``mkdir`` is called
without ``parents=True``, so a nested ``#output_dir:`` fails. Pinned as
current behaviour; see ``notes/bugs/config-output-dir-no-parents.md``.

**An output name is shorter than the input name.** ``with_suffix`` truncates
at the last dot, so ``v1.2_model.in`` becomes ``v1.h5``. Two models can
collide on one file. See ``notes/bugs/config-output-path-with-suffix.md``.

Test Catalog — ``test_utilities.py``
------------------------------------

**101 tests** from 64 test functions across 10 classes.

Rounding, sorting and the small helpers everything else is built on.

Nothing in this file is more than a dozen lines of source, and that is
precisely the risk. These functions are called from geometry construction,
from grid discretisation and from the source and receiver placement code —
places where the caller assumes "round" means what Python's ``round`` means.
It does not.

Two deliberate departures from the standard library run through this suite:

- ``round_int`` uses ``ROUND_HALF_DOWN``: a tie goes toward **zero**, so
  ``0.5`` is ``0`` and ``-0.5`` is ``0``. Python's built-in ``round`` uses
  banker's rounding (ties to even), giving ``round(0.5) == 0`` but
  ``round(1.5) == 2`` — the two agree on the first and disagree on the
  second.

- ``round_float`` uses ``ROUND_FLOOR``: it truncates toward **negative
  infinity**, not toward zero, so ``-1.2345`` at two places is ``-1.24`` and
  not ``-1.23``.

Both choices are correct for their purpose — a cell index must not be
rounded up past the domain edge — but neither is discoverable from the call
site. The tables below exist so that a change to either is a failing test
rather than a geometry that is one cell out.

``get_terminal_width`` is included because two other modules build padded
banner strings from it, and its zero-width fallback is the only branch that
matters on a CI runner with no tty.

TestTerminalWidth
^^^^^^^^^^^^^^^^^

``get_terminal_width`` — the width every banner is padded to.

``test_returns_the_reported_width``
   Undocumented.

``test_a_zero_width_falls_back_to_one_hundred``
   A pipe or a CI log can report zero columns.

   Without the fallback the padding expressions elsewhere (``'-' *
   (get_terminal_width() - 1 - len(s))``) would go negative and silently
   produce no separator at all.

``test_only_the_first_element_is_used``
   Columns, not lines — the two are easy to transpose.

``test_the_real_call_returns_a_positive_integer``
   Unpatched, on whatever the runner is.

TestAtoi
^^^^^^^^

``atoi`` — convert if it looks like a number, otherwise pass through.

``test_digits_become_an_integer``
   Undocumented.

``test_non_digits_are_returned_unchanged``
   Undocumented.

``test_an_empty_string_is_returned_unchanged``
   ``"".isdigit()`` is ``False``, so this must not raise.

``test_leading_zeros_are_dropped``
   Undocumented.

``test_a_negative_sign_prevents_conversion``
   ``isdigit`` rejects the minus sign, so ``"-1"`` stays a string.

   Harmless for its actual use — sorting file names — but it means ``atoi``
   is not a general string-to-int conversion.

``test_a_decimal_point_prevents_conversion``
   Undocumented.

TestNaturalKeys
^^^^^^^^^^^^^^^

``natural_keys`` — the sort order humans expect for numbered files.

``test_digits_and_text_are_split_apart``
   Undocumented.

``test_a_string_with_no_digits_is_a_single_element``
   Undocumented.

``test_it_sorts_numerically_not_lexically``
   The whole point: ``model10`` must come after ``model9``.

``test_lexical_sorting_would_get_this_wrong``
   Stated explicitly so the value of the helper is visible.

``test_multiple_number_groups_are_all_converted``
   Undocumented.

``test_it_orders_snapshot_files_correctly``
   Undocumented.

TestRoundInt
^^^^^^^^^^^^

``round_int`` — ties toward zero, not toward even.

``test_non_tie_values_round_to_nearest``
   Undocumented.

``test_ties_go_toward_zero``
   ``ROUND_HALF_DOWN`` — magnitude never increases on a tie.

   This is the property the grid code depends on: a coordinate exactly half
   a cell past the last node must not round to a node that does not exist.

``test_it_disagrees_with_the_builtin_on_odd_ties``
   Stated as a test so the difference is not folded into a comment.

   The two rules coincide on *even* ties — banker's rounding sends ``2.5``
   to ``2``, and so does half-down — which is why the difference is easy to
   miss in casual testing. It shows up only on odd ties.

``test_it_agrees_with_the_builtin_on_even_ties``
   Undocumented.

``test_it_returns_a_python_int``
   Undocumented.

``test_an_integer_argument_is_accepted``
   Undocumented.

``test_a_numpy_float_is_rejected``
   ``decimal.Decimal`` accepts ``float`` and ``int``, but not
   ``np.float32``.

   Callers holding a value from an array must convert first. Pinned because
   the failure is a bare ``TypeError`` from inside ``decimal`` with no
   mention of gprMax, and is written up in ``notes/bugs/utilities-rounding-
   rejects-numpy-scalars.md``.

``test_a_numpy_float64_is_accepted``
   ``np.float64`` subclasses ``float``, so it slips through.

   The inconsistency with ``np.float32`` is the reason the restriction is
   worth writing down.

TestRoundFloat
^^^^^^^^^^^^^^

``round_float`` — truncation toward negative infinity.

``test_positive_values_truncate_downward``
   Undocumented.

``test_negative_values_also_go_downward``
   Toward −∞, so a negative number becomes *more* negative.

   ``ROUND_FLOOR`` is not ``ROUND_DOWN``; the latter truncates toward zero
   and would give ``-1.23`` for the first case.

``test_it_is_not_symmetric_about_zero``
   The consequence of ``ROUND_FLOOR``, stated once and plainly.

``test_zero_places_gives_an_integral_value``
   ``"1."`` with no zeros is still a valid quantisation target.

``test_it_returns_a_python_float``
   Undocumented.

``test_many_places_leave_the_value_alone``
   Undocumented.

``test_a_negative_place_count_is_silently_treated_as_zero``
   ``'0' * -1`` is the empty string, so the target becomes ``"1."``.

   ``Decimal("1.")`` is a valid integral quantiser, so a negative place
   count neither raises nor multiplies by a power of ten — it rounds to a
   whole number. Surprising, but harmless, and pinned so the behaviour is
   documented somewhere.

TestRoundValue
^^^^^^^^^^^^^^

``round_value`` — the dispatcher the rest of the package calls.

``test_zero_places_uses_the_integer_rounding``
   Undocumented.

``test_zero_places_is_the_default``
   Undocumented.

``test_a_nonzero_place_count_uses_the_float_rounding``
   Undocumented.

``test_zero_places_returns_an_integer``
   Undocumented.

``test_nonzero_places_returns_a_float``
   Undocumented.

``test_the_return_type_depends_on_the_place_count``
   One argument changes the *type* of the result, not just its value.

   Worth an explicit test: a caller passing ``decimalplaces`` through from
   configuration gets an ``int`` or a ``float`` depending on data.

TestRound32
^^^^^^^^^^^

``round32`` — round *up* to a multiple of 32, for kernel block sizes.

``test_it_rounds_up_to_the_next_multiple``
   Undocumented.

``test_an_exact_multiple_is_unchanged``
   Otherwise every launch would over-allocate by a whole block.

``test_the_result_is_always_a_multiple_of_thirty_two``
   Undocumented.

``test_the_result_is_never_smaller_than_the_input``
   Undocumented.

``test_it_returns_a_python_int``
   ``np.ceil`` returns a float; the cast matters for array shapes.

``test_a_float_input_is_accepted``
   Undocumented.

``test_a_string_input_is_accepted``
   ``float(value)`` first, so a value straight from a hash command works.

TestFftPower
^^^^^^^^^^^^

``fft_power`` — the spectrum used for numerical-dispersion analysis.

``test_the_maximum_power_is_zero_decibels``
   The whole array is shifted so the peak sits at 0 dB.

``test_no_power_exceeds_the_peak``
   Undocumented.

``test_the_frequency_bins_match_the_waveform_length``
   Undocumented.

``test_the_first_bin_is_direct_current``
   Undocumented.

``test_the_bin_spacing_is_the_reciprocal_of_the_record_length``
   Undocumented.

``test_the_peak_lands_on_the_signal_frequency``
   A pure tone must show up in the bin it belongs to.

``test_an_all_zero_waveform_does_not_produce_infinities``
   ``log10(0)`` is ``-inf``; the function replaces non-finite values.

   A zeroed receiver trace is entirely normal — a source that has not fired
   yet — so this path is reached in ordinary use.

``test_a_waveform_with_a_zero_bin_does_not_produce_infinities``
   A real signal can still have an exactly zero frequency component.

``test_scaling_the_waveform_does_not_change_the_spectrum``
   Power is relative to its own maximum, so amplitude cancels out.

TestTimer
^^^^^^^^^

``timer`` — a monotonic clock, wrapped for a single import point.

``test_it_returns_a_float``
   Undocumented.

``test_it_does_not_go_backwards``
   ``perf_counter``, not ``time`` — immune to the system clock moving.

   Solve times are differences of two of these, so a wall-clock jump during
   a long run would otherwise produce a negative duration.

``test_it_is_the_performance_counter``
   Pinned by behaviour: two calls sit within one counter tick.

TestLogo
^^^^^^^^

``logo`` — the banner printed once at startup.

``test_it_returns_a_string``
   Undocumented.

``test_the_version_appears``
   Undocumented.

``test_the_current_year_appears_in_the_copyright``
   Generated from the clock, so a stale year is a real possibility.

``test_the_project_url_appears``
   Undocumented.

``test_the_authors_appear``
   Undocumented.

``test_the_licence_is_named``
   Undocumented.

``test_no_line_exceeds_the_terminal_width``
   The reason the width is read at all.

   Colour escape sequences are stripped first — they occupy no columns but
   do occupy characters.

``test_it_does_not_print``
   The caller logs the returned string; the function is pure.

When these fail
~~~~~~~~~~~~~~~

**A rounding assertion disagrees with Python's ``round``.** It is meant to.
``round_int`` uses ``ROUND_HALF_DOWN`` — ties toward **zero** — and
``round_float`` uses ``ROUND_FLOOR`` — toward **−∞**. The two rules coincide
on *even* ties (``2.5 → 2`` either way) and differ on odd ones (``1.5 → 1``
here, ``2`` for the builtin), which is why the difference is easy to miss.

**``TypeError`` from inside ``decimal``.** A ``np.float32`` was passed.
``Decimal`` accepts ``float`` and ``int``; ``np.float64`` slips through
because it subclasses ``float``, and ``np.float32`` does not. See
``notes/bugs/utilities-rounding-rejects-numpy-scalars.md``.

**A ``logo`` or terminal-width assertion is off.** ``get_terminal_width``
reads the real terminal. The ``TestLogo`` class pins it to 100 with an
autouse fixture; without that, the line-length assertion is environment-
dependent.

**An ``fft_power`` test fails on infinities.** ``log10(0)`` is ``-inf``, and
the function replaces every non-finite value with ``0`` *before* shifting
the peak to 0 dB. A zeroed receiver trace — a source that has not fired yet
— reaches this path in ordinary use.

Test Catalog — ``test_logging.py``
----------------------------------

**51 tests** from 46 test functions across 7 classes.

The custom log level, the colour formatter, and the logger setup.

gprMax does not use ``print``. Everything a user sees — the banner, the host
description, the per-model progress, the memory warnings — arrives through
``logger.basic(...)``, a method that does not exist in the standard library.
``gprMax/utilities/logging.py`` creates it at **import time**, by mutating
the ``logging`` module itself:

- ``logging.addLevelName(25, "BASIC")`` registers the name;

- ``logging.BASIC = 25`` publishes the constant;

- ``logging.Logger.basic = basic`` bolts a method onto the stdlib class.

None of those are reversible, and all three are process-wide: any library in
the same interpreter now sees a ``BASIC`` level it never asked for. That is
a deliberate trade — it makes ``logger.basic`` available everywhere without
an import — but it means the tests below are asserting on the state of the
*standard library*, not on a gprMax object. They are written to be read that
way.

Level 25 sits between ``INFO`` (20) and ``WARNING`` (30), which is what
makes it useful: a user running at the default level sees ``BASIC`` output,
and ``--log-level 30`` silences it while leaving warnings intact. The
numeric ordering is therefore a contract, not an implementation detail, and
it is pinned below.

``logging_config`` is tested through the handlers it installs rather than
through captured output. ``caplog`` attaches its own handler at the root and
``propagate`` is set to ``False`` here, so the two do not compose;
inspecting ``logger.handlers`` is both more direct and more honest about
what the function actually does.

TestTheBasicLevel
^^^^^^^^^^^^^^^^^

Level 25 — registered in the standard library at import.

``test_the_number_is_twenty_five``
   Undocumented.

``test_it_sits_between_info_and_warning``
   The property that makes the level useful, not its value.

   A user at the default ``INFO`` threshold sees ``BASIC`` messages; raising
   the threshold to ``WARNING`` hides them without hiding warnings.

``test_the_name_is_registered_with_the_standard_library``
   Undocumented.

``test_the_reverse_lookup_also_works``
   ``getLevelName`` is bidirectional; ``--log-level BASIC`` needs this.

``test_the_constant_is_published_on_the_logging_module``
   ``logging.BASIC`` — so callers need not import from gprMax.

``test_the_method_is_attached_to_every_logger``
   Including loggers created before gprMax was imported.

   ``Logger.basic`` is set on the *class*, so it appears retroactively.

TestBasicMessages
^^^^^^^^^^^^^^^^^

``logger.basic(...)`` — what the method actually emits.

``test_a_message_is_recorded_at_level_twenty_five``
   Undocumented.

``test_the_level_name_on_the_record_is_basic``
   Undocumented.

``test_the_message_survives``
   Undocumented.

``test_arguments_are_interpolated_lazily``
   ``%``-style arguments, as with every other level method.

``test_it_is_suppressed_above_its_level``
   The point of putting it below ``WARNING``.

``test_a_warning_still_passes_at_that_threshold``
   Paired with the previous test: only ``BASIC`` is silenced.

``test_the_enabled_check_happens_before_the_log_call``
   ``isEnabledFor`` guards the body, so a disabled call is cheap.

   Asserted through behaviour: nothing is recorded, and no formatting error
   is raised by the deliberately broken format string.

TestTheColourMapping
^^^^^^^^^^^^^^^^^^^^

``MAPPING`` — one colour per level name.

``test_every_standard_level_has_a_colour``
   Undocumented.

``test_warnings_and_errors_are_visually_distinct``
   The reason the mapping exists at all.

``test_the_quiet_levels_share_one_colour``
   Undocumented.

TestCustomFormatter
^^^^^^^^^^^^^^^^^^^

``CustomFormatter`` — colour wrapped around level name and message.

``test_the_message_appears_in_the_output``
   Undocumented.

``test_the_level_name_appears_when_the_pattern_asks_for_it``
   Undocumented.

``test_colour_escapes_are_added``
   Undocumented.

``test_the_colour_is_reset_afterwards``
   Otherwise every subsequent line in the terminal stays coloured.

``test_every_level_formats``
   Undocumented.

``test_an_unmapped_level_gets_the_fallback_colour``
   A level gprMax did not register still formats rather than raising.

``test_the_original_record_is_not_modified``
   The formatter copies first.

   A file handler attached to the same logger must not receive escape
   sequences because a console handler formatted the record first.

``test_a_message_with_no_arguments_formats``
   The only shape gprMax actually uses — every call site is an f-string.

``test_a_message_with_arguments_raises``
   ``getMessage()`` runs twice, so lazy ``%``-style logging is broken.

   ``format`` interpolates the arguments and assigns the *result* back to
   ``msg`` — but leaves ``args`` in place. The base ``Formatter.format``
   then calls ``getMessage()`` again on the already-interpolated string, and
   the surviving arguments have nothing left to fill.

   The idiomatic stdlib call ``logger.basic("%d cores", 6)`` therefore fails
   inside the handler. It has never been noticed because every call site in
   gprMax passes a pre-built f-string. Pinned as the current behaviour;
   written up in ``notes/bugs/logging-custom-formatter-double-
   interpolation.md``.

TestLoggingConfig
^^^^^^^^^^^^^^^^^

``logging_config`` — the setup call the CLI and API both make.

``test_a_handler_is_installed``
   Undocumented.

``test_the_handler_writes_to_stdout``
   Not stderr — gprMax's normal output is not an error stream.

``test_the_logger_itself_is_set_to_debug``
   Filtering happens on the handler, so the logger must let all through.

   This is what allows a file handler at ``DEBUG`` to coexist with a console
   handler at ``INFO``.

``test_the_requested_level_lands_on_the_handler``
   Undocumented.

``test_propagation_is_disabled``
   Otherwise every message would also reach the root logger's handlers,
   printing it twice for anyone who called ``basicConfig``.

``test_calling_twice_does_not_accumulate_handlers``
   The API can be called repeatedly in one interpreter session.

``test_the_default_name_is_gprmax``
   Every module logger is ``gprMax.<something>``, so this is the root.

``test_the_default_level_is_info``
   Undocumented.

TestFormatStyles
^^^^^^^^^^^^^^^^

``format_style`` — the terse default and the diagnostic alternative.

``test_the_standard_style_is_the_message_alone``
   Undocumented.

``test_the_full_style_carries_the_source_location``
   Undocumented.

``test_the_full_style_carries_a_timestamp``
   Undocumented.

``test_debug_level_forces_the_full_style``
   ``--log-level DEBUG`` upgrades the format without a second flag.

   Anyone asking for debug output wants to know where it came from.

``test_the_formatter_is_the_colour_one``
   Undocumented.

TestFileLogging
^^^^^^^^^^^^^^^

``log_file=True`` — a second, uncoloured handler on disk.

``test_a_second_handler_is_added``
   Undocumented.

``test_the_second_handler_writes_to_a_file``
   Undocumented.

``test_the_file_name_starts_with_the_logger_name``
   Undocumented.

``test_the_file_handler_records_everything``
   ``DEBUG`` on disk regardless of the console level.

   A user reporting a problem can be asked for the log file without being
   asked to re-run at a different verbosity.

``test_the_file_is_not_coloured``
   Escape sequences in a text file are noise, not colour.

``test_the_file_always_uses_the_full_format``
   Even when the console is terse.

``test_messages_reach_the_file``
   End to end: configure, log, close, read back.

``test_the_level_name_reaches_the_file``
   Undocumented.

When these fail
~~~~~~~~~~~~~~~

**A level assertion fails and the number is not 25.** ``logging.py`` mutates
the standard library at **import**: ``addLevelName(25, "BASIC")``,
``logging.BASIC = 25``, and ``logging.Logger.basic = basic``. None of it is
reversible, and all of it is process-wide. These tests assert on the state
of ``logging``, not on a gprMax object.

**A ``CustomFormatter`` test raises ``TypeError``.** The formatter
interpolates the message and assigns the result back to ``msg`` while
leaving ``args`` in place, so ``Formatter.format`` interpolates a second
time. Every ``%``-style logging call fails inside the handler. It has never
been noticed because every call site in gprMax passes a pre-built f-string.
See ``notes/bugs/logging-custom-formatter-double-interpolation.md``.

**A handler assertion fails after another test ran.**
``logging.getLogger(name)`` caches by name in a process-wide dictionary. The
``temporary_logger`` fixture creates uniquely named loggers and closes their
handlers on teardown; reusing ``"gprMax"`` would leave handlers attached for
the rest of the session, including in suites that assert on log output.

**``caplog`` records nothing from a configured logger.** ``logging_config``
sets ``propagate = False``, so messages never reach the root handler
``caplog`` attaches. These tests inspect ``logger.handlers`` instead, which
is both more direct and more honest about what the function does.

Test Catalog — ``test_host_info.py``
------------------------------------

**66 tests** from 66 test functions across 7 classes.

``get_host_info`` — the probe that describes the machine gprMax runs on.

This is the most-shelled-out function in the package and, until now, the
least tested. It runs three or four external commands, parses their stdout
with string surgery, and returns nine keys that appear in the run banner,
drive the OpenMP thread count and gate the memory warnings.

**Why it is worth this much attention.** The Windows branch contains the
student's own merged contribution: Microsoft removed ``wmic`` in Windows 11
25H2, so ``subprocess.check_output(["wmic", ...])`` raises
``FileNotFoundError`` rather than ``CalledProcessError``, which the original
``except`` clause did not catch — gprMax crashed on startup before printing
anything. The fix (``ce2c456e``) widened the clause and added a PowerShell
``Get-CimInstance`` fallback. ``gsocDocs/feats/setup-and-wmic-fix.rst``
records that it was verified *by temporarily adding print statements inside
each except block*. These tests replace that with something a CI runner can
check.

**Why the fakes are total.** Every ``subprocess.check_output`` call is
served from a table (``fake_subprocess``), ``sys.platform`` is patched, and
so are the five ``platform`` lookups and the two ``psutil`` counts. That is
heavy — but it is the only way to get three properties that matter:

- all three platform branches run on all three CI runners, instead of two
  being dead code on each;

- the wmic-absent path can be *forced*, which is impossible on a machine
  that still has wmic — the exact situation in which the bug was missed;

- the suite gives the same answer on every machine, so a failure is a code
  change and never a hardware difference.

A few tests pin behaviour that is defective rather than contractual. They
say so, and name the file in ``notes/bugs/``. They are here because the
failure modes are silent — a wrong socket count does not raise, it just
prints a wrong banner and, on a workstation, chooses the wrong number of
OpenMP threads.

TestTheReturnedDictionary
^^^^^^^^^^^^^^^^^^^^^^^^^

Shape and provenance of the result, independent of platform.

``test_exactly_nine_keys_are_returned``
   Undocumented.

``test_the_hostname_comes_from_the_platform_module``
   Not from any subprocess — the one field that never shells out.

``test_the_core_counts_come_from_psutil``
   Undocumented.

``test_the_memory_total_comes_from_psutil``
   Undocumented.

``test_the_physical_core_count_falls_back_to_the_logical_one``
   ``psutil.cpu_count(logical=False)`` returns ``None`` on some machines.

   Containers and some ARM hosts do not expose the topology. Without the
   fallback, ``set_omp_threads`` would set ``OMP_NUM_THREADS`` to
   ``"None"``.

``test_a_zero_physical_core_count_also_falls_back``
   ``not 0`` is true, so zero takes the same branch as ``None``.

TestWindowsWithWmic
^^^^^^^^^^^^^^^^^^^

The Windows branch when ``wmic`` is present and working.

``test_the_manufacturer_is_read_from_wmic``
   Undocumented.

``test_the_model_is_read_from_wmic``
   Undocumented.

``test_the_machine_id_joins_manufacturer_and_model``
   Undocumented.

``test_the_wmic_header_line_is_skipped``
   ``wmic`` prints the column name first; element ``[1]`` is the value.

   Without the skip the banner would read ``Vendor Model``.

``test_carriage_returns_are_stripped``
   Windows commands emit ``\r\n``; the parser splits on ``\n`` only.

``test_a_single_line_response_is_used_as_is``
   Some ``wmic`` builds omit the header; the parser handles both.

``test_the_cpu_name_is_read_from_wmic``
   Undocumented.

``test_one_cpu_line_means_one_socket``
   Sockets are *counted*, not queried — one output line each.

``test_two_cpu_lines_mean_two_sockets``
   Undocumented.

``test_blank_lines_do_not_count_as_sockets``
   ``wmic`` pads its output; a blank line is not a processor.

``test_internal_whitespace_in_the_cpu_name_is_collapsed``
   ``wmic`` pads processor names with runs of spaces.

``test_the_os_version_names_windows_and_its_bit_width``
   Undocumented.

``test_a_thirty_two_bit_machine_is_reported_as_such``
   Undocumented.

``test_hyperthreading_is_detected_from_the_core_counts``
   Undocumented.

``test_equal_core_counts_mean_no_hyperthreading``
   Undocumented.

TestWindowsWithoutWmic
^^^^^^^^^^^^^^^^^^^^^^

The reason this file exists: Windows 11 25H2 removed ``wmic``.

``subprocess.check_output`` raises ``FileNotFoundError`` — not
``CalledProcessError`` — when the executable does not exist. The original
code caught only the latter, so gprMax died on import with a traceback from
inside ``host_info``. Each test below forces that exact condition.

``test_a_missing_wmic_does_not_crash``
   The regression the fix exists to prevent.

``test_the_powershell_vendor_command_is_issued``
   The *exact* argv, not merely "something with powershell in it".

``test_the_powershell_model_command_is_issued``
   Undocumented.

``test_the_powershell_cpu_command_is_issued``
   Undocumented.

``test_powershell_is_tried_only_after_wmic_fails``
   Order matters: ``wmic`` is faster, so it stays the first choice.

``test_the_powershell_output_has_no_header_to_skip``
   ``Select-Object -ExpandProperty`` prints the value alone.

   So the fallback parses with ``.strip()`` only. Feeding it wmic-shaped
   output with a header would produce the header as the answer — the
   asymmetry is why the two paths cannot share a parser.

``test_all_three_fall_back_independently``
   A machine with no wmic at all takes every fallback in one run.

``test_the_socket_count_still_works_through_the_fallback``
   PowerShell prints one line per processor, so the count is unchanged.

``test_a_failing_wmic_also_falls_back``
   The original clause caught ``CalledProcessError``; it still must.

   Widening the ``except`` must not have lost the case it already handled —
   a ``wmic`` that exists but exits non-zero.

``test_both_probes_failing_leaves_the_field_unknown``
   The default set at the top of the function survives.

   A machine with neither wmic nor a working ``Get-CimInstance`` still gets
   a banner, just a vaguer one.

``test_a_missing_powershell_is_not_caught``
   The inner ``except`` still catches only ``CalledProcessError``.

   So the very failure mode the outer clause was widened for is unhandled
   one level down: on a machine with neither ``wmic`` nor ``powershell.exe``
   on ``PATH`` — a stripped Windows container, or a ``PATH`` that has lost
   ``System32`` — gprMax crashes exactly as it did before the fix. Pinned as
   the current behaviour; written up in ``notes/bugs/host-info-powershell-
   fallback-filenotfound.md``.

``test_no_cpu_information_at_all_leaves_the_socket_count_at_zero``
   ``sockets`` is reset to ``0`` before the loop, not left ``"unknown"``.

   The banner then reads ``0 x unknown``, which is at least honest.

TestMacOs
^^^^^^^^^

The ``darwin`` branch — three ``sysctl`` calls.

``test_the_manufacturer_is_hard_coded``
   No probe; Apple hardware has exactly one vendor.

``test_the_model_comes_from_sysctl``
   Undocumented.

``test_the_socket_count_is_an_integer``
   ``sysctl`` returns text; the conversion is what makes it usable.

``test_the_cpu_name_comes_from_the_brand_string``
   Undocumented.

``test_internal_whitespace_in_the_cpu_name_is_collapsed``
   Undocumented.

``test_the_os_version_reports_the_mac_release``
   Undocumented.

``test_a_failing_model_probe_leaves_the_field_unknown``
   Undocumented.

``test_a_failing_cpu_probe_leaves_both_cpu_fields_unknown``
   The two probes share one ``try``, so one failure loses both.

``test_apple_silicon_has_no_brand_string``
   ``machdep.cpu.brand_string`` does not exist on M-series chips.

   ``sysctl`` exits non-zero, so ``cpuID`` stays ``"unknown"`` on exactly
   the hardware gprMax users are most likely to be running today — and
   ``set_omp_threads`` then takes the ``ACTIVE`` wait-policy branch, which
   Apple's own tuning guide advises against for Apple silicon. Pinned as the
   current behaviour; written up in ``notes/bugs/host-info-apple-silicon-
   cpuid.md``.

``test_a_missing_sysctl_is_not_caught``
   The same blind spot the wmic fix closed, still open here.

   ``except subprocess.CalledProcessError`` without ``FileNotFoundError`` —
   one of four surviving instances outside the Windows branch. Pinned as the
   current behaviour; written up in ``notes/bugs/host-info-remaining-
   filenotfound-blind-spots.md``.

TestLinux
^^^^^^^^^

The ``linux`` branch — DMI files, ``/proc/cpuinfo`` and ``lscpu``.

``test_the_manufacturer_comes_from_the_dmi_tree``
   Undocumented.

``test_the_model_comes_from_the_dmi_tree``
   Undocumented.

``test_the_cpu_name_comes_from_proc_cpuinfo``
   Undocumented.

``test_the_last_model_name_line_wins``
   The loop assigns without breaking, so it ends on the final core.

   Harmless on a homogeneous machine, which is every machine gprMax runs on,
   but worth pinning: it is not the *first* match.

``test_the_socket_count_comes_from_lscpu``
   Undocumented.

``test_hyperthreading_is_two_threads_per_core``
   Unlike the other two branches, this one ignores ``psutil``.

``test_one_thread_per_core_means_no_hyperthreading``
   Undocumented.

``test_the_os_version_is_the_platform_string``
   Undocumented.

``test_the_locale_is_forced_to_english``
   ``lscpu`` translates its labels, and the parser matches on English.

   Asserted through the environment handed to the subprocess rather than
   through the output, since the fake cannot speak French.

``test_a_two_digit_socket_count_is_misread``
   ``int(line.strip()[-1])`` reads only the **last character**.

   ``"Socket(s): 12"`` therefore parses as ``2``. The banner under-reports,
   and on a large multi-socket node — precisely the machines gprMax is run
   on — the figure is silently wrong. Pinned as the current behaviour;
   written up in ``notes/bugs/host-info-lscpu-last-character-parse.md``.

``test_a_failing_dmi_read_leaves_the_machine_id_unknown``
   Reading the DMI tree needs root on some distributions.

``test_a_failing_lscpu_leaves_the_socket_count_unknown``
   ``lscpu`` is not installed in every minimal container image.

TestAnUnrecognisedPlatform
^^^^^^^^^^^^^^^^^^^^^^^^^^

No ``else`` on the platform chain.

``test_an_unknown_platform_raises_an_unbound_local_error``
   FreeBSD, Cygwin and AIX all reach the end with nothing assigned.

   ``machineID``, ``hyperthreading`` and ``osversion`` are only bound inside
   the three branches, so the dictionary construction raises
   ``UnboundLocalError`` — a failure that names a local variable rather than
   the unsupported platform. This is the same missing-terminal-``else``
   pattern found four other times in this PR's scope. Pinned as the current
   behaviour; written up in ``notes/bugs/host-info-no-terminal-else.md``.

TestPrintHostInfo
^^^^^^^^^^^^^^^^^

``print_host_info`` — the one line the user actually reads.

``test_the_hostname_is_printed``
   Undocumented.

``test_the_machine_id_is_printed``
   Undocumented.

``test_the_socket_count_and_cpu_are_printed_together``
   Undocumented.

``test_the_memory_is_printed_in_human_units``
   16 GiB, not 17179869184.

``test_hyperthreading_is_mentioned_when_present``
   Undocumented.

``test_hyperthreading_is_not_mentioned_when_absent``
   Undocumented.

``test_the_thread_count_line_reports_the_physical_cores``
   OpenMP is sized by physical cores, and the banner says so.

``test_it_logs_at_the_basic_level``
   Level 25, so ``--log-level 30`` suppresses the banner.

``test_it_emits_exactly_two_lines``
   Undocumented.

``test_the_first_two_fields_come_from_the_global_not_the_argument``
   A genuine inconsistency, pinned so it is not "fixed" by accident.

   ``hostname`` and ``machineID`` are read from
   ``config.sim_config.hostinfo``, while every other field comes from the
   ``hostinfo`` argument. Passing a different dictionary — as a caller
   reasonably might, to describe a remote node — silently mixes the two.
   Written up in ``notes/bugs/host-info-print-mixes-sources.md``.

When these fail
~~~~~~~~~~~~~~~

**A test reaches the real shell.** ``fake_subprocess`` raises
``FileNotFoundError`` for any argv not registered, so an unanticipated
command fails loudly. If a test starts depending on the runner's hardware,
that fixture has been bypassed.

**A parse assertion fails on whitespace.** ``wmic`` prints a header line and
pads with ``\r\n``; the parser splits on ``\n`` and takes element ``[1]``.
PowerShell's ``Select-Object -ExpandProperty`` prints the value alone and is
parsed with ``.strip()`` only. The two paths **cannot share a parser**, and
the fixtures reproduce both output shapes exactly.

**A PowerShell command assertion fails.** The tests assert the *exact* argv,
not "something containing powershell". The three commands live in the
``powershell_commands`` fixture so a change to the fix is a one-line update
rather than a search.

**A socket count is wrong by a factor of ten.** ``int(line.strip()[-1])``
reads only the last character, so ``"Socket(s): 12"`` parses as ``2``. See
``notes/bugs/host-info-lscpu-last-character-parse.md``.

**``UnboundLocalError`` on an unrecognised platform.** The three-way
``sys.platform`` chain has no ``else``. See ``notes/bugs/host-info-no-
terminal-else.md``.

**24 ``DeprecationWarning``\ s appear.** ``re.sub`` is called with a
positional ``count``, removed in Python 3.15. Nothing previously ran the
Linux branch on a Windows or macOS runner, so this suite is what surfaced
it. See ``notes/bugs/host-info-re-sub-positional-count.md``.

Test Catalog — ``test_omp_threads.py``
--------------------------------------

**33 tests** from 32 test functions across 6 classes.

``set_omp_threads`` — the five environment variables that size the solver.

The CPU solver's inner loops are OpenMP-parallel Cython. OpenMP is
configured entirely through environment variables read by the runtime *when
the first parallel region is entered*, which means this function has to run
before any kernel does, and that whatever it writes is what the solver gets.
There is no API to query it afterwards and no way to change it later in the
process.

That makes ``os.environ`` the return value, in practice. ``nthreads`` is
returned too, but four of the five variables the function sets are never
read back by gprMax — they exist purely to be seen by libgomp. So the tests
below assert on the environment rather than on the return value, and every
one of them goes through ``monkeypatch``: a leaked ``OMP_NUM_THREADS`` would
change how the *rest of the test session* runs, not merely this file.

Three behaviours are worth naming before reading the tests:

- **The thread count is chosen from physical cores, not logical ones.**
  Hyperthreads share an execution unit, so running two OpenMP threads on one
  core makes an FDTD kernel slower, not faster. This is the single most
  performance-relevant line in the utilities package.

- **An existing ``OMP_NUM_THREADS`` wins.** A user, a scheduler or an MPI
  launcher may have set it deliberately; gprMax defers.

- **Windows Subsystem for Linux gets special treatment.** ``OMP_PLACES`` and
  ``OMP_PROC_BIND`` are *deleted* there, because binding threads to cores
  under WSL hangs (microsoft/WSL#785).

``clean_omp_environment`` in ``conftest.py`` removes any inherited ``OMP_*``
before each test, so a developer who exports one in their shell sees the
same results as CI.

TestTheThreadCount
^^^^^^^^^^^^^^^^^^

How many threads, and where the number comes from.

``test_it_defaults_to_the_physical_core_count``
   Not the logical count — hyperthreads hurt an FDTD kernel.

``test_the_default_is_written_to_the_environment``
   Undocumented.

``test_an_explicit_count_is_used``
   Undocumented.

``test_an_explicit_count_is_written_to_the_environment``
   Undocumented.

``test_an_explicit_count_overrides_the_environment``
   ``-n 2`` on the command line beats an inherited variable.

``test_an_inherited_count_is_respected``
   A scheduler or MPI launcher may have set it deliberately.

``test_an_inherited_count_is_returned_as_an_integer``
   It arrives as a string; callers do arithmetic with it.

``test_an_inherited_count_is_not_rewritten``
   Undocumented.

``test_a_count_of_zero_falls_through_to_the_default``
   ``if nthreads:`` is a truthiness test, so ``0`` is not "no threads".

   Arguably right — zero OpenMP threads is meaningless — but it means the
   argument cannot be used to request a serial run.

``test_an_empty_inherited_value_falls_through_to_the_default``
   ``os.environ.get`` returns ``""``, which is falsy.

   Without this branch the ``int("")`` would raise.

``test_the_physical_core_count_is_read_from_the_global_config``
   Not probed again — the value ``get_host_info`` found at startup.

TestTheFixedSettings
^^^^^^^^^^^^^^^^^^^^

Three variables set on every platform, for every run.

``test_dynamic_thread_adjustment_is_disabled``
   The runtime must not shrink the team mid-solve.

   An FDTD timestep is a fixed amount of work; a varying team size makes
   per-iteration timings meaningless and can leave cores idle.

``test_places_are_cores``
   Undocumented.

``test_threads_are_bound_to_their_places``
   Without binding the OS migrates threads and destroys cache locality.

``test_the_three_are_set_regardless_of_the_thread_count``
   They precede the ``nthreads`` branching, so no path skips them.

TestTheWaitPolicyOnMacOs
^^^^^^^^^^^^^^^^^^^^^^^^

``OMP_WAIT_POLICY`` — set only on ``darwin``, and only there.

``test_apple_silicon_gets_a_passive_wait``
   Apple's tuning guide: spinning threads steal power budget.

   On an efficiency-core design a busy-wait does not merely waste a core, it
   lowers the clock available to the cores doing real work.

``test_an_intel_mac_gets_an_active_wait``
   Spinning is the faster choice on a conventional x86 Mac.

``test_the_branch_is_chosen_by_a_substring_of_the_cpu_name``
   ``"Apple" in cpuID`` — the only signal available.

``test_an_unknown_cpu_gets_the_active_wait``
   The case that actually occurs on Apple silicon.

   ``machdep.cpu.brand_string`` does not exist on M-series chips, so
   ``get_host_info`` leaves ``cpuID`` as ``"unknown"`` — and this function
   then chooses ``ACTIVE``, the opposite of what the hardware wants. The two
   defects compound; written up together in ``notes/bugs/host-info-apple-
   silicon-cpuid.md``.

TestTheWaitPolicyElsewhere
^^^^^^^^^^^^^^^^^^^^^^^^^^

Not set at all on Windows or Linux — the default is left alone.

``test_it_is_not_set``
   Undocumented.

``test_an_inherited_value_is_left_untouched``
   Undocumented.

TestWindowsSubsystemForLinux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Binding hangs under WSL, so the binding variables are removed again.

``test_affinity_is_disabled``
   microsoft/WSL#785 — thread affinity is not implemented there.

``test_the_places_variable_is_removed``
   Set unconditionally a few lines earlier, then deleted here.

``test_the_binding_variable_is_removed``
   Undocumented.

``test_the_thread_count_is_still_set``
   Only the binding is dropped; parallelism is not.

``test_dynamic_adjustment_is_still_disabled``
   Undocumented.

``test_the_detection_is_a_substring_of_the_os_version``
   ``"Microsoft" in osversion`` — capital M, and case-sensitive.

   WSL2 kernels report ``microsoft-standard-WSL2`` in lower case, so this
   check misses them entirely. That is arguably correct — WSL2 is a real VM
   and does support affinity — but it is accidental rather than intended.
   Recorded in the analogy doc's observations table.

``test_an_ordinary_linux_keeps_its_binding``
   Undocumented.

TestEnvironmentHygiene
^^^^^^^^^^^^^^^^^^^^^^

The function's whole effect is a side effect; pin its extent.

``test_exactly_five_variables_are_written``
   A sixth would be a silent change to how the solver runs.

``test_nothing_unrelated_is_removed``
   Undocumented.

``test_calling_twice_is_idempotent``
   The context loop configures once per model in a multi-model run.

``test_the_second_call_sees_its_own_first_call``
   A subtle consequence: the first call sets ``OMP_NUM_THREADS``.

   So the second takes the *inherited* branch rather than recomputing from
   the core count. Identical result here, but it means an explicit argument
   on a later call is the only way to change the number.

When these fail
~~~~~~~~~~~~~~~

**A result differs between your machine and CI.** ``set_omp_threads``
branches on whether ``OMP_NUM_THREADS`` is already set, so a shell export
changes the path taken. The autouse ``clean_omp_environment`` fixture
deletes six variables before every test.

**An environment variable leaks into a later test.** Every write goes
through ``monkeypatch``, which restores ``os.environ`` afterwards.
``test_exactly_four_variables_are_written`` pins the full extent of the
function's side effects — a fifth would be a silent change to how the solver
runs.

**A thread count is the logical core count.** It should be the **physical**
one. Hyperthreads share an execution unit, so two OpenMP threads on one core
make an FDTD kernel slower. This is the most performance-relevant line in
the utilities package.

**A wait-policy test fails on Apple silicon.** The branch tests ``"Apple" in
cpuID``, and on M-series chips ``cpuID`` is ``"unknown"`` — the sysctl key
the probe reads does not exist there. So the one branch written for Apple
silicon is the one it cannot reach. See ``notes/bugs/host-info-apple-
silicon-cpuid.md``.

Test Catalog — ``test_mem_checks.py``
-------------------------------------

**35 tests** from 35 test functions across 5 classes.

The memory estimates gprMax makes before it allocates anything.

An FDTD grid is a dozen dense arrays sized ``(nx+1)(ny+1)(nz+1)``. A model
that will not fit fails with a ``MemoryError`` from inside NumPy, minutes
into a build, with no indication of which grid was too large. These
functions exist to say so first, in bytes the user recognises.

Three things are being tested, and they are different in kind:

**Arithmetic.** ``mem_check_run_all`` and ``mem_check_build_all`` accumulate
into two places at once — ``get_model_config().mem_use``, which is per
*model* and spans every grid including subgrids, and ``grid.mem_use``, which
is per grid. Getting one and not the other is the obvious failure, and it is
silent: the warning threshold is simply never reached.

**Conditionals.** Dispersive materials add coefficient arrays; snapshots add
a field copy each; fractal volumes add complex arrays during construction
only. Each is counted only when present, which is four branches whose
absence costs nothing and whose presence can double the estimate.

**A warning that must fire.** ``mem_check_host`` compares against the RAM
figure ``get_host_info`` found at startup. It is the last thing standing
between a user and an out-of-memory kill.

The grids here are ``SimpleNamespace`` stand-ins with four method
attributes. A real ``FDTDGrid`` would allocate the very arrays these
functions are trying to predict, which would make the tests both slow and
circular.

TestMemCheckHost
^^^^^^^^^^^^^^^^

``mem_check_host`` — one comparison, one warning.

``test_a_small_request_is_silent``
   Undocumented.

``test_an_oversized_request_warns``
   Undocumented.

``test_the_requested_amount_appears_in_the_warning``
   Humanised, so the user can compare it with what they have.

``test_the_available_amount_appears_in_the_warning``
   Undocumented.

``test_exactly_the_available_amount_does_not_warn``
   A strict ``>``; using every last byte is allowed.

``test_one_byte_over_warns``
   Undocumented.

``test_it_only_warns_and_never_raises``
   The user may know something the estimate does not — swap, or a machine
   whose RAM was mis-detected. The run is not blocked.

``test_the_limit_comes_from_the_global_host_info``
   Not probed live, so a long run is judged against startup figures.

   Shown by lowering the recorded RAM to an absurd figure: a request of two
   kilobytes then warns, which no real machine would provoke.

TestMemCheckRunAll
^^^^^^^^^^^^^^^^^^

``mem_check_run_all`` — the estimate made just before solving.

``test_the_basic_estimate_is_added_to_the_model_total``
   Undocumented.

``test_the_basic_estimate_is_added_to_the_grid_total``
   Both tallies are kept; the per-grid one is what the banner prints.

``test_the_model_total_is_returned``
   Undocumented.

``test_every_grid_contributes``
   Subgrids are separate ``FDTDGrid`` objects in the same list.

``test_one_string_is_produced_per_grid``
   Undocumented.

``test_each_string_names_its_grid``
   Undocumented.

``test_each_string_is_humanised``
   Undocumented.

``test_an_empty_grid_list_returns_the_existing_total``
   The 65 MB interpreter overhead ``ModelConfig`` starts with.

TestDispersiveMaterials
^^^^^^^^^^^^^^^^^^^^^^^

The dispersive coefficient arrays, counted only when poles exist.

``test_no_poles_means_no_extra_memory``
   Undocumented.

``test_poles_add_the_dispersive_estimate``
   Undocumented.

``test_the_grid_tally_is_also_increased``
   Undocumented.

``test_the_pole_count_is_read_once_per_model_not_per_grid``
   ``maxpoles`` is a model-wide property; subgrids share it.

TestSnapshots
^^^^^^^^^^^^^

Snapshots are copies of the field arrays, held until they are written.

``test_no_snapshots_add_nothing``
   Undocumented.

``test_each_snapshot_is_counted``
   Undocumented.

``test_the_grid_tally_is_also_increased``
   Undocumented.

``test_snapshots_across_grids_are_all_counted``
   Undocumented.

``test_a_snapshot_heavy_model_can_trigger_the_host_warning``
   The reason snapshots are counted at all.

   The field arrays fit; the hundred copies of them do not.

TestMemCheckBuildAll
^^^^^^^^^^^^^^^^^^^^

``mem_check_build_all`` — the estimate made before geometry is built.

``test_the_basic_estimate_is_counted``
   Undocumented.

``test_fractal_volumes_add_their_estimate``
   Fractal construction needs complex arrays the solve never sees.

``test_no_fractal_volumes_means_no_extra_memory``
   Undocumented.

``test_it_does_not_mutate_the_model_tally``
   Unlike ``mem_check_run_all``, the build estimate is transient.

   Build-time memory is released before the solve; folding it into
   ``mem_use`` would double-count against the run-time check that follows.

``test_it_does_not_mutate_the_grid_tally``
   Undocumented.

``test_dispersive_materials_are_not_counted``
   They do not exist yet — materials are assigned after geometry.

``test_one_string_is_produced_per_grid``
   Undocumented.

``test_each_string_reports_only_its_own_grid``
   A running total here would misattribute the second grid's size.

``test_an_oversized_build_warns``
   Undocumented.

``test_an_empty_grid_list_returns_the_existing_total``
   Undocumented.

When these fail
~~~~~~~~~~~~~~~

**A memory tally is right in one place and wrong in the other.** Both
``mem_check_run_all`` and the grid objects keep a total:
``get_model_config().mem_use`` spans every grid in the model including
subgrids, and ``grid.mem_use`` is per grid. Updating one and not the other
is silent — the warning threshold is simply never reached.

**A build estimate leaks into the run estimate.** It must not.
``mem_check_build_all`` deliberately does *not* mutate either tally: build-
time memory is released before the solve, so folding it in would double-
count against the run-time check that follows.

**A humanised string does not match.** ``humanize.naturalsize(n)`` is
decimal (MB, GB) and ``naturalsize(n, True)`` is binary (MiB, GiB). The two
are used in different places on purpose — required memory in decimal,
detected memory in binary — and the strings in these assertions follow the
source.

**The grids look too simple.** They are ``SimpleNamespace`` stand-ins with
four method attributes. A real ``FDTDGrid`` would allocate the very arrays
these functions are trying to predict, making the tests both slow and
circular.

**``mem_check_device_snaps`` is untested.** Two of its four solver cases
leave ``device_mem`` unbound, so a test would be asserting the bug. See
``notes/bugs/mem-check-device-snaps-no-else.md`` and ``notes/bugs/mem-check-
run-all-precedence.md`` — the two must be fixed together.

Test Catalog — ``test_device_detection.py``
-------------------------------------------

**58 tests** from 50 test functions across 7 classes.

Finding, describing and reporting compute devices.

gprMax can solve on a CPU, on CUDA through pycuda, on OpenCL through
pyopencl, or on Apple Metal through pyobjc. Three of those four backends are
optional imports, and the code that finds them follows the same shape each
time: a ``has_*`` predicate that swallows ``ImportError``, a ``detect_*``
that returns a dictionary of device objects keyed by ID, and a
``print_*_info`` that turns that dictionary into the indented tree the user
sees at startup.

**The testing problem, and the way round it.** None of the three packages is
installed in the test environment, and installing them would not help:
pycuda imports fine on a machine with no NVIDIA card and then fails at
``drv.init()``, while pyopencl needs an ICD loader. So the modules are
*fabricated* — ``fake_module`` inserts a ``ModuleType`` into
``sys.modules``, and the ``import pycuda.driver as drv`` inside the function
under test picks it up like any other import. That is what makes the success
paths reachable at all; without it only the "not installed" branch could
ever be tested, which is the branch that matters least.

The fabricated modules are deliberately minimal — a device object here has a
``name`` and a memory figure and nothing else, because that is all the code
touches. Anything more would be inventing an API contract that the real
libraries define.

Two of the tests pin defects. ``print_opencl_info`` leaves a local unbound
for any device that is neither CPU nor GPU, and ``detect_metal`` stores a
``None`` device when the Metal framework reports no hardware. Both are
silent in the common case, which is exactly why they need writing down.

TestTheAvailabilityPredicates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``has_pycuda`` / ``has_pyopencl`` / ``has_metal`` — importable or not.

``test_an_installed_module_is_reported_present``
   Truthiness, not identity — see the shadowing test below.

``test_two_of_the_three_return_the_module_instead_of_true``
   ``import pycuda`` rebinds the local that held ``True``.

   Each predicate sets a local to ``True``, then does ``import <name>``
   *inside the same function* — and the import statement binds the module to
   that same name. So ``has_pycuda`` and ``has_pyopencl`` return a module
   object on success. ``has_metal`` escapes only by accident: its local is
   ``metal`` and the module is ``Metal``, and the case differs.

   Harmless today, because every caller uses the result in an ``if``. It
   matters the moment one is compared with ``True`` or serialised. Written
   up in ``notes/bugs/host-info-has-predicates-shadowing.md``.

``test_a_missing_module_is_reported_absent``
   Undocumented.

``test_a_missing_module_does_not_raise``
   The whole point — these are called unconditionally at startup.

   A user with none of the three optional backends must still be able to run
   gprMax on a CPU.

``test_the_result_is_a_boolean``
   Callers use it in ``if``; a truthy module object would also work, but the
   annotation-free signature makes the type worth pinning.

TestDetectCudaGpus
^^^^^^^^^^^^^^^^^^

``detect_cuda_gpus`` — pycuda's device list, keyed by device ID.

``test_no_pycuda_returns_an_empty_dictionary``
   Undocumented.

``test_no_pycuda_warns_with_installation_instructions``
   The user asked for ``-gpu``; silence would be unhelpful.

``test_every_device_is_returned``
   Undocumented.

``test_devices_are_keyed_by_their_identifier``
   Undocumented.

``test_the_driver_is_initialised``
   ``drv.init()`` must run before any other pycuda call.

``test_no_devices_warns``
   pycuda installed but no card — a common misconfiguration.

``test_no_devices_returns_an_empty_dictionary``
   Undocumented.

``test_the_visible_devices_variable_restricts_the_list``
   Schedulers set ``CUDA_VISIBLE_DEVICES``; gprMax must honour it.

   Ignoring it on a shared node would mean grabbing another job's GPU.

``test_a_single_visible_device_is_parsed``
   Undocumented.

``test_the_visible_devices_variable_is_ignored_when_there_are_none``
   The zero-device check comes first, so the warning still fires.

TestDetectOpencl
^^^^^^^^^^^^^^^^

``detect_opencl`` — a flat dictionary across all platforms.

``test_no_pyopencl_returns_an_empty_dictionary``
   Undocumented.

``test_no_pyopencl_warns_with_installation_instructions``
   Undocumented.

``test_every_device_is_returned``
   Undocumented.

``test_devices_are_numbered_from_zero``
   Undocumented.

``test_devices_across_platforms_share_one_numbering``
   A machine with an integrated GPU and a discrete one has two platforms;
   the user selects a device by a single ID, not a pair.

``test_no_platforms_returns_an_empty_dictionary``
   Undocumented.

``test_a_failing_platform_query_warns``
   A bare ``except`` — an ICD loader that raises is not fatal.

``test_a_failing_platform_query_returns_an_empty_dictionary``
   Undocumented.

TestDetectMetal
^^^^^^^^^^^^^^^

``detect_metal`` — one device, or a placeholder for one.

``test_no_metal_returns_an_empty_dictionary``
   Undocumented.

``test_no_metal_warns_with_installation_instructions``
   Undocumented.

``test_the_system_default_device_is_returned``
   Undocumented.

``test_it_is_keyed_at_zero``
   Metal exposes one system default; there is no device list.

``test_no_hardware_still_produces_an_entry``
   ``MTLCreateSystemDefaultDevice`` returns ``None`` with no GPU.

   The return value is stored unconditionally, so the dictionary is non-
   empty and truthy while containing nothing usable. Anything that later
   tests ``if devs:`` is misled, and the failure surfaces as an
   ``AttributeError`` on ``None`` inside ``print_metal_info``. Pinned as the
   current behaviour; written up in ``notes/bugs/host-info-detect-metal-
   stores-none.md``.

TestPrintCudaInfo
^^^^^^^^^^^^^^^^^

``print_cuda_info`` — one line per GPU, under a heading.

``test_a_heading_is_printed``
   Undocumented.

``test_the_device_name_is_printed``
   Undocumented.

``test_the_device_identifier_is_printed``
   The number the user passes to ``-gpu``.

``test_the_memory_is_humanised``
   Undocumented.

``test_internal_whitespace_in_the_name_is_collapsed``
   Driver-reported names are padded.

``test_one_line_per_device``
   Undocumented.

``test_an_empty_dictionary_prints_only_the_heading``
   Undocumented.

TestPrintOpenclInfo
^^^^^^^^^^^^^^^^^^^

``print_opencl_info`` — devices grouped under their platform.

``test_a_heading_is_printed``
   Undocumented.

``test_the_platform_name_is_printed``
   Undocumented.

``test_the_device_type_is_printed``
   Undocumented.

``test_a_cpu_device_is_labelled_as_such``
   OpenCL on a CPU is a supported gprMax configuration.

``test_the_device_name_and_memory_are_printed``
   Undocumented.

``test_one_platform_heading_for_several_devices_on_it``
   The grouping is the reason the loop tracks the previous platform.

``test_a_second_platform_gets_its_own_heading``
   Undocumented.

``test_an_empty_dictionary_prints_only_the_heading``
   Undocumented.

``test_a_device_that_is_neither_cpu_nor_gpu_raises``
   ``type`` is assigned in two ``if``s with no ``else``.

   An FPGA or a custom accelerator reports neither ``"CPU"`` nor ``"GPU"``,
   so the local is never bound and the line that formats it raises
   ``UnboundLocalError`` — while merely *listing* devices, before anything
   has been selected. Worse, if a CPU or GPU device was printed first, the
   stale value is silently reused and the accelerator is mislabelled. Pinned
   as the current behaviour; written up in ``notes/bugs/host-info-print-
   opencl-unbound-type.md``.

TestPrintMetalInfo
^^^^^^^^^^^^^^^^^^

``print_metal_info`` — the shortest of the three.

``test_a_heading_is_printed``
   Undocumented.

``test_the_device_name_is_printed``
   Undocumented.

``test_the_device_identifier_is_printed``
   Undocumented.

``test_internal_whitespace_in_the_name_is_collapsed``
   Undocumented.

``test_no_memory_figure_is_reported``
   Unlike CUDA and OpenCL — Metal uses unified memory, so a per-device
   figure would repeat the host's RAM.

``test_it_logs_at_the_basic_level``
   Undocumented.

When these fail
~~~~~~~~~~~~~~~

**A ``has_*`` predicate does not return ``True``.** ``has_pycuda`` and
``has_pyopencl`` return the **module**: ``import pycuda`` rebinds the local
that held ``True``. ``has_metal`` escapes only because its local is
``metal`` and the module is ``Metal``. See ``notes/bugs/host-info-has-
predicates-shadowing.md``.

**A fabricated module leaks into another test.** ``fake_module`` uses
``monkeypatch.setitem`` on ``sys.modules``, so it is removed afterwards. A
fabricated ``pycuda`` surviving the test would make every later
``has_pycuda`` call return ``True`` on a machine without it.

**``import pycuda.driver as drv`` does not see the fake.** Submodules must
be registered under their dotted name *and* as an attribute of the parent,
because that statement consults both. ``fake_module`` does both.

**``UnboundLocalError`` from ``print_opencl_info``.** The device is neither
CPU nor GPU. Worse, if a CPU or GPU was printed first, the stale label is
silently reused and the accelerator is mislabelled. See ``notes/bugs/host-
info-print-opencl-unbound-type.md``.

**``detect_metal`` returns a non-empty dictionary with nothing in it.**
``MTLCreateSystemDefaultDevice()`` returns ``None`` when there is no
hardware, and the return value is stored unconditionally. See
``notes/bugs/host-info-detect-metal-stores-none.md``.

Test Catalog — ``test_vtkhdf_base.py``
--------------------------------------

**64 tests** from 64 test functions across 9 classes.

``VtkHdfFile`` — opening, naming, marking and closing a VTKHDF file.

Before any data is written, four things have to be right or VTK will refuse
the file, or worse, misread it:

- the **file name** must end in ``.vtkhdf``;

- there must be a group called ``VTKHDF`` at the root;

- that group must carry a ``Version`` attribute VTK recognises;

- and a ``Type`` attribute naming the dataset class, stored as *fixed-length
  ASCII* — not a Python string, not UTF-8.

The last of those is the subtle one. ``h5py`` will happily write a variable
length UTF-8 string for a Python ``str``, and the file will open in ``h5py``
and look correct in an HDF5 viewer, while VTK reads nothing. The constructor
therefore encodes the type and hands ``h5py`` an explicit fixed-length ASCII
dtype. That is asserted here by dtype, not by value.

The extension rewriting deserves its own attention. ``Path.with_suffix``
replaces everything after the *last* dot, so a name that contains a version
number or a dimension loses part of itself — ``model_1.5`` becomes
``model_1.vtkhdf``. The warning fires for the extension it replaced, not for
the characters it dropped. Both are pinned below.

TestClassConstants
^^^^^^^^^^^^^^^^^^

The format constants, which are also the compatibility contract.

``test_the_version_is_two_two``
   VTKHDF 2.2 — the version whose layout this writer implements.

``test_the_extension_is_vtkhdf``
   Undocumented.

``test_the_root_group_is_named_vtkhdf``
   VTK looks for this exact group; nothing else in the file matters.

``test_the_two_required_attributes_are_named``
   Undocumented.

``test_the_file_type_enum_has_two_members``
   ImageData for voxels, UnstructuredGrid for lines.

``test_the_file_type_enum_members_are_strings``
   ``VtkFileType`` subclasses ``str``, so ``.encode`` works directly.

   The constructor relies on that: ``vtk_file_type.encode("ascii")``.

TestFileNaming
^^^^^^^^^^^^^^

The extension is rewritten, not merely checked.

``test_a_correct_extension_is_kept``
   Undocumented.

``test_a_wrong_extension_is_replaced``
   Undocumented.

``test_a_missing_extension_is_added``
   Undocumented.

``test_a_wrong_extension_warns``
   Undocumented.

``test_the_offending_extension_is_named_in_the_warning``
   Undocumented.

``test_a_missing_extension_does_not_warn``
   Nothing was replaced, so there is nothing to report.

``test_a_correct_extension_does_not_warn``
   Undocumented.

``test_the_directory_is_preserved``
   Undocumented.

``test_the_file_is_written_where_the_name_says``
   Undocumented.

``test_a_dot_in_the_name_truncates_it``
   ``with_suffix`` replaces everything after the last dot.

   A snapshot called ``model_1.5`` — a perfectly ordinary name for a 1.5 ns
   time point — is written as ``model_1.vtkhdf``, and the next snapshot at
   ``model_1.7`` overwrites it. The warning that fires says the extension
   ``'.5'`` was invalid, which is technically true and entirely unhelpful.
   Pinned as the current behaviour; written up in ``notes/bugs/vtkhdf-
   filename-suffix-truncation.md``.

``test_the_filename_is_stored_as_a_path``
   Callers read ``handler.filename`` to log where output went.

TestRootGroup
^^^^^^^^^^^^^

The ``VTKHDF`` group, and the two attributes VTK requires on it.

``test_the_root_group_exists_on_disk``
   Undocumented.

``test_the_root_group_is_a_group``
   Undocumented.

``test_the_version_attribute_is_written``
   Undocumented.

``test_the_version_attribute_is_integral``
   VTK parses it as a pair of integers, not a string.

``test_the_type_attribute_is_written``
   Undocumented.

``test_the_type_attribute_is_bytes_not_a_string``
   The whole reason for the explicit dtype.

   A Python ``str`` would be stored as variable-length UTF-8, which VTK does
   not read.

``test_the_type_attribute_is_fixed_length_ascii``
   Length exactly ``len("ImageData")`` — 9 bytes, no terminator.

``test_the_unstructured_type_gets_its_own_length``
   16 bytes for ``UnstructuredGrid`` — the length is per-value.

``test_exactly_two_attributes_are_written``
   A third would be non-standard and might confuse a reader.

``test_no_datasets_are_written_by_the_base_constructor``
   Undocumented.

TestExistingFiles
^^^^^^^^^^^^^^^^^

Reopening — the attributes are checked rather than rewritten.

``test_an_existing_file_is_truncated_in_write_mode``
   Undocumented.

``test_a_matching_version_does_not_warn``
   Undocumented.

``test_a_mismatched_version_warns``
   A file written by a future gprMax should not be silently trusted.

``test_a_readonly_file_missing_the_attributes_warns``
   Read-only mode cannot repair the file, so it reports instead.

``test_a_readonly_file_does_not_gain_the_attributes``
   Undocumented.

``test_a_readwrite_file_gains_the_attributes``
   Mode ``r+`` is the one branch that repairs a file in place.

``test_a_missing_file_in_read_mode_raises``
   Undocumented.

``test_the_root_group_is_created_if_absent``
   ``require_group``, so an empty HDF5 file is upgraded rather than
   rejected.

TestTheContextManager
^^^^^^^^^^^^^^^^^^^^^

``with VtkHdfFile(...) as f:`` — the intended way to use it.

``test_entering_returns_the_handler``
   Undocumented.

``test_exiting_closes_the_file``
   Undocumented.

``test_it_closes_even_when_the_body_raises``
   Otherwise a failed export would leave a locked, truncated file.

``test_it_does_not_suppress_exceptions``
   ``__exit__`` returns ``None``, so the error still propagates.

``test_the_data_is_readable_after_exit``
   The point of closing: buffers are flushed.

TestClose
^^^^^^^^^

``close`` — also callable directly, and more than once.

``test_it_closes_the_handle``
   Undocumented.

``test_calling_it_twice_is_harmless``
   h5py treats closing a closed file as a no-op, so the ``__exit__`` of a
   handler that was already closed cannot raise.

``test_closing_inside_a_context_is_harmless``
   Undocumented.

``test_writing_after_close_raises``
   Undocumented.

TestDatasetPaths
^^^^^^^^^^^^^^^^

``_build_dataset_path`` and ``_get_dataset`` — the plumbing beneath.

``test_a_single_component_hangs_off_the_root``
   Undocumented.

``test_several_components_are_joined``
   Undocumented.

``test_no_components_gives_the_root_group``
   Undocumented.

``test_an_existing_dataset_is_returned``
   Undocumented.

``test_a_missing_dataset_raises_a_key_error``
   Undocumented.

``test_the_missing_path_message_is_never_the_one_reported``
   ``h5py`` returns ``None`` for an absent path, not ``"default"``.

   So the ``cls == "default"`` branch — and its clear "Path does not exist"
   message — is unreachable, and a simple typo in a dataset name surfaces as
   ``Dataset not found. Found 'None' instead``, which reads like a type
   problem rather than a missing key. Pinned as the current behaviour;
   written up in ``notes/bugs/vtkhdf-unreachable-missing-path-branch.md``.

``test_a_path_pointing_at_a_group_raises``
   Asking for the root group as a dataset is a caller error.

   The message names what was found, which is the difference between a typo
   and a structural mistake.

TestRootAttributes
^^^^^^^^^^^^^^^^^^

``_set_root_attribute`` / ``_get_root_attribute`` / ``_has_...``.

``test_an_attribute_can_be_set_and_read_back``
   Undocumented.

``test_setting_an_attribute_twice_replaces_it``
   Undocumented.

``test_an_explicit_dtype_is_honoured``
   ``WholeExtent`` must be integral or VTK reads garbage extents.

``test_a_present_attribute_is_reported_present``
   Undocumented.

``test_an_absent_attribute_is_reported_absent``
   Undocumented.

``test_reading_an_absent_attribute_raises``
   Undocumented.

``test_the_error_names_the_group``
   So a user can tell a missing attribute from a missing group.

``test_attributes_persist_to_disk``
   Undocumented.

TestCellTypes
^^^^^^^^^^^^^

``VtkCellType`` — the numeric codes VTK defines, mirrored here.

``test_the_line_type_is_three``
   The only one gprMax writes: geometry views in line mode.

``test_the_voxel_type_is_eleven``
   Undocumented.

``test_the_values_are_the_vtk_ones``
   Copied from ``vtkCellType.h``; a drift here is a silent corruption of
   every file written, since VTK trusts the number.

``test_the_members_are_unsigned_bytes``
   The ``Types`` dataset is written as ``uint8``.

``test_a_cell_type_can_be_used_as_a_number``
   Undocumented.

When these fail
~~~~~~~~~~~~~~~

**A ``Type`` attribute assertion fails on the dtype rather than the value.**
That is the assertion that matters. ``h5py`` will happily store a Python
``str`` as variable-length UTF-8, and the file will open in ``h5py`` and
look correct in an HDF5 viewer while VTK reads nothing. The constructor
hands it an explicit fixed-length ASCII dtype — ``S9`` for ``ImageData``,
``S16`` for ``UnstructuredGrid``.

**A file name is shorter than requested.** ``with_suffix`` replaces
everything after the last dot, so ``model_1.5`` becomes ``model_1.vtkhdf``
and the warning reports the extension ``'.5'`` as invalid. See
``notes/bugs/vtkhdf-filename-suffix-truncation.md``.

**A ``KeyError`` message is not the one you expected.** ``h5py`` returns
``None`` for a missing path, not the string ``"default"``, so the clear
"Path does not exist" branch is unreachable and a typo reports ``Found
'None' instead``. See ``notes/bugs/vtkhdf-unreachable-missing-path-
branch.md``.

**An attribute is not written.** ``_check_root_attribute`` only *writes*
when ``file_handler.mode == "r+"``. h5py reports ``"r+"`` for files opened
``"w"``, ``"a"`` and ``"r+"``; a file opened ``"r"`` gets a warning instead.
That is the one branch where the constructor reports rather than repairs.

Test Catalog — ``test_vtkhdf_datasets.py``
------------------------------------------

**55 tests** from 55 test functions across 6 classes.

``_write_dataset`` — everything that actually puts bytes in the file.

One method, about a hundred lines, and every dataset in every VTKHDF file
gprMax writes goes through it. It does four separable jobs, and the tests
below are grouped by them:

**Coercion.** Anything array-like is accepted — a list, a scalar, a NumPy
array — and a scalar is expanded to shape ``(1,)`` so HDF5 has something to
store. This is why ``NumberOfCells`` can be written as a plain ``int``.

**String conversion.** HDF5 has no UTF-32, so NumPy's ``'U'`` dtype cannot
be stored, and VTKHDF only reads *variable-length ASCII*. Any string data is
therefore rewritten to that one representation, and the two ways of asking
for something else — a fixed length, or UTF-8 — each produce a warning
explaining what happened instead. Three of the four warning paths only
trigger when the caller passes an explicit ``dtype``, which is why the tests
do so.

**The transpose.** VTKHDF stores arrays in ZYX order; gprMax works in XYZ.
``_write_dataset`` therefore calls ``data.transpose()`` — which reverses
*all* axes — and flips ``shape`` and ``offset`` to match. This is the single
most consequential line in the package for anyone reading the output: get it
wrong and the model loads mirrored, with no error anywhere.
``xyz_data_ordering`` defaults to ``True``, and **no public method exposes
it** — only ``add_field_data`` sets it to ``False``, internally. Its
consequences for 1-D and 2-D data are pinned here because they are not
obvious: a 1-D transpose is a no-op, and a 2-D one swaps rows and columns.

**Partial writes.** ``shape`` plus ``offset`` place a block inside a larger
dataset, which is how MPI ranks each write their own slab. Three separate
validation errors guard it, and each is asserted on its message, because the
message is the only thing a user has to work with when a rank's arithmetic
is wrong.

Everything is verified by reopening the file. Where a test asserts on
layout, it reads the raw HDF5 array — *not* transposed back — because the
on-disk order is exactly the thing being pinned.

TestScalarsAndSequences
^^^^^^^^^^^^^^^^^^^^^^^

Coercion of anything array-like into something HDF5 can store.

``test_a_list_is_accepted``
   Undocumented.

``test_a_scalar_becomes_a_one_element_dataset``
   ``NumberOfCells`` is written this way — a bare Python ``int``.

``test_the_scalar_value_survives``
   Undocumented.

``test_an_array_is_written_unchanged``
   Undocumented.

``test_the_dtype_is_deduced_from_the_data``
   Undocumented.

``test_an_explicit_dtype_overrides_the_data``
   Undocumented.

``test_floating_point_data_keeps_its_precision``
   Snapshots are ``float32``; promoting them would double file size.

``test_an_empty_array_is_written``
   A model with no cells of a given type is legitimate.

``test_a_duplicate_dataset_raises``
   ``create_dataset`` will not overwrite; the second write fails.

   Relevant to any caller that names two datasets alike — geometry views and
   snapshots share a file in some configurations.

``test_a_nested_path_creates_intermediate_groups``
   ``CellData/Material`` needs a ``CellData`` group that nothing made.

TestTheTranspose
^^^^^^^^^^^^^^^^

ZYX on disk, XYZ in memory — reversal of *all* axes.

``test_a_three_dimensional_array_is_reversed``
   The shape on disk is the reverse of the shape in memory.

   Asserted on the raw HDF5 dataset, because that is what VTK reads.

``test_the_values_land_where_the_reversal_says``
   Shape alone would pass for a wrong permutation; values pin it.

``test_the_round_trip_recovers_the_original``
   A reader that transposes back gets exactly what was written.

``test_a_one_dimensional_array_is_unaffected``
   Transposing a vector is a no-op, so scalars and IDs are safe.

``test_a_two_dimensional_array_is_swapped``
   An ``(N, 3)`` array of vectors becomes ``(3, N)`` on disk.

   VTK expects ``(nTuples, nComponents)``, so any caller writing vector data
   through the default path gets it stored the wrong way round. This is why
   ``Points`` is written with ``xyz_data_ordering=False``.

``test_it_can_be_switched_off``
   The escape hatch ``Points`` and field data use.

``test_it_is_on_by_default``
   Stated explicitly: a caller who says nothing gets the transpose.

``test_no_public_method_exposes_the_switch``
   The transpose cannot be turned off from outside the package.

   ``add_point_data``, ``add_cell_data`` and the ``VtkImageData`` /
   ``VtkUnstructuredGrid`` constructors take no such argument, so every
   externally written dataset is transposed. Worth pinning: it means the 2-D
   swap above is not an edge case a caller can avoid.

TestStringData
^^^^^^^^^^^^^^

VTKHDF reads variable-length ASCII, and nothing else.

``test_a_list_of_strings_is_written``
   Undocumented.

``test_strings_are_stored_as_variable_length``
   Fixed-length padding would be read by VTK as trailing nulls.

``test_strings_are_stored_as_ascii``
   Undocumented.

``test_a_single_string_becomes_a_one_element_dataset``
   Undocumented.

``test_an_explicit_unicode_dtype_warns``
   HDF5 has no UTF-32; the conversion is silent otherwise.

``test_an_explicit_unicode_dtype_still_writes_ascii``
   Undocumented.

``test_a_utf8_string_dtype_warns``
   Undocumented.

``test_a_fixed_length_string_dtype_warns``
   Serial I/O converts to variable length and says so.

``test_a_fixed_length_string_dtype_is_converted``
   Undocumented.

``test_a_bytes_array_is_also_converted``
   ``'S'`` dtype takes the same path as ``'U'``.

``test_a_plain_string_array_does_not_warn``
   The warnings only fire for an explicitly requested dtype.

   Passing a ``'U'`` array without naming a dtype converts silently — the
   common case, and arguably the one most worth reporting.

TestPartialWrites
^^^^^^^^^^^^^^^^^

``shape`` plus ``offset`` — one rank's slab inside the whole dataset.

``test_the_full_dataset_is_created``
   Undocumented.

``test_the_data_lands_at_the_offset``
   Undocumented.

``test_the_rest_of_the_dataset_is_zero``
   Ranks that never write leave zeros, not uninitialised memory.

``test_a_shape_equal_to_the_data_needs_no_offset``
   The single-rank case: shape is given but describes the whole thing.

``test_a_larger_shape_without_an_offset_raises``
   Undocumented.

``test_a_shape_of_the_wrong_rank_raises``
   Undocumented.

``test_an_offset_of_the_wrong_rank_raises``
   Undocumented.

``test_data_that_overruns_the_dataset_raises``
   The MPI failure mode: a rank computed its offset wrongly.

``test_the_overrun_message_shows_the_arithmetic``
   ``[2] + (2,) = [4] > [3]`` — enough to debug without a rerun.

``test_a_two_dimensional_partial_write_places_correctly``
   Both ``shape`` and ``offset`` are flipped alongside the data.

``test_the_offset_is_flipped_with_the_data``
   Stated as its own test: the flip must apply to all three, or the block
   lands in the wrong corner of a correctly shaped dataset.

``test_a_partial_write_without_the_transpose_uses_the_given_order``
   Undocumented.

TestCreateDataset
^^^^^^^^^^^^^^^^^

``_create_dataset`` — reserve space without writing anything.

Used when some MPI ranks have no data but must still take part in the
collective creation call.

``test_a_dataset_of_the_requested_shape_is_created``
   Undocumented.

``test_it_is_filled_with_zeros``
   Undocumented.

``test_the_requested_dtype_is_used``
   Undocumented.

``test_a_unicode_dtype_warns``
   Undocumented.

``test_a_unicode_dtype_becomes_variable_length_ascii``
   Undocumented.

``test_a_utf8_dtype_warns``
   Undocumented.

``test_no_data_and_no_shape_raises``
   Undocumented.

``test_no_data_and_no_dtype_raises``
   Undocumented.

TestFieldData
^^^^^^^^^^^^^

``add_field_data`` — the only public writer on the base class.

``test_it_writes_under_the_field_data_group``
   Undocumented.

``test_it_does_not_transpose``
   Field data is metadata, not a spatial array — ZYX is meaningless.

   The one place ``xyz_data_ordering=False`` is passed from inside the
   package.

``test_a_scalar_is_accepted``
   Undocumented.

``test_an_explicit_dtype_is_honoured``
   Undocumented.

``test_several_fields_coexist``
   Undocumented.

``test_a_partial_field_write_is_supported``
   Undocumented.

When these fail
~~~~~~~~~~~~~~~

**A shape assertion looks transposed.** It is. ``numpy.transpose`` with no
argument reverses **all** axes, and ``xyz_data_ordering`` defaults to
``True``. A 3-D ``(2, 3, 4)`` array is stored ``(4, 3, 2)``; a 2-D ``(N,
3)`` array is stored ``(3, N)``. These tests read the raw on-disk array and
do **not** transpose it back, because the on-disk order is the thing being
pinned.

**A partial write lands in the wrong corner.** ``shape`` and ``offset`` are
flipped alongside the data. All three must be flipped together, or the block
lands in the wrong place inside a correctly shaped dataset —
``test_the_offset_is_flipped_with_the_data`` is the assertion that separates
the two failures.

**A string warning does not fire.** Three of the four warning paths only
trigger when the caller passes an **explicit** ``dtype``. Handing in a plain
``'U'`` array converts silently — the common case, and arguably the one most
worth reporting.

**A duplicate-dataset test does not raise.** ``create_dataset`` will not
overwrite; the second write to the same path raises ``ValueError``. Relevant
to any caller that names two datasets alike.

**An error-message assertion fails.** The three partial-write errors are
asserted on their text, because the text is all a user has when a rank's
arithmetic is wrong — including the arithmetic itself, ``[2] + (2,) = [4] >
[3]``.

Test Catalog — ``test_vtk_image_data.py``
-----------------------------------------

**51 tests** from 51 test functions across 8 classes.

``VtkImageData`` — the voxel writer behind every geometry view and snapshot.

An ImageData file describes a regular grid implicitly: an origin, a spacing
and an extent are enough to place every cell, so no coordinates are stored.
That makes it compact — a 200³ geometry view is one integer array rather
than eight million points — and it makes the four root attributes load-
bearing. Get ``Spacing`` wrong and the model renders at the wrong physical
size with no error; get ``WholeExtent`` wrong and VTK reads past the data.

Three things here are specific to this class and covered nowhere else:

**Shape padding.** VTKHDF ImageData is always three-dimensional. A 1-D or
2-D shape is padded with ones, so a 2-D gprMax model writes a file one cell
deep rather than failing. The padding also decides what ``add_cell_data``
will accept afterwards, since the stored ``self.shape`` is the padded one.

**The extent convention.** ``WholeExtent`` is six numbers — a ``[min, max]``
pair per axis — and the maxima are *cell* counts, so a shape of ``(2, 3,
4)`` gives ``[0, 2, 0, 3, 0, 4]``. Points are one more than cells in each
direction, which is why ``add_point_data`` demands ``shape + 1``.

**The dimension checks.** ``add_cell_data`` and ``add_point_data`` each
raise before writing anything if the array does not match, which is the
difference between a clear error and a file VTK cannot open.

PR 10's outputs suite already pins the *values* these attributes take for a
real geometry view. This file tests the writer's own contract: the defaults,
the validation, and the padding — none of which a round-trip through
``write_vtk`` can reach.

TestConstruction
^^^^^^^^^^^^^^^^

What a newly created file contains before anything is added.

``test_the_type_attribute_says_image_data``
   Undocumented.

``test_the_file_is_always_opened_for_writing``
   The constructor hard-codes mode ``"w"`` — there is no read path.

   A ``VtkImageData`` always truncates; it is a writer, not a reader.

``test_the_shape_is_stored``
   Undocumented.

``test_the_four_attributes_are_written``
   Undocumented.

``test_no_datasets_are_written``
   Geometry is implicit; only the attributes describe the grid.

``test_the_dimension_count_is_three``
   Undocumented.

TestWholeExtent
^^^^^^^^^^^^^^^

Six numbers: a ``[min, max]`` pair per axis, in cells.

``test_it_is_derived_from_the_shape``
   Undocumented.

``test_the_minima_are_zero``
   gprMax models always start at the origin.

``test_the_maxima_are_the_cell_counts``
   Undocumented.

``test_it_is_integral``
   A float extent makes VTK compute fractional indices.

``test_it_persists_to_disk``
   Undocumented.

TestShapePadding
^^^^^^^^^^^^^^^^

1-D and 2-D models become 3-D files, padded with ones.

``test_a_two_dimensional_shape_gains_a_third``
   A 2-D gprMax model is one cell deep, not an error.

``test_a_one_dimensional_shape_gains_two``
   Undocumented.

``test_the_padded_extent_is_still_six_numbers``
   Undocumented.

``test_an_empty_shape_raises``
   Undocumented.

``test_a_four_dimensional_shape_raises``
   There is no fourth spatial axis to pad away.

``test_the_padded_shape_is_what_cell_data_must_match``
   The padding is not cosmetic: it changes the validation downstream.

TestOrigin
^^^^^^^^^^

Where the grid's first corner sits in physical space.

``test_it_defaults_to_the_coordinate_origin``
   Undocumented.

``test_a_supplied_origin_is_used``
   Undocumented.

``test_it_can_be_changed_afterwards``
   Snapshots of a subgrid share a writer but not an origin.

``test_a_two_element_origin_raises``
   Undocumented.

``test_a_four_element_origin_raises``
   Undocumented.

``test_it_persists_to_disk``
   Undocumented.

TestSpacing
^^^^^^^^^^^

Cell size — the attribute that sets the model's physical scale.

``test_it_defaults_to_unit_cells``
   Undocumented.

``test_a_supplied_spacing_is_used``
   gprMax passes the grid discretisation here — usually millimetres.

``test_anisotropic_spacing_is_kept_per_axis``
   Undocumented.

``test_it_can_be_changed_afterwards``
   Undocumented.

``test_a_two_element_spacing_raises``
   Undocumented.

``test_it_persists_to_disk``
   Undocumented.

TestDirection
^^^^^^^^^^^^^

The nine-element basis; the identity unless a model is rotated.

``test_it_defaults_to_the_identity``
   Undocumented.

``test_a_flat_array_is_accepted``
   Undocumented.

``test_a_nested_array_is_flattened``
   The docstring promises the two forms are equivalent.

``test_it_is_always_stored_flat``
   Undocumented.

``test_it_can_be_changed_afterwards``
   Undocumented.

``test_a_three_element_direction_raises``
   Undocumented.

``test_a_four_by_four_direction_raises``
   Undocumented.

TestAddCellData
^^^^^^^^^^^^^^^

One value per cell — materials, and every snapshot field.

``test_a_matching_array_is_written``
   Undocumented.

``test_it_is_stored_in_zyx_order``
   The shape on disk is reversed, as for every spatial dataset.

``test_the_values_survive_the_round_trip``
   Undocumented.

``test_a_mismatched_shape_raises``
   Undocumented.

``test_the_error_shows_both_shapes``
   Undocumented.

``test_a_partial_write_bypasses_the_shape_check``
   With an offset, the array is a slab and need not match.

``test_the_slab_lands_at_the_offset``
   The MPI case: each rank writes its own x-slab.

``test_several_cell_arrays_coexist``
   A geometry view holds materials; a snapshot holds six fields.

``test_the_dtype_is_preserved``
   Material IDs are integers; storing them as floats doubles the file.

TestAddPointData
^^^^^^^^^^^^^^^^

One value per *point* — one more than the cells in each direction.

``test_an_array_one_larger_in_each_dimension_is_accepted``
   Undocumented.

``test_the_cell_shape_is_rejected``
   The off-by-one that would otherwise be silent in a viewer.

``test_the_error_shows_the_expected_shape``
   Undocumented.

``test_it_is_stored_in_zyx_order``
   Undocumented.

``test_a_partial_write_bypasses_the_shape_check``
   Undocumented.

``test_point_and_cell_data_coexist``
   Undocumented.

When these fail
~~~~~~~~~~~~~~~

**An extent assertion is off by one.** ``WholeExtent`` is six numbers, a
``[min, max]`` pair per axis, and the maxima are **cell** counts. Points are
one more than cells in each direction, which is why ``add_point_data``
demands ``shape + 1`` and ``add_cell_data`` demands ``shape``.

**A 2-D model's cell data is rejected.** VTKHDF ImageData is always three-
dimensional, so a 1-D or 2-D shape is padded with ones — and the *padded*
shape is what the validation then compares against. A ``(2, 3)`` model needs
``(2, 3, 1)`` cell data.

**A default is not what you expected.** Origin defaults to ``[0, 0, 0]``,
spacing to ``[1, 1, 1]``, direction to the flattened identity. gprMax passes
the grid discretisation as spacing — usually millimetres — and getting it
wrong renders the model at the wrong physical size with no error anywhere.

**A shape test passes but a value test fails.** Shape alone will accept a
wrong *permutation*. ``test_the_values_survive_the_round_trip`` compares the
transposed-back array element for element, which is the assertion that
distinguishes them. The default fixture shape is deliberately anisotropic
for the same reason.

Test Catalog — ``test_vtk_unstructured_grid.py``
------------------------------------------------

**46 tests** from 46 test functions across 7 classes.

``VtkUnstructuredGrid`` — explicit points and cells, for line geometry.

Where ImageData describes a grid implicitly, an unstructured grid stores
everything: N point coordinates, C cell types, and a connectivity array cut
into cells by an offsets array. gprMax uses it for geometry views in *line*
mode, where each cell is a two-point edge along a material boundary.

The four arrays have to agree, and none of the agreements is checkable by
VTK after the fact — a connectivity index one too large reads a point that
is not there. The constructor therefore validates three relationships before
writing anything:

- ``cell_offsets`` is one longer than ``cell_types`` — C cells need C+1
  boundaries;

- ``cell_offsets`` ascends — a cell cannot end before it starts;

- ``connectivity`` is at least as long as the final offset — every cell's
  points must exist.

The fourth case, a connectivity array *longer* than the offsets require, is
a warning rather than an error: the surplus is simply unreferenced.

**Seven datasets** are written on construction, and their names are fixed by
the VTKHDF specification. They are asserted as a set here, because a missing
one produces a file VTK opens and then renders as nothing at all — the most
expensive kind of failure to diagnose.

**``Points`` is the exception to the transpose.** It is written with
``xyz_data_ordering=False``, keeping the ``(N, 3)`` layout VTK expects for
coordinates. Every *other* dataset goes through the default path — which is
what makes the 2-D point-data case below worth pinning.

Only the serial path is exercised. The MPI branch needs a real communicator
with more than one rank; ``tests/unit/outputs/test_mpi_grid_view.py`` covers
that ground from above.

TestConstruction
^^^^^^^^^^^^^^^^

A minimal grid: two points joined by one line.

``test_the_type_attribute_says_unstructured_grid``
   Undocumented.

``test_all_seven_datasets_are_written``
   Undocumented.

``test_the_dataset_names_are_the_specification_ones``
   Pinned against the enum, so a rename is caught at the source.

``test_the_file_is_always_opened_for_writing``
   Undocumented.

``test_the_serial_partition_is_zero``
   Without a communicator there is one partition, numbered zero.

TestCounts
^^^^^^^^^^

The three count datasets, and the properties that mirror them.

``test_the_cell_count_is_the_number_of_cell_types``
   Undocumented.

``test_the_point_count_is_the_number_of_coordinates``
   Undocumented.

``test_the_connectivity_count_is_the_array_length``
   Undocumented.

``test_the_counts_are_written_to_disk``
   Undocumented.

``test_the_counts_are_one_element_datasets``
   One entry per partition; serial writes a single element.

   The format expects an array here even when there is one rank, which is
   why the scalar is expanded rather than written as an attribute.

``test_the_global_counts_match_the_local_ones``
   Serial: no reduction happens, so global equals local.

``test_the_offsets_start_at_zero``
   Serial has nothing before it to skip.

``test_the_point_offsets_are_a_pair``
   Two dimensions, because ``Points`` is an ``(N, 3)`` array.

TestPointsDataset
^^^^^^^^^^^^^^^^^

Coordinates — the one dataset written without the ZYX transpose.

``test_the_shape_is_points_by_three``
   ``(N, 3)`` on disk, exactly as VTK expects for coordinates.

``test_the_coordinates_are_not_transposed``
   A distinctive point makes the orientation unambiguous.

``test_the_coordinates_survive_the_round_trip``
   Undocumented.

TestCellsAndConnectivity
^^^^^^^^^^^^^^^^^^^^^^^^

How points are grouped into cells.

``test_the_cell_types_are_written``
   Undocumented.

``test_the_cell_types_are_unsigned_bytes``
   One byte per cell; a geometry view can hold millions.

``test_the_connectivity_is_written``
   Undocumented.

``test_the_offsets_are_written``
   Undocumented.

``test_the_offsets_are_one_longer_than_the_cells``
   C+1 boundaries for C cells — the invariant the check enforces.

``test_a_grid_with_no_cells_is_accepted``
   An empty geometry view is a legitimate, if dull, output.

TestValidation
^^^^^^^^^^^^^^

Three raises and one warning, before anything is written.

``test_too_few_offsets_raises``
   Undocumented.

``test_too_many_offsets_raises``
   Undocumented.

``test_unsorted_offsets_raise``
   A cell that ends before it starts would read backwards.

``test_equal_consecutive_offsets_are_allowed``
   A zero-length cell is degenerate but not out of order.

``test_a_short_connectivity_array_raises``
   The last cell would reference points past the end of the array.

``test_a_long_connectivity_array_warns``
   Surplus entries are unreferenced, not wrong — a warning suffices.

``test_a_long_connectivity_array_is_still_written_in_full``
   The surplus is stored; only the offsets decide what is read.

``test_a_valid_grid_does_not_warn``
   Undocumented.

``test_a_failed_construction_still_leaves_a_file``
   The file is opened — and truncated — before validation runs.

   So a rejected grid destroys any earlier file of the same name and leaves
   an empty one in its place, with the handle never closed. Pinned as the
   current behaviour; written up in ``notes/bugs/vtkhdf-truncates-before-
   validating.md``.

TestAddCellData
^^^^^^^^^^^^^^^

One value, or one 3-vector, per cell.

``test_a_matching_array_is_written``
   Undocumented.

``test_a_three_component_array_is_accepted``
   Vectors per cell — the ``(C, 3)`` layout VTK defines.

``test_a_single_component_array_is_accepted``
   Undocumented.

``test_a_wrong_length_raises``
   Undocumented.

``test_the_error_names_the_partition``
   Under MPI the rank is what a user needs to know.

``test_a_three_dimensional_array_raises``
   Undocumented.

``test_a_two_component_array_raises``
   VTK has scalars and 3-vectors; nothing in between.

``test_two_dimensional_cell_data_is_stored_transposed``
   ``(C, 3)`` goes in as ``(3, C)`` — the opposite of VTK's layout.

   ``Points`` is written with ``xyz_data_ordering=False`` for exactly this
   reason, but ``add_cell_data`` and ``add_point_data`` are not, so vector-
   valued cell data comes out with components and tuples swapped. gprMax
   only ever writes scalar cell data, where a 1-D transpose is a no-op,
   which is why it has never been noticed. Pinned as the current behaviour;
   written up in ``notes/bugs/vtkhdf-unstructured-vector-data-
   transposed.md``.

TestAddPointData
^^^^^^^^^^^^^^^^

One value, or one 3-vector, per point.

``test_a_matching_array_is_written``
   Undocumented.

``test_a_three_component_array_is_accepted``
   Undocumented.

``test_a_wrong_length_raises``
   Undocumented.

``test_a_three_dimensional_array_raises``
   Undocumented.

``test_a_two_component_array_raises``
   Undocumented.

``test_point_and_cell_data_coexist``
   Undocumented.

``test_the_seven_required_datasets_are_undisturbed``
   Adding data must not perturb the structural datasets.

When these fail
~~~~~~~~~~~~~~~

**A required dataset is missing.** Seven are written on construction, and
their names are fixed by the VTKHDF specification. A missing one produces a
file VTK opens and then renders as nothing at all — the most expensive kind
of failure to diagnose, which is why they are asserted as a complete set.

**A ``Points`` orientation assertion fails.** ``Points`` is the **one**
dataset written with ``xyz_data_ordering=False``, keeping the ``(N, 3)``
layout VTK expects for coordinates. Every other dataset goes through the
default transposing path.

**Vector-valued attribute data comes out ``(3, C)``.** Correct as tested,
wrong as a format: ``add_point_data`` and ``add_cell_data`` accept an ``(N,
3)`` array and then write it transposed. gprMax only ever writes scalar
attribute data, where a 1-D transpose is a no-op. See ``notes/bugs/vtkhdf-
unstructured-vector-data-transposed.md``.

**A validation test leaves a file behind.** It does, and that is pinned. The
constructor calls ``super().__init__`` — mode ``"w"`` — before checking any
argument, so a rejected grid destroys any existing file of that name and
leaks the handle. See ``notes/bugs/vtkhdf-truncates-before-validating.md``.

**Only the serial path is here.** The MPI branch needs a real communicator
with more than one rank. ``tests/unit/outputs/test_mpi_grid_view.py`` covers
that ground from above.

Deliberately Untested Paths
---------------------------

The standing rule from PR 9 onward is that **no test asserts broken behaviour
and none is marked** ``xfail``. Where a defect made a contract untestable, the
test was omitted. That leaves coverage holes which would otherwise look like
oversights, so they are named here.

Each has a maintainer write-up carrying the tests its fix should add, so
closing one of these is a matter of applying the fix and pasting in the test.

**An unknown precision.**
   ``_set_precision`` dispatches on ``general["precision"]`` with no terminal
   ``else``, so any other value leaves ``self.dtypes`` undefined. The
   ``SimulationConfig`` constructs successfully and the failure arrives as an
   ``AttributeError`` from whichever module allocates an array first.

**An unsupported platform.**
   ``get_host_info``'s three-way ``sys.platform`` chain has no ``else``, so
   FreeBSD, Cygwin and AIX reach the dictionary construction with
   ``machineID``, ``hyperthreading`` and ``osversion`` unbound. The five *other*
   locals in that function do have ``"unknown"`` defaults; these three do not.

**An unknown logging format style.**
   ``logging_config``'s two-way chain has no ``else``, leaving ``format``
   unbound. Uniquely among these, the failure happens *before logging is
   configured*, so a traceback is all the user gets.

**``mem_check_device_snaps`` on Metal or CPU.**
   Two branches, four solvers. ``device_mem`` is unbound for the other two. The
   Metal path is reachable in principle and is currently prevented only by an
   operator-precedence bug in the caller's guard — the two must be fixed
   together.

**An OpenCL device that is neither CPU nor GPU.**
   ``print_opencl_info`` assigns ``type`` in two ``if``\ s with no ``else``. An
   ``ACCELERATOR`` or ``CUSTOM`` device either raises ``UnboundLocalError``
   while merely listing devices, or — worse — silently inherits the previous
   device's label.

**A dispersive dispatch before the material types are set.**
   ``materials["dispersivedtype"]`` defaults to ``None``, and
   ``None == np.complex64`` is ``False``, so calling ``set_dispersive_updates``
   too early silently selects the **real** kernel for a complex-pole model. The
   run completes and the dispersive materials are wrong.

**Restarted multi-model runs.**
   ``model_configs`` is sized ``n`` while the model indices run to
   ``(i-1)+n``, so every run with ``-i > 1`` raises ``IndexError`` when its
   first model's config is stored. The arithmetic mismatch is asserted; driving
   the failure needs the whole context loop.

**Combined accelerator arguments.**
   The "cannot combine CUDA/OpenCL/Metal" guard uses ``list.count(True)`` on
   arguments that are device-ID *lists*, so it never fires. Three independent
   ``if``\ s then run in sequence and the last one wins, leaving the CUDA
   detection's work discarded. The suite pins that the guard does not fire
   rather than asserting the resulting state is correct.

**Output paths containing a dot.**
   ``with_suffix`` truncates at the last dot, in both ``config.py`` and the
   VTKHDF constructor. Two models named ``v1.2_model`` and ``v1.9_model`` write
   to the same file. Pinned as current behaviour in both suites, with the
   collision case left for the fix to add.

**Vector-valued point and cell data.**
   ``VtkUnstructuredGrid.add_point_data`` and ``add_cell_data`` explicitly
   accept an ``(N, 3)`` array and then write it transposed, because they use
   the default ZYX path. gprMax only ever writes scalar attribute data, where a
   1-D transpose is a no-op, so no real output is affected.

**The error path of the VTKHDF constructors.**
   Both concrete writers call ``super().__init__`` — which opens with mode
   ``"w"`` — before validating any argument. A rejected grid therefore destroys
   any existing file of that name and leaks the handle. The suite pins that a
   file is left behind; it does not assert that this is acceptable.

**``MetalUpdates`` as an** ``Updates`` **implementation.**
   It does not subclass ``Updates``, so ABCMeta never checks it. It happens to
   implement all eleven methods today. One test pins the non-conformance and
   names the parametrisation to add once the base class is declared.

**Parallel I/O in the VTKHDF writers.**
   Every ``comm``-aware branch — the MPI partition arithmetic in
   ``VtkUnstructuredGrid``, the fixed-length string handling, the
   ``_create_dataset`` variable-length guard — needs a real multi-rank
   communicator and parallel HDF5. ``h5py.get_config().mpi`` is ``False`` here
   and on CI. The serial paths are covered unconditionally; **this is an
   environment constraint, not a decision**.

**The accelerator backends' own update methods.**
   ``CUDAUpdates``, ``OpenCLUpdates`` and ``MetalUpdates`` are tested only for
   their conformance to the ``Updates`` contract and their selection by
   ``create_solver``. Their bodies import ``pycuda`` / ``pyopencl`` /
   ``Metal`` and drive real devices. That is PR 12, behind hardware ``skipif``
   guards.

**Five more no-terminal-**\ ``else`` **instances** appear above:
``_set_precision``, the platform chain, the format-style chain,
``mem_check_device_snaps`` and ``print_opencl_info``. With PR 9's five and PR
10's four that is **fourteen across three PRs**, and it is now clearly one issue
about the codebase's dispatch idiom rather than fourteen tickets. The pattern is
always the same: a chain that is total on the production path, in a function
whose arguments are public.

Out of Scope
------------

**No source changes under** ``gprMax/``. Not the missing ``else`` branches, not
the ``lscpu`` parse, not the seven remaining ``FileNotFoundError`` clauses.
Tests only, per the standing decision from PR 9 onward, with every defect
written up for the maintainers instead.

**No changes to the upstream sketch.** ``tests/updates/test_cpu_updates.py``
and ``tests/grid/test_fdtd_grid.py`` are left exactly as they are. The new
suite sits alongside them in ``tests/unit/``, and *Closing the Upstream
Sketch's Gaps* above maps each of the six unimplemented stubs to the test that
now covers it.

**No CI change.** Unlike PR 10, no workflow edit proved necessary. The suite is
green under ``OMP_NUM_THREADS=1`` and ``MPI4PY_RC_FINALIZE=0``, and with the
four new directories collected first, last, and interleaved with the existing
ones.

**No physics validation.** ``Solver.solve`` is tested for the order it calls
things in and the kernels for the region they write, never for the numbers that
come out. That needs an analytic reference solution and belongs in an
integration suite.

**No GPU execution.** The accelerator device probes are driven against
fabricated ``pycuda`` / ``pyopencl`` / ``Metal`` modules inserted into
``sys.modules``. That is what makes the *success* paths reachable at all on a
CPU-only runner; without it, only the "not installed" branch could ever be
tested, which is the branch that matters least.

**No** ``contexts.py`` **or** ``model.py``. ``create_solver`` is tested;
what calls it is not. The context loop is where ``set_current_model`` is driven
and where the ``-i`` index bug would surface, and it remains an orchestrator
outside the unit suite's scope.

**No re-testing of what PR 10 already pins.** The VTKHDF suite is complementary
by construction: PR 10's ``write_vtk`` round-trips establish the on-disk layout
for real geometry views and snapshots, and this suite covers the writers' own
contract — file naming, root-attribute dtypes, shape padding, string
conversion, partial writes, the error messages — none of which a round-trip can
reach.
