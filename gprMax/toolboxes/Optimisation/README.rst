.. _optimisation-toolbox:

Optimisation toolbox
====================

The toolbox connects a user's gprMax model to an external optimiser. You define
what may change and how to score a completed simulation. The toolbox proposes
values, builds and runs the model, reads your score, and returns it to the
optimiser. Geometry can change on every trial; fixed-geometry Studies remain a
separate workflow.

The user guide lives beside the code in ``gprMax/toolboxes/Optimisation`` and is included
in the main gprMax documentation. Start here; the campaign and developer guides
are optional references for more advanced uses.

.. contents:: On this page
   :local:
   :depth: 2

Start with a small working example
----------------------------------

Use the Python environment in which this gprMax development checkout runs. From
the checkout directory, install the starter's optional optimisation package:

.. code-block:: console

   python -m pip install optuna==4.9.0
   mkdir -p my_optimisation
   cp gprMax/toolboxes/Optimisation/examples/start_here.py my_optimisation/my_model.py
   python my_optimisation/my_model.py

The example creates a small dielectric-block model, measures the peak receiver
field and adjusts permittivity to approach a target peak. Its twelve evaluations
are a short demonstration, not a convergence guarantee. It saves results beside
the copied script. Set ``RESULTS`` to a fresh directory before running again.

:download:`Copy the starter <../../gprMax/toolboxes/Optimisation/examples/start_here.py>`.
Save callbacks in a Python file; workers must be able to import them. Keep the
launch under ``if __name__ == "__main__":`` as shown in the example. Additional
input files and imported helpers should be listed in ``files=`` so changes are
tracked. Finish editing before launching a campaign.

The four sections you edit
--------------------------

.. list-table:: Your model file
   :header-rows: 1
   :widths: 22 38 40

   * - Section
     - What you supply
     - Why it is needed
   * - 1. Parameters and targets
     - ``PARAMETERS`` and your target constants
     - Defines allowed trial values and the desired physical response.
   * - 2. Model
     - ``build_model(parameters)``
     - Uses those values to return a complete, unexecuted ``gprMax.Scene``.
   * - 3. Criterion
     - ``evaluate(parameters, output)``
     - Reads and processes the result; returns one finite number to minimise.
   * - 4. Launch
     - ``optimise(...)``
     - Chooses the optimiser, budget, resources and results directory.

The keys in ``PARAMETERS`` are your names. For example,
``parameters["length_mm"]`` is the proposed physical length, not an internal
optimiser coordinate. The same dictionary reaches both user functions.

Parameter values and units
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gprMax.toolboxes.Optimisation import Real, Integer, Categorical

   PARAMETERS = {
       "length_mm": Integer(26, 36, "mm", step=2),  # 26, 28, 30, 32, 34, 36
       "permittivity": Real(2.0, 8.0),
       "material_name": Categorical(("dielectric_a", "dielectric_b")),
   }

``Real`` uses continuous bounds; ``Integer`` uses inclusive integer bounds and
an optional positive integer step; ``Categorical`` chooses one of the supplied
values. Both integer endpoints must lie on the declared lattice. ``unit`` is a
label: convert millimetres to metres yourself in the model with ``value * 1e-3``.
The toolbox does not change the mesh or interpret your parameter names.

Use TPE or RF for categorical parameters. GA, PSO and DE currently accept numeric
parameters only. All five support integer steps. Do not round to a different
geometry in your model without recording what was actually built. Choose a mesh
and parameter increments that represent your intended dimensions.

Build and check one model first
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the domain, mesh, materials, geometry, source, receivers and any required
monitors in ``build_model``. Return the Scene without calling ``gprMax.run``.
The toolbox runs it in an isolated worker. To check one concrete design, put
this code in a saved script:

.. code-block:: python

   from gprMax.toolboxes.Optimisation import simulate
   from gprMax.toolboxes.Optimisation.examples.start_here import build_model, evaluate

   if __name__ == "__main__":
       values = {"permittivity": 4.0}
       output = simulate(model=build_model, parameters=values,
                         directory="results/one_model")
       print(output.file)          # Path to the solver's main HDF5 file.
       print(output.datasets())    # Dataset names that were actually saved.
       print(evaluate(values, output))

``simulate`` takes concrete values, with no optimiser or parameter ranges.
Verify the geometry and output before spending a large optimisation budget.

Choose the criterion and stopping condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The line returned by ``evaluate`` defines the criterion. The optimiser always
minimises: ``-30`` is better than ``-20``. To maximise a positive quantity such
as gain, return its negative. To match a waveform, return a scalar error between
simulated and measured traces. You may use NumPy, an existing postprocessor or
your own functions; the optimiser never needs to understand the raw output.

These are separate settings:

* ``objective=evaluate`` supplies the function that computes the score.
* ``evaluations=120`` limits candidate evaluations. Repeated candidates count.
  Population methods use whole batches and may leave a remainder unused.
* ``target_value=1.0`` optionally stops when the best score is at most 1.0.
  All candidates in the current batch finish first. Omit it for a budget-only run.
  ``stop_on_target=False`` records target attainment while continuing the budget.

The target has the same units and sign as the returned score. For an S11 objective
in dB, ``target_value=-30.0`` means an acceptable S11 of -30 dB or better.
Neither a used-up budget nor a flat best-score curve proves global optimality.
Missing or invalid outputs raise an error; correct the model or evaluation
before treating such cases as a deliberate physical penalty.

Run, inspect and change the optimiser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gprMax.toolboxes.Optimisation import optimise
   from gprMax.toolboxes.Optimisation.examples.start_here import PARAMETERS, build_model, evaluate

   if __name__ == "__main__":
       result = optimise(
           parameters=PARAMETERS, model=build_model, objective=evaluate,
           directory="results/my_de_run", optimiser="de",
           evaluations=120, population_size=10, seed=7,
       )
       print(result.best_parameters)
       print(result.best_value)
       print(result.stop_reason)

Change ``optimiser`` to ``"tpe"``, ``"rf"``, ``"ga"``, ``"pso"`` or ``"de"``
after installing its package. ``population_size`` controls GA/PSO/DE;
``batch_size`` controls simultaneous proposals for TPE/RF. They are independent
of the number of available worker processes.

Results and resources
---------------------

``directory`` is a fresh campaign directory, not an input-output folder you must
populate yourself. The toolbox creates one candidate directory per evaluation,
keeps its parameter record and solver log, and passes the correct result to your
objective. ``output.file`` identifies that candidate's main HDF5 output.
``result.trials`` records the evaluated parameters, scores and statuses;
``optimiser/result.json`` saves the search result. Processing artifacts saved
with ``output.save_npz`` belong to the evaluation that produced them.

The default runs serially on the CPU. A local pool can run separate models in
parallel, including one per GPU supported by your gprMax installation:

.. code-block:: python

   from gprMax.toolboxes.Optimisation import LocalPool

   execution = LocalPool(workers=4, cpu_threads_per_worker=2)
   # GPU alternative, with these device IDs visible to this process:
   # execution = LocalPool(solver="cuda", devices=[0, 1], cpu_threads_per_worker=2)
   # Add execution=execution to the optimise() call above.

Every worker builds a complete model; each needs enough memory. This distributes
candidate simulations, not one FDTD domain. Use enough candidates per batch to
fill the available workers. MPI, resource profiles, caching and native optimiser
checkpoint recovery are covered in the
:download:`execution guide <../../gprMax/toolboxes/Optimisation/execution-and-processing.md>`.
The simple ``optimise`` call creates a new campaign; use the explicit Campaign
interface for checkpoint recovery and shared simulation caches.

.. include:: ../../gprMax/toolboxes/Optimisation/OPTIMISERS.rst

.. include:: ../../gprMax/toolboxes/Optimisation/OUTPUT_READERS.rst

.. include:: ../../gprMax/toolboxes/Optimisation/EXAMPLES.rst

Further reference
-----------------

* :download:`Campaign API <../../gprMax/toolboxes/Optimisation/CAMPAIGN.rst>`: multiple scenarios, builders and stored records.
* :download:`Code walkthrough <../../gprMax/toolboxes/Optimisation/CODE_WALKTHROUGH.md>`: follows a candidate from proposal to score.
* :download:`Developer guide <../../gprMax/toolboxes/Optimisation/DEVELOPER_GUIDE.md>`: module responsibilities and extension points.
* :download:`Adapter contract <../../gprMax/toolboxes/Optimisation/ADAPTERS.md>`: add another optimiser without changing model code.
