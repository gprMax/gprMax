External optimisation packages
------------------------------

Install packages in the **same Python environment as gprMax**. The versions below
were used for the toolbox trials; they are tested versions, not a claim that they
are the latest available releases. Optional packages are imported when their
adapter is used.

.. list-table:: Tried integrations
   :header-rows: 1
   :widths: 15 22 18 45

   * - Selection
     - Package tested
     - Public adapter
     - Method and parameter support
   * - ``"tpe"``
     - Optuna 4.9.0
     - ``OptunaTPE``
     - Tree-structured Parzen Estimator; real, integer and categorical.
   * - ``"rf"``
     - scikit-optimize 0.10.2
     - ``SkoptRF``
     - Random-forest surrogate with expected improvement by default; real, integer and categorical.
   * - ``"ga"``
     - pymoo 0.6.2
     - ``PymooGA``
     - Genetic algorithm; real and integer.
   * - ``"pso"``
     - pymoo 0.6.2
     - ``PymooPSO``
     - Particle swarm optimisation; real and integer.
   * - ``"de"``
     - pymoo 0.6.2
     - ``PymooDE``
     - Differential evolution; real and integer.
   * - Direct helper
     - SciPy 1.18.0
     - ``minimise_integer``
     - A separate bounded scalar search for one integer variable.

For one package, use the appropriate command:

.. code-block:: console

   python -m pip install optuna==4.9.0
   python -m pip install scikit-optimize==0.10.2
   python -m pip install pymoo==0.6.2

To install the five general adapters together, use
``python -m pip install -r gprMax/toolboxes/Optimisation/requirements-optimisers.txt``.
SciPy is already a gprMax dependency; the direct helper is described separately
below. Check the active interpreter with ``python -m pip show optuna pymoo
scikit-optimize scipy`` if an import fails.

Switching packages without changing the problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`optimiser_choices.py <../../gprMax/toolboxes/Optimisation/examples/optimiser_choices.py>`
imports the starter's parameter definitions, builder and objective. Only the
launch settings change. For example, from the checkout:

.. code-block:: console

   python -m gprMax.toolboxes.Optimisation.examples.optimiser_choices results/tpe --optimiser tpe
   python -m gprMax.toolboxes.Optimisation.examples.optimiser_choices results/rf --optimiser rf
   python -m gprMax.toolboxes.Optimisation.examples.optimiser_choices results/ga --optimiser ga
   python -m gprMax.toolboxes.Optimisation.examples.optimiser_choices results/pso --optimiser pso
   python -m gprMax.toolboxes.Optimisation.examples.optimiser_choices results/de --optimiser de

Run only the methods you want to try. Each command creates a separate twelve-trial
search on the same small model. Population methods use six candidates per batch.
These short runs demonstrate the connection to the solver; they do not rank the
algorithms or establish convergence.

RF and TPE learn from evaluated parameter/score pairs. RF predicts promising
candidates with its forest, but the objective sent back to the optimiser is
still computed from the actual gprMax result. This is distinct from training a
surrogate to replace the solver permanently.

GA, PSO and DE propose populations. Integer coordinates are mapped to the declared
physical step before reaching the model. PSO retains its continuous internal
positions and velocities; GA and DE repair their coordinates onto the allowed
lattice. This implementation detail does not change the dictionary your model
receives. Use the same parameter definitions and objective when comparing methods.

Setting algorithm options
~~~~~~~~~~~~~~~~~~~~~~~~~

Pass an adapter object instead of a string when you want package-specific
settings. Use one of these objects as ``optimiser=adapter`` in the ordinary
``optimise`` call. Its seed and batch/population size take precedence over the
string-selection settings:

.. code-block:: python

   from gprMax.toolboxes.Optimisation import OptunaTPE, SkoptRF, PymooGA, PymooPSO, PymooDE

   adapter = OptunaTPE(seed=7, batch_size=2)
   adapter = SkoptRF(seed=7, n_initial_points=8, batch_size=2)
   adapter = PymooGA(seed=7, population_size=10)
   adapter = PymooPSO(seed=7, population_size=10, restart_after=20)
   adapter = PymooDE(seed=7, population_size=10, options={"variant": "DE/rand/1/bin"})

These are alternatives: choose one assignment. ``restart_after`` is an optional
toolbox PSO policy that starts a fresh swarm after that many populations without
improvement, while retaining the campaign's best result and total budget. It is
disabled by default. Native pymoo PSO defaults remain active unless overridden
through ``options``. A larger population or more evaluations can change results;
a seed makes a particular configuration repeatable, not universally successful.

The official package documentation explains their algorithms and native options:

* `Optuna TPE <https://optuna.readthedocs.io/en/v4.9.0/reference/samplers/generated/optuna.samplers.TPESampler.html>`_.
* `scikit-optimize ask/tell Optimizer <https://scikit-optimize.readthedocs.io/en/stable/modules/generated/skopt.Optimizer.html>`_.
* pymoo: `GA <https://pymoo.org/algorithms/soo/ga.html>`_, `PSO <https://pymoo.org/algorithms/soo/pso.html>`_, `DE <https://pymoo.org/algorithms/soo/de.html>`_.

SciPy: a separate one-variable example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`scipy_integer.py <../../gprMax/toolboxes/Optimisation/examples/scipy_integer.py>`
uses ``minimise_integer`` to call the same small model at integer permittivities:

.. code-block:: console

   python -m gprMax.toolboxes.Optimisation.examples.scipy_integer results/scipy

This helper wraps
`SciPy's bounded minimize_scalar <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html>`_,
remembers scores for repeated integers, and checks endpoints and neighbours. It
is intended for an approximately unimodal interval. Its callback owns the
simulation calls and storage. ``maxiter`` limits SciPy's internal iterations;
explicit endpoint/neighbour checks are additional evaluations.

There is no ``optimiser="scipy"`` alias in the general framework. Use the five
adapters above for several parameters, parallel populations, shared campaign
caching or native optimiser checkpoints.

What has been tried
~~~~~~~~~~~~~~~~~~~

TPE, RF, GA, PSO and DE have been exercised on the thin-wire dipole, as well as on
small integration problems. The direct SciPy integer helper was also tried on
the dipole. PSO and DE both found the published Antenna I dimensions in our
rectangular-patch adaptation. A short budget did not make every dipole method
converge equally; the examples demonstrate reusable mechanics, not an optimiser
leaderboard. See the patch comparison and its modelling limits below.
