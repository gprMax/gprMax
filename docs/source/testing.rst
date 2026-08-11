.. _developer-testing:

**********************
Testing and validation
**********************

gprMax uses several complementary forms of testing. They are kept separate
because a fast unit test, a comparison with an analytical solution, an
inter-code research study, and an HPC regression campaign provide different
evidence. A numerical comparison with another solver, for example, must not be
reported as validation against ground truth.

.. list-table:: Testing structure
    :header-rows: 1
    :widths: 22 24 54

    * - Location
      - Mechanism
      - Purpose
    * - ``tests``
      - pytest
      - Automated unit, compact integration, and real-GPU tests
    * - ``testing/validation``
      - Manual scientific drivers
      - Production gprMax results compared with independent analytical
        solutions and quantitative acceptance criteria
    * - ``testing/other_codes``
      - Reproducible studies
      - Comparisons with other numerical solvers; neither result is treated
        as ground truth
    * - ``testing/backend_consistency``
      - Diagnostic drivers
      - CPU/GPU, precision, orientation, and equivalent-configuration parity
    * - ``testing/regression``
      - Diagnostic drivers
      - Larger behavioural matrices without an independent analytical result
    * - ``testing/benchmarking``
      - Benchmark drivers
      - Runtime, memory, throughput, and scaling measurements
    * - ``reframe_tests``
      - ReFrame
      - Whole-model and HPC regression campaigns, including MPI and task-farm
        configurations

The ``testing/models_basic`` and ``testing/models_pmls`` directories contain
legacy compact model collections used by the historical regression workflow.
They can be migrated into the categories above as that workflow is
modernised.

Automated pytest suite
======================

``tests`` is the standard pytest test-suite directory. Install the development
requirements and run the usual CPU selection from the repository root:

.. code-block:: console

    $ python -m pip install -r requirements.txt
    $ python -m pytest -m "not gpu and not slow"

This is also the selection run by the GitHub Actions workflow for pull
requests and pushes to ``devel``. Run the complete locally available suite
with:

.. code-block:: console

    $ python -m pytest

The markers registered in ``pyproject.toml`` are:

``integration``
    Exercises multiple gprMax components together or runs a complete compact
    model. The automated Hertzian-dipole and PEC-sphere comparisons are
    examples.

``gpu``
    Executes on a real CUDA device. Tests that inspect generated GPU source or
    use mocks do not receive this marker because they remain ordinary CPU
    tests.

``slow``
    Normally takes more than ten seconds on a development machine.

Markers may overlap. Useful selections include:

.. code-block:: console

    $ python -m pytest -m integration
    $ python -m pytest -m slow
    $ python -m pytest -m gpu --gpu-device 0
    $ python -m pytest --durations=25

The CUDA index can alternatively be supplied through ``GPRMAX_TEST_GPU``.
Real-device tests skip when the selected device is unavailable. The shared
pytest configuration also selects the local ``shm`` OFI provider by default on
Linux, avoiding unrelated MPI provider initialisation failures during test
collection while preserving an explicit user setting.

:download:`The pytest README <../../tests/README.rst>` contains the concise
command reference, and :download:`the workflow
<../../.github/workflows/pytest.yml>` records the automated CI environment.

Adding pytest coverage
----------------------

Prefer the smallest test that can detect the failure:

* use a unit test for a formula, parser, array transformation, or generated
  kernel contract;
* use ``integration`` when the behaviour requires a compact production FDTD
  run or several gprMax subsystems;
* add ``gpu`` only when real device execution is essential, and obtain the
  device index from the ``gpu_device`` fixture rather than hard-coding zero;
* add ``slow`` when the measured runtime normally exceeds ten seconds; and
* write all temporary solver output below pytest's ``tmp_path`` or
  ``tmp_path_factory`` location.

Run ``python -m pytest --collect-only`` after adding markers. Strict-marker
checking is enabled, so misspelled or unregistered markers fail collection.
New tests should also pass the repository's Black, isort, and pre-commit
checks.

Analytical validation
=====================

The higher-resolution drivers in ``testing/validation`` compare normal-incidence
plane-wave reflection, Hertzian-dipole fields and antenna metrics, rational
lumped-network reflection, and PEC or dielectric sphere RCS with independent
Fresnel, dipole, circuit/TEM-guide, and Mie solutions.
The dispersive studies compare exact pole-residue interface averaging with
the non-averaged staircased representation for planar layers, homogeneous
spheres, and a Debye-core/Lorentz-shell sphere.
They write CSV data, PNG plots, ``summary.json`` acceptance results, and a
human-readable report. Run, for example:

.. code-block:: console

    $ python -m testing.validation.validate_hertzian_dipole --gpu 0
    $ python -m testing.validation.validate_rational_network_literature
    $ python -m testing.validation.validate_pec_sphere_rcs --gpu 0
    $ python -m testing.validation.validate_debye_sphere_averaging --gpu 0
    $ python -m testing.validation.dispersive_averaging.validate_multilayer_fdtd
    $ python -m testing.validation.dispersive_averaging.validate_core_shell_fdtd --gpu 0

Omit ``--gpu`` to use the CPU solver. These full-resolution cases can take
minutes and are intentionally not all part of routine CI. Compact FDTD
counterparts are retained in pytest where they provide practical regression
coverage.

See :ref:`Analytical comparisons <analytical-comparisons>` for the formulae,
plots, and current quantitative results, and :download:`the validation README
<../../testing/validation/README.rst>` for commands and output policy.

Other-code and diagnostic studies
=================================

``testing/other_codes`` contains reproducible inter-code comparisons. The
current MATLAB Antenna Toolbox suite covers dipole, monopole, bow-tie, patch,
helix, array, port, and PEC-plate RCS models. Retained MAT files allow the
Python comparison scripts to be rerun without access to MATLAB. These studies
report agreement and modelling differences but do not define correctness
solely from another numerical solver.

See :ref:`Numerical comparisons <numerical-comparisons>` for representative
plots and links to every case. Backend-only parity studies belong under
``testing/backend_consistency``; larger diagnostic matrices without an exact
reference belong under ``testing/regression``.

Generated-data policy
=====================

Source-controlled evidence should be compact and reviewable: scripts, input
models, analytical or independently generated reference data, CSV tables,
JSON metrics, reports, and selected plots. Large gprMax HDF5 outputs, VTK
geometry, NumPy caches, and Python bytecode are reproducible working products
and should normally remain ignored. A saved output from an earlier gprMax run
is regression data, not an independent analytical reference.

ReFrame suite
=============

ReFrame is retained for whole-model and HPC regression testing. It launches
complete gprMax jobs, manages machine configurations and dependencies, and can
exercise serial, MPI, task-farm, snapshot, geometry, and Python-API variants.
It is not a replacement for focused pytest unit tests or analytical
validation. See the :doc:`ReFrame Test Suite <reframe_test_suite>` page for
installation, filtering, test classes, mixins, and regression checks.
