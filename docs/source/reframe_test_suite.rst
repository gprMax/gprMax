******************
ReFrame Test Suite
******************

gprMax includes a test suite built using `ReFrame <https://reframe-hpc.readthedocs.io>`_.

This is not a unit testing framework, instead it provides a mechanism to perform regression checks on whole model runs. Reference files for regression checks are automatically generated and stored in ``reframe_tests/regression_checks``.

.. attention::

    The regression checks are sensitive to floating point precision errors so are currently specific to the `ARCHER2 <https://www.archer2.ac.uk/>`_ system. Additional work is required to make them portable between systems.

Run the test suite
==================

Running the test suite requires ReFrame to be installed:

.. code-block:: console

    $ pip install reframe-hpc

The full test suite can be run with:

.. code-block:: console

    $ cd reframe_tests
    $ reframe -c tests/ -r

If you are running on a HPC system, you will need to be provide a configuration file:

.. code-block:: console

    $ reframe -C configuration/archer2_settings.py -c tests/ -r

A ReFrame configuration script for `ARCHER2 <https://www.archer2.ac.uk/>`_ is provided in the ``reframe_tests/configuration`` folder. Configurations for additional machines can be added here.

.. tip::

    The full test suite is quite large. ReFrame provides a number of ways to filter the tests you want to run such as ``-n`` and ``-t`` (by name and tag respectively). There is much more information in the `ReFrame documentation <https://reframe-hpc.readthedocs.io/en/stable/manpage.html#test-filtering>`_.

There is also an example job submission script for running the suite as a long running job on ARCHER2. Any additional arguments are passed forwarded to ReFrame. E.g.

.. code-block:: console

    $ sbatch job_scripts/archer2_tests.slurm -n Snapshot

would run all tests with "Snapshot" in the test name.

Developer guide
===============

Tests are defined in the ``reframe_tests/tests`` folder with gprMax input files stored in ``reframe_tests/tests/src``.

Base test classes
-----------------

Every regression test inherits from the :ref:`GprMaxBaseTest` class. This class contains all the logic for launching a gprMax job, checking the simulation completed, and running any regression checks.

Additionally, every test depends on the :ref:`CreatePyenvTest` class. It creates a new Python environment that all other tests will use.

.. currentmodule:: reframe_tests.tests.base_tests

.. autosummary::
    :template: class.rst
    :toctree: developer_reference
    :nosignatures:

    CreatePyenvTest
    GprMaxBaseTest

.. tip::

    Avoid rebuilding the Python environment every time you run the test suite by running ReFrame with the ``--restore-session`` flag.

Adding a new test
-----------------

The easiest way to learn how to write a new test is by looking at the existing tests. The test below runs the B-scan model provided with gprMax:

.. code-block:: python

    import reframe as rfm
    from reframe.core.builtins import parameter

    from reframe_tests.tests.mixins import BScanMixin, ReceiverMixin

    @rfm.simple_test
    class TestBscan(BScanMixin, ReceiverMixin, GprMaxBaseTest):
        tags = {"test", "serial", "bscan"}
        sourcesdir = "src/bscan_tests"
        model = parameter(["cylinder_Bscan_2D"])
        num_models = parameter([64])

- ``@rfm.simple_test`` - marks the class as a ReFrame test.
- :ref:`BScanMixin` and :ref:`ReceiverMixin` - mixin classes alter the behaviour to test specific gprMax functionality.
- ``tags`` - set tags that can be used to filter tests.
- ``sourcesdir`` - path to test source directory.
- ``model`` - gprMax input filename (without file extension). This is a ReFrame parameter so it can take multiple values to run the same test with multiple input files.
- ``num_models`` - parameter specific to the :ref:`BScanMixin`.

Test dependencies
-----------------

Tests can also define a test dependency. This uses the ReFrame test dependency mechanism to link tests. The dependent test can access the resources and outputs of the test it depends on. This means we can create a test that should produce an identical result to another test, but is configured differently. For example, to test the MPI domain decomposition functionality using the previous B-scan model, we can add:

.. code-block:: python

    from reframe_tests.tests.mixins import MpiMixin

    @rfm.simple_test
    class TestBscanMPI(MpiMixin, TestBscan):
        tags = {"test", "mpi", "bscan"}
        mpi_layout = parameter([[2, 2, 1]])
        test_dependency = TestBscan

- Our new class inherits from the above ``TestBscan`` class.
- Use the :ref:`MpiMixin` to run with the gprMax domain decomposition functionality.
- Override ``tags``.
- ``mpi_layout`` - parameter specific to the :ref:`MpiMixin`.
- ``test_dependency`` - Depend on the ``TestBscan`` class. It is not sufficient to just inherit from the class. The output from this test will compared with the output from the ``TestBscan`` test.

.. note::

    Some parameters, such as ``model`` are unified between test dependencies. I.e. the dependent test and test dependency will have the same value for the parameter.

    If a mixin class adds a new parameter, this may need to be unified as well. For an example of how to do this, see the :ref:`BscanMixin` class and the ``num_models`` parameter.

Mixin classes
-------------

The different mixin classes are used to alter the behaviour of a given test to support testing gprMax functionality - snapshots, geometry objects, geometry views - and different runtime configurations such as task farms, MPI, and the Python API.

.. important::

    When creating a new test, the mixin class must be specified earlier in the inheritance list than the base ReFrame test class::

        class TestAscan(ReceiverMixin, GprMaxBaseTest):
            pass

.. currentmodule:: reframe_tests.tests.mixins

.. autosummary::
    :template: class_stub.rst
    :toctree: developer_reference
    :nosignatures:

    BScanMixin
    GeometryObjectsReadMixin
    GeometryObjectsWriteMixin
    GeometryOnlyMixin
    GeometryViewMixin
    MpiMixin
    PythonApiMixin
    ReceiverMixin
    SnapshotMixin
    TaskfarmMixin

Regression checks
-----------------

.. note::

    To make the test suite portable between systems, the main changes would be to these regression check classes. Specifically the way hdf5 files are compared.

There are a number of classes that perform regression checks.

.. currentmodule:: reframe_tests.tests.regression_checks

.. autosummary::
    :template: class.rst
    :toctree: developer_reference
    :nosignatures:

    RegressionCheck
    H5RegressionCheck
    ReceiverRegressionCheck
    GeometryObjectRegressionCheck
    GeometryObjectMaterialsRegressionCheck
    GeometryViewRegressionCheck
    SnapshotRegressionCheck
