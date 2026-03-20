.. _python-api:

******************************
Using gprMax as a Python Library
******************************

gprMax can be used directly as a Python library via its ``api()`` function. This allows you to run simulations programmatically from Python scripts, notebooks, or other tools — without using the command line.

.. code-block:: python

    from gprMax.gprMax import api


API Reference
=============

.. code-block:: python

    api(
        inputfile,
        n=1,
        task=None,
        restart=None,
        mpi=False,
        mpi_no_spawn=False,
        mpicomm=None,
        gpu=None,
        benchmark=False,
        geometry_only=False,
        geometry_fixed=False,
        write_processed=False,
        opt_taguchi=False
    )

**Arguments**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``inputfile``
     - *(required)*
     - Path to the ``.in`` input file. Equivalent to the positional argument on the CLI.
   * - ``n``
     - ``1``
     - Number of times to run the model. Equivalent to CLI ``-n``. Useful for parameter sweeps where each run uses a different geometry via ``current_model_run``.
   * - ``task``
     - ``None``
     - Task number when using a job array (e.g. HPC). Equivalent to CLI ``--task``.
   * - ``restart``
     - ``None``
     - Run number to restart a previously interrupted batch from. Equivalent to CLI ``--restart``.
   * - ``mpi``
     - ``False``
     - Enable MPI task farming for multiple model runs. Equivalent to CLI ``--mpi``.
   * - ``mpi_no_spawn``
     - ``False``
     - Use MPI without spawning workers (advanced MPI usage). Equivalent to CLI ``--mpi-no-spawn``.
   * - ``mpicomm``
     - ``None``
     - An existing MPI communicator object to use (for embedding in MPI-aware applications).
   * - ``gpu``
     - ``None``
     - List of GPU device ID(s) to use, e.g. ``[0]`` or ``[0, 1]``. Equivalent to CLI ``--gpu``. Pass ``None`` to use CPU only.
   * - ``benchmark``
     - ``False``
     - Run in benchmarking mode to measure runtime scaling. Equivalent to CLI ``--benchmark``.
   * - ``geometry_only``
     - ``False``
     - Build and export the model geometry without running the simulation. Equivalent to CLI ``--geometry-only``.
   * - ``geometry_fixed``
     - ``False``
     - Optimise for a geometry that does not change between runs. Equivalent to CLI ``--geometry-fixed``.
   * - ``write_processed``
     - ``False``
     - Write a processed version of the input file after Python blocks are executed. Equivalent to CLI ``--write-processed``.
   * - ``opt_taguchi``
     - ``False``
     - Enable Taguchi optimisation mode. Equivalent to CLI ``--opt-taguchi``. See :ref:`user_libs_opt_taguchi`.


Examples
========

Single Run and Reading Results
-------------------------------

The simplest use case is running a single simulation and reading the output with the ``h5py`` library.

.. code-block:: python

    import h5py
    import numpy as np
    from gprMax.gprMax import api

    # Run the simulation
    api('my_model/cylinder_Ascan_2D.in', n=1)

    # Read the output file
    with h5py.File('my_model/cylinder_Ascan_2D.out', 'r') as f:
        # Read the Ez field component recorded at the receiver
        ez = f['rxs/rx1/Ez'][:]
        dt = f.attrs['dt']  # timestep in seconds
        iterations = f.attrs['Iterations']

    time = np.arange(0, iterations) * dt
    print(f'Simulation complete. {iterations} iterations, dt={dt:.3e}s')
    print(f'Ez shape: {ez.shape}')

.. note::

    The output ``.out`` file is written to the same directory as the input file by default. Make sure your working directory or input file path is set correctly before calling ``api()``.


Simple Parameter Sweep
-----------------------

You can run multiple simulations with different parameters by combining ``n`` with Python scripting in the input file. Each run can access ``current_model_run`` to vary geometry or material properties.

**Input file** (``sweep_model.in``):

.. code-block:: none

    #python:
    depth = 0.05 + (current_model_run - 1) * 0.01  # vary depth from 0.05m to 0.14m
    box(0.05, depth, 0, 0.10, depth + 0.02, 0.002, 'pec')
    #end_python:

**Python script**:

.. code-block:: python

    import h5py
    import numpy as np
    from gprMax.gprMax import api

    n_runs = 10
    results = []

    # Run all 10 simulations in one call
    api('sweep_model.in', n=n_runs)

    # Read each output file
    for i in range(1, n_runs + 1):
        filename = f'sweep_model{i}.out'
        with h5py.File(filename, 'r') as f:
            ez = f['rxs/rx1/Ez'][:]
            results.append(ez)

    results = np.array(results)
    print(f'Sweep complete. Results shape: {results.shape}')

.. tip::

    When running a sweep with ``n > 1``, output files are numbered sequentially: ``modelname1.out``, ``modelname2.out``, etc.


GPU Selection
--------------

To run on a specific GPU, pass its device ID as a list:

.. code-block:: python

    from gprMax.gprMax import api

    # Run on GPU 0
    api('my_model.in', n=1, gpu=[0])

    # Run on GPU 1
    api('my_model.in', n=1, gpu=[1])

    # Run multiple models across two GPUs (with MPI)
    api('my_model.in', n=4, mpi=True, gpu=[0, 1])

.. note::

    GPU support requires a CUDA-capable Nvidia GPU and PyCUDA to be installed. See :ref:`gpu` for full GPU setup instructions.


Best Practices
==============

**Working directory**

gprMax resolves input file paths and writes output files relative to the location of the input file. It is good practice to use absolute paths:

.. code-block:: python

    import os
    from gprMax.gprMax import api

    inputfile = os.path.join(os.path.dirname(__file__), 'models', 'my_model.in')
    api(inputfile, n=1)

**Error handling**

Wrap ``api()`` calls in a try/except block to handle simulation errors gracefully:

.. code-block:: python

    from gprMax.gprMax import api
    from gprMax.exceptions import GeneralError

    try:
        api('my_model.in', n=1)
    except GeneralError as e:
        print(f'gprMax error: {e}')
    except FileNotFoundError:
        print('Input file not found.')

**Geometry checking before running**

Use ``geometry_only=True`` to verify your model geometry is correct before committing to a full simulation run:

.. code-block:: python

    from gprMax.gprMax import api

    # Check geometry first
    api('my_model.in', geometry_only=True)

    # Then run the full simulation
    api('my_model.in', n=1)

**Logging**

gprMax prints simulation progress to stdout. To capture this output in a script or notebook, you can redirect stdout:

.. code-block:: python

    import sys
    import io
    from gprMax.gprMax import api

    log = io.StringIO()
    sys.stdout = log
    api('my_model.in', n=1)
    sys.stdout = sys.__stdout__

    print(log.getvalue())  # print captured log