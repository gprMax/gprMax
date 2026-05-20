.. _python-scripting:

************************
Scripting the input file
************************

The input file has now been made scriptable by permitting blocks of Python code to be specified between ``#python`` and ``#end_python`` commands. The code is executed when the input file is read by gprMax. You don't need any external tools, such as MATLAB, to generate larger, more complex input files for building intricate models. Python scripting means that gprMax now includes :ref:`libraries of more complex objects, such as antennas <antennas>`, that can be easily inserted into a model. You can also access a number of built-in constants from your Python code.

Constants/variables
===================

You can access the following built-in constants from your Python code:

* ``c`` is the speed of light in vacuum :math:`c=2.9979245 \times 10^8` m/s
* ``e0`` is the permittivity of free space :math:`\epsilon_0=8.854187 \times 10^{-12}` F/m
* ``m0`` is the permeability of free space :math:`\mu_0=1.256637 \times 10^{-6}` H/m
* ``z0`` is the impedance of free space :math:`z_0=376.7303134` Ohms

You can access the following built-in variables from your Python code:

* ``current_model_run`` is the current run number of the model that is been executed.
* ``inputfile`` is the path and name of the input file.
* ``number_model_runs`` is the total number of runs specified when the model was initially executed, i.e. from ``python -m gprMax my_input_file -n number_of_model_runs``


Functions for input commands
============================

To make it easier to create commands within a block of Python code, there is a built-in module which contains some of the most commonly used input commands in functional form. For example, to use Python to generate a series of cylinders in a model:

.. code-block:: none

    #python:
    from gprMax.input_cmd_funcs import *
    domain = domain(0.2, 0.2, 0.2)
    for x in range(0, 8):
        cylinder(0.02 + x * 0.02, 0.05, 0, 0.020 + x * 0.02, 0.05, domain[2], 0.005, 'pec’)
    #end_python:

The ``domain`` function will print the ``#domain`` command to the input file and return a variable with the extent of the domain that can be used elsewhere in a Python code block, e.g. in this case with the ``cylinder`` function. The ``cylinder`` function is just a functional version of the ``#cylinder`` command which prints it to the input file.

Running gprMax from the Python API
===================================

Instead of using the command line, gprMax can be run directly from a Python script or notebook using the :mod:`gprMax.api` module. This is useful for parameter sweeps, automated workflows, and building interactive tools such as reactive notebooks.

Basic A-scan
------------

.. code-block:: python

    from gprMax.api import run

    # Run a single A-scan simulation
    run('user_models/cylinder_Ascan_2D.in', n=1, geometry_only=False)

Running a B-scan
----------------

A B-scan is made up of multiple A-scan runs with the source and receiver
stepping along a profile. Use the ``n`` parameter to set the number of traces:

.. code-block:: python

    from gprMax.api import run

    # Run 30 traces for a B-scan
    run('user_models/cylinder_Bscan_2D.in', n=30)

Checking geometry only
-----------------------

To visualise the model geometry without running the full FDTD simulation:

.. code-block:: python

    from gprMax.api import run

    run('user_models/cylinder_Ascan_2D.in', n=1, geometry_only=True)

Reading output files
--------------------

Simulation results are saved as HDF5 ``.out`` files. Field components are
stored under ``/rxs/rx1/`` and can be read with ``h5py``:

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File('user_models/cylinder_Ascan_2D.out', 'r') as f:
        ez = np.array(f['/rxs/rx1/Ez'])   # Ez field component
        dt = f.attrs['dt']                 # time step in seconds

    print(f'Ez shape:  {ez.shape}')
    print(f'Time step: {dt:.3e} s')

Plotting an A-scan
------------------

.. code-block:: python

    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    with h5py.File('user_models/cylinder_Ascan_2D.out', 'r') as f:
        ez = np.array(f['/rxs/rx1/Ez'])
        dt = f.attrs['dt']

    time = np.arange(len(ez)) * dt * 1e9  # convert to nanoseconds
    plt.plot(time, ez)
    plt.xlabel('Time (ns)')
    plt.ylabel('Ez field strength (V/m)')
    plt.title('A-scan')
    plt.grid(True)
    plt.show()