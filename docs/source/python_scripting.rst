.. _python-scripting:

****************
Python scripting
****************

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
* ``number_model_runs`` is the total number of runs specified when the model was initially executed, i.e. from ``python -m gprMax my_input_file -n number_of_model_runs``
* ``inputdirectory`` is the path to the directory where your input file is located.

Antenna models
==============

You can also access a user library of antenna models. This library currently features models of antennas similar to:

* a Geophysical Survey Systems, Inc. (GSSI) 1.5 GHz (Model 5100) antenna (http://www.geophysical.com)
* a MALA Geoscience 1.2 GHz antenna (http://www.malags.com/)

These antenna models can be accessed from within a block of Python code in your simulation. For example, to use Python to include an antenna model similar to a GSSI 1.5 GHz antenna at a location 0.125m, 0.094m, 0.100m (x,y,z) using a 1mm spatial resolution:

.. code-block:: none

    #python:
    from user_libs.antennas import antenna_like_GSSI_1500
    antenna_like_GSSI_1500(0.125, 0.094, 0.100, 0.001)
    #end_python:

Functions for input commands
============================

To make it easier to create commands within a block of Python code, there is a built-in module which contains some of the most commonly used input commands in functional form. For example, to use Python to generate a series of cylinders in a model:

.. code-block:: none

    #python:
    from gprMax.input_cmd_funcs import *
    domain = domain(0.2 0.2 0.2)
    for x in range(0, 8)
        cylinder(0.02 + x * 0.02, 0.05, 0, 0.020 + x * 0.02, 0.05, domain[2], 0.005, 'pec’)
    #end_python:

The ``domain`` function will print the ``#domain`` command to the input file and return a variable with the extent of the domain that can be used elsewhere in a Python code block, e.g. in this case with the ``cylinder`` function. The ``cylinder`` function is just a functional version of the ``#cylinder`` command which prints it to the input file.

input_cmd_funcs.py
------------------

.. automodule:: gprMax.input_cmd_funcs



