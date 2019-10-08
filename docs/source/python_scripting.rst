.. _python-scripting:

************************
Scripting the input file
************************

INTEGRATE THIS WHICH API DOCS

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
