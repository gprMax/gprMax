User libraries is a sub-package where useful Python modules contributed by users are stored.

**UNDER CONSTRUCTION**

**********************
Optimisation - Taguchi
**********************

Information
===========

Author/Contact: Craig Warren (Craig.Warren@ed.ac.uk), University of Edinburgh

License: Creative Commons Attribution-ShareAlike 4.0 International License (http://creativecommons.org/licenses/by-sa/4.0/)

.. code-block:: python

    # Copyright (C) 2015-2016, Craig Warren
    #
    # This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
    # To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
    #
    # Please use the attribution at http://dx.doi.org/10.1190/1.3548506

The package features an optimisation technique based on Taguchi's method. It allows the user to define parameters in an input file and optimise their values based on a fitness function.


Taguchi's method
----------------

Taguchi's method is based on the concept of the Orthogonal Array (OA) and has the following advantages:

* Simple to implement
* Effective in reduction of experiments
* Fast convergence speed
* Global optimum results
* Independence from initial values of optimisation parameters

Details of Taguchi's method in the context of electromagnetics can be found in [WEN2007a]_ and [WEN2007b]_. The process by which Taguchi's method optimises parameters is illustrated in the following figure.

.. figure:: images/taguchi_process.png
    :width: 300 px

    Process associated with Taguchi's method.


Package overview
================

.. code-block:: none

    OA_9_4_3_2.npy
    OA_18_7_3_2.npy
    optimisation_taguchi_fitness.py
    optimisation_taguchi_plot.py

* `OA_9_4_3_2.npy` and `OA_18_7_3_2.npy` are NumPy archive containing pre-built OAs from http://neilsloane.com/oadir/
* `optimisation_taguchi_fitness.py` is a module containing fitness functions. There are some pre-built ones but users should add their own here.
* `optimisation_taguchi_plot.py` is a module for plotting the results, such as parameter values and convergence history, from an optimisation process when it has completed.


How to use the package
======================

Parameters to optimise
----------------------

The module will select from 2 pre-built OAs (http://neilsloane.com/oadir/) depending on the number of parameters to optimise. Currently, up to 7 independent parameters can be optimised, although a method to construct OAs of any size is under testing.


Fitness functions
-----------------

A fitness function is required to set a goal against which to compare results from the optimisation process. A number of pre-built fitness functions can be found in the `optimisation_taguchi_fitness.py` module, such as `minvalue`, `maxvalue` and `xcorr`. Users can easily add their own fitness functions to this module. All fitness functions must take two arguments and return a single fitness value which will be maximised. The arguments must be:

* `filename` a string containing the full path and filename of the output file
* `args` a dictionary which can contain any number of additional arguments for the function, e.g. names (IDs) of outputs (rxs) from input file


Example
-------