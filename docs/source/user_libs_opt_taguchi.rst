User libraries is a sub-package where useful Python modules contributed by users are stored.

**UNDER CONSTRUCTION**

**********************
Optimisation - Taguchi
**********************

Information
===========

**Author/Contact**: Craig Warren (Craig.Warren@ed.ac.uk), University of Edinburgh

**License**: Creative Commons Attribution-ShareAlike 4.0 International License (http://creativecommons.org/licenses/by-sa/4.0/)

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

Details of Taguchi's method in the context of electromagnetics can be found in [WEN2007a]_ and [WEN2007b]_.


Package overview
================

.. code-block:: none

    OA_9_4_3_2.npy
    OA_18_7_3_2.npy
    optimisation_taguchi_fitness.py
    optimisation_taguchi_plot.py

* ``OA_9_4_3_2.npy`` and ``OA_18_7_3_2.npy`` are NumPy archives containing pre-built OAs from http://neilsloane.com/oadir/
* ``optimisation_taguchi_fitness.py`` is a module containing fitness functions. There are some pre-built ones but users should add their own here.
* ``optimisation_taguchi_plot.py`` is a module for plotting the results, such as parameter values and convergence history, from an optimisation process when it has completed.

Implementation
--------------

The process by which Taguchi's method optimises parameters is illustrated in the following figure.

.. figure:: images/taguchi_process.png
    :width: 300 px

    Process associated with Taguchi's method.

In stage 1a, one of the 2 pre-built OAs will automatically be chosen depending on the number of parameters to optimise. Currently, up to 7 independent parameters can be optimised, although a method to construct OAs of any size is under testing.

In stage 1b, a fitness function is required to set a goal against which to compare results from the optimisation process. A number of pre-built fitness functions can be found in the ``optimisation_taguchi_fitness.py`` module, e.g. ``minvalue``, ``maxvalue`` and ``xcorr``. Users can also easily add their own fitness functions to this module. All fitness functions must take two arguments:

* ``filename`` a string containing the full path and filename of the output file
* ``args`` a dictionary which can contain any number of additional arguments for the function, e.g. names (IDs) of outputs (rxs) from input file

Additionally all fitness functions must return a single fitness value which the optimsation process will aim to maximise.

Stages 2-6 are iterated by the optimisation process.

Parameters and settings for the optimisation process are specified within a special Python block defined by ``#taguchi`` and ``#end_taguchi`` in the input file. The parameters to optimise must be defined in a dictionary named `optparams` and their initial ranges specified as lists with lower and upper values. The fitness function, it's parameters, and a stopping value are defined in dictionary named ``fitness`` which has keys for:

* ``name``, a string that is the name of the fitness function to be used.
* ``args``, a dictionary containing arguments to be passed to the fitness function. Within ``args`` there must be a key called ``outputs`` which contains a string or list of the names of one or more outputs in the model.
* ``stop``, a value which when exceeded the optimisation should stop.

Optionally a variable called ``maxiterations`` maybe specified within the ``#taguchi`` block which will set a maximum number of iterations after which the optimisation process will terminate irrespective of any other criteria.


How to use the package
======================

The package requires ``#python`` and ``#end_python`` to be used in the input file, as well as ``#taguchi`` and ``#end_taguchi`` for specifying parameters and setting for the optimisation process. A Taguchi optimisation is run using the command line option ``--optimisation-taguchi``.

Example
-------