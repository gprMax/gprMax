User libraries is a sub-package where useful Python modules contributed by users are stored.

**UNDER CONSTRUCTION**

**********************
Optimisation - Taguchi
**********************

.. code-block:: python

    # Copyright (C) 2015-2016, Craig Warren
    #
    # This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
    # To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
    #
    # Please use the attribution at http://dx.doi.org/10.1190/1.3548506

The module features an optimisation technique based on Taguchi's method. It allows the user to define parameters in an input file and optimise their values based on a user-defined fitness function.

Taguchi's method
================

Taguchi's method is based on the concept of Orthogonal Array (OA) and has the following advantages:

* Simple to implement
* Effective in reduction of experiments
* Fast convergence speed
* Global optimum results
* Independence from initial values of optimisation parameters

Details of Taguchi's method in the context of electromagnetics can be found in [WEN2007a]_ and [WEN2007b]_. The process by which Taguchi's method optimises parameters is illustrated in the following figure.

.. figure:: images/taguchi_process.png
    :width: 300 px

    Process associated with Taguchi's method.

Implementation
==============

The module will select from 2 pre-built OAs (http://neilsloane.com/oadir/) depending on the number of parameters to optimise. Currently, up to 7 independent parameters can be optimised, although a method to construct OAs of any size is under testing.

Fitness functions
-----------------

A fitness function is required