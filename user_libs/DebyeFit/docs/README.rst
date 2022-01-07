User libraries is a sub-package where useful Python modules contributed by users are stored.

********
DebyeFit
********

Information
===========

**Author/Contact**: Iraklis Giannakis (iraklis.giannakis@abdn.ac.uk), University of Aberdeen, UK and Sylwia Majchrowska (Sylwia.Majchrowska1993@gmail.com)

This module was created as part of the `Google Summer of Code <https://summerofcode.withgoogle.com/>`_ programme 2021 which gprMax participated.

**License**: `Creative Commons Attribution-ShareAlike 4.0 International License <http://creativecommons.org/licenses/by-sa/4.0/>`_

**Attribution/cite**: Giannakis, I., & Giannopoulos, A. (2014). A novel piecewise linear recursive convolution approach for dispersive media using the finite-difference time-domain method. *IEEE Transactions on Antennas and Propagation*, 62(5), 2669-2678. (http://dx.doi.org/10.1109/TAP.2014.2308549)

Electric permittivity is a complex function with both real and imaginary parts.
In general, as a hard and fast rule, the real part dictates the velocity of the medium while the imaginary part is related to the electromagnetic losses.
The generic form of dispersive media is

.. math::

   \epsilon(\omega) = \epsilon^{'}(\omega) - j\epsilon^{''}(\omega),

where :math:`\omega` is the angular frequency, :math:`\epsilon^{'}` and :math:`\epsilon^{''}` are the real and imaginary parts of the permittivity respectively. 

This package provides scripts and tools which can be used to fit a multi-Debye expansion to dielectric data, defined as

.. math::

   \epsilon(\omega) = \epsilon_{\infty} + \sum_{i=1}^{N}\frac{\Delta\epsilon_{i}}{1+j\omega t_{0,i}},

where :math:`\epsilon(\omega)` is frequency dependent dielectric permittivity, :math:`\Delta\epsilon` - difference between the real permittivity at zero and infinite frequency.
:math:`\tau_{0}` is relaxation time (seconds),  :math:`\epsilon_{\infty}` - real part of relative permittivity at infinite frequency, and :math:`N` is number of the Debye poles.

To fit the data to a multi-Debye expansion, you can choose between Havriliak-Negami, Jonscher, or Complex Refractive Index Mixing (CRIM) models, as well as arbitrary dielectric data derived experimentally or calculated using a different function.

.. figure:: images/epsilon.png
    :width: 600 px

    Real and imaginary parts of frequency-dependent permittivity


Package contents
================

There are two main scripts:

* ```Debye_fit.py``` contains definitions of all Relaxation functions classes
* ```optimization.py``` contains definitions of three choosen global optimization methods


Relaxation Class
----------------

This class is designed for modelling different relaxation functions, like Havriliak-Negami (```HavriliakNegami```), Jonscher (```Jonscher```), Complex Refractive Index Mixing (```CRIM```) models, and arbitrary dielectric data derived experimentally or calculated using some other function (```Rawdata```).

More about the ``Relaxation`` class structure can be found in the :doc:`Relaxation doc <relaxation.rst>`.

Havriliak-Negami Function
^^^^^^^^^^^^^^^^^^^^^^^^^

The Havriliak-Negami relaxation is an empirical modification of the Debye relaxation model in electromagnetism, which in addition to the Debye equation has two exponential parameters

.. math::

    \epsilon(\omega) = \epsilon_{\infty} + \frac{\Delta\epsilon}{\left(1+\left(j\omega t_{0}\right)^{a}\right)^{b}}


The ``HavriliakNegami`` class has the following structure:

.. code-block:: none

    HavriliakNegami(f_min, f_max,
                    alpha, beta, e_inf, de, tau_0,
                    sigma, mu, mu_sigma, material_name,
                    number_of_debye_poles=-1, f_n=50,
                    plot=False, save=True,
                    optimizer=PSO_DLS,
                    optimizer_options={})


* ``f_min`` is first bound of the frequency range used to approximate the given function (Hz),
* ``f_max`` is second bound of the frequency range used to approximate the given function (Hz),
* ``alpha`` is real positive float number which varies 0 < $\alpha$ < 1,
* ``beta`` is real positive float number which varies 0 < $\beta$ < 1,
* ``e_inf`` is a real part of relative permittivity at infinite frequency,
* ``de`` is a difference between the real permittivity at zero and infinite frequency,
* ``tau_0`` is a relaxation time (seconds),
* ``sigma`` is a conductivity (Siemens/metre),
* ``mu`` is a relative permeability,
* ``mu_sigma`` is a magnetic loss (Ohms/metre),
* ``material_name`` is the material name,
* ``number_of_debye_poles`` is the chosen number of Debye poles,
* ``f_n`` is the chosen number of frequences,
* ``plot`` is a switch to turn on the plotting,
* ``save`` is a switch to turn on saving final material properties,
* ``optimizer`` is a chosen optimizer to fit model to dielectric data,
* ``optimizer_options`` is a dict for options of chosen optimizer.

Jonscher Function
^^^^^^^^^^^^^^^^^

Jonscher function is mainly used to describe the dielectric properties of concrete and soils. The frequency domain expression of Jonscher
function is given by

.. math::

    \epsilon(\omega) = \epsilon_{\infty} + a_{p}*\left( -j*\frac{\omega}{\omega_{p}} \right)^{n}


The ``Jonscher`` class has the following structure:

.. code-block:: none

    Jonscher(f_min, f_max,
            e_inf, a_p, omega_p, n_p,
            sigma, mu, mu_sigma,
            material_name, number_of_debye_poles=-1,
            f_n=50, plot=False, save=True,
            optimizer=PSO_DLS,
            optimizer_options={})


* ``f_min`` is first bound of the frequency range used to approximate the given function (Hz),
* ``f_max`` is second bound of the frequency range used to approximate the given function (Hz),
* ``e_inf`` is a real part of relative permittivity at infinite frequency,
* ``a_p``` is a Jonscher parameter. Real positive float number,
* ``omega_p`` is a Jonscher parameter. Real positive float number,
* ``n_p`` Jonscher parameter, 0 < n_p < 1.

Complex Refractive Index Mixing (CRIM) Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CRIM is the most mainstream approach for estimating the bulk permittivity of heterogeneous materials and has been widely applied for GPR applications. The function takes form of

.. math::

    \epsilon(\omega)^{d} = \sum_{i=1}^{m}f_{i}\epsilon_{m,i}(\omega)^{d}


The ``CRIM`` class has the following structure:

.. code-block:: none

    CRIM(f_min, f_max, a, volumetric_fractions,
        materials, sigma, mu, mu_sigma, material_name, 
        number_of_debye_poles=-1, f_n=50,
        plot=False, save=True,
        optimizer=PSO_DLS,
        optimizer_options={})


* ``f_min`` is first bound of the frequency range used to approximate the given function (Hz),
* ``f_max`` is second bound of the frequency range used to approximate the given function (Hz),
* ``a`` is a shape factor,
* ``volumetric_fractions`` is a volumetric fraction for each material,
* ``materials`` are arrays of materials properties, for each material [e_inf, de, tau_0].

Rawdata Class
^^^^^^^^^^^^^

This package also has the ability to model dielectric properties obtained experimentally by fitting multi-Debye functions to data given from a file.
The format of the file should be three columns: the first column contains the frequencies (Hz) associated with the electric permittivity; the second column contains the real part of the relative permittivity; the third column contains the imaginary part of the relative permittivity. The columns should separated by a coma by default, but it is also possible to define a different separator.

The ``Rawdata`` class has the following structure:

.. code-block:: none

    Rawdata(self, filename,
            sigma, mu, mu_sigma,
            material_name, number_of_debye_poles=-1,
            f_n=50, delimiter =',',
            plot=False, save=True,
            optimizer=PSO_DLS,
            optimizer_options={})


* ``filename`` is a path to text file which contains three columns,
* ``delimiter`` is a separator for three data columns.

Class Optimizer
---------------

This class supports global optimization algorithms (particle swarm, dual annealing, evolutionary algorithms) for finding an optimal set of relaxation times that minimise the error between the actual and the approximated electric permittivity, and calculates optimised weights for the given relaxation times.
Code written here is mainly based on external libraries, like ```scipy``` and ```pyswarm```.

More about the ``Optimizer`` class structure can be found in the :doc:`Optimisation doc <optimisation.rst>`.

PSO_DLS Class
^^^^^^^^^^^^^

Creation of hybrid Particle Swarm-Damped Least Squares optimisation object with predefined parameters.
The code is a modified version of the pyswarm package which can be found at https://pythonhosted.org/pyswarm/.

DA_DLS Class
^^^^^^^^^^^^

Creation of Dual Annealing-Damped Least Squares optimisation object with predefined parameters. The class is a modified version of the scipy.optimize package which can be found at:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing.

DE_DLS Class
^^^^^^^^^^^^

Creation of Differential Evolution-Damped Least Squares object with predefined parameters. The class is a modified version of the scipy.optimize package which can be found at:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution.

DLS function
^^^^^^^^^^^^

Finding the weights using a non-linear least squares (LS) method, the Levenberg-Marquardt algorithm (LMA or just LM), also known as the damped least-squares (DLS) method.

How to use the package
======================

Examples
--------

In the examples directory you will find Jupyter notebooks, scripts, and data that demonstrate different cases of how to use the main script ```DebyeFit.py```:

* ```example_DebyeFitting.ipynb```: simple cases of using all available implemented relaxation functions,
* ```example_BiologicalTissues.ipynb```: simple cases of using Cole-Cole function for biological tissues,
* ```example_ColeCole.py```: simple cases of using Cole-Cole function in case of 3, 5 and automatically chosen number of Debye poles,
* ```Test.txt```: raw data for testing ```Rawdata``` class, file contains 3 columns: the first column contains the frequencies (Hz) associated with the value of the permittivity; the second column contains the real part of the relative permittivity; and the third column contains the imaginary part of the relative permittivity.

The following code shows a basic example of how to use the Havriliak-Negami function:

.. code-block:: python

    # set Havrilak-Negami function with initial parameters
    setup = HavriliakNegami(f_min=1e4, f_max=1e11,
                            alpha=0.3, beta=1,
                            e_inf=3.4, de=2.7, tau_0=.8e-10,
                            sigma=0.45e-3, mu=1, mu_sigma=0,
                            material_name="dry_sand", f_n=100,
                            plot=True, save=False,
                            number_of_debye_poles=3,
                            optimizer_options={'swarmsize':30,
                                               'maxiter':100,
                                               'omega':0.5,
                                               'phip':1.4,
                                               'phig':1.4,
                                               'minstep':1e-8,
                                               'minfun':1e-8,
                                               'seed': 111,
                                               'pflag': True})
    # run optimization
    setup.run()