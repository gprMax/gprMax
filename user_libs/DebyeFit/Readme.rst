Fitting multi-pole Debye model to dielectric data
=================================================

In the ``user_libs`` sub-package is a module called ``DebyeFit`` which can be used to to fit a multi-Debye expansion to dielectric data, defined as

.. math::

   \epsilon(\omega) = \epsilon_{\infty} + \sum_{i=1}^{N}\frac{\Delta\epsilon_{i}}{1+j\omega t_{0,i}},

where :math:`\epsilon(\omega)` is frequency dependent dielectric properties, :math:`\Delta\epsilon` - difference between the real permittivity at zero and infinity frequency.
:math:`\tau_{0}` is relaxation time,  :math:`\epsilon_{\infty}` - real part of relative permittivity at infinity frequency, and :math:`N` is number of the Debye poles.

The user can choose between Havriliak-Negami, Jonsher, Complex Refractive Index Mixing models, and arbitrary dielectric data derived experimentally
or calculated using some other function.

Havriliak-Negami Function
#########################

The Havriliak–Negami relaxation is an empirical modification of the Debye relaxation model in electromagnetism, which in additionto the Debye equation has two exponential parameters

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
* ``e_inf`` is a real part of relative permittivity at infinity frequency,
* ``de`` is a difference between the real permittivity at zero and infinity frequency,
* ``tau_0`` is a relaxation time,
* ``sigma`` is a conductivity (Siemens/metre),
* ``mu`` is a relative permeability,
* ``mu_sigma`` is a magnetic loss,
* ``material_name`` is definition of material name,
* ``number_of_debye_poles`` is choosen number of Debye poles,
* ``f_n`` is choosen number of frequences,
* ``plot`` is a switch to turn on the plotting,
* ``save`` is a switch to turn on the saving final material properties,
* ``optimizer`` is a choosen optimizer to fit model to dielectric data,
* ``optimizer_options`` is a dict for options of choosen optimizer.

Jonsher Function
################

Jonscher function is mainly used to describe the dielectric properties of concrete and soils. The frequency domain expression of Jonscher
function is given by

.. math::

    \epsilon(\omega) = \epsilon_{\infty} - a_{p}*\left( -j*\frac{\omega}{\omega_{p}} \right)^{n}


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
* ``e_inf`` is a real part of relative permittivity at infinity frequency,
* ``a_p``` is a Jonscher parameter. Real positive float number,
* ``omega_p`` is a Jonscher parameter. Real positive float number,
* ``n_p`` Jonscher parameter, 0 < n_p < 1.

Complex Refractive Index Mixing (CRIM) Function
###############################################

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
#############

The present package has the ability to model dielectric properties obtained experimentally by fitting multi-Debye functions to data given from a file.
The format of the file should be three columns. The first column contains the frequencies (Hz) associated with the electric permittivity point.
The second column contains the real part of the relative permittivity. The third column contains the imaginary part of the relative permittivity.
The columns should separated by coma by default (is it posible to define different separator).

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

Code structure
==============

The ``user_libs`` sub-package contains two main scripts:

* ```Debye_fit.py``` with definition of all Relaxation functions classes,
* ```optimization.py``` with definition of three choosen global optimization methods.

Examples
########

In directory ```examples```, we provided jupyter notebooks, scripts and data to show how use stand alone script ```DebyeFit.py```:

* ```example_DebyeFitting.ipynb```: simple cases of using all available implemented relaxation functions,
* ```example_BiologicalTissues.ipynb```: simple cases of using Cole-Cole function for biological tissues,
* ```example_ColeCole.py```: simple cases of using Cole-Cole function in case of 3, 5 and automatically chosen number of Debye poles,
* ```Test.txt```: raw data for testing ```Rawdata Class```, file contains 3 columns: the first column contains the frequencies (Hz) associated with the value of the permittivity, second column contains the real part of the relative permittivity, and the third one the imaginary part of the relative permittivity. The columns should separated by comma.

Dispersive material commands
============================

gprMax has implemented an optimisation approach to fit a multi-Debye expansion to dielectric data.
The user can choose between Havriliak-Negami, Johnsher and Complex Refractive Index Mixing models, fit arbitrary dielectric data derived experimentally or calculated using some other function.
Notice that Havriliak-Negami is an inclusive function that holds as special cases the widely-used **Cole-Cole** and **Cole-Davidson** functions.

.. note::

    The technique employed here as a default is a hybrid linear-nonlinear optimisation proposed by Kelley et. al (2007).
    Their method was slightly adjusted to overcome some instability issues and thus making the process more robust and faster.
    In particular, in the case of negative weights we inverse the sign in order to introduce a large penalty in the optimisation process thus indirectly constraining the weights
    to be always positive. This made dumbing factors unnecessary and consequently they were removed from the algorithm. Furthermore we added the real part to the cost action
    to avoid possible instabilities to arbitrary given functions that does not follow the Kramers–Kronig relationship.

.. warning::

    * The fitting accuracy depends on the number of the Debye poles as well as the fitted function. It is advised to check if the resulted accuracy is sufficient for your application. 
	* Increasing the number of Debye poles will make the approximation more accurate but it will increase the overall computational resources of FDTD.

#HavriliakNegami:
#################

Allows you to model dielectric properties by fitting multi-Debye functions to Havriliak-Negami function. The syntax of the command is:

.. code-block:: none

    #HavriliakNegami: f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 i1 str1 [i2]

* ``f1`` is the lower frequency bound (in Hz).
* ``f2`` is the upper frequency bound (in Hz).
* ``f3`` is the :math:`\alpha` parameter beetwen bonds :math:`\left(0 < \alpha < 1 \right)`.
* ``f4`` is the :math:`\beta` parameter beetwen bonds :math:`\left(0 < \beta < 1 \right)`.
* ``f5`` is the real relative permittivity at infinity frequency, :math:`\epsilon_{\infty}`.
* ``f6`` is the difference between the real permittivity at zero and infinity frequency, :math:`\Delta\epsilon`.
* ``f7`` is the relaxation time, :math:`t_{0}`.
* ``f8`` is the conductivity (Siemens/metre), :math:`\sigma`
* ``f9`` is the relative permeability, :math:`\mu_r`
* ``f10`` is the magnetic loss (Ohms/metre), :math:`\sigma_*`
* ``i1`` is the number of Debye poles, set to -1 will be automatically calculated tends to minimize the relative absolut error.
* ``str1`` is an identifier for the material.
* ``i2`` is an optional parameter which controls the seeding of the random number generator used in stochastic global optimizator. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.


For example ``#HavriliakNegami: 1e4 1e11 0.3 1 3.4 2.7 0.8e-10 4.5e-4 1 0 5 dry_sand`` creates a material called ``dry_sand`` which 

#Jonscher:
##########

Allows you to model dielectric properties by fitting multi-Debye functions to Jonscher function. The syntax of the command is:

.. code-block:: none

    #Jonscher: f1 f2 f3 f4 f5 f6 f7 f8 f9 i1 str1 [i2]

* ``f1`` is the lower frequency bound (in Hz).
* ``f2`` is the upper frequency bound (in Hz).
* ``f3`` is the real relative permittivity at infinity frequency, :math:`\epsilon_{\infty}`.
* ``f4`` is the :math:`a_{p}` parameter.
* ``f5`` is the :math:`\omega_{p}` parameter.
* ``f6`` is the :math:`n_{p}` parameter.
* ``f7`` is the conductivity (Siemens/metre), :math:`\sigma`
* ``f8`` is the relative permeability, :math:`\mu_r`
* ``f9`` is the magnetic loss (Ohms/metre), :math:`\sigma_*`
* ``i1`` is the number of Debye poles, set to -1 will be automatically calculated tends to minimize the relative absolut error.
* ``str1`` is an identifier for the material.
* ``i2`` is an optional parameter which controls the seeding of the random number generator used in stochastic global optimizator. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.

For example ``#Jonscher: 1e6 1e-5 50 1 1e5 0.7 0.1 1 0.1 4 M2`` creates a material called ``M2`` which 

#Crim:
######

Allows you to model dielectric properties by fitting multi-Debye functions to CRIM function. The syntax of the command is:

.. code-block:: none

    #Crim: f1 f2 f3 v1 v2 f4 f5 f6 i1 str1 [i2]

* ``f1`` is the lower frequency bound (in Hz).
* ``f2`` is the upper frequency bound (in Hz).
* ``f3`` is the shape factor, :math:`a`
* ``v1`` is the vector (paramiter given in input file with `[]`) of volumetric fractions [f1, f2 .... ]. The nuber of paramiters depend on number of materials.
* ``v2`` is the vector (paramiter given in input file with `[]`) containing the materials properties [:math:`\epsilon_{1\infty}`, :math:`\Delta\epsilon_{1}`, :math:`t_{0}_{1}`, :math:`\epsilon_{2\infty}`, :math:`\Delta\epsilon_{2}`, :math:`t_{0}_{2}` .... ]. The number of material vector must be divisible by three.
* ``f4`` is the conductivity (Siemens/metre), :math:`\sigma`
* ``f5`` is the relative permeability, :math:`\mu_r`
* ``f6`` is the magnetic loss (Ohms/metre), :math:`\sigma_*`
* ``i1`` is the number of Debye poles, set to -1 will be automatically calculated tends to minimize the relative absolut error.
* ``str1`` is an identifier for the material.
* ``i2`` is an optional parameter which controls the seeding of the random number generator used in stochastic global optimizator. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.

For example ``#Crim: 1e-1 1e-9 0.5 [0.5,0.5] [3,25,1e6,3,0,1e3] 0.1 1 0 2 M3`` creates a material called ``M3`` which ...

#Rawdata:
#########

Allows you to model dielectric properties obtained experimentally by fitting multi-Debye functions to data given from a file. The syntax of the command is:

.. code-block:: none

    #Rawdata: file1 f1 f2 f3 i1 str1 [i2]

* ``file1`` is an path to text file with experimental data points.
* ``f1`` is the conductivity (Siemens/metre), :math:`\sigma`
* ``f2`` is the relative permeability, :math:`\mu_r`
* ``f3`` is the magnetic loss (Ohms/metre), :math:`\sigma_*`
* ``i1`` is the number of Debye poles, set to -1 will be automatically calculated tends to minimize the relative absolut error.
* ``str1`` is an identifier for the material.
* ``i2`` is an optional parameter which controls the seeding of the random number generator used in stochastic global optimizator. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.

For example ``#Rawdata: user_libs/DebyeFit/Test.txt 0.1 1 0.1 3 M4`` creates a material called ``M4`` which ...

