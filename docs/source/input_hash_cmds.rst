.. _input-hash-cmds:

**************
Model Building
**************

Introduction
============

gprMax has a choice of two methods for building a model to simulate:

1. A **text-based (ASCII) input file**, which can be created with any text editor, and uses a series of gprMax commands which begin with the hash character (``#``). This method is recommended for beginners and those not familiar with Python, and is described in this section of the documentation.
2. A **Python API**, which includes all the functionality of method 1 as well as several more advanced features. This method is recommended for those who prefer to use Python or need access to specific API-only advanced features, and is documented in the :ref:`Python API <input-api>` section.

The general syntax of the hash commands is:

.. code-block:: none

    #command_name: parameter1 parameter2 parameter3 ...

A command and associated parameters should occupy a single line of the input file, and only one command per line is allowed. Hence, the first character of a line containing a command **must** be the hash character (``#``). If the line starts with **any other character** it is ignored by the program. Therefore, user comments or descriptions can be included in the input file. If a line starts with a hash character (``#``) the program will expect a valid command. If the name of the command is not correct the program will abandon execution and issue an error message. When a command requires more than one parameter then these should be separated using a white space character.

The order of commands in the input file is not important with the exception of object construction commands.

To describe the commands that can be used in the input file and their parameters the following conventions are used:

* ``f`` means a real number which can be entered using either a ``[.]`` separating the integral from the decimal part, e.g. 1.5, or in scientific notation, e.g. 15e-1 or 0.15e1.
* ``i`` means an integer number.
* ``c`` means a single character, e.g. ``y``.
* ``str`` means a string of characters with **no** white spaces in between, e.g ``sand``.
* ``file`` means a filename.
* ``[ ]`` square brackets are used to indicate optional parameters.

Unless otherwise specified, the SI system of units is used throughout gprMax:

* All parameters associated with simulated space (i.e. size of the model, spatial increments, etc...) should be specified in **metres**.
* All parameters associated with time (i.e. total simulation time, time instants, etc...) should be specified in **seconds**.
* All parameters denoting frequency should be specified in **Hertz**.
* All parameters associated with spatial coordinates in the model should  be specified in **metres**. The origin of the coordinate system **(0,0)** is at the lower left corner of the model.

It is important to note that gprMax converts spatial and temporal parameters given in **metres** and **seconds** to integer values corresponding to **FDTD cell coordinates** and **iteration number** respectively. Therefore, rounding to the nearest integer number of the user defined values is performed.

The fundamental spatial and temporal discretization steps are denoted as :math:`\Delta x` , :math:`\Delta y`, :math:`\Delta z` and :math:`\Delta t` respectively.

The commands have been grouped into six categories:

* **Essential** - required to run any model, such as the domain size and spatial discretization
* **General** - provide further control over the model
* **Material** - used to introduce different materials into the model
* **Object construction** - used to build geometric shapes with different constitutive parameters
* **Source and output** - used to place source and output points in the model
* **PML** - provide advanced customisation and optimisation of the absorbing boundary conditions

Essential commands
==================

Most of the commands are optional but there are some essential commands which are necessary in order to construct any model. For example, none of the media and object commands are necessary to run a model. However, without specifying any objects in the model gprMax will simulate free space (air), which on its own, is not particularly useful for GPR modelling. If you have not specified a command which is essential in order to run a model, for example the size of the model, gprMax will terminate execution and issue an appropriate error message.

The essential commands are:

#domain:
--------

Allows you to specify the size of the model. The syntax of the command is:

.. code-block:: none

    #domain: f1 f2 f3

where ``f1 f2 f3`` are the size of the model in the x, y, and z directions
respectively. For example, to specify a 500 x 500 x 1000 mm model use
``#domain: 0.5 0.5 1.0``.

For an explicitly declared 2D model, use ``inf`` for its invariant axis and
select the polarisation with ``#domain_mode``. gprMax resolves ``inf`` to the
internal Yee-grid thickness required by the chosen mode; it does not create an
infinite allocation. For example, the following specifies a model invariant in
z:

.. code-block:: none

    #domain_mode: TM
    #domain: 0.5 0.5 inf

The legacy convention of giving one spatial cell on one axis remains supported
and is interpreted as TM mode, but ``#domain_mode`` with ``inf`` is recommended
for new models and is required for TE mode.

#domain_mode:
-------------

Selects whether the domain is three-dimensional or uses a two-dimensional TM
or TE field reduction. The syntax is:

.. code-block:: none

    #domain_mode: str1

``str1`` is ``TM``, ``TE``, or ``3D`` (case-insensitive). ``3D`` is the
default when this command is omitted. For TM or TE, exactly one coordinate of
``#domain`` must be ``inf``; that coordinate selects the invariant axis. If
``inf`` is used without ``#domain_mode``, gprMax defaults to TM for backwards
compatibility.

For example, these declarations select TMz and TEz respectively:

.. code-block:: none

    #domain_mode: TM
    #domain: 0.5 0.5 inf

.. code-block:: none

    #domain_mode: TE
    #domain: 0.5 0.5 inf

TM uses one internal cell and TE uses two internal cells on the invariant axis.
The physical coordinates on that axis may be written as ``inf`` in commands
that accept points or bounds; gprMax resolves lower and upper bounds to the
appropriate faces of the reduced domain. For a single source or receiver,
``inf`` on the invariant axis selects the active interior reference layer
(rather than a TE boundary layer whose fields are constrained). On an
in-plane axis, ``-inf`` selects the lower domain face and ``inf`` the upper
face.

.. note::

    * A 2D model must have exactly one invariant axis.
    * Subgrids, symmetry boundaries, transmission-line sources, and magnetic
      edges are not currently supported in 2D mode.
    * A source polarisation must be one of the active components for the
      selected plane and mode. gprMax rejects incompatible electric and
      magnetic sources rather than silently creating a zero source.

#dx_dy_dz:
----------

Allows you to specify the discretization of space in the x , y and z directions respectively (i.e. :math:`\Delta x` , :math:`\Delta y`, :math:`\Delta z`). The syntax of the command is:

.. code-block:: none

    #dx_dy_dz: f1 f2 f3

where ``f1`` is the spatial step in the x direction (:math:`\Delta x`), ``f2`` is the spatial step in the y direction (:math:`\Delta y`) and ``f3`` is the spatial step in the z direction (:math:`\Delta z`). The spatial discretization controls the maximum permissible time step :math:`\Delta t` with which the solution advances in time in order to reach the required simulated time window. The relation between :math:`\Delta t` and :math:`\Delta x` , :math:`\Delta y`, :math:`\Delta z` is:

.. math:: \Delta t \leq \frac{1}{c\sqrt{\frac{1}{(\Delta x)^2}+\frac{1}{(\Delta y)^2}+\frac{1}{(\Delta z)^2}}},

where :math:`c` is the speed of light. In gprMax the equality is used to determine :math:`\Delta t` from :math:`\Delta x` , :math:`\Delta y`, and :math:`\Delta z`. Small values of :math:`\Delta x` , :math:`\Delta y`, and :math:`\Delta z` result in small values for :math:`\Delta t` which means more iterations in order to reach a given simulated time. However, it is important to note that the smaller the values of :math:`\Delta x` , :math:`\Delta y`, :math:`\Delta z` and :math:`\Delta t` are the more accurate your model will be. See the :ref:`guidance` section for tips on choosing a spatial discretisation.

#time_window:
-------------

Allows you to specify the total required simulated time. The syntax of the command is:

.. code-block:: none

    #time_window: f1

or

.. code-block:: none

    #time_window: i1

In the first case the ``f1`` parameter determines the required simulated time in seconds. For example, if you want to simulate a GPR trace of 20 nanoseconds then ``#time_window: 20e-9`` can be used. gprMax will perform the necessary number of iterations in order to reach the required simulated time. Alternatively, if the command is specified with an ``i1`` gprMax will interpret this value as a total number of iterations. Hence the command ``#time_window: 100`` means that 100 iterations will be performed. The number of iterations and the total simulated time window are related by:

.. math:: t_w = \Delta t × N_{it},

where :math:`t_w` is the time window in seconds, :math:`\Delta t` the time step, and :math:`N_{it}` the number of iterations. gprMax converts the specified time window in seconds to a number of iterations internally using the aforementioned equation. The result of the division is rounded to the nearest integer.


General commands
================

#include_file:
--------------

Allows you to include commands from a file. It will insert the commands from the specified file at the location where the ``#include_file`` command is placed. The syntax of the command is:

.. code-block:: none

    #include_file: file1

``file1`` can be the name of the file containing the commands in the same directory as the input file, or ``file`` can be the full path to the file containing the commands (allowing you to specify any location).


#time_step_stability_factor:
----------------------------

Allows you to alter the value of the time step :math:`\Delta t` used by gprMax. gprMax uses the equality in the CFL condition, hence the maximum permissible time step. If a smaller time step is required then the syntax of the command is:

.. code-block:: none

    #time_step_stability_factor: f1

where ``f1`` can take values :math:`0 < \textrm{f1} \leq 1`. Then the actual time step used will be :math:`\textrm{f1} \times \Delta t`, where :math:`\Delta t` is calculated using the equality from the CFL condition.

#title:
-------

Allows you to include a title for your model. This title is saved in the output file(s). The syntax of the command is:

.. code-block:: none

    #title: str1

where ``str1`` can contain white space characters to separate individual words. The title has to be contained in a single line.

#output_dir:
------------

Allows you to control the directory where output file(s) will be stored.  The syntax of the command is:

.. code-block:: none

    #output_dir: str1

where ``str1`` can be either the absolute path to the directory for the output file(s) or a path relative to the directory of the input files. The default value is the same as the directory of the input files.


#omp_threads:
-------------

Allows you to control how many OpenMP threads (usually the number of physical CPU cores available) are used when running the model. The most computationally intensive parts of gprMax, which are the FDTD solver loops, have been parallelised using `OpenMP <http://openmp.org>`_ which supports multi-platform shared memory multiprocessing. The syntax of the command is:

.. code-block:: none

    #omp_threads: i1

where ``i1`` is the number of OpenMP threads to use. If ``#omp_threads`` is not specified gprMax will first look to see if the environment variable ``OMP_NUM_THREADS`` exists, and if not will detect and use all available physical CPU cores on the machine.

For compatibility with input files created for earlier versions of gprMax,
``#num_threads: i1`` is accepted as an exact alias for ``#omp_threads: i1``.
``#omp_threads`` is the preferred name because it distinguishes OpenMP CPU
threads from other forms of parallel execution. Do not specify both commands
in the same input file.


.. _materials:

Material commands
=================

Built-in materials
------------------

gprMax has three built-in materials which can be used by specifying their identifiers:

* ``pec`` is a perfect electric conductor (PEC), represented by infinite electric conductivity.
* ``pmc`` is a perfect magnetic conductor (PMC), represented by infinite magnetic conductivity.
* ``free_space`` is free space, with :math:`\epsilon_r = \mu_r = 1` and :math:`\sigma = \sigma_* = 0`.

The identifiers ``grass`` and ``water`` are reserved for internal use and should not be used unless you intentionally want to change their properties.

#material:
----------

Allows you to introduce a material into the model described by a set of constitutive parameters. The syntax of the command is:

.. code-block:: none

    #material: f1 f2 f3 f4 str1

* ``f1`` is the relative permittivity, :math:`\epsilon_r`
* ``f2`` is the conductivity (Siemens/metre), :math:`\sigma`
* ``f3`` is the relative permeability, :math:`\mu_r`
* ``f4`` is the magnetic loss (Ohms/metre), :math:`\sigma_*`
* ``str1`` is an identifier for the material.

For example ``#material: 3 0.01 1 0 my_sand`` creates a material called ``my_sand`` which has a relative permittivity (frequency independent) of :math:`\epsilon_r = 3`, a conductivity of :math:`\sigma = 0.01` S/m, and is non-magnetic, i.e. :math:`\mu_r = 1` and :math:`\sigma_* = 0`

#material_from_database:
------------------------

Creates a material from an official or local versioned JSON material
database. See :doc:`material_databases` for the schema, lookup policy, and
provenance information. The syntax is:

.. code-block:: none

    #material_from_database: str1 str2 [str3]

* ``str1`` is the database name, without ``.json``.
* ``str2`` is the entry key.
* ``str3`` is an optional local material ID. It defaults to ``str2`` and is
  useful for giving the material a model-specific name, avoiding a name
  collision, or matching the name expected by imported geometry.

For example ``#material_from_database: fundamental vacuum`` creates the
official ``vacuum`` entry and uses ``vacuum`` as its local ID. To refer to the
same material as ``model_vacuum`` within the model, use
``#material_from_database: fundamental vacuum model_vacuum``.

.. note::

    The Debye, Lorentz, and Drude commands below describe **electric** dispersion. They cannot be applied to PEC or PMC materials, including user-defined materials with infinite electric or magnetic conductivity.


#add_dispersion_debye:
----------------------

Allows you to add dispersive properties to an already defined ``#material`` based on a multiple pole Debye formulation (see :ref:`capabilities` section). For example, the susceptibility function for a single-pole Debye material is given by:

.. math::

    \chi_p (t) = \frac{\Delta \epsilon_{rp}}{\tau_p} e^{-t/\tau_p},

where :math:`\Delta \epsilon_{rp} = \epsilon_{rsp} - \epsilon_{r \infty}`, :math:`\epsilon_{rsp}` is the zero-frequency relative permittivity for the pole, :math:`\epsilon_{r \infty}` is the relative permittivity at infinite frequency, and :math:`\tau_p` is the pole relaxation time.

The syntax of the command is:

.. code-block:: none

    #add_dispersion_debye: i1 f1 f2 f3 f4 ... str1

* ``i1`` is the number of Debye poles.
* ``f1`` is the difference between the zero-frequency relative permittivity and the relative permittivity at infinite frequency, i.e. :math:`\Delta \epsilon_{rp1} = \epsilon_{rsp1} - \epsilon_{r \infty}` , for the first Debye pole.
* ``f2`` is the relaxation time (seconds), :math:`\tau_{p1}`, for the first Debye pole.
* ``f3`` is the difference between the zero-frequency relative permittivity and the relative permittivity at infinite frequency, i.e. :math:`\Delta \epsilon_{rp2} = \epsilon_{rsp2} - \epsilon_{r \infty}` , for the second Debye pole.
* ``f4`` is the relaxation time (seconds), :math:`\tau_{p2}`, for the second Debye pole.
* ...
* ``str1`` identifies the material to add the dispersive properties to.

For example to create a model of water with a single Debye pole, :math:`\epsilon_{rsp1} = 80.1`, :math:`\epsilon_{r \infty} = 4.9` and :math:`\tau_{p1} = 9.231\times 10^{-12}` seconds use: ``#material: 4.9 0 1 0 my_water`` and ``#add_dispersion_debye: 1 75.2 9.231e-12 my_water``.

.. note::

    * You can continue to add pairs of values for :math:`\Delta \epsilon_{rp}` and :math:`\tau_p` for as many Debye poles as you have specified with ``i1``.
    * The relative permittivity in the ``#material`` command should be given as the relative permittivity at infinite frequency, i.e. :math:`\epsilon_{r \infty}`.
    * Temporal values associated with pole frequencies and relaxation times should always be greater than the time step :math:`\Delta t` used in the model.


#add_dispersion_lorentz:
------------------------

Allows you to add dispersive properties to an already defined ``#material`` based on a multiple pole Lorentz formulation (see :ref:`capabilities` section). For example, the susceptability function for a single-pole Lorentz material is given by:

.. math::

    \chi_p (t) = \Re \left\{ -j\gamma_p e^{(-\delta_p + j\beta_p)t} \right\},

where

.. math::

    \beta_p = \sqrt{\omega_p^2 - \delta_p^2} \quad \textrm{and} \quad \gamma_p = \frac{\omega_p^2 \Delta \epsilon_{rp}}{\beta_p},

where :math:`\Delta \epsilon_{rp} = \epsilon_{rsp} - \epsilon_{r \infty}`, :math:`\epsilon_{rsp}` is the zero-frequency relative permittivity for the pole, :math:`\epsilon_{r \infty}` is the relative permittivity at infinite frequency, :math:`\omega_p` is the frequency (Hertz) of the pole pair, :math:`\delta_p` is the damping coefficient (Hertz) , and :math:`j=\sqrt{-1}`.

The syntax of the command is:

.. code-block:: none

    #add_dispersion_lorentz: i1 f1 f2 f3 f4 f5 f6 ... str1

* ``i1`` is the number of Lorentz poles.
* ``f1`` is the difference between the zero-frequency relative permittivity and the relative permittivity at infinite frequency, i.e. :math:`\Delta \epsilon_{rp1} = \epsilon_{rsp1} - \epsilon_{r \infty}` , for the first Lorentz pole.
* ``f2`` is the frequency (Hertz), :math:`\omega_{p1}`, for the first Lorentz pole.
* ``f3`` is the damping coefficient (Hertz), :math:`\delta_{p1}`, for the first Lorentz pole.
* ``f4`` is the difference between the zero-frequency relative permittivity and the relative permittivity at infinite frequency, i.e. :math:`\Delta \epsilon_{rp2} = \epsilon_{rsp2} - \epsilon_{r \infty}` , for the second Lorentz pole.
* ``f5`` is the frequency (Hertz), :math:`\omega_{p2}`, for the second Lorentz pole.
* ``f6`` is the damping coefficient (Hertz), :math:`\delta_{p2}`, for the second Lorentz pole.
* ...
* ``str1`` identifies the material to add the dispersive properties to.

.. note::

    * You can continue to add triplets of values for :math:`\Delta \epsilon_{rp}`, :math:`\omega_p` and :math:`\delta_p` for as many Lorentz poles as you have specified with ``i1``.
    * The relative permittivity in the ``#material`` command should be given as the relative permittivity at infinite frequency, i.e. :math:`\epsilon_{r \infty}`.
    * Temporal values associated with pole frequencies and relaxation times should always be greater than the time step :math:`\Delta t` used in the model.


#add_dispersion_drude:
----------------------

Allows you to add dispersive properties to an already defined ``#material`` based on a multiple pole Drude formulation (see :ref:`capabilities` section). For example, the susceptability function for a single-pole Drude material is given by:

.. math::

    \chi_p (t) = \frac{\omega_p^2}{\gamma_p} (1-e^{-\gamma_p t}),

where :math:`\omega_p` is the frequency (Hertz) of the pole, and :math:`\gamma_p` is the inverse of the pole relaxation time (Hertz).

The syntax of the command is:

.. code-block:: none

    #add_dispersion_drude: i1 f1 f2 f3 f4 ... str1

* ``i1`` is the number of Drude poles.
* ``f1`` is the frequency (Hertz), :math:`\omega_{p1}`, for the first Drude pole.
* ``f2`` is the inverse of the relaxation time (Hertz), :math:`\gamma_{p1}`, for the first Drude pole.
* ``f3`` is the frequency (Hertz), :math:`\omega_{p2}`, for the second Drude pole.
* ``f4`` is the inverse of the relaxation time (Hertz), :math:`\gamma_{p2}` for the second Drude pole.
* ...
* ``str1`` identifies the material to add the dispersive properties to.

.. note::

    * You can continue to add pairs of values for :math:`\omega_p` and :math:`\gamma_p` for as many Drude poles as you have specified with ``i1``.
    * Temporal values associated with pole frequencies and relaxation times should always be greater than the time step :math:`\Delta t` used in the model.


#material_range:
----------------

Allows you to create a series of materials with properties specified by ranges of relative permittivity, conductivity, relative permeability, and magnetic loss. The command is designed to be used in conjunction with the ``#fractal_box`` command for spatial distributions of dielectric properties. The syntax of the command is:

.. code-block:: none

    #material_range: f1 f2 f3 f4 f5 f6 f7 f8 str1

* ``f1`` is the lower end of the range of relative permittivity values.
* ``f2`` is the upper end of the range of relative permittivity values.
* ``f3`` is the lower end of the range of conductivity values.
* ``f4`` is the upper end of the range of conductivity values.
* ``f5`` is the lower end of the range of relative permeability values.
* ``f6`` is the upper end of the range of relative permeability values.
* ``f7`` is the lower end of the range of magnetic loss values.
* ``f8`` is the upper end of the range of magnetic loss values.
* ``str1`` is an identifier for the material range.

For example to create a series of 10 materials with relative permittivity ranging between 2 and 6, :math:`\sigma=0`, :math:`\mu_r=1`, and :math:`\sigma_*=0`, distributed using a fractal approach, use: ``#material_range: 2 6 0 0 1 1 0 0 er2_6`` and ``#fractal_box: 0 0 0 0.15 0.15 0.15 1.5 1 1 1 10 er2_6 my_frac_box``.


#material_list:
----------------

Allows you to create a list of pre-defined materials that can be used in conjunction with the ``#fractal_box`` command for spatial distributions of dielectric properties. The syntax of the command is:

.. code-block:: none

    #material_list: str1 str2 ... str3

* ``str1`` and ``str2`` are identifiers for materials. You can have identifiers for as many pre-defined materials as required.
* ``str3`` is an identifier for the material list.

For example to create a fractal distribution of two different sand materials and water use: ``#material: 3 0 1 0 sand1``, ``#material: 4 0.1 1 0 sand2``, ``#material: 4.9 0.001 1 0 my_water``, ``#add_dispersion_debye: 1 75.2 9.231e-12 my_water``, ``#material_list: sand1 sand2 my_water my_list``, ``#fractal_box: 0 0 0 0.15 0.15 0.15 1.5 1 1 1 3 my_list my_frac_box``.


#soil_peplinski:
----------------

Allows you to use a mixing model for soils proposed by Peplinski (http://dx.doi.org/10.1109/36.387598), valid for frequencies in the range 0.3GHz to 1.3GHz. The command is designed to be used in conjunction with the ``#fractal_box`` command for creating soils with realistic dielectric and geometric properties. The syntax of the command is:

.. code-block:: none

    #soil_peplinski: f1 f2 f3 f4 f5 f6 str1

* ``f1`` is the sand fraction of the soil.
* ``f2`` is the clay fraction of the soil.
* ``f3`` is the bulk density of the soil in grams per centimetre cubed.
* ``f4`` is the density of the sand particles in the soil in grams per centimetre cubed.
* ``f5`` and ``f6`` define a range for the volumetric water fraction of the soil.
* ``str1`` is an identifier for the soil.

For example for a soil with sand fraction 0.5, clay fraction 0.5, bulk density :math:`2~g/cm^3`, sand particle density of :math:`2.66~g/cm^3`, and a volumetric water fraction range of 0.001 - 0.25 use: ``#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.25 my_soil``.

.. note::

    Further information on the Peplinski soil model and our implementation can be found in 'Giannakis, I. (2016). Realistic numerical modelling of Ground Penetrating Radar for landmine detection. The University of Edinburgh, United Kingdom. (http://hdl.handle.net/1842/20449)'

.. _object-construction-commands:

Object construction commands
============================

Object construction commands are processed in the order they appear in the input file. Therefore space in the model allocated to a specific material using for example the ``#box`` command can be reallocated to another material using the same or any other object construction command. Space in the model can be regarded as a canvas in which objects are introduced and one can be overlaid on top of the other overwriting its properties in order to produce the desired geometry. The object construction commands can therefore be used to create complex shapes and configurations.

Anisotropy
----------

It is possible to specify objects that have diagonal anisotropy which allows materials such as wood and fibre-reinforced composites, often imaged with GPR, to be more accurately modelled.

.. math::

    \bar{\bar{\epsilon}} = \left[ \begin{array}{ccc}
    \epsilon_{xx} & 0 & 0 \\
    0 & \epsilon_{yy} & 0 \\
    0 & 0 & \epsilon_{zz}
    \end{array} \right],\quad
    \bar{\bar{\sigma}}= \left[ \begin{array}{ccc}
    \sigma_{xx} & 0 & 0 \\
    0 & \sigma_{yy} & 0 \\
    0 & 0 & \sigma_{zz}
    \end{array} \right]

Standard isotropic objects specify one material identifier that defines the same properties in x, y, and z directions. However, every volumetric object building command can also be specified with three material identifiers, which allows properties for the x, y, and z directions to be separately defined. The ``#plate`` command, which defines a surface, can specify up to two material identifiers, and the ``#edge`` command, which defines a line, continues to take one material identifier. For example to create a box with different material properties in each of the x, y, and z directions use:

.. code-block:: none

    #material: 41 10 1 0 matX
    #material: 35 10 1 0 matY
    #material: 33 1 1 0 matZ
    #box: 0 0 0 0.1 0.1 0.1 matX matY matZ

As another example, to create a cylinder of radius 10 mm that has the same properties in the x and y directions but different properties in the z direction use:

.. code-block:: none

    #material: 41 10 1 0 matXY
    #material: 33 1 1 0 matZ
    #cylinder: 0.1 0.1 0.1 0.5 0.1 0.1 0.01 matXY matXY matZ


Dielectric smoothing
--------------------

At the boundaries between different materials in the model there is the question of which material properties to use. Should the last object to be defined at that location dictate the properties? Should an average set of properties of the materials of the objects that share that location be used? This latter option is often referred to as dielectric smoothing and has been shown to result in more accurate simulations [LUE1994]_ [BOU1996]_. To address this question gprMax includes an option to turn dielectric smoothing on or off for volumetric object building commands. The default behaviour (if no option is specified) is for dielectric smoothing to be on. The option can be specified with a single character ``y`` (on) or ``n`` (off) given after the material identifier in each object command. For example to specify a sphere of material ``sand`` with dielectric smoothing turned off use: ``#sphere: 0.5 0.5 0.5 0.1 sand n``.

.. note::

    * Dispersive materials are not smoothed by default. Use
      ``#dispersive_averaging: y`` to include Debye, Lorentz, and Drude media
      in electric-edge smoothing.
    * If an object is anistropic then dielectric smoothing is automatically turned off for that object.
    * Non-volumetric object building commands, ``#edge``, ``#plate``, and ``#triangle`` (applies to triangular patch not triangular prism) cannot have dielectric smoothing.


#magnetic_averaging:
---------------------

Selects the mixing rule used for magnetic-field components at smoothed material interfaces. Each H component is constructed from the two cells stacked along its own axis. Because the normal component of magnetic flux density is continuous across an interface, the harmonic mean of relative permeability (:math:`\mu_r`) and magnetic loss (:math:`\sigma_*`) is used by default. Electric-field smoothing is unchanged and continues to use its arithmetic four-cell average. The syntax is:

.. code-block:: none

    #magnetic_averaging: str1

* ``str1`` is ``harmonic`` or ``arithmetic`` (case-insensitive).

.. note::

    * This command is optional; the default is ``harmonic``.
    * Earlier versions of gprMax used an arithmetic magnetic average. Add ``#magnetic_averaging: arithmetic`` when exact reproduction of those results is required.
    * The command chooses the magnetic mixing rule only; it does not enable or disable dielectric smoothing.


#dispersive_averaging:
----------------------

Enables or disables interface averaging for dispersive materials. When
enabled, Debye, Lorentz, and Drude media use the same arithmetic four-cell
electric-edge average as nondispersive dielectrics. The Debye case follows the
contour-path formulation developed by [HAR2020]_; its extension to all three
dispersion families uses the inclusive susceptibility representation of
[GIA2014]_. The syntax is:

.. code-block:: none

    #dispersive_averaging: c1

* ``c1`` is ``y`` to enable dispersive averaging or ``n`` to disable it.

For surrounding-cell weights :math:`w_m`, the effective high-frequency
permittivity and conductivity are

.. math::

    \begin{aligned}
    \epsilon_{\infty,\mathrm{eff}}
      &= \sum_m w_m\epsilon_{\infty,m}, \\
    \sigma_{\mathrm{eff}}
      &= \sum_m w_m\sigma_m.
    \end{aligned}

Writing a constituent susceptibility in inclusive pole-residue form as

.. math::

    \chi_m(\omega) = \sum_p
    \frac{W_{m,p}}{j\omega-Q_{m,p}},

where a complex Lorentz term implicitly includes its conjugate contribution,
the effective response is

.. math::

    \epsilon_{r,\mathrm{eff}}(\omega) =
    \sum_m w_m\left[
      \epsilon_{\infty,m}+
      \frac{\sigma_m}{j\omega\epsilon_0}+
      \chi_m(\omega)
    \right].

Consequently, every pole location :math:`Q_{m,p}` is retained and its residue
:math:`W_{m,p}` is multiplied by the corresponding cell weight. For a Debye
term this is equivalent to retaining its relaxation time and scaling
:math:`\Delta\epsilon`:

.. math::

    \chi_{m,p}(\omega) =
    \frac{w_m\Delta\epsilon_{m,p}}
         {1+j\omega\tau_{m,p}}.

The four surrounding cells each contribute :math:`1/4`; repeated materials
therefore naturally produce weights of :math:`1/2` or :math:`3/4`. Terms with
identical pole locations are combined exactly. Two different single-pole
materials generally produce a two-pole effective material, even if each bulk
material uses only one pole.

.. note::

    * This command is optional; the default is ``n``. Enabling it is
      recommended when interface accuracy is more important than the possible
      increase in memory use.
    * Nondispersive dielectric smoothing remains controlled by the optional
      ``y``/``n`` argument of each volumetric geometry command.
    * Dispersive state arrays use the model-wide maximum pole count for
      computational efficiency. An averaged interface containing additional
      distinct pole locations can therefore increase memory use throughout
      the grid. The resolved maximum is included in gprMax's memory estimate.
    * The production solver preserves the exact pole union. It does not apply
      automatic pole reduction because any reduced model is valid only over a
      chosen frequency band and error tolerance.


.. _geometryview:

#geometry_view:
---------------

Allows you output to file(s) information about the geometry of model. The file(s) use the open source `Visualization ToolKit (VTK) <http://www.vtk.org>`_ format which can be viewed in many free readers, such as `Paraview <http://www.paraview.org>`_. The command can be used to create several 3D views of the model which are useful for checking that it has been constructed as desired. The syntax of the command is:

.. code-block:: none

    #geometry_view: f1 f2 f3 f4 f5 f6 f7 f8 f9 file1 c1

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the volume of the geometry view in metres.
* ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the volume of the geometry view in metres.
* ``f7 f8 f9`` are the spatial discretisation of the geometry view in metres. Typically these will be the same as the spatial discretisation of the model but they can be courser if desired.
* ``file1`` is the filename of the file where the geometry view will be stored in the same directory as the input file.
* ``c1`` can be either n (normal) or f (fine) which specifies whether to output the geometry information on a per-cell basis (n) or a per-cell-edge basis (f). The fine mode should be reserved for viewing detailed parts of the geometry that occupy small volumes, as using this mode can generate geometry files with large file sizes.

.. tip::

    When you want to just check the geometry of your model, run gprMax using the optional command line argument ``--geometry-only``. This will build the model and produce any geometry view files, but will not run the simulation.


#edge:
------

Allows you to introduce a wire with specific properties into the model. A wire is an edge of a Yee cell and it can be useful to model resistors or thin wires. The syntax of the command is:

.. code-block:: none

    #edge: f1 f2 f3 f4 f5 f6 str1

* ``f1 f2 f3`` are the starting (x,y,z) coordinates of the edge, and ``f4 f5 f6`` are the ending (x,y,z) coordinates of the edge. The coordinates should define a single line.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials.

For example to specify a x-directed wire that is a perfect electric conductor, use: ``#edge: 0.5 0.5 0.5 0.7 0.5 0.5 pec``. Note that the y and z coordinates are identical.


#thin_wire:
-----------

Allows a conducting wire whose physical radius is smaller than the Yee cell to
be represented by an axis-aligned thin-wire model. The logarithmic radius
correction is based on Umashankar, Taflove, and Beker [UMA1987]_, with the
improved electric/magnetic contour factors proposed by Mäkinen, Juntunen, and
Kivikoski [MAK2002]_. The wire occupies electric edges like a PEC ``#edge``,
while the magnetic updates on the four surrounding Yee edges are corrected to
represent the specified sub-cell radius. The syntax is:

.. code-block:: none

    #thin_wire: f1 f2 f3 f4 f5 f6 f7

* ``f1 f2 f3`` and ``f4 f5 f6`` are the start and end coordinates of one
  non-zero, axis-aligned wire in metres.
* ``f7`` is the physical wire radius :math:`a` in metres. It must be positive
  and smaller than half the minimum transverse cell size.

For a wire along :math:`w`, consider a surrounding :math:`H_v` edge whose
radial direction is :math:`u` (so :math:`u`, :math:`v`, and :math:`w` are the
three Cartesian axes). The Umashankar radius factor is

.. math::

    F_u = \frac{2}{\ln(\Delta u/a)},

and the Mäkinen contour factors are

.. math::

    k_{H_v} = \frac{\Delta u}{\Delta v}
              \tan^{-1}\!\left(\frac{\Delta v}{\Delta u}\right),
    \qquad
    k_{E_u} = \frac{1}{k_{H_v}}.

gprMax stores the projected Yee-edge value
:math:`\widetilde{H}_v=k_{H_v}H_v`. In that representation, Mäkinen's
magnetic update is implemented by multiplying the radial curl coefficient by
:math:`F_u k_{H_v}`. The coefficient for the derivative along the wire remains
the background value because :math:`k_{H_v}k_{E_u}=1`. The ordinary electric
update then consumes :math:`\widetilde{H}_v` directly, thereby applying the
required :math:`k_H` correction to the radial electric field without a special
runtime kernel. On a square transverse mesh,
:math:`k_H=\pi/4` and :math:`k_E=4/\pi`.

The magnetic permeability and magnetic conductivity at each affected H
component are inherited from its already-resolved background material,
including magnetic material averaging. The magnetic-source coefficient is
also multiplied by :math:`k_H`; therefore a co-located
``#magnetic_frill_source`` must not apply :math:`k_H` a second time. It does,
however, apply the feed-cell factor :math:`F_u`, as required by Hyun's
discrete magnetic-current equation.

.. important::

    Receiver samples on the four H edges immediately surrounding the wire are
    the stored projected values :math:`\widetilde{H}`. Divide such a sample by
    its orientation-specific :math:`k_H` if the unprojected point value is
    required. Current loops and ordinary electric updates should use the
    stored values directly.

For example, a z-directed wire of radius 0.1 mm is specified by:
``#thin_wire: 0.05 0.05 0.02 0.05 0.05 0.12 0.0001``.

The wire may lie on a transverse domain face only when that face is a PMC
symmetry boundary. A wire and its surrounding magnetic stencil must not touch
a PML region. MPI domain decomposition is supported: each electric edge and
each member of the surrounding magnetic stencil is constructed on its owning
rank, including a stencil that crosses an internal rank boundary. Thin wires
in 2D models and overlapping sub-cell wire junctions are not currently
supported. The charge-based end-cap treatment
from [MAK2002]_ is not implemented, so an isolated open end retains the usual
staircasing/electrically-long end error; the improved straight-section update
is used up to the final wire edge. No special runtime solver is used: the
corrected material coefficients are consumed by the normal CPU, CUDA, OpenCL,
and Metal field-update kernels.


#magnetic_edge:
----------------

Allows you to introduce a single magnetic-field edge with specific properties into the model. It is the magnetic dual of ``#edge``. The syntax is:

.. code-block:: none

    #magnetic_edge: f1 f2 f3 f4 f5 f6 str1

* ``f1 f2 f3`` are the starting (x,y,z) coordinates of the edge, and ``f4 f5 f6`` are its ending coordinates. The coordinates must define a single axis-aligned line.
* ``str1`` is a material identifier that has already been defined, or one of the built-in materials.

For example, an x-directed perfect magnetic conductor is specified with ``#magnetic_edge: 0.5 0.5 0.5 0.7 0.5 0.5 pmc``.

.. note::

    ``#magnetic_edge`` is not currently supported in 2D mode.

#plate:
-------

Allows you to introduce a plate with specific properties into the model. A plate is a surface of a Yee cell and it can be useful to model objects thinner than a Yee cell. The syntax of the command is:

.. code-block:: none

    #plate: f1 f2 f3 f4 f5 f6 str1

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the plate, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the plate. The coordinates should define a surface and not a 3D object like the ``#box`` command.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials.

For example to specify a xy oriented plate that is a perfect electric conductor, use: ``#plate: 0.5 0.5 0.5 0.7 0.8 0.5 pec``. Note that the z coordinates are identical.

#triangle:
----------

Allows you to introduce a triangular patch or a triangular prism with specific properties into the model. The patch is just a triangular surface made as a collection of staircased Yee cells, and the triangular prism extends the triangular patch in the direction perpendicular to the plane. The syntax of the command is:

.. code-block:: none

    #triangle: f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 str1 [c1]

* ``f1 f2 f3`` are the coordinates (x,y,z) of the first apex of the triangle, ``f4 f5 f6`` the coordinates (x,y,z) of the second apex, and ``f7 f8 f9`` the coordinates (x,y,z) of the third apex.
* ``f10`` is the thickness of the triangular prism. If the thickness is zero then a triangular patch is created.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing. For use only when creating a triangular prism, not a triangular patch.

For example, to specify a xy orientated triangular patch that is a perfect electric conductor, use: ``#triangle: 0.5 0.5 0.5 0.6 0.4 0.5 0.7 0.9 0.5 0.0 pec``. Note that the z coordinates are identical and the thickness is zero.

#box:
-----

Allows you to introduce an orthogonal parallelepiped with specific properties into the model. The syntax of the command is:

.. code-block:: none

    #box: f1 f2 f3 f4 f5 f6 str1 [c1]

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the parallelepiped, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the parallelepiped.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

#sphere:
--------

Allows you to introduce a spherical object with specific parameters into the model. The syntax of the command is:

.. code-block:: none

    #sphere: f1 f2 f3 f4 str1 [c1]

* ``f1 f2 f3`` are the coordinates (x,y,z) of the centre of the sphere.
* ``f4`` is its radius.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

For example, to specify a sphere with centre at (0.5, 0.5, 0.5), radius 100 mm, and with constitutive parameters of ``my_sand``, use: ``#sphere: 0.5 0.5 0.5 0.1 my_sand``.

.. note::

    * Sphere objects are permitted to extend outwith the model domain if desired, however, only parts of object inside the domain will be created.

#cylinder:
----------

Allows you to introduce a circular cylinder into the model. The orientation of the cylinder axis can be arbitrary, i.e. it does not have align with one of the Cartesian axes of the model. The syntax of the command is:

.. code-block:: none

    #cylinder: f1 f2 f3 f4 f5 f6 f7 str1 [c1]

* ``f1 f2 f3`` are the coordinates (x,y,z) of the centre of one face of the cylinder, and ``f4 f5 f6`` are the coordinates (x,y,z) of the centre of the other face.
* ``f7`` is the radius of the cylinder.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

For example, to specify a cylinder with its axis in the y direction, a length of 0.7 m, a radius of 100 mm, and that is a perfect electric conductor, use: ``#cylinder: 0.5 0.1 0.5 0.5 0.8 0.5 0.1 pec``.

.. note::

    * Cylinder objects are permitted to extend outwith the model domain if desired, however, only parts of object inside the domain will be created.


#cylindrical_sector:
--------------------

Allows you to introduce a cylindrical sector (shaped like a slice of pie) into the model. The syntax of the command is:

.. code-block:: none

    #cylindrical_sector: c1 f1 f2 f3 f4 f5 f6 f7 str1 [c1]

* ``c1`` is the direction of the axis of the cylinder from which the sector is defined and can be ``x``, ``y``, or ``z``.
* ``f1 f2`` are the coordinates of the centre of the cylindrical sector.
* ``f3 f4`` are the lower and higher coordinates of the axis of the cylinder from which the sector is defined (in effect they specify the thickness of the sector).
* ``f5`` is the radius of the cylindrical sector.
* ``f6`` is the starting angle (in degrees) for the cylindrical sector (with zero degrees defined on the positive first axis of the plane of the cylindrical sector).
* ``f7`` is the angle (in degrees) swept by the cylindrical sector (the finishing angle of the sector is always anti-clockwise from the starting angle).
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

For example, to specify a cylindrical sector with its axis in the z direction, radius of 0.25 m, thickness of 2 mm, a starting angle of 330 :math:`^\circ`, a sector angle of 60 :math:`^\circ`, and that is a perfect electric conductor, use: ``#cylindrical_sector: z 0.34 0.24 0.500 0.502 0.25 330 60 pec``.

.. note::

    * Cylindrical sector objects are permitted to extend outwith the model domain if desired, however, only parts of object inside the domain will be created.

#cone:
------

Allows you to introduce a cone into the model. The orientation of the cylinder axis can be arbitrary, i.e. it does not have align with one of the Cartesian axes of the model. The syntax of the command is:

.. code-block:: none

    #cone: f1 f2 f3 f4 f5 f6 f7 f8 str1 [c1]

* ``f1 f2 f3`` are the coordinates (x,y,z) of the centre of the first face of the cone, and ``f4 f5 f6`` are the coordinates (x,y,z) of the centre of the other face.
* ``f7`` is the radius of the first face of the cone, and ``f8`` is the radius of the other face of the cone.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

For example, to specify a cone with centres at (0.02, 0.075, 0.075) and (0.08, 0.075, 0.075), radii of 30 mm and 0 mm, and with constitutive parameters of ``my_sand``, use: ``cone: 0.02 0.075 0.075 0.08 0.075 0.075 0.03 0 my_sand``.

.. note::

    * Cone objects are permitted to extend outwith the model domain if desired, however, only parts of object inside the domain will be created.

#ellipsoid:
-----------

Allows you to introduce an ellipsoid into the model. The syntax of the command is:

.. code-block:: none

    #ellipsoid: f1 f2 f3 f4 f5 f6 str1 [c1]

* ``f1 f2 f3`` are the coordinates (x,y,z) of the centre of the ellipsoid.
* ``f4 f5 f6`` are the coordinates (x,y,z) of the semi-axes of the ellipsoid.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

For example, to specify an ellipsoid with centre at (0.045, 0.045, 0.045), and semi-axes (0.03, 0.02, 0.03), and with constitutive parameters of ``my_sand``, use: ``#ellipsoid: 0.045 0.045 0.045 0.03 0.02 0.03 my_sand``.

.. note::

    * Ellipsoidal objects are permitted to extend outwith the model domain if desired, however, only parts of object inside the domain will be created.

.. _fractals:

#fractal_box:
-------------

Allows you to introduce an orthogonal parallelepiped with fractal distributed properties which are related to a mixing model or normal material into the model. The syntax of the command is:

.. code-block:: none

    #fractal_box: f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 i1 str1 str2 [i2] [c1]

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the parallelepiped, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the parallelepiped.
* ``f7`` is the fractal dimension which, for an orthogonal parallelepiped, should take values between zero and three.
* ``f8`` is used to weight the fractal in the x direction.
* ``f9`` is used to weight the fractal in the y direction.
* ``f10`` is used to weight the fractal in the z direction.
* ``i1`` is the number of materials to use for the fractal distribution (defined according to the associated mixing model). This should be set to one if using a normal material instead of a mixing model.
* ``str1`` is an identifier for the associated mixing model or material.
* ``str2`` is an identifier for the fractal box itself.
* ``i2`` is an optional parameter which controls the seeding of the random number generator used to create the fractals. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing. If ``c1`` is specified then a value for ``i2`` must also be present.

For example, to create an orthogonal parallelepiped with fractal distributed properties using a Peplinski mixing model for soil, with 50 different materials over a range of water volumetric fractions from 0.001 - 0.25, you should first define the mixing model using: ``#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.25 my_soil`` and then specify the fractal box using ``#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 50 my_soil my_fractal_box``.

.. note::

    * Currently (2024) we are not aware of a formulation of Perfectly Matched Layer (PML) absorbing boundary that can specifically handle distributions of material properties (such as those created by fractals) throughout the thickness of the PML, i.e. this is a required area of research. Our PML formulations can work to an extent depending on your modelling scenario and requirements. You may need to increase the thickness of the PML and/or consider tuning the parameters of the PML (:ref:`pml-tuning`) to improve performance for your specific model.

#add_surface_roughness:
-----------------------

Allows you to add rough surfaces to a ``#fractal_box`` in the model. A fractal distribution is used for the profile of the rough surface. The syntax of the command is:

.. code-block:: none

    #add_surface_roughness: f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 str1 [i1]

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of a surface on a ``#fractal_box``, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of a surface on a ``#fractal_box``. The coordinates must locate one of the six surfaces of a ``#fractal_box`` but do not have to extend over the entire surface.
* ``f7`` is the fractal dimension which, for an orthogonal parallelepiped, should take values between zero and three.
* ``f8`` is used to weight the fractal in the first direction of the surface.
* ``f9`` is used to weight the fractal in the second direction of the surface.
* ``f10 f11`` define lower and upper limits for a range over which the roughness can vary. These limits should be specified relative to the dimensions of the ``#fractal_box`` that the rough surface is being applied.
* ``str1`` is an identifier for the ``#fractal_box`` that the rough surface should be applied to.
* ``i1`` is an optional parameter which controls the seeding of the random number generator used to create the fractals. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.

Up to six ``#add_rough_surface commands`` can be given for any ``#fractal_box`` corresponding to the six surfaces.

For example, if a ``#fractal_box`` has been specified using: ``#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 50 my_soil my_fractal_box`` then to apply a rough surface that varys between 85 mm and 110 mm (i.e. valleys that are up to 15 mm deep and peaks that are up to 10 mm tall) to the surface that is in the positive z direction, use ``#add_surface_roughness: 0 0 0.1 0.1 0.1 0.1 1.5 1 1 0.085 0.110 my_fractal_box``.

#add_surface_water:
-------------------

Allows you to add surface water to a ``#fractal_box`` in the model that has had a rough surface applied. The syntax of the command is:

.. code-block:: none

    #add_surface_water: f1 f2 f3 f4 f5 f6 f7 str1

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of a surface on a ``#fractal_box``, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of a surface on a ``#fractal_box``. The coordinates must locate one of the six surfaces of a ``#fractal_box`` but do not have to extend over the entire surface.
* ``f7`` defines the depth of the water, which should be specified relative to the dimensions of the ``#fractal_box`` that the surface water is being applied.
* ``str1`` is an identifier for the ``#fractal_box`` that the surface water should be applied to.

For example, to add surface water that is 5 mm deep to an existing ``#fractal_box`` that has been specified using ``#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 50 my_soil my_fractal_box`` and has had a rough surface applied using ``#add_surface_roughness: 0 0 0.1 0.1 0.1 0.1 1.5 1 1 0.085 0.110 my_fractal_box``, use ``#add_surface_water: 0 0 0.1 0.1 0.1 0.1 0.105 my_fractal_box``.

.. note::

    * The water is modelled using a single-pole Debye formulation with properties :math:`\epsilon_{rs} = 80.1`, :math:`\epsilon_{\infty} = 4.9`, and a relaxation time of :math:`\tau = 9.231 \times 10^{-12}` seconds (http://dx.doi.org/10.1109/TGRS.2006.873208). If you prefer, gprMax will use your own definition for water as long as it is named ``water``.

#add_grass:
-----------

Allows you to add grass with roots to a ``#fractal_box`` in the model. The blades of grass are randomly distributed over the specified surface area and a fractal distribution is used to vary the height of the blades of grass and depth of the grass roots. The syntax of the command is:

.. code-block:: none

    #add_grass: f1 f2 f3 f4 f5 f6 f7 f8 f9 i1 str1 [i2]

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of a surface on a ``#fractal_box``, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of a surface on a ``#fractal_box``. The coordinates must locate one of three surfaces (in the positive axis direction) of a ``#fractal_box`` but do not have to extend over the entire surface.
* ``f7`` is the fractal dimension which, for an orthogonal parallelepiped, should take values between zero and three.
* ``f8 f9`` define lower and upper limits for a range over which the height of the blades of grass can vary. These limits should be specified relative to the dimensions of the ``#fractal_box`` that the grass is being applied.
* ``i1`` is the number of blades of grass that should be applied to the surface area.
* ``str1`` is an identifier for the ``#fractal_box`` that the grass should be applied to.
* ``i2`` is an optional parameter which controls the seeding of the random number generator used to create the fractals. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.

For example, to apply 100 blades of grass that vary in height between 100 and 150 mm to the entire surface in the positive z direction of a ``#fractal_box`` that had been specified using ``#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 50 my_soil my_fractal_box``, use ``#add_grass: 0 0 0.1 0.1 0.1 0.1 1.5 0.2 0.25 100 my_fractal_box``.

.. note::

    * The grass is modelled using a single-pole Debye formulation with properties :math:`\epsilon_{rs} = 18.5087`, :math:`\epsilon_{\infty} = 12.7174`, and a relaxation time of :math:`\tau = 1.0793 \times 10^{-11}` seconds (http://dx.doi.org/10.1007/BF00902994). If you prefer, gprMax will use your own definition for grass if you use a material named ``grass``. The geometry of the blades of grass are defined by the parametric equations: :math:`x = x_c +s_x {\left( \frac{t}{b_x} \right)}^2`, :math:`y = y_c +s_y {\left( \frac{t}{b_y} \right)}^2`, and :math:`z=t`, where :math:`s_x` and :math:`s_y` can be -1 or 1 which are randomly chosen, and where the constants :math:`b_x` and :math:`b_y` are random numbers based on a Gaussian distribution.

#geometry_objects_read:
-----------------------

Allows you to insert pre-defined geometry into a model. New geometry files
store compact integer arrays in HDF5, stable entry names in
``/material_keys``, and constitutive properties in a companion JSON material
database. The syntax of the command is:

.. code-block:: none

    #geometry_objects_read: f1 f2 f3 file1 file2

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates in the domain where the lower left corner of the geometry array should be placed.
* ``file1`` is the path to and filename of the HDF5 file that contains an integer array which defines the geometry.
* ``file2`` is the material database name without ``.json``. It is resolved beside ``file1``.

.. note::

    * The integer numbers in the HDF5 file must be stored as a NumPy array at the root named ``data`` with type ``np.int16``.
    * ``/material_keys`` maps each non-negative integer to an entry key in the JSON database.
    * You can use an integer of -1 in the HDF5 file to indicate not to build any material at that location, i.e. whatever material is already in the model at that location.
    * The spatial resolution of the geometry objects must match the spatial resolution defined in the model.
    * The spatial resolution must be specified as a root attribute of the HDF5 file with the name ``dx_dy_dz`` equal to a tuple of floats, e.g. (0.002, 0.002, 0.002)
    * Legacy material command files remain supported. Supply the ``.txt`` filename as ``file2``; files ending in ``.txt`` select the legacy reader.

For example, after converting a downloaded 2 mm AustinMan model to the
current HDF5/JSON format as described in the :doc:`AustinMan/AustinWoman
toolbox <inc_AustinMan>`, insert it with its lower-left corner 40 mm from the
domain origin using ``#geometry_objects_read: 0.04 0.04 0.04
AustinMan_v2.6_2x2x2_gprmax.h5 AustinMan_v2_6_2mm_materials``.

#geometry_objects_write:
------------------------

Allows you to write geometry generated in a model to file. The file can be read back into gprMax using the ``#geometry_objects_read`` command. This allows complex geometry that can take some time to generate to be saved to file and more quickly imported into subsequent models. Geometry arrays and stable material keys are saved in HDF5 and corresponding material definitions are saved in a versioned ``_materials.json`` database. The syntax of the command is:

.. code-block:: none

    #geometry_objects_write: f1 f2 f3 f4 f5 f6 file1

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the parallelepiped, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the parallelepiped.
* ``file1`` is the basename for the files where geometry and material information will be stored.

.. note::

    * The structure of the HDF5 file is the same as that described for the ``#geometry_objects_read`` command.
    * Objects are stored using spatial resolution defined in the model.


Source and output commands
==========================

#waveform:
----------

Allows you to specify common waveform shapes to use with sources in the model. The syntax of the command is:

.. code-block:: none

    #waveform: str1 f1 f2 str2

* ``str1`` is the type of waveform which can be:

    * ``gaussian`` which is a Gaussian waveform.
    * ``gaussiandot`` which is the first derivative of a Gaussian waveform.
    * ``gaussiandotnorm`` which is the normalised first derivative of a Gaussian waveform.
    * ``gaussiandotdot`` which is the second derivative of a Gaussian waveform.
    * ``gaussiandotdotnorm`` which is the normalised second derivative of a Gaussian waveform.
    * ``ricker`` which is a Ricker (or Mexican hat) waveform, i.e. the negative, normalised second derivative of a Gaussian waveform.
    * ``gaussianprime`` which is the first derivative of a Gaussian waveform, directly derived from the aforementioned ``gaussian`` (see notes below).
    * ``gaussiandoubleprime`` which is the second derivative of a Gaussian waveform, directly derived from the aforementioned ``gaussian`` (see notes below).
    * ``sine`` which is a single cycle of a sine waveform.
    * ``contsine`` which is a continuous sine waveform. In order to avoid introducing noise into the calculation the amplitude of the waveform is modulated for the first cycle of the sine wave (ramp excitation).
* ``f1`` is the scaling of the maximum amplitude of the waveform (for a ``#hertzian_dipole`` the units will be Amps, for a ``#voltage_source`` or ``#transmission_line`` the units will be Volts).
* ``f2`` is the centre frequency of the waveform (Hertz). In the case of the Gaussian waveform it is related to the pulse width.
* ``str2`` is an identifier for the waveform used to assign it to a source.

For example, to specify the normalised first derivative of a Gaussian waveform with an amplitude of one and a centre frequency of 1.2GHz, use: ``#waveform: gaussiandotnorm 1 1.2e9 my_gauss_pulse``.

.. note::

    * The functions used to create the waveforms can be found in the ``toolboxes/Plotting`` package.
    * ``gaussiandot``, ``gaussiandotnorm``, ``gaussiandotdot``, ``gaussiandotdotnorm``, ``ricker`` waveforms have their centre frequencies specified by the user, i.e. they are not derived to the 'base' ``gaussian``
    * ``gaussianprime`` and ``gaussiandoubleprime`` waveforms are the first derivative and second derivative of the 'base' ``gaussian`` waveform, i.e. the centre frequencies of the waveforms will rise for the first and second derivatives.


#excitation_file:
-----------------

Allows you to specify an ASCII file that contains amplitude values that specify custom waveform(s) that can be used with sources in the model.

The first row of each column must begin with a identifier string that will be used as the name of each waveform. Subsequent rows should contain amplitude values for the custom waveform you want to use. You can import multiple different waveforms (as columns of amplitude data) in a single file.

Ideally, there should be the same number of amplitude values as number of iterations in your model. If there are less amplitude values than the number of iterations in the model, the end of the sequence of amplitude values will be padded with zero values up to the number of iterations. If extra amplitude values are specified than needed then they are ignored.

Optionally, in the first column of the file you may specify your own time vector of values (which must use the identifier ``time``) to use with the amplitude values of the waveform.

The amplitude values will be interpolated using either the aforementioned user specified time vector, or if none was supplied, a vector of time values corresponding to the simulation time step and number of iterations will be used. Key parameters used for the interpolation can be specified in the command.

 The syntax of the command is:

.. code-block:: none

    #excitation_file: file1 [str1 str2]

* ``file1`` can be the name of the file containing the specified waveform in the same directory as the input file, or ``file`` can be the full path to the file containing the specified waveform (allowing you to specify any location).
* ``str1`` and ``str2`` are an optional parameter pair that allow values for ``kind`` and ``fill_value`` to be passed to the interpolation function (`scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_). If they are not given the default values for the function will be used.

For example, to specify the file ``my_waves.txt``, which contains two custom waveform shapes, use: ``#excitation_file: my_waves.txt``. The contents of the file ``my_waves.txt`` would take the form:

.. code-block:: none

    time my_pulse1 my_pulse2
    0 0 0
    1.926e-12 1.2e-6 0
    3.852e-12 1.3e-6 1.0e-1
    5.778e-12 5.0e-6 1.5e-1
    ...       ...    ...
    ...       ...    ...
    ...       ...    ...

Then to use ``my_pulse1`` custom waveform shape with, for example, a z-polarised Hertzian dipole source:

.. code-block:: none

    #hertzian_dipole: z 0.5 0.5 0.5 my_pulse1

.. note::

    * The ``#waveform`` command is not necessary when using a custom waveform excitation, only the ``#excitation_file`` command and whatever source is going to be used with the custom waveform excitation.

#hertzian_dipole:
-----------------

Allows you to specify a current density term at an electric field location - the simplest excitation, often referred to as an additive or soft source.

.. math::

    J_s = \frac{I \Delta l}{\Delta x \Delta y \Delta z},

where :math:`J_s` is the current density, :math:`I` is the current, :math:`\Delta l` is the length of the infinitesimal electric dipole, and :math:`\Delta x`, :math:`\Delta y`, and :math:`\Delta z` are the spatial resolution of the grid.

.. note::

    * :math:`\Delta l` is set equal to :math:`\Delta x`, :math:`\Delta y`, or :math:`\Delta z` depending on the specified polarisation.

The syntax of the command is:

.. code-block:: none

    #hertzian_dipole: c1 f1 f2 f3 str1 [f4 f5]

* ``c1`` is the polarisation of the source and can be ``x``, ``y``, or ``z``.
* ``f1 f2 f3`` are the coordinates (x,y,z) of the source in the model.
* ``f4 f5`` are optional parameters. ``f4`` is a time delay in starting the source. ``f5`` is a time to remove the source. If the time window is longer than the source removal time then the source will stop after the source removal time. If the source removal time is longer than the time window then the source will be active for the entire time window. If ``f4 f5`` are omitted the source will start at the beginning of time window and stop at the end of the time window.
* ``str1`` is the identifier of the waveform that should be used with the source.

For example, to use a x-polarised Hertzian dipole with unit amplitude and a 600 MHz centre frequency Ricker waveform, use: ``#waveform: ricker 1 600e6 my_ricker_pulse`` and ``#hertzian_dipole: x 0.05 0.05 0.05 my_ricker_pulse``.

.. note::

    * When a ``#hertzian_dipole`` is used in a 2D simulation it acts as a line source of current in the invariant (geometry) direction of the simulation.


#magnetic_dipole:
-----------------

This will simulate an infinitesimal magnetic dipole. This is often referred to as an additive or soft source. The syntax of the command is:

.. code-block:: none

    #magnetic_dipole: c1 f1 f2 f3 str1 [f4 f5]

* ``c1`` is the polarisation of the source and can be ``x``, ``y``, or ``z``.
* ``f1 f2 f3`` are the coordinates (x,y,z) of the source in the model.
* ``f4 f5`` are optional parameters. ``f4`` is a time delay in starting the source. ``f5`` is a time to remove the source. If the time window is longer than the source removal time then the source will stop after the source removal time. If the source removal time is longer than the time window then the source will be active for the entire time window. If ``f4 f5`` are omitted the source will start at the beginning of time window and stop at the end of the time window.
* ``str1`` is the identifier of the waveform that should be used with the source.

#rational_network:, #network_terminal:, #network_excitation:
------------------------------------------------------------

These commands connect a linear one-port network to one electric Yee edge.
The reusable network model is expressed as a rational driving-point
admittance

.. math::

    Y(s) = G + sC + \sum_{m=1}^{M}\frac{r_m}{s-p_m},

where :math:`G` is the direct conductance, :math:`C` is the direct
capacitance, and :math:`p_m` and :math:`r_m` are pole-residue pairs. The
syntax is

.. code-block:: none

    #rational_network: str1 f1 f2 i1 [f3 f4 f5 f6 ...] [c1]
    #network_terminal: c2 f7 f8 f9 str1 str2
    #network_excitation: str2 str3 [f10 f11]

* ``str1`` is the reusable network-model ID.
* ``f1`` and ``f2`` are :math:`G` in siemens and :math:`C` in farads.
* ``i1`` is the number of poles. Every pole then has four values:
  ``pole_real pole_imag residue_real residue_imag``. Poles are in rad/s and
  residues are in S/s. Non-real poles and residues must occur in conjugate
  pairs so that the time-domain current is real.
* ``c1`` is an optional ``y``/``n`` flag permitting an active model. It is
  ``n`` by default. Passive models are checked over the FDTD band and all
  poles must be stable.
* ``c2`` is the terminal polarisation ``x``, ``y``, or ``z``; ``f7 f8 f9``
  are its coordinates; and ``str2`` is its unique terminal ID.
* ``str3`` is an existing waveform ID. The optional ``f10 f11`` pair gives
  its start and stop times. Omitting ``#network_excitation`` creates a
  passive load or receiving terminal.

For common elements, a resistor uses :math:`G=1/R`, a capacitor uses the
direct :math:`C` term, and an inductor uses :math:`p=0`, :math:`r=1/L`. A
series :math:`RL` branch has :math:`p=-R/L` and :math:`r=1/L`. For example,
a driven 50 Ohm terminal is

.. code-block:: none

    #waveform: ricker 1 2e9 pulse
    #rational_network: source50 0.02 0 0
    #network_terminal: z 0.05 0.05 0.02 source50 feed
    #network_excitation: feed pulse
    #network_port: feed 50 10

The optional excitation is a Thévenin open-circuit voltage applied through
the complete rational impedance. The arbitrary linear circuit-to-edge
formulation follows [PER1999]_ and [CHE2007]_. gprMax improves the underlying
classic PLRC time placement using the exponential recursive-convolution
approach of Giannakis and Giannopoulos [GIA2014]_. In particular, every pole
state is evaluated analytically at the electric half-step under a linearly
varying terminal voltage; it is not estimated by averaging its two integer-
time values. Each terminal stores only its own poles and applies one locally
implicit edge correction, rather than allocating dispersive state throughout
the mesh.

A driven network with :math:`G=1/R`, no direct capacitance, and no poles is
discretely identical to a finite-resistance ``#voltage_source`` having the
same :math:`R`, waveform, position, and polarisation. A zero-resistance hard
source is not equivalent.

This implementation supports 3-D models on the CPU, CUDA, OpenCL, and Metal
solvers, including domain-decomposed MPI CPU models, and a nondispersive
terminal edge; dispersive materials may exist elsewhere in the model. In an
MPI model the sparse terminal state is advanced only by the rank that owns its
electric edge, then gathered for final port processing. A terminal may be
placed in a CPU subgrid, where it uses the fine spatial and temporal steps. On
an accelerator the complete recurrence and local field correction remain
device-resident during time
stepping. Several independent terminals may be used, but a coupled multiport
admittance matrix is not yet supported. See [CHE2007]_ for the general PLRC
lumped-network formulation and :ref:`Analytical comparisons
<rational-network-validation>` for a complete loaded-guide comparison.


#voltage_source:
----------------

Allows you to introduce a voltage source at an electric field location. It can be a hard source if it's resistance is zero, i.e. the time variation of the specified electric field component is prescribed, or if it's resistance is non-zero it behaves as a resistive voltage source. It is useful for exciting antennas when the physical properties of the antenna are included in the model. The syntax of the command is:

.. code-block:: none

    #voltage_source: c1 f1 f2 f3 f4 str1 [f5 f6 [f7]]

* ``c1`` is the polarisation of the source and can be ``x``, ``y``, or ``z``.
* ``f1 f2 f3`` are the coordinates (x,y,z) of the source in the model.
* ``f4`` is the internal resistance of the voltage source in Ohms. If ``f4`` is set to zero then the voltage source is a hard source. That means it prescribes the value of the electric field component. If the waveform becomes zero then the source is perfectly reflecting.
* ``f5 f6`` are optional parameters. ``f5`` is a time delay in starting the source. ``f6`` is a time to remove the source. If the time window is longer than the source removal time then the source will stop after the source removal time. If the source removal time is longer than the time window then the source will be active for the entire time window. If ``f5 f6`` are omitted the source will start at the beginning of time window and stop at the end of the time window.
* ``f7`` is the optional positive wave-reference impedance in Ohms used by a coincident ``#rx_port``. A hard source defaults to 50 Ohms. For a finite-resistance source, ``f7`` must equal ``f4``. In the positional hash syntax, ``f5`` and ``f6`` must be supplied before ``f7``; the Python API does not have this restriction.
* ``str1`` is the identifier of the waveform that should be used with the source.

For example, to specify a y directed voltage source with an internal resistance of 50 Ohms, an amplitude of five, and a 1.2 GHz centre frequency Gaussian waveform use: ``#waveform: gaussian 5 1.2e9 my_gauss_pulse`` and ``#voltage_source: y 0.05 0.05 0.05 50 my_gauss_pulse``.

.. note::

    * Where a resistive voltage source is placed at a location that is not free space, the conductivity (determined from the resistance) of the voltage source will be added to the underlying conductivity of the existing material at that location. For example, if a resistive voltage source of 50 Ohms is placed at a location where the material has a relative permittivity of 4 and conductivity of 0.1 S/m, the conductivity of that cell edge will become 0.12 S/m.

#transmission_line:
-------------------

Allows you to introduce a one-dimensional transmission line model [MAL1994]_ at an electric field location. The transmission line can have a specified resistance greater than zero and less than the impedance of free space (376.73 Ohms). It is useful for exciting antennas when the physical properties of the antenna are included in the model. Transmission lines are supported by the CPU, CUDA, OpenCL, and Metal solvers. The syntax of the command is:

.. code-block:: none

    #transmission_line: c1 f1 f2 f3 f4 str1 [f5 f6]

* ``c1`` is the polarisation of the transmission line and can be ``x``, ``y``, or ``z``.
* ``f1 f2 f3`` are the coordinates (x,y,z) of the transmission line in the model.
* ``f4`` is the characteristic resistance of the transmission line source in Ohms. It can be any value greater than zero and less than the impedance of free space (376.73 Ohms).
* ``f5 f6`` are optional parameters. ``f5`` is a time delay in starting the excitation of the transmission line. ``f6`` is a time to remove the excitation of the transmission line. If the time window is longer than the excitation of the transmission line removal time then the excitation of the transmission line will stop after the excitation of the transmission line removal time. If the excitation of the transmission line removal time is longer than the time window then the excitation of the transmission line will be active for the entire time window. If ``f5 f6`` are omitted the excitation of the transmission line will start at the beginning of time window and stop at the end of the time window.
* ``str1`` is the identifier of the waveform that should be used with the source.

Time histories of incident and total voltage and current are saved to the
output file. gprMax also calculates S11, input impedance, and input admittance
automatically after the simulation. The line resistance is used as the S11
reference impedance; ``Zin`` is derived from S11, while an independently
de-embedded current result is saved as ``Zin_current`` for verification. The
frequency axis, validity masks, and lambda/10 mesh limit are stored with the
results. No separate ``#rx_port`` command is required for a transmission-line
source. The complete schema and equations are documented in the
:ref:`Simulation Output <output>` section.

For example, to specify a z directed transmission line source with a resistance of 75 Ohms, an amplitude of five, and a 1.2 GHz centre frequency Gaussian waveform use: ``#waveform: gaussian 5 1.2e9 my_gauss_pulse`` and ``#transmission_line: z 0.05 0.05 0.05 75 my_gauss_pulse``.

An example antenna model using a transmission line can be found in the :ref:`examples <example-wire-dipole>` section.

#magnetic_frill_source:
------------------------

Allows you to introduce a magnetic-frill (equivalent-feed) source [HYU2009]_ at an
electric field location, for an antenna driven through a PEC ground plane by a
coaxial line - the antenna's own inner conductor passes continuously through the
plane, unlike ``#voltage_source``/``#transmission_line``'s gap-feed model.
Complements ``#transmission_line``: it is a different, well-established
formulation (not a variant of the two-wire line, and not a general-purpose
alternative to ``#voltage_source``), building on the magnetic frill generator of
King and Harrison and on Maloney, Smith, & Scott's FDTD implementation of it.
There is no explicit one-dimensional line, no absorbing boundary, and no
"magic time step". The coax's sub-cell aperture is represented by an
equivalent magnetic surface current entering Faraday's law at the four Yee
magnetic-field components immediately surrounding the feed point. The
corrected Hyun feed-cell formulation is supported by the CPU, CUDA, OpenCL,
and Metal solvers, and by domain-decomposed MPI CPU models. In MPI, the four
magnetic feed edges may cross internal rank boundaries: their Ampere-loop
terms are combined before the common terminal state is advanced, and each
field deposit is applied by its owning rank. The syntax is:

.. code-block:: none

    #magnetic_frill_source: c1 f1 f2 f3 f4 str1 [f5 f6]

* ``c1`` is the polarisation of the source and can be ``x``, ``y``, or ``z``
  - the antenna axis the source drives current along, following the same
  electrical sign convention already used by gprMax's
  ``Ix``/``Iy``/``Iz`` current output.
* ``f1 f2 f3`` are the coordinates (x,y,z) of the feed point in the model - the
  position of the antenna's own inner conductor where it passes through the
  ground plane.
* ``f4`` is the coax's characteristic impedance ``Zcoax`` (Ohms), and must be
  finite and greater than zero.
* ``str1`` is the identifier of the waveform that should be used with the source.
* ``f5 f6`` are optional parameters specifying a delay before the incident
  waveform starts and a time at which that waveform stops. They gate only the
  incident wave; the coaxial terminal relation remains connected for the rest
  of the simulation so that late antenna reflections are treated correctly.

The source must be co-located with a Yee edge of a ``#thin_wire`` of the same
orientation. gprMax obtains the inner-conductor radius :math:`a`
from that wire. This is necessary because the radius occurs explicitly in
Hyun's discrete feed-cell equation. An ordinary PEC edge has no unambiguous
physical radius and is therefore rejected. In particular, :math:`a` cannot be
assumed to equal the cell size: that would make the logarithmic correction
singular. The normal ``#thin_wire`` condition
:math:`a < \min(\Delta u,\Delta v)/2` applies in the two transverse directions.

The physical outer-conductor/aperture radius :math:`b` and the coax filler are
represented through ``Zcoax`` and are not separate numerical inputs. For a
lossless TEM coax with inner-conductor radius :math:`a`, outer-conductor inner
radius :math:`b`, and filler relative permittivity and permeability
:math:`\varepsilon_{r,c}` and :math:`\mu_{r,c}`, respectively,

.. math::

    Z_\mathrm{coax}
    = \frac{1}{2\pi}
      \sqrt{\frac{\mu_0\mu_{r,c}}{\varepsilon_0\varepsilon_{r,c}}}
      \ln\!\left(\frac{b}{a}\right)
    = \frac{\eta_0}{2\pi}
      \sqrt{\frac{\mu_{r,c}}{\varepsilon_{r,c}}}
      \ln\!\left(\frac{b}{a}\right).

For the usual nonmagnetic filler (:math:`\mu_{r,c}=1`), this is commonly
written

.. math::

    Z_\mathrm{coax} \simeq
    \frac{60}{\sqrt{\varepsilon_{r,c}}}
    \ln\!\left(\frac{b}{a}\right)\ \Omega,
    \qquad
    b = a\exp\!\left(
        \frac{Z_\mathrm{coax}\sqrt{\varepsilon_{r,c}}}{60}
    \right).

Here :math:`\eta_0=\sqrt{\mu_0/\varepsilon_0}` is the impedance of free space.
The permittivity in these equations is that of the **coax filler**, not
necessarily the material above the ground plane surrounding the antenna.
Thus a measured or datasheet value of ``Zcoax`` may be supplied directly;
otherwise it can be calculated from :math:`a`, :math:`b`, and the filler.
gprMax obtains :math:`a` from the attached ``#thin_wire`` but does not infer or
check :math:`b`.

.. warning::

    This formulation is only strictly valid while the coax's aperture radius
    ``b`` is sub-cell: smaller than the discretisation (:math:`\Delta`) in the
    plane perpendicular to the polarisation axis -
    :math:`b<\min(\Delta y,\Delta z)` for ``x`` polarisation,
    :math:`b<\min(\Delta z,\Delta x)` for ``y``, or
    :math:`b<\min(\Delta x,\Delta y)` for ``z``. This is a model-validity
    boundary, not a refinement axis: a physical aperture that is not sub-cell
    does not converge toward the true coax-fed antenna by refining the mesh,
    it converges toward a different problem (a continuous PEC plane carrying a
    fictitious current sheet) - mesh the coax explicitly with
    ``#cylinder``/``#box`` commands instead in that case. Because gprMax is
    never given ``b`` (only ``Zcoax``), **this is not checked automatically -
    it is the user's responsibility to confirm it holds** for their coax and
    mesh before trusting the result.

The tangential electric edges of the ground plane at the feed must already be
PEC (for example via ``#plate`` or ``#box``). The axial edge is supplied by the
attached ``#thin_wire``. The sub-cell aperture is invisible to the grid and is
represented by the frill term; no gap is cut in the PEC plane.

In Hyun's notation, the equivalent magnetic current and coax load relation are

.. math::

    M_\phi^n =
    \frac{-2V_\mathrm{inc}^n + Z_0 I_\mathrm{tot}^n}
         {(\Delta\rho/2)\ln(\Delta\rho/a)},
    \qquad
    V_\mathrm{ab}^n = 2V_\mathrm{inc}^n-Z_0 I_\mathrm{tot}^n.

gprMax generalises the cylindrical feed cell to a rectangular Cartesian Yee
cell. For each transverse radial direction :math:`u`, the frill applies

.. math::

    F_u = \frac{2}{\ln(\Delta u/a)}

to the magnetic-source coefficient. The attached improved thin-wire material
already supplies Mäkinen's orientation-specific :math:`k_H` projection, so the
complete source coefficient contains :math:`F_u k_H`. The frill must supply
:math:`F_u`; omitting it would model the wrong inner-conductor radius.

The leapfrog current is evaluated using the time-average approximation
recommended in [HYU2009]_,

.. math::

    I_\mathrm{tot}^n = \frac{1}{2}
    \left(I^{n-1/2}+I^{n+1/2}\right).

Because :math:`I^{n+1/2}` depends on the frill voltage applied during the same
update, gprMax solves this small implicit relation in closed form. If
:math:`G_f` is the precomputed feed-cell self-admittance and
:math:`I_\mathrm{bulk}^{n+1/2}` is the current after the ordinary magnetic
update but before the new frill deposit, then

.. math::

    I^{n+1/2} =
    \frac{I_\mathrm{bulk}^{n+1/2} + 2G_f V_\mathrm{inc}^n
          - (G_f Z_0/2) I^{n-1/2}}
         {1+G_f Z_0/2}.

This avoids a forward-time predictor iteration and implements equations
(8)--(11) of [HYU2009]_ directly. The command waveform follows gprMax's
Thevenin-generator convention, so the stored incident wave is one half of the
specified waveform amplitude.

Time histories of incident and total voltage (:math:`V_\mathrm{inc}`,
:math:`V_\mathrm{ab}`) and total current (:math:`I_\mathrm{tot}`) are saved to
the output file, along with automatically-calculated S11, input impedance, and
input admittance, following the same conventions as ``#transmission_line``. No
separate ``#rx_port`` command is required. If ``#rx_port`` is placed at the
same position it does not create a second, independent measurement - it can
only override the automatic output's spectrum limit. The complete schema and
equations are documented in the :ref:`Simulation Output <output>` section.

For example, a z-directed, 0.1 mm radius inner conductor driven through a
ground plane at z = 0 by a 50 Ohm coax is:

.. code-block:: none

    #waveform: ricker 1 10e9 my_pulse
    #plate: 0 0 0 0.1 0.1 0 pec
    #thin_wire: 0.05 0.05 0 0.05 0.05 0.04 0.0001
    #magnetic_frill_source: z 0.05 0.05 0 50 my_pulse

The hash command creates a main-grid source and remains valid in a model that
also contains subgrids. To place the frill, its attached thin wire, and its
ground plane inside a ``SubGridHSG``, use the Python API and add all four
objects (including the waveform) to the same subgrid object.

.. note::

    * This source can be placed at a symmetry-plane corner declared with
      ``#symmetry_boundary`` (for example a monopole fed at the domain origin,
      to simulate only a quarter of the structure) - but only at the '0'-type
      faces transverse to ``c1`` (``y0``/``z0`` for ``x`` polarisation,
      ``z0``/``x0`` for ``y``, or ``x0``/``y0`` for ``z``); the corresponding
      'max'-type symmetry corners are not yet supported.
    * A feed point placed exactly at a domain-minimum boundary without the
      matching ``#symmetry_boundary`` declared there is rejected outright,
      since gprMax's underlying current-loop calculation cannot otherwise
      distinguish "domain edge" from "symmetry plane".
    * Two frill sources may not share a surrounding H edge. Such adjacent or
      duplicate feeds form a coupled feed-cell system and cannot be advanced
      as independent scalar terminal relations.
    * MPI symmetry boundaries are supported. A frill and its thin wire may
      cross internal rank boundaries or use PMC image completion at supported
      minimum-face symmetry corners.
    * This source is a "Path A" (through-ground-plane, continuous-conductor)
      feed model. It is not intended for a dipole/bow-tie style gap feed
      (:math:`E_z \neq 0` at the feed) - use ``#voltage_source`` for that case.

#plane_wave_angles:
---------------------

Allows you to introduce a discrete plane wave source [TAN2010]_. Plane wave sources are a useful tool in multiple different scenarios of electromagnetic simulations, especially when the wave is emitted by a source that is quite far away from the target. The plane wave can originate from any direction and it is assumed that it propagates in a homogeneous background medium. The syntax of the command is:

.. code-block:: none

    #plane_wave_angles: f1 f2 f3 f4 f5 f6 f7 f8 f9 str1 [str2 f10 f11]

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the total field, scattered field (TFSF) box, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the total field, scattered field (TFSF) box.
* ``f7`` is theta which defines the polar propagation angle (degrees) of the incident plane wave.
* ``f8`` is phi which defines the azimuthal propagation angle (degrees) of the incident plane wave.
* ``f9`` is psi which defines the polarisation angle (degrees) of the incident plane wave.
* ``str1`` is the identifier of the waveform that should be used with the source.
* ``str2 f10 f11`` are optional parameters. ``str2`` is a material identifier that is the background material that the plane wave propagates through. The default value is ``free_space``. This material must also be the background material of your full model. ``f10`` is a time delay in starting the excitation of the discrete plane wave. ``f11`` is a time to remove the excitation of the discrete plane wave. If the time window is longer than the excitation of the discrete plane wave removal time then the excitation of the discrete plane wave will stop after the excitation of the discrete plane wave removal time. If the excitation of the discrete plane wave removal time is longer than the time window then the excitation of the discrete plane wave will be active for the entire time window. If ``f10 f11`` are omitted the excitation of the discrete plane wave will start at the beginning of time window and stop at the end of the time window.


For example, to specify a discrete plane wave in a TFSF box (0.010, 0.010, 0.010 to 0.040, 0.040, 0.040) with a polarisation angle :math:`\psi` of 90 degrees, azimuthal propagation angle :math:`\phi` of 63.4 degrees, polar propagation angle :math:`\theta` of 36.7 degrees, and using the waveform defined by the identifier ``mypulse`` use: ``#plane_wave_angles: 0.010 0.010 0.010 0.040 0.040 0.040 36.7 63.4 90.0 mypulse``.

.. note::

    * Plane waves support non-dispersive dielectric backgrounds and multi-pole Debye, Lorentz, and Drude media. They do not currently support ``user``-defined waveforms.
    * The plane-wave command must be defined on the main grid. Its TFSF box may contain a complete subgrid, but must strictly enclose the subgrid's HSG outer coupling surface wherever the two regions overlap.
    * MPI domain decomposition is supported. Every rank advances an identical, small auxiliary one-dimensional DPW, while TFSF corrections are restricted to the Yee components owned by that rank. This adds no per-timestep plane-wave communication beyond the normal field-halo exchange.
    * This plane wave implementation was based on an initial implementation made possible by a `Google Summer of Code <https://summerofcode.withgoogle.com/>`_ (GSoC) project and `more details can be found in the original pull request <https://github.com/gprMax/gprMax/pull/373>`_.
    * Internally, theta and phi are approximated by an integer direction vector (Mx, My, Mz) found to within a maximum acceptable angular difference of 3 arc minutes (0.05 degrees) by default. This tolerance can be relaxed or tightened using the ``max_angle_diff`` parameter (in degrees) when using the Python API.

#plane_wave_vector:
---------------------

Allows you to introduce a discrete plane wave source [TAN2010]_. Plane wave sources are a useful tool in multiple different scenarios of electromagnetic simulations, especially when the wave is emitted by a source that is quite far away from the target. The plane wave can originate from any direction and it is assumed that it propagates in a homogeneous background medium. The syntax of the command is:

.. code-block:: none

    #plane_wave_vector: f1 f2 f3 f4 f5 f6 i1 i2 i3 f7 str1 [str2 f10 f11]

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the total field, scattered field (TFSF) box, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the total field, scattered field (TFSF) box.
* ``i1 i2 i3`` are integers that specify the direction of the wave vector (Mx, My, Mz) of the incident plane wave.
* ``f7`` is psi which defines the polarisation angle (degrees) of the incident plane wave.
* ``str1`` is the identifier of the waveform that should be used with the source.
* ``str2 f10 f11`` are optional parameters. ``str2`` is a material identifier that is the background material that the plane wave propagates through. The default value is ``free_space``. This material must also be the background material of your full model. ``f10`` is a time delay in starting the excitation of the discrete plane wave. ``f11`` is a time to remove the excitation of the discrete plane wave. If the time window is longer than the excitation of the discrete plane wave removal time then the excitation of the discrete plane wave will stop after the excitation of the discrete plane wave removal time. If the excitation of the discrete plane wave removal time is longer than the time window then the excitation of the discrete plane wave will be active for the entire time window. If ``f10 f11`` are omitted the excitation of the discrete plane wave will start at the beginning of time window and stop at the end of the time window.


For example, to specify a discrete plane wave in a TFSF box (0.010, 0.010, 0.010 to 0.040, 0.040, 0.040) propagating along the diagonal of your grid using a polarisation angle :math:`\psi` of 90 degrees, you can use as a vector (1,1,1) resulting in an azimuthal propagation angle :math:`\phi` of 45.0  degrees, polar propagation angle :math:`\theta` of approximately 54.736 degrees, and using the waveform defined by the identifier ``mypulse`` use: ``#plane_wave_vector: 0.010 0.010 0.010 0.040 0.040 0.040 1 1 1 90.0 mypulse``.

.. note::

    * Plane waves support non-dispersive dielectric backgrounds and multi-pole Debye, Lorentz, and Drude media. They do not currently support ``user``-defined waveforms.
    * The plane-wave command must be defined on the main grid. Its TFSF box may contain a complete subgrid, but must strictly enclose the subgrid's HSG outer coupling surface wherever the two regions overlap.
    * MPI domain decomposition is supported. Every rank advances an identical, small auxiliary one-dimensional DPW, while TFSF corrections are restricted to the Yee components owned by that rank. This adds no per-timestep plane-wave communication beyond the normal field-halo exchange.
    * This plane wave implementation was based on an initial implementation made possible by a `Google Summer of Code <https://summerofcode.withgoogle.com/>`_ (GSoC) project and `more details can be found in the original pull request <https://github.com/gprMax/gprMax/pull/373>`_.


#plane_wave_axial:
---------------------

Allows you to introduce a discrete plane wave source [TAN2010]_. Plane wave sources are a useful tool in multiple different scenarios of electromagnetic simulations, especially when the wave is emitted by a source that is quite far away from the target. This command introduces a plane wave that propagates along one of the three grid axes and can be normally incident on multi-layer setups that span the entire model domain perpendicular to the direction of propagation. It takes its media properties from the background materials of the grid at the direction of the axis that it propagates. This allows for half-space simulations but only for normally incident plane waves. The syntax of the command is:

.. code-block:: none

    #plane_wave_axial: f1 f2 f3 f4 f5 f6 f7 c1 str1 [f10 f11]

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the total field, scattered field (TFSF) box, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the total field, scattered field (TFSF) box.
* ``c1`` is a character that specifies the axis along which the incident plane wave propagates and can be ``x``, ``y``, or ``z``. originating at the lower left corner of the TFSF box and propagating in the positive axis direction.
* ``f7`` is psi which defines the polarisation angle (degrees) of the incident plane wave.
* ``str1`` is the identifier of the waveform that should be used with the source.
* ``f10 f11`` are optional parameters. ``f10`` is a time delay in starting the excitation of the discrete plane wave. ``f11`` is a time to remove the excitation of the discrete plane wave. If the time window is longer than the excitation of the discrete plane wave removal time then the excitation of the discrete plane wave will stop after the excitation of the discrete plane wave removal time. If the excitation of the discrete plane wave removal time is longer than the time window then the excitation of the discrete plane wave will be active for the entire time window. If ``f10 f11`` are omitted the excitation of the discrete plane wave will start at the beginning of time window and stop at the end of the time window.


For example, to specify a discrete plane wave in a TFSF box (0.010, 0.010, 0.010 to 0.040, 0.040, 0.040) propagating along the positive ``x`` direction using a polarisation angle :math:`\psi` of 90 degrees and using the waveform defined by the identifier ``mypulse`` use: ``#plane_wave_axial: 0.010 0.010 0.010 0.040 0.040 0.040 90.0 x mypulse``.

.. note::

    * For simulations that do not involve half-space setups it is recommended to use either the ``#plane_wave_angles`` or ``#plane_wave_vector`` commands instead as the formulations are more efficient and faster if the background medium of propagation for the plane wave is homogeneous.
    * Plane waves support non-dispersive dielectric layers and multi-pole Debye, Lorentz, and Drude layers. They do not currently support ``user``-defined waveforms.
    * The plane-wave command must be defined on the main grid. Its TFSF box may contain a complete subgrid, but must strictly enclose the subgrid's HSG outer coupling surface wherever the two regions overlap.
    * MPI domain decomposition is supported. The layered one-dimensional material profile is assembled collectively once during model construction using the actual electric, magnetic, and dispersive update-coefficient rows. The compact auxiliary profile is then replicated, and the timestep loop requires no plane-wave-specific MPI communication.
    * This plane wave implementation was based on an initial implementation made possible by a `Google Summer of Code <https://summerofcode.withgoogle.com/>`_ (GSoC) project and `more details can be found in the original pull request <https://github.com/gprMax/gprMax/pull/373>`_.



#eigenmode_band:
----------------

Defines the single frequency band shared by every eigenmode port in the model:

.. code-block:: none

    #eigenmode_band: str1 f1 f2 i1

* ``str1`` is a non-empty band identifier.
* ``f1`` and ``f2`` are the inclusive DFT start and stop frequencies in Hertz.
* ``i1`` is the number of uniformly spaced DFT points. A one-point band
  requires ``f1=f2``; a multi-point band requires ``f2>f1``.

Exactly one band is required when eigenmode ports are present. Defining the
band once guarantees identical DFT bins at every port.

#eigenmode_port:
----------------

Defines an active or passive modal reference plane. The same command is used
for every port; ``#eigenmode_excitation`` separately selects the one active
port and mode.

.. code-block:: none

    #eigenmode_port: i1 f1 f2 f3 f4 f5 f6 c1 i2[,i3 ...] str1|f7 [f8 ...] [c2]

* ``i1`` is the unique, one-based port number.
* ``f1 f2 f3`` and ``f4 f5 f6`` are opposite port-plane points in metres.
  In 3D exactly one finite coordinate pair must match, defining the normal.
  In 2D the normal must be in-plane and the port must span the complete
  invariant thickness; use ``inf`` for the upper invariant coordinate.
* ``c1`` is ``+`` or ``-`` and points into the device from the port.
* ``i2[,i3 ...]`` is a comma-separated, strictly increasing list of positive
  one-based modes to monitor, for example ``1`` or ``1,2``.
* ``str1`` can be ``auto``. One common automatic anchor list covers the
  requested band and the significant sampled source spectrum, including its
  transition regions, and is used by every automatic port.
* Alternatively, ``f7 [f8 ...]`` are explicit, strictly increasing modal
  anchor frequencies. Multiple anchors must cover the required range or the
  model is rejected with suggested anchors. A single explicit anchor is
  accepted intentionally as a constant modal basis across the complete band.
* ``c2`` optionally controls field plots: ``y`` always writes them and ``n``
  always suppresses them. If omitted, geometry-only runs write the plots and
  normal runs do not.

For example, these two rectangular-waveguide ports share one DFT band and one
automatic anchor list:

.. code-block:: none

    #eigenmode_band: wg_band 45e9 65e9 81
    #eigenmode_port: 1 0.002 0.001 0.001 0.002 0.007 0.005 + 1 auto
    #eigenmode_port: 2 0.011 0.001 0.001 0.011 0.007 0.005 - 1 auto

Consecutive anchors are checked using normalized modal-field overlap. If
explicit multiple anchors show a severe mismatch, such as at a degeneracy or
mode crossing, the run stops and recommends one explicit anchor. With
``auto``, a failure confined to a spectral guard outside the requested band
trims that tail only for the affected port and mode and uses its nearest
retained modal profile for endpoint extrapolation. A failure in the requested
band makes that port and mode warn and use its band-centre anchor. Candidate
frequencies remain common, but retained masks and fallbacks are per port and
per mode.

#eigenmode_excitation:
----------------------

Optionally selects the single active port and mode after the band and ports
have been defined. Omit this command only when every port is a passive virtual
guide; such a model writes raw modal spectra but no S matrix:

.. code-block:: none

    #eigenmode_excitation: i1 i2 [str1] [f1] [c1]

* ``i1`` is an existing port number.
* ``i2`` is one of that port's monitored mode indices.
* ``str1`` is ``auto`` by default. It constructs a finite, real band-pass
  pulse whose Gaussian-smoothed lower and upper edges adapt to the requested
  band, simulation time step, and Nyquist limit. The same waveform drives the
  source and its modal spectrum analysis. The pulse is placed at the earliest
  causal time that preserves its significant temporal support, leaving the
  remaining time window for propagation and ring-down.
* A custom waveform identifier may be supplied instead. Its exact sampled
  spectrum is checked before any modal solve. Significant DC or Nyquist bins
  are discarded with a warning. More than one percent spectral power outside
  the requested band remains an error. Use a band-limited waveform, or select
  ``auto`` to synthesize one automatically for a finite frequency band.
* ``f1`` is an optional amplitude scale and is valid only with ``auto``.
* ``c1`` optionally controls the waveform/DFT plot: ``y`` always writes it and
  ``n`` always suppresses it. If omitted, geometry-only runs write the plot and
  normal runs do not. The flag may follow ``i2`` directly when the default
  waveform and amplitude are used.

A complete excitation for the ports above is:

.. code-block:: none

    #eigenmode_excitation: 1 1 auto y

A single-frequency band cannot use the automatic finite-band pulse. Supply a
matching continuous waveform and one explicit modal anchor instead.

At every FDTD time step the source port and all passive ports project the
cell-centred transverse fields onto their requested modes. gprMax writes
``<output>_sparameters.csv`` and the corresponding ``/eigenmode_ports`` HDF5
groups. Source-port rows contain modal S11 results; other ports contain S21
and modal-conversion results. See :doc:`eigenmode_port` for the phasor convention,
FDFD formulation, power normalization, TF/SF injection, broadband synthesis,
and output definitions.

.. note::

    * Hash commands define eigenmode ports on the main grid, where the CPU,
      CUDA, OpenCL, and Metal solvers are supported. Direct eigenmode ports may
      also be added to an HSG subgrid through the Python API; they then use the
      fine-grid material slice, spatial step, time step, and CPU update cycle.
      The complete port stencil must remain strictly inside the subgrid working
      region. See :doc:`eigenmode_port`.
    * Domain-decomposed MPI CPU models are supported. Modal material slices
      are assembled collectively once, TF/SF corrections are restricted to
      locally owned Yee components, and modal DFT projections are reduced at
      finalisation.

#virtual_waveguide:
-------------------

Replaces the continuation behind an eigenmode-port plane with a finite
auxiliary FDTD waveguide terminated by a PML:

.. code-block:: none

    #virtual_waveguide: i1 [i2] [i3] [i4] [str1]

* ``i1`` is the one-based number of an existing ``#eigenmode_port``.
* ``i2`` is the total auxiliary-guide length in cells (default 30).
* ``i3`` is the remote PML thickness in cells (default 12).
* ``i4`` is the number of clear cells between an active modal source and the
  PML (default 6).
* ``str1`` optionally selects a reusable PML profile defined by named
  ``#pml_formulation`` and ``#pml_cfs`` commands. Use ``None`` to retain the
  global PML configuration while specifying all preceding positional values.

The guide direction and cross-section are inherited from the referenced port.
The material and Yee-component cross-section at the port is repeated through
the auxiliary grid. The normal H field and tangential E fields are coupled in
both directions at the aperture, so reflected modes enter the guide and are
absorbed by its remote PML.

If the referenced port is selected by ``#eigenmode_excitation``, the modal
source is placed inside the auxiliary guide. Otherwise the guide is passive.
All ports may be passive when every ``#eigenmode_port`` has a
``#virtual_waveguide``; raw incident and outgoing modal spectra are then saved,
but no S-parameters can be normalized without an active port.

The port plane must be internal, locally uniform along the guide axis, and at
least two cells wide in each transverse direction. The minimum guide length
is ``i3 + i4 + 3`` cells. Main-grid virtual waveguides support 3D,
non-dispersive guide cross-sections with the CPU, CUDA, OpenCL, and Metal
solvers. Through the Python API, a virtual waveguide may instead be attached
to an HSG-subgrid port; it then inherits that subgrid's fine material slice,
spatial and temporal steps, and CPU update cycle. Domain-decomposed MPI CPU
models are supported. The compact auxiliary guide is replicated, while one
aperture-sized collective communicates the three required H sheets after each
magnetic halo exchange.

Unlike a direct eigenmode source, a virtual-waveguide source lies outside the
main FDTD domain. A closed equivalent-current or KSIR NTFF surface may
therefore enclose the antenna and its physical feed aperture without omitting
a face. The integration surface must not intersect the virtual aperture.

#rx:
----

Allows you to introduce output points into the model. These are locations where the values of the electric and magnetic field components over the number of iterations of the model will be saved to file. The syntax of the command is:

.. code-block:: none

    #rx: f1 f2 f3 [str1 str2]

* ``f1 f2 f3`` are the coordinates (x,y,z) of the receiver in the model.
* ``str1`` is the identifier of the receiver.
* ``str2`` is a list of outputs with this receiver. It can be any selection from ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``, ``Ix``, ``Iy``, or ``Iz``.

.. note::

    * When the optional parameters ``str1`` and ``str2`` are not given all the electric and magnetic field components will be output with the receiver point.
    * On CUDA, OpenCL, and Metal, ``Ix``, ``Iy``, and ``Iz`` use the same
      single-cell Ampere loops as the CPU solver. Their device histories are
      allocated only for explicitly requested current components.

#rx_array:
----------

Provides a simple method of defining multiple output points in the model. The syntax of the command is:

.. code-block:: none

    #rx_array: f1 f2 f3 f4 f5 f6 f7 f8 f9

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the output line/rectangle/volume, and ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the output line/rectangle/volume.
* ``f7 f8 f9`` are the increments (x,y,z) which define the number of output points in each direction. ``f7``, ``f8``, or  ``f9`` can be set to zero to prevent any output points in a particular direction. Otherwise, the minimum value of ``f7`` is :math:`\Delta x`, the minimum value of ``f8`` is :math:`\Delta y`, and the minimum value of ``f9`` is :math:`\Delta z`.

#rx_port:
---------

Calculates the complex reflection coefficient and input impedance of a
single-cell voltage-source port. The output point must coincide
exactly with one ``#voltage_source`` after both positions have been resolved to
the Yee grid. A separate ``#rx`` command is not required. The syntax is:

.. code-block:: none

    #rx_port: f1 f2 f3 [str1 [str2]]

* ``f1 f2 f3`` are the source coordinates (x,y,z).
* ``str1`` is the optional port/output identifier. If omitted, ``port1``,
  ``port2``, and so on are generated.
* ``str2`` is the optional spectrum limit. A number specifies the minimum
  cells per shortest material wavelength; the default is 10. The keyword
  ``nyquist`` requests every native non-negative FFT bin for research and
  diagnostic use.

The voltage source supplies the reference impedance :math:`Z_0`. For example:

.. code-block:: none

    #voltage_source: z 0.050 0.050 0.020 50 source_wave
    #rx_port: 0.050 0.050 0.020 feed
    #rx_port: 0.050 0.050 0.020 feed nyquist
    #voltage_source: z 0.060 0.050 0.020 0 source_wave
    #rx_port: 0.060 0.050 0.020 ideal_feed
    # Custom 75 Ohm hard-source reference; start/stop precede it positionally
    #voltage_source: z 0.070 0.050 0.020 0 source_wave 0 10e-9 75
    #rx_port: 0.070 0.050 0.020 ideal_feed_75

``#rx_port`` is supported in domain-decomposed MPI CPU models. The source and
its internal field monitor belong to one rank; for a hard source, magnetic
halos are synchronised before the next current sample so an Ampere loop may
cross an internal rank face or corner. Port histories are gathered and the
frequency-domain quantities are calculated once on the coordinator rank.


For a finite-resistance source, the voltage-source resistance is the
reference impedance :math:`Z_0`; a hard source defaults to 50 Ohms unless
``f7`` is supplied. At the source plane, the known generator
spectrum :math:`V_g` and sampled total gap voltage :math:`V` give

.. math::

    S_{11,\mathrm{source}} = \frac{2V-V_g}{V_g}.

No current calculation is required in this finite-resistance case. gprMax
removes the parallel capacitance and background conductance of the source Yee
edge before reporting the antenna-terminal result. With
:math:`c=Z_0Y_\mathrm{gap}`, this correction and the input impedance are

.. math::

    S_{11} =
    \frac{2S_{11,\mathrm{source}}+c(1+S_{11,\mathrm{source}})}
         {2-c(1+S_{11,\mathrm{source}})},
    \qquad
    Z_\mathrm{in}=Z_0\frac{1+S_{11}}{1-S_{11}}.

For a zero-resistance source, the gap voltage is prescribed at integer
electric-field times. gprMax calculates the Ampere-loop current from the four
surrounding magnetic components. The voltage and current samples retain their
exact Yee times,

.. math::

    V=V^{n+1}, \qquad I_\mathrm{loop}=I_\mathrm{loop}^{n+1/2},

and the engineering-convention transforms apply the corresponding
:math:`\Delta t` and :math:`\Delta t/2` time offsets. This corrects their
relative phase without attenuating current by interpolation. The terminal
current is then

.. math::

    I_\mathrm{terminal}=I_\mathrm{loop}-Y_\mathrm{gap}V.

For this integer-voltage/half-step-current pairing, the discrete parallel-gap
admittance is

.. math::

    Y_\mathrm{gap} = G_\mathrm{bg}\cos\left(\frac{\omega\Delta t}{2}\right)
    +j\frac{2C_\mathrm{gap}}{\Delta t}
    \sin\left(\frac{\omega\Delta t}{2}\right).

This is the FDTD analogue of an ideal delta-gap MoM excitation: voltage is
imposed and the antenna current is a solved response. The user-supplied
:math:`Z_0` (or its 50 Ohm default) defines the travelling-wave normalisation
only. The reported
quantities are calculated directly as

.. math::

    Z_\mathrm{in}=\frac{V}{I_\mathrm{terminal}},
    \qquad
    V^\pm=\frac{V\pm Z_0 I_\mathrm{terminal}}{2},
    \qquad
    S_{11}=\frac{V^-}{V^+}.

The gap capacitance and conductance use the effective electric-edge material
before any artificial source resistance is added. The appropriate discrete
admittance is used in each source mode so the correction is consistent with
the trapezoidal Yee update and the mode's voltage/current sampling times.

By default, output stops at the first native FFT bin that does not have at
least 10 cells per shortest wavelength in the model. For nonmagnetic,
nondispersive media this is the material with the largest :math:`\epsilon_r`.
A numeric ``str2`` changes this sampling requirement; values below 10 produce
a warning and values below 3 are rejected. ``nyquist`` deliberately retains
the full spectrum but does not claim it is accurate: the lambda/10 limit and
per-bin mesh/source validity masks are still written to HDF5. The actual
stored range, native frequency resolution, Nyquist bound, and limiting
material are reported when the model is built.

.. note::

    * The implementation supports one finite-resistance or zero-resistance
      voltage source on the main 3D grid with the CPU, CUDA, OpenCL, or Metal
      solver. A hard-source port uses a 50 Ohm default :math:`Z_0`, which can
      be overridden on ``#voltage_source``.
    * MPI, subgrids, 2D modes, geometry-fixed runs, grouped sources, sources
      inside a PML, and dispersive material on the source edge are currently
      rejected. Dispersive materials elsewhere in the model are supported.
      A hard-source current loop cannot lie on a domain-minimum transverse
      boundary.
    * ``S11`` remains the primary result. ``Zin`` is singular near an open
      circuit (:math:`S_{11}=1`), so gprMax also stores ``Yin`` and separate
      validity masks.
    * A time trace that has not decayed before the end of the model window can
      contaminate the spectrum. gprMax reports a tail-level warning rather
      than hiding or clipping the result.

#network_port:
--------------

Requests time- and frequency-domain port quantities for an existing
``#network_terminal``. The terminal ID is also the HDF5 port ID. The syntax is

.. code-block:: none

    #network_port: str1 [f1 [f2]]

* ``str1`` is the network-terminal ID.
* ``f1`` is the positive real power-wave reference impedance in ohms; it is
  50 Ohms by default.
* ``f2`` is the minimum number of cells per shortest material wavelength; it
  is 10 by default. The literal ``nyquist`` retains the full native spectrum
  with validity masks for research use.

The output contains terminal voltage and current, incident and reflected
power-wave spectra, ``S11``, ``Zin``, and ``Yin``. gprMax removes the
background Yee-gap capacitance and conductance from the reported terminal
current. A driven network obtains ``S11`` and incident power; an unexcited
network remains useful as a passive measured port, for which ``Zin`` and
``Yin`` are still available when numerically defined.

#src_steps: and #rx_steps:
--------------------------

Provides a simple method to allow you to move the location of all simple sources (``#src_steps``) or all receivers (``#rx_steps``) between runs of a model. The syntax of the commands is:

.. code-block:: none

    #src_steps: f1 f2 f3
    #rx_steps: f1 f2 f3

``f1 f2 f3`` are increments (x,y,z) to move all simple sources (``#hertzian_dipole`` or ``#magnetic_dipole``) or all receivers (created using either ``#rx`` or ``#rx_array`` commands).

.. note::

    * ``#src_steps`` and ``#rx_steps`` are not suitable for moving sources which have associated geometry, e.g. antenna models.

#study:
-------

Runs a sequence of source/receiver cases while building the model geometry
only once. This is the general counterpart to ``#src_steps`` and
``#rx_steps``: positions need not follow a regular increment, individual
sources can be activated or scaled, and every output file records the exact
case that produced it. The syntax is:

.. code-block:: none

    #study: gpr file1

The study type may be ``gpr`` for irregular source/receiver acquisition,
``source`` for fixed-topology stateful terminal sources, ``port`` for a
finite-resistance voltage-source S-parameter study, ``eigenmode`` for a
modal-port S-parameter study, or ``plane_wave`` for an angular TFSF/RCS
study:

.. code-block:: none

    #study: source file1
    #study: port file1
    #study: eigenmode file1
    #study: plane_wave file1

``file1`` is a CSV table. Its path is resolved relative to the main input
file. The required columns are ``case_id`` and ``object_id``. The optional
columns are ``active``, ``x_m``, ``y_m``, ``z_m``, ``waveform_id``,
``start_s``, ``stop_s``, ``scale``, ``record``, ``port``, and ``mode``. Blank
cells mean "use the object's baseline value". All three position columns must
be supplied together and contain absolute coordinates in metres. ``port`` and
``mode`` are positive integers used only by an eigenmode study.

A plane-wave table uses the deterministic object ID ``plane_wave_1``. Its
additional optional columns are ``theta_deg``, ``phi_deg``, ``psi_deg``,
``axis``, ``m_x``, ``m_y``, and ``m_z``. The permitted columns follow the
plane-wave command used in the model: angles use theta/phi/psi, vector sources
use all three integer mapping columns and psi, and axial sources use axis and
psi. ``waveform_id``, ``start_s``, ``stop_s``, and a non-zero ``scale`` are
available for every form. For example:

.. code-block:: text

    case_id,object_id,theta_deg,phi_deg,psi_deg,scale
    x_incidence,plane_wave_1,90,0,90,1
    y_incidence,plane_wave_1,90,90,90,1

The TFSF box and its background material remain fixed. gprMax rebuilds the
auxiliary one-dimensional discrete plane wave and resets/recompiles
declarative NTFF accumulators for each row, while retaining the main model
geometry. NTFF observation directions also remain fixed: request all angles
needed across the cases in the input file and select the appropriate direction
from each numbered output file.

A ``source`` study manages main-grid ``#transmission_line``,
``#magnetic_frill_source``, and ``#network_excitation`` commands. Their
deterministic IDs are ``transmission_line_1``, ``magnetic_frill_source_1``,
and ``network_excitation_1`` (and so on within each family). Positions,
impedances, thin-wire/coax geometry, and rational-network definitions are
fixed. The CSV may vary ``active``, ``waveform_id``, ``start_s``, ``stop_s``,
and ``scale``. For example:

.. code-block:: text

    case_id,object_id,active,waveform_id,start_s,stop_s,scale
    full_drive,transmission_line_1,true,pulse,0,4e-9,1
    half_drive,transmission_line_1,true,pulse,0,4e-9,0.5
    passive,transmission_line_1,false,,,,

Any number of listed sources may be active in a case. An omitted or inactive
source keeps its physical terminal and acts as a passive termination with
zero generator drive. gprMax resets the terminal's internal field, recurrence,
history, and derived port state before every case. Declarative NTFF outputs
are also rebuilt with pristine accumulators. ``source`` studies are therefore
suitable for comparing multiple-feed antenna excitations, but do not
calculate a complete S matrix; use ``port`` or ``eigenmode`` for that purpose.
The normal CPU, CUDA, OpenCL, and Metal implementations are available.
Stateful sources inside subgrids, MPI, and task farming are not currently
supported by this study type.

For example:

.. code-block:: text

    case_id,object_id,active,x_m,y_m,z_m,waveform_id,start_s,stop_s,scale,record
    trace_1,hertzian_dipole_1,true,0.100,0.050,0.030,,,,1,
    trace_1,rx_1,,0.140,0.050,0.030,,,,,true
    trace_2,hertzian_dipole_1,true,0.102,0.052,0.030,,,,0.8,
    trace_2,rx_1,,0.145,0.052,0.030,,,,,true

Objects receive deterministic IDs from their order of appearance within each
object family: ``hertzian_dipole_1``, ``hertzian_dipole_2``,
``magnetic_dipole_1``, ``voltage_source_1``, and ``rx_1``. An explicit
``#rx`` identifier is also accepted as an alias. A source listed in a case is
active by default; a source omitted from that case, or listed with
``active=false``, is inactive. A receiver omitted from a case remains at its
baseline position and is still recorded. ``record=false`` is reserved for
future selective-output support and is currently rejected rather than
silently ignored.

For a ``port`` study, every ``#voltage_source`` must have finite, non-zero
resistance and a coincident ``#rx_port`` with a unique ID. The CSV must contain
one case for every voltage source and drive exactly one source in each case.
Omitted sources retain their fixed source resistance but receive a zero
generator waveform, so they behave as passive matched terminations. For
example, a two-port schedule is:

.. code-block:: text

    case_id,object_id,active,scale
    drive_port1,voltage_source_1,true,1
    drive_port2,voltage_source_2,true,1

Source position and resistance cannot vary in a port study because they are
part of the built electric-edge material. Hard sources are not accepted: zero
drive on a hard source enforces zero field and is not a matched termination.
Each case output stores its source-plane S-matrix column. After the cases
finish, ``<output>_study.h5`` stores the complete source-plane and
gap-corrected matrices using
``S[frequency, output_port, input_port]``. The correction removes all numerical
gap capacitances/conductances through the full admittance matrix, including
the coupled off-diagonal terms.

For an ``eigenmode`` study, every case contains the single deterministic
object ``eigenmode_excitation_1`` and selects one declared port/mode channel.
Every mode on every ``#eigenmode_port`` must be selected exactly once:

.. code-block:: text

    case_id,object_id,port,mode
    p1m1,eigenmode_excitation_1,1,1
    p2m1,eigenmode_excitation_1,2,1

All port modal anchors are solved during the first geometry build, including
source-grade spectral-guard anchors. Later cases reuse those bases, reset the
modal DFT and any ``#virtual_waveguide`` fields/PML history, and switch the
active channel without another FDFD solve. The aggregate file stores
``S[frequency, output_channel, input_channel]`` together with ``channel_ports``
and ``channel_modes``.

The number of CSV cases determines the number of model runs, so ``-n`` is not
required. ``-i N`` restarts at case ``N`` and retains absolute output numbering.
The original geometry, materials, PMLs, and grid allocation are reused, but
field arrays and receiver histories are reset before every case. Each output
contains a ``/study`` group with the case ID, resolved parameters, and a copy
of the CSV source.

.. note::

    GPR studies support top-level ``#hertzian_dipole``,
    ``#magnetic_dipole``, and ``#rx`` objects; port studies additionally
    support finite-resistance ``#voltage_source``/``#rx_port`` pairs;
    eigenmode studies support ``#eigenmode_port``, ``#eigenmode_excitation``,
    and ``#virtual_waveguide``. MPI domain decomposition, task farming, plane
    waves, transmission lines, and rational/frill ports remain excluded from
    studies until their family-specific state reset hooks are implemented.

#snapshot:
----------

Allows you to obtain information about the electromagnetic fields within a volume of the model at a given time instant. The file(s) use the open source `Visualization ToolKit (VTK) <http://www.vtk.org>`_ format which can be viewed in many free readers, such as `Paraview <http://www.paraview.org>`_. The syntax of this command is:

.. code-block:: none

    #snapshot: f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 file1

or

.. code-block:: none

    #snapshot: f1 f2 f3 f4 f5 f6 f7 f8 f9 i1 file1

* ``f1 f2 f3`` are the lower left (x,y,z) coordinates of the volume of the snapshot in metres.
* ``f4 f5 f6`` are the upper right (x,y,z) coordinates of the volume of the snapshot in metres.
* ``f7 f8 f9`` are the spatial discretisation of the snapshot in metres.
* ``f10`` or ``i1`` are the time in seconds (float) or the iteration number (integer) which denote the point in time at which the snapshot will be taken.
* ``file1`` is the name of the file where the snapshot will be stored. Snapshot files are automatically stored in a directory with the name of the input file appended with '_snaps'. For multiple model runs each model run will have its own directory, i.e. '_snaps1', 'snaps2' etc...

For example to save a snapshot of the electromagnetic fields in the model at a simulated time of 3 nanoseconds use: ``#snapshot: 0 0 0 1 1 1 0.1 0.1 0.1 3e-9 snap1``

.. tip::
    A series of snapshots can be more easily defined using a loop and our :ref:`Python API <input-api>`, see :ref:`outputs-snaps`.

    The Python API can also add a snapshot to an HSG subgrid. Such a snapshot
    uses the subgrid's finer spatial discretisation and time step, while its
    file origin remains in the global model coordinate system. The hash
    command above always defines a main-grid snapshot.

Near-to-far-field transformation commands
==========================================

The NTFF commands separate the integration surface from the formulation and
its output points. A closed surface can therefore be reused by KSIR and
equivalent-current transforms, and by many output directions, without
repeating the six surface coordinates. An open surface is specific to the
frequency-domain Huygens/equivalent-current transform, which can use any
user-selected nonempty subset of the six faces. All optional parameters
use the traditional positional gprMax syntax; ``name=value`` tokens are not
used.
The derivations, field normalisation, Yee placement, and comparison of the
three implemented transforms are given in :ref:`ntff-formulations`.

The following conventions apply to every NTFF command:

* coordinates and radii are in metres, frequencies are in Hz, and angles are
  in degrees;
* requested frequencies must not exceed the temporal Nyquist limit
  :math:`1/(2\Delta t)` for the model time step;
* :math:`\theta` is the polar angle measured from ``+z`` and :math:`\phi` is
  the azimuth measured from ``+x`` towards ``+y``;
* a spherical coordinate is relative to the centre of its integration
  surface. The Python API can instead give a custom surface origin;
* Cartesian outputs are ``Ex Ey Ez Hx Hy Hz``. Spherical outputs are
  ``Er Etheta Ephi Hr Htheta Hphi``. Far-field derived outputs are described
  under ``#ksir_far_field`` below;
* every exact time- or frequency-domain point must be strictly outside the
  completed integration surface. A point may be outside the FDTD model domain;
* the sampled surface and exterior must be one homogeneous, lossless,
  non-dispersive material. gprMax determines its wave speed and impedance
  from the Yee material IDs; these are not user-entered command parameters;
* surface samples must remain outside the PML and clear of the TFSF correction
  stencil. A closed surface must enclose the radiating source or the complete
  TFSF box and scatterer. An open Huygens surface instead permits an impressed
  source outside only through one of its omitted faces;
* the implementation requires a three-dimensional model and does not support
  geometry-fixed reuse. NTFF commands are main-grid objects, but their closed
  surface may contain complete subgrids. A surface that overlaps a subgrid
  must strictly enclose its HSG outer coupling surface; it cannot touch or cut
  the coupling region. Both time- and frequency-domain KSIR collection and
  frequency-domain equivalent-current collection are available with the
  serial CPU, CUDA, OpenCL, Metal, and MPI domain-decomposition solvers.
  Equivalent-current transient far fields are available with the same
  solvers. MPI uses the CPU field-update backend;
* CPU collection uses the Cython/OpenMP implementation. Accelerator surface
  state and time-domain output storage remain on the device during FDTD
  iterations and are transferred to the host once, after the solve. CUDA and
  OpenCL are hardware-qualified on the development server. Metal has complete
  source-generation and dispatch coverage, but still requires execution tests
  on suitable Apple hardware. With MPI, every surface patch is sampled by the
  rank which owns its inside Yee sample; the neighbouring outside sample is
  read from the normal one-cell halo. There is therefore no additional NTFF
  communication inside the FDTD iteration. Compact time histories are reduced
  and frequency-domain surface phasors are assembled on the coordinator after
  time stepping, before the normal HDF5 output is written.

Directivity, gain, efficiency, and port normalisation are post-processing
operations after the FDTD solve. Angular KSIR and equivalent-current
evaluation uses Cython/OpenMP kernels on the host for CPU and accelerator
simulations alike;
it does not add a new per-iteration GPU operation or transfer fields back to
the CPU during time stepping.

A minimal dipole workflow can reuse one surface for an exact KSIR time-domain
point and an equivalent-current frequency-domain radiation pattern:

.. code-block:: none

    #domain: 0.1 0.1 0.1
    #dx_dy_dz: 0.002 0.002 0.002
    #time_window: 10e-9

    #waveform: gaussiandot 1 1e9 pulse
    #hertzian_dipole: z 0.05 0.05 0.05 pulse

    #ntff_surface: 0.03 0.03 0.03 0.07 0.07 0.07 radiation_surface
    #ksir_time_rx: 0.12 0.05 0.05 radiation_surface transient Ez first_arrival
    #ntff_frequency: radiation_surface antenna_band 0.8e9 1.0e9 1.2e9 hann
    #ntff_far_field_array: 0 180 5 0 360 5 antenna_band pattern Etheta Ephi radiation_intensity

The observation point may lie outside the FDTD domain because KSIR evaluates
the homogeneous exterior analytically. The integration surface itself must be
inside the non-PML FDTD region and enclose the source.

KSIR independently reconstructs each Cartesian field component and is the
only formulation that provides finite-distance fields. The conventional
equivalent-current formulation combines tangential electric and magnetic
fields and provides far-zone fields only. Its commands use the ``ntff``
prefix. Both frequency-domain formulations support the same far-field,
antenna, and RCS outputs, so they can be requested together for comparison.

#ntff_surface:
--------------

Defines a reusable Yee-aligned cuboidal integration surface:

.. code-block:: none

    #ntff_surface: x1 y1 z1 x2 y2 z2 surface_id [omit_face1 ... omit_face5]

``x1 y1 z1`` and ``x2 y2 z2`` are the lower and upper logical corners.
``surface_id`` must be unique and must not contain ``/``. Zero to five trailing
face names may be supplied from ``x0``, ``xmax``, ``y0``, ``ymax``, ``z0``,
and ``zmax``; duplicates are not allowed. Listed faces are omitted from the
frequency-domain Huygens/equivalent-current integral. At least one face must
remain active. An impressed source outside the Huygens volume must enter
through an omitted face. A feed crossing an opening should remain uniform to
the corresponding PML, with its impressed source plane outside the volume.
Every sampled face must lie in the homogeneous exterior. This open-surface form
is rejected by every KSIR/Ramahi command and by the transient equivalent-current
transform.

An arbitrary open surface is not a mathematically exact closure. It is useful
when the omitted contribution is intentionally excluded, but its far field can
depend on the selected faces. Check convergence by moving and enlarging the
sampled faces. The edge rows shared with an omitted face are excluded from the
homogeneous-material check, allowing an active side face to terminate on a PEC
backplane.

.. warning::

   The NTFF integration surface is not closed. Equivalent-current NTFF normally
   assumes a closed Huygens surface. This option is intended for configurations
   where the omitted face is associated with an eigenmode port or other
   modelling scenarios that require an open Huygens surface. Results may be
   incomplete or inaccurate if the omitted field contribution is not
   represented correctly or is significant for your calculations.

Without omitted face names, the surface is physically closed unless one or more
faces coincide exactly with a declared
``#symmetry_boundary``. Coincident PEC/PMC faces are then omitted from direct
sampling and completed automatically using the exact image parity, reflected
normal, edge quadrature, and propagation distance for every component.

For example:

.. code-block:: none

    #ntff_surface: 0.034 0.034 0.034 0.066 0.066 0.066 radiation_surface

An antenna fed from the negative x direction can omit its feed face:

.. code-block:: none

    #ntff_surface: 0.034 0.020 0.020 0.080 0.080 0.080 radiation_surface x0

A leaky-wave antenna can omit both waveguide end faces so that a passive
eigenmode receiver beyond ``xmax`` still measures S21:

.. code-block:: none

    #ntff_surface: 0.034 0.020 0.020 0.080 0.080 0.080 radiation_surface x0 xmax

A surface whose lower side terminates on a PEC backplane can omit ``z0``:

.. code-block:: none

    #ntff_surface: 0.020 0.020 0.030 0.080 0.080 0.080 radiation_surface z0

#ksir_frequency:
----------------

Declares a streaming frequency transform for a previously defined surface:

.. code-block:: none

    #ksir_frequency: surface_id transform_id f1 [f2 ...] [window]

``transform_id`` must be globally unique. At least one non-negative frequency
is required. The optional final ``window`` is ``rectangular`` (the default) or
``hann``. Frequencies above the temporal Nyquist limit are rejected rather
than being silently aliased. Frequencies are accumulated directly during time
stepping; field histories are not retained. Surface phasors are saved under
the transform's HDF5 group so that it remains useful even when it has no receiver
or far-field command.

The engineering convention is used throughout: phasors have time dependence
``exp(+j*omega*t)``, the forward transform kernel is ``exp(-j*omega*t)``, and
the outgoing Green function contains ``exp(-j*k*R)``.

A frequency transform still requires a sufficiently long FDTD time window.
The ``hann`` window reduces leakage from a truncated non-zero tail, but it is
not a replacement for allowing the physical surface fields to decay. A
``rectangular`` window retains the unmodified engineering phasor and is
required by gain normalisation, but is more sensitive to end-of-record
truncation.

.. code-block:: none

    #ksir_frequency: radiation_surface antenna_band 0.8e9 1.0e9 1.2e9 hann

#ntff_frequency:
----------------

Declares a streaming frequency transform using the conventional
equivalent-current method of Luebbers *et al.* [LUE1991]_:

.. code-block:: none

    #ntff_frequency: surface_id transform_id f1 [f2 ...] [window]

The frequency, window, engineering convention, Nyquist check, and globally
unique transform-ID rules are identical to ``#ksir_frequency``. The
tangential Yee fields are arithmetically collocated on common cell-face
centres and form

.. math::

    \mathbf J_s=\hat{\mathbf n}\times\mathbf H,
    \qquad
    \mathbf M_s=-\hat{\mathbf n}\times\mathbf E.

If

.. math::

    \mathbf N=\oint_S\mathbf J_s
      e^{+jk\hat{\mathbf r}\cdot(\mathbf r'-\mathbf r_0)}\,\mathrm dS',
    \qquad
    \mathbf L=\oint_S\mathbf M_s
      e^{+jk\hat{\mathbf r}\cdot(\mathbf r'-\mathbf r_0)}\,\mathrm dS',

the stored range-normalized electric far field is

.. math::

    \mathbf F_E
    =-\frac{jk}{4\pi}
      \left[\eta\left(\mathbf N-
      \hat{\mathbf r}(\hat{\mathbf r}\cdot\mathbf N)\right)
      -\hat{\mathbf r}\times\mathbf L\right].

Here :math:`\mathbf r_0` is the surface phase origin. This transform cannot be
used by the finite-distance ``#ksir_frequency_rx`` commands. It is consumed by
``#ntff_far_field``, ``#ntff_far_field_array``, and optionally
``#ntff_antenna_ports``. It accepts either all six physical integration faces
or any nonempty subset selected by trailing omitted face names on
``#ntff_surface``. Symmetry-completed equivalent-current
surfaces are not yet enabled.

.. code-block:: none

    #ntff_frequency: radiation_surface current_band 0.8e9 1.0e9 1.2e9 hann

#ksir_antenna_ports:
--------------------

Associates a complete set of physical antenna ports with a frequency
transform so that accepted power, gain, realized gain, and efficiency can be
calculated:

.. code-block:: none

    #ksir_antenna_ports: transform_id port_id1 [port_id2 ...]

For a voltage source, ``port_id`` is the ID of the coincident ``#rx_port``.
Transmission-line and magnetic-frill sources provide automatic port IDs
``tl1``, ``tl2``, ... and ``frill1``, ``frill2``, ... respectively, in source
creation order. The association is not required for electric or magnetic far
fields, radiation intensity, RCS, or directivity. It is required when a
far-field command asks for gain or efficiency.

A direct eigenmode source cannot be used with any Ramahi/KSIR command. When
its active port has a ``#virtual_waveguide``, the source is moved outside the
main FDTD domain and a closed KSIR or equivalent-current surface may be used.
Without a virtual guide, use the frequency-domain equivalent-current Huygens
commands ``#ntff_frequency``, ``#ntff_far_field`` or
``#ntff_far_field_array``, and ``#ntff_antenna_ports`` instead.

A port on a subgrid is named as ``subgrid_id/port_id``. For example,
``fine_grid/feed`` identifies an ``#rx_port`` called ``feed`` on subgrid
``fine_grid``; automatic source ports use forms such as ``fine_grid/tl1`` and
``fine_grid/frill1``. Main-grid IDs remain unqualified. Each subgrid port is
post-processed using its owning grid's finer spatial and temporal steps.

The listed set must include **every** physical voltage, transmission-line, and
magnetic-frill port in the model. Every voltage source must therefore have a
coincident ``#rx_port``. This requirement makes the net accepted power
unambiguous in coupled multiport antennas. A source whose waveform amplitude is
zero is still a terminated physical port: list it normally. It has zero
incident power, but coupling from driven ports can make its accepted power
negative because it delivers coupled energy into its termination. gprMax sums
all signed port powers when normalising antenna gain.

Gain normalisation currently requires the transform to use the
``rectangular`` window. Active Hertzian electric or magnetic dipoles and
plane-wave sources cannot be mixed with a port-normalised antenna result,
because their input power is not represented by this port set.
The normal per-port wavelength-sampling limit also applies to gain validity.
For a voltage-source port, an explicit ``nyquist`` research override on its
``#rx_port`` retains the full temporal band, including spatially
under-resolved values, as it does for S11 and impedance. A coincident
``#rx_port`` can apply the same override to a magnetic-frill output.

For example, a two-element array with one driven and one terminated element
uses:

.. code-block:: none

    #waveform: ricker 1 1e9 driven
    #waveform: ricker 0 1e9 terminated
    #voltage_source: z 0.045 0.050 0.050 50 driven
    #voltage_source: z 0.055 0.050 0.050 50 terminated
    #rx_port: 0.045 0.050 0.050 element1
    #rx_port: 0.055 0.050 0.050 element2
    #ksir_frequency: radiation_surface antenna_band 0.8e9 1.0e9 1.2e9 rectangular
    #ksir_antenna_ports: antenna_band element1 element2
    #ksir_far_field_array: 0 180 2 0 360 2 antenna_band pattern gain realized_gain

#ntff_antenna_ports:
--------------------

Associates antenna ports with an equivalent-current frequency transform:

.. code-block:: none

    #ntff_antenna_ports: transform_id port_id1 [port_id2 ...]

Its port-set, rectangular-window, multiport-power, subgrid, and validity rules
are the same as for ``#ksir_antenna_ports``. The separate command name prevents
accidentally associating a port set with a transform from the other
formulation.

This is the antenna-port association to use with an eigenmode excitation.
Every ``#eigenmode_port`` uses ``portN``, where ``N`` is its explicit port
number. The listed set must include every physical conventional and modal
port in the model. A passive eigenmode port has zero generator incident power
and contributes signed net modal power to the accepted-power balance.

For every associated eigenmode port, the transform frequencies must exactly
match that port's direct-DFT bins. Degenerate modes and mode crossings should
use a single-frequency modal solve rather than broadband profile
interpolation.

#ksir_time_rx: and #ksir_time_rx_spherical:
--------------------------------------------

Request exact physical time-domain fields at one Cartesian or spherical
point:

.. code-block:: none

    #ksir_time_rx: x y z surface_id [rx_id [output1 output2 ... [time_origin]]]
    #ksir_time_rx_spherical: r theta phi surface_id [rx_id [output1 output2 ... [time_origin]]]

The Cartesian command defaults to all six Cartesian components. The spherical
command defaults to all six spherical components. ``rx_id`` is optional and
is generated as ``rx1``, ``rx2``, and so on when omitted. ``time_origin`` is
the final token and is either:

* ``simulation`` (default): retain time from the start of the FDTD run; or
* ``first_arrival``: omit the guaranteed retarded-time zero prefix separately
  for each point while recording its absolute physical origin.

Optional parameters are positional. Therefore an ID must be supplied before
component names, and both the ID and any desired components must precede
``time_origin``. For example:

.. code-block:: none

    #ksir_time_rx: 0.074 0.05 0.051 radiation_surface outside Ez Hy first_arrival
    #ksir_time_rx_spherical: 0.25 90 0 radiation_surface principal Etheta Ephi simulation

The spherical radius is explicit because these commands return the actual
finite-distance field, including all ``1/R`` and ``1/R^2`` terms. It is not a
normalization constant.

The output file stores both ``fully_supported_lengths`` and ``valid_lengths``.
Use the former by default: it stops before retarded propagation causes any
surface patch to run beyond the available FDTD history. The latter exposes the
longer partial tail for research use. gprMax also records a terminal decay
ratio and warns when the field has not decayed adequately within the fully
supported interval; in that case the simulation time window should be
increased.

#ksir_time_rx_array:
--------------------

Defines a Cartesian line, plane, or volume of exact time-domain points:

.. code-block:: none

    #ksir_time_rx_array: x1 y1 z1 x2 y2 z2 dx dy dz surface_id [rx_id [output1 output2 ... [time_origin]]]

The bounds are inclusive and must contain an integer number of increments.
An increment can be zero only on an axis whose lower and upper coordinates are
equal. All points share one output ID and are stored as the first dimension of
each field dataset.

#ntff_time_far_field: and #ntff_time_far_field_array:
-----------------------------------------------------

Request transient far-zone fields using the modified one-step
equivalent-current construction of Giannopoulos *et al.* [GIAFF1997]_:

.. code-block:: none

    #ntff_time_far_field: theta phi surface_id [output_id [output1 output2 ...]]
    #ntff_time_far_field_array: theta1 theta2 dtheta phi1 phi2 dphi surface_id [output_id [output1 output2 ...]]

The default outputs are ``Etheta Ephi``. Any Cartesian or spherical electric
or magnetic component may be requested. These are range-normalized far-zone
waveforms, not finite-distance receivers, so the commands deliberately have no
radius. If :math:`\tau=t-r/c_b`, the electric result is

.. math::

    \mathbf F_E(\hat{\mathbf r},\tau)
    =-\frac{1}{4\pi c_b}\oint_S
      \left[\eta\,\dot{\mathbf J}_{s,t}
      -\hat{\mathbf r}\times\dot{\mathbf M}_s\right]
      \left(\tau+\frac{\hat{\mathbf r}\cdot
      (\mathbf r'-\mathbf r_0)}{c_b}\right)\,\mathrm dS'.

Here :math:`\mathbf J_{s,t}` is the component of
:math:`\mathbf J_s=\hat{\mathbf n}\times\mathbf H` transverse to the
observation direction and
:math:`\mathbf M_s=-\hat{\mathbf n}\times\mathbf E`. The electric and
magnetic current differences retain their natural half-step offset: the
derivative of :math:`\mathbf M_s^n` is placed at
:math:`(n-1/2)\Delta t`, and the derivative of
:math:`\mathbf J_s^{n+1/2}` at :math:`n\Delta t`. No extra interpolation is
used to force them onto one FDTD time level. This is the modification to the
original Luebbers method [LUE1991]_ introduced by Giannopoulos *et al.*
[GIAFF1997]_. Linear interpolation is used only for fractional propagation
delays.

The stored reduced-time axis excludes both the range-dependent leading-zero
delay and all final bins that are not supported by every integration patch.
This prevents an incomplete retarded-time tail being mistaken for a physical
late-time response. Increase ``#time_window`` if the stored terminal-decay
test fails.

The current implementation supports the CPU, CUDA, OpenCL, and Metal solvers,
requires a homogeneous lossless background and all six physical faces, and
supports 3-D serial models. KSIR
remains available for finite-distance time-domain points and for
symmetry-completed surfaces.

.. code-block:: none

    #ntff_time_far_field_array: 0 180 2 0 360 2 radiation_surface transient Etheta Ephi

#ksir_frequency_rx: and #ksir_frequency_rx_spherical:
------------------------------------------------------

Request the exact finite-distance physical phasor at one point using a
previously declared transform:

.. code-block:: none

    #ksir_frequency_rx: x y z transform_id [rx_id [output1 output2 ...]]
    #ksir_frequency_rx_spherical: r theta phi transform_id [rx_id [output1 output2 ...]]

The Cartesian and spherical component defaults match their time-domain
counterparts. These results retain the full outgoing Green function and are
not range normalized. Their arrays have shape ``(nfrequencies, npoints)``;
``npoints`` is one for these two commands.

.. code-block:: none

    #ksir_frequency_rx: 0.074 0.05 0.051 antenna_band near_phasor Ez
    #ksir_frequency_rx_spherical: 0.25 90 0 antenna_band spherical_phasor Etheta Ephi

#ksir_frequency_rx_array:
-------------------------

Defines a Cartesian line, plane, or volume of exact frequency-domain points:

.. code-block:: none

    #ksir_frequency_rx_array: x1 y1 z1 x2 y2 z2 dx dy dz transform_id [rx_id [output1 output2 ...]]

The inclusive bounds and zero-increment rule are the same as for
``#ksir_time_rx_array``.

#ksir_far_field:
----------------

Requests a range-normalized far field in one spherical direction:

.. code-block:: none

    #ksir_far_field: theta phi transform_id [output_id [output1 output2 ...]]

The default outputs are ``Etheta Ephi``. Cartesian and spherical electric or
magnetic components may be requested. The derived outputs are
``radiation_intensity``, ``directivity``, ``directivity_dbi``, ``gain``,
``gain_dbi``, ``realized_gain``, ``realized_gain_dbi``,
``radiation_efficiency``, ``total_efficiency``, and ``rcs``. Linear and dBi
forms are separate outputs; gprMax does not silently convert one into the
other.

For the range-normalized electric field, radiation intensity is

.. math::

    U(\theta,\phi,f)
    = \frac{|F_\theta|^2+|F_\phi|^2}{2\eta},

where :math:`\eta` is the wave impedance of the homogeneous material around
the NTFF surface. When directivity or either efficiency is requested, gprMax
also evaluates a temporary full sphere using Gauss--Legendre quadrature in
:math:`\cos\theta` and periodic quadrature in :math:`\phi`:

.. math::

    P_\mathrm{rad}(f) = \int_{4\pi}U(\theta,\phi,f)\,\mathrm{d}\Omega,
    \qquad
    D(\theta,\phi,f) = \frac{4\pi U(\theta,\phi,f)}{P_\mathrm{rad}(f)}.

The quadrature order is selected from the largest requested value of
:math:`ka`, where :math:`a` is the bounding radius of the completed
integration surface. The temporary
full-sphere fields are processed in blocks and are not stored; only the
radiated power, estimated pattern maximum and its direction, and quadrature
metadata are retained. The maximum estimate is additionally refined with the
directions explicitly requested for that output, so a fine user grid cannot
report a larger directivity than the stored maximum. Therefore a user may request only a principal-plane
cut and still obtain correctly full-sphere-normalised directivity.

For an associated antenna port set, the exact-frequency terminal spectra give

.. math::

    P_\mathrm{acc}(f)
    = \sum_p \frac{1}{2}\Re\{V_p I_p^*\},
    \qquad
    P_\mathrm{inc}(f)
    = \sum_p \frac{|V_p^+|^2}{2Z_{0p}}.

The requested gain quantities are

.. math::

    G = \frac{4\pi U}{P_\mathrm{acc}}, \qquad
    G_\mathrm{realized} = \frac{4\pi U}{P_\mathrm{inc}},

and the scalar efficiencies stored for each frequency are

.. math::

    \eta_\mathrm{rad}=\frac{P_\mathrm{rad}}{P_\mathrm{acc}}, \qquad
    \eta_\mathrm{total}=\frac{P_\mathrm{rad}}{P_\mathrm{inc}}.

All surface and terminal spectra use the same engineering DFT and transform
scale, which cancels in these dimensionless ratios. Frequencies below
``-40 dB`` of the peak total incident spectrum, invalid terminal samples, or
non-positive normalising powers are written as ``NaN`` and marked invalid in
the HDF5 port metadata. A radiation efficiency materially above unity emits a
warning and normally indicates an insufficient time window, mesh error,
integration-surface error, or inconsistent port definition.

``rcs`` requests bistatic radar cross section and requires a TFSF plane-wave
source. The NTFF surface must strictly enclose the TFSF box and be clear of
its field-correction stencil. With hash commands, exactly one plane wave must
be present and it is associated automatically. Use one plane wave per
simulation for an unambiguous RCS result. RCS and port-normalised gain belong
to different excitation workflows and cannot be combined in one result.

Unlike the exact spherical receiver commands, ``#ksir_far_field`` has no
radius. Each field component is the range-normalized quantity

.. math::

    F_\mathrm{s}(\theta,\phi,f)
    = r\,\exp(+jkr)\,E_\mathrm{s}(r,\theta,\phi,f),

in the far-zone limit, where the subscript ``s`` denotes the scattered field.
The RCS is

.. math::

    \sigma(\theta,\phi,f)
    = 4\pi
      \frac{|F_{\mathrm{s},\theta}|^2+|F_{\mathrm{s},\phi}|^2}
      {|E_{\mathrm{inc},x}|^2+|E_{\mathrm{inc},y}|^2
       +|E_{\mathrm{inc},z}|^2}.

The incident spectrum is not inferred from the nominal waveform amplitude.
gprMax samples the actual numerically propagated field of the associated
discrete plane wave and transforms it using the same frequencies and time
window as the NTFF surface data. Plane-wave start and stop times are therefore
included automatically. Frequencies at which the incident spectrum is zero
produce ``NaN``; results where it is very small can be poorly conditioned and
should not be used.

``rcs`` is stored as a real, linear quantity in square metres, not in dBsm. It
can be converted using

.. math::

    \sigma_\mathrm{dBsm}
    = 10\log_{10}\!\left(\frac{\sigma}{1\,\mathrm{m}^2}\right).

The requested :math:`(\theta,\phi)` specifies the observation direction. For
monostatic RCS it must be opposite to the incident propagation direction. For
example, a plane wave propagating along ``+x`` is observed in backscatter at
``theta=90`` and ``phi=180`` degrees. The far-field normalization and
engineering phase convention are also written as HDF5 attributes.

.. code-block:: none

    #ksir_far_field: 90 180 antenna_band backscatter Etheta Ephi rcs

#ksir_far_field_array:
----------------------

Requests the Cartesian product of inclusive theta and phi ranges:

.. code-block:: none

    #ksir_far_field_array: theta1 theta2 dtheta phi1 phi2 dphi transform_id [output_id [output1 output2 ...]]

Each range must contain an integer number of positive increments. For example,
the following requests a five-degree full-sphere pattern:

.. code-block:: none

    #ksir_far_field_array: 0 180 5 0 360 5 antenna_band pattern Etheta Ephi radiation_intensity

#ntff_far_field: and #ntff_far_field_array:
------------------------------------------------

Request conventional equivalent-current frequency-domain far fields:

.. code-block:: none

    #ntff_far_field: theta phi transform_id [output_id [output1 output2 ...]]
    #ntff_far_field_array: theta1 theta2 dtheta phi1 phi2 dphi transform_id [output_id [output1 output2 ...]]

``transform_id`` must refer to ``#ntff_frequency``. Angles, default field
components, range and increment rules, range normalization, derived radiation
quantities, RCS, and antenna-port normalization are identical to the
corresponding ``#ksir_far_field`` commands. Because the underlying surface
integral is independent, requesting both formulations on the same
``#ntff_surface`` provides a useful numerical cross-check.

.. code-block:: none

    #ntff_far_field_array: 0 180 5 0 360 5 current_band current_pattern Etheta Ephi directivity_dbi

Symmetry-completed surface example
----------------------------------

A symmetry plane can coincide with an integration-surface face. No closure
option is entered on the KSIR command:

.. code-block:: none

    #symmetry_boundary: x0 pmc
    #ntff_surface: 0 0.034 0.034 0.026 0.066 0.066 half_surface
    #ksir_time_rx: 0.04 0.05 0.051 half_surface half_fields Ez

The physical ``x0`` face is not sampled. The other five faces and their
reflected images form the completed closed surface. Observation points must be
outside this completed physical-plus-image surface, not merely outside the
simulated half. This KSIR workflow is supported by the local CPU, CUDA,
OpenCL, and Metal solvers for nondispersive models. Equivalent-current
transforms do not yet support symmetry image completion; physical faces can
instead be omitted explicitly for an open frequency-domain Huygens surface.
OpenCL has end-to-end qualification on the development server. Metal has
source-generation and dispatch coverage but still requires qualification on
suitable Apple hardware.


PML commands
============

The default behaviour for the absorbing boundary conditions (ABC) is first order Complex Frequency Shifted (CFS) Perfectly Matched Layers (PML), with thicknesses of 10 cells on each of the six sides of the model domain.

#pml_cells:
------------

Allows you to control the number of cells (thickness) of PML that are used on the six sides of the model domain. The PML is defined within the model domain, i.e. it is not added to the domain size. The syntax of the command is:

.. code-block:: none

    #pml_cells: i1 [i2 i3 i4 i5 i6]

* ``i1`` is the number of cells of PML to use on all sides of the model domain (can be set to zero to completely switch off the PML), or ``i1`` is the number of cells of PML to use on the side of the model domain nearest the origin of the x-axis (x0).
* ``i2`` is the number of cells of PML to use on the side of the model domain nearest the origin of the y-axis (y0).
* ``i3`` is the number of cells of PML to use on the side of the model domain nearest the origin of the z-axis (z0).
* ``i4`` is the number of cells of PML to use on the side of the model domain furthest from the origin of the x-axis (xmax).
* ``i5`` is the number of cells of PML to use on the side of the model domain furthest from the origin of the y-axis (ymax).
* ``i6`` is the number of cells of PML to use on the side of the model domain furthest from the origin of the z-axis (zmax).
* ``i1 i2 i3 i4 i5 i6`` may be set to zero to turn off the PML on a specific side of the model domain.

For example to use a PML with 20 cells (thicker than the default 10 cells) on only the z-axis sides of the domain use:

.. code-block:: none

    #pml_cells: 10 10 20 10 10 20

#pml_formulation:
-----------------

Allows you to alter the formulation used for the PML. The current options are to use the Higher Order RIPML (HORIPML) - https://doi.org/10.1109/TAP.2011.2180344, or Multipole RIPML (MRIPML) - https://doi.org/10.1109/TAP.2018.2823864. The syntax of the command is:

.. code-block:: none

    #pml_formulation: str1 [str2]

* ``str1`` can be either ``HORIPML`` or ``MRIPML``.
* ``str2`` is an optional reusable PML profile ID. Without an ID the
  formulation is applied globally. With an ID it is stored for use by a
  ``#pml_slab`` and does not change the domain-boundary PMLs.

For example to use the Multipole RIPML:

.. code-block:: none

    #pml_formulation: MRIPML

#pml_cfs:
---------

Allows you (advanced) control of the parameters that are used to build each order of the PML. Up to a second order PML can currently be specified, i.e. by using two ``#pml_cfs`` commands. The syntax of the command is:

.. code-block:: none

    #pml_cfs: str1 str2 f1 f2 str3 str4 f3 f4 str5 str6 f5 f6 [str7]

* ``str1`` is the type of scaling to use for the CFS :math:`\alpha` parameter. It can be ``constant``, ``linear``, ``quadratic``, ``cubic``, ``quartic``, ``quintic`` and ``sextic``.
* ``str2`` is the direction of the scaling to use for the CFS :math:`\alpha` parameter. It can be ``forward`` or ``reverse``.
* ``f1 f2`` are the minimum and maximum values for the CFS :math:`\alpha` parameter.
* ``str3`` is the type of scaling to use for the CFS :math:`\kappa` parameter. It can be ``constant``, ``linear``, ``quadratic``, ``cubic``, ``quartic``, ``quintic`` and ``sextic``.
* ``str4`` is the direction of the scaling to use for the CFS :math:`\kappa` parameter. It can be ``forward`` or ``reverse``.
* ``f3 f4`` are the minimum and maximum values for the CFS :math:`\kappa` parameter. The minimum value for the CFS :math:`\kappa` parameter is one.
* ``str5`` is the type of scaling to use for the CFS :math:`\sigma` parameter. It can be ``constant``, ``linear``, ``quadratic``, ``cubic``, ``quartic``, ``quintic`` and ``sextic``.
* ``str6`` is the direction of the scaling to use for the CFS :math:`\sigma` parameter. It can be ``forward`` or ``reverse``.
* ``f5 f6`` are the minimum and maximum values for the CFS :math:`\sigma` parameter.
* ``str7`` is an optional reusable PML profile ID. It must match the ID on a
  named ``#pml_formulation`` and may then be selected by ``#pml_slab``.

The CFS values (which are internally specified) used for the default standard first order PML are: ``#pml_cfs: constant forward 0 0 constant forward 1 1 quartic forward 0 None``. Specifying 'None' for the maximum value of :math:`\sigma` forces gprMax to calculate it internally based on the relative permittivity and permeability of the underlying materials in the model.

The parameters will be applied to all slabs of the PML that are switched on.
When a profile ID is supplied, they are applied only to slabs that select that
profile. A profile can contain one or two CFS terms. A named formulation with
no named ``#pml_cfs`` uses the default first-order CFS parameters.

.. tip::

    ``forward`` direction implies minimum parameter value at the inner boundary of the PML and maximum parameter value at the edge of computational domain, ``reverse`` is the opposite.

#pml_slab:
----------

Places an experimental, one-axis RIPML correction in an axis-aligned region of
the main 3D grid. This can form a local matched load inside a PEC guiding
structure, or replace one domain-boundary PML with an independently configured
profile. The syntax is:

.. code-block:: none

    #pml_slab: f1 f2 f3 f4 f5 f6 str1 [str2 [str3]]

* ``f1 f2 f3`` and ``f4 f5 f6`` are the lower and upper corners of the slab.
* ``str1`` is the maximum-stretch face: ``x0``, ``y0``, ``z0``, ``xmax``,
  ``ymax``, or ``zmax``. The opposite face is the zero-stretch entrance.
* ``str2`` is an optional profile ID defined by ``#pml_formulation`` and,
  optionally, ``#pml_cfs``. If omitted, the global PML configuration is used.
* ``str3`` optionally controls automatic PEC enclosure generation: ``y``
  (default) or ``n``. To specify ``str3`` without a profile, use ``None`` for
  ``str2``.

For example, this replaces a disabled ``x0`` boundary PML with a 12-cell slab
using a locally defined MRIPML recipe:

.. code-block:: none

    #pml_cells: 0 10 10 10 10 10
    #pml_formulation: MRIPML port_load
    #pml_slab: 0 0 0 0.012 0.080 0.060 x0 port_load

For an internal absorber, gprMax normally creates five PEC plates after all
user geometry has been processed: four transverse walls and a backing plate on
the maximum-stretch face. Plates coincident with model-domain faces are not
needed and are omitted. The zero-stretch entrance remains open. Existing
geometry on an automatically enclosed face is therefore made PEC.

Advanced experiments may disable this behaviour while retaining the global
PML profile:

.. code-block:: none

    #pml_slab: 0.010 0.020 0.020 0.022 0.060 0.050 x0 None n

With automatic plates disabled, gprMax checks the final Yee edges and warns
about every exposed transverse or maximum-stretch face, but permits the model
to run for research and custom-enclosure experiments. Incomplete enclosures
have no stability guarantee: long-duration testing has demonstrated
exponential late-time growth for exposed transverse and open maximum-stretch
terminations. The material cross-section must remain constant along the
absorption direction; inconsistently filled slabs, same-axis slab overlaps,
and overlaps with a native PML remain errors because they invalidate the
current formulation.

An automatically generated internal identifier is reported in the log. The
hash command defines a slab on the main 3D grid. With the Python API a slab may
instead be added to an HSG subgrid, provided the complete slab lies inside its
working region and does not overlap the HSG coupling or auxiliary-PML regions.
Subgrid slabs use the CPU solver and the subgrid's finer spatial and temporal
discretisation. Domain-decomposed MPI CPU models are also supported. A slab may
cross any number of rank boundaries. Its CFS grading is evaluated over the
complete global thickness and sliced consistently between ranks, while each
rank allocates only the PML history arrays for its local part. Consequently,
the timestep loop needs no slab-specific communication in addition to the
normal electric- and magnetic-field halo exchanges. Automatic PEC enclosure,
material-extrusion validation, custom profiles, and boundary-replacement slabs
have the same behaviour as in serial models. Orthogonal PML slabs may overlap,
as domain PMLs do at edges and corners.


#symmetry_boundary:
--------------------

Sets a PEC or PMC symmetry boundary on one face of the model domain, replacing the PML on that face. The command may be used more than once to set different faces. The syntax is:

.. code-block:: none

    #symmetry_boundary: str1 str2

* ``str1`` is ``x0``, ``y0``, ``z0``, ``xmax``, ``ymax``, or ``zmax``.
* ``str2`` is ``pec`` or ``pmc``.

For example:

.. code-block:: none

    #symmetry_boundary: x0 pec
    #symmetry_boundary: ymax pmc

.. note::

    * The PML thickness on a symmetry face is set to zero automatically.
    * PEC and PMC boundaries, including PMC boundaries in models containing
      dispersive materials, are supported by the CPU, CUDA, OpenCL, Metal, and
      domain-decomposed MPI CPU solvers.
    * In MPI models, each face is constructed and updated only by ranks that
      touch that global domain face. Domain-edge corrections are likewise
      restricted to ranks touching both physical faces; internal rank seams
      are not treated as symmetry edges.
    * Symmetry boundaries are not currently supported in 2D mode or on a
      subgrid. They may be used on the main grid of a model that contains
      subgrids.
