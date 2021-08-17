.. _commands:

*********************************
Random Parameter Generation Mode
*********************************

This new gprMax feature allows the user to generate random parameters for a specific model. Syntactically, instead of entering single values for each **numerical** parameter in a given hash command, the user is allowed to enter values in pairs. The following convention is followed to activate the random parameter generation mode:

.. code-block:: none

    #command_name: distr parameter_1.1 parameter_1.2 parameter_2.1 parameter_2.2 parameter_3.1 parameter_3.2 ...

``distr`` specifies the Probability Distribution Function (PDF) from which random numbers are drawn. Currently, the following distributions are supported and corresponding to each of them, the values entered in pairs specify the distribution parameters:

* ``u`` - **Uniform Distribution**

  ``parameter_x.1`` = Lower bound, ``parameter_x.2`` = Upper bound

* ``n`` - **Normal Distribution**

  ``parameter_x.1`` = Mean (:math:`\mu`), ``parameter_x.2`` = Standard deviation (:math:`\sigma>0`)

* ``ln`` - **Log-Normal Distribution**

  ``parameter_x.1`` = Mean (:math:`\mu`), ``parameter_x.2`` = Standard deviation (:math:`\sigma>0`)

* ``lg`` - **Logistic Distribution**

  ``parameter_x.1`` = Mean (:math:`\mu`), ``parameter_x.2`` = Scale (s>0)

* ``lp`` - **Laplace (double exponential) Distribution**

  ``parameter_x.1`` = Mean (:math:`\mu`), ``parameter_x.2`` = Exponential decay factor (:math:`\lambda`)

* ``b`` - **Beta Distribution**

  ``parameter_x.1`` = :math:`\alpha` (>0), ``parameter_x.2`` = :math:`\beta` (>0)


**Note**

* This mode is built on the ``numpy.random`` module. For more information on probability distributions and the associated parameters, check their `documentation <https://numpy.org/doc/1.16/reference/routines.random.html>`_ 

* In case the generated random parameter exceeds the model domain bounds, it is automatically constrained to fit inside the domain, which ensures that the execution is not stopped midway.

* In case the upper coordinate for a certain geometry object is smaller than the lower coordinate, it is automatically incremented (by the floating point precision) to just exceed the lower coordinate.

* The random generation mode can only be used for certain geometry and multi-use hash commands. The following single-use hash commands are not compatible with this feature: 
  
  ``#title``, ``#output_dir``, ``#cpu_threads``, ``#dx_dy_dz``, ``#domain``, ``#time_step_stability_factor``, ``#time_window``, ``#pml_cells``, ``#src_steps``, ``#rx_steps``, ``#excitation_file``, ``#pml_cells``, ``#pml_formulation``, ``#pml_cfs``

* In case you want a subset of the parameters to remain constant **inside the same hash command** (and vary only the remaining ones), you would have to enter those parameters twice.

  For example: If you would like to randomly vary only the relative permittivity (:math:`\epsilon_r`) inside a ``#material`` command, you would have to enter the following: 

  .. code-block:: none

      #material: u 2 5 0.01 0.01 1 1 0 0 my_sand

  This creates a material called ``my_sand`` which has a relative permittivity :math:`\epsilon_r` drawn from a Uniform Distribution (``u``) within the range ``[2,5]``, a conductivity of :math:`\sigma = 0.01` S/m, and is non-magnetic, i.e. :math:`\mu_r = 1` and :math:`\sigma_* = 0`

* String literals are supposed to be entered **only once**. Only numerical parameters can be entered in pairs. All other conventions while entering a hash command remain the same. For example, to use a randomly positioned x-polarised Hertzian dipole with a Ricker waveform whose amplitude and frequency are drawn from a uniform distribution, use: 
  
  .. code-block:: none

    #waveform: u ricker 1 3 500e6 750e6 my_ricker_pulse
    #hertzian_dipole: u x 0.05 0.10 0.05 0.10 0.05 0.10 my_ricker_pulse


Saving Randomly Generated Parameters
====================================

All the randomly generated parameters are **automatically** saved to a pickle file in the same directory as the input file. Each column in this file correspoinds to a specific model parameter.

This feature can easily be used along with the ``-n`` command line argument. For every iteration, a new set of random parameters would be generated and a new row would be appended to the pickle file. 

We also introduced a new command line flag: ``--no-h5``, which instructs gprMax to skip saving the output .h5 files.

.. code-block:: none

    (gprMax)$ python -m gprMax path_to_folder/name_of_input_file.in -n 5000 --no-h5

For every command line execution, the following attributes are saved:

* All the randomly generated parameters are saved to - ``path_to_folder/name_of_input_file_{rand_params}.pkl``
* All redundant features are removed from the file generated above and the compressed file is saved to - ``path_to_folder/name_of_input_file_{rand_params}_{compressed}.pkl``. This might be useful for using the dataset for subsequent purposes (such as Machine Learning)
* All A-scans for each receiver in the model are saved to - ``path_to_folder/name_of_input_file_{field_outputs}.pkl``

After the simulation is complete, the data labels corresponding to the random parameters are displayed on the terminal (in the same order as they are saved in the pickle file)

For more information on reading and extracting data from the output pickle files, check `this Jupyter Notebook <https://github.com/utsav-akhaury/gprMax/blob/devel/ML/ML.ipynb>`_


.. _materials:

Material commands
=================

gprMax has two builtin materials which can be used by specifying the identifiers ``pec`` and ``free_space``. These simulate a perfect electric conductor and air, i.e. a non-magnetic material with :math:`\epsilon_r = 1`, :math:`\sigma = 0`, respectively. Additionally the identifiers ``grass`` and ``water`` are currently reserved for internal use and should not be used unless you intentionally want to change their properties.

#material:
----------

Allows you to introduce a material into the model described by a set of constitutive parameters. The syntax of the command is:

.. code-block:: none

    #material: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 str1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` specify the PDF parameters for choosing the relative permittivity, :math:`\epsilon_r`
* ``f2.2 f2.2`` specify the PDF parameters for choosing the conductivity (Siemens/metre), :math:`\sigma`
* ``f3.1 f3.2`` specify the PDF parameters for choosing the relative permeability, :math:`\mu_r`
* ``f4.1 f4.2`` specify the PDF parameters for choosing the magnetic loss (Ohms/metre), :math:`\sigma_*`
* ``str1`` is an identifier for the material.

For example ``#material: u 3 5 0.01 0.01 1 1 0 0 my_sand`` creates a material called ``my_sand`` which has a relative permittivity (frequency independent) of :math:`\epsilon_r` drawn from a Uniform Distribution (``u``) within the range ``[3,5]``, a conductivity of :math:`\sigma = 0.01` S/m, and is non-magnetic, i.e. :math:`\mu_r = 1` and :math:`\sigma_* = 0`


#add_dispersion_debye:
----------------------

Allows you to add dispersive properties to an already defined ``#material`` based on a multiple pole Debye formulation (see :ref:`capabilities` section). For example, the susceptability function for a single-pole Debye material is given by:

.. math::

    \chi_p (t) = \frac{\Delta \epsilon_{rp}}{\tau_p} e^{-t/\tau_p},

where :math:`\Delta \epsilon_{rp} = \epsilon_{rsp} - \epsilon_{r \infty}`, :math:`\epsilon_{rsp}` is the zero-frequency relative permittivity for the pole, :math:`\epsilon_{r \infty}` is the relative permittivity at infinite frequency, and :math:`\tau_p` is the pole relaxation time.

The syntax of the command is:

.. code-block:: none

    #add_dispersion_debye: distr i1 f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 ... str1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``i1`` is the number of Debye poles.
* ``f1.1 f1.2`` specify the PDF parameters for choosing the difference between the zero-frequency relative permittivity and the relative permittivity at infinite frequency, i.e. :math:`\Delta \epsilon_{rp1} = \epsilon_{rsp1} - \epsilon_{r \infty}` , for the first Debye pole.
* ``f2.1 f2.2`` specify the PDF parameters for choosing the relaxation time (seconds), :math:`\tau_{p1}`, for the first Debye pole.
* ``f3.1 f3.2`` specify the PDF parameters for choosing the difference between the zero-frequency relative permittivity and the relative permittivity at infinite frequency, i.e. :math:`\Delta \epsilon_{rp2} = \epsilon_{rsp2} - \epsilon_{r \infty}` , for the second Debye pole.
* ``f4.1 f4.2`` specify the PDF parameters for choosing the relaxation time (seconds), :math:`\tau_{p2}`, for the second Debye pole.
* ...
* ``str1`` identifies the material to add the dispersive properties to.

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

    #add_dispersion_lorentz: distr i1 f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 ... str1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``i1`` is the number of Lorentz poles.
* ``f1.1 f1.2`` specify the PDF parameters for choosing the difference between the zero-frequency relative permittivity and the relative permittivity at infinite frequency, i.e. :math:`\Delta \epsilon_{rp1} = \epsilon_{rsp1} - \epsilon_{r \infty}` , for the first Lorentz pole.
* ``f2.1 f2.2`` specify the PDF parameters for choosing the frequency (Hertz), :math:`\omega_{p1}`, for the first Lorentz pole.
* ``f3.1 f3.2`` specify the PDF parameters for choosing the damping coefficient (Hertz), :math:`\delta_{p1}`, for the first Lorentz pole.
* ``f4.1 f4.2`` specify the PDF parameters for choosing the difference between the zero-frequency relative permittivity and the relative permittivity at infinite frequency, i.e. :math:`\Delta \epsilon_{rp2} = \epsilon_{rsp2} - \epsilon_{r \infty}` , for the second Lorentz pole.
* ``f5.1 f5.2`` specify the PDF parameters for choosing the frequency (Hertz), :math:`\omega_{p2}`, for the second Lorentz pole.
* ``f6.1 f6.2`` specify the PDF parameters for choosing the damping coefficient (Hertz), :math:`\delta_{p2}`, for the second Lorentz pole.
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

    #add_dispersion_drude: distr i1 f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 ... str1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``i1`` is the number of Drude poles.
* ``f1.1 f1.2`` specify the PDF parameters for choosing the frequency (Hertz), :math:`\omega_{p1}`, for the first Drude pole.
* ``f2.1 f2.2`` specify the PDF parameters for choosing the inverse of the relaxation time (Hertz), :math:`\gamma_{p1}`, for the first Drude pole.
* ``f3.1 f3.2`` specify the PDF parameters for choosing the frequency (Hertz), :math:`\omega_{p2}`, for the second Drude pole.
* ``f4.1 f4.2`` specify the PDF parameters for choosing the inverse of the relaxation time (Hertz), :math:`\gamma_{p2}` for the second Drude pole.
* ...
* ``str1`` identifies the material to add the dispersive properties to.

.. note::

    * You can continue to add pairs of values for :math:`\omega_p` and :math:`\gamma_p` for as many Drude poles as you have specified with ``i1``.
    * Temporal values associated with pole frequencies and relaxation times should always be greater than the time step :math:`\Delta t` used in the model.


#soil_peplinski:
----------------

Allows you to use a mixing model for soils proposed by Peplinski (http://dx.doi.org/10.1109/36.387598), valid for frequencies in the range 0.3GHz to 1.3GHz. The command is designed to be used in conjunction with the ``#fractal_box`` command for creating soils with realistic dielectric and geometric properties. The syntax of the command is:

.. code-block:: none

    #soil_peplinski: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 str1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` specify the PDF parameters for choosing the sand fraction of the soil.
* ``f2.1 f2.2`` specify the PDF parameters for choosing the clay fraction of the soil.
* ``f3.1 f3.2`` specify the PDF parameters for choosing the bulk density of the soil in grams per centimetre cubed.
* ``f4.1 f4.2`` specify the PDF parameters for choosing the density of the sand particles in the soil in grams per centimetre cubed.
* ``f5.1 f5.2`` and ``f6.1 f6.2``specify the PDF parameters for defining a range for the volumetric water fraction of the soil.
* ``str1`` is an identifier for the soil.

For example for a soil with sand fraction 0.5, clay fraction 0.5, bulk density :math:`2~g/cm^3`, sand particle density of :math:`2.66~g/cm^3`, and a volumetric water fraction range of 0.001 - 0.25 use: ``#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.25 my_soil``.

.. note::

    Further information on the Peplinski soil model and our implementation can be found in 'Giannakis, I. (2016). Realistic numerical modelling of Ground Penetrating Radar for landmine detection. The University of Edinburgh. (http://hdl.handle.net/1842/20449)'


Object construction commands
============================

Object construction commands are processed in the order they appear in the input file. Therefore space in the model allocated to a specific material using for example the ``#box`` command can be reallocated to another material using the same or any other object construction command. Space in the model can be regarded as a canvas in which objects are introduced and one can be overlaid on top of the other overwriting its properties in order to produce the desired geometry. The object construction commands can therefore be used to create complex shapes and configurations.

.. _geometryview:

#geometry_view:
---------------

Allows you output to file(s) information about the geometry of model. The file(s) use the open source `Visualization ToolKit (VTK) <http://www.vtk.org>`_ format which can be viewed in many free readers, such as `Paraview <http://www.paraview.org>`_. The command can be used to create several 3D views of the model which are useful for checking that it has been constructed as desired. The syntax of the command is:

.. code-block:: none

    #geometry_view: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 f8.1 f8.2 f9.1 f9.2 file1 c1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of the volume of the geometry view in metres respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of the volume of the geometry view in metres respectively.
* ``f7.1 f7.2`` and ``f8.1 f8.2`` and ``f9.1 f9.2`` specify the PDF parameters for choosing the spatial discretisation of the geometry view in metres respectively. Typically these will be the same as the spatial discretisation of the model but they can be courser if desired.
* ``file1`` is the filename of the file where the geometry view will be stored in the same directory as the input file.
* ``c1`` can be either n (normal) or f (fine) which specifies whether to output the geometry information on a per-cell basis (n) or a per-cell-edge basis (f). The fine mode should be reserved for viewing detailed parts of the geometry that occupy small volumes, as using this mode can generate geometry files with large file sizes.

.. tip::

    When you want to just check the geometry of your model, run gprMax using the optional command line argument ``--geometry-only``. This will build the model and produce any geometry view files, but will not run the simulation.


#edge:
------

Allows you to introduce a wire with specific properties into the model. A wire is an edge of a Yee cell and it can be useful to model resistors or thin wires. The syntax of the command is:

.. code-block:: none

    #edge: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 str1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the starting (x,y,z) coordinates of the edge respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the ending (x,y,z) coordinates of the edge respectively. The coordinates should define a single line.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials ``pec`` or ``free_space``.

For example to specify a x-directed wire of random length that is a perfect electric conductor, use: ``#edge: u 0.4 0.6 0.5 0.5 0.5 0.5 0.7 0.9 0.5 0.5 0.5 0.5 pec``. Note that the y and z coordinates are identical.

#plate:
-------

Allows you to introduce a plate with specific properties into the model. A plate is a surface of a Yee cell and it can be useful to model objects thinner than a Yee cell. The syntax of the command is:

.. code-block:: none

    #plate: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 str1

* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of the plate respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of the plate respectively. The coordinates should define a surface and not a 3D object like the ``#box`` command.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials ``pec`` or ``free_space``.

For example to specify a xy oriented plate of random surface area that is a perfect electric conductor, use: ``#plate: u 0.4 0.6 0.4 0.6 0.5 0.5 0.7 0.8 0.8 0.9 0.5 0.5 pec``. Note that the z coordinates are identical.

#triangle:
----------

Allows you to introduce a triangular patch or a triangular prism with specific properties into the model. The patch is just a triangular surface made as a collection of staircased Yee cells, and the triangular prism extends the triangular patch in the direction perpendicular to the plane. The syntax of the command is:

.. code-block:: none

    #triangle: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 f8.1 f8.2 f9.1 f9.2 f10.1 f10.2 str1 [c1]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the first apex of the triangle respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the second apex respectively.
* ``f7.1 f7.2`` and ``f8.1 f8.2`` and ``f9.1 f9.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the third apex respectively.
* ``f10.1 f10.2`` specify the PDF parameters for choosing the thickness of the triangular prism. If the thickness is zero then a triangular patch is created.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials ``pec`` or ``free_space``.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing. For use only when creating a triangular prism, not a triangular patch.

For example, to specify a xy orientated triangular patch that is a perfect electric conductor, use: ``#triangle: u 0.4 0.6 0.4 0.6 0.5 0.5 0.4 0.6 0.4 0.6 0.5 0.5 0.7 0.8 0.9 1.0 0.5 0.5 0.0 0.0 pec``. Note that the z coordinates are identical and the thickness is zero.

#box:
-----

Allows you to introduce an orthogonal parallelepiped with specific properties into the model. The syntax of the command is:

.. code-block:: none

    #box: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 str1 [c1]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of the parallelepiped respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of the parallelepiped respectively.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials ``pec`` or ``free_space``.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

#sphere:
--------

Allows you to introduce a spherical object with specific parameters into the model. The syntax of the command is:

.. code-block:: none

    #sphere: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 str1 [c1]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the centre of the sphere respectively.
* ``f4.1 f4.2`` specify the PDF parameters for choosing its radius.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials ``pec`` or ``free_space``.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

For example, to specify a randomly centered sphere with random radius and with constitutive parameters of ``my_sand``, use: ``#sphere: u 0.4 0.6 0.4 0.6 0.4 0.6 0.1 0.4 my_sand``.

.. note::

    * Sphere objects are permitted to extend outwith the model domain if desired, however, only parts of object inside the domain will be created.

#cylinder:
----------

Allows you to introduce a circular cylinder into the model. The orientation of the cylinder axis can be arbitrary, i.e. it does not have align with one of the Cartesian axes of the model. The syntax of the command is:

.. code-block:: none

    #cylinder: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 str1 [c1]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the centre of one face of the cylinder repectively
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the centre of the other face repectively.
* ``f7.1 f7.2`` specify the PDF parameters for choosing the radius of the cylinder.
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials ``pec`` or ``free_space``.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

For example, to specify a cylinder with its axis in the y direction and that is a perfect electric conductor, use: ``#cylinder: u 0.5 0.5 0.1 0.3 0.5 0.5 0.5 0.5 0.6 0.8 0.5 0.5 0.1 0.4 pec``.

.. note::

    * Cylinder objects are permitted to extend outwith the model domain if desired, however, only parts of object inside the domain will be created.


#cylindrical_sector:
--------------------

Allows you to introduce a cylindrical sector (shaped like a slice of pie) into the model. The syntax of the command is:

.. code-block:: none

    #cylindrical_sector: distr n1 f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 str1 [c1]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``n1`` is the direction of the axis of the cylinder from which the sector is defined and can be ``x``, ``y``, or ``z``.
* ``f1.1 f1.2`` and ``f2.1 f2.2``specify the PDF parameters for choosing the coordinates of the centre of the cylindrical sector respectively.
* ``f3.1 f3.2`` and ``f4.1 f4.2``specify the PDF parameters for choosing the lower and higher coordinates of the axis of the cylinder from which the sector is defined (in effect they specify the thickness of the sector).
* ``f5.1 f5.2`` specify the PDF parameters for choosing the radius of the cylindrical sector.
* ``f6.1 f6.2`` specify the PDF parameters for choosing the starting angle (in degrees) for the cylindrical sector (with zero degrees defined on the positive first axis of the plane of the cylindrical sector).
* ``f7.1 f7.2`` specify the PDF parameters for choosing the angle (in degrees) swept by the cylindrical sector (the finishing angle of the sector is always anti-clockwise from the starting angle).
* ``str1`` is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials ``pec`` or ``free_space``.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing.

For example, to specify a cylindrical sector with its axis in the z direction and that is a perfect electric conductor, use: ``#cylindrical_sector: u z 0.30 0.35 0.20 0.25 0.400 0.500 0.550 0.600 0.25 0.30 330 350 60 80 pec``.

.. note::

    * Cylindrical sector objects are permitted to extend outwith the model domain if desired, however, only parts of object inside the domain will be created.

.. _fractals:

#fractal_box:
-------------

Allows you to introduce an orthogonal parallelepiped with fractal distributed properties which are related to a mixing model or normal material into the model. The syntax of the command is:

.. code-block:: none

    #fractal_box: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 f8.1 f8.2 f9.1 f9.2 f10.1 f10.2 i1 str1 str2 [i2] [c1]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of the parallelepiped respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of the parallelepiped respectively.
* ``f7.1 f7.2`` specify the PDF parameters for choosing the fractal dimension, which for an orthogonal parallelepiped, should take values between zero and three.
* ``f8.1 f8.2`` specify the PDF parameters for choosing the weights for the fractal in the x direction.
* ``f9.1 f9.2`` specify the PDF parameters for choosing the weights for the fractal in the y direction.
* ``f10.1 f10.2`` specify the PDF parameters for choosing the weights for the fractal in the z direction.
* ``i1`` is the number of materials to use for the fractal distribution (defined according to the associated mixing model). This should be set to one if using a normal material instead of a mixing model.
* ``str1`` is an identifier for the associated mixing model or material.
* ``str2`` is an identifier for the fractal box itself.
* ``i2`` is an optional parameter which controls the seeding of the random number generator used to create the fractals. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.
* ``c1`` is an optional parameter which can be ``y`` or ``n``, used to switch on and off dielectric smoothing. If ``c1`` is specified then a value for ``i2`` must also be present.

For example, to create an orthogonal parallelepiped with fractal distributed properties using a Peplinski mixing model for soil, with 50 different materials over a range of water volumetric fractions from 0.001 - 0.25, you should first define the mixing model using: ``#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.25 my_soil`` and then specify the fractal box using ``#fractal_box: u 0 0 0 0 0 0 0.1 0.1 0.1 0.1 0.1 0.1 1.5 3 1 2 1 2 1 2 50 my_soil my_fractal_box``.

#add_surface_roughness:
-----------------------

Allows you to add rough surfaces to a ``#fractal_box`` in the model. A fractal distribution is used for the profile of the rough surface. The syntax of the command is:

.. code-block:: none

    #add_surface_roughness: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 f8.1 f8.2 f9.1 f9.2 f10.1 f10.2 f11.1 f11.2 str1 [i1]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of a surface on a ``#fractal_box`` respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of a surface on a ``#fractal_box`` repectively. The coordinates must locate one of the six surfaces of a ``#fractal_box`` but do not have to extend over the entire surface.
* ``f7.1 f7.2`` specify the PDF parameters for choosing the fractal dimension, which for an orthogonal parallelepiped, should take values between zero and three.
* ``f8.1 f8.2`` specify the PDF parameters for choosing the weights for the fractal in first direction of the surface.
* ``f9.1 f9.2`` specify the PDF parameters for choosing the weights for the fractal in the second direction of the surface.
* ``f10.1 f10.2`` and ``f11.1 f11.2`` specify the PDF parameters for defining the lower and upper limits for a range over which the roughness can vary. These limits should be specified relative to the dimensions of the ``#fractal_box`` that the rough surface is being applied.
* ``str1`` is an identifier for the ``#fractal_box`` that the rough surface should be applied to.
* ``i1`` is an optional parameter which controls the seeding of the random number generator used to create the fractals. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.

Up to six ``#add_rough_surface commands`` can be given for any ``#fractal_box`` corresponding to the six surfaces.

For example, if a ``#fractal_box`` has been specified using: ``#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 50 my_soil my_fractal_box`` then to apply a rough surface that varys between 85 mm and 110 mm (i.e. valleys that are up to 15 mm deep and peaks that are up to 10 mm tall) to the surface that is in the positive z direction, use ``#add_surface_roughness: u 0 0 0 0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1.5 1.5 1 1 1 1 0.085 0.090 0.110 0.115 my_fractal_box``.

#add_surface_water:
-------------------

Allows you to add surface water to a ``#fractal_box`` in the model that has had a rough surface applied. The syntax of the command is:

.. code-block:: none

    #add_surface_water: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 str1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of a surface on a ``#fractal_box`` respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of a surface on a ``#fractal_box`` repectively. The coordinates must locate one of the six surfaces of a ``#fractal_box`` but do not have to extend over the entire surface.
* ``f7.1 f7.2`` specify the PDF parameters for choosing the depth of the water, which should be specified relative to the dimensions of the ``#fractal_box`` that the surface water is being applied.
* ``str1`` is an identifier for the ``#fractal_box`` that the surface water should be applied to.

For example, to add surface water with random depth between 5-10 mm to an existing ``#fractal_box`` that has been specified using ``#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 50 my_soil my_fractal_box`` and has had a rough surface applied using ``#add_surface_roughness: 0 0 0.1 0.1 0.1 0.1 1.5 1 1 0.085 0.110 my_fractal_box``, use ``#add_surface_water: u 0 0 0 0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.105 0.110 my_fractal_box``.

.. note::

    * The water is modelled using a single-pole Debye formulation with properties :math:`\epsilon_{rs} = 80.1`, :math:`\epsilon_{\infty} = 4.9`, and a relaxation time of :math:`\tau = 9.231 \times 10^{-12}` seconds (http://dx.doi.org/10.1109/TGRS.2006.873208). If you prefer, gprMax will use your own definition for water as long as it is named ``water``.

#add_grass:
-----------

Allows you to add grass with roots to a ``#fractal_box`` in the model. The blades of grass are randomly distributed over the specified surface area and a fractal distribution is used to vary the height of the blades of grass and depth of the grass roots. The syntax of the command is:

.. code-block:: none

    #add_grass: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 f8.1 f8.2 f9.1 f9.2 i1 str1 [i2]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of a surface on a ``#fractal_box`` respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of a surface on a ``#fractal_box`` respectively. The coordinates must locate one of three surfaces (in the positive axis direction) of a ``#fractal_box`` but do not have to extend over the entire surface.
* ``f7.1 f7.2`` specify the PDF parameters for choosing the fractal dimension, which for an orthogonal parallelepiped, should take values between zero and three.
* ``f8.1 f8.2`` and ``f9.1 f9.2`` specify the PDF parameters for defining the lower and upper limits for a range over which the height of the blades of grass can vary. These limits should be specified relative to the dimensions of the ``#fractal_box`` that the grass is being applied.
* ``i1`` is the number of blades of grass that should be applied to the surface area.
* ``str1`` is an identifier for the ``#fractal_box`` that the grass should be applied to.
* ``i2`` is an optional parameter which controls the seeding of the random number generator used to create the fractals. By default (if you don't specify this parameter) the random number generator will be seeded by trying to read data from ``/dev/urandom`` (or the Windows analogue) if available or from the clock otherwise.

For example, to apply 100 blades of grass that randomly vary in height to the entire surface in the positive z direction of a ``#fractal_box`` that had been specified using ``#fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 50 my_soil my_fractal_box``, use ``#add_grass: u 0 0 0 0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1.5 1.5 0.2 0.2 0.25 0.40 100 my_fractal_box``.

.. note::

    * The grass is modelled using a single-pole Debye formulation with properties :math:`\epsilon_{rs} = 18.5087`, :math:`\epsilon_{\infty} = 12.7174`, and a relaxation time of :math:`\tau = 1.0793 \times 10^{-11}` seconds (http://dx.doi.org/10.1007/BF00902994). If you prefer, gprMax will use your own definition for grass if you use a material named ``grass``. The geometry of the blades of grass are defined by the parametric equations: :math:`x = x_c +s_x {\left( \frac{t}{b_x} \right)}^2`, :math:`y = y_c +s_y {\left( \frac{t}{b_y} \right)}^2`, and :math:`z=t`, where :math:`s_x` and :math:`s_y` can be -1 or 1 which are randomly chosen, and where the constants :math:`b_x` and :math:`b_y` are random numbers based on a Gaussian distribution.

#geometry_objects_read:
-----------------------

Allows you to insert pre-defined geometry into a model. The geometry is specified using a 3D array of integer numbers stored in a HDF5 file. The integer numbers must correspond to the order of a list of ``#material`` commands specified in a text file. The syntax of the command is:

.. code-block:: none

    #geometry_objects_read: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 file1 file2

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates in the domain where the lower left corner of the geometry array should be placed.
* ``file1`` is the path to and filename of the HDF5 file that contains an integer array which defines the geometry.
* ``file2`` is the path to and filename of the text file that contains ``#material`` commands.

.. note::

    * The integer numbers in the HDF5 file must be stored as a NumPy array at the root named ``data`` with type ``np.int16``.
    * The integer numbers in the HDF5 file correspond to the order of material commands in the materials text file, i.e. if ``#sand: 3 0 1 0`` is the first material in the materials file, it will be associated with any integers that are zero in the HDF5 file.
    * You can use an integer of -1 in the HDF5 file to indicate not to build any material at that location, i.e. whatever material is already in the model at that location.
    * The spatial resolution of the geometry objects must match the spatial resolution defined in the model.
    * The spatial resolution must be specified as a root attribute of the HDF5 file with the name ``dx_dy_dz`` equal to a tuple of floats, e.g. (0.002, 0.002, 0.002)
    * If the geometry objects being imported were originally generated using gprMax, i.e. exported using #geometry_objects_write, then you can use dielectric smoothing as you like when generating the original geometry objects. However, if the geometry objects being imported were generated by an external method then dielectric smoothing will not take place.

#geometry_objects_write:
------------------------

Allows you to write geometry generated in a model to file. The file can be read back into gprMax using the ``#geometry_objects_read`` command. This allows complex geometry that can take some time to generate to be saved to file and more quickly imported into subsequent models. The geometry information is saved as a 3D array of integer numbers stored in a HDF5 file, and corresponding material information is stored in a text file. The integer numbers correspond to the order of a list of ``#material`` commands specified in the text file. The syntax of the command is:

.. code-block:: none

    #geometry_objects_write: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 file1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of the parallelepiped respectively.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of the parallelepiped respectively.
* ``file1`` is the basename for the files where geometry and material information will be stored.

.. note::

    * The structure of the HDF5 file is the same as that described for the ``#geometry_objects_read`` command.
    * Objects are stored using spatial resolution defined in the model.


Source and output commands
==========================

#waveform:
----------

Allows you to specify waveforms to use with sources in the model. The syntax of the command is:

.. code-block:: none

    #waveform: distr str1 f1.1 f1.2 f2.1 f2.2 str2

* ``distr`` specifies the PDF from which random numbers are drawn
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
* ``f1.1 f1.2`` specify the PDF parameters for choosing the scaling of the maximum amplitude of the waveform (for a ``#hertzian_dipole`` the units will be Amps, for a ``#voltage_source`` or ``#transmission_line`` the units will be Volts).
* ``f2.1 f2.2`` specify the PDF parameters for choosing the centre frequency of the waveform (Hertz). In the case of the Gaussian waveform it is related to the pulse width.
* ``str2`` is an identifier for the waveform used to assign it to a source.

For example, to specify the normalised first derivate of a Gaussian waveform with a random amplitude and centre frequency, use: ``#waveform: u gaussiandotnorm 1 2 1.0e9 1.5e9 my_gauss_pulse``.

.. note::

    * The functions used to create the waveforms can be found in the :ref:`tools section <waveforms>`.
    * ``gaussiandot``, ``gaussiandotnorm``, ``gaussiandotdot``, ``gaussiandotdotnorm``, ``ricker`` waveforms have their centre frequencies specified by the user, i.e. they are not derived to the 'base' ``gaussian``
    * ``gaussianprime`` and ``gaussiandoubleprime`` waveforms are the first derivative and second derivative of the 'base' ``gaussian`` waveform, i.e. the centre frequencies of the waveforms will rise for the first and second derivatives.


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

    #hertzian_dipole: distr c1 f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 str1 [f4.1 f4.2 f5.1 f5.2]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``c1`` is the polarisation of the source and can be ``x``, ``y``, or ``z``.
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the source in the model.
* ``f4.1 f4.2 f5.1 f5.2`` are optional parameters. ``f4.1 f4.2`` specify the PDF parameters for choosing the time delay in starting the source. ``f5.1 f5.2`` specify the PDF parameters for choosing the time to remove the source. If the time window is longer than the source removal time then the source will stop after the source removal time. If the source removal time is longer than the time window then the source will be active for the entire time window. If ``f4.1 f4.2 f5.1 f5.2`` are omitted the source will start at the beginning of time window and stop at the end of the time window.
* ``str1`` is the identifier of the waveform that should be used with the source.

For example, to use a randomly placed x-polarised Hertzian dipole with unit amplitude and a 600 MHz centre frequency Ricker waveform, use: ``#waveform: ricker 1 600e6 my_ricker_pulse`` and ``#hertzian_dipole: u x 0.05 0.05 0.05 0.08 0.08 0.08 my_ricker_pulse``.

.. note::

    * When a ``#hertzian_dipole`` is used in a 2D simulation it acts as a line source of current in the invariant (geometry) direction of the simulation.


#magnetic_dipole:
-----------------

This will simulate an infinitesimal magnetic dipole. This is often referred to as an additive or soft source. The syntax of the command is:

.. code-block:: none

    #magnetic_dipole: distr c1 f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 str1 [f4.1 f4.2 f5.1 f5.2]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``c1`` is the polarisation of the source and can be ``x``, ``y``, or ``z``.
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the source in the model.
* ``f4.1 f4.2 f5.1 f5.2`` are optional parameters. ``f4.1 f4.2`` specify the PDF parameters for choosing the time delay in starting the source. ``f5.1 f5.2`` specify the PDF parameters for choosing the time to remove the source. If the time window is longer than the source removal time then the source will stop after the source removal time. If the source removal time is longer than the time window then the source will be active for the entire time window. If ``f4.1 f4.2 f5.1 f5.2`` are omitted the source will start at the beginning of time window and stop at the end of the time window.
* ``str1`` is the identifier of the waveform that should be used with the source.

#voltage_source:
----------------

Allows you to introduce a voltage source at an electric field location. It can be a hard source if it's resistance is zero, i.e. the time variation of the specified electric field component is prescribed, or if it's resistance is non-zero it behaves as a resistive voltage source. It is useful for exciting antennas when the physical properties of the antenna are included in the model. The syntax of the command is:

.. code-block:: none

    #voltage_source: distr c1 f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 str1 [f5.1 f5.2 f6.1 f6.2]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``c1`` is the polarisation of the source and can be ``x``, ``y``, or ``z``.
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the source in the model.
* ``f4.1 f4.2`` specify the PDF parameters for choosing the internal resistance of the voltage source in Ohms. If it is set to zero, then the voltage source is a hard source. That means it prescribes the value of the electric field component. If the waveform becomes zero then the source is perfectly reflecting.
* ``f5.1 f5.2 f6.1 f6.2`` are optional parameters. ``f5.1 f5.2`` specify the PDF parameters for choosing the time delay in starting the source. ``f6.1 f6.2`` specify the PDF parameters for choosing the time to remove the source. If the time window is longer than the source removal time then the source will stop after the source removal time. If the source removal time is longer than the time window then the source will be active for the entire time window. If ``f4.1 f4.2 f5.1 f5.2`` are omitted the source will start at the beginning of time window and stop at the end of the time window.
* ``str1`` is the identifier of the waveform that should be used with the source.

For example, to specify a randomly placed y-directed voltage source with random internal resistance between 50 & 100 Ohms, an amplitude of five, and a 1.2 GHz centre frequency Gaussian waveform use: ``#waveform: gaussian 5 1.2e9 my_gauss_pulse`` and ``#voltage_source: u y 0.05 0.05 0.05 0.08 0.08 0.08 50 100 my_gauss_pulse``.

#transmission_line:
-------------------

Allows you to introduce a one-dimensional transmission line model [MAL1994]_ at an electric field location. The transmission line can have a specified resistance greater than zero and less than the impedance of free space (376.73 Ohms). It is useful for exciting antennas when the physical properties of the antenna are included in the model. The syntax of the command is:

.. code-block:: none

    #transmission_line: distr c1 f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 str1 [f5.1 f5.2 f6.1 f6.2]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``c1`` is the polarisation of the transmission line and can be ``x``, ``y``, or ``z``.
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the transmission line in the model.
* ``f4.1 f4.2`` specify the PDF parameters for choosing the characteristic resistance of the transmission line source in Ohms. It can be any value greater than zero and less than the impedance of free space (376.73 Ohms).
* ``f5.1 f5.2 f6.1 f6.2`` are optional parameters. ``f5.1 f5.2`` specify the PDF parameters for choosing the time delay in starting the source. ``f6.1 f6.2`` specify the PDF parameters for choosing the time to remove the source. If the time window is longer than the source removal time then the source will stop after the source removal time. If the source removal time is longer than the time window then the source will be active for the entire time window. If ``f4.1 f4.2 f5.1 f5.2`` are omitted the source will start at the beginning of time window and stop at the end of the time window.
* ``str1`` is the identifier of the waveform that should be used with the source.

Time histories of voltage and current values in the transmission line are saved to the output file. These are documented in the :ref:`output file section <output>`. These parameters are useful for calculating characteristics of an antenna such as the input impedance or S-parameters. gprMax includes a Python module (in the ``tools`` package) to help you view the input impedance and s11 parameter from an antenna model fed using a transmission line. Details of how to use this module is given in the :ref:`tools section <plotting>`.

For example, to specify a randomly placed z-directed transmission line source with a random resistance between 50 & 100 Ohms, an amplitude of five, and a 1.2 GHz centre frequency Gaussian waveform use: ``#waveform: gaussian 5 1.2e9 my_gauss_pulse`` and ``#transmission_line: u z 0.05 0.05 0.05 0.08 0.08 0.08 50 100 my_gauss_pulse``.

An example antenna model using a transmission line can be found in the :ref:`examples section <example-wire-dipole>`.

#rx:
----

Allows you to introduce output points into the model. These are locations where the values of the electric and magnetic field components over the number of iterations of the model will be saved to file. The syntax of the command is:

.. code-block:: none

    #rx: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 [str1 str2]

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the coordinates (x,y,z) of the receiver in the model.
* ``str1`` is the identifier of the receiver.
* ``str2`` is a list of outputs with this receiver. It can be any selection from ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``, ``Ix``, ``Iy``, or ``Iz``.

.. note::

    * When the optional parameters ``str1`` and ``str2`` are not given all the electric and magnetic field components will be output with the receiver point.

#rx_array:
----------

Provides a simple method of defining multiple output points in the model. The syntax of the command is:

.. code-block:: none

    #rx_array: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 f8.1 f8.2 f9.1 f9.2

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of the output line/rectangle/volume.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of the output line/rectangle/volume.
* ``f7.1 f7.2`` and ``f8.1 f8.2`` and ``f9.1 f9.2`` specify the PDF parameters for choosing the increments (x,y,z) which define the number of output points in each direction. They can be set to zero to prevent any output points in a particular direction. Otherwise, the minimum value of ``f7.1`` and ``f7.2`` is :math:`\Delta x`, the minimum value of ``f8.1`` and ``f8.2``is :math:`\Delta y`, and the minimum value of ``f9.1`` and ``f9.2`` is :math:`\Delta z`.

#snapshot:
----------

Allows you to obtain information about the electromagnetic fields within a volume of the model at a given time instant. The file(s) use the open source `Visualization ToolKit (VTK) <http://www.vtk.org>`_ format which can be viewed in many free readers, such as `Paraview <http://www.paraview.org>`_. The syntax of this command is:

.. code-block:: none

    #snapshot: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 f8.1 f8.2 f9.1 f9.2 f10 file1

or

.. code-block:: none

    #snapshot: distr f1.1 f1.2 f2.1 f2.2 f3.1 f3.2 f4.1 f4.2 f5.1 f5.2 f6.1 f6.2 f7.1 f7.2 f8.1 f8.2 f9.1 f9.2 i1 file1

* ``distr`` specifies the PDF from which random numbers are drawn
* ``f1.1 f1.2`` and ``f2.1 f2.2`` and ``f3.1 f3.2`` specify the PDF parameters for choosing the lower left (x,y,z) coordinates of the volume of the snapshot in metres.
* ``f4.1 f4.2`` and ``f5.1 f5.2`` and ``f6.1 f6.2`` specify the PDF parameters for choosing the upper right (x,y,z) coordinates of the volume of the snapshot in metres.
* ``f7.1 f7.2`` and ``f8.1 f8.2`` and ``f9.1 f9.2`` specify the PDF parameters for choosing the spatial discretisation of the snapshot in metres.
* ``f10`` or ``i1`` are the time in seconds (float) or the iteration number (integer) which denote the point in time at which the snapshot will be taken.
* ``file1`` is the name of the file where the snapshot will be stored. Snapshot files are automatically stored in a directory with the name of the input file appended with '_snaps'. For multiple model runs each model run will have its own directory, i.e. '_snaps1', 'snaps2' etc...

For example to save a random snapshot of the electromagnetic fields in the model at a simulated time of 3 nanoseconds use: ``#snapshot: 0 0 0 0 0 0 1 1 1 1.5 1.5 1.5 0.1 0.1 0.1 0.1 0.1 0.1 3e-9 snap1``