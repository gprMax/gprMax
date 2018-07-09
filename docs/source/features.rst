.. _capabilities:

*****************
Software Features
*****************

This section begins with an overview, primarily for previous users, about what is new, what has changed, and what has been retired in this version of gprMax. It is then followed by more general descriptions some of the key features of gprMax that are useful for GPR modelling as well as more general electromagnetic simulations.

What's new/changed?
===================

A brief summary of what each input command does is given. Please refer to the :ref:`commands` section for a detailed description of the syntax of each command.

The code has been completely re-written in Python/Cython. In the process a lot of changes have been made to improve efficiency, speed, and usability. Many new features have been implemented, which have been focussed on the following areas:

* Scripting in the input file
* Built-in library of antenna models
* Anisotropic material modelling
* Dispersive material modelling using multiple pole Debye, Lorentz or Drude formulations
* Building heterogeneous objects using fractal distributions
* Building objects with rough surfaces
* Modelling soils with realistic dielectric and geometric properties
* Improved PML (RIPML) performance

New commands
------------

* ``#python`` and ``#end_python`` are used to define blocks of the input file where Python code will be executed. This allows the user to use scripting directly in the input file.
* ``#material`` replaces ``#medium`` with a new syntax
* ``#add_dispersion_debye`` is used to add Debye dispersive properties to a ``#material``
* ``#add_dispersion_lorentz`` is used to add Lorentz dispersive properties to a ``#material``
* ``#add_dispersion_drude`` is used to add Drude dispersive properties to a ``#material``
* ``#soil_peplinski`` is a soil mixing model that can be used with ``#fractal_box`` to generate soil(s) with more realistic dielectric and geometric properties
* ``#cylindrical_sector`` is a new object building command
* ``#geometry_view`` replaces ``#geometry_file`` or ``#geometry_vtk`` and is used to create views of the geometry of the model in open source `Visualization Toolkit (VTK) <http://www.vtk.org>`_ format which can be viewed in many free readers, such as `Paraview <http://www.paraview.org>`_
* ``#fractal_box`` is used to create a volume with a fractal distribution of properties
* ``#add_surface_roughness`` is used to add a rough surface to a ``#fractal_box``
* ``#add_surface_water`` is used to add surface water to a ``#fractal_box`` that has a rough surface
* ``#add_grass`` is used to add grass to a ``#fractal_box``
* ``#waveform`` is used to specify a waveform shape, and works in conjunction with the changed source commands
* ``#magnetic_dipole`` is used to introduce a magnetic dipole source, i.e. current on a small loop
* ``#pml_cfs`` is an advanced command for adjusting the CFS parameters used in the new PML


Changed commands
----------------

* All object building commands support anisotropy via additional material identifiers, e.g. ``#box: 0.0 0.0 0.0 0.1 0.1 0.1 matX matY matZ``
* Dielectric smoothing can be turned on (the default setting) or off for any volumetric object building command by specifying a character ``y`` (on) or ``n`` (off) after the material identifier, e.g. ``#sphere: 0.5 0.5 0.5 0.25 sand n``
* ``#triangle`` can now create both triangular patches (2D) and triangular prisms (3D) via a thickness parameter
* ``#pml_cells`` replaces ``#pml_layers`` and can now be used to control the number of cells of PML on the six faces of the model domain. The number of cells can be set to zero on any of the faces to turn that PML off if desired. The default behaviour (if this command is not specified) is to use 10 cells.
* ``#hertzian_dipole`` and ``#voltage_source`` now specify polarisation, location, any additional parameters, and an identifier to link to a ``#waveform`` command
* ``#snapshot`` no longer requires a type, as only VTK snapshot files are now produced
* ``#num_of_procs`` is now called ``#num_threads``
* ``#tx_steps`` is now called ``#src_steps``


Retired commands
----------------

* ``#analysis`` and ``#end_analysis`` are no longer required as: sources and receivers can be specified anywhere is the input file; the output file automatically has the same name as the input file with a ``.out`` extension; the format of the output file is now `HDF5 <http://www.hdfgroup.org/HDF5/>`_; gprMax can be run with the syntax ``gprMax my_input_file -n number_of_runs``, where ``number_of_runs`` can be used to rerun the model for creating scans and/or moving geometry between runs.
* ``#tx`` is no longer required as the polarisation and position of a source is now specified in the source command, e.g. ``#hertzian_dipole: y 0.05 0.05 0.05 myPulse``
* ``#cylinder_new`` has become ``#cylinder``
* ``#cylindrical_segment`` was under-used and its effect can be created by cutting a ``#cylinder`` with a ``#box``
* ``#bowtie`` can be created using the new behaviour of ``#triangle``
* ``#number_of_media`` is not required with the new Python code
* ``#nips_number`` is not required with the new Python code
* ``#media_file`` was under-used
* ``#geometry_file`` has been replaced with the ``#geometry_view`` command
* ``#medium`` has been replaced with the ``#material`` command
* ``#abc_type``, ``#abc_order``, ``#abc_stability_factors``, ``#abc_optimization_angles``, and ``#abc_mixing_parameters`` are not required as the Higdon ABCs have been removed
* ``#huygens_surface`` will become part of the new ``#plane_wave`` command which is yet to be implemented


Commands yet to be implemented
------------------------------

There are commands from previous versions of gprMax that are planned for this version, but are yet to be implemented. These will be introduced in a future update. They are ``#thin_wire`` and ``#plane_wave``.


Migrating old input files
-------------------------

gprMax includes a Python module (in the ``tools`` package) to help you migrate old input files, written for previous versions of gprMax, to the syntax of the new commands. The module will do its best to convert the old file and write a new one, however, you should still carefully check the new file to make sure it is what you intended! Usage (from the top-level gprMax directory) is: ``python -m tools.inputfile_old2new my_old_inputfile.in``.


Key features
============

Python scriptable input files
-----------------------------

The input file has now been made scriptable by permitting blocks of Python code to be specified between ``#python`` and ``#end_python`` commands. The code is executed when the input file is read by gprMax. You don't need any external tools, such as MATLAB, to generate larger, more complex input files for building intricate models. Python scripting means that gprMax now includes :ref:`libraries of more complex objects, such as antennas <antennas>`, that can be easily inserted into a model. You can also access a number of built-in constants from your Python code. For further details see the :ref:`Python section <python-scripting>`.

Dispersive materials
--------------------

gprMax has always included the ability to represent dispersive materials using a single-pole Debye model. Many materials can be adequately represented using this approach for the typical frequency ranges associated with GPR. However, multi-pole Debye, Drude and Lorentz functions are often used to simulate the electric susceptibility of materials such as: water [PIE2009]_, human tissue [IRE2013]_, cold plasma [LI2013]_, gold [VIA2005]_, and soils [BER1998]_, [GIAK2012]_, [TEI1998]_. Electric susceptibility relates the polarization density to the electric field, and includes both the real and imaginary parts of the complex electric permittivity variation. In the new version of gprMax a recursive convolution based method is used to express dispersive properties as apparent current density sources [GIA2014]_. A major advantage of this implementation is that it creates an inclusive susceptibility function that holds, as special cases, Debye, Drude and Lorentz materials. For further details see the :ref:`material commands section <materials>`.

Realistic soils, heterogeneous objects and rough surfaces
---------------------------------------------------------

The inclusion of improved models of soils is important for many GPR simulations. gprMax can now be used to create soils with more realistic dielectric and geometrical properties. A semi-empirical model, initially suggested by [DOB1985]_, is used to describe the dielectric properties of the soil. The model relates relative permittivity of the soil to bulk density, sand particle density, sand fraction, clay fraction and water volumetric fraction. Using this approach, a more realistic soil with a stochastic distribution of the aforementioned parameters can be modelled. The real and imaginary parts of this semi-empirical model can be approximated using a multi-pole Debye function plus a conductive term. This can now be achieved in gprMax using the new dispersive material functionality. For further details see the :ref:`material commands section <materials>`.

Fractals are scale invariant functions which can express the topography of the earth for a wide range of scales with sufficient detail [TUR1987]_. For this reason fractals have been chosen to represent the topography of soils. Fractals can be generated by the convolution of Gaussian noise with an inverse Fourier transform of :math:`\frac{1}{kb}`, where :math:`k` is the wavenumber and :math:`b` is a constant related to the fractal dimension [TUR1997]_. gprMax can now generate heterogeneous volumes (boxes) with realistic soil properties that can have rough surfaces applied. For further details see the :ref:`fractal object building commands section <fractals>`.

Fractal correlated noise [TUR1997]_ is used to describe the stochastic distribution of the properties of soils. This approach has been chosen because it has been shown that soil-related environmental properties frequently obey fractal laws [BUR1981]_, [HILL1998]_. For further details see the :ref:`material commands section <materials>` and the :ref:`fractal object building commands section <fractals>`.

.. _antennas:

Library of antenna models
-------------------------

gprMax now includes Python modules with pre-defined models of antennas that behave similarly to commercial antennas [WAR2011]_ [STA2017]_. Currently models of antennas similar to `Geophysical Survey Systems, Inc. (GSSI) <http://www.geophysical.com>`_ 1.5 GHz (Model 5100) antenna, and 400 MHz antenna, as well as `MALA Geoscience <http://www.malags.com/>`_ 1.2 GHz antenna are included. By taking advantage of Python scripting in input files, using such complex structures in a model is straightforward without having to be built step-by-step by the user. For further details see the :ref:`Python section <python>`.

Anisotropic materials
---------------------

It is possible to specify objects that have diagonal anisotropy which allows materials such as wood and fibre-reinforced composites, often imaged with GPR, to be more accurately modelled. Standard isotropic objects specify one material identifier that defines the same properties in x, y, and z directions. However, every volumetric object building command can also be specified with three material identifiers, which allows properties for the x, y, and z directions to be separately defined.

Dielectric smoothing
--------------------

At the boundaries between different materials in a model there is the question of what electric and magnetic material properties to use?

* Should the last object to be defined at that location dictate the electric and magnetic properties?
* Should an average set of electric and magnetic properties of the materials of the objects that share that location be used?

This latter option is often referred to as dielectric smoothing and has been shown to result in more accurate simulations [LUE1994]_ [BOU1996]_ [WHI2009]_. To address this question gprMax includes an option to turn dielectric smoothing on or off for volumetric object building commands. The default behaviour (if no option is specified) is for dielectric smoothing to be on. The option can be specified with a single character ``y`` (on) or ``n`` (off) given after the material identifier in each object command. When dielectric smoothing is on gprMax calculates the arithmetic mean of the electric and magnetic properties of the surrounding Yee cells, to use for the single Yee cell edge (boundary) of interest.

Perfectly Matched Layer (PML) boundary conditions
-------------------------------------------------

With increased research into quantitative information from GPR, it has become necessary for models to be able to have more efficient and better-performing Perfectly Matched Layer (PML) absorbing boundary conditions. Since 2005 gprMax has featured PML absorbing boundary conditions based on the uniaxial PML (UPML) [GED1998]_ formulation. A PML based on a recursive integration approach to the complex frequency shifted (CFS) PML [GIA2012]_ has been adopted in the new version of gprMax. A general formulation of this RIPML, which can be used to develop any order of PML, has been used to implement first and second order CFS stretching functions. One of the attractions of the RIPML is that it is easily applied as a correction to the field quantities after the complete FDTD grid has been updated using the standard FDTD update equations. gprMax now offers the ability (for advanced users) to customise the parameters of the PML which allows its performance to be better optimised for specific applications. Additionally, since the RIPML is media agnostic it can be used without change to problems involving dispersive and anisotropic materials. For further details see the :ref:`PML commands section <pml-commands>`.

Open source, robust, file formats
---------------------------------

Alongside improvements to the input file there is a new output file format – `HDF5 <http://www.hdfgroup.org/HDF5/>`_ – to manage the larger and more complex data sets that are being generated. HDF5 is a robust, portable and extensible format with a number of free readers available. For further details see the :ref:`output file section <output>`.

In addition, the `Visualization Toolkit (VTK) <http://www.vtk.org>`_ is being used for improved handling and viewing of the detailed 3D FDTD geometry meshes. The VTK is an open-source system for 3D computer graphics, image processing and visualisation. It also has a number of free readers available including `Paraview <http://www.paraview.org>`_. For further details see the :ref:`geometry view command <geometryview>`.
