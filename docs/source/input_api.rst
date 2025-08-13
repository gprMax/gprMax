.. _input-api:

************************************
Advanced Model Building (Python API)
************************************

Introduction
============

gprMax has a choice of two methods for building a model to simulate:

1. A **text-based (ASCII) input file**, which can be created with any text editor, and uses a series of gprMax commands which begin with the hash character (``#``). This method is recommended for beginners and those not familiar with Python, and is described in the :ref:`input-hash-cmds` section.
2. A **Python API**, which includes all the functionality of method 1 as well as several more advanced features. This method is recommended for those who prefer to use Python or need access to specific API-only advanced features, and is described in this section of the documentation.

The Python API in gprMax allows users to access to gprMax functions directly from Python through importing the gprMax module. There are several advantages to using the API:

* Users can take advantage of the Python language - for instance, the structural elements of Python can be utilised more easily.
* gprMax objects can be used directly within functions, classes, modules and packages. In this way collections of components can be defined, reused and modified. For example, complex targets can be imported from a separate module and combined with an antenna from another module.
* The API can interface with other Python libraries. For example, the API could be used to create a parametric antenna and the external library Scipy could then be used to optimise its parameters.

The syntax of the API is generally more verbose than the input file (hash) command syntax. However, for input file commands where there are an undefined number of parameters, such as adding dispersive properties, the user may find the API more manageable.

.. note::

    In prior versions of gprMax (<4) the input file could be scripted using Python inserted between two commands (`#python:` and `#end_python:`). This feature is now deprecated and will be removed entirely in later versions. Users are encouraged to move to the new Python API. Antenna models can still be inserted between `#python:` and `#end_python:` commands but will need to make a small change to their input file. An example of this is provided in `examples/antenna_like_GSSI_1500_fs.in`. Alternatively a switch to the Python API can be made using the example in `examples/antenna_like_GSSI_1500_fs.py`.

Example
=======

:download:`antenna_wire_dipole_fs.py <../../examples/antenna_wire_dipole_fs.py>`

This example is used to give an introduction to the gprMax Python API.

.. literalinclude:: ../../examples/antenna_wire_dipole_fs.py
    :language: python
    :linenos:

1. Import the gprMax module.
2. Objects for the model are created from the gprMax module by passing object parameters as key=value arguments. The example shows the creation of objects and also their equivalent input file (hash) command for clarity.
3. Create a :class:`gprMax.scene.Scene` object. The scene is a container for all the objects required in a simulation. Simulations with multiple models, e.g. A-scans, should have a separate scene for each model (A-scan). Each scene must contain the essential functions and objects required for that particular model.
4. Add objects are to the scene.
5. Run the simulation.

Unless otherwise specified, the SI system of units is used throughout gprMax:

* All parameters associated with simulated space (i.e. size of model, spatial increments, etc...) should be specified in **metres**.
* All parameters associated with time (i.e. total simulation time, time instants, etc...) should be specified in **seconds**.
* All parameters denoting frequency should be specified in **Hertz**.
* All parameters associated with spatial coordinates in the model should  be specified in **metres**. The origin of the coordinate system **(0,0)** is at the lower left corner of the model.

It is important to note that gprMax converts spatial and temporal parameters given in **metres** and **seconds** to integer values corresponding to **FDTD cell coordinates** and **iteration number** respectively. Therefore, rounding to the nearest integer number of the user defined values is performed.

The fundamental spatial and temporal discretization steps are denoted as :math:`\Delta x` , :math:`\Delta y`, :math:`\Delta z` and :math:`\Delta t` respectively.

The functions have been grouped into six categories:

* **Essential** - required to run any model, such as the domain size and spatial discretization
* **General** - provide further control over the model
* **Material** - used to introduce different materials into the model
* **Object construction** - used to build geometric shapes with different constitutive parameters
* **Source and output** - used to place source and output points in the model
* **PML** - provide advanced customisation and optimisation of the absorbing boundary conditions

Essential functions
===================
Most of the functions are optional but there are some essential functions which are necessary in order to construct any model. For example, none of the media and object functions are necessary to run a model. However, without specifying any objects in the model gprMax will simulate free space (air), which on its own, is not particularly useful for GPR modelling. If you have not specified a functions which is essential in order to run a model, for example the size of the model, gprMax will terminate execution and issue an appropriate error message.

The essential functions are:

Running model(s)
----------------
.. autofunction:: gprMax.run

Creating a model scene
----------------------
.. autoclass:: gprMax.Scene
    :members: add

Domain
------
.. autoclass:: gprMax.user_objects.cmds_singleuse.Domain

Discretisation
--------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.Discretisation

Time Window
-----------
.. autoclass:: gprMax.user_objects.cmds_singleuse.TimeWindow

General functions
=================

Title
-----
.. autoclass:: gprMax.user_objects.cmds_singleuse.Title

Number of OpenMP threads
------------------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.OMPThreads

Time Step Stability Factor
--------------------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.TimeStepStabilityFactor

Output Directory
----------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.OutputDir

Material functions
==================

Material
--------
.. autoclass:: gprMax.user_objects.cmds_multiuse.Material

Debye Dispersion
----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.AddDebyeDispersion

Lorentz Dispersion
------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.AddLorentzDispersion

Drude Dispersion
----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.AddDrudeDispersion

Soil Peplinski
--------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.SoilPeplinski


Object construction functions
=============================

Object construction commands are processed in the order they appear in the scene. Therefore space in the model allocated to a specific material using for example the :class:`gprMax.user_objects.cmds_geometry.box.Box` command can be reallocated to another material using the same or any other object construction command. Space in the model can be regarded as a canvas in which objects are introduced and one can be overlaid on top of the other overwriting its properties in order to produce the desired geometry. The object construction commands can therefore be used to create complex shapes and configurations.

Box
---
.. autoclass:: gprMax.user_objects.cmds_geometry.box.Box

Cone
----
.. autoclass:: gprMax.user_objects.cmds_geometry.cone.Cone

Cylinder
--------
.. autoclass:: gprMax.user_objects.cmds_geometry.cylinder.Cylinder

Cylindrical Sector
------------------
.. autoclass:: gprMax.user_objects.cmds_geometry.cylindrical_sector.CylindricalSector

Edge
----
.. autoclass:: gprMax.user_objects.cmds_geometry.edge.Edge

Ellipsoid
---------
.. autoclass:: gprMax.user_objects.cmds_geometry.ellipsoid.Ellipsoid

Plate
-----
.. autoclass:: gprMax.user_objects.cmds_geometry.plate.Plate

Sphere
------
.. autoclass:: gprMax.user_objects.cmds_geometry.sphere.Sphere

Triangle
--------
.. autoclass:: gprMax.user_objects.cmds_geometry.triangle.Triangle

Fractal Box
-----------
.. autoclass:: gprMax.user_objects.cmds_geometry.fractal_box.FractalBox

.. note::

    * Currently (2024) we are not aware of a formulation of Perfectly Matched Layer (PML) absorbing boundary that can specifically handle distributions of material properties (such as those created by fractals) throughout the thickness of the PML, i.e. this is a required area of research. Our PML formulations can work to an extent depending on your modelling scenario and requirements. You may need to increase the thickness of the PML and/or consider tuning the parameters of the PML (:ref:`pml-tuning`) to improve performance for your specific model.

Add Grass
---------
.. autoclass:: gprMax.user_objects.cmds_geometry.add_grass.AddGrass

Add Surface Roughness
---------------------
.. autoclass:: gprMax.user_objects.cmds_geometry.add_surface_roughness.AddSurfaceRoughness

Add Surface Water
-----------------
.. autoclass:: gprMax.user_objects.cmds_geometry.add_surface_water.AddSurfaceWater

Geometry View
-------------
.. autoclass:: gprMax.user_objects.cmds_output.GeometryView

Geometry Objects Read
----------------------
.. autoclass:: gprMax.user_objects.cmds_geometry.geometry_objects_read.GeometryObjectsRead

Geometry Objects Write
----------------------
.. autoclass:: gprMax.user_objects.cmds_output.GeometryObjectsWrite

Source and output functions
===========================

Waveform
--------
.. autoclass:: gprMax.user_objects.cmds_multiuse.Waveform

Voltage Source
--------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.VoltageSource

Hertzian Dipole Source
----------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.HertzianDipole

Magnetic Dipole Source
----------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.MagneticDipole

Transmission Line
-----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.TransmissionLine

Discrete Plane Wave
-------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.DiscretePlaneWave

Excitation File
---------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.ExcitationFile

Receiver
--------
.. autoclass:: gprMax.user_objects.cmds_multiuse.Rx

Receiver Array
--------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.RxArray

Source Steps
------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.SrcSteps

Receiver Steps
--------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.RxSteps

Snapshot
--------
.. autoclass:: gprMax.user_objects.cmds_output.Snapshot

Subgrid
-------
.. autoclass:: gprMax.SubGridHSG


.. _pml-tuning:

PML functions
=============

The default behaviour for the absorbing boundary conditions (ABC) is first order Complex Frequency Shifted (CFS) Perfectly Matched Layers (PML), with thicknesses of 10 cells on each of the six sides of the model domain. The PML can be customised using the following commands:

PML Formulation
---------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.PMLFormulation

PML Thickness
-------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.PMLThickness

PML Properties
--------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.PMLProps

PML CFS
-------
Allows you control of the specific parameters that are used to build each order of the PML. Up to a second order PML can currently be specified, i.e. by using two ``PMLCFS`` commands.

.. autoclass:: gprMax.user_objects.cmds_multiuse.PMLCFS

The CFS values (which are internally specified) used for the default standard first order PML are:
* ``alphascalingprofile = 'constant'``
* ``alphascalingdirection = 'forward'``
* ``alphamin = 0``
* ``alphamax = 0``
* ``kappascalingprofile = 'constant'``
* ``kappascalingdirection = 'forward'``
* ``kappamin = 1``
* ``kappamax = 1``
* ``sigmascalingprofile = 'quartic'``
* ``sigmascalingdirection = 'forward``
* ``sigmamin = 0``
* ``sigmamax = None``

.. note::

    * The parameters will be applied to all slabs of the PML that are switched on.
    * Using ``None`` for the maximum value of :math:`\sigma` forces gprMax to calculate it internally based on the relative permittivity and permeability of the underlying materials in the model.
    * ``forward`` direction implies a minimum parameter value at the inner boundary of the PML and maximum parameter value at the edge of the computational domain, ``reverse`` is the opposite.
