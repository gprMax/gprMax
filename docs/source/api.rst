.. _api:

*******************
API
*******************

Introduction
==================
In additional to input file command interface gprMax can also be run using its API. The usage of the API differs from the use of the Python blocks syntax as follows. In the API gprMax functionality is called directly from any Python file via the gprMax module. Using the Python blocks syntax the Python code is executed within an embedded interpreter. The API has the advantage that it can be included within any Python file and can be included within any Python script.

There are several advantages to using API. Firstly, users can take advantage of the Python language. For instance, the structural elements of Python can be utilised more easily. gprMax objects can be used directly within functions, classes, modules and packages. In this way collections of components can be defined, reused and modified. For example, multiple SMA type connectors can be imported as a module and combined with an antenna from another module.

The API also allows gprMax to interface with other Python libraries. For example, the API could be used to create a parametric antenna and the external library Scipy could then be used to optimise its parameters. Although, this is possible using Python blocks syntax, the script file can also be debugged.

The syntax of the API is generally more verbose than the input file command syntax. However, for input file commands where there are an undefined number of parameters, such as adding dispersive properties, the user may find the API more manageable.

Example
==================

The following example is used to give an introduction to the gprMax API. the example file is found in
``user_models/antenna_wire_dipole_fs.py``.

First, import the gprMax module.

.. code-block:: python

  import gprMax

Next, simulation objects for the simulation are created from the gprMax module. Each input file command is available as an object. Simulation objects are created by passing the object parameters as key=value option arguments. The following example shows the creation of simulation objects and also their equivalent input file command for clarity.

.. code-block:: python

  #title: Wire antenna - half-wavelength dipole in free-space
  title = gprMax.Title(name="Wire antenna - half-wavelength dipole in free-space")
  #domain: 0.050 0.050 0.200
  domain = gprMax.Domain(p1=(0.050, 0.050, 0.200))
  #dx_dy_dz: 0.001 0.001 0.001
  dxdydz = gprMax.Discretisation(p1=(0.001, 0.001, 0.001))
  #time_window: 60e-9
  time_window = gprMax.TimeWindow(time=10e-9)
  #waveform: gaussian 1 1e9 mypulse
  waveform = gprMax.Waveform(wave_type='gaussian', amp=1, freq=1e9, id='mypulse')
  #transmission_line: z 0.025 0.025 0.100 73 mypulse
  transmission_line = gprMax.TransmissionLine(polarisation='z',
                                              p1=(0.025, 0.025, 0.100),
                                              resistance=73,
                                              waveform_id='mypulse')
  ## 150mm length
  #edge: 0.025 0.025 0.025 0.025 0.025 0.175 pec
  e1 = gprMax.Edge(p1=(0.025, 0.025, 0.025),
                   p2=(0.025, 0.025, 0.175),
                   material_id='pec')

  ## 1mm gap at centre of dipole
  #edge: 0.025 0.025 0.100 0.025 0.025 0.101 free_space
  e2 = gprMax.Edge(p1=(0.025, 0.025, 0.100),
                   p2=(0.025, 0.025, 0.100),
                   material_id='free_space')

  #geometry_view: 0.020 0.020 0.020 0.030 0.030 0.180 0.001 0.001 0.001 antenna_wire_dipole_fs f
  gv = gprMax.GeometryView(p1=(0.020, 0.020, 0.020),
                           p2=(0.030, 0.030, 0.180),
                           dl=(0.001, 0.001, 0.001),
                           filename='antenna_wire_dipole_fs',
                           output_type='n')


Next a :class:`gprMax.scene.Scene` object is created. The scene is a container for all the objects required in a simulation. The objects are added to the scene as follows:

.. code-block:: python

  # Create a scene
  scene = gprMax.Scene()
  # Add the simulation objects to the scene
  scene.add(title)
  scene.add(domain)
  scene.add(dxdydz)
  scene.add(time_window)
  scene.add(waveform)
  scene.add(transmission_line)
  scene.add(e1)
  scene.add(e2)
  scene.add(gv)


Once the simulation objects have been added to the scene the simulation is run as follows:

.. code-block:: python

  # run the simulation
  gprMax.run(scenes=[scene], n=1, outputfile='mysimulation')

The run function arguments are similar to the flags in the CLI. The most notable difference is that a file path for the data output must be provided.

Multiple simulation can be specified by providing multiple scene objects to the run function. Each scene must contain the essential commands and each user object required for that particular model.

Reference
=========

The commands have been grouped into six categories:

* **Essential** - required to run any model, such as the domain size and spatial discretization
* **General** - provide further control over the model
* **Material** - used to introduce different materials into the model
* **Object construction** - used to build geometric shapes with different constitutive parameters
* **Source and output** - used to place source and output points in the model
* **PML** - provide advanced customisation and optimisation of the absorbing boundary conditions

Essential
==================
Most of the commands are optional but there are some essential commands which are necessary in order to construct any model. For example, none of the media and object commands are necessary to run a model. However, without specifying any objects in the model gprMax will simulate free space (air), which on its own, is not particularly useful for GPR modelling. If you have not specified a command which is essential in order to run a model, for example the size of the model, gprMax will terminate execution and issue an appropriate error message.

The essential commands are:

Domain
------
.. autoclass:: gprMax.cmds_single_use.Domain

Discretisation
--------------
.. autoclass:: gprMax.cmds_single_use.Discretisation

Time Window
-----------
.. autoclass:: gprMax.cmds_single_use.TimeWindow

General
=======

Messages
--------
.. autoclass:: gprMax.cmds_single_use.Messages

Title
-----
.. autoclass:: gprMax.cmds_single_use.Title

Number of Threads
-----------------
.. autoclass:: gprMax.cmds_single_use.NumThreads

Time Step Stability Factor
--------------------------
.. autoclass:: gprMax.cmds_single_use.TimeStepStabilityFactor

Output Directory
--------------------------
.. autoclass:: gprMax.cmds_single_use.OutputDir

Number of Model Runs
--------------------
.. autoclass:: gprMax.cmds_single_use.NumberOfModelRuns



Material
========

Material
--------
.. autoclass:: gprMax.cmds_multiple.Material

Debye Dispersion
----------------
.. autoclass:: gprMax.cmds_multiple.AddDebyeDispersion

Lorentz Dispersion
------------------
.. autoclass:: gprMax.cmds_multiple.AddLorentzDispersion

Drude Dispersion
----------------
.. autoclass:: gprMax.cmds_multiple.AddDrudeDispersion

Soil Peplinski
--------------
.. autoclass:: gprMax.cmds_multiple.SoilPeplinski


Object Construction
===================

Object construction commands are processed in the order they appear in the scene. Therefore space in the model allocated to a specific material using for example the :class:`gprMax.cmds_geometry.box.Box` command can be reallocated to another material using the same or any other object construction command. Space in the model can be regarded as a canvas in which objects are introduced and one can be overlaid on top of the other overwriting its properties in order to produce the desired geometry. The object construction commands can therefore be used to create complex shapes and configurations.

Box
---
.. autoclass:: gprMax.cmds_geometry.box.Box

Cylinder
--------
.. autoclass:: gprMax.cmds_geometry.cylinder.Cylinder

Cylindrical Sector
------------------
.. autoclass:: gprMax.cmds_geometry.cylindrical_sector.CylindricalSector

Edge
----
.. autoclass:: gprMax.cmds_geometry.edge.Edge

Plate
-----
.. autoclass:: gprMax.cmds_geometry.plate.Plate

Triangle
-----
.. autoclass:: gprMax.cmds_geometry.triangle.Triangle

Sphere
-----
.. autoclass:: gprMax.cmds_geometry.sphere.Sphere

Fractal Box
-----
.. autoclass:: gprMax.cmds_geometry.fractal_box.FractalBox

Add Grass
---------
.. autoclass:: gprMax.cmds_geometry.add_grass.AddGrass

Add Surface Roughness
---------------------
.. autoclass:: gprMax.cmds_geometry.add_surface_roughness.AddSurfaceRoughness

Add Surface Water
-----------------
.. autoclass:: gprMax.cmds_geometry.add_surface_water.AddSurfaceWater

Geometry View
-------------
.. autoclass:: gprMax.cmds_multiple.GeometryView

Geometry Objects Write
----------------------
.. autoclass:: gprMax.cmds_multiple.GeometryObjectsWrite

Source and Output
=================
Waveform
--------
.. autoclass:: gprMax.cmds_multiple.Waveform

Voltage Source
--------------
.. autoclass:: gprMax.cmds_multiple.VoltageSource

Hertzian Dipole Source
----------------------
.. autoclass:: gprMax.cmds_multiple.HertzianDipole

Magnetic Dipole Source
----------------------
.. autoclass:: gprMax.cmds_multiple.MagneticDipole

Transmission Line
-----------------
.. autoclass:: gprMax.cmds_multiple.TransmissionLine

Excitation File
---------------
.. autoclass:: gprMax.cmds_single_use.ExcitationFile

Rx
--
.. autoclass:: gprMax.cmds_multiple.Rx

Rx Array
--------
.. autoclass:: gprMax.cmds_multiple.RxArray

Source Steps
------------
.. autoclass:: gprMax.cmds_single_use.SrcSteps

Rx Steps
------------
.. autoclass:: gprMax.cmds_single_use.RxSteps

Snapshot
--------
.. autoclass:: gprMax.cmds_multiple.Snapshot

PML
===

The default behaviour for the absorbing boundary conditions (ABC) is first order Complex Frequency Shifted (CFS) Perfectly Matched Layers (PML), with thicknesses of 10 cells on each of the six sides of the model domain. This can be altered by using the following command

PML Cells
--------------------------
.. autoclass:: gprMax.cmds_single_use.PMLCells

PML CFS
--------------------------
.. autoclass:: gprMax.cmds_multiple.PMLCFS


Additional API objects
======================

Function to run the simulation
------------------------------
.. autofunction:: gprMax.gprMax.run
.. autoclass:: gprMax.scene.Scene
