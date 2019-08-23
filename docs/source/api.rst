.. _commands:

*******************
API
*******************

gprMax can be also be run using its API in additional to input file commands. For instance,

.. code-block:: python

  import gprMax

  # Make simulation objects

  #title: GSSI 400MHz 'like' antenna in free-space
  #domain: 0.380 0.380 0.360
  #dx_dy_dz: 0.001 0.001 0.001
  #time_window: 12e-9

  # equivalent to 'title: API example'
  title = gprMax.Title(name='API example')
  # equivalent to 'dx_dy_dz: 1e-3 1e-3 1e-3'
  dxdydz = gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3))
  # equivalent to 'time_window: 6e-9'
  tw = gprMax.TimeWindow(time=6e-9)
  # equivalent to 'domain: 0.15 0.15 0.15'
  domain = gprMax.Domain(p1=(0.15, 0.15, 0.15))

  # equivalent to #waveform: ricker 1 1.5e9 myricker
  waveform = gprMax.Waveform(wave_type='ricker', amp=1, freq=1.5e9, id='my_ricker')
  # equivalent to 'hertzian_dipole: y 0.045 0.075 0.085 my_ricker'
  dipole = gprMax.HertzianDipole(p1=(0.045, 0.075, 0.085), polarisation='y', waveform_id='my_ricker')
  # equivalent to 'rx: 0.045, 0.075 + 10e-3, 0.085'
  rx = gprMax.Rx(p1=(0.045, 0.075 + 10e-3, 0.085))

  # make a container for the simulation
  scene = gprMax.Scene()
  # add the objects to the container
  scene.add(dxdydz)
  scene.add(tw)
  scene.add(domain)
  scene.add(title)
  scene.add(waveform)
  scene.add(dipole)

  # run the simulation
  gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile='mysimulation')


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
Object Construction
===================
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

PML
===
PML Cells
--------------------------
.. autoclass:: gprMax.cmds_single_use.PMLCells
