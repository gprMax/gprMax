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

    In prior versions of gprMax (<4) the input file could be scripted using Python inserted between two commands (`#python:` and `#end_python:`). This feature is now deprecated and will be removed entirely in later versions. Users are encouraged to move to the new Python API. Antenna models can still be inserted between `#python:` and `#end_python:` commands but will need to make a small change to their input file. An example of this is provided in `examples/gpr/antennas/gssi_1500/antenna_like_GSSI_1500_fs.in`. Alternatively a switch to the Python API can be made using the adjacent `examples/gpr/antennas/gssi_1500/antenna_like_GSSI_1500_fs.py` example.

Example
=======

:download:`antenna_wire_dipole_fs.py <../../examples/antennas/wire_dipole/antenna_wire_dipole_fs.py>`

The equivalent hash-command model is
:download:`antenna_wire_dipole_fs.in <../../examples/antennas/wire_dipole/antenna_wire_dipole_fs.in>`.

This example is used to give an introduction to the gprMax Python API.

.. literalinclude:: ../../examples/antennas/wire_dipole/antenna_wire_dipole_fs.py
    :language: python
    :linenos:

1. Import the gprMax module.
2. Objects for the model are created from the gprMax module by passing object parameters as key=value arguments. The adjacent ``.in`` model shows their equivalent positional hash commands.
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

Domain Mode
-----------
.. autoclass:: gprMax.user_objects.cmds_singleuse.DomainMode

For an explicit 2D model, set one component of ``Domain.p1`` to
``float('inf')`` and add ``DomainMode('TM')`` or ``DomainMode('TE')`` before
the domain is built. The infinite value identifies the invariant axis; it is
resolved internally to the one-cell TM or two-cell TE Yee-grid thickness.

.. code-block:: python

    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.DomainMode(mode='TE'))
    scene.add(gprMax.Domain(p1=(0.24, 0.21, float('inf'))))

Discretisation
--------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.Discretisation

Time Window
-----------
.. autoclass:: gprMax.user_objects.cmds_singleuse.TimeWindow

A minimal three-dimensional scene contains the three essential model objects:

.. code-block:: python

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.30, 0.20, 0.15)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=12e-9))

    gprMax.run(scenes=[scene], n=1, outputfile='minimal_model')

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

Magnetic Averaging
------------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.MagneticAveraging

Typical general settings are added directly to the scene:

.. code-block:: python

    scene.add(gprMax.Title(name='buried_target'))
    scene.add(gprMax.OMPThreads(n=8))
    scene.add(gprMax.TimeStepStabilityFactor(f=0.95))
    scene.add(gprMax.OutputDir(dir='results'))
    scene.add(gprMax.MagneticAveraging(mode='harmonic'))

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

The dispersion objects modify materials that have already been added to the
same scene. Each material ID below is therefore unique:

.. code-block:: python

    scene.add(gprMax.Material(
        er=6, se=0.01, mr=1, sm=0, id='half_space'
    ))

    scene.add(gprMax.Material(
        er=4, se=0, mr=1, sm=0, id='debye_medium'
    ))
    scene.add(gprMax.AddDebyeDispersion(
        poles=1, er_delta=(2.0,), tau=(1e-10,),
        material_ids=('debye_medium',),
    ))

    scene.add(gprMax.Material(
        er=3, se=0, mr=1, sm=0, id='lorentz_medium'
    ))
    scene.add(gprMax.AddLorentzDispersion(
        poles=1, er_delta=(2.0,), omega=(2e10,), delta=(5e9,),
        material_ids=('lorentz_medium',),
    ))

    scene.add(gprMax.Material(
        er=1, se=0, mr=1, sm=0, id='drude_medium'
    ))
    scene.add(gprMax.AddDrudeDispersion(
        poles=1, omega=(2e10,), alpha=(5e9,),
        material_ids=('drude_medium',),
    ))

The Peplinski object is a mixing model for a :class:`FractalBox`, rather than
a homogeneous material:

.. code-block:: python

    scene.add(gprMax.SoilPeplinski(
        sand_fraction=0.5, clay_fraction=0.2,
        bulk_density=2.0, sand_density=2.66,
        water_fraction_lower=0.05, water_fraction_upper=0.25,
        id='soil_mix',
    ))


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

Thin Wire
---------
.. autoclass:: gprMax.user_objects.cmds_geometry.thin_wire.ThinWire

.. code-block:: python

    scene.add(gprMax.ThinWire(
        p1=(0.10, 0.10, 0.02),
        p2=(0.10, 0.10, 0.12),
        radius=0.0001,
    ))

Magnetic Edge
-------------
.. autoclass:: gprMax.user_objects.cmds_geometry.magnetic_edge.MagneticEdge

.. code-block:: python

    scene.add(gprMax.MagneticEdge(
        p1=(0.5, 0.5, 0.5),
        p2=(0.7, 0.5, 0.5),
        material_id='pmc',
    ))

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

The following compact examples show the required geometry keywords. The
referenced material IDs must already exist; ``pec`` and ``pmc`` are built in.

.. code-block:: python

    scene.add(gprMax.Box(
        p1=(0, 0, 0), p2=(0.30, 0.20, 0.08), material_id='half_space'
    ))
    scene.add(gprMax.Cone(
        p1=(0.05, 0.05, 0.08), p2=(0.05, 0.05, 0.13),
        r1=0.02, r2=0, material_id='pec',
    ))
    scene.add(gprMax.Cylinder(
        p1=(0.15, 0.02, 0.05), p2=(0.15, 0.18, 0.05),
        r=0.01, material_id='pec',
    ))
    scene.add(gprMax.CylindricalSector(
        normal='z', ctr1=0.15, ctr2=0.10,
        extent1=0.04, extent2=0.08, r=0.04,
        start=0, end=90, material_id='half_space',
    ))
    scene.add(gprMax.Edge(
        p1=(0.10, 0.10, 0.02), p2=(0.10, 0.10, 0.12),
        material_id='pec',
    ))
    scene.add(gprMax.ThinWire(
        p1=(0.12, 0.10, 0.02), p2=(0.12, 0.10, 0.12),
        radius=0.0001,
    ))
    scene.add(gprMax.Ellipsoid(
        p1=(0.20, 0.10, 0.08), xr=0.03, yr=0.02, zr=0.01,
        material_id='half_space',
    ))
    scene.add(gprMax.Plate(
        p1=(0.04, 0.04, 0.06), p2=(0.10, 0.10, 0.06),
        material_id='pec',
    ))
    scene.add(gprMax.Sphere(
        p1=(0.24, 0.10, 0.08), r=0.015, material_id='pec'
    ))
    scene.add(gprMax.Triangle(
        p1=(0.04, 0.04, 0.04), p2=(0.10, 0.04, 0.04),
        p3=(0.04, 0.10, 0.04), thickness=0.01,
        material_id='half_space',
    ))

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

A fractal volume can use either a normal material or a mixing model such as
``soil_mix`` from the material example above. Surface modifiers refer to the
fractal box by its ID and must be added after it:

.. code-block:: python

    scene.add(gprMax.FractalBox(
        p1=(0, 0, 0), p2=(0.30, 0.20, 0.10),
        frac_dim=1.5, weighting=(1, 1, 1), n_materials=20,
        mixing_model_id='soil_mix', id='ground', seed=1,
    ))
    scene.add(gprMax.AddSurfaceRoughness(
        p1=(0, 0, 0.10), p2=(0.30, 0.20, 0.10),
        frac_dim=1.5, weighting=(1, 1), limits=(0.08, 0.12),
        fractal_box_id='ground', seed=1,
    ))
    scene.add(gprMax.AddSurfaceWater(
        p1=(0, 0, 0.10), p2=(0.30, 0.20, 0.10),
        depth=0.105, fractal_box_id='ground',
    ))
    scene.add(gprMax.AddGrass(
        p1=(0, 0, 0.10), p2=(0.30, 0.20, 0.10),
        frac_dim=1.5, limits=(0.01, 0.03), n_blades=100,
        fractal_box_id='ground', seed=1,
    ))

Geometry View
-------------
.. autoclass:: gprMax.user_objects.cmds_output.GeometryView

Geometry Objects Read
----------------------
.. autoclass:: gprMax.user_objects.cmds_geometry.geometry_objects_read.GeometryObjectsRead

Geometry Objects Write
----------------------
.. autoclass:: gprMax.user_objects.cmds_output.GeometryObjectsWrite

Geometry views are visualisations. Geometry-object files instead preserve
material-index geometry for insertion into another model:

.. code-block:: python

    scene.add(gprMax.GeometryView(
        p1=(0, 0, 0), p2=(0.30, 0.20, 0.15),
        dl=(0.002, 0.002, 0.002),
        filename='model_geometry', output_type='n',
    ))
    scene.add(gprMax.GeometryObjectsWrite(
        p1=(0, 0, 0), p2=(0.30, 0.20, 0.15),
        filename='reusable_geometry',
    ))

In a different scene, the saved geometry can be inserted at a chosen origin:

.. code-block:: python

    scene.add(gprMax.GeometryObjectsRead(
        p1=(0.05, 0.05, 0),
        geofile='reusable_geometry.h5',
        matfile='reusable_geometry_materials.txt',
    ))

Source and output functions
===========================

Waveform
--------
.. autoclass:: gprMax.user_objects.cmds_multiuse.Waveform

The constructor has three forms. Arguments should be supplied by keyword:

.. code-block:: python

    # Built-in analytic waveform
    gprMax.Waveform(wave_type='ricker', amp=1, freq=1e9, id='pulse')

    # User-defined Python function
    gprMax.Waveform(wave_type='user', user_func=function, id='pulse')

    # User-defined sample arrays
    gprMax.Waveform(
        wave_type='user', user_values=values, user_time=times,
        kind='linear', fill_value=0, id='pulse'
    )

The callable keyword is ``user_func`` (not ``usr_func``). It provides a bespoke waveform shape as an alternative to the ``user_values``/``user_time`` sample arrays and is available only through the Python API because a function object cannot be represented in a text input file. The callable must accept one scalar time in seconds and return the complete numeric amplitude at that time; ``amp`` and ``freq`` are therefore not required for ``wave_type='user'``.

For example, the following creates a callable waveform and assigns it to a Hertzian dipole:

.. code-block:: python

    import numpy as np
    import gprMax

    def my_waveform(time):
        return np.sin(2 * np.pi * 1e9 * time) * np.exp(-time / 1e-9)

    scene.add(gprMax.Waveform(wave_type='user', user_func=my_waveform, id='mywave'))
    scene.add(gprMax.HertzianDipole(
        p1=(0.05, 0.05, 0.05), polarisation='z', waveform_id='mywave'
    ))

gprMax first calls the function at :math:`t=0` to check its signature and return type. It is then sampled at the whole and/or half time steps required by each source while the source arrays are prepared. It is never called from the FDTD time-stepping loop, although a computationally expensive function can increase model setup time. Imports used by the function must be available in the scope where the function is *defined*, following normal Python name resolution.

The callable can also be a closure, which is a convenient way to generate a family of related waveforms without duplicating code:

.. code-block:: python

    def make_waveform(freq, decay):
        def waveform(time):
            return np.sin(2 * np.pi * freq * time) * np.exp(-time / decay)
        return waveform

    for i, freq in enumerate([0.5e9, 1e9, 2e9, 4e9]):
        scene.add(gprMax.Waveform(
            wave_type='user', user_func=make_waveform(freq, decay=1e-9), id=f'wave_{i}'
        ))

Exactly one of ``user_func`` and ``user_values`` must be supplied. When
``user_values`` is used without ``user_time``, gprMax associates the samples
with its simulation time vector. ``kind`` and ``fill_value`` are passed to
``scipy.interpolate.interp1d`` and apply only to sampled waveforms. User-defined
waveforms can drive local Hertzian or magnetic dipoles, voltage sources,
transmission lines, and magnetic-frill sources. The discrete-plane-wave
formulation currently requires a built-in analytic waveform.

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

Every transmission-line source automatically writes its incident and terminal
voltage/current histories together with ``frequency``, ``S11``, ``Zin``, and
``Yin`` beneath ``/tls/tlN`` in the model HDF5 output. ``Zin`` is derived from
the voltage-wave S11 result; ``Zin_current`` is an independent, stagger-aware
current-wave check. An additional :class:`RxPort` is only needed for a
resistive voltage source, not for a transmission line. See
:ref:`Simulation Output <output>` for the equations and validity masks.

Magnetic Frill Source
---------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.MagneticFrillSource

A magnetic frill represents a sub-cell coaxial aperture through a PEC ground
plane. It must share an axial Yee edge with a ``ThinWire``; the source uses
that object's physical radius :math:`a` in Hyun's feed-cell equation. The
``zcoax`` argument is the characteristic impedance of the physical coax. For
a lossless TEM coax with outer-conductor inner radius :math:`b` and filler
properties :math:`\varepsilon_{r,c}` and :math:`\mu_{r,c}`,

.. math::

    Z_\mathrm{coax}
    = \frac{\eta_0}{2\pi}
      \sqrt{\frac{\mu_{r,c}}{\varepsilon_{r,c}}}
      \ln\!\left(\frac{b}{a}\right).

For the usual nonmagnetic filler,

.. math::

    Z_\mathrm{coax} \simeq
    \frac{60}{\sqrt{\varepsilon_{r,c}}}
    \ln\!\left(\frac{b}{a}\right)\ \Omega,
    \qquad
    b = a\exp\!\left(
        \frac{Z_\mathrm{coax}\sqrt{\varepsilon_{r,c}}}{60}
    \right).

The filler permittivity is the value inside the coax, which need not equal the
antenna-side material above the ground plane. gprMax obtains :math:`a` from
``ThinWire``; :math:`b` is not an input and the user must confirm that the
resulting aperture remains sub-cell. For example:

.. code-block:: python

    scene.add(gprMax.Waveform(
        wave_type='ricker', amp=1, freq=1e9, id='pulse'
    ))
    scene.add(gprMax.Plate(
        p1=(0, 0, 0.02), p2=(0.10, 0.10, 0.02), material_id='pec'
    ))
    scene.add(gprMax.ThinWire(
        p1=(0.05, 0.05, 0.02), p2=(0.05, 0.05, 0.08), radius=0.0001
    ))
    scene.add(gprMax.MagneticFrillSource(
        p1=(0.05, 0.05, 0.02), polarisation='z', zcoax=50,
        waveform_id='pulse',
    ))

The corrected formulation is supported by the CPU, CUDA, OpenCL, and Metal
solvers. It is also supported inside a CPU ``SubGridHSG``. Add the waveform,
PEC ground plane, thin wire, and magnetic frill to the same subgrid object,
using the same global-coordinate convention as other subgrid sources when
``autotranslate=True``:

.. code-block:: python

    subgrid.add(gprMax.Waveform(
        wave_type='ricker', amp=1, freq=5e9, id='fine_feed_wave'
    ))
    subgrid.add(gprMax.Plate(
        p1=(0.08, 0.06, 0.05), p2=(0.10, 0.08, 0.05),
        material_id='pec',
    ))
    subgrid.add(gprMax.ThinWire(
        p1=(0.09, 0.07, 0.05), p2=(0.09, 0.07, 0.06),
        radius=0.0001,
    ))
    subgrid.add(gprMax.MagneticFrillSource(
        p1=(0.09, 0.07, 0.05), polarisation='z', zcoax=50,
        waveform_id='fine_feed_wave',
    ))

The complete frill stencil and attached wire should remain within the
subgrid's working region; objects traversing its outer surface produce the
usual advanced-use warning, while thin-wire or frill placement in its PML is
rejected. Symmetry boundaries are not supported on a subgrid, so subgrid
frills cannot use symmetry-plane completion. The source
writes its time-domain terminal histories and derived ``S11``, ``Zin``, and
``Yin`` automatically beneath ``/frills/frillN`` on the main grid or
``/subgrids/<subgrid ID>/frills/frillN`` on a subgrid. See
:ref:`Simulation Output <output>`.

All local sources refer to the ID of a waveform that has already been added to
the scene. The following illustrates their required arguments; a model would
normally contain only the source or sources that it needs:

.. code-block:: python

    scene.add(gprMax.Waveform(
        wave_type='ricker', amp=1, freq=1e9, id='pulse'
    ))
    scene.add(gprMax.VoltageSource(
        p1=(0.04, 0.05, 0.05), polarisation='z', resistance=50,
        waveform_id='pulse',
    ))
    scene.add(gprMax.HertzianDipole(
        p1=(0.05, 0.05, 0.05), polarisation='z', waveform_id='pulse'
    ))
    scene.add(gprMax.MagneticDipole(
        p1=(0.06, 0.05, 0.05), polarisation='y', waveform_id='pulse'
    ))
    scene.add(gprMax.TransmissionLine(
        p1=(0.07, 0.05, 0.05), polarisation='z', resistance=50,
        waveform_id='pulse',
    ))

Plane Wave Angles
-------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.DiscretePlaneWaveAngles

Plane Wave Vector
-------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.DiscretePlaneWaveVector

Plane Wave Axial
-------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.DiscretePlaneWaveAxial

The angle, propagation-vector, and axial classes are alternative ways of
describing a total-field/scattered-field plane wave. The two points define the
total-field box. For example, choose one of:

.. code-block:: python

    scene.add(gprMax.DiscretePlaneWaveAngles(
        theta=90, phi=0, psi=90,
        p1=(0.02, 0.02, 0.02), p2=(0.08, 0.08, 0.08),
        waveform_id='pulse',
    ))

    scene.add(gprMax.DiscretePlaneWaveVector(
        m_vec=(1, 0, 0), psi=90,
        p1=(0.02, 0.02, 0.02), p2=(0.08, 0.08, 0.08),
        waveform_id='pulse',
    ))

    scene.add(gprMax.DiscretePlaneWaveAxial(
        axis='x', psi=90,
        p1=(0.02, 0.02, 0.02), p2=(0.08, 0.08, 0.08),
        waveform_id='pulse',
    ))

Here ``pulse`` must identify a built-in analytic waveform. Discrete plane waves
use the CPU and CUDA solvers. Homogeneous angle/vector plane waves and layered
axial plane waves support non-dispersive materials and multi-pole Debye,
Lorentz, and Drude materials. Their auxiliary dispersive state uses the same
real or complex precision selected for the main grid. A discrete plane wave
must be added to the main scene, not to a subgrid. Its TFSF box may contain a
complete subgrid; where the two regions overlap, the box must strictly enclose
the subgrid's HSG outer coupling surface so that the TFSF correction stencil
remains on the main grid.

Excitation File
---------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.ExcitationFile

.. code-block:: python

    scene.add(gprMax.ExcitationFile(
        filepath='measured_waveforms.txt', kind='linear', fill_value=0
    ))

Receiver
--------
.. autoclass:: gprMax.user_objects.cmds_multiuse.Rx

Receiver Array
--------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.RxArray

Voltage-source S11 and input impedance
--------------------------------------
.. autoclass:: gprMax.user_objects.cmds_output.RxPort

``RxPort`` binds to one single-Yee-edge ``VoltageSource`` at the same
discretised coordinate. It creates the necessary field monitors automatically
and calculates corrected complex ``S11``, ``Zin``, and ``Yin`` after the
solve. A finite-resistance source uses its resistance as the reference
impedance:

.. code-block:: python

    port = gprMax.RxPort(
        p1=(0.050, 0.050, 0.020),
        id='feed',
        spectrum_limit=10,
    )
    scene.add(port)

The default ``spectrum_limit=10`` retains frequencies having at least ten
cells per shortest material wavelength. Because optional hash-command
arguments are positional, an ``id`` is required when the Python object uses a
non-default spectrum limit. A research run can explicitly request all native
non-negative FFT bins while retaining the normal validity metadata:

.. code-block:: python

    scene.add(gprMax.RxPort(
        p1=(0.050, 0.050, 0.020),
        id='feed_full',
        spectrum_limit='nyquist',
    ))

A hard voltage source can instead emulate an ideal MoM delta-gap excitation.
Set the source resistance to zero. Its travelling-wave reference impedance
defaults to 50 Ohms and can be changed on the source:

.. code-block:: python

    scene.add(gprMax.VoltageSource(
        p1=(0.050, 0.050, 0.020),
        polarisation='z',
        resistance=0,
        waveform_id='source_wave',
        reference_impedance=50,
    ))
    hard_port = gprMax.RxPort(
        p1=(0.050, 0.050, 0.020),
        id='ideal_feed',
    )
    scene.add(hard_port)

After ``gprMax.run`` completes, ``port.result`` provides the same numerical
arrays that are stored under ``/ports/feed`` in the model HDF5 file. For a
hard source, gprMax obtains terminal current from the surrounding magnetic-
field loop and accounts explicitly for the half-step phase difference from
the integer-time voltage during transformation.

An ``RxPort`` may also be placed inside a ``SubGridHSG``. Add the waveform,
voltage source, and port to the same subgrid object; the port is then sampled
using that subgrid's finer spatial and temporal discretisation. With
``autotranslate=True``, the port and source use the same global physical
coordinate:

.. code-block:: python

    subgrid.add(gprMax.Waveform(
        wave_type='ricker', amp=1, freq=5e9, id='feed_wave'
    ))
    subgrid.add(gprMax.VoltageSource(
        p1=(0.090, 0.070, 0.060),
        polarisation='z', resistance=50,
        waveform_id='feed_wave',
    ))
    fine_port = gprMax.RxPort(
        p1=(0.090, 0.070, 0.060), id='fine_feed'
    )
    subgrid.add(fine_port)

The result is stored at
``/subgrids/<subgrid ID>/ports/<port ID>``. The source and port must belong to
the same grid object so that their discretised coordinates, material edge,
``dl``, and ``dt`` are unambiguous.

Source Steps
------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.SrcSteps

Receiver Steps
--------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.RxSteps

Snapshot
--------
.. autoclass:: gprMax.user_objects.cmds_output.Snapshot

Receivers can request selected field components, while an array produces
regularly spaced receivers over a line, plane, or volume. Source and receiver
steps are applied between repeated model runs:

.. code-block:: python

    scene.add(gprMax.Rx(
        p1=(0.08, 0.05, 0.05), id='surface_rx', outputs=['Ez', 'Hy']
    ))
    scene.add(gprMax.RxArray(
        p1=(0.02, 0.10, 0.05), p2=(0.12, 0.10, 0.05),
        dl=(0.01, 0, 0),
    ))
    scene.add(gprMax.SrcSteps(p1=(0.002, 0, 0)))
    scene.add(gprMax.RxSteps(p1=(0.002, 0, 0)))
    scene.add(gprMax.Snapshot(
        p1=(0, 0, 0), p2=(0.15, 0.12, 0.10),
        dl=(0.002, 0.002, 0.002), time=2e-9,
        filename='fields_2ns', fileext='.h5', outputs=['Ez', 'Hy'],
    ))

Reusable NTFF integration surface
---------------------------------
.. autoclass:: gprMax.user_objects.cmds_output.NTFFSurface

The reusable Python interface has a one-to-one mapping to the supported NTFF
hash commands:

.. list-table:: Reusable NTFF interfaces
    :header-rows: 1
    :widths: 42 38

    * - Python class
      - Hash command
    * - ``NTFFSurface``
      - ``#ntff_surface``
    * - ``KSIRFrequencyTransform``
      - ``#ksir_frequency``
    * - ``NTFFFrequencyTransform``
      - ``#ntff_frequency``
    * - ``KSIRAntennaPorts``
      - ``#ksir_antenna_ports``
    * - ``NTFFAntennaPorts``
      - ``#ntff_antenna_ports``
    * - ``KSIRTimeRx``
      - ``#ksir_time_rx``
    * - ``KSIRTimeRxSpherical``
      - ``#ksir_time_rx_spherical``
    * - ``KSIRTimeRxArray``
      - ``#ksir_time_rx_array``
    * - ``KSIRFrequencyRx``
      - ``#ksir_frequency_rx``
    * - ``KSIRFrequencyRxSpherical``
      - ``#ksir_frequency_rx_spherical``
    * - ``KSIRFrequencyRxArray``
      - ``#ksir_frequency_rx_array``
    * - ``KSIRFarField``
      - ``#ksir_far_field``
    * - ``KSIRFarFieldArray``
      - ``#ksir_far_field_array``
    * - ``NTFFFarField``
      - ``#ntff_far_field``
    * - ``NTFFFarFieldArray``
      - ``#ntff_far_field_array``
    * - ``NTFFTimeFarField``
      - ``#ntff_time_far_field``
    * - ``NTFFTimeFarFieldArray``
      - ``#ntff_time_far_field_array``

The mapping covers the reusable operations and their normal options. The
Python API also exposes advanced keyword arguments that cannot be entered
positionally in a hash command: ``NTFFSurface.origin``, and
``save_surface_dft`` and ``plane_wave_index`` on either frequency-transform
class. Hash
commands use the default surface centre, save the surface DFT, and associate
an enclosed plane wave automatically.

``EigenmodeSource`` cannot be combined with any of the ``KSIR*`` classes.
Use ``NTFFFrequencyTransform``, ``NTFFFarField`` or
``NTFFFarFieldArray``, and ``NTFFAntennaPorts`` for an eigenmode-fed
antenna.

``NTFFSurface(omit_faces=('x0', 'xmax'))`` creates an open frequency-domain
Huygens surface. ``omit_faces`` accepts one to five distinct Cartesian face
names; at least one of the six faces must remain active. A feed crossing an
opening continues uniformly into its PML, with the impressed source plane
outside the Huygens volume. KSIR and transient equivalent-current outputs
reject omitted physical faces.

NTFF definitions are main-grid objects, but their notional closed integration
surface may enclose complete HSG subgrids. A surface must not touch or cut an
HSG outer coupling surface: overlapping regions require the NTFF surface to
strictly enclose that outer surface. Sources and scatterers inside the subgrid
then contribute through the normal HSG field exchange. A disjoint subgrid is
permitted. NTFF surfaces cannot be defined inside a subgrid.

The following example reuses one surface for an exact time-domain point and a
frequency-domain radiation pattern. Python keyword arguments replace the
positional optional parameters used by the equivalent hash commands.

.. code-block:: python

    scene.add(gprMax.NTFFSurface(
        p1=(0.03, 0.03, 0.03),
        p2=(0.07, 0.07, 0.07),
        id='radiation_surface',
    ))
    scene.add(gprMax.KSIRTimeRx(
        position=(0.12, 0.05, 0.05),
        surface_id='radiation_surface',
        id='transient',
        outputs=('Ez',),
        time_origin='first_arrival',
    ))
    scene.add(gprMax.KSIRFrequencyTransform(
        surface_id='radiation_surface',
        id='antenna_band',
        frequencies=(0.8e9, 1.0e9, 1.2e9),
        window='hann',
    ))
    scene.add(gprMax.KSIRFarFieldArray(
        theta_start=0,
        theta_stop=180,
        theta_step=5,
        phi_start=0,
        phi_stop=360,
        phi_step=5,
        transform_id='antenna_band',
        id='pattern',
        outputs=('Etheta', 'Ephi', 'radiation_intensity'),
    ))

For a completed time receiver, ``result.point_times(q)`` and
``result.point_field(output, q)`` return only the interval supported by every
surface patch. ``point_raw_times`` and ``point_raw_field`` deliberately expose
the additional partial retarded tail for research use. Check
``result.terminal_decay_ok[q]``; a false value means that the FDTD time window
should be increased.

KSIR frequency transform
------------------------
.. autoclass:: gprMax.user_objects.cmds_output.KSIRFrequencyTransform

KSIR antenna-port association
-----------------------------
.. autoclass:: gprMax.user_objects.cmds_output.KSIRAntennaPorts

The association is needed only for gain and efficiency. It must name every
physical port, including a zero-amplitude source that acts as a termination
and every eigenmode source or receiver.
Main-grid port IDs are used directly. A subgrid port is qualified by its
subgrid ID, for example ``fine_grid/feed``, ``fine_grid/tl1``, or
``fine_grid/frill1``. Its voltage and current spectra are transformed with the
owning subgrid's finer time step.
For voltage sources, use the ID of the coincident :class:`RxPort`; automatic
transmission-line and magnetic-frill IDs are ``tl1``, ... and ``frill1``, ...
respectively. An eigenmode source is ``portN`` for its explicit port index;
an eigenmode receiver uses its configured ID. Eigenmode transform
frequencies must exactly match the modal direct-DFT bins. Their gain
normalization uses the full modal power matrix rather than an artificial
voltage/current or reference impedance. For example:

.. code-block:: python

    scene.add(gprMax.KSIRFrequencyTransform(
        surface_id='radiation_surface',
        id='antenna_band',
        frequencies=(0.8e9, 1.0e9, 1.2e9),
        window='rectangular',
    ))
    scene.add(gprMax.KSIRAntennaPorts(
        transform_id='antenna_band',
        port_ids=('element1', 'element2'),
    ))
    scene.add(gprMax.KSIRFarFieldArray(
        theta_start=0,
        theta_stop=180,
        theta_step=2,
        phi_start=0,
        phi_stop=360,
        phi_step=2,
        transform_id='antenna_band',
        id='array_pattern',
        outputs=(
            'directivity_dbi',
            'gain_dbi',
            'realized_gain_dbi',
            'radiation_efficiency',
            'total_efficiency',
        ),
    ))

The same surface may independently supply conventional equivalent-current
far fields. For example:

.. code-block:: python

    scene.add(gprMax.NTFFFrequencyTransform(
        surface_id='radiation_surface',
        id='current_band',
        frequencies=(0.8e9, 1.0e9, 1.2e9),
        window='hann',
    ))
    scene.add(gprMax.NTFFFarFieldArray(
        theta_start=0,
        theta_stop=180,
        theta_step=5,
        phi_start=0,
        phi_stop=360,
        phi_step=5,
        transform_id='current_band',
        id='current_pattern',
        outputs=('Etheta', 'Ephi', 'directivity_dbi'),
    ))
    scene.add(gprMax.NTFFTimeFarField(
        theta=90,
        phi=0,
        surface_id='radiation_surface',
        id='current_transient',
        outputs=('Etheta', 'Ephi'),
    ))

The full sphere needed to normalise directivity and efficiency is generated
internally. The directions stored for ``array_pattern`` remain exactly those
requested above. Gain uses the coherent net accepted power of the complete
port set, so amplitudes and delays applied to the source waveforms can model
array steering. For broadband pulses, a time delay represents true-time-delay
steering; a fixed phase shift is frequency-specific.

KSIR exact time-domain receivers
--------------------------------
.. autoclass:: gprMax.user_objects.cmds_output.KSIRTimeRx

.. autoclass:: gprMax.user_objects.cmds_output.KSIRTimeRxSpherical

.. autoclass:: gprMax.user_objects.cmds_output.KSIRTimeRxArray

KSIR exact frequency-domain receivers
-------------------------------------
.. autoclass:: gprMax.user_objects.cmds_output.KSIRFrequencyRx

.. autoclass:: gprMax.user_objects.cmds_output.KSIRFrequencyRxSpherical

.. autoclass:: gprMax.user_objects.cmds_output.KSIRFrequencyRxArray

KSIR range-normalized far fields
--------------------------------
.. autoclass:: gprMax.user_objects.cmds_output.KSIRFarField

.. autoclass:: gprMax.user_objects.cmds_output.KSIRFarFieldArray

The spherical receiver radius is explicit and produces an exact physical
finite-distance field. :class:`KSIRFarField` deliberately has no radius and
returns ``r * exp(+j*k*r) * field``. All spherical angles use theta from
``+z`` and phi from ``+x`` towards ``+y``. A surface face that coincides with
a declared PEC or PMC symmetry boundary is completed automatically by image
theory.

Equivalent-current far fields
-----------------------------
.. autoclass:: gprMax.user_objects.cmds_output.NTFFFrequencyTransform

.. autoclass:: gprMax.user_objects.cmds_output.NTFFFarField

.. autoclass:: gprMax.user_objects.cmds_output.NTFFFarFieldArray

.. autoclass:: gprMax.user_objects.cmds_output.NTFFAntennaPorts

These classes provide the conventional Love-current frequency transform and
share the KSIR far-field output and antenna-metric definitions. They do not
provide finite-distance receivers. See :ref:`ntff-formulations` for the
surface-current equations and engineering phasor convention.

Modified one-step transient far fields
---------------------------------------
.. autoclass:: gprMax.user_objects.cmds_output.NTFFTimeFarField

.. autoclass:: gprMax.user_objects.cmds_output.NTFFTimeFarFieldArray

These CPU-only classes implement the modified time-domain equivalent-current
method of Giannopoulos *et al.* [GIAFF1997]_. Their ``times`` are reduced
times for range-normalized far fields, and only samples supported by every
surface patch are returned. The time placement of both current derivatives is
defined in :ref:`ntff-formulations`.

Subgrid
-------
.. autoclass:: gprMax.SubGridHSG

A subgrid is added to the main scene, but its materials and geometry are added
to the subgrid object. With ``autotranslate=True`` these objects can use main
grid coordinates:

.. code-block:: python

    subgrid = gprMax.SubGridHSG(
        p1=(0.06, 0.04, 0.03), p2=(0.12, 0.10, 0.09),
        ratio=3, id='fine_grid',
    )
    scene.add(subgrid)

    subgrid.add(gprMax.Material(
        er=4, se=0, mr=1, sm=0, id='subgrid_material'
    ))
    subgrid.add(gprMax.Sphere(
        p1=(0.09, 0.07, 0.06), r=0.01,
        material_id='subgrid_material',
    ))

    gprMax.run(
        scenes=[scene], n=1, outputfile='subgrid_model',
        subgrid=True, autotranslate=True,
    )


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

For example, select the multipole formulation and set the thickness of each
domain face independently, in the order ``x0, y0, z0, xmax, ymax, zmax``:

.. code-block:: python

    scene.add(gprMax.PMLFormulation(formulation='MRIPML'))
    scene.add(gprMax.PMLThickness(thickness=(12, 12, 10, 12, 12, 10)))

Symmetry Boundary
-----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.SymmetryBoundary

.. code-block:: python

    scene.add(gprMax.SymmetryBoundary(face='x0', type='pmc'))

PML Properties
--------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.PMLProps

.. warning::

    ``PMLProps`` is retained for compatibility with older Python models. New
    models should use ``PMLFormulation``, ``PMLThickness``, and, when detailed
    coefficient control is needed, ``PMLCFS``.

PML CFS
-------
Allows you control of the specific parameters that are used to build each order of the PML. Up to a second order PML can currently be specified, i.e. by using two ``PMLCFS`` commands.

.. autoclass:: gprMax.user_objects.cmds_multiuse.PMLCFS

For example, the following explicitly requests the coefficient profiles used
by the default first-order PML:

.. code-block:: python

    scene.add(gprMax.PMLCFS(
        alphascalingprofile='constant',
        alphascalingdirection='forward',
        alphamin=0, alphamax=0,
        kappascalingprofile='constant',
        kappascalingdirection='forward',
        kappamin=1, kappamax=1,
        sigmascalingprofile='quartic',
        sigmascalingdirection='forward',
        sigmamin=0, sigmamax=None,
    ))

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
* ``sigmascalingdirection = 'forward'``
* ``sigmamin = 0``
* ``sigmamax = None``

.. note::

    * The parameters will be applied to all slabs of the PML that are switched on.
    * Using ``None`` for the maximum value of :math:`\sigma` forces gprMax to calculate it internally based on the relative permittivity and permeability of the underlying materials in the model.
    * ``forward`` direction implies a minimum parameter value at the inner boundary of the PML and maximum parameter value at the edge of the computational domain, ``reverse`` is the opposite.
