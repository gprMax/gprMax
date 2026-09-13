.. _input-api:

************************************
Advanced Model Building (Python API)
************************************

Introduction
============

Practical workflows are in :doc:`sources_ports` and :doc:`studies`.
The entries here describe the model objects and include their output
contracts; complete HDF5 schemas are in :doc:`output`.

gprMax has a choice of two methods for building a model to simulate:

1. A **text-based (ASCII) input file**, which can be created with any text editor, and uses a series of gprMax commands which begin with the hash character (``#``). This method is recommended for beginners and those not familiar with Python, and is described in the :ref:`input-hash-cmds` section.
2. A **Python API**, which includes all the functionality of method 1 as well as several more advanced features. This method is recommended for those who prefer to use Python or need access to specific API-only advanced features, and is described in this section of the documentation.

The Python API in gprMax allows users to access gprMax functions directly from Python by importing the gprMax module. There are several advantages to using the API:

* Users can take advantage of the Python language - for instance, the structural elements of Python can be utilised more easily.
* gprMax objects can be used directly within functions, classes, modules and packages. In this way collections of components can be defined, reused and modified. For example, complex targets can be imported from a separate module and combined with an antenna from another module.
* The API can interface with other Python libraries. For example, the API could be used to create a parametric antenna and the external library Scipy could then be used to optimise its parameters.

The API syntax is generally more verbose than the hash-command syntax.
However, for commands with a variable number of parameters, such as those
that add dispersive properties, the API may be more manageable.

Source/receiver positions and output bounds containing ``inf`` are resolved
against each grid when it is built. The declaration retains its symbolic
coordinates, so reusing it with a different grid spacing or domain does not
freeze the first build's resolved position.

``str(user_object)`` is a readable, hash-command-style diagnostic, not a general
API-to-input-file exporter. In particular, omitted optional fields do not
always round-trip through the positional hash-command syntax. Use the
documented command syntax when writing an input file.

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

For deliberate mesh-resolution experiments, ``allow_underresolved=True`` changes
the pre-solve wavelength-sampling rejection into a warning. The default is
``False``; material stability and output-validity checks remain enabled. See
:ref:`spatial-resolution diagnostic <spatial-resolution-diagnostic>` for the scope of this override and the
exclusion of zero-amplitude passive-source waveforms.

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

Declaring ``SurfaceImpedance`` automatically caps the effective factor at
0.99, preserves any smaller user factor, and reports an automatic reduction
in the build log. See :ref:`impedance-automatic-timestep`.

Dispersive materials also undergo the :ref:`dispersive_timestep_check` before
time stepping. If it fails, the run stops without changing the chosen timestep
or material parameters. Use ``TimeStepStabilityFactor(f=...)`` to explicitly
request a smaller timestep; the diagnostic gives a checked candidate when
one is found.

Output Directory
----------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.OutputDir

Relative ``OutputDir(dir=...)`` paths are resolved against the working directory
when the Scene is built. This differs from paths in the ``#output_dir`` hash command, which are
relative to the top-level input file. The resolved directory is retained during
``geometry_fixed=True`` repetition, with distinct numbered model and snapshot
paths.

Magnetic Averaging
------------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.MagneticAveraging

Dispersive Averaging
--------------------
.. autoclass:: gprMax.user_objects.cmds_singleuse.DispersiveAveraging

Reusable parameter studies
--------------------------

See :doc:`studies` for the task-selection guide, case semantics, restrictions,
CSV/Python examples, result types and restart behaviour.
:class:`gprMax.Study`, :class:`gprMax.StudyCase` and
:class:`gprMax.ObjectState` define the common scheduling interface.

.. autoclass:: gprMax.studies.Study
    :members: from_csv

.. autoclass:: gprMax.studies.GPRStudy

Fixed-topology terminal-source studies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`SourceStudy <study-source>` schedules fixed transmission-line, frill
and rational-network generators without removing their passive loads.

.. autoclass:: gprMax.studies.SourceStudy

Finite-resistance voltage-port studies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`PortStudy <study-port>` computes a complete S matrix from
finite-resistance voltage ports. Hard sources are not matched terminations.

.. autoclass:: gprMax.studies.PortStudy

.. autoclass:: gprMax.studies.PortStudyResult

Eigenmode-port studies and array synthesis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`EigenmodeStudy <study-eigenmode>` provides modal S matrices,
:class:`gprMax.ArrayCodebook` states and embedded far-field synthesis.

.. autoclass:: gprMax.studies.EigenmodeStudy

.. autoclass:: gprMax.studies.EigenmodeStudyResult
    :members: from_hdf5, excitation_weights, outgoing, evaluate_array_state, evaluate_codebook

.. autoclass:: gprMax.studies.ModalWeight

.. autoclass:: gprMax.studies.ArrayState

.. autoclass:: gprMax.studies.ArrayCodebook

.. autoclass:: gprMax.studies.EmbeddedFarFieldSpec

.. autoclass:: gprMax.studies.EmbeddedFarFieldBank

.. autoclass:: gprMax.studies.ArrayStateResult

.. autoclass:: gprMax.studies.ArrayFarFieldResult

.. autofunction:: gprMax.studies.modal_array_weights

.. autofunction:: gprMax.studies.combine_embedded_modal_responses

Plane-wave and RCS studies
^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`PlaneWaveStudy <study-plane-wave>` schedules incident directions and
polarisation. :ref:`GPRStudy <study-gpr>` schedules point-source acquisitions.

.. autoclass:: gprMax.studies.PlaneWaveStudy

.. autoclass:: gprMax.studies.StudyCase

.. autoclass:: gprMax.studies.ObjectState

Other general settings
----------------------

Typical general settings are added directly to the scene:

.. code-block:: python

    scene.add(gprMax.Title(name='buried_target'))
    scene.add(gprMax.OMPThreads(n=8))
    scene.add(gprMax.TimeStepStabilityFactor(f=0.95))
    scene.add(gprMax.OutputDir(dir='results'))
    scene.add(gprMax.MagneticAveraging(mode='harmonic'))
    scene.add(gprMax.DispersiveAveraging(enabled=True))

Material functions
==================

Material
--------
.. autoclass:: gprMax.user_objects.cmds_multiuse.Material

User-defined material IDs must not contain ``+``. It is reserved for averaged
and other internally generated materials; use ``_`` instead. This also applies
to the optional ``id`` of :class:`gprMax.MaterialFromDatabase`. Generated names
stored by :class:`gprMax.GeometryObjectsWrite` remain valid when restored with
:class:`gprMax.GeometryObjectsRead`.

Surface impedance
-----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.SurfaceImpedance

.. include:: _includes/surface_impedance_parameters.rstinc

Assign the ID to a closed, cell-occupying geometry object. For model choices,
fit-band selection, geometry, and supported configurations, see
:doc:`impedance_surfaces`. The boundary equations are in
:doc:`impedance_surfaces_theory`; stored metadata is in
:ref:`impedance-output`.

Material from database
----------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.MaterialFromDatabase

See :doc:`material_databases` for database lookup, schema, provenance, and
geometry-file migration.

Material mass density
---------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.MaterialDensity

Mass density is optional cell-centred physical metadata in SI units
(:math:`\mathrm{kg\,m^{-3}}`). It does not alter electromagnetic update
coefficients or dielectric smoothing. Derived dosimetry outputs require a
finite, positive density for every selected material.

Material range
--------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.MaterialRange

``MaterialRange`` defines bounded stochastic constitutive properties for a
:class:`gprMax.FractalBox`; it does not create a homogeneous material. See
:ref:`#material_range <material_range>` for how the sampled values are used.

Material list
-------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.MaterialList

``MaterialList`` groups existing material IDs for stochastic assignment by a
:class:`gprMax.FractalBox`. See :ref:`#material_list <material_list>` for the
corresponding input-file command.

Debye Dispersion
----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.AddDebyeDispersion

Lorentz Dispersion
------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.AddLorentzDispersion

Drude Dispersion
----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.AddDrudeDispersion

Material CRIM
-------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.MaterialCrim

See :ref:`#material_crim <material_crim>` for the CRIM mixing formula and how it is
specialised to the matrix/dispersive-phase/air case used here.

The CRIM object, like the Peplinski object below, is a mixing model for a
:class:`FractalBox`, rather than a homogeneous material. It combines a
fixed-fraction non-dispersive matrix material with a single-pole Debye
dispersive material (assumed water or brine); the remaining volume fraction
is assumed to be air. Both materials must already exist in the scene, and
the dispersive material must have exactly one Debye pole. Conductivity is
mixed separately by volume fraction as described for the hash command:

.. code-block:: python

    scene.add(gprMax.Material(er=5, se=0, mr=1, sm=0, id='sand'))

    scene.add(gprMax.Material(er=4.9, se=0, mr=1, sm=0, id='water'))
    scene.add(gprMax.AddDebyeDispersion(
        poles=1, er_delta=(73.3389,), tau=(8.0994e-12,),
        material_ids=('water',),
    ))

    scene.add(gprMax.MaterialCrim(
        matrix_id='sand', matrix_fraction=0.6,
        dispersive_id='water', fraction_lower=0.02, fraction_upper=0.35,
        f_min=1e6, f_max=3e9, a=0.5,
        id='wetsand',
    ))

    scene.add(gprMax.FractalBox(
        p1=(0, 0, 0), p2=(0.1, 0.1, 0.08),
        frac_dim=1.5, weighting=(1, 1, 1),
        n_materials=10, mixing_model_id='wetsand',
        id='my_fractal_box',
    ))

Soil Peplinski
--------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.SoilPeplinski

The dispersion objects modify materials that have already been added to the
same scene. Each material ID below is therefore unique. These objects describe
electric dispersion and cannot target PEC or PMC materials, including custom
materials with infinite electric or magnetic conductivity:

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

Cell-centred geometry tags
--------------------------

The volumetric commands ``Box``, ``Sphere``, ``Cylinder``, ``Cone``,
``CylindricalSector``, ``Ellipsoid``, a ``Triangle`` with non-zero thickness,
and ``FractalBox`` accept the optional keyword ``tag``. A tag is semantic
metadata independent of the electromagnetic material, for example
``tag='cranial_bone'``. Reusing the same string on several primitives makes
them one semantic region without retaining a list of the individual
primitives.

Tags follow the same ordered overwrite semantics as geometry. A tagged
primitive writes its tag to every cell it occupies; an untagged primitive
writes tag ID zero and therefore clears an older tag in its cells. This makes
constructive geometry work naturally. For example, an untagged free-space
cylinder drawn inside a tagged material cylinder leaves a tagged shell and an
untagged hollow interior. A free-space volume may itself be tagged when that
region is intentionally significant. Dielectric smoothing does not alter tag
membership.

.. code-block:: python

    scene.add(gprMax.Cylinder(
        p1=(0.10, 0.10, 0.05), p2=(0.10, 0.10, 0.15),
        r=0.04, material_id='plastic', tag='container',
    ))
    scene.add(gprMax.Cylinder(
        p1=(0.10, 0.10, 0.05), p2=(0.10, 0.10, 0.15),
        r=0.03, material_id='free_space',
    ))

Tag ID zero is permanently reserved for ``untagged``. The model stores one
compact integer per cell only when at least one tag is present; tag arrays are
not part of the FDTD field update and are not transferred to accelerators.
Tags are flat rather than hierarchical. Larger groups such as ``head`` can be
formed later by selecting several leaf tags such as ``brain``, ``eyes``, and
``cranial_bone``.

Closed impedance volumes
------------------------

Assign a :class:`gprMax.SurfaceImpedance` ID directly as the ``material_id``
of a supported ordinary geometry object. No separate conversion object is
needed. ``SurfaceImpedance`` defines the boundary response; assigning its ID
to geometry creates an opaque impedance volume. Interior fields are excluded,
not solved using a bulk conductivity. Even a one-cell-thick volume is opaque,
not a transmissive sheet.

.. code-block:: python

    scene.add(gprMax.SurfaceImpedance(
        id='metal',
        preset='copper',
        fit_frequency_range=(8e9, 12e9),
    ))
    scene.add(gprMax.Sphere(
        p1=(0.15, 0.10, 0.08), r=0.025,
        material_id='metal', averaging='n',
    ))

Use a scalar ``material_id`` and finite-volume geometry that occupies mesh cells. Sheets,
lines, directional assignments, and dielectric smoothing are unavailable
for this one-sided opaque boundary. Geometry follows declaration order:
later objects overwrite earlier ones. The final voxel topology is checked after all cutouts.
Tags remain optional metadata.

For supported primitives, PEC/PMC contacts, symmetry, 2D extrusion,
clearance, and import/export limitations, see :doc:`impedance_surfaces`.
Uniform propagation through PML and virtual-guide apertures has additional
host-material and coverage requirements in :ref:`sibc-pml`. Infinite
surface resistance gives exact voxel-face PMC; built-in PMC volume geometry
retains its legacy reflection-plane limitation.

The compiled boundary uses the same discrete surface response in FDTD and
modal solves. The bulk modal operator uses Yee temporal and longitudinal
difference symbols, while bulk dispersive poles still use their analytic
physical-frequency response. See :doc:`impedance_surfaces_theory` for the
boundary reduction and :doc:`eigenmode_port_theory` for modal operators.

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

Fractal definitions can be reused for a geometry preview followed by a solve,
or in multiple rebuilt Scenes. Each fresh grid gets its own fractal volume and
material-bin mapping. Supply a seed for reproducible geometry. With
``geometry_fixed=True``, the already-built geometry is retained instead.

.. note::

    * We are not aware of a formulation of Perfectly Matched Layer (PML) absorbing boundary that can specifically handle distributions of material properties (such as those created by fractals) throughout the thickness of the PML, i.e. this is a required area of research. Our PML formulations can work to an extent depending on your modelling scenario and requirements. You may need to increase the thickness of the PML and/or consider tuning the parameters of the PML (:ref:`pml-tuning`) to improve performance for your specific model.

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
        mixing_model_id='soil_mix', id='ground', seed=1, tag='soil_layer_1',
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
---------------------
.. autoclass:: gprMax.user_objects.cmds_geometry.geometry_objects_read.GeometryObjectsRead

The ``matfile`` argument is no longer supported. To reuse an old HDF5 geometry
with text material commands, follow :ref:`legacy_geometry_conversion`, then
use ``geofile="geometry_converted.h5", material_database="geometry_materials"``.
The insertion coordinates and averaging option remain unchanged.

Geometry Objects Write
----------------------
.. autoclass:: gprMax.user_objects.cmds_output.GeometryObjectsWrite

Geometry views are visualisations. Geometry-object files instead preserve
material-index geometry for insertion into another model. If semantic tags
exist, both outputs also preserve their cell IDs and the ID-to-name table.
This permits a costly tagged anatomy or geological model to be written once
and inserted into later simulations without rebuilding it:

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

A geometry view can be added to an HSG subgrid. Its points and VTK origin are
written in the global model coordinate system, even though its material data
are sampled from the fine local grid. ``GeometryObjectsWrite`` remains a
main-grid operation.

In a different scene, the saved geometry can be inserted at a chosen origin:

.. code-block:: python

    scene.add(gprMax.GeometryObjectsRead(
        p1=(0.05, 0.05, 0),
        geofile='reusable_geometry.h5',
        material_database='reusable_geometry_materials',
        averaging='y',
    ))

``averaging='y'`` reconstructs a voxel-only file with dielectric interface
averaging; its backward-compatible default is ``'n'``. If the file contains
complete ``/ID``, ``/rigidE``, and ``/rigidH`` arrays, those Yee-component
arrays are authoritative and the option is ignored. Dispersive tissue
interfaces additionally require ``gprMax.DispersiveAveraging(enabled=True)``.
Material density and cell-centred geometry tags remain discrete per cell.

Source and output functions
===========================

.. _point-source-bounds:

Point-source positions and boundary components
----------------------------------------------

After coordinates are rounded to grid indices, a Hertzian dipole, voltage
source, or transmission line must lie on a physical electric component;
a magnetic dipole must lie on a physical magnetic component. For a grid
with ``nx``, ``ny``, and ``nz`` cells, the inclusive index ranges are:

.. list-table:: Physical component indices
   :header-rows: 1

   * - Component
     - x index
     - y index
     - z index
   * - Ex
     - 0 .. nx-1
     - 0 .. ny
     - 0 .. nz
   * - Ey
     - 0 .. nx
     - 0 .. ny-1
     - 0 .. nz
   * - Ez
     - 0 .. nx
     - 0 .. ny
     - 0 .. nz-1
   * - Hx
     - 0 .. nx
     - 0 .. ny-1
     - 0 .. nz-1
   * - Hy
     - 0 .. nx-1
     - 0 .. ny
     - 0 .. nz-1
   * - Hz
     - 0 .. nx-1
     - 0 .. ny-1
     - 0 .. nz

For example, ``Ex[i,j,k]`` is located at
``((i+0.5)*dx, j*dy, k*dz)``. Its last physical x index is ``nx-1``,
although the field array also contains a padded entry at ``nx``. By
contrast, its transverse terminal indices ``j=ny`` and ``k=nz`` remain
valid. The boundary condition still determines which components can be
excited there; the bounds check does not change PEC or PMC enforcement.

The checks also apply to dipole positions reached through ``SrcSteps`` or
a study. In 2D, sources must remain on the active invariant layer (index
0 for TM, index 1 for TE). MPI checks use the global physical grid, not
the edge of an individual rank. Subgrid checks use the translated local
fine-grid indices. Receiver and extended-source bounds are separate.

Sources sharing a field component
---------------------------------

Hertzian dipoles or magnetic dipoles assigned to the same discretised Yee
component add their contributions in source-list order. This also applies
when different physical coordinates round to the same component. Different
polarisations address different component arrays.

CPU and accelerator solvers use the same conventional-source ordering:
voltage sources, then transmission lines, then Hertzian dipoles for the
electric update; magnetic dipoles precede magnetic frills for the magnetic
update. Within each family, input order is preserved. The accelerator
kernels process ordinary source lists and transmission-line electric
updates sequentially, so coincident writes do not race.

Not every source adds to the field. An active hard voltage source or
transmission line assigns its electric component; a later assignment to
that component takes precedence. This reproduces the CPU update rules,
not a combined circuit model of several feeds sharing one terminal.
Existing restrictions on overlapping magnetic-frill stencils and duplicate
network-terminal edges still apply.

Waveform
--------
.. autoclass:: gprMax.user_objects.cmds_multiuse.Waveform

The constructor has three forms. Arguments should be supplied by keyword:

.. code-block:: python

    # Built-in analytic waveform
    gprMax.Waveform(wave_type='ricker', amp=1, freq=1e9, id='pulse')

    # MATLAB-default Gaussian-modulated cosine (50% bandwidth at -6 dB)
    gprMax.Waveform(wave_type='gauspulse', amp=1, freq=1e9, id='rf_pulse')

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
with exactly ``iterations`` times, ``arange(iterations) * dt``. The number of
values must match; supply an explicit, strictly increasing ``user_time`` for
another sampling grid. Sampled waveforms default to linear interpolation
inside the supplied time axis and **zero outside it**. ``kind`` and
``fill_value`` override those defaults (including explicit ``'extrapolate'``)
and apply only to sampled waveforms. Numeric fill values are honoured outside
both ends of the time axis. Callables retain their own boundary behaviour.
User-defined
waveforms can drive local Hertzian or magnetic dipoles, voltage sources,
transmission lines, and magnetic-frill sources. The discrete-plane-wave
formulation currently requires a built-in analytic waveform.

Hard voltage sources evaluate only whole-step samples, including the initial
electric field at time zero. Resistive voltage sources and Hertzian dipoles
use their existing half-step current samples; a final half-step outside a
sampled waveform's time axis uses the selected fill value. Zero waveform
amplitude does not remove an active hard-source clamp: its start/stop window
still controls whether the electric edge is prescribed.

For a Gaussian envelope with a cosine carrier, see the
:ref:`modulated Gaussian waveform example <waveform-modulated-gaussian>`.
It includes the built-in ``gauspulse`` definition, time trace, power spectrum,
intrinsic delay, and a custom-bandwidth callable. The built-in pulse uses
MATLAB's default fractional bandwidth of 0.5 at -6 dB and starts at the
-60 dB envelope level; its peak occurs about ``2.781 / freq`` seconds after
the source start time.

Eigenmode band, ports, excitation, and virtual guides
-----------------------------------------------------

.. autoclass:: gprMax.user_objects.cmds_multiuse.EigenmodeBand

.. include:: _includes/eigenmode_band_parameters.rstinc

.. autoclass:: gprMax.user_objects.cmds_multiuse.EigenmodePort

.. include:: _includes/eigenmode_port_outputs.rstinc

.. include:: _includes/eigenmode_port_parameters.rstinc

.. autoclass:: gprMax.user_objects.cmds_multiuse.EigenmodeExcitation

.. include:: _includes/eigenmode_port_outputs.rstinc

.. include:: _includes/eigenmode_excitation_parameters.rstinc

.. autoclass:: gprMax.user_objects.cmds_multiuse.VirtualWaveguide

.. include:: _includes/virtual_waveguide_parameters.rstinc

.. autoclass:: gprMax.user_objects.cmds_output.EigenmodeFieldOutput

Use ``scene.add(gprMax.EigenmodeFieldOutput(filename="bank", ports=(1, 2)))``
to export prepared tracked modal profiles to ``bank.modes.h5`` in the run
output directory, including during geometry-only runs. Omit ``ports`` to
export all prepared physical ports; ``filename`` defaults to ``port_modes``.
Export requires a serial 3D main grid. The equivalent hash command and output
restrictions are described under :ref:`hash-eigenmode-field-output`.

A band selects output frequencies; a port defines a reference plane and
monitored modes; an excitation drives a channel; a virtual guide provides
a separate matched continuation. The full workflows and equivalent hash
commands are in :doc:`eigenmode_port`. See :doc:`input_hash_cmds` for
the complete command syntax, :doc:`output` for stored arrays, and
:doc:`eigenmode_port_theory` for numerical methods.

A direct eigenmode model may also be placed wholly inside one HSG subgrid.
Add its band, ports, waveform (when one is used), and excitation to that same
subgrid object. With ``autotranslate=True``, the plane coordinates remain
global physical coordinates:

.. code-block:: python

    fine_grid.add(gprMax.Waveform(
        wave_type='contsine', amp=1, freq=22e9, id='fine_wave',
    ))
    fine_grid.add(gprMax.EigenmodeBand(
        id='fine_band', fmin=22e9, fmax=22e9, points=1,
    ))
    fine_grid.add(gprMax.EigenmodePort(
        port=1,
        p1=(0.045, 0.039, 0.039),
        p2=(0.045, 0.051, 0.049),
        direction='+', modes=(1,), anchors=(22e9,),
    ))
    fine_grid.add(gprMax.EigenmodeExcitation(
        port=1, mode=1, waveform='fine_wave',
    ))

The FDFD solver samples the final component-resolved material slice of the
fine Yee grid and uses its two transverse spatial steps. Injection and modal
observation then run at every fine-grid time step. The complete plane and its
adjacent staggered Yee stencil must lie strictly inside the subgrid working
region; a plane touching the HSG coupling surface or entering its auxiliary
or PML region is rejected. Ports cannot be divided between the main grid and
a subgrid, or between different subgrids. ``VirtualWaveguide`` can be added
to the same subgrid and referenced to one of its ports. The auxiliary guide
then inherits the fine grid's spatial and temporal steps, material
cross-section, update coefficients, and iteration count; it is not resampled
from the coarse main grid.

Voltage Source
--------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.VoltageSource
    :members: result

.. include:: _includes/voltage_port_outputs.rstinc

A finite-resistance voltage source is a one-cell Thévenin gap source. It also
acts as a port: after a supported 3-D simulation, ``source.result`` provides
the sampled gap voltage together with the frequency-domain port
quantities. Its physical resistance is the wave-reference impedance. A zero
resistance creates a hard source with an independent wave-reference
impedance: ``reference_impedance`` defaults to 50 Ohms and does not add a
physical resistance to the gap.

.. code-block:: python

    feed = gprMax.VoltageSource(
        p1=(0.05, 0.05, 0.02),
        polarisation='z',
        resistance=50,
        waveform_id='pulse',
        id='feed',
    )
    scene.add(feed)

The waveform amplitude is the generator voltage in volts. See
:ref:`#voltage_source <voltage_source>` for the one-cell source equation and
the definition of the automatic port spectra.

A hard source also prescribes the initial electric field before sample zero
is stored. Subsequent prescriptions use the new electric time level
:math:`(n+1)\Delta t`. ``start`` and ``stop`` are inclusive at these physical
times. A zero waveform still clamps the edge while active; ``start=0`` and
:math:`0<\mathtt{stop}<\Delta t` apply only the initial impulse and then release
the edge. In a subgrid, use its local :math:`\Delta t`.

Voltage-source S11 and input impedance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Every 3-D single-Yee-edge :class:`gprMax.VoltageSource` normally owns the
necessary hidden field monitor and calculates corrected complex ``S11``,
``Zin``, and ``Yin`` after the solve. The exception is a zero-resistance hard
source on a domain-minimum transverse boundary: the source remains valid, but
gprMax warns and omits its automatic port because the complete Ampere current
loop lies partly outside the grid. A finite-resistance source uses its
physical resistance as the reference impedance:

.. code-block:: python

    port = gprMax.VoltageSource(
        p1=(0.050, 0.050, 0.020),
        polarisation='z',
        resistance=50,
        waveform_id='source_wave',
        id='feed',
        spectrum_limit=10,
    )
    scene.add(port)

The default ``spectrum_limit=10`` retains frequencies having at least ten
cells per shortest material wavelength. A research run can explicitly request
all native non-negative FFT bins while retaining the normal validity metadata:

.. code-block:: python

    scene.add(gprMax.VoltageSource(
        p1=(0.050, 0.050, 0.020),
        polarisation='z', resistance=50,
        waveform_id='source_wave',
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
        id='ideal_feed',
        reference_impedance=50,
    ))

After ``gprMax.run`` completes, ``port.result`` provides the same numerical
arrays that are stored under ``/ports/feed`` in the model HDF5 file. For a
hard source, gprMax obtains terminal current from the surrounding magnetic-
field loop and accounts explicitly for the half-step phase difference from
the integer-time voltage during transformation.

A finite-resistance source on a dispersive edge includes the background
material's complete complex permittivity in the Yee-gap correction. The same
correction is used for terminal current and accepted power when calculating
antenna parameters or normalising SAR. A hard
source on a dispersive edge is not yet supported because its sampled
Ampere-loop current requires the discrete polarisation-current contribution
to be separated explicitly.

The automatic voltage-source port also supports domain-decomposed MPI CPU
models. The owning rank
stores the voltage history; hard-source current loops may cross internal rank
faces because magnetic halos are synchronised before sampling. Histories are
gathered and transformed once on the coordinator rank, while ``port.result``
is rebound to that final result for Python API use.

The source may also be placed inside a ``SubGridHSG``. Add the waveform and
voltage source to the subgrid object; the port is then sampled
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
        id='fine_feed',
    ))

The result is stored at
``/subgrids/<subgrid ID>/ports/<port ID>``. The source belongs to the owning
grid object, so its discretised coordinate, material edge,
``dl``, and ``dt`` are unambiguous.

Hertzian Dipole Source
----------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.HertzianDipole

.. include:: _includes/dipole_outputs.rstinc

This additive electric source uses the waveform amplitude as current
:math:`I` in amperes and an effective dipole length equal to the cell step in
the selected direction. It therefore impresses

.. math::

   J_s = \frac{I\,\Delta l}{\Delta x\,\Delta y\,\Delta z}.

For example:

.. code-block:: python

    scene.add(gprMax.HertzianDipole(
        p1=(0.05, 0.05, 0.05),
        polarisation='z',
        waveform_id='pulse',
    ))

It is an ideal field excitation, not a circuit port, and consequently does
not produce S-parameters or an input impedance.

Magnetic Dipole Source
----------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.MagneticDipole

.. include:: _includes/dipole_outputs.rstinc

This is the magnetic-current dual of the additive Hertzian source. It is
placed at a magnetic Yee-field location and does not represent a circuit
port. For example:

.. code-block:: python

    scene.add(gprMax.MagneticDipole(
        p1=(0.05, 0.05, 0.05),
        polarisation='y',
        waveform_id='pulse',
    ))

The permitted source component and invariant-axis index in a 2-D model follow
the surviving TM or TE magnetic-field components and are validated when the
scene is built.

Rational lumped-network terminal
--------------------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.RationalNetwork
.. autoclass:: gprMax.user_objects.cmds_multiuse.NetworkTerminal
.. autoclass:: gprMax.user_objects.cmds_multiuse.NetworkExcitation

.. include:: _includes/network_port_outputs.rstinc

``RationalNetwork`` defines the reusable driving-point admittance

.. math::

    Y(s)=G+sC+\sum_m\frac{r_m}{s-p_m}.

``NetworkTerminal`` places it on one electric Yee edge. It is passive unless
a ``NetworkExcitation`` supplies a Thévenin open-circuit waveform. For
example, a 50 Ohm driven port is

.. code-block:: python

    scene.add(gprMax.RationalNetwork(
        id='source50', conductance=1 / 50, capacitance=0,
    ))
    scene.add(gprMax.NetworkTerminal(
        p1=(0.05, 0.05, 0.02), polarisation='z',
        network_id='source50', id='feed',
    ))
    scene.add(gprMax.NetworkExcitation(
        terminal_id='feed', waveform_id='pulse',
    ))
    network_port = gprMax.NetworkPort(terminal_id='feed')
    scene.add(network_port)

An inductor is represented by ``poles=(0,)`` and ``residues=(1/L,)``;
a series :math:`RL` branch uses ``poles=(-R/L,)`` and
``residues=(1/L,)``. Complex terms must be supplied as conjugate pairs.
The circuit-to-edge formulation follows the arbitrary linear lumped-network
FDTD approaches of [PER1999]_ and [CHE2007]_. Their underlying classic PLRC
time discretisation is improved here using the exponential recursive-
convolution treatment of Giannakis and Giannopoulos [GIA2014]_: every pole
current is evaluated analytically at the electric half-step for a linearly
varying voltage, rather than estimated by averaging its two integer-time
values. State is stored only for placed terminals. Independent one-port
rational networks are supported in 3-D on the CPU, CUDA, OpenCL, and Metal
solvers, including domain-decomposed MPI CPU models; terminals inside
HSG subgrids use the CPU or CUDA solver. An MPI terminal is advanced only on
the rank that owns its electric edge, and its histories are gathered for port
post-processing. Device runs keep the network recurrence and field correction
on the compute device
and copy the completed histories back after the solve. Coupled multiport
admittance matrices are reserved for a later extension.

For :math:`Y=1/R`, ``NetworkExcitation`` and a conventional finite-resistance
``VoltageSource`` are the same discrete Thévenin source when their position,
resistance, and waveform are identical. This equivalence does not apply to a
zero-resistance hard voltage source.

Transmission Line
-----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.TransmissionLine

.. include:: _includes/transmission_line_outputs.rstinc

Every transmission-line source automatically writes its incident and terminal
voltage/current histories together with ``frequency``, ``S11``, ``Zin``, and
``Yin`` beneath ``/tls/tlN`` in the model HDF5 output. ``Zin`` is derived from
the voltage-wave S11 result; ``Zin_current`` is an independent, stagger-aware
current-wave check. Voltage and transmission-line sources both own their
terminal outputs, so no separate receiver-port object is required. See
:ref:`Simulation Output <output>` for the equations and validity masks.

Magnetic Frill Source
---------------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.MagneticFrillSource

.. include:: _includes/frill_port_outputs.rstinc

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
solvers and by domain-decomposed MPI CPU models. Its four magnetic feed edges
may cross internal MPI rank boundaries, and PMC image completion is supported
at minimum-face symmetry corners. It is also supported inside a CPU or CUDA
``SubGridHSG``. Add the waveform, PEC ground plane, thin wire, and magnetic
frill to the same subgrid object,
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
-----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.DiscretePlaneWaveAngles

.. include:: _includes/plane_wave_outputs.rstinc

Plane Wave Vector
-----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.DiscretePlaneWaveVector

.. include:: _includes/plane_wave_outputs.rstinc

Plane Wave Axial
----------------
.. autoclass:: gprMax.user_objects.cmds_multiuse.DiscretePlaneWaveAxial

.. include:: _includes/plane_wave_outputs.rstinc

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
use the CPU, CUDA, OpenCL, and Apple Metal solvers.
Homogeneous angle/vector plane waves and layered axial plane waves support
non-dispersive materials and multi-pole Debye, Lorentz, and Drude materials.
Their auxiliary dispersive state uses the same real or complex precision
selected for the main grid. A discrete plane wave must be added to the main
scene, not to a subgrid. Its TFSF box may contain a complete subgrid; where the
two regions overlap, the box must strictly enclose the subgrid's HSG outer
coupling surface so that the TFSF correction stencil remains on the main grid.
MPI domain decomposition is supported. The auxiliary one-dimensional wave is
replicated on every rank, and each rank applies only the TFSF corrections for
the Yee components that it owns. For an axial plane wave, the layered material
profile is assembled once from the distributed grid's actual update
coefficients, including multi-pole dispersive coefficients; no additional
plane-wave communication occurs during timestepping.

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


Specific absorption rate (SAR)
------------------------------

.. autoclass:: gprMax.user_objects.cmds_output.SAR

See the :ref:`electric-only absorption limitation <electric-only-absorption>`
for the magnetic-material warning and its scope, including directional
materials and ideal PMC constraints.

Both isotropic and grid-aligned diagonal anisotropic volume materials are
supported. For a primitive with ``material_ids=(mx, my, mz)``, absorption is
the sum of the x-, y-, and z-component contributions using each material's
frequency-dependent electric loss, not the mean of the three conductivities.
The three constituent materials must have the same positive mass density for
SAR. Missing or conflicting densities are rejected; ``Radiometry`` uses the
same electric absorption calculation without density. See :doc:`output` for
the equations and the supported tensor convention.

``SAR`` selects the final voxelised cells belonging to one or more semantic
geometry tags. Every selected material must first have a mass density in
kg/m\ :sup:`3`, assigned with :class:`MaterialDensity` or supplied by a
material database. For example:

.. code-block:: python

    scene.add(gprMax.MaterialDensity(density=1040, material_ids=('brain',)))
    scene.add(gprMax.SAR(
        frequencies=np.linspace(0.8e9, 1.2e9, 41),
        waveform_id='pulse',
        tags=('brain_region',),
        id='brain_sar',
        target_amplitude=1.0,
        spectrum_limit=10,
        averaging_masses=(0.001, 0.01),
    ))

``target_amplitude`` uses the source's native excitation units. In
particular, it is incident electric-field amplitude in V/m for a discrete
plane wave and generator voltage in V for a voltage source. The resulting
SAR is therefore tied to that explicitly stated source normalisation.
For a 3-D Hertzian dipole, use ``normalisation='current_moment'`` and give
``target_amplitude`` in A m. For a discrete plane wave, use
``normalisation='incident_flux'`` and ``target_flux`` in W/m\ :sup:`2` when
power flux rather than electric-field amplitude is the required physical
normalisation.
Spatial mass averaging is opt-in. With the default ``averaging_masses=()``,
gprMax writes local cell SAR, absorbed-power density, and per-tag summaries
without running the potentially expensive mass-averaging stage. Supply
``(0.001, 0.01)`` to request the standard 1 g and 10 g results, or any other
positive masses in kg for research applications. Density is constant within
each final tagged cell; only the included volume fraction of a cell changes
at an averaging-cube boundary.

For a one-watt accepted-power result from a named port, use:

.. code-block:: python

    scene.add(gprMax.SAR(
        frequencies=np.linspace(0.8e9, 1.2e9, 41),
        tags=('brain_region',), id='brain_sar_1W',
        normalisation='accepted_power', port_id='feed', target_power=1.0,
    ))

``incident_power`` is also available. ``target_power`` is in W for 3-D models
and W/m for invariant 2-D models. Power normalisation currently requires
a rectangular transform window and a physical, valid port-power result at
each requested frequency.

Selected tagged cells that lie in boundary or internal PML regions are
excluded automatically because PML loss is not physical material absorption.

``SAR`` may be added to a :class:`SubGridHSG` in the same way as other
subgrid outputs. Fields are transformed using the fine-grid ``dt`` and the
result is written below ``/subgrids/<subgrid ID>/sar/<output ID>``. The
normalising source may be on the main grid or the subgrid; its waveform DFT
uses the timestep of the grid that owns the source. This is useful when a
main-grid plane wave or antenna illuminates finely resolved tagged tissue.
Mass averaging is local to the selected subgrid tag volume, so the complete
tissue region required by an averaging cube should be contained inside the
subgrid working region.

Reduced 2-D TM and TE main-grid models are supported. SAR is evaluated only
on the genuine field plane (invariant index zero for TM and one for TE), and
only the active electric components are transformed. Tag-integrated absorbed
power and mass are written per unit invariant length. Spatial mass averaging
is not yet available in 2-D, so ``averaging_masses`` must remain empty for
these models.

See :ref:`sar-2d-cylinder-validation` for analytical TMz and TEz validation
against homogeneous lossy-cylinder series over fat-, skin-, and muscle-like
material properties.

The default permits output only while the shortest wavelength in any model
material is sampled by at least ten cells. Use ``spectrum_limit=8`` for a
lambda/8 criterion. ``spectrum_limit='nyquist'`` is an explicit research
override: it retains the requested frequencies but does not imply spatial
accuracy. Three- and two-dimensional CPU, CUDA, OpenCL, and Metal models are
supported on the main grid. MPI domain-decomposed CPU models and
three-dimensional CPU and CUDA HSG subgrids are also supported. Under MPI, source and
port normalisation and any requested spatial mass averaging are completed
globally on the coordinator, so tag volumes and averaging cubes may cross rank
boundaries. See :ref:`sar-output` for the formulation and HDF5 schema.

Radiometric absorption weighting
--------------------------------

.. autoclass:: gprMax.user_objects.cmds_output.Radiometry

The :ref:`electric-only absorption limitation <electric-only-absorption>`
and its magnetic-material warning apply to every radiometry normalisation.

``Radiometry`` is the density-independent counterpart of ``SAR``. It uses the
same tagged-cell field transforms and loss calculation, but writes absorbed
power and a source-normalised absorption weighting without requiring
``MaterialDensity``. A plane-wave absorption cross section is requested with:

.. code-block:: python

    scene.add(gprMax.Radiometry(
        frequencies=np.linspace(0.5e9, 2e9, 61),
        waveform_id='incident', tags=('subsurface_layer',),
        id='layer_absorption', normalisation='incident_flux',
        target_flux=1.0,
    ))

For an antenna or local probe with a physical port, omit ``waveform_id`` and
normalise to port power:

.. code-block:: python

    scene.add(gprMax.Radiometry(
        frequencies=np.linspace(0.5e9, 2e9, 61),
        tags=('subsurface_layer',), id='probe_weighting',
        normalisation='accepted_power', port_id='feed', target_power=1.0,
    ))

For a portless Hertzian source, ``current_moment`` gives an absorption kernel
per squared A m. ``waveform`` remains available for every source and retains
that source's native excitation units. Outputs and their dimensional meaning
are described in :ref:`radiometry-output`.

Rational-network S11 and input impedance
----------------------------------------
.. autoclass:: gprMax.user_objects.cmds_output.NetworkPort

.. include:: _includes/network_port_outputs.rstinc

``NetworkPort`` requests the output for an existing ``NetworkTerminal``. Its
terminal ID becomes the HDF5 port ID:

.. code-block:: python

    network_port = gprMax.NetworkPort(
        terminal_id='feed', reference_impedance=50, spectrum_limit=10,
    )
    scene.add(network_port)

After the solve, ``network_port.result`` contains aligned voltage/current
histories and ``S11``, ``Zin``, and ``Yin`` spectra. The background Yee-gap
capacitance and conductance are removed from terminal current. Omitting
``NetworkExcitation`` leaves a passive measurable port; it has no meaningful
source-normalised S11 but can still report impedance/admittance where the
response is numerically defined. A port inside a Python API subgrid is written
beneath ``/subgrids/<subgrid ID>/ports/<terminal ID>`` and uses that subgrid's
fine ``dl`` and ``dt``.

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

A snapshot can also be added to an HSG subgrid. It is sampled on every fine
subgrid time step, uses the subgrid's spatial discretisation, and records its
origin in the global model coordinate system. The requested time or iteration
is interpreted against the owning subgrid's ``dt`` and iteration count.
In a reduced 2-D model, any requested invariant-axis extent is collapsed to
the single genuine field plane: index zero for TM or index one for TE.

Snapshot values are evaluated at the regular output-cell centres written to
the file. Coarse ``dl`` uses linear interpolation of native Yee samples, not
volume averaging; native-cell spacing preserves the original arithmetic.
Non-dividing interior ROIs keep ``ceil((p2-p1)/dl)`` output cells. Construction
rejects a final regular cell whose centre or interpolation stencil requires
samples outside the physical Yee grid. Unlike older versions, this does not
silently clip or enlarge ``p2`` or alter live-axis ``dl``. Choose a smaller
extent or spacing instead. MPI exchanges only the required native samples at
snapshot iterations; ranks with no output cells still participate.

The snapshot ``time`` is rounded to the nearest full electric-field time
level :math:`n\Delta t` (halfway values round to the earlier step), while
``iterations=n`` selects the same zero-based level directly. Snapshot electric
fields are at :math:`n\Delta t` and magnetic fields retain their native Yee
stagger at :math:`(n-1/2)\Delta t`; there is no temporal averaging. Valid
indices are ``0 <= n <`` the owning grid's number of iterations. This temporal
definition is separate from the existing spatial collocation of field
components in a snapshot cell.

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
    * - ``NTFFLayeredBackground``
      - ``#ntff_layered_background``
    * - ``NTFFLayeredFrequencyTransform``
      - ``#ntff_layered_frequency``
    * - ``NTFFLayeredTimeTransform``
      - ``#ntff_layered_time``
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
    * - ``NTFFLayeredTimeFarField``
      - ``#ntff_layered_time_far_field``
    * - ``NTFFLayeredTimeFarFieldArray``
      - ``#ntff_layered_time_far_field_array``

The mapping covers the reusable operations and their normal options. The
Python API also exposes advanced keyword arguments that cannot be entered
positionally in a hash command: ``NTFFSurface.origin``, and
``save_surface_dft`` and ``plane_wave_index`` on either frequency-transform
class. Hash
commands use the default surface centre, save the surface DFT, and associate
an enclosed plane wave automatically.

The ``KSIR*`` classes require a closed ``NTFFSurface``; this rule is independent
of whether an eigenmode source is present. Traditionally a real guide extends
through the domain PML and the crossed Huygens face is omitted, which makes
that open surface unsuitable for KSIR. A ``VirtualWaveguide`` moves the matched
continuation outside the main domain, allowing the eigenmode-fed antenna to be
enclosed by either a closed KSIR or closed equivalent-current surface.

``NTFFSurface(omit_faces=('x0', 'xmax'))`` creates an open frequency-domain
Huygens surface. ``omit_faces`` accepts one to five distinct Cartesian face
names; at least one of the six faces must remain active. A feed crossing an
opening continues uniformly into its PML, with the impressed source plane
outside the Huygens volume. KSIR and ordinary transient equivalent-current
outputs reject omitted physical faces. The exception is a direct layered-time
transform whose one omitted face coincides exactly with its declared terminal
PEC plane. This is the grounded-slab construction described in
:ref:`ntff-formulations`.

A PEC-backed substrate can be declared through the same background class;
``pec`` is the terminal entry and the adjacent interface is its physical
coordinate:

.. code-block:: python

    scene.add(gprMax.NTFFLayeredBackground(
        id='grounded', axis='z',
        materials=('free_space', 'substrate', 'pec'),
        interfaces=(0.0, -0.002),
    ))

This declaration does not build the corresponding substrate or PEC geometry.
Only directions in the open hemisphere may be requested.

NTFF definitions are main-grid objects, but their notional closed integration
surface may enclose complete HSG subgrids. A surface must not touch or cut an
HSG outer coupling surface: overlapping regions require the NTFF surface to
strictly enclose that outer surface. Sources and scatterers inside the subgrid
then contribute through the normal HSG field exchange. A disjoint subgrid is
permitted. NTFF surfaces cannot be defined inside a subgrid.

The same interface is available with MPI domain decomposition. Surface patches
are distributed between ranks and use the existing field halos, while the
completed results retain the same HDF5 schema as a serial run. MPI NTFF does
not introduce a surface collective during each FDTD iteration; accumulated
time histories or frequency-domain phasors are combined once at finalisation.
Geometry-fixed reuse remains unsupported.

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
physical conventional port, including a zero-amplitude source that acts as
a termination and explicitly monitored rational-network terminals.
Main-grid port IDs are used directly. A subgrid port is qualified by its
subgrid ID, for example ``fine_grid/feed``, ``fine_grid/tl1``, or
``fine_grid/frill1``. Its voltage and current spectra are transformed with the
owning subgrid's finer time step.
For voltage sources, use the source's ``id`` (or its automatic ``portN`` ID); automatic
transmission-line and magnetic-frill IDs are ``tl1``, ... and ``frill1``, ...
respectively. Rational-network ports use their terminal IDs. For active modal
sources use an equivalent-current transform with ``NTFFAntennaPorts``, not
this conventional KSIR gain-normalisation path. Raw KSIR fields on a compatible
closed surface remain a separate capability. See :ref:`port-power-accounting`.
For a conventional two-terminal antenna:

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

.. autoclass:: gprMax.user_objects.cmds_output.NTFFLayeredBackground

.. autoclass:: gprMax.user_objects.cmds_output.NTFFLayeredFrequencyTransform

.. autoclass:: gprMax.user_objects.cmds_output.NTFFLayeredTimeTransform

.. autoclass:: gprMax.user_objects.cmds_output.NTFFFarField

.. autoclass:: gprMax.user_objects.cmds_output.NTFFFarFieldArray

.. autoclass:: gprMax.user_objects.cmds_output.NTFFAntennaPorts

These classes provide the conventional Love-current frequency transform and
share the KSIR far-field output and antenna-metric definitions. They do not
provide finite-distance receivers. See :ref:`ntff-formulations` for the
surface-current equations and engineering phasor convention.

Advanced NTFF utilities
-----------------------

The following low-level API utilities support specialised post-processing
and research workflows. They are not scene objects and should not be passed
to ``Scene.add``.

.. autoclass:: gprMax.SymmetryCompletion

.. autoclass:: gprMax.ExperimentalMask

.. autofunction:: gprMax.evaluate_saved_surface_dft

.. autofunction:: gprMax.spherical_observation_points

``SymmetryCompletion`` requests an exact image-theory completion at declared
PEC or PMC symmetry boundaries. ``ExperimentalMask`` deliberately makes a
KSIR surface mathematically open; it is provided for controlled research
experiments and generally does not produce a physical field without a
separately justified closure. ``evaluate_saved_surface_dft`` evaluates a
compatible saved surface spectrum after a run, while
``spherical_observation_points`` constructs Cartesian exact-field receiver
locations from spherical coordinates.

A layered transform reuses the same far-field request classes. For example,
air above a finite dielectric layer and a lower dielectric half-space can be
declared independently of the geometry objects which build those layers:

.. code-block:: python

    scene.add(gprMax.NTFFLayeredBackground(
        id='ground',
        axis='z',
        materials=('free_space', 'dry_soil', 'wet_soil'),
        interfaces=(0.0, -0.1),
    ))
    scene.add(gprMax.NTFFLayeredFrequencyTransform(
        surface_id='radiation_surface',
        id='ground_band',
        background_id='ground',
        frequencies=(0.8e9, 1.0e9, 1.2e9),
        window='rectangular',
    ))
    scene.add(gprMax.NTFFFarField(
        theta=30,
        phi=0,
        transform_id='ground_band',
        id='upper_pattern',
        outputs=(
            'Etheta',
            'Ephi',
            'directivity_dbi',
            'exterior_power',
            'exterior_maximum',
        ),
    ))

The two exterior materials must be lossless and an explicitly requested
direction must not be exactly grazing to the layer normal. Internal layers
may be conductive or electrically dispersive. See :ref:`ntff-formulations`
for the TE/TM transmission-line formulation.
Far-field and antenna metrics support different lossless exterior materials;
coherent array-codebook synthesis currently requires equal,
frequency-independent exterior impedances.

The grouped ``exterior_power`` output stores the radiated power and fraction
in each positive- and negative-axis exterior. ``exterior_maximum`` stores the
maximum radiation intensity, conventionally full-sphere-normalised
directivity, and its direction in each exterior. These two requests do not
require a port. ``exterior_efficiency`` additionally stores each exterior's
accepted- and incident-power-normalised coupling efficiency and maximum gain;
it therefore requires :class:`NTFFAntennaPorts`, every physical port, and a
rectangular transform window. For example:

.. code-block:: python

    scene.add(gprMax.NTFFAntennaPorts(
        transform_id='ground_band',
        port_ids=('feed',),
    ))
    pattern = gprMax.NTFFFarFieldArray(
        theta_start=1,
        theta_stop=179,
        theta_step=2,
        phi_start=0,
        phi_stop=358,
        phi_step=2,
        transform_id='ground_band',
        id='ground_pattern',
        outputs=(
            'Etheta',
            'Ephi',
            'gain_dbi',
            'realized_gain_dbi',
            'exterior_power',
            'exterior_efficiency',
            'exterior_maximum',
        ),
    )
    scene.add(pattern)

Here the layered transform must use ``window='rectangular'`` because the
second request is normalised by the port spectra. The exact grazing direction
is omitted from this example; it is singular in the layered far-zone
representation.
After the run, the same summaries are available without reopening the HDF5
file through ``pattern.result.radiation_metrics.exterior``. Its arrays use
region order ``(positive_axis, negative_axis)`` and then frequency; the HDF5
writer presents those two rows as named groups.

Time-domain equivalent-current far fields
-----------------------------------------
.. autoclass:: gprMax.user_objects.cmds_output.NTFFTimeFarField

.. autoclass:: gprMax.user_objects.cmds_output.NTFFTimeFarFieldArray

.. autoclass:: gprMax.user_objects.cmds_output.NTFFLayeredTimeFarField

.. autoclass:: gprMax.user_objects.cmds_output.NTFFLayeredTimeFarFieldArray

``NTFFTimeFarField`` and ``NTFFTimeFarFieldArray`` compute homogeneous
time-domain equivalent-current far fields on the CPU, CUDA, OpenCL, and Metal
solvers. The calculation includes the time-staggering modification of
Giannopoulos *et al.* [GIAFF1997]_, described in
:ref:`ntff-equivalent-current-time`. Their ``times`` are reduced times for
range-normalized far fields, and only samples supported by every surface
patch are returned.

``NTFFLayeredTimeTransform`` and its request classes replace homogeneous
propagation by the direct TE/TM impulse responses of a lossless planar stack
[CAP2007]_. They reuse ``NTFFLayeredBackground``. Every layer must be
positive, lossless, and nondispersive, and each requested direction must be
propagating in every layer and non-grazing. CPU and MPI execution use a
Cython/OpenMP accumulation kernel; CUDA, OpenCL, and Metal use device-resident
gather and sparse-deposition kernels. Use ``NTFFLayeredFrequencyTransform``
for conductive, dispersive, or evanescent cases.

For example:

.. code-block:: python

    scene.add(gprMax.NTFFLayeredTimeTransform(
        surface_id='radiation_surface',
        id='ground_transient',
        background_id='ground',
    ))
    scene.add(gprMax.NTFFLayeredTimeFarFieldArray(
        theta_start=5,
        theta_stop=175,
        theta_step=5,
        phi_start=0,
        phi_stop=355,
        phi_step=5,
        transform_id='ground_transient',
        id='transient_pattern',
        outputs=('Etheta', 'Ephi'),
    ))

Subgrid
-------
.. autoclass:: gprMax.SubGridHSG

A subgrid is added to the main scene, but its materials and geometry are added
to the subgrid object. With ``autotranslate=True`` these objects can use main
grid coordinates. Refining subgrids support the double-precision CPU and CUDA
solvers. OpenCL, Metal, and distributed MPI subgrid execution are not supported.

Pass ``subgrid=True`` even for a geometry-only preview; a Scene containing
subgrids without that flag is rejected before model construction. Every subgrid
must have a nonempty ``id`` unique within its Scene. IDs cannot contain path
separators or NUL, or be ``.`` or ``..``, because they identify HDF5 groups.

``ratio=1`` selects an **equal-resolution embedded region**. This is exposed
through ``SubGridHSG`` to avoid a second, overlapping object API, but it does
not refine space or time: values are transferred directly on the shared Yee
lattice, the subgrid-boundary PML and filter are disabled, and no spatial or
temporal interpolation is performed. It inherits ``cpu_precision`` or
``gpu_precision`` from the main CPU or CUDA grid respectively. This mode is
useful, for example, for confining dispersive material
storage and updates to a local part of a larger model.

Only the auxiliary PML around the embedded region is disabled at ``ratio=1``.
Explicitly added ``PMLSlab`` absorbers remain active at each local time step.

For refining grids, ``interpolation=1`` (linear) and ``filter=True`` are the
standard defaults. The API also accepts FITPACK spline degrees 2–5 on both CPU
and CUDA, provided each coarse precursor axis contains more samples than the
chosen degree. CUDA uses the same spline boundary behaviour as CPU and keeps
field interpolation on the device. Higher degrees are experimental: numerical
parity does not establish stability or improve accuracy for every model. Both
options are ignored for ``ratio=1``.

.. note::

    The HSG formulation fixes ``pml_separation`` at ``ratio // 2 + 2``. The
    constructor retains the argument for API compatibility, but any supplied
    value is intentionally ignored. Changing this separation is experimental
    and requires modifying the formulation in the source code.

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

Domain-decomposed MPI CPU models support the same PEC and PMC boundaries.
Only ranks touching a selected global face construct and update it, and
domain-edge corrections are not applied at internal MPI seams.

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

.. _pml-higher-order-stability:

Selecting a stable second-order HORIPML profile
-----------------------------------------------

Duplicating two unshifted first-order HORIPML factors can create an unstable
absorber. Coefficient construction rejects the demonstrated negative-real
product-stretch profiles and identifies the slab/profile, electric or
magnetic sample, and parameters. Reducing the timestep does not cure that
profile instability.

Use one CFS factor, or repair the investigated unit-kappa configuration with
an unshifted first factor by matching the second alpha grading to
``1.1 * sigma1``. Match both polynomial order and direction at every electric
and magnetic sample. Alpha and sigma use the same units in these commands.
This recipe is specific to that configuration, not arbitrary custom profiles.
The full analytical condition and the tested 20,000-step cases are in
:ref:`impedance-pml-profile-theory`.

The check covers boundary, internal, and virtual-guide PMLs, including
terminal samples and the global profile before partitioning. Explicit
``sigmamax=0`` remains zero; only ``None`` requests automatic conductivity.
Passing the check excludes the demonstrated failure mechanism, not every
material, geometry, or mesh instability. The product argument is specific
to HORIPML and must not be applied to MRIPML.

Reusable profiles and internal PML slabs
----------------------------------------

An optional ``id`` on :class:`PMLFormulation` defines a reusable local profile
instead of changing the global PML formulation. One or two :class:`PMLCFS`
objects may be associated with it through ``profile_id``. A named formulation
without named CFS terms uses the default first-order CFS parameters.

.. autoclass:: gprMax.user_objects.cmds_multiuse.PMLSlab

For example, a local MRIPML load can be placed inside a PEC rectangular guide:

.. code-block:: python

    scene.add(gprMax.PMLFormulation(formulation='MRIPML', id='port_load'))
    scene.add(gprMax.PMLSlab(
        p1=(0.005, 0.010, 0.010),
        p2=(0.015, 0.020, 0.020),
        maximum_face='x0',
        profile_id='port_load',
        id='feed_absorber',       # optional API-only label
        build_pec=True,           # default: generate the PEC enclosure
    ))

The automatically generated PEC enclosure is applied after user geometry but
before component averaging and PML coefficient generation. Four transverse
walls and the maximum-stretch backing plate are generated unless a face is
already on a model boundary; the opposite, zero-stretch face is the open
entrance. Set ``build_pec=False`` for manually constructed or deliberately
open experiments. gprMax then warns about exposed faces rather than rejecting
the model. Such incomplete enclosures have no stability guarantee and require
case-specific long-duration testing. The material cross-section must be
invariant through the slab.

When ``id`` is omitted, gprMax assigns ``internal_pml_1``,
``internal_pml_2``, and so on. Internal slabs support the CPU, CUDA, OpenCL,
and Metal solvers on the main 3D grid. A slab may also be added to an HSG
subgrid, where it uses the CPU or CUDA solver and the local-grid update cycle,
including equal-resolution regions with ``ratio=1``. A
subgrid-owned slab must lie wholly within the working region: overlap with its
HSG coupling or auxiliary-PML regions is rejected.

Domain-decomposed MPI CPU models are supported. The user declaration remains
in global coordinates and may cross normal or transverse rank boundaries.
Each participating rank allocates only its local PML history arrays, but its
coefficient slice is taken from the complete global CFS profile so a partition
does not restart the grading. The ordinary field-halo exchanges join these
local corrections; no additional slab-specific communication is required per
timestep. Automatic PEC enclosure, collective material-extrusion checks,
custom profiles, and use as a replacement for a disabled boundary PML retain
their serial behaviour.
