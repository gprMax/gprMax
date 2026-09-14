******************
Material databases
******************

Material databases provide reusable, versioned electromagnetic properties
without copying long lists of dispersion poles into every model. They are
JSON data, not executable gprMax input. gprMax validates and translates only
the entries requested by a model, so an unused multipole material does not
increase model-wide dispersive storage.

Using a material
================

The hash-command form is:

.. code-block:: none

    #material_from_database: fundamental vacuum

The equivalent Python API is:

.. code-block:: python

    scene.add(gprMax.MaterialFromDatabase(
        database="fundamental",
        material="vacuum",
    ))

The material is then used by geometry objects in exactly the same way as one
created with :class:`gprMax.Material`.

The optional local ID is useful when the model needs a more descriptive name,
must avoid a collision with another material, or must match the material name
expected by an imported geometry. If it is omitted, the database entry key is
used. For example, these forms load the ``vacuum`` entry but make it available
to geometry objects as ``model_vacuum``:

.. code-block:: none

    #material_from_database: fundamental vacuum model_vacuum

.. code-block:: python

    scene.add(gprMax.MaterialFromDatabase(
        database="fundamental",
        material="vacuum",
        id="model_vacuum",
    ))

User-defined local material IDs must not contain ``+``: gprMax uses it when
constructing averaged and other generated material IDs. Use ``_`` instead.
This restriction applies to both ``Material`` and ``MaterialFromDatabase``
and their hash commands. It is not a restriction on descriptive JSON ``name``
fields, nor on generated IDs stored in paired geometry databases. Full
``GeometryObjectsWrite``/``GeometryObjectsRead`` round trips preserve those
generated definitions and their component assignments.

Database lookup
===============

Official database names are registered and reserved by gprMax. The initial
catalogues are ``fundamental``, ``gpr``, ``antenna``, and ``bioem``. A local
database named ``laboratory.json`` is selected as ``laboratory``. For an input
file it must be beside the ``.in`` file; for a direct Python API model it must
be in the execution directory. A local file cannot silently shadow an
official name.

Official files are installed in ``gprMax/data/materials`` and are versioned
with the source code. Do not edit installed files: package updates would
replace those edits. Put project or laboratory data in a local database.

List or validate a database from the command line with:

.. code-block:: console

    python -m gprMax.toolboxes.MaterialDatabase list antenna
    python -m gprMax.toolboxes.MaterialDatabase validate laboratory --directory path/to/model

Initial catalogues
==================

The ``fundamental`` database contains the exact vacuum, PEC, and PMC
definitions. The empirical ``gpr``, ``antenna``, and ``bioem`` catalogues are
registered but intentionally empty in this first release of the database
framework. Entries will be added only when their sources, applicable
conditions, frequency range, fit quality, and redistribution rights have been
reviewed. Registering the names now reserves a stable public namespace without
presenting generic or insufficiently documented values as authoritative.

For example, the current, versioned
`IT'IS Tissue Properties Database
<https://itis.swiss/virtual-population/tissue-properties/database>`_ is a
strong source for BioEM work, but its website copyright terms do not permit
redistribution in gprMax without written consent. Users can export the
frequency-dependent data for their scientific work and fit it to a gprMax
multi-pole Debye material using the :doc:`DebyeFit toolbox <inc_DebyeFit>`.
The :doc:`AustinMan/AustinWoman toolbox <inc_AustinMan>` retains its historic
900 MHz and three-pole material mappings for reproducibility and explains how
to convert a downloaded voxel model to the modern HDF5/JSON format.

The :doc:`Materials toolbox <inc_Materials>` supplies the existing Eccosorb LS
three-pole Debye fits as a local ``eccosorb.json`` database, with an editable
slab example. Copy the JSON beside your input file and select a grade with
``#material_from_database: eccosorb eccosorb_ls22``. Direct Python API models
use ``MaterialFromDatabase`` and look for the JSON in their working directory.
The fitted coefficients are preserved from the original toolbox; these entries
are separate from the reserved ``antenna`` catalogue.

Schema and provenance
=====================

The machine-readable schema is supplied as
``gprMax/data/materials/schema-v1.json``. A minimal constant material is:

.. code-block:: json

    {
      "schema": "gprMax-material-database",
      "schema_version": 1,
      "database": {
        "id": "laboratory",
        "name": "Laboratory measurements",
        "version": "1.0.0"
      },
      "materials": {
        "sample_a": {
          "name": "Sample A",
          "model": "constant",
          "base": {
            "relative_permittivity": 4.2,
            "electric_conductivity_s_per_m": 0.012,
            "relative_permeability": 1.0,
            "magnetic_conductivity_s_per_m": 0.0
          },
          "mass_density_kg_per_m3": 1040,
          "metadata": {
            "conditions": {"temperature_c": 20},
            "validity": {"frequency_hz": [100000000, 1000000000]},
            "citations": ["Laboratory report 2026-01"]
          }
        }
      }
    }

The supported models are ``constant``, ``debye``, ``lorentz``, ``drude``,
and the general inclusive pole representation used internally by gprMax.
Field names carry units explicitly; for example a Debye pole uses
``relative_permittivity_difference`` and ``relaxation_time_s``. Perfect
conductors use the ``builtin`` model rather than non-standard JSON infinity.
The optional ``mass_density_kg_per_m3`` field is finite and strictly positive.
It is retained as cell-centred physical metadata and does not participate in
electromagnetic material averaging. When omitted, density is unspecified;
ordinary simulations remain valid, while derived dosimetry calculations must
reject selected material cells without a density.
Lorentz and Drude resonance or plasma frequencies are specified in hertz,
whereas their damping and collision coefficients are specified per second.
The recursive formulation requires pole frequencies below ``1 / dt``; this
coefficient-construction limit must not be mistaken for the stricter Nyquist
limit on resolved simulation output. The Lorentz representation is
underdamped and therefore also requires the damping coefficient to be less
than ``2 * pi`` times the resonance frequency.

Each selected material records the database ID, database version, entry key,
canonical entry SHA-256, source path, and whether it came from an official
catalogue. These values are written to the output HDF5
``material_database_provenance`` group. This makes a simulation auditable even
if a local database is later edited.

Database and grid representations
=================================

The loader first resolves aliases and validates a grid-independent material
specification. Building that specification assigns the material's numeric grid
ID and checks dispersion constraints that depend on the model time step.
Loading a document alone does not allocate dispersion histories. The maximum
pole count is updated when a dispersive material is built, not when an unused
entry is encountered in the database.

The JSON pole keys state their units explicitly. Internally, the historical
``DispersiveMaterial.tau`` array contains relaxation times in seconds for Debye
materials, but pole frequencies in hertz for Lorentz and Drude materials. It
must not be interpreted as an array of time constants without first checking
the material model. The JSON translation preserves these conventions for the
existing coefficient builders.

Material definitions describe constitutive properties, whereas geometry tags
describe final cell membership. Neither a tag ID nor a geometry file's compact
material index is a portable model-wide material ID: import remaps tags by name
and database-backed geometry imports remap material indices through the file's
material-key table.

Tagged geometry files pair ``/tag_data`` with a ``/tag_names`` catalogue whose
first entry is ``untagged``. The tag array must have the same cell shape as
``/data`` and contain non-negative integer indices into that catalogue. Tag
ID zero clears a tag wherever the imported geometry writes a cell. In contrast,
material index ``-1`` in ``/data`` means leave the existing cell and its tag
unchanged; ``-1`` is not a valid tag ID.

Property limitations
====================

An entry is a model under stated conditions, not a universal property of a
trade name. FR-4, soils, concrete, and biological tissues can vary strongly
with composition, manufacture, moisture, temperature, and frequency. New
official empirical entries should therefore include traceable citations,
explicit validity bands and conditions, and fit-quality information for fitted
dispersion models. A generic engineering estimate must be labelled as such
rather than presented as a measured vendor grade.

Curation workflow
=================

An official empirical entry should be added only after the following checks:

#. identify the primary measurement, manufacturer, or standards source and
   record its version or publication identifier;
#. record frequency range, temperature, moisture, density, orientation, and
   other conditions that affect the result;
#. translate the source response to a passive causal model supported by
   gprMax, retaining the original sampled data and fitting script outside the
   runtime catalogue;
#. report complex-permittivity or complex-permeability fit error over the
   declared band and validate the resulting time-domain material against an
   analytical reflection or scattering problem; and
#. review redistribution terms before copying a third-party dataset into an
   official catalogue.

A loss tangent quoted at one frequency is not, by itself, a broadband
time-domain material. It may be documented as a narrowband approximation, or
used as source data for a causal fit, but should not be converted silently to
a frequency-independent conductivity and advertised outside that band.

Geometry object files
=====================

New geometry object files contain a ``/material_keys`` dataset. It maps every
compact integer in ``/data`` and ``/ID`` to an entry in the companion JSON
database. For example:

.. code-block:: none

    #geometry_objects_read: 0 0 0 antenna.h5 antenna_materials

The database is resolved beside the HDF5 geometry file. This is the only
material-file format accepted by ``GeometryObjectsRead`` in version 4. Legacy
text files are inputs to the migration utility below, not to the simulator.

An existing model material is reused only when its numerical constitutive
parameters, pole definitions, and density match the imported definition
exactly. Otherwise the imported material receives a database-qualified name,
such as ``tissue{anatomy_materials}``. If that qualified name already exists
with different properties, the import stops instead of substituting materials.

Exports containing diagonal anisotropic volumes also retain ordered x/y/z
references in each generated cell record's ``metadata.directional_materials``.
These are three keys in the same JSON database, not numeric grid IDs. The
scalar base values on that record are display/estimate means, not a substitute
for the directional constitutive definitions. ``GeometryObjectsRead`` restores
the referenced materials before the cell record, including dispersion and
density. Such files require the complete ``ID``, ``rigidE``, and ``rigidH``
arrays; scalar voxel-only reconstruction would lose the directional model
and is rejected. Do not load a generated directional record with
``MaterialFromDatabase`` as an ordinary scalar material. Existing isotropic
geometry files require no new metadata.

Regenerate older anisotropic geometry exports from their input models if they
lack these directional references: scalar mean cell properties alone do not
retain the constitutive tensor or its directional dispersive poles.

New PNG-derived geometry should be created with
``python -m gprMax.toolboxes.Utilities.convert_png2h5``. The utility writes both the
current HDF5 material-key mapping and an adjacent editable JSON database. Its
entries initially contain null constitutive values because electromagnetic
properties cannot be inferred from image colours; complete them before using
``GeometryObjectsRead``. The selected RGB/RGBA values remain in material
metadata for an auditable colour-to-material mapping. See the
:doc:`Utilities toolbox <inc_Utilities>` for the complete workflow.

.. _legacy_geometry_conversion:

Migrating legacy geometry and text materials
--------------------------------------------

Convert an existing pair once, without modifying the originals:

.. code-block:: console

    python -m gprMax.toolboxes.MaterialDatabase convert-geometry geometry.h5 materials.txt

For this command the outputs are ``geometry_converted.h5`` and
``geometry_materials.json``. The utility prints their paths and the database
name to use. Update the input line, retaining the original insertion position
and any averaging flag:

.. code-block:: none

    #geometry_objects_read: 0 0 0 geometry_converted.h5 geometry_materials

Or, with the Python API:

.. code-block:: python

    scene.add(gprMax.GeometryObjectsRead(
        p1=(0, 0, 0), geofile="geometry_converted.h5",
        material_database="geometry_materials", averaging="n",
    ))

Custom output names can be supplied with ``--output-geometry`` and
``--output-database``. Both outputs must be in the same existing directory;
the database filename must be a valid database name followed by ``.json``.
When only ``--output-geometry`` is supplied, the JSON file is placed beside
that output. Existing output files are never overwritten. Keep the pair
together when copying or sharing it.

The converter copies the HDF5 contents without rebuilding or renumbering
geometry. It preserves cell and component material indices, ``-1`` transparent
entries, rigid flags, compression, attributes, and existing cell tags and their
name table. It adds material keys in the original text-file declaration order,
including built-ins; it does not assume v3 and v4 have the same built-in
indices. Historical ``dx, dy, dz`` spacing metadata gains the current
``dx_dy_dz`` attribute without changing the spacing.

Supported text commands are ``#material``, ``#add_dispersion_debye``,
``#add_dispersion_lorentz``, ``#add_dispersion_drude``, and
``#material_density``. Pole parameters and density in kg/m³ are transferred
without fitting or averaging. Blank lines and ``##`` comments are allowed.
Other commands are rejected, not executed. The text file must contain the
complete original material table, including any background or averaged
materials referenced by the geometry. The utility cannot reconstruct missing
definitions from numeric indices alone.

Before writing, the converter validates the JSON schema, spacing, material
index coverage, component-array shapes and any tag catalogue. Large material
and tag volumes are scanned in bounded blocks. A partial component mesh or
malformed input stops conversion. Failed writes remove outputs created by that
call; neither source file is changed. Files must be self-contained: external
links, external raw storage and HDF5 virtual datasets must be materialised
before conversion, since moving a file copy can break those dependencies.
Timestep-dependent material restrictions
are still checked when the converted model is built for simulation.

Regression tests compare the original and converted datasets and attributes,
and compare converted voxel-model fields with direct geometry builds. Migration
is not regeneration: rebuilding an old model with a newer gprMax version can
change its geometry or interface materials because of separate solver
developments. Conversion itself does not apply those changes.
