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

    python -m toolboxes.MaterialDatabase list antenna
    python -m toolboxes.MaterialDatabase validate laboratory --directory path/to/model

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

Each selected material records the database ID, database version, entry key,
canonical entry SHA-256, source path, and whether it came from an official
catalogue. These values are written to the output HDF5
``material_database_provenance`` group. This makes a simulation auditable even
if a local database is later edited.

Property limitations
====================

An entry is a model under stated conditions, not a universal property of a
trade name. FR-4, soils, concrete, and biological tissues can vary strongly
with composition, manufacture, moisture, temperature, and frequency. New
official empirical entries should therefore include traceable citations,
explicit validity bands and conditions, and fit-quality information for fitted
dispersion models. The migrated Eccosorb fits are explicitly marked as legacy
where that information was not retained. A generic engineering estimate must
be labelled as such rather than presented as a measured vendor grade.

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

The database is resolved beside the HDF5 geometry file. Legacy text material
files remain readable when their filename ends in ``.txt``.

Convert an existing pair non-destructively with:

.. code-block:: console

    python -m toolboxes.MaterialDatabase convert-geometry geometry.h5 materials.txt

The source files are not modified. The converter copies all HDF5 arrays and
attributes, adds stable material keys, writes JSON, and verifies that every
non-negative material index is declared. This works with voxel-only v3 files
and current files containing ``ID``, ``rigidE``, and ``rigidH``.
