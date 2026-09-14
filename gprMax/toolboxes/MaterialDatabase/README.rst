Material database tools
=======================

Use this toolbox to list and validate JSON material databases, or to convert
an older geometry/material pair. The material definitions themselves live in
their own databases; for example, ``gprMax/toolboxes/Materials/eccosorb.json`` supplies
the Eccosorb absorber fits.

List and validate materials
===========================

From the repository root:

.. code-block:: console

    python -m gprMax.toolboxes.MaterialDatabase list eccosorb --directory gprMax/toolboxes/Materials
    python -m gprMax.toolboxes.MaterialDatabase validate eccosorb --directory gprMax/toolboxes/Materials

``list`` prints each entry's key, model type and name. Both commands validate
every entry. ``--directory`` selects the folder containing a local JSON file;
it defaults to the current directory. Use a database name without ``.json``.

To use a material, copy that JSON beside your input file and select an entry:

.. code-block:: none

    #material_from_database: eccosorb eccosorb_ls22

The Python equivalent is ``gprMax.MaterialFromDatabase(database="eccosorb",
material="eccosorb_ls22")``, added to your scene. Direct API models look for
local databases in the working directory. The Materials toolbox README has
a complete example and explains how to edit the properties.

Convert legacy geometry
========================

Version 4 geometry imports use an HDF5 file with material keys and an adjacent
JSON material database. Legacy text material files are no longer accepted by
``GeometryObjectsRead``. Convert an existing pair once:

.. code-block:: console

    python -m gprMax.toolboxes.MaterialDatabase convert-geometry geometry.h5 materials.txt

This creates ``geometry_converted.h5`` and ``geometry_materials.json``. Neither
original is modified, and existing output files are not overwritten. Keep the
two converted files together and update your model:

.. code-block:: none

    #geometry_objects_read: 0 0 0 geometry_converted.h5 geometry_materials

For the API, use ``GeometryObjectsRead(p1=(0, 0, 0),
geofile="geometry_converted.h5", material_database="geometry_materials")``.
Keep your model's original insertion coordinates and averaging option.

Use ``--output-geometry`` and ``--output-database`` for custom names. Outputs
must be in the same existing directory. The JSON basename is the database
name, not a path argument to ``GeometryObjectsRead``.

Conversion retains the original material-table order, voxel/component IDs,
rigid flags, tags and metadata. Material, Debye, Lorentz, Drude and density
commands are translated without fitting or rerasterising. Missing material
definitions cannot be inferred from a geometry array; provide the complete
original text file. Unsupported commands and malformed data are rejected.

Inspect or validate the resulting database with:

.. code-block:: console

    python -m gprMax.toolboxes.MaterialDatabase list geometry_materials
    python -m gprMax.toolboxes.MaterialDatabase validate geometry_materials

Use ``--directory`` when the JSON file is not in the current directory.
