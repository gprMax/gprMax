***************
Landmine Models
***************

Ready-to-use geometry and material files for PMA-1, PMN and TS-50 landmines,
and a metal can as a false target. Each model is supplied at 1 mm and 2 mm
cubic mesh spacing.

Files to use
============

Use the HDF5 geometry and JSON materials in this folder. Both resolutions of
a model share one material file; for example:

.. code-block:: none

    LandmineModels/
        PMN_1x1x1.h5
        PMN_2x2x2.h5
        PMN_materials.json
        examples/free_space.py
        legacy/                 # Original HDF5 and text files for older readers.

The same naming applies to ``PMA``, ``TS50`` and ``can``. Keep the material
file beside the geometry files. The JSON lists named materials and their
electromagnetic properties; both resolutions use those same properties.

The simulation mesh must match the selected file's spacing. These are the
actual stored dimensions, in millimetres:

.. list-table:: Geometry extents (x × y × z)
    :header-rows: 1
    :widths: 20 40 40

    * - Model
      - 1 mm geometry
      - 2 mm geometry
    * - PMA
      - 140 × 64 × 33
      - 140 × 64 × 32
    * - PMN
      - 116 × 156 × 49
      - 116 × 156 × 48
    * - TS50
      - 90 × 90 × 44
      - 90 × 90 × 44
    * - can
      - 76 × 76 × 108
      - 76 × 76 × 108

Run the example
===============

From the repository root, in your gprMax Python environment:

.. code-block:: console

    python -m gprMax.toolboxes.LandmineModels.examples.free_space --model PMN --resolution-mm 2

This imports the target, adds a theoretical source and receiver above it,
and runs a short CPU simulation. The output is
``landmine_results/PMN_2mm/target.h5``. Choose another target with ``--model PMA``,
``--model TS50`` or ``--model can``, and use ``--resolution-mm 1`` for 1 mm spacing.
Use ``--directory results/new_run`` to choose a fresh output folder.

:download:`Edit the example <../../gprMax/toolboxes/LandmineModels/examples/free_space.py>`
to set your background, antenna, target position and recording time.
The default 3 ns run checks model import and execution; it is not a calibrated
GPR measurement. The example shows how to build the model, execute it, and
read the receiver's electric field from the output file.

Add a target to your model
==========================

In a saved Python script launched from the repository root:

.. code-block:: python

    from pathlib import Path
    import gprMax

    geometry = Path("gprMax/toolboxes/LandmineModels/PMN_2x2x2.h5").resolve()
    # Your scene must already define a 2 mm mesh, a sufficiently large domain,
    # and the background geometry. Add the target after the background.
    target = gprMax.GeometryObjectsRead(
        p1=(0.032, 0.032, 0.032),  # Lower corner of the imported array, in metres.
        geofile=geometry,
        material_database="PMN_materials",  # JSON filename without .json.
        averaging=False,
    )
    # scene.add(target)

In a hash-command input file, use the equivalent command below, with the
geometry and JSON files beside your input file:

.. code-block:: none

    #geometry_objects_read: 0.032 0.032 0.032 PMN_2x2x2.h5 PMN_materials

Build the soil or other background before importing the target. Transparent
voxels (``-1`` in the geometry array) leave that background unchanged; explicitly
stored air voxels insert air. Changing the simulation mesh does not resample
the stored target.

Reference and licence
=====================

**Author/Contact**: Iraklis Giannakis (iraklis.giannakis@abdn.ac.uk), University of Aberdeen, UK

**License**: `Creative Commons Attribution-ShareAlike 4.0 International License <https://creativecommons.org/licenses/by-sa/4.0/>`_

**Attribution/cite**: Giannakis, I., Giannopoulos, A., Warren, C. (2016).
A Realistic FDTD Numerical Modeling Framework of Ground Penetrating Radar for
Landmine Detection. *IEEE Journal of Selected Topics in Applied Earth Observations
and Remote Sensing*, 9(1), 37–51. https://doi.org/10.1109/JSTARS.2015.2468597

The paper describes the models and how their dielectric properties were fitted
to laboratory measurements of scattered fields in free space. The same licence
and attribution apply to the current files and the unchanged originals in
``legacy/``.

.. figure:: ../../images_shared/PMA.png
    :width: 600 px

    FDTD geometry mesh showing the PMA-1 landmine model.

.. figure:: ../../images_shared/PMN.png
    :width: 600 px

    FDTD geometry mesh showing the PMN landmine model.
