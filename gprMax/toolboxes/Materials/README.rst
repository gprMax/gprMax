*********
Materials
*********

Reusable material definitions for gprMax models. ``eccosorb.json`` contains
the existing three-pole Debye fits for Eccosorb LS14, LS16, LS18, LS20, LS22,
LS26, LS28 and LS30.

Files to use
============

.. code-block:: none

    Materials/
        eccosorb.json           # Material properties and Debye poles.
        eccosorb_slab.in        # Complete example using LS22.
        legacy/eccosorb.txt     # Original definitions for older models.

The JSON uses the original material names, such as ``eccosorb_ls22``, so your
geometry can keep those names. Every original coefficient is preserved.
The text file is retained unchanged in ``legacy/``.

Use a material in your model
============================

Hash-command input
------------------

Copy ``eccosorb.json`` beside your input file. Load the required material with
``#material_from_database`` and use its name in your geometry:

.. code-block:: none

    #material_from_database: eccosorb eccosorb_ls22
    #box: 0.010 0.010 0.022 0.038 0.038 0.028 eccosorb_ls22 n

Here ``eccosorb`` is the JSON filename without its extension, and
``eccosorb_ls22`` is an entry inside it. These two lines belong in a complete
model with a domain, discretisation, time window and excitation.

To update a model that used ``#include_file: .../eccosorb.txt``, replace that
line with one ``#material_from_database`` line for each grade your geometry
uses. The JSON already includes all three Debye poles; no separate dispersion
command is needed. Only the selected materials are loaded.

Python API
----------

For a model built directly with the Python API, copy ``eccosorb.json`` into
the directory from which you run Python. Add the material to your scene before
using it in geometry:

.. code-block:: python

    import gprMax

    # Add these to your scene after defining the domain, mesh and time window.
    scene.add(gprMax.MaterialFromDatabase(
        database="eccosorb",
        material="eccosorb_ls22",
    ))
    scene.add(gprMax.Box(
        p1=(0.010, 0.010, 0.022),
        p2=(0.038, 0.038, 0.028),
        material_id="eccosorb_ls22",
        averaging=False,
    ))

The direct API looks for a local database in the working directory, even if
the output file is written elsewhere. For a hash-command model it looks beside
the input file. See :doc:`Material databases <material_databases>` for local
material aliases and the complete schema.

Run the example
===============

From the repository root, in your gprMax Python environment:

.. code-block:: console

    mkdir -p materials_results
    python -m gprMax gprMax/toolboxes/Materials/eccosorb_slab.in -o materials_results/eccosorb_slab

The :download:`editable slab example <../../gprMax/toolboxes/Materials/eccosorb_slab.in>`
uses a 1 mm mesh, a 3 GHz Ricker source, and a receiver on the other side of a
6 mm thick LS22 slab. Change the material name in the material and box lines
to try another grade. The comments separate the model settings, material,
geometry, and excitation.

The output is ``materials_results/eccosorb_slab.h5``. Read the receiver's
electric field and its time axis with:

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File("materials_results/eccosorb_slab.h5", "r") as output:
        ex = output["rxs/rx1/Ex"][...]
        time_s = np.arange(ex.size) * float(output.attrs["dt"])

This is a small example of loading a dispersive material and running a model.
Its receiver signal includes propagation and diffraction around the finite
slab; it is not a measurement of the material's attenuation per unit length.

Inspect or edit the properties
==============================

List the entries or check the database without running a simulation:

.. code-block:: console

    python -m gprMax.toolboxes.MaterialDatabase list eccosorb --directory gprMax/toolboxes/Materials
    python -m gprMax.toolboxes.MaterialDatabase validate eccosorb --directory gprMax/toolboxes/Materials

In each JSON entry, ``model`` is ``debye`` and ``base`` contains the
infinite-frequency relative permittivity, electric conductivity in S/m,
relative permeability, and magnetic conductivity in Ohm/m. Each item in
``poles`` contains a relative-permittivity increment and its relaxation time
in seconds. The static relative permittivity is the base value plus the sum
of the three increments. Loss is included in these poles even though the
separate electric conductivity is zero.

Edit a project copy of the JSON when changing properties. Keep its filename
and ``database.id`` consistent if you rename the database.

Fit reference
=============

These are the original gprMax fits to manufacturer permittivity data, with
the original comparison figures below. They have not been refitted to current
product data. Current product information and datasheet downloads are available
on the `manufacturer's Eccosorb LS page <https://www.laird.com/products/absorbers/microwave-absorbing-foams/single-layer-foams/eccosorb-ls>`_.
The figures show the fit quality; check its suitability over your operating band.

.. figure:: ../../images_shared/eccosorb_ls14.png
    :width: 600 px

    3-pole Debye fit for Eccosorb LS14 absorber (HN indicates data from manufacturer datasheet)

.. figure:: ../../images_shared/eccosorb_ls16.png
    :width: 600 px

    3-pole Debye fit for Eccosorb LS16 absorber (HN indicates data from manufacturer datasheet)

.. figure:: ../../images_shared/eccosorb_ls18.png
    :width: 600 px

    3-pole Debye fit for Eccosorb LS18 absorber (HN indicates data from manufacturer datasheet)

.. figure:: ../../images_shared/eccosorb_ls20.png
    :width: 600 px

    3-pole Debye fit for Eccosorb LS20 absorber (HN indicates data from manufacturer datasheet)

.. figure:: ../../images_shared/eccosorb_ls22.png
    :width: 600 px

    3-pole Debye fit for Eccosorb LS22 absorber (HN indicates data from manufacturer datasheet)

.. figure:: ../../images_shared/eccosorb_ls26.png
    :width: 600 px

    3-pole Debye fit for Eccosorb LS26 absorber (HN indicates data from manufacturer datasheet)

.. figure:: ../../images_shared/eccosorb_ls28.png
    :width: 600 px

    3-pole Debye fit for Eccosorb LS28 absorber (HN indicates data from manufacturer datasheet)

.. figure:: ../../images_shared/eccosorb_ls30.png
    :width: 600 px

    3-pole Debye fit for Eccosorb LS30 absorber (HN indicates data from manufacturer datasheet)
