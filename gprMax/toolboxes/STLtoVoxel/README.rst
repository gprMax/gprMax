Toolboxes is a sub-package where useful Python modules contributed by users are stored.

**********
STLtoVoxel
**********

Information
===========

**Author/Contact**: Kartik Bansal (kartikbn21000@gmail.com)

This package provides the ability to directly model real objects without having to build their geometries manually using geometry primitives such as the ``#edge``, ``#plate``, ``#box`` etc.. commands. It specifically provides a tool to convert a `STL file <https://en.wikipedia.org/wiki/STL_(file_format)>`_, which can be produced by many CAD software packages, to a voxelised mesh (FDTD Yee cells) which is saved as a geometry file in HDF5 format suitable for directly importing into gprMax.

This package was created as part of the `Google Summer of Code <https://summerofcode.withgoogle.com/>`_ programme 2021 which gprMax participated. Its first versions used the `stl-to-voxel <https://github.com/cpederkoff/stl-to-voxel>`_ Python library by Christian Pederkoff. It now voxelises with the scanline voxeliser of the ``STEPtoVoxel`` toolbox, which needs no CAD kernel.

**License**: `Creative Commons Attribution-ShareAlike 4.0 International License <http://creativecommons.org/licenses/by-sa/4.0/>`_

**Attribution/cite**: TBC

Package contents
================

* ``stltovoxel.py`` is the main script which should be executed to convert a STL file to a voxelised mesh which is saved as a geometry file in HDF5 format suitable for directly importing into gprMax.
* ``examples`` is a folder containing example STL files as well as gprMax input files that can be used to import the resulting HDF5 geometry files.
* ``convert.py`` reads the STL files, applies any rotation, scaling and translation, and voxelises them on the model's cell grid.
* ``license.md`` is the license for the `stl-to-voxel <https://github.com/cpederkoff/stl-to-voxel>`_ Python library by Christian Pederkoff, which earlier versions of this package adapted.

Installation
============

``STLtoVoxel`` and its ``numpy-stl`` dependency are installed automatically
with gprMax. No separate toolbox installation is required when gprMax is
installed from PyPI or from source.

How to use the package
======================

The main script is ``stltovoxel.py`` which should be run at the command line and takes two arguments:

* ``path`` is base path to the folder containing the STL file(s) to convert.
* ``-dxdydz`` is the spatial discretisation of the generated voxelised mesh, in metres. It should be given as a floating point number and must equal the model's ``#dx_dy_dz``: ``GeometryObjectsRead`` refuses a file whose cell size differs from the model's.

Optional arguments:

* ``--unit m|mm|um`` is the unit of the STL coordinates (default ``mm``).
* ``--rotate RX RY RZ`` rotates the geometry by ``RX``, ``RY`` and ``RZ`` degrees about the x, y and z axes, applied in that order about fixed axes (anticlockwise when looking along each axis towards the origin).
* ``--rotate-about bbox|origin`` is the centre of rotation: the centre of the bounding box of all included STL files (default), or the coordinate origin.
* ``--scale S`` scales the geometry uniformly by ``S`` about the coordinate origin, for example ``--scale 25.4`` with ``--unit mm`` for an STL drawn in inches.
* ``--translate DX DY DZ`` moves the geometry, in the ``--unit`` unit, after rotating and scaling.
* ``--supersample N`` samples each cell ``N`` times along each axis and fills it when most samples are inside. The default, 1, fills a cell when its centre is inside a closed surface, which is what gprMax's own geometry primitives do.
* ``--pad-cells N`` adds ``N`` empty cells on every side (default 0). Empty cells leave the model unchanged when imported.

The settings used are recorded as JSON in the ``conversion_settings`` attribute of the HDF5 file.

STL has no standard coordinate unit. The historical toolbox assumed
millimetres and this remains the default. Use ``--unit m``, ``--unit mm``, or
``--unit um`` explicitly for new workflows.

For a directory containing several parts, create an editable assignment table
before conversion:

.. code-block:: console

    python -m gprMax.toolboxes.STLtoVoxel.stltovoxel anatomy \
        --prepare anatomy.csv

Then convert with:

.. code-block:: console

    python -m gprMax.toolboxes.STLtoVoxel.stltovoxel anatomy \
        -dxdydz 0.002 --assignments anatomy.csv

The CSV maps ``file`` independently to ``material_name`` and ``geometry_tag``.
For example, left and right eyes may share a vitreous material while retaining
separate tags. Higher-priority files overwrite lower-priority files wherever
closed surfaces overlap. Filename stems become tags when no assignment file is
provided.

Material names must not contain ``+``: gprMax reserves it for automatically
averaged material IDs. Use ``_`` instead. Without an assignment CSV, the STL
filename stem is also the material name; if it contains ``+``, use
``--prepare`` and edit ``material_name``, then convert with ``--assignments``.
The source filename itself need not change. The same restriction applies to
``name`` and ``metadata.original_id`` when the converter reuses an edited
companion JSON database. This does not restrict averaged materials restored
from files produced by gprMax's ``GeometryObjectsWrite``.

Where the cells are and how to place them
=========================================

The converter fills each cell whose centre lies inside a closed STL surface,
on a grid aligned with the model's: the grid's lower corner is a whole number
of cells from the coordinate origin. It is written to the HDF5 file as the
``origin_xyz`` attribute (in metres) and printed after conversion, for example:

.. code-block:: none

    Grid origin (lower corner of the first cell): (-0.024, -0.043, 0.005) m
    To keep the STL coordinates, place it with GeometryObjectsRead p1 = (-0.024, -0.043, 0.005) m; ...

Placing the file with ``GeometryObjectsRead`` at ``p1`` equal to this origin
puts every cell where the STL is, the same cells that gprMax's ``Box``,
``Sphere`` and other primitives would fill for the same shape. To move the
object, shift ``p1`` by whole cells (gprMax rounds ``p1`` to the nearest cell
corner and warns when it does), or convert again with ``--translate``.
The object's size depends only on the STL and on the cell size.

Overlapping files are resolved per cell: a file with a higher priority in the
assignment CSV (or, for equal priorities, the later file) overwrites a lower
one.

Rotating a STL file
===================

The quickest way to turn an object is ``--rotate`` (see above), for example
to turn the Stanford bunny upside down:

.. code-block:: none

    python -m gprMax.toolboxes.STLtoVoxel.stltovoxel gprMax/toolboxes/STLtoVoxel/examples/stl/Stanford_Bunny.stl -dxdydz 0.001 --rotate 180 0 0

The object can also be oriented in another application before conversion.

**Author/Contact**: Owen Wilkinson (owenwilkinson39@gmail.com)

`Bambu Studio
<https://bambulab.com/en/download/studio>`_, is a free, open-source application commonly used for slicing 3D printing models that provides a simple solution to this limitation. Importantly, Bambu Studio uses the same axis orientation as gprMax, making it well-suited for preparing stl, 3mf, oltp, stp, step,svg, amf or obj files before importing. The software is lightweight, intuitive, and requires no account creation. Skipping all setup steps will open a basic workspace with a printing base plate visible. Using the previous example of the Stanford Bunny, opening its STL file in Bambu Studio will display the model positioned on the base plate, as shown below.

.. figure:: ../../images_shared/Rotated_Bunny_Import.png
    :width: 600 px

    Image of the Stanford bunny STL file imported into Bambu Studios

The information panel in the bottom-right corner displays the STL object’s dimensions in the same format used in gprMax. To rotate the object about any axis, use the rotate icon located in the top toolbar. As the STL object is rotated, the dimensions shown in the bottom-right panel will update accordingly.

.. figure:: ../../images_shared/Rotated_Bunny_Upside_down.png
    :width: 600 px

    Image of the Stanford bunny STL file rotated upside down

Finally, once the object is correctly oriented in the x, y, and z planes for the gprMax model, use the menu path File → Export → Export all objects as one STL. This will generate a new STL file containing the updated orientation. As before, convert this new STL file into an HDF5 file for use within gprMax, following the procedure outlined in the previous section.

.. figure:: ../../images_shared/Rotated_Bunny_Export.png
    :width: 600 px

    Image of the Stanford bunny STL file exported with new orientation

.. figure:: ../../images_shared/Rotated_Bunny_ParaView.png
    :width: 600 px

    Image of the Stanford bunny rotated inside Paraview


Example
-------

To create a voxelised mesh (HDF5 geometry file) from the ubiquitous `Stanford bunny <https://en.wikipedia.org/wiki/Stanford_bunny>`_ STL file, using a spatial discretisation of 1mm:

.. code-block:: none

    python -m gprMax.toolboxes.STLtoVoxel.stltovoxel gprMax/toolboxes/STLtoVoxel/examples/stl/Stanford_Bunny.stl -dxdydz 0.001

The grid is 109 x 89 x 109 cells of 1mm, so the imported geometry occupies 0.109 x 0.089 x 0.109m from ``p1``, and its origin is (-0.024, -0.043, 0.005) m. The example script below places it at ``p1 = (0.020, 0.020, 0.020)``, which moves the bunny by whole cells into the model's domain.

The converter also creates an editable ``_materials.json`` database beside
the generated HDF5 file. Its
constitutive values are initially ``null`` because the STL file contains no
electromagnetic properties. They must be filled in before running gprMax. The
stable keys in the database are also stored in ``/material_keys`` in the HDF5
file, so their order cannot become detached from the voxel indices. Rerunning
the conversion preserves an existing database when its material keys still
match, so user-entered properties are not overwritten.

The HDF5 file also contains ``tag_data`` and ``tag_names``. Tag IDs describe
the final voxelised cells after overlap resolution and are imported by
``GeometryObjectsRead``. They are not material IDs and dielectric smoothing
does not alter their membership.

.. literalinclude:: ../../gprMax/toolboxes/STLtoVoxel/examples/stl/Stanford_Bunny_geo_materials.json
    :language: json
    :linenos:

The following Python script imports the generated geometry and selects its
database by name. The example database assigns sand to the bunny.

.. literalinclude:: ../../gprMax/toolboxes/STLtoVoxel/examples/bunny.py
    :language: python
    :linenos:

.. figure:: ../../images_shared/stanford_bunny_stl.png
    :width: 600 px

    Image of the Stanford bunny STL file

.. figure:: ../../images_shared/stanford_bunny.png
    :width: 600 px

    FDTD geometry mesh showing the Stanford bunny
