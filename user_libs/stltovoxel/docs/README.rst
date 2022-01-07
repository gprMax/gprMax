User libraries is a sub-package where useful Python modules contributed by users are stored.

**********
stltovoxel
**********

Information
===========

**Author/Contact**: Kartik Bansal (kartikbn21000@gmail.com), India

This module provides the ability to directly model real objects without having to build their geometries manually using geometry primitives such as the ``#edge``, ``#plate``, ``#box`` etc.. commands. It specifically provides a tool to convert a `STL file <https://en.wikipedia.org/wiki/STL_(file_format)>`_, which can be produced by many CAD software packages, to a voxelised mesh (FDTD Yee cells) which is saved as a geometry file in HDF5 format suitable for directly importing into gprMax. 

This module was created as part of the `Google Summer of Code <https://summerofcode.withgoogle.com/>`_ programme 2021 which gprMax participated. The module is based on the `stl-to-voxel <https://github.com/cpederkoff/stl-to-voxel>`_ Python library by Christian Pederkoff.

**License**: `Creative Commons Attribution-ShareAlike 4.0 International License <http://creativecommons.org/licenses/by-sa/4.0/>`_

**Attribution/cite**: TBC

Module overview
===============

* ``stltovoxel.py`` is the main script which should be executed to convert a STL file to a voxelised mesh which is saved as a geometry file in HDF5 format suitable for directly importing into gprMax.
* ``examples`` is a folder containing example STL files as well as gprMax input files that can be used to import the resulting HDF5 geometry files.
* ``convert.py``, ``perimeter.py``, ``slice.py`` are modules adapted from the `stl-to-voxel <https://github.com/cpederkoff/stl-to-voxel>`_ Python library by Christian Pederkoff.
* ``license.md`` is the license for the `stl-to-voxel <https://github.com/cpederkoff/stl-to-voxel>`_ Python library by Christian Pederkoff.

How to use the module
=====================

The main script is ``stltovoxel.py`` which should be run at the command line and takes three arguments:

* ``stlfilename`` is name of STL file to convert including the path.
* ``-matindex`` is an integer which represents the index of the material to be used from the materials file which will accompany the generated geometry file (HDF5 format).
* ``-dxdyz`` is the spatial discretisation of the generated voxelised mesh. It should be given as three floating point numbers.

The physical dimensions of the voxelised object will depend on the size of the object in the original STL file and the spatial discretisation chosen.

Example
-------

To create a voxelised mesh (HDF5 geometry file) from the ubiquitous `Stanford bunny <https://en.wikipedia.org/wiki/Stanford_bunny>`_ STL file, using a spatial discretisation of 1mm and selecting material index 2 from a materials file:

.. code-block:: none

    python -m user_libs.stltovoxel.stltovoxel user_libs/stltovoxel/examples/stl/Stanford_Bunny.stl -matindex 2 -dxdydz 0.001 0.001 0.001

Since the number of voxels are 108 x 88 108 and the spatial discretisation chosen is 1mm, the physical dimensions of the Stanford bunny when imported into gprMax will be 0.108 x 0.088 x 0.108mm.

The following is an example of a ``materials.txt`` file that can be used with the generated geometry file (HDF5 format) when importing into gprMax. Since ``-matindex`` is set to 2 the material with name ``hdpe``, i.e. a plastic, will be used.

.. literalinclude:: ../examples/materials.txt
    :language: none
    :linenos:

The following Python script (using the gprMax API) can be used to import the generated geometry file ``Stanford_Bunny.h5`` and materials file ``materials.txt`` into a gprMax model:

.. literalinclude:: ../examples/bunny.py
    :language: python
    :linenos:

.. figure:: images/stanford_bunny_stl.png
    :width: 600 px

    Image of the Stanford bunny STL file

.. figure:: images/stanford_bunny.png
    :width: 600 px

    FDTD geometry mesh showing the Stanford bunny