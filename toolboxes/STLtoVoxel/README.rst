Toolboxes is a sub-package where useful Python modules contributed by users are stored.

**********
STLtoVoxel
**********

Information
===========

**Author/Contact**: Kartik Bansal (kartikbn21000@gmail.com)

This package provides the ability to directly model real objects without having to build their geometries manually using geometry primitives such as the ``#edge``, ``#plate``, ``#box`` etc.. commands. It specifically provides a tool to convert a `STL file <https://en.wikipedia.org/wiki/STL_(file_format)>`_, which can be produced by many CAD software packages, to a voxelised mesh (FDTD Yee cells) which is saved as a geometry file in HDF5 format suitable for directly importing into gprMax.

This package was created as part of the `Google Summer of Code <https://summerofcode.withgoogle.com/>`_ programme 2021 which gprMax participated. The package uses the `stl-to-voxel <https://github.com/cpederkoff/stl-to-voxel>`_ Python library by Christian Pederkoff.

**License**: `Creative Commons Attribution-ShareAlike 4.0 International License <http://creativecommons.org/licenses/by-sa/4.0/>`_

**Attribution/cite**: TBC

Package contents
================

* ``stltovoxel.py`` is the main script which should be executed to convert a STL file to a voxelised mesh which is saved as a geometry file in HDF5 format suitable for directly importing into gprMax.
* ``examples`` is a folder containing example STL files as well as gprMax input files that can be used to import the resulting HDF5 geometry files.
* ``convert.py``, ``perimeter.py``, ``slice.py`` are modules adapted from the `stl-to-voxel <https://github.com/cpederkoff/stl-to-voxel>`_ Python library by Christian Pederkoff.
* ``license.md`` is the license for the `stl-to-voxel <https://github.com/cpederkoff/stl-to-voxel>`_ Python library by Christian Pederkoff.

How to use the package
======================

The main script is ``stltovoxel.py`` which should be run at the command line and takes two arguments:

* ``path`` is base path to the folder containing the STL file(s) to convert.
* ``-dxdyz`` is the spatial discretisation of the generated voxelised mesh. It should be given as a floating point number.

The physical dimensions of the voxelised object will depend on the size of the object in the original STL file and the spatial discretisation chosen.

Method of rotating STL file
===========================

The STLtoVoxel command enables real-world objects saved in stl format to be converted and imported directly into gprMax. However, this command currently does not support directional manipulation of geometries once inside gprMax.  `Bambu Studio
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

    python -m toolboxes.STLtoVoxel.stltovoxel toolboxes/STLtoVoxel/examples/stl/Stanford_Bunny.stl -dxdydz 0.001

Since the number of voxels are 108 x 88 x 108 and the spatial discretisation chosen is 1mm, the physical dimensions of the Stanford bunny when imported into gprMax will be 0.108 x 0.088 x 0.108mm.

The following is an example of a ``materials.txt`` file that can be used with the generated geometry file (HDF5 format) when importing into gprMax. The material index used in the HDF5 geometry file corresponds to the number of STL files converted, e.g. it will be zero if only a single STL file is converted.

.. literalinclude:: ../../toolboxes/STLtoVoxel/examples/materials.txt
    :language: none
    :linenos:

The following Python script (using our Python API) can be used to import the generated geometry file ``Stanford_Bunny.h5`` and materials file ``materials.txt`` into a gprMax model. The bunny material will be sand, i.e. index zero in the materials file.

.. literalinclude:: ../../toolboxes/STLtoVoxel/examples/bunny.py
    :language: python
    :linenos:

.. figure:: ../../images_shared/stanford_bunny_stl.png
    :width: 600 px

    Image of the Stanford bunny STL file

.. figure:: ../../images_shared/stanford_bunny.png
    :width: 600 px

    FDTD geometry mesh showing the Stanford bunny
