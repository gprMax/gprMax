Toolboxes is a sub-package where useful Python modules contributed by users are stored.

*********
Utilities
*********

Information
===========

This package contains various scripts and helper functions.

Package contents
================

HPC
---

This folder contains scripts to assist with running gprMax on high-performance computing (HPC) systems.

MATLAB
------

This folder contains scripts designed as a base to help getting started with plotting data (A-scans and B-scans) from simulations. They do not feature extensive error checking.


Paraview
--------

This folder contains a Python macro to be installed into Paraview. The macro enables materials to be easily visualised when geometry files are loaded into Paraview.


convert_png2h5.py
-----------------

This module enables a Portable Network Graphics (PNG) image file to be converted into a HDF5 file that can then be used to import geometry into gprMax (see the ``#geometry_objects_read`` command for information on how to use the HDF5 file with a materials file to import the geometry). The resulting geometry will be 2D but maybe extended in the z-(invariate) direction if a 3D model was desired. Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m toolboxes.Utilities.convert_png2h5 imagefile dxdydz

where:

* ``imagefile`` is the name of the PNG image file including the path
* ``dxdydz`` is the spatial discretisation to be used in the model

There is an optional command line argument:

* ``-zcells`` is the number of cells to extend the geometry in the z-(invariate) direction of the model

For example create a HDF5 geometry objects file from the PNG image ``my_layers.png`` with a spatial discretisation of :math:`\Delta x = \Delta y = \Delta z = 0.002` metres, and extending 150 cells in the z-(invariate) direction of the model:

.. code-block:: none

    python -m toolboxes.Utilities.convert_png2h5 my_layers.png 0.002 0.002 0.002 -zcells 150

The module will display the PNG image and allow the user to select colours that will be used to define discrete materials in the model. When the user has finished selecting colours the window should be closed, whereupon the HDF5 file will be written.


get_host_spec.py
----------------

This module prints information about the host machine capabilities for OpenMP/CUDA/OpenCL.


outputfiles_merge.py
--------------------

gprMax produces a separate output file for each trace (A-scan) in a B-scan. This module combines the separate output files into a single file, and can remove the separate output files afterwards. Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m toolboxes.Utilities.outputfiles_merge basefilename --remove-files

where:

* ``basefilename`` is the base name file of the output file series, e.g. for ``myoutput1.h5``, ``myoutput2.h5`` the base file name would be ``myoutput``
* ``remove-files`` is an optional argument (flag) that when given will remove the separate output files after the merge.