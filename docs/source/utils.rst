.. _utils:

**************
File utilities
**************

This section provides information on how to use the other Python modules in the ``tools`` package to help manage gprMax files.

inputfile_old2new.py
--------------------

This modules assists with the process of migrating input files from the syntax of the old (pre v.3) version of gprMax to the new syntax. It will produce a new input file with the old syntax (attempted to be!) translated to the new syntax. Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m tools.inputfile_old2new inputfile

where ``inputfile`` is the name of input file including the path.


outputfiles_merge.py
--------------------

gprMax produces a separate output file for each trace (A-scan) in a B-scan. This module combines the separate output files into a single file, and can remove the separate output files afterwards. Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m tools.outputfiles_merge basefilename --remove-files

where:

* ``basefilename`` is the base name file of the output file series, e.g. for ``myoutput1.out``, ``myoutput2.out`` the base file name would be ``myoutput``
* ``remove-files`` is an optional argument (flag) that when given will remove the separate output files after the merge.


convert_png2h5.py
-----------------

This module enables a Portable Network Graphics (PNG) image file to be converted into a HDF5 file that can then be used to import geometry into gprMax (see the ``#geometry_objects_read`` command for information on how to use the HDF5 file with a materials file to import the geometry). The resulting geometry will be 2D but maybe extended in the z-(invariate) direction if a 3D model was desired. Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m tools.convert_png2h5 imagefile dxdydz

where:

* ``imagefile`` is the name of the PNG image file including the path
* ``dxdydz`` is the spatial discretisation to be used in the model

There is an optional command line argument:

* ``-zcells`` is the number of cells to extend the geometry in the z-(invariate) direction of the model

For example create a HDF5 geometry objects file from the PNG image ``my_layers.png`` with a spatial discretisation of :math:`\Delta x = \Delta y = \Delta z = 0.002` metres, and extending 150 cells in the z-(invariate) direction of the model:

.. code-block:: none

    python -m tools.convert_png2h5 my_layers.png 0.002 0.002 0.002 -zcells 150

The module will display the PNG image and allow the user to select colours that will be used to define discrete materials in the model. When the user has finished selecting colours the window should be closed, whereupon the HDF5 file will be written.
