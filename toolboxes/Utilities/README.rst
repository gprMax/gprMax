Toolboxes is a sub-package where useful Python modules contributed by users are stored.

*********
Utilities
*********

Information
===========

This package contains various scripts and helper functions.

Package contents
================

* ``HPC`` is a folder with scripts to assist with running gprMax on high-performance computing (HPC) systems.
* ``MATLAB`` is a folder containing scripts are designed as a base to help getting started with plotting data (A-scans and B-scans) from simulations. They do not feature extensive error checking.
* ``Paraview.py`` is a folder containing a Python macro to be installed into Paraview. The macro enables materials to be easily visualised when geometry files are loaded into Paraview.
* ``convert_png2h5.py`` is a script to convert a PNG image to a HDF5 file that can be used to import geometry (#geometry_objects_read) into a 2D model.
* ``get_host_spec.py`` is a script that prints information about the host machine capabilities for OpenMP/CUDA/OpenCL.
* ``outputfiles_merge.py`` is a script that merges traces (A-scans) from multiple output files into one new file, then optionally removes the series of output files.