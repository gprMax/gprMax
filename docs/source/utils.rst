.. _utils:

**************
File utilities
**************

This section provides information on how to use the other Python modules in the ``tools`` package to help manage gprMax files.

inputfile_old2new.py
--------------------

This modules assists with the process of migrating input files from the syntax of the old (pre v.3) version of gprMax to the new syntax. It will produce a new input file with the old syntax (attempted to be!) translated to the new syntax. Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m tools.inputfile_new2old inputfile

where ``inputfile`` is the name of input file including the path.


outputfiles_merge.py
--------------------

gprMax produces a separate output file for each trace (A-scan) in a B-scan. This module combines the separate output files into a single file, and offers to remove the separate output files afterwards. Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m tools.outputfiles_merge basefilename modelruns

where:

* ``basefilename`` is the base name file of the output file series, e.g. for ``myoutput1.out``, ``myoutput2.out`` the base file name would be ``myoutput``
* ``modelruns`` is the number of output files to combine