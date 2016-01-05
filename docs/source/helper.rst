.. _helper:

****************
Helper utilities
****************

This section provides information on how to use the Python modules (in the ``tools`` package) that help manage gprMax files.

inputfile_old2new.py
--------------------

INSERT DESCRIPTION OF USAGE


outputfiles_merge.py
--------------------

gprMax produces a separate output file for each trace (A-scan) in a B-scan. Combine the separate output files into one file using the Python module ``outputfiles_merge.py``. Usage (from the top-level gprMax directory) is: ``python -m tools.outputfiles_merge basefilename modelruns``, where ``basefilename`` is the base name file of the output file series, e.g. for ``myoutput1.out``, ``myoutput2.out`` the base file name would be ``myoutput``, and ``modelruns`` is the number of output files to combine.

UPDATE DESCRIPTION OF USAGE
