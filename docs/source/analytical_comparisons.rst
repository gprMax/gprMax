**********************
Analytical comparisons
**********************

This section presents comparisons between analytical and modelled solutions.

Hertzian dipole in free space
=============================

:download:`hertzian_dipole_fs.in <../../tests/analytical/hertzian_dipole_fs/hertzian_dipole_fs.in>`

This example is of a Hertzian dipole, i.e. an additive source (electric current density), in free space.

.. literalinclude:: ../../tests/analytical/hertzian_dipole_fs/hertzian_dipole_fs.in
    :language: none
    :linenos:

The function ``hertzian_dipole_fs`` in the ``analytical solutions`` module in the ``tests`` sub-package was written to compute the analytical solution.

.. autofunction:: tests.analytical_solutions.hertzian_dipole_fs

Results
-------

:numref:`hertzian_dipole_fs_results` shows the time history of the electric and magnetic field components of the modelled and analytical solutions. The responses overlap one another due to their similarity. Therefore, :numref:`hertzian_dipole_fs_results_diffs` shows the percentage differences between the solutions.

.. _hertzian_dipole_fs_results:

.. figure:: ../../tests/analytical/hertzian_dipole_fs/hertzian_dipole_fs_vs_analytical.png
    :width: 600 px

    Time history of the electric and magnetic field components of the modelled and analytical solutions.

.. _hertzian_dipole_fs_results_diffs:

.. figure:: ../../tests/analytical/hertzian_dipole_fs/hertzian_dipole_fs_vs_analytical_diffs.png
    :width: 600 px

    Percentage differences between the modelled and analytical solutions.








