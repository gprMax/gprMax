*********************
Numerical comparisons
*********************

This section presents some comparisons of models using different numerical modelling techniques.

FDTD/MoM
========

The Finite-Difference Time-Domain (FDTD) method from gprMax is compared with the Method of Moments (MoM) from the MATLAB antenna toolbox (http://uk.mathworks.com/products/antenna/).

Bowtie antenna in free space
----------------------------

:download:`antenna_bowtie_fs.in <../../tests/numerical/vs_MoM_MATLAB/antenna_bowtie_fs/antenna_bowtie_fs.in>`

This example considers the input impedance of a planar bowtie antenna in free space. The length and height of the bowtie are 100mm, giving a flare angle of :math:`90^\circ`.

.. literalinclude:: ../../tests/numerical/vs_MoM_MATLAB/antenna_bowtie_fs/antenna_bowtie_fs.in
    :language: none
    :linenos:

For the MoM, the bowtie antenna was created in MATLAB using the ``bowtieTriangular`` class:

.. code-block:: matlab

    bowtie = bowtieTriangular('Length', 0.1)

Results
-------

:numref:`antenna_bowtie_fs_zin_results` shows the input impedance (resistive and reactive) for the FDTD (gprMax) and MoM (MATLAB) models.

.. _hertzian_dipole_fs_results:

.. figure:: ../../tests/numerical/vs_MoM_MATLAB/antenna_bowtie_fs/antenna_bowtie_fs_zin_results.png
    :width: 600 px

    Input impedance (resistive and reactive) of a bowtie antenna in free space using FDTD (gprMax) and MoM (MATLAB) models.

The match between the FDTD and MoM solutions is generally very good.








