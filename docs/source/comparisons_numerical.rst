*********************
Numerical comparisons
*********************

This section presents comparisons of models using different numerical modelling techniques.

FDTD/MoM
========

The Finite-Difference Time-Domain (FDTD) method from gprMax is compared with the Method of Moments (MoM) from the `MATLAB antenna toolbox <http://uk.mathworks.com/products/antenna/>`_.

Bowtie antenna in free space
----------------------------

This example considers the input impedance of a planar bowtie antenna in free space. The length and height of the bowtie are 100mm, giving a flare angle of :math:`90^\circ`.

FDTD model
^^^^^^^^^^

:download:`antenna_bowtie_fs.in <../../tests/other_codes/vs_MoM_MATLAB/antenna_bowtie_fs/antenna_bowtie_fs.in>`

.. literalinclude:: ../../tests/other_codes/vs_MoM_MATLAB/antenna_bowtie_fs/antenna_bowtie_fs.in
    :language: none
    :linenos:

A Gaussian waveform with a centre frequency of 1.5GHz was used to excite the antenna, which was fed by a transmission line with a characteristic impedance of 50 Ohms.

The module ``plot_antenna_params`` from the ``tools`` subpackage was used to calculate and plot the input impedance from the FDTD model.

MoM model
^^^^^^^^^

The bowtie antenna was created using the antenna toolbox in MATLAB, and the ``bowtieTriangular`` class.

.. code-block:: matlab

    bowtie = bowtieTriangular('Length', 0.1)
    zin = impedance(bowtie, 33.33e6:33.33e6:6e9)

Results
-------

:numref:`antenna_bowtie_fs_ant_params` shows the input impedance (resistive and reactive) for the FDTD (gprMax) and MoM (MATLAB) models. The frequency resolution for the FFT used in both models was :math:`\Delta f = 33.33~MHz`.

.. _antenna_bowtie_fs_ant_params:

.. figure:: ../../tests/other_codes/vs_MoM_MATLAB/antenna_bowtie_fs/antenna_bowtie_fs_ant_params.png
    :width: 600 px

    Input impedance (resistive and reactive) of a bowtie antenna in free space using FDTD (gprMax) and MoM (MATLAB) models.

The results from the FDTD and MoM modelling techniques are in very good agreement. The biggest mismatch occurs in the resistive part of the input impedance at frequencies above 3GHz.

